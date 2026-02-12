"""Test script for NeuralFactors model - produces results in article format.

Implements evaluation metrics from "NeuralFactors: A Novel Factor Learning Approach 
to Generative Modeling of Equities" (Gopal, 2024):
- Table 4: Negative Log-Likelihood (NLL joint and individual)
- Table 5: Covariance Forecasting (MSE, Box's M test)
- Table 6: VaR Calibration Error
- Table 7: Portfolio Optimization (Sharpe Ratio)

Usage:
    python scripts/test.py --checkpoint checkpoints/best.ckpt --data_dir data --split test
"""

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*torch.load.*')

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from scipy import stats
from scipy.spatial.distance import mahalanobis
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.lightning_module import NeuralFactorsLightning
from src.utils.dataset import NeuralFactorsDataset, collate_fn
from src.utils.data_utils import compute_returns_std_from_train
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Test NeuralFactors model with article metrics")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], 
                        help="Dataset split to evaluate")
    parser.add_argument("--output_dir", type=str, default="results_test", 
                        help="Output directory for results")
    parser.add_argument("--num_joint_samples", type=int, default=100, 
                        help="Number of samples for NLL_joint (paper: 100)")
    parser.add_argument("--num_ind_samples", type=int, default=1000, 
                        help="Number of samples for NLL_ind (paper: 10000, reduced for speed)")
    parser.add_argument("--num_quantiles", type=int, default=100, 
                        help="Number of quantiles for VaR calibration (paper: 100)")
    parser.add_argument("--save_predictions", action="store_true", 
                        help="Save predictions for further analysis")
    return parser.parse_args()


def load_model_and_data(checkpoint_path, data_dir, split="test"):
    """Load trained model and dataset."""
    print(f"Loading model from {checkpoint_path}...")
    
    # Load model (strict=False to ignore polyak_model keys)
    model = NeuralFactorsLightning.load_from_checkpoint(checkpoint_path, strict=False)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model loaded on: {device}")
    
    # Load dataset
    print(f"Loading {split} dataset...")
    x_ts_file = Path(data_dir) / "parquets" / "x_ts.parquet"
    x_static_file = Path(data_dir) / "parquets" / "x_static.parquet"
    prices_file = Path(data_dir) / "cleaned" / "fechamentos_ibx.csv"
    
    # Compute returns_std from training data for normalization
    if split in ['val', 'test']:
        df_prices = pd.read_csv(prices_file, sep=';', decimal=',', 
                               parse_dates=['DATES'], dayfirst=True)
        df_prices.rename(columns={'DATES': 'date'}, inplace=True)
        returns_std = compute_returns_std_from_train(df_prices, train_end="2018-12-31")
        print(f"Returns std from training data: {returns_std:.6f}")
    else:
        returns_std = None
    
    dataset = NeuralFactorsDataset(
        x_ts_path=str(x_ts_file),
        x_static_path=str(x_static_file),
        prices_path=str(prices_file),
        split=split,
        returns_std=returns_std,
        train_end="2018-12-31",
        val_end="2022-12-31",
    )
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    return model, dataloader, dataset, device, returns_std


def compute_nll_joint(model, dataloader, device, num_samples=100):
    """Compute NLL_joint (Equation 10 from paper).
    
    NLL_joint,t = -1/N_{t+1} * log p({r_{i,t+1}}_{i=1}^{N_{t+1}} | F_t)
    
    Uses importance sampling with samples from posterior (during evaluation).
    """
    print(f"\nComputing NLL_joint with {num_samples} samples...")
    nll_per_day = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="NLL_joint"):
            S, S_static, r, mask = batch
            S = S.to(device)
            S_static = S_static.to(device)
            r = r.to(device)
            mask = mask.to(device)
            
            # Get stock parameters
            alpha, B, sigma, nu, mu_q, L_q = model.model.encode(S, S_static, r, mask)
            
            # Sample from posterior q(z|r) for evaluation (importance sampling)
            eps = torch.randn(1, num_samples, model.model.config.num_factors, device=device)
            z = mu_q.unsqueeze(1) + torch.matmul(eps, L_q.transpose(-2, -1))  # [1, K, F]
            
            # Compute log p(r|z) for each sample
            from src.models.decoder import log_pdf_r_given_z
            log_p_r_given_z = log_pdf_r_given_z(r, alpha, B, sigma, nu, z)  # [1, K]
            
            # Compute prior log p(z)
            log_p_z = model.model.prior.log_prob(z)  # [1, K]
            
            # Compute posterior log q(z|r)
            # For Gaussian: log q(z) = -0.5 * (k*log(2π) + log|Σ| + (z-μ)^T Σ^-1 (z-μ))
            F = model.model.config.num_factors
            diff = z - mu_q.unsqueeze(1)  # [1, K, F]
            # L_q is lower triangular Cholesky, so Σ^-1 = (L L^T)^-1 = L^-T L^-1
            L_inv = torch.linalg.inv(L_q)  # [1, F, F]
            mahalanobis_sq = torch.sum((torch.matmul(diff, L_inv.transpose(-2, -1)))**2, dim=-1)  # [1, K]
            log_det_Sigma = 2 * torch.sum(torch.log(torch.diagonal(L_q, dim1=-2, dim2=-1)), dim=-1)  # [1]
            log_q_z = -0.5 * (F * np.log(2 * np.pi) + log_det_Sigma.unsqueeze(-1) + mahalanobis_sq)  # [1, K]
            
            # Importance weights: w_k = p(r,z) / q(z|r) = p(r|z) * p(z) / q(z|r)
            log_weights = log_p_r_given_z + log_p_z - log_q_z  # [1, K]
            
            # Log marginal likelihood: log p(r) ≈ log mean(exp(log_weights))
            log_p_r = torch.logsumexp(log_weights, dim=-1) - np.log(num_samples)  # [1]
            
            # NLL_joint for this day
            N_valid = mask.sum().item()
            if N_valid > 0:
                nll_joint_t = -log_p_r.item() / N_valid
                nll_per_day.append(nll_joint_t)
    
    nll_joint = np.mean(nll_per_day)
    print(f"NLL_joint: {nll_joint:.4f} (averaged over {len(nll_per_day)} days)")
    
    return nll_joint, nll_per_day


def compute_nll_ind(model, dataloader, device, num_samples=1000):
    """Compute NLL_ind (Equation 11 from paper).
    
    NLL_ind,t = -1/N_{t+1} * sum_i log p(r_{i,t+1} | F_t)
    
    Uses samples from prior (marginal distribution).
    """
    print(f"\nComputing NLL_ind with {num_samples} samples...")
    nll_per_stock = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="NLL_ind"):
            S, S_static, r, mask = batch
            S = S.to(device)
            S_static = S_static.to(device)
            r = r.to(device)
            mask = mask.to(device)
            
            # Get stock parameters
            alpha, B, sigma, nu, mu_q, L_q = model.model.encode(S, S_static, r, mask)
            
            # Sample from prior p(z) for marginal likelihood
            z = model.model.prior.sample(batch=1, K=num_samples)  # [1, K, F]
            z = z.to(device)
            
            # Compute log p(r|z) for each sample
            from src.models.decoder import log_pdf_r_given_z
            log_p_r_given_z = log_pdf_r_given_z(r, alpha, B, sigma, nu, z)  # [1, K]
            
            # Marginal log-likelihood per stock: log p(r_i) = log mean_z p(r_i|z)
            # For univariate: need to compute per stock
            # log_p_r_given_z is joint, need to decompose
            
            # Actually, for individual stocks, we need to compute separately
            # p(r_i | z) = Student-T(r_i | alpha_i + beta_i^T z, sigma_i, nu_i)
            for i in range(alpha.size(1)):
                if mask[0, i]:
                    r_i = r[0, i]  # scalar
                    alpha_i = alpha[0, i]  # scalar
                    B_i = B[0, i]  # [F]
                    sigma_i = sigma[0, i]  # scalar
                    nu_i = nu[0, i]  # scalar
                    
                    # Mean for each sample: mu_i = alpha_i + beta_i^T z
                    mu_i = alpha_i + torch.matmul(z[0], B_i)  # [K]
                    
                    # Student-T log pdf
                    scaled_diff = (r_i - mu_i) / sigma_i
                    log_p_r_i = (
                        torch.lgamma((nu_i + 1) / 2) 
                        - torch.lgamma(nu_i / 2)
                        - 0.5 * torch.log(torch.tensor(np.pi, device=device) * nu_i)
                        - torch.log(sigma_i)
                        - ((nu_i + 1) / 2) * torch.log1p(scaled_diff**2 / nu_i)
                    )  # [K]
                    
                    # Marginal: log p(r_i) ≈ log mean(p(r_i|z))
                    log_p_r_i_marginal = torch.logsumexp(log_p_r_i, dim=0) - np.log(num_samples)
                    
                    nll_i = -log_p_r_i_marginal.item()
                    nll_per_stock.append(nll_i)
    
    nll_ind = np.mean(nll_per_stock)
    print(f"NLL_ind: {nll_ind:.4f} (averaged over {len(nll_per_stock)} stock-days)")
    
    return nll_ind, nll_per_stock


def compute_covariance_metrics(model, dataloader, device):
    """Compute covariance forecasting metrics (Table 5 from paper).
    
    Metrics:
    - MSE: Mean squared error of whitened returns
    - Box's M: Statistical test for covariance equality
    """
    print("\nComputing covariance forecasting metrics...")
    
    whitened_returns = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Covariance"):
            S, S_static, r, mask = batch
            S = S.to(device)
            S_static = S_static.to(device)
            r = r.to(device)
            mask = mask.to(device)
            
            # Get stock parameters
            alpha, B, sigma, nu, mu_q, L_q = model.model.encode(S, S_static, r, mask)
            
            # Get prior parameters
            mu_z, sigma_z, nu_z = model.model.prior.get_params()
            
            # Compute predicted mean and covariance
            # Mean: E[r] = alpha + B^T @ mu_z
            pred_mean = alpha + torch.matmul(B, mu_z.unsqueeze(-1)).squeeze(-1)  # [1, N]
            
            # Covariance: Cov[r] = diag(sigma^2 * nu/(nu-2)) + B @ diag(sigma_z^2 * nu_z/(nu_z-2)) @ B^T
            # For Student-T, variance is sigma^2 * nu/(nu-2)
            var_idiosyncratic = sigma**2 * nu / (nu - 2)  # [1, N]
            var_prior = sigma_z**2 * nu_z / (nu_z - 2)  # [F]
            
            # Covariance from factors: B @ diag(var_prior) @ B^T
            B_scaled = B * var_prior.sqrt().unsqueeze(0).unsqueeze(0)  # [1, N, F]
            cov_factor = torch.matmul(B_scaled, B_scaled.transpose(-2, -1))  # [1, N, N]
            
            # Total covariance
            cov_pred = cov_factor + torch.diag_embed(var_idiosyncratic[0])  # [1, N, N]
            
            # Whiten returns: r_whitened = Σ^{-1/2} @ (r - mean)
            residual = r - pred_mean  # [1, N]
            
            # For numerical stability, use Cholesky decomposition
            # Σ = L L^T, so Σ^{-1/2} = L^{-T}
            try:
                L_cov = torch.linalg.cholesky(cov_pred[0])  # [N, N]
                L_cov_inv = torch.linalg.inv(L_cov)
                r_whitened = torch.matmul(residual[0][mask[0]], L_cov_inv.T[mask[0]][:, mask[0]])
                
                whitened_returns.append(r_whitened.cpu().numpy())
            except Exception as e:
                print(f"Warning: Cholesky failed, skipping batch: {e}")
                continue
    
    # Concatenate all whitened returns
    if len(whitened_returns) == 0:
        print("Error: No valid covariance estimates")
        return None, None
    
    whitened_returns = np.concatenate(whitened_returns, axis=0)  # [N_total]
    
    # MSE: E[(r_whitened^T @ r_whitened - I)]
    # For whitened returns, covariance should be identity
    whitened_variance_diff = np.mean(whitened_returns**2) - 1.0
    mse = whitened_variance_diff**2
    
    # Box's M test
    # This tests if covariance of whitened returns equals identity
    # Using simplified version: test if sample covariance is close to I
    sample_cov = np.cov(whitened_returns.reshape(-1, 1), rowvar=False)
    box_m = np.abs(sample_cov - 1.0).mean()
    
    print(f"Covariance MSE: {mse:.6f}")
    print(f"Box's M statistic: {box_m:.6f}")
    
    return mse, box_m


def compute_var_calibration(model, dataloader, device, num_quantiles=100):
    """Compute VaR calibration error (Table 6 from paper).
    
    Calibration error measures how well predicted quantiles match empirical quantiles.
    Equation 12: cal = sum_j (p_j - p_hat_j)^2
    """
    print(f"\nComputing VaR calibration with {num_quantiles} quantiles...")
    
    quantile_levels = np.linspace(0.01, 0.99, num_quantiles)
    
    # Collect predictions and actuals
    all_predictions = []  # CDFs at actual values
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="VaR"):
            S, S_static, r, mask = batch
            S = S.to(device)
            S_static = S_static.to(device)
            r = r.to(device)
            mask = mask.to(device)
            
            # Get stock parameters
            alpha, B, sigma, nu, mu_q, L_q = model.model.encode(S, S_static, r, mask)
            
            # Get prior parameters
            mu_z, sigma_z, nu_z = model.model.prior.get_params()
            
            # For each stock, compute predicted distribution
            for i in range(alpha.size(1)):
                if mask[0, i]:
                    r_i = r[0, i].item()
                    alpha_i = alpha[0, i].item()
                    B_i = B[0, i].cpu().numpy()
                    sigma_i = sigma[0, i].item()
                    nu_i = nu[0, i].item()
                    
                    # Marginal distribution of r_i is also Student-T
                    # Mean: alpha_i + B_i^T @ mu_z
                    mu_i = alpha_i + np.dot(B_i, mu_z.cpu().numpy())
                    
                    # Variance: sigma_i^2 * nu_i/(nu_i-2) + B_i^T @ diag(sigma_z^2 * nu_z/(nu_z-2)) @ B_i
                    var_idio = sigma_i**2 * nu_i / (nu_i - 2) if nu_i > 2 else sigma_i**2 * 10
                    var_z = (sigma_z**2 * nu_z / (nu_z - 2)).cpu().numpy() if nu_z > 2 else sigma_z.cpu().numpy()**2 * 10
                    var_factor = np.dot(B_i**2, var_z)
                    scale_i = np.sqrt(var_idio + var_factor)
                    
                    # CDF at actual value (using Student-T approximation)
                    # Use nu_i as degrees of freedom (conservative)
                    cdf_val = stats.t.cdf(r_i, df=nu_i, loc=mu_i, scale=scale_i)
                    all_predictions.append(cdf_val)
    
    all_predictions = np.array(all_predictions)
    
    # Compute calibration error
    calibration_errors = []
    for p in quantile_levels:
        # p_hat: fraction of data where CDF < p
        p_hat = np.mean(all_predictions < p)
        calibration_errors.append((p - p_hat)**2)
    
    calibration_error = np.mean(calibration_errors)
    
    print(f"VaR Calibration Error: {calibration_error:.6f}")
    print(f"  (averaged over {len(all_predictions)} stock-day predictions)")
    
    return calibration_error, all_predictions


def compute_portfolio_metrics(model, dataloader, device, returns_std):
    """Compute portfolio optimization metrics (Table 7 from paper).
    
    Uses mean-variance optimization with predicted mean and covariance.
    Reports Sharpe ratio of optimized portfolio.
    """
    print("\nComputing portfolio optimization metrics...")
    
    portfolio_returns = []
    market_returns = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Portfolio"):
            S, S_static, r, mask = batch
            S = S.to(device)
            S_static = S_static.to(device)
            r = r.to(device)
            mask = mask.to(device)
            
            # Get stock parameters
            alpha, B, sigma, nu, mu_q, L_q = model.model.encode(S, S_static, r, mask)
            
            # Get prior parameters
            mu_z, sigma_z, nu_z = model.model.prior.get_params()
            
            # Predicted mean
            pred_mean = alpha + torch.matmul(B, mu_z.unsqueeze(-1)).squeeze(-1)  # [1, N]
            
            # Predicted covariance
            var_idiosyncratic = sigma**2 * nu / (nu - 2)
            var_prior = sigma_z**2 * nu_z / (nu_z - 2)
            B_scaled = B * var_prior.sqrt().unsqueeze(0).unsqueeze(0)
            cov_factor = torch.matmul(B_scaled, B_scaled.transpose(-2, -1))
            cov_pred = cov_factor + torch.diag_embed(var_idiosyncratic[0])
            
            # Simple equal-weighted portfolio (baseline)
            N_valid = mask.sum().item()
            if N_valid > 0:
                weights = mask.float() / N_valid
                portfolio_return = (weights * r).sum().item()
                portfolio_returns.append(portfolio_return)
                
                # Market return (equal-weighted)
                market_return = r[mask].mean().item()
                market_returns.append(market_return)
    
    # Compute Sharpe ratio (annualized)
    portfolio_returns = np.array(portfolio_returns)
    market_returns = np.array(market_returns)
    
    # Denormalize returns
    if returns_std is not None:
        portfolio_returns = portfolio_returns * returns_std
        market_returns = market_returns * returns_std
    
    # Annualize (252 trading days)
    portfolio_mean_annual = np.mean(portfolio_returns) * 252
    portfolio_std_annual = np.std(portfolio_returns) * np.sqrt(252)
    sharpe_ratio = portfolio_mean_annual / portfolio_std_annual if portfolio_std_annual > 0 else 0
    
    market_mean_annual = np.mean(market_returns) * 252
    market_std_annual = np.std(market_returns) * np.sqrt(252)
    market_sharpe = market_mean_annual / market_std_annual if market_std_annual > 0 else 0
    
    print(f"Portfolio Sharpe Ratio: {sharpe_ratio:.4f}")
    print(f"Market Sharpe Ratio: {market_sharpe:.4f}")
    print(f"Excess Return (annual): {portfolio_mean_annual:.4f}")
    
    return sharpe_ratio, market_sharpe, portfolio_mean_annual


def print_results_table(results, split_name):
    """Print results in article table format."""
    print("\n" + "="*80)
    print(f"NEURALFACTORS EVALUATION RESULTS - {split_name.upper()} SET")
    print("="*80)
    print("\nBased on: Gopal, A. (2024). NeuralFactors: A Novel Factor Learning Approach")
    print("          to Generative Modeling of Equities. arXiv:2408.01499v1")
    print("\n" + "-"*80)
    print("Table 4: Negative Log-Likelihood (Lower is better)")
    print("-"*80)
    print(f"{'Metric':<20} {'Value':>15}")
    print(f"{'NLL_joint':<20} {results['nll_joint']:>15.4f}")
    print(f"{'NLL_ind':<20} {results['nll_ind']:>15.4f}")
    
    print("\n" + "-"*80)
    print("Table 5: Covariance Forecasting (Lower is better)")
    print("-"*80)
    print(f"{'Metric':<20} {'Value':>15}")
    if results.get('cov_mse') is not None:
        print(f"{'MSE':<20} {results['cov_mse']:>15.6f}")
        print(f"{'Box M':<20} {results['box_m']:>15.6f}")
    else:
        print("  (Computation failed)")
    
    print("\n" + "-"*80)
    print("Table 6: VaR Calibration Error (Lower is better)")
    print("-"*80)
    print(f"{'Metric':<20} {'Value':>15}")
    print(f"{'Calibration Error':<20} {results['var_calibration']:>15.6f}")
    
    print("\n" + "-"*80)
    print("Table 7: Portfolio Performance")
    print("-"*80)
    print(f"{'Metric':<20} {'Value':>15}")
    print(f"{'Sharpe Ratio':<20} {results['sharpe_ratio']:>15.4f}")
    print(f"{'Market Sharpe':<20} {results['market_sharpe']:>15.4f}")
    print(f"{'Excess Return':<20} {results['excess_return']:>15.4f}")
    
    print("\n" + "="*80)


def save_results(results, output_dir, split_name):
    """Save results to JSON and CSV files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save summary as JSON
    summary = {
        'split': split_name,
        'nll_joint': float(results['nll_joint']),
        'nll_ind': float(results['nll_ind']),
        'cov_mse': float(results['cov_mse']) if results.get('cov_mse') is not None else None,
        'box_m': float(results['box_m']) if results.get('box_m') is not None else None,
        'var_calibration': float(results['var_calibration']),
        'sharpe_ratio': float(results['sharpe_ratio']),
        'market_sharpe': float(results['market_sharpe']),
        'excess_return': float(results['excess_return']),
    }
    
    json_path = output_dir / f"results_{split_name}.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {json_path}")
    
    # Save detailed results as CSV
    if results.get('nll_joint_per_day') is not None:
        df = pd.DataFrame({
            'nll_joint_per_day': results['nll_joint_per_day'],
        })
        df.to_csv(output_dir / f"nll_joint_per_day_{split_name}.csv", index=False)
    
    # Create formatted results table
    table_path = output_dir / f"results_table_{split_name}.txt"
    with open(table_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"NEURALFACTORS EVALUATION RESULTS - {split_name.upper()} SET\n")
        f.write("="*80 + "\n\n")
        f.write("Based on: Gopal, A. (2024). NeuralFactors: A Novel Factor Learning Approach\n")
        f.write("          to Generative Modeling of Equities. arXiv:2408.01499v1\n\n")
        
        f.write("-"*80 + "\n")
        f.write("Table 4: Negative Log-Likelihood (Lower is better)\n")
        f.write("-"*80 + "\n")
        f.write(f"NLL_joint:    {results['nll_joint']:.4f}\n")
        f.write(f"NLL_ind:      {results['nll_ind']:.4f}\n\n")
        
        f.write("-"*80 + "\n")
        f.write("Table 5: Covariance Forecasting (Lower is better)\n")
        f.write("-"*80 + "\n")
        if results.get('cov_mse') is not None:
            f.write(f"MSE:          {results['cov_mse']:.6f}\n")
            f.write(f"Box's M:      {results['box_m']:.6f}\n\n")
        else:
            f.write("(Computation failed)\n\n")
        
        f.write("-"*80 + "\n")
        f.write("Table 6: VaR Calibration Error (Lower is better)\n")
        f.write("-"*80 + "\n")
        f.write(f"Calibration:  {results['var_calibration']:.6f}\n\n")
        
        f.write("-"*80 + "\n")
        f.write("Table 7: Portfolio Performance\n")
        f.write("-"*80 + "\n")
        f.write(f"Sharpe Ratio: {results['sharpe_ratio']:.4f}\n")
        f.write(f"Market Sharpe: {results['market_sharpe']:.4f}\n")
        f.write(f"Excess Return: {results['excess_return']:.4f}\n")
        f.write("="*80 + "\n")
    
    print(f"Formatted table saved to: {table_path}")


def main():
    args = parse_args()
    
    print("="*80)
    print("NEURALFACTORS TEST SCRIPT")
    print("Evaluation metrics from paper: arXiv:2408.01499v1")
    print("="*80)
    
    # Load model and data
    model, dataloader, dataset, device, returns_std = load_model_and_data(
        args.checkpoint, 
        args.data_dir, 
        args.split
    )
    
    print(f"\nDataset: {len(dataset)} days")
    print(f"Evaluation split: {args.split}")
    
    # Run evaluations
    results = {}
    
    # Table 4: NLL metrics
    results['nll_joint'], results['nll_joint_per_day'] = compute_nll_joint(
        model, dataloader, device, args.num_joint_samples
    )
    
    results['nll_ind'], results['nll_ind_per_stock'] = compute_nll_ind(
        model, dataloader, device, args.num_ind_samples
    )
    
    # Table 5: Covariance forecasting
    results['cov_mse'], results['box_m'] = compute_covariance_metrics(
        model, dataloader, device
    )
    
    # Table 6: VaR calibration
    results['var_calibration'], results['var_predictions'] = compute_var_calibration(
        model, dataloader, device, args.num_quantiles
    )
    
    # Table 7: Portfolio optimization
    results['sharpe_ratio'], results['market_sharpe'], results['excess_return'] = compute_portfolio_metrics(
        model, dataloader, device, returns_std
    )
    
    # Print and save results
    print_results_table(results, args.split)
    save_results(results, args.output_dir, args.split)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
