"""Quantitative evaluation script for NeuralFactors model.

Computes paper metrics on test set:
- Negative Log-Likelihood (NLL_joint, NLL_ind)
- Covariance prediction (MSE)
- Value at Risk (VaR) calibration
- Portfolio backtest with benchmark comparison

Usage:
    # Debug mode (fast, first 50 dates)
    python scripts/test.py --checkpoint checkpoints/neuralfactors/last.ckpt --mode debug
    
    # Full evaluation (paper mode)
    python scripts/test.py --checkpoint checkpoints/neuralfactors/last.ckpt --mode paper
"""

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*torch.load.*')

import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from scipy.optimize import minimize
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.lightning_module import NeuralFactorsLightning
from src.utils.dataset import NeuralFactorsDataset, collate_fn
from src.utils.data_utils import compute_returns_std_from_train
from torch.utils.data import DataLoader
import src.models.decoder as decoder


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="NeuralFactors Quantitative Evaluation")
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data', 
                       help='Data directory')
    parser.add_argument('--output_dir', type=str, default='results/evaluation', 
                       help='Output directory for results')
    parser.add_argument('--experiment_name', type=str, default='neuralfactors', 
                       help='Experiment name for organizing outputs')
    parser.add_argument('--split', type=str, default='test', 
                       choices=['train', 'val', 'test'], 
                       help='Dataset split to evaluate')
    parser.add_argument('--num_samples', type=int, default=100, 
                       help='Number of samples for NLL computation (overridden by mode)')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed for reproducibility')
    parser.add_argument('--mode', type=str, default='paper', 
                       choices=['debug', 'paper'],
                       help='Evaluation mode: debug (fast, 50 dates) or paper (full)')
    return parser.parse_args()


# =============================================================================
# 1. Model and Data Loading
# =============================================================================
def load_model_and_data(checkpoint_path, data_dir, split='test'):
    """Load trained model and dataset.
    
    Args:
        checkpoint_path: Path to model checkpoint
        data_dir: Data directory containing parquets and cleaned data
        split: Dataset split ('train', 'val', or 'test')
        
    Returns:
        tuple: (model, dataloader, dataset, returns_std, device)
    """
    print("="*80)
    print("LOADING MODEL AND DATA")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model from checkpoint (strict=False to ignore polyak_model keys)
    print(f"Loading model from {checkpoint_path}...")
    model = NeuralFactorsLightning.load_from_checkpoint(checkpoint_path, strict=False)
    model.eval()
    model = model.to(device)
    print("✓ Model loaded successfully")
    
    # Load config to get returns_std
    checkpoint_dir = Path(checkpoint_path).parent
    config_path = checkpoint_dir / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        returns_std = config['training']['returns_std']
        print(f"✓ Returns std from config: {returns_std:.6f}")
    else:
        # Fallback: compute from data
        print("Config not found, computing returns_std from training data...")
        prices_file = Path(data_dir) / "cleaned" / "fechamentos_ibx.csv"
        df_prices = pd.read_csv(prices_file, sep=';', decimal=',', 
                               parse_dates=['DATES'], dayfirst=True)
        df_prices.rename(columns={'DATES': 'date'}, inplace=True)
        returns_std = compute_returns_std_from_train(df_prices, train_end="2018-12-31")
        print(f"✓ Computed returns_std: {returns_std:.6f}")
    
    # Create dataset
    print(f"Loading {split} dataset...")
    x_ts_file = Path(data_dir) / "parquets" / "x_ts.parquet"
    x_static_file = Path(data_dir) / "parquets" / "x_static.parquet"
    prices_file = Path(data_dir) / "cleaned" / "fechamentos_ibx.csv"
    
    dataset = NeuralFactorsDataset(
        x_ts_path=str(x_ts_file),
        x_static_path=str(x_static_file),
        prices_path=str(prices_file),
        split=split,
        returns_std=returns_std,
        train_end="2018-12-31",
        val_end="2022-12-31",
    )
    
    print(f"✓ Dataset loaded: {len(dataset)} dates")
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    return model, dataloader, dataset, returns_std, device


def setup_output_dirs(output_dir, experiment_name):
    """Create output directory structure.
    
    Args:
        output_dir: Base output directory
        experiment_name: Experiment name for subdirectory
        
    Returns:
        Path: Full output directory path
    """
    output_path = Path(output_dir) / experiment_name
    
    # Create subdirectories
    (output_path / "metrics").mkdir(parents=True, exist_ok=True)
    (output_path / "timeseries").mkdir(parents=True, exist_ok=True)
    (output_path / "plots").mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Output directories created at: {output_path}")
    
    return output_path


# =============================================================================
# 2. Negative Log-Likelihood (NLL) Metrics
# =============================================================================
def compute_nll_metrics(model, dataloader, dataset, num_samples, mode, device):
    """Compute NLL_joint and NLL_ind on dataset.
    
    Args:
        model: Trained NeuralFactors model
        dataloader: DataLoader for iteration
        dataset: Dataset for accessing dates
        num_samples: Number of samples for IWAE loss
        mode: 'debug' or 'paper' (controls max_dates)
        device: torch device
        
    Returns:
        pd.DataFrame: Results with columns [date, nll_joint, nll_ind, log_p_r_z, kl, n_stocks]
    """
    print("\n" + "="*80)
    print("COMPUTING NEGATIVE LOG-LIKELIHOOD METRICS")
    print("="*80)
    
    max_dates = 50 if mode == 'debug' else None
    if max_dates:
        print(f"Debug mode: Processing first {max_dates} dates")
    
    results = []
    model.eval()
    
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, desc="Computing NLL")):
            # Stop if we reached max_dates in debug mode
            if max_dates and idx >= max_dates:
                break
            
            S, S_static, r, mask = batch
            S = S.to(device)
            S_static = S_static.to(device)
            r = r.to(device)
            mask = mask.to(device)
            
            # Compute IWAE loss (returns negative ELBO components)
            output = model.model.compute_iwae_loss(S, S_static, r, num_samples, mask)
            
            # Extract metrics
            nll_joint = output['loss'].item()  # Joint NLL across all stocks
            log_p_r_z = output['log_likelihood'].item()  # E[log p(r|z)]
            kl = output['kl_divergence'].item()  # E[KL(q||p)]
            n_stocks = mask.sum().item()  # Number of valid stocks
            
            # Compute per-stock NLL (average)
            nll_ind = nll_joint / max(n_stocks, 1)
            
            # Get date
            date = dataset.dates[idx]
            
            results.append({
                'date': date,
                'nll_joint': nll_joint,
                'nll_ind': nll_ind,
                'log_p_r_z': log_p_r_z,
                'kl': kl,
                'n_stocks': n_stocks
            })
    
    df = pd.DataFrame(results)
    
    # Print summary
    print(f"\n✓ NLL Computation Complete")
    print(f"  Dates processed: {len(df)}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Mean NLL_joint: {df['nll_joint'].mean():.4f}")
    print(f"  Std NLL_joint: {df['nll_joint'].std():.4f}")
    print(f"  Mean NLL_ind: {df['nll_ind'].mean():.4f}")
    print(f"  Mean log p(r|z): {df['log_p_r_z'].mean():.4f}")
    print(f"  Mean KL: {df['kl'].mean():.4f}")
    
    return df


def save_nll_results(nll_df, output_dir):
    """Save NLL results to CSV.
    
    Args:
        nll_df: DataFrame with NLL metrics
        output_dir: Output directory
    """
    output_path = output_dir / "metrics" / "nll_timeseries.csv"
    nll_df.to_csv(output_path, index=False)
    print(f"✓ NLL results saved to: {output_path}")


def plot_nll_timeseries(nll_df, output_dir):
    """Plot NLL timeseries.
    
    Args:
        nll_df: DataFrame with NLL metrics
        output_dir: Output directory
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot NLL_joint
    axes[0].plot(nll_df['date'], nll_df['nll_joint'], alpha=0.7, linewidth=1.5)
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('NLL Joint')
    axes[0].set_title('Negative Log-Likelihood (Joint) Over Time')
    axes[0].grid(True, alpha=0.3)
    
    # Plot KL divergence
    axes[1].plot(nll_df['date'], nll_df['kl'], color='orange', alpha=0.7, linewidth=1.5)
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('KL Divergence')
    axes[1].set_title('KL Divergence Over Time')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / "plots" / "nll_timeseries.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ NLL plot saved to: {output_path}")


# =============================================================================
# 3. Value at Risk (VaR) Metrics
# =============================================================================
def compute_var_metrics(model, dataloader, dataset, num_samples, mode, returns_std, device):
    """Compute VaR calibration metrics.
    
    Args:
        model: Trained model
        dataloader: DataLoader
        dataset: Dataset for dates
        num_samples: Number of samples for predictions
        mode: 'debug' or 'paper'
        returns_std: Returns standard deviation for denormalization
        device: torch device
        
    Returns:
        pd.DataFrame: Results with columns [quantile, theoretical, empirical, error]
    """
    print("\n" + "="*80)
    print("COMPUTING VALUE AT RISK CALIBRATION")
    print("="*80)
    print(f"Number of samples: {num_samples}")
    
    quantiles = [0.01, 0.05, 0.10]
    max_dates = 50 if mode == 'debug' else None
    if max_dates:
        print(f"Debug mode: Processing first {max_dates} dates")
    
    all_predictions = []
    all_actuals = []
    
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, desc="Computing VaR")):
            if max_dates and idx >= max_dates:
                break
            
            S, S_static, r, mask = batch
            S = S.to(device)
            S_static = S_static.to(device)
            r = r.to(device)
            mask = mask.to(device)
            
            # Get predictions
            pred_output = model.model.predict(S, S_static, num_samples=num_samples)
            r_samples = pred_output['r_samples']  # [batch=1, N, K]
            
            # Remove batch dimension and move to CPU
            r_samples = r_samples[0].cpu().numpy()  # [N, K]
            r_actual = r[0].cpu().numpy()  # [N]
            mask_np = mask[0].cpu().numpy()  # [N]
            
            # Denormalize
            r_samples = r_samples * returns_std
            r_actual = r_actual * returns_std
            
            # Keep only valid stocks
            r_samples = r_samples[mask_np]  # [N_valid, K]
            r_actual = r_actual[mask_np]  # [N_valid]
            
            all_predictions.append(r_samples)
            all_actuals.append(r_actual)
    
    # Stack all data
    predictions = np.concatenate(all_predictions, axis=0)  # [N_total, K]
    actuals = np.concatenate(all_actuals)  # [N_total]
    
    print(f"\nTotal observations: {len(actuals)}")
    print(f"Predictions shape: {predictions.shape}")
    
    # Compute calibration for each quantile
    results = []
    for q in quantiles:
        # Compute theoretical quantiles from predictions
        theoretical_quantiles = np.quantile(predictions, q, axis=1)  # [N_total]
        
        # Count violations (actual < quantile)
        violations = (actuals < theoretical_quantiles).sum()
        empirical_prob = violations / len(actuals)
        
        # Calibration error
        error = abs(empirical_prob - q)
        
        results.append({
            'quantile': q,
            'theoretical': q,
            'empirical': empirical_prob,
            'error': error
        })
        
        quality = 'Good' if error < 0.02 else ('OK' if error < 0.05 else 'Poor')
        print(f"  {q:.2f}: empirical={empirical_prob:.4f}, error={error:.4f} [{quality}]")
    
    df = pd.DataFrame(results)
    print(f"\n✓ VaR Calibration Complete")
    
    return df


def save_var_results(var_df, output_dir):
    """Save VaR results to CSV.
    
    Args:
        var_df: DataFrame with VaR calibration metrics
        output_dir: Output directory
    """
    output_path = output_dir / "metrics" / "var_calibration.csv"
    var_df.to_csv(output_path, index=False)
    print(f"✓ VaR results saved to: {output_path}")


def plot_var_calibration(var_df, output_dir):
    """Plot VaR calibration.
    
    Args:
        var_df: DataFrame with VaR calibration metrics
        output_dir: Output directory
    """
    plt.figure(figsize=(8, 8))
    
    # Scatter plot: theoretical vs empirical
    plt.scatter(var_df['theoretical'], var_df['empirical'], s=100, alpha=0.7)
    
    # Diagonal line (perfect calibration)
    min_val = var_df['theoretical'].min()
    max_val = var_df['theoretical'].max()
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect Calibration')
    
    # Labels and formatting
    for _, row in var_df.iterrows():
        plt.annotate(f"{row['quantile']:.2f}", 
                    (row['theoretical'], row['empirical']),
                    xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('Theoretical Quantile')
    plt.ylabel('Empirical Quantile')
    plt.title('VaR Calibration')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    output_path = output_dir / "plots" / "var_calibration.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ VaR plot saved to: {output_path}")


# =============================================================================
# 4. Covariance Metrics
# =============================================================================
def compute_covariance_metrics(model, dataloader, dataset, mode, returns_std, device):
    """Compare predicted vs empirical covariance.
    
    Args:
        model: Trained model
        dataloader: DataLoader
        dataset: Dataset for dates
        mode: 'debug' or 'paper'
        returns_std: Returns standard deviation
        device: torch device
        
    Returns:
        pd.DataFrame: Results with columns [date, mse_cov, n_stocks]
    """
    print("\n" + "="*80)
    print("COMPUTING COVARIANCE METRICS")
    print("="*80)
    
    window_size = 20
    max_dates = 50 if mode == 'debug' else None
    print(f"Rolling window size: {window_size} days")
    if max_dates:
        print(f"Debug mode: Processing first {max_dates} dates")
    
    results = []
    model.eval()
    
    # Store returns history
    returns_history = []
    
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, desc="Computing Covariance")):
            if max_dates and idx >= max_dates:
                break
            
            S, S_static, r, mask = batch
            S = S.to(device)
            S_static = S_static.to(device)
            r = r.to(device)
            mask = mask.to(device)
            
            # Store current returns for empirical covariance
            r_np = r[0].cpu().numpy() * returns_std
            mask_np = mask[0].cpu().numpy()
            returns_history.append((r_np, mask_np))
            
            # Skip first window_size dates (need history)
            if len(returns_history) <= window_size:
                continue
            
            # Get model predictions
            # Encode to get model components (returns tuple: alpha, B, sigma, nu, mu_q, L_q)
            alpha, B, sigma, nu, mu_q, L_q = model.model.encode(S, S_static, r, mask)
            
            # Get prior parameters
            mu_z, Sigma_z = model.model.prior.to_normal_params()
            
            # Compute predicted covariance
            cov_pred = decoder.marginal_covariance(B[0], Sigma_z, sigma[0])  # [N, N]
            cov_pred = cov_pred.cpu().numpy() * (returns_std ** 2)
            
            # Compute empirical covariance from rolling window
            recent_returns = returns_history[-window_size:]
            
            # Get valid stocks in current period
            valid_now = mask_np
            n_stocks = valid_now.sum()
            
            if n_stocks < 2:
                continue  # Need at least 2 stocks for covariance
            
            # Build returns matrix for valid stocks
            returns_matrix = []
            for r_hist, mask_hist in recent_returns:
                # Only include returns where stock was valid both then and now
                valid_both = valid_now & mask_hist
                r_valid = np.where(valid_both, r_hist, np.nan)
                returns_matrix.append(r_valid[valid_now])
            
            returns_matrix = np.array(returns_matrix)  # [window, N_valid]
            
            # Remove any stocks with NaN (not available throughout window)
            valid_cols = ~np.isnan(returns_matrix).any(axis=0)
            if valid_cols.sum() < 2:
                continue
            
            returns_matrix = returns_matrix[:, valid_cols]
            n_stocks_final = returns_matrix.shape[1]
            
            # Compute empirical covariance
            try:
                cov_emp = np.cov(returns_matrix, rowvar=False)  # [N_valid, N_valid]
                
                # Extract corresponding predicted covariance
                valid_indices = np.where(valid_now)[0][valid_cols]
                cov_pred_sub = cov_pred[np.ix_(valid_indices, valid_indices)]
                
                # Compute MSE
                mse = np.mean((cov_pred_sub - cov_emp) ** 2)
                
                # Get date
                date = dataset.dates[idx]
                
                results.append({
                    'date': date,
                    'mse_cov': mse,
                    'n_stocks': n_stocks_final
                })
            except Exception as e:
                # Handle numerical issues
                print(f"\nWarning at date {dataset.dates[idx]}: {str(e)}")
                continue
    
    df = pd.DataFrame(results)
    
    if len(df) > 0:
        print(f"\n✓ Covariance Computation Complete")
        print(f"  Dates processed: {len(df)}")
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"  Mean MSE: {df['mse_cov'].mean():.6f}")
        print(f"  Std MSE: {df['mse_cov'].std():.6f}")
    else:
        print("\nWarning: No covariance metrics computed")
    
    return df


def save_cov_results(cov_df, output_dir):
    """Save covariance results to CSV.
    
    Args:
        cov_df: DataFrame with covariance metrics
        output_dir: Output directory
    """
    if len(cov_df) > 0:
        output_path = output_dir / "metrics" / "cov_metrics.csv"
        cov_df.to_csv(output_path, index=False)
        print(f"✓ Covariance results saved to: {output_path}")


def plot_cov_metrics(cov_df, output_dir):
    """Plot covariance MSE timeseries.
    
    Args:
        cov_df: DataFrame with covariance metrics
        output_dir: Output directory
    """
    if len(cov_df) == 0:
        return
    
    plt.figure(figsize=(12, 6))
    plt.plot(cov_df['date'], cov_df['mse_cov'], alpha=0.7, linewidth=1.5)
    plt.xlabel('Date')
    plt.ylabel('MSE')
    plt.title('Covariance Prediction Error Over Time')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = output_dir / "plots" / "cov_mse_timeseries.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Covariance plot saved to: {output_path}")


# =============================================================================
# 5. Portfolio Backtest
# =============================================================================
def optimize_portfolio(r_mean, r_cov, method='min_variance'):
    """Optimize portfolio weights.
    
    Args:
        r_mean: Expected returns [N]
        r_cov: Covariance matrix [N, N]
        method: Optimization method ('equal_weight' or 'min_variance')
        
    Returns:
        np.array: Portfolio weights [N]
    """
    N = len(r_mean)
    
    if method == 'equal_weight':
        return np.ones(N) / N
    
    elif method == 'min_variance':
        # Minimize portfolio variance: w^T Sigma w
        # Subject to: sum(w) = 1, w >= 0
        
        def objective(w):
            return w @ r_cov @ w
        
        # Constraints
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        
        # Bounds: long-only
        bounds = [(0, 1) for _ in range(N)]
        
        # Initial guess: equal weight
        w0 = np.ones(N) / N
        
        # Optimize
        try:
            result = minimize(objective, w0, method='SLSQP', 
                            bounds=bounds, constraints=constraints,
                            options={'maxiter': 1000})
            if result.success:
                return result.x
            else:
                print(f"Warning: Optimization failed, using equal weight")
                return np.ones(N) / N
        except Exception as e:
            print(f"Warning: Optimization error ({str(e)}), using equal weight")
            return np.ones(N) / N
    
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_max_drawdown(returns):
    """Compute maximum drawdown from returns series.
    
    Args:
        returns: Array of returns
        
    Returns:
        float: Maximum drawdown (negative value)
    """
    cumulative = (1 + returns).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()


def load_ibovespa_returns(data_dir, start_date, end_date):
    """Load Ibovespa benchmark returns.
    
    Args:
        data_dir: Data directory
        start_date: Start date
        end_date: End date
        
    Returns:
        pd.DataFrame or None: DataFrame with columns [date, return]
    """
    ibov_path = Path(data_dir) / "cleaned" / "ibovespa.csv"
    
    if not ibov_path.exists():
        print(f"Warning: Ibovespa data not found at {ibov_path}. Skipping benchmark comparison.")
        return None
    
    try:
        df = pd.read_csv(ibov_path, sep=';', decimal=',', parse_dates=['date'])
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        df['return'] = df['price'].pct_change()
        df = df[['date', 'return']].dropna()
        return df
    except Exception as e:
        print(f"Warning: Error loading Ibovespa data: {str(e)}")
        return None


def compute_portfolio_metrics(model, dataset, returns_std, mode, device, output_dir):
    """Run portfolio backtest with benchmark comparison.
    
    Args:
        model: Trained model
        dataset: Dataset
        returns_std: Returns standard deviation
        mode: 'debug' or 'paper'
        device: torch device
        output_dir: Output directory
        
    Returns:
        tuple: (returns_df, metrics_dict)
    """
    print("\n" + "="*80)
    print("PORTFOLIO BACKTEST")
    print("="*80)
    
    max_dates = 50 if mode == 'debug' else None
    method = 'min_variance'
    print(f"Portfolio method: {method}")
    if max_dates:
        print(f"Debug mode: Processing first {max_dates} dates")
    
    portfolio_returns = []
    dates = []
    
    model.eval()
    with torch.no_grad():
        for idx in tqdm(range(len(dataset) - 1), desc="Running Backtest"):
            if max_dates and idx >= max_dates - 1:
                break
            
            # Get current date data
            S, S_static, r, mask = dataset[idx]
            
            # Get next date returns for evaluation
            _, _, r_next, mask_next = dataset[idx + 1]
            
            # Add batch dimension and move to device
            S = S.unsqueeze(0).to(device)
            S_static = S_static.unsqueeze(0).to(device)
            r = r.unsqueeze(0).to(device)
            mask = mask.unsqueeze(0).to(device)
            
            # Get model predictions
            enc_output = model.model.encode(S, S_static, r, mask)
            alpha = enc_output['alpha']  # [1, N]
            B = enc_output['B']  # [1, N, F]
            sigma = enc_output['sigma']  # [1, N]
            
            # Get prior parameters
            mu_z, Sigma_z = model.model.prior.to_normal_params()
            
            # Compute expected returns and covariance
            r_mean = decoder.marginal_mean(alpha[0], B[0], mu_z)  # [N]
            r_cov = decoder.marginal_covariance(B[0], Sigma_z, sigma[0])  # [N, N]
            
            # Move to CPU and denormalize
            r_mean = r_mean.cpu().numpy() * returns_std
            r_cov = r_cov.cpu().numpy() * (returns_std ** 2)
            mask_np = mask[0].cpu().numpy()
            
            # Filter valid stocks
            r_mean_valid = r_mean[mask_np]
            valid_indices = np.where(mask_np)[0]
            r_cov_valid = r_cov[np.ix_(valid_indices, valid_indices)]
            
            # Optimize portfolio
            weights_valid = optimize_portfolio(r_mean_valid, r_cov_valid, method=method)
            
            # Create full weight vector
            weights = np.zeros(len(mask_np))
            weights[valid_indices] = weights_valid
            
            # Compute portfolio return using next period's realized returns
            r_next_np = r_next.numpy() * returns_std
            mask_next_np = mask_next.numpy()
            
            # Only use stocks that are valid in both periods
            valid_both = mask_np & mask_next_np
            if valid_both.sum() > 0:
                # Renormalize weights for stocks valid in both periods
                weights_both = weights[valid_both]
                weights_both = weights_both / weights_both.sum() if weights_both.sum() > 0 else weights_both
                r_next_both = r_next_np[valid_both]
                
                port_return = np.dot(weights_both, r_next_both)
                portfolio_returns.append(port_return)
                dates.append(dataset.dates[idx + 1])
    
    # Create returns DataFrame
    returns_df = pd.DataFrame({
        'date': dates,
        'return': portfolio_returns
    })
    
    print(f"\n✓ Backtest Complete")
    print(f"  Periods: {len(returns_df)}")
    print(f"  Date range: {returns_df['date'].min()} to {returns_df['date'].max()}")
    
    # Compute performance metrics
    returns_array = returns_df['return'].values
    
    total_return = (1 + returns_array).prod() - 1
    annualized_return = returns_array.mean() * 252
    annualized_vol = returns_array.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0
    max_dd = compute_max_drawdown(returns_array)
    
    metrics = {
        'total_return': float(total_return),
        'annualized_return': float(annualized_return),
        'annualized_vol': float(annualized_vol),
        'sharpe_ratio': float(sharpe_ratio),
        'max_drawdown': float(max_dd)
    }
    
    # Load and compare with benchmark
    benchmark_df = load_ibovespa_returns(
        Path(output_dir).parent.parent / "data",
        returns_df['date'].min(),
        returns_df['date'].max()
    )
    
    if benchmark_df is not None:
        # Merge returns
        merged = returns_df.merge(benchmark_df, on='date', suffixes=('_strategy', '_benchmark'))
        
        if len(merged) > 0:
            bench_returns = merged['return_benchmark'].values
            bench_total = (1 + bench_returns).prod() - 1
            bench_annual = bench_returns.mean() * 252
            bench_vol = bench_returns.std() * np.sqrt(252)
            bench_sharpe = bench_annual / bench_vol if bench_vol > 0 else 0
            bench_dd = compute_max_drawdown(bench_returns)
            
            # Compute excess metrics
            strat_returns = merged['return_strategy'].values
            excess_returns = strat_returns - bench_returns
            excess_annual = excess_returns.mean() * 252
            tracking_error = excess_returns.std() * np.sqrt(252)
            info_ratio = excess_annual / tracking_error if tracking_error > 0 else 0
            
            metrics['benchmark_total_return'] = float(bench_total)
            metrics['benchmark_annualized_return'] = float(bench_annual)
            metrics['benchmark_sharpe'] = float(bench_sharpe)
            metrics['benchmark_max_drawdown'] = float(bench_dd)
            metrics['excess_return'] = float(excess_annual)
            metrics['information_ratio'] = float(info_ratio)
            
            print(f"\n  Benchmark comparison:")
            print(f"    Ibovespa Ann. Return: {bench_annual:.2%}")
            print(f"    Excess Return: {excess_annual:.2%}")
            print(f"    Information Ratio: {info_ratio:.2f}")
    
    # Print metrics
    print(f"\n  Performance Metrics:")
    print(f"    Total Return: {total_return:.2%}")
    print(f"    Annualized Return: {annualized_return:.2%}")
    print(f"    Annualized Volatility: {annualized_vol:.2%}")
    print(f"    Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"    Max Drawdown: {max_dd:.2%}")
    
    # Save results
    returns_path = output_dir / "timeseries" / "backtest_returns.csv"
    returns_df.to_csv(returns_path, index=False)
    print(f"\n✓ Returns saved to: {returns_path}")
    
    metrics_path = output_dir / "metrics" / "backtest_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Metrics saved to: {metrics_path}")
    
    return returns_df, metrics


def plot_cumulative_returns(returns_df, output_dir, data_dir):
    """Plot cumulative returns.
    
    Args:
        returns_df: DataFrame with strategy returns
        output_dir: Output directory
        data_dir: Data directory (for loading benchmark)
    """
    plt.figure(figsize=(12, 6))
    
    # Strategy cumulative returns
    strategy_cum = (1 + returns_df['return']).cumprod()
    plt.plot(returns_df['date'], strategy_cum, label='NeuralFactors Min-Variance', 
             linewidth=2, alpha=0.8)
    
    # Try to load and plot benchmark
    benchmark_df = load_ibovespa_returns(
        data_dir,
        returns_df['date'].min(),
        returns_df['date'].max()
    )
    
    if benchmark_df is not None:
        # Merge to align dates
        merged = returns_df[['date']].merge(benchmark_df, on='date', how='left')
        merged['return'] = merged['return'].fillna(0)
        benchmark_cum = (1 + merged['return']).cumprod()
        plt.plot(returns_df['date'], benchmark_cum, label='Ibovespa', 
                linewidth=2, alpha=0.8, linestyle='--')
    
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.title('Cumulative Returns: NeuralFactors vs Benchmark')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = output_dir / "plots" / "cumulative_returns.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Cumulative returns plot saved to: {output_path}")


# =============================================================================
# 6. Results Saving and Reporting
# =============================================================================
def generate_summary_report(output_dir, nll_df, cov_df, var_df, backtest_metrics):
    """Generate summary report of all metrics.
    
    Args:
        output_dir: Output directory
        nll_df: NLL results DataFrame
        cov_df: Covariance results DataFrame
        var_df: VaR results DataFrame
        backtest_metrics: Dictionary of backtest metrics
    """
    print("\n" + "="*80)
    print("GENERATING SUMMARY REPORT")
    print("="*80)
    
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("NEURALFACTORS MODEL EVALUATION SUMMARY")
    report_lines.append("="*80)
    report_lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # 1. NLL Summary
    report_lines.append("-" * 80)
    report_lines.append("1. NEGATIVE LOG-LIKELIHOOD")
    report_lines.append("-" * 80)
    if not nll_df.empty:
        report_lines.append(f"Dates processed: {len(nll_df)}")
        report_lines.append(f"Date range: {nll_df['date'].min()} to {nll_df['date'].max()}")
        report_lines.append(f"Mean NLL_joint: {nll_df['nll_joint'].mean():.4f} ± {nll_df['nll_joint'].std():.4f}")
        report_lines.append(f"Mean NLL_ind: {nll_df['nll_ind'].mean():.4f} ± {nll_df['nll_ind'].std():.4f}")
        report_lines.append(f"Mean log p(r|z): {nll_df['log_p_r_z'].mean():.4f}")
        report_lines.append(f"Mean KL divergence: {nll_df['kl'].mean():.4f}")
    else:
        report_lines.append("No NLL metrics computed")
    report_lines.append("")
    
    # 2. Covariance Summary
    report_lines.append("-" * 80)
    report_lines.append("2. COVARIANCE PREDICTION")
    report_lines.append("-" * 80)
    if not cov_df.empty:
        report_lines.append(f"Dates processed: {len(cov_df)}")
        report_lines.append(f"Date range: {cov_df['date'].min()} to {cov_df['date'].max()}")
        report_lines.append(f"Mean MSE: {cov_df['mse_cov'].mean():.6f}")
        report_lines.append(f"Std MSE: {cov_df['mse_cov'].std():.6f}")
        report_lines.append(f"Min MSE: {cov_df['mse_cov'].min():.6f}")
        report_lines.append(f"Max MSE: {cov_df['mse_cov'].max():.6f}")
    else:
        report_lines.append("No covariance metrics computed")
    report_lines.append("")
    
    # 3. VaR Calibration
    report_lines.append("-" * 80)
    report_lines.append("3. VALUE AT RISK CALIBRATION")
    report_lines.append("-" * 80)
    if not var_df.empty:
        report_lines.append(f"{'Quantile':<12} {'Theoretical':<15} {'Empirical':<15} {'Error':<10} {'Quality'}")
        report_lines.append("-" * 70)
        for _, row in var_df.iterrows():
            q = row['quantile']
            t = row['theoretical']
            e = row['empirical']
            err = row['error']
            quality = 'Good' if err < 0.02 else ('OK' if err < 0.05 else 'Poor')
            report_lines.append(f"{q:<12.2f} {t:<15.4f} {e:<15.4f} {err:<10.4f} {quality}")
    else:
        report_lines.append("No VaR metrics computed")
    report_lines.append("")
    
    # 4. Portfolio Backtest
    report_lines.append("-" * 80)
    report_lines.append("4. PORTFOLIO BACKTEST (MIN-VARIANCE)")
    report_lines.append("-" * 80)
    if backtest_metrics:
        report_lines.append(f"Total Return: {backtest_metrics['total_return']:.2%}")
        report_lines.append(f"Annualized Return: {backtest_metrics['annualized_return']:.2%}")
        report_lines.append(f"Annualized Volatility: {backtest_metrics['annualized_vol']:.2%}")
        report_lines.append(f"Sharpe Ratio: {backtest_metrics['sharpe_ratio']:.2f}")
        report_lines.append(f"Max Drawdown: {backtest_metrics['max_drawdown']:.2%}")
        
        if 'benchmark_total_return' in backtest_metrics:
            report_lines.append("")
            report_lines.append("Benchmark Comparison (Ibovespa):")
            report_lines.append(f"  Benchmark Ann. Return: {backtest_metrics['benchmark_annualized_return']:.2%}")
            report_lines.append(f"  Benchmark Sharpe: {backtest_metrics['benchmark_sharpe']:.2f}")
            report_lines.append(f"  Excess Return: {backtest_metrics['excess_return']:.2%}")
            report_lines.append(f"  Information Ratio: {backtest_metrics['information_ratio']:.2f}")
    else:
        report_lines.append("No backtest metrics computed")
    report_lines.append("")
    
    report_lines.append("="*80)
    report_lines.append("END OF REPORT")
    report_lines.append("="*80)
    
    # Write to file
    report_text = "\n".join(report_lines)
    report_path = output_dir / "evaluation_summary.txt"
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    print(f"\n✓ Summary report saved to: {report_path}")
    
    # Also print to console
    print("\n" + report_text)


# =============================================================================
# 7. Main Function
# =============================================================================
def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Configure based on mode
    if args.mode == 'debug':
        num_samples_nll = 10
        num_samples_var = 100
        print("\n" + "="*80)
        print("RUNNING IN DEBUG MODE")
        print("="*80)
        print("- NLL samples: 10")
        print("- VaR samples: 100")
        print("- Max dates: 50")
    else:  # paper
        num_samples_nll = args.num_samples
        num_samples_var = 1000
        print("\n" + "="*80)
        print("RUNNING IN PAPER MODE (FULL EVALUATION)")
        print("="*80)
        print(f"- NLL samples: {num_samples_nll}")
        print(f"- VaR samples: {num_samples_var}")
        print("- Max dates: All")
    
    start_time = time.time()
    
    try:
        # 1. Load model and data
        model, dataloader, dataset, returns_std, device = load_model_and_data(
            args.checkpoint, args.data_dir, args.split
        )
        
        # 2. Setup output directories
        output_dir = setup_output_dirs(args.output_dir, args.experiment_name)
        
        # 3. Compute NLL metrics
        phase_start = time.time()
        nll_df = compute_nll_metrics(
            model, dataloader, dataset, num_samples_nll, args.mode, device
        )
        save_nll_results(nll_df, output_dir)
        plot_nll_timeseries(nll_df, output_dir)
        print(f"  Time elapsed: {time.time() - phase_start:.1f}s")
        
        # 4. Compute Covariance metrics
        phase_start = time.time()
        cov_df = compute_covariance_metrics(
            model, dataloader, dataset, args.mode, returns_std, device
        )
        if not cov_df.empty:
            save_cov_results(cov_df, output_dir)
            plot_cov_metrics(cov_df, output_dir)
            print(f"  Time elapsed: {time.time() - phase_start:.1f}s")
        
        # 5. Compute VaR metrics
        phase_start = time.time()
        var_df = compute_var_metrics(
            model, dataloader, dataset, num_samples_var, args.mode, returns_std, device
        )
        if not var_df.empty:
            save_var_results(var_df, output_dir)
            plot_var_calibration(var_df, output_dir)
            print(f"  Time elapsed: {time.time() - phase_start:.1f}s")
        
        # 6. Portfolio backtest
        phase_start = time.time()
        returns_df, backtest_metrics = compute_portfolio_metrics(
            model, dataset, returns_std, args.mode, device, output_dir
        )
        if not returns_df.empty:
            plot_cumulative_returns(returns_df, output_dir, Path(args.data_dir))
            print(f"  Time elapsed: {time.time() - phase_start:.1f}s")
        
        # 7. Generate summary report (Phase 4)
        generate_summary_report(output_dir, nll_df, cov_df, var_df, backtest_metrics)
        
        # Final summary
        total_time = time.time() - start_time
        print("\n" + "="*80)
        print("EVALUATION COMPLETE")
        print("="*80)
        print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"Results saved to: {output_dir}")
        
    except Exception as e:
        print("\n" + "="*80)
        print("ERROR DURING EVALUATION")
        print("="*80)
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())