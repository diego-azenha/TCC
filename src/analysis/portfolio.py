"""Portfolio backtest and optimization for NeuralFactors evaluation."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from scipy.optimize import minimize
from tqdm import tqdm

import src.models.decoder as decoder


# =============================================================================
# Helpers
# =============================================================================

def compute_max_drawdown(returns):
    """Compute maximum drawdown from a returns array."""
    cumulative = (1 + returns).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()


def optimize_portfolio(r_mean, r_cov, method='min_variance'):
    """Compute portfolio weights.

    Args:
        r_mean: Expected returns [N]
        r_cov: Covariance matrix [N, N]
        method: 'equal_weight' or 'min_variance'

    Returns:
        np.array: Portfolio weights [N]
    """
    N = len(r_mean)

    if method == 'equal_weight':
        return np.ones(N) / N

    # min_variance: minimize w^T Sigma w s.t. sum(w)=1, w>=0
    def objective(w):
        return w @ r_cov @ w

    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, 1)] * N
    w0 = np.ones(N) / N

    try:
        result = minimize(objective, w0, method='SLSQP',
                          bounds=bounds, constraints=constraints,
                          options={'maxiter': 1000})
        return result.x if result.success else w0
    except Exception as e:
        print(f"Warning: Optimization error ({e}), using equal weight")
        return w0


def load_ibovespa_returns(data_dir, start_date, end_date):
    """Load Ibovespa benchmark returns.

    Returns:
        pd.DataFrame with [date, return] or None if not available
    """
    ibov_path = Path(data_dir) / "cleaned" / "ibovespa.csv"
    if not ibov_path.exists():
        print(f"Warning: Ibovespa data not found at {ibov_path}. Skipping benchmark.")
        return None

    try:
        df = pd.read_csv(ibov_path, sep=';', decimal=',', parse_dates=['date'])
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        df['return'] = df['price'].pct_change()
        return df[['date', 'return']].dropna()
    except Exception as e:
        print(f"Warning: Error loading Ibovespa data: {e}")
        return None


# =============================================================================
# Main backtest
# =============================================================================

def compute_portfolio_metrics(model, dataset, returns_std, mode, device, output_dir):
    """Run minimum-variance portfolio backtest.

    Args:
        model: Trained model
        dataset: Dataset
        returns_std: Returns std for denormalization
        mode: 'debug' or 'paper'
        device: torch device
        output_dir: Output directory (used to infer data_dir for benchmark)

    Returns:
        tuple: (returns_df, metrics_dict)
    """
    print("\n" + "=" * 80)
    print("PORTFOLIO BACKTEST")
    print("=" * 80)

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

            S, S_static, r, mask = dataset[idx]
            _, _, r_next, mask_next = dataset[idx + 1]

            S = S.unsqueeze(0).to(device)
            S_static = S_static.unsqueeze(0).to(device)
            r = r.unsqueeze(0).to(device)
            mask = mask.unsqueeze(0).to(device)

            alpha, B, sigma, nu, mu_q, L_q = model.model.encode(S, S_static, r, mask)
            mu_z, Sigma_z = model.model.prior.to_normal_params()

            r_mean = decoder.marginal_mean(alpha[0], B[0], mu_z)
            r_cov = decoder.marginal_covariance(B[0], Sigma_z, sigma[0])

            if r_mean.dim() == 2:
                r_mean = r_mean[0]
            if r_cov.dim() == 3:
                r_cov = r_cov[0]

            r_mean = r_mean.cpu().numpy() * returns_std
            r_cov = r_cov.cpu().numpy() * (returns_std ** 2)
            mask_np = mask[0].cpu().numpy().astype(bool)

            valid_idx = np.where(mask_np)[0]
            r_mean_valid = r_mean[mask_np]
            r_cov_valid = r_cov[np.ix_(valid_idx, valid_idx)]

            weights_valid = optimize_portfolio(r_mean_valid, r_cov_valid, method=method)
            weights = np.zeros(len(mask_np))
            weights[valid_idx] = weights_valid

            r_next_np = r_next.numpy() * returns_std
            mask_next_np = mask_next.numpy().astype(bool)

            # Ticker-level alignment to handle variable IBX universe size across days
            date_today = dataset.dates[idx]
            date_next = dataset.dates[idx + 1]
            tickers_today = list(dataset._returns_cache.get(date_today, {}).keys())
            tickers_next  = list(dataset._returns_cache.get(date_next,  {}).keys())
            today_to_idx = {t: i for i, t in enumerate(tickers_today)}
            next_to_idx  = {t: i for i, t in enumerate(tickers_next)}
            valid_today_set = {t for i, t in enumerate(tickers_today) if i < len(mask_np)      and mask_np[i]}
            valid_next_set  = {t for i, t in enumerate(tickers_next)  if i < len(mask_next_np) and mask_next_np[i]}
            common = sorted(valid_today_set & valid_next_set)

            if len(common) > 0:
                today_idx = [today_to_idx[t] for t in common]
                next_idx  = [next_to_idx[t]  for t in common]
                w_both = weights[today_idx]
                w_both = w_both / w_both.sum() if w_both.sum() > 0 else w_both
                port_return = np.dot(w_both, r_next_np[next_idx])
                portfolio_returns.append(port_return)
                dates.append(dataset.dates[idx + 1])

    returns_df = pd.DataFrame({'date': dates, 'return': portfolio_returns})

    print(f"\n✓ Backtest Complete")
    print(f"  Periods: {len(returns_df)}")
    print(f"  Date range: {returns_df['date'].min()} to {returns_df['date'].max()}")

    arr = returns_df['return'].values
    total_return = (1 + arr).prod() - 1
    ann_return = arr.mean() * 252
    ann_vol = arr.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0
    max_dd = compute_max_drawdown(arr)

    metrics = {
        'total_return': float(total_return),
        'annualized_return': float(ann_return),
        'annualized_vol': float(ann_vol),
        'sharpe_ratio': float(sharpe),
        'max_drawdown': float(max_dd),
    }

    # Benchmark comparison
    data_dir = Path(output_dir).parent.parent / "data"
    benchmark_df = load_ibovespa_returns(data_dir, returns_df['date'].min(), returns_df['date'].max())
    if benchmark_df is not None:
        merged = returns_df.merge(benchmark_df, on='date', suffixes=('_strategy', '_benchmark'))
        if len(merged) > 0:
            bench = merged['return_benchmark'].values
            strat = merged['return_strategy'].values
            excess = strat - bench
            bench_ann = bench.mean() * 252
            bench_vol = bench.std() * np.sqrt(252)
            excess_ann = excess.mean() * 252
            te = excess.std() * np.sqrt(252)
            metrics.update({
                'benchmark_total_return': float((1 + bench).prod() - 1),
                'benchmark_annualized_return': float(bench_ann),
                'benchmark_sharpe': float(bench_ann / bench_vol if bench_vol > 0 else 0),
                'benchmark_max_drawdown': float(compute_max_drawdown(bench)),
                'excess_return': float(excess_ann),
                'information_ratio': float(excess_ann / te if te > 0 else 0),
            })
            print(f"\n  Benchmark comparison:")
            print(f"    Ibovespa Ann. Return: {bench_ann:.2%}")
            print(f"    Excess Return: {excess_ann:.2%}")
            print(f"    Information Ratio: {metrics['information_ratio']:.2f}")

    print(f"\n  Performance Metrics:")
    print(f"    Total Return:        {total_return:.2%}")
    print(f"    Annualized Return:   {ann_return:.2%}")
    print(f"    Annualized Vol:      {ann_vol:.2%}")
    print(f"    Sharpe Ratio:        {sharpe:.2f}")
    print(f"    Max Drawdown:        {max_dd:.2%}")

    # Save
    returns_path = output_dir / "timeseries" / "backtest_returns.csv"
    returns_df.to_csv(returns_path, index=False)
    print(f"\n✓ Returns saved to: {returns_path}")

    metrics_path = output_dir / "metrics" / "backtest_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Metrics saved to: {metrics_path}")

    return returns_df, metrics


def plot_cumulative_returns(returns_df, output_dir, data_dir):
    """Plot cumulative strategy returns vs Ibovespa benchmark."""
    plt.figure(figsize=(12, 6))

    cum = (1 + returns_df['return']).cumprod()
    plt.plot(returns_df['date'], cum, label='NeuralFactors Min-Variance', linewidth=2, alpha=0.8)

    benchmark_df = load_ibovespa_returns(data_dir, returns_df['date'].min(), returns_df['date'].max())
    if benchmark_df is not None:
        merged = returns_df[['date']].merge(benchmark_df, on='date', how='left')
        merged['return'] = merged['return'].fillna(0)
        bench_cum = (1 + merged['return']).cumprod()
        plt.plot(returns_df['date'], bench_cum, label='Ibovespa', linewidth=2, alpha=0.8, linestyle='--')

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
