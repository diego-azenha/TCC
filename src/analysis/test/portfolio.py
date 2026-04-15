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


def compute_downside_std(returns, risk_free=0.0):
    """Annualised downside deviation (Sortino denominator)."""
    excess = np.asarray(returns) - risk_free / 252
    neg = excess[excess < 0]
    if len(neg) < 2:
        return 0.0
    return float(np.sqrt(np.mean(neg ** 2)) * np.sqrt(252))


def compute_turnover(w_new, w_prev):
    """One-way turnover: sum |w_new - w_prev_drifted|."""
    return float(np.sum(np.abs(w_new - w_prev)))


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
    ibov_path = Path(data_dir) / "ibovespa.csv"
    if not ibov_path.exists():
        print(f"Warning: Ibovespa data not found at {ibov_path}. Skipping benchmark.")
        return None

    try:
        df = pd.read_csv(ibov_path, sep=';', decimal=',', parse_dates=['date'])
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        df = df.sort_values('date').reset_index(drop=True)
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
    turnover_series = []
    max_weight_series = []
    eff_n_series = []
    prev_weights_by_ticker = {}  # {ticker: weight} from previous period

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

            alpha, B, sigma, mu_q, log_sigma_q = model.model.encode(S, S_static, r, mask)

            r_mean = decoder.marginal_mean(alpha[0], B[0])
            r_cov = decoder.marginal_covariance(B[0], sigma[0])

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

                # -- Concentration metrics --
                max_weight_series.append(float(w_both.max()))
                hhi = float(np.sum(w_both ** 2))
                eff_n_series.append(1.0 / hhi if hhi > 0 else len(w_both))

                # -- Turnover --
                new_w_dict = {t: w for t, w in zip(common, w_both)}
                if prev_weights_by_ticker:
                    all_tickers = set(new_w_dict) | set(prev_weights_by_ticker)
                    turnover = sum(
                        abs(new_w_dict.get(t, 0.0) - prev_weights_by_ticker.get(t, 0.0))
                        for t in all_tickers
                    )
                    turnover_series.append(turnover)
                prev_weights_by_ticker = new_w_dict

    returns_df = pd.DataFrame({
        'date': dates,
        'return': portfolio_returns,
        'eff_n': eff_n_series,
    })

    print(f"\n✓ Backtest Complete")
    print(f"  Periods: {len(returns_df)}")
    print(f"  Date range: {returns_df['date'].min()} to {returns_df['date'].max()}")

    arr = returns_df['return'].values
    ann = 252
    total_return = (1 + arr).prod() - 1
    ann_return = float((1 + total_return) ** (ann / len(arr)) - 1)
    ann_vol = arr.std() * np.sqrt(ann)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0
    max_dd = compute_max_drawdown(arr)
    downside = compute_downside_std(arr)
    sortino = ann_return / downside if downside > 0 else float('nan')
    calmar = ann_return / abs(max_dd) if max_dd != 0 else float('nan')

    # Turnover & concentration summaries
    avg_turnover = float(np.mean(turnover_series)) if turnover_series else float('nan')
    ann_turnover = avg_turnover * ann if turnover_series else float('nan')
    avg_max_weight = float(np.mean(max_weight_series)) if max_weight_series else float('nan')
    avg_eff_n = float(np.mean(eff_n_series)) if eff_n_series else float('nan')

    # Transaction costs (proportional, one-way)
    tc_bps = 10  # 10 bps per one-way trade
    tc_rate = tc_bps / 10_000
    tc_daily = [tc_rate * t for t in turnover_series] if turnover_series else []
    arr_net = arr.copy()
    if tc_daily:
        # First return has no turnover cost; subsequent returns are net of TC
        for i, tc in enumerate(tc_daily):
            arr_net[i + 1] -= tc
    total_return_net = float((1 + arr_net).prod() - 1)
    ann_return_net = float((1 + total_return_net) ** (ann / len(arr_net)) - 1)
    ann_vol_net = float(arr_net.std() * np.sqrt(ann))
    sharpe_net = ann_return_net / ann_vol_net if ann_vol_net > 0 else 0

    metrics = {
        'total_return': float(total_return),
        'annualized_return': float(ann_return),
        'annualized_vol': float(ann_vol),
        'sharpe_ratio': float(sharpe),
        'sortino_ratio': float(sortino),
        'calmar_ratio': float(calmar),
        'max_drawdown': float(max_dd),
        'avg_turnover': float(avg_turnover),
        'annualized_turnover': float(ann_turnover),
        'avg_max_weight': float(avg_max_weight),
        'avg_effective_n': float(avg_eff_n),
        'transaction_cost_bps': tc_bps,
        'total_return_net': float(total_return_net),
        'annualized_return_net': float(ann_return_net),
        'sharpe_ratio_net': float(sharpe_net),
    }

    # Benchmark comparison — ibovespa.csv lives in the top-level data/ folder
    data_dir = Path(__file__).parent.parent.parent.parent / "data"
    benchmark_df = load_ibovespa_returns(data_dir, returns_df['date'].min(), returns_df['date'].max())
    if benchmark_df is not None:
        merged = returns_df.merge(benchmark_df, on='date', suffixes=('_strategy', '_benchmark'))
        if len(merged) > 0:
            bench = merged['return_benchmark'].values
            strat = merged['return_strategy'].values
            excess = strat - bench
            bench_total = float((1 + bench).prod() - 1)
            bench_ann = float((1 + bench_total) ** (ann / len(bench)) - 1)
            bench_vol = bench.std() * np.sqrt(ann)
            excess_total = float((1 + excess).prod() - 1)
            excess_ann = float((1 + excess_total) ** (ann / len(excess)) - 1)
            te = excess.std() * np.sqrt(ann)
            metrics.update({
                'benchmark_total_return': float(bench_total),
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
    print(f"    Sortino Ratio:       {sortino:.2f}")
    print(f"    Calmar Ratio:        {calmar:.2f}")
    print(f"    Max Drawdown:        {max_dd:.2%}")
    print(f"    Avg Turnover/day:    {avg_turnover:.4f}")
    print(f"    Ann. Turnover:       {ann_turnover:.1f}x")
    print(f"    Avg Max Weight:      {avg_max_weight:.2%}")
    print(f"    Avg Effective N:     {avg_eff_n:.1f}")
    print(f"    Sharpe (net {tc_bps}bps):  {sharpe_net:.2f}")

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
    fig, ax = plt.subplots(figsize=(14, 7))

    cum = (1 + returns_df['return']).cumprod()
    ax.plot(returns_df['date'], cum, label='NeuralFactors Min-Variance',
            linewidth=1.8, color='#1f77b4')

    benchmark_df = load_ibovespa_returns(data_dir, returns_df['date'].min(), returns_df['date'].max())
    if benchmark_df is not None:
        merged = returns_df[['date']].merge(benchmark_df, on='date', how='left')
        merged['return'] = merged['return'].fillna(0)
        bench_cum = (1 + merged['return']).cumprod()
        ax.plot(returns_df['date'], bench_cum, label='Ibovespa',
                linewidth=1.8, linestyle='--', color='grey')

    ax.axhline(1.0, color='black', linewidth=0.8, linestyle=':')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Return (base 1)', fontsize=12)
    ax.set_title('NeuralFactors — Portfolio Backtest: Cumulative Returns',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    output_path = output_dir / "plots" / "cumulative_returns.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Cumulative returns plot saved to: {output_path}")


def plot_portfolio_diagnostics(returns_df, output_dir, data_dir):
    """Three-panel portfolio diagnostic figure.

    Panel 1: Cumulative returns (NeuralFactors vs Ibovespa).
    Panel 2: Rolling 60-day annualised Sharpe ratio.
    Panel 3: Effective number of stocks (1/HHI) over time.

    Args:
        returns_df: DataFrame with columns [date, return, eff_n].
        output_dir: Base Path for results (plots/ subdir is used).
        data_dir: Root data directory (for Ibovespa lookup).
    """
    output_dir = Path(output_dir)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    arr     = returns_df['return'].values
    dates   = returns_df['date']
    eff_n   = returns_df['eff_n'].values if 'eff_n' in returns_df.columns else None

    # Rolling Sharpe — auto-reduce window if period is short
    n_obs = len(arr)
    roll_window = min(60, max(10, n_obs // 3))
    s_ret = pd.Series(arr, index=dates)
    roll_mean = s_ret.rolling(roll_window).mean()
    roll_std  = s_ret.rolling(roll_window).std()
    roll_ann  = np.sqrt(252)
    rolling_sharpe = (roll_mean * 252) / (roll_std * roll_ann)
    rolling_sharpe = rolling_sharpe.where(s_ret.rolling(roll_window).count() >= roll_window)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=True)
    fig.suptitle('Portfolio Diagnostics', fontsize=16, fontweight='bold')

    # ── Panel 1: Cumulative returns ───────────────────────────────────────────
    ax1 = axes[0]
    cum = (1 + arr).cumprod()
    ax1.plot(dates, cum, label='NeuralFactors Min-Variance',
             linewidth=1.8, color='#1f77b4')

    benchmark_df = load_ibovespa_returns(data_dir, dates.min(), dates.max())
    if benchmark_df is not None:
        merged = returns_df[['date']].merge(benchmark_df, on='date', how='left')
        merged['return'] = merged['return'].fillna(0)
        bench_cum = (1 + merged['return']).cumprod()
        ax1.plot(dates, bench_cum, label='Ibovespa',
                 linewidth=1.8, linestyle='--', color='grey')

    ax1.axhline(1.0, color='black', linewidth=0.8, linestyle=':')
    ax1.set_ylabel('Cumulative Return (base 1)', fontsize=11)
    ax1.set_title('Cumulative Returns', fontsize=12)
    ax1.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: Rolling Sharpe ───────────────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(rolling_sharpe.index, rolling_sharpe.values,
             color='#2ca02c', linewidth=1.5,
             label=f'Rolling {roll_window}-day Ann. Sharpe')
    ax2.axhline(0.0, color='red', linestyle='--', linewidth=1.0, alpha=0.7)
    ax2.fill_between(
        rolling_sharpe.index,
        rolling_sharpe.values,
        0,
        where=rolling_sharpe.values > 0,
        alpha=0.15, color='#2ca02c',
        interpolate=True,
    )
    ax2.fill_between(
        rolling_sharpe.index,
        rolling_sharpe.values,
        0,
        where=rolling_sharpe.values < 0,
        alpha=0.15, color='#d62728',
        interpolate=True,
    )
    ax2.set_ylabel('Ann. Sharpe Ratio', fontsize=11)
    ax2.set_title(f'Rolling {roll_window}-day Annualised Sharpe', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: Effective N ──────────────────────────────────────────────────
    ax3 = axes[2]
    if eff_n is not None:
        ax3.plot(dates, eff_n, color='#9467bd', linewidth=1.5,
                 label='Effective N  (1/HHI)')
        # "Concentrated signal" reference: N/4 where N ≈ median effective N
        n_ref = float(np.nanmedian(eff_n)) / 2.0 if np.any(~np.isnan(eff_n)) else None
        if n_ref is not None:
            ax3.axhline(n_ref, color='darkorange', linestyle='--', linewidth=1.2,
                        label=f'Median/2 reference ({n_ref:.0f})')
    else:
        ax3.text(0.5, 0.5, 'eff_n not available', transform=ax3.transAxes,
                 ha='center', va='center', fontsize=12)

    ax3.set_ylabel('Effective number of stocks', fontsize=11)
    ax3.set_title(
        'Portfolio Concentration — 1/HHI\n'
        '(Low = concentrated; equal-weight at N stocks → N)',
        fontsize=12,
    )
    ax3.legend(fontsize=10)
    ax3.set_xlabel('Date', fontsize=11)
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = plot_dir / "portfolio_diagnostics.png"
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Portfolio diagnostics saved to: {out_path}")
