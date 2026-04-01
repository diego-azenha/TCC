"""Portfolio backtest and optimization for PPCA evaluation.

Minimum-variance portfolio using the PPCA model covariance W W^T + sigma2 I.
Ticker alignment and benchmark loading mirror src/analysis/portfolio.py exactly.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tqdm import tqdm

from PPCA import model as ppca_model


# =============================================================================
# Helper utilities
# =============================================================================

def compute_max_drawdown(returns: np.ndarray) -> float:
    """Max drawdown of a return series (as a negative fraction)."""
    cumulative = (1 + returns).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    return float(drawdown.min())


def compute_downside_std(returns: np.ndarray, risk_free: float = 0.0) -> float:
    """Annualised downside deviation (Sortino denominator)."""
    excess = np.asarray(returns) - risk_free / 252
    neg = excess[excess < 0]
    if len(neg) < 2:
        return 0.0
    return float(np.sqrt(np.mean(neg ** 2)) * np.sqrt(252))


def optimize_portfolio(r_cov: np.ndarray) -> np.ndarray:
    """Min-variance portfolio weights: argmin w^T Sigma w  s.t. sum(w)=1, w>=0.

    Parameters
    ----------
    r_cov : (N, N) covariance matrix (denormalised, real return scale)

    Returns
    -------
    (N,) weight vector summing to 1 (falls back to equal-weight on failure)
    """
    N = r_cov.shape[0]
    w0 = np.ones(N) / N

    def objective(w):
        return w @ r_cov @ w

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = [(0.0, 1.0)] * N
    try:
        res = minimize(objective, w0, method="SLSQP",
                       bounds=bounds, constraints=constraints,
                       options={"maxiter": 1000})
        return res.x if res.success else w0
    except Exception as e:
        print(f"    Warning: Optimisation failed ({e}), using equal weight")
        return w0


def load_ibovespa_returns(
    data_dir: str | Path,
    start_date: str,
    end_date: str,
) -> Optional[pd.DataFrame]:
    """Load Ibovespa benchmark daily returns.

    Returns pd.DataFrame [date, return] or None if file not found.
    """
    ibov_path = Path(data_dir) / "ibovespa.csv"
    if not ibov_path.exists():
        print(f"  Warning: Ibovespa data not found at {ibov_path}. Skipping benchmark.")
        return None
    try:
        df = pd.read_csv(ibov_path, sep=";", decimal=",", parse_dates=["date"])
        df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
        df = df.sort_values("date").reset_index(drop=True)
        return df[["date", "return"]].dropna()
    except Exception as e:
        print(f"  Warning: Error loading Ibovespa data: {e}")
        return None


# =============================================================================
# Main backtest
# =============================================================================

def compute_portfolio_metrics(
    returns_wide: pd.DataFrame,
    eval_indices: List[int],
    F: int,
    window_size: int,
    mode: str,
    returns_std: float,
    output_dir: Path,
    data_dir: Path | str = "data",
) -> Tuple[pd.DataFrame, dict]:
    """Run minimum-variance portfolio backtest using PPCA-estimated covariance.

    For each consecutive pair of evaluation dates (today, tomorrow):
    1. Fit PPCA on ``[idx-window_size : idx]`` using tickers valid for the window.
    2. Build covariance ``W W^T + sigma2 I`` (denormalised).
    3. Optimise min-variance weights on today's valid-ticker universe.
    4. Realise the portfolio return using the intersection of tickers available
       on both today and tomorrow.

    Parameters
    ----------
    returns_wide : (T, N) normalised-returns DataFrame
    eval_indices : positional row indices for the split
    F            : number of PPCA factors
    window_size  : look-back window for fitting
    mode         : ``'debug'`` or ``'paper'``
    returns_std  : normalisation std (for denormalisation)
    output_dir   : root directory of this experiment (used to infer data_dir)

    Returns
    -------
    returns_df      : pd.DataFrame [date, return]
    backtest_metrics: dict of scalar performance metrics
    """
    print("\n" + "=" * 80)
    print("PORTFOLIO BACKTEST")
    print("=" * 80)

    max_dates = 50 if mode == "debug" else None
    print(f"Portfolio method: min_variance  |  F = {F}  |  Window = {window_size}")
    if max_dates:
        print(f"Debug mode: Processing first {max_dates} dates")

    # Work with consecutive pairs from eval_indices
    pairs = list(zip(eval_indices[:-1], eval_indices[1:]))
    if max_dates:
        pairs = pairs[:max_dates]

    portfolio_returns = []
    dates_out = []
    turnover_series = []
    max_weight_series = []
    eff_n_series = []
    prev_weights_by_ticker = {}  # {ticker: weight} from previous period

    for idx_today, idx_next in tqdm(pairs, desc="Running Backtest"):
        window = returns_wide.iloc[idx_today - window_size : idx_today]

        # Fit tickers: must be valid across the whole fitting window
        valid_fit = window.columns[window.notna().all(axis=0)].tolist()
        if len(valid_fit) < F + 2:
            continue

        # Today's and next-day's returns
        today_row = returns_wide.iloc[idx_today]
        next_row  = returns_wide.iloc[idx_next]

        valid_today = set(t for t in valid_fit if pd.notna(today_row[t]))
        valid_next  = set(returns_wide.columns[next_row.notna()])
        common = sorted(valid_today & valid_next)
        if len(common) < 2:
            continue

        R_fit = window[valid_fit].values.astype(np.float64)
        try:
            fitted = ppca_model.fit(R_fit, F, valid_fit)
        except Exception as e:
            print(f"\n  Warning (fit) at {returns_wide.index[idx_today]}: {e}")
            continue

        # Covariance submatrix for common tickers (denormalised)
        ticker_to_idx = {t: i for i, t in enumerate(valid_fit)}
        common_in_fit = [t for t in common if t in ticker_to_idx]
        if len(common_in_fit) < 2:
            continue
        fit_idx = [ticker_to_idx[t] for t in common_in_fit]
        W_sub   = fitted.W[fit_idx, :]
        cov_sub = (W_sub @ W_sub.T + fitted.sigma2 * np.eye(len(common_in_fit))) * (returns_std ** 2)

        weights = optimize_portfolio(cov_sub)

        # Realised return: next-day actual returns (denormalised)
        r_next  = next_row[common_in_fit].values.astype(np.float64) * returns_std
        port_r  = float(np.dot(weights, r_next))
        portfolio_returns.append(port_r)
        dates_out.append(returns_wide.index[idx_next])

        # -- Concentration metrics --
        max_weight_series.append(float(weights.max()))
        hhi = float(np.sum(weights ** 2))
        eff_n_series.append(1.0 / hhi if hhi > 0 else len(weights))

        # -- Turnover --
        new_w_dict = {t: float(w) for t, w in zip(common_in_fit, weights)}
        if prev_weights_by_ticker:
            all_tickers = set(new_w_dict) | set(prev_weights_by_ticker)
            turnover = sum(
                abs(new_w_dict.get(t, 0.0) - prev_weights_by_ticker.get(t, 0.0))
                for t in all_tickers
            )
            turnover_series.append(turnover)
        prev_weights_by_ticker = new_w_dict

    returns_df = pd.DataFrame({"date": dates_out, "return": portfolio_returns})

    print(f"\n  Backtest complete — {len(returns_df)} periods")
    if returns_df.empty:
        return returns_df, {}

    print(f"  Date range: {returns_df['date'].min().date()} to {returns_df['date'].max().date()}")

    arr = returns_df["return"].values
    ann = 252
    total_return    = float((1 + arr).prod() - 1)
    ann_return      = float((1 + total_return) ** (ann / len(arr)) - 1)
    ann_vol         = float(arr.std() * np.sqrt(ann))
    sharpe          = ann_return / ann_vol if ann_vol > 0 else np.nan
    max_dd          = compute_max_drawdown(arr)
    downside        = compute_downside_std(arr)
    sortino         = ann_return / downside if downside > 0 else float("nan")
    calmar          = ann_return / abs(max_dd) if max_dd != 0 else float("nan")

    # Turnover & concentration summaries
    avg_turnover  = float(np.mean(turnover_series)) if turnover_series else float("nan")
    ann_turnover  = avg_turnover * ann if turnover_series else float("nan")
    avg_max_weight = float(np.mean(max_weight_series)) if max_weight_series else float("nan")
    avg_eff_n     = float(np.mean(eff_n_series)) if eff_n_series else float("nan")

    # Transaction costs (proportional, one-way)
    tc_bps = 10  # 10 bps per one-way trade
    tc_rate = tc_bps / 10_000
    tc_daily = [tc_rate * t for t in turnover_series] if turnover_series else []
    arr_net = arr.copy()
    if tc_daily:
        for i, tc in enumerate(tc_daily):
            arr_net[i + 1] -= tc
    total_return_net = float((1 + arr_net).prod() - 1)
    ann_return_net   = float((1 + total_return_net) ** (ann / len(arr_net)) - 1)
    ann_vol_net      = float(arr_net.std() * np.sqrt(ann))
    sharpe_net       = ann_return_net / ann_vol_net if ann_vol_net > 0 else float("nan")

    metrics: dict = {
        "total_return":         total_return,
        "annualized_return":    ann_return,
        "annualized_vol":       ann_vol,
        "sharpe_ratio":         sharpe,
        "sortino_ratio":        sortino,
        "calmar_ratio":         calmar,
        "max_drawdown":         max_dd,
        "avg_turnover":         avg_turnover,
        "annualized_turnover":  ann_turnover,
        "avg_max_weight":       avg_max_weight,
        "avg_effective_n":      avg_eff_n,
        "transaction_cost_bps": tc_bps,
        "total_return_net":     total_return_net,
        "annualized_return_net": ann_return_net,
        "sharpe_ratio_net":     sharpe_net,
    }

    print(f"  Total Return:      {total_return:.2%}")
    print(f"  Annualised Return: {ann_return:.2%}")
    print(f"  Annualised Vol:    {ann_vol:.2%}")
    print(f"  Sharpe Ratio:      {sharpe:.2f}")
    print(f"  Sortino Ratio:     {sortino:.2f}")
    print(f"  Calmar Ratio:      {calmar:.2f}")
    print(f"  Max Drawdown:      {max_dd:.2%}")
    print(f"  Avg Turnover/day:  {avg_turnover:.4f}")
    print(f"  Ann. Turnover:     {ann_turnover:.1f}x")
    print(f"  Avg Max Weight:    {avg_max_weight:.2%}")
    print(f"  Avg Effective N:   {avg_eff_n:.1f}")
    print(f"  Sharpe (net {tc_bps}bps): {sharpe_net:.2f}")

    # Benchmark
    start_str = str(returns_df["date"].min().date())
    end_str   = str(returns_df["date"].max().date())
    ibov_df = load_ibovespa_returns(Path(data_dir), start_str, end_str)

    if ibov_df is not None:
        ibov_arr = ibov_df["return"].values
        ibov_total  = float((1 + ibov_arr).prod() - 1)
        ibov_ann    = float((1 + ibov_total) ** (ann / len(ibov_arr)) - 1)
        ibov_vol    = float(ibov_arr.std() * np.sqrt(ann))
        ibov_sharpe = ibov_ann / ibov_vol if ibov_vol > 0 else np.nan
        excess_returns = arr - np.interp(
            np.arange(len(arr)), np.arange(len(ibov_arr)), ibov_arr
        )
        tracking_err = float(excess_returns.std() * np.sqrt(ann))
        ir = (ann_return - ibov_ann) / tracking_err if tracking_err > 0 else np.nan
        metrics.update({
            "benchmark_total_return":      ibov_total,
            "benchmark_annualized_return": ibov_ann,
            "benchmark_sharpe":            ibov_sharpe,
            "excess_return":               ann_return - ibov_ann,
            "information_ratio":           ir,
        })

    # Save portfolio returns CSV
    csv_path = output_dir / "metrics" / "portfolio_returns.csv"
    returns_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    return returns_df, metrics


def plot_cumulative_returns(
    returns_df: pd.DataFrame,
    output_dir: Path,
    data_dir: Path,
) -> None:
    """Plot cumulative portfolio return vs Ibovespa benchmark."""
    fig, ax = plt.subplots(figsize=(14, 7))

    cumret = (1 + returns_df["return"].values).cumprod()
    ax.plot(returns_df["date"], cumret, label="PPCA Min-Variance",
            linewidth=1.8, color="#1f77b4")

    start_str = str(returns_df["date"].min().date())
    end_str   = str(returns_df["date"].max().date())
    ibov_df = load_ibovespa_returns(data_dir, start_str, end_str)
    if ibov_df is not None:
        ibov_cum = (1 + ibov_df["return"].values).cumprod()
        ax.plot(ibov_df["date"], ibov_cum, label="Ibovespa", linewidth=1.8,
                linestyle="--", color="grey")

    ax.axhline(1.0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Cumulative Return (base 1)", fontsize=12)
    ax.set_title("PPCA — Portfolio Backtest: Cumulative Returns",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path = output_dir / "plots" / "cumulative_returns.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {output_path}")
