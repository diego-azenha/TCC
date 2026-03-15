"""Covariance prediction metrics for PPCA evaluation.

Predicted covariance: W W^T + sigma2 I  (from the PPCA fit on a 252-day window).
Empirical covariance: np.cov() over the last 20 days of the same window.

The 20-day empirical window matches the NeuralFactors evaluation protocol
in src/analysis/covariance.py for a fair comparison.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from PPCA import model as ppca_model


def compute_covariance_metrics(
    returns_wide: pd.DataFrame,
    eval_indices: List[int],
    F: int,
    window_size: int,
    mode: str,
    returns_std: float,
) -> pd.DataFrame:
    """Compare PPCA-predicted covariance against 20-day empirical covariance.

    For each date at positional index ``idx``:
    1. Fit PPCA on rows ``[idx-window_size : idx]`` (valid tickers only).
    2. Predicted cov = ``W W^T + sigma2 I`` denormalised by ``returns_std²``.
    3. Empirical cov = ``np.cov`` of the **last 20 rows** of the same window,
       restricted to tickers with non-NaN values in those 20 rows.
    4. MSE over the upper-triangle elements of the common submatrix.

    Parameters
    ----------
    returns_wide : (T, N) normalised-returns DataFrame
    eval_indices : positional indices in the split
    F            : number of PPCA factors
    window_size  : look-back window for fitting
    mode         : ``'debug'`` or ``'paper'``
    returns_std  : normalisation std (used to denormalise covariances)

    Returns
    -------
    pd.DataFrame  columns: [date, mse_cov, n_stocks]
    """
    print("\n" + "=" * 80)
    print("COMPUTING COVARIANCE METRICS")
    print("=" * 80)

    EMP_WINDOW = 20   # same as NeuralFactors src/analysis/covariance.py
    print(f"Empirical window: {EMP_WINDOW} days  |  Fitting window: {window_size} days")

    max_dates = 50 if mode == "debug" else None
    if max_dates:
        print(f"Debug mode: Processing first {max_dates} dates")

    indices = eval_indices[:max_dates] if max_dates else eval_indices
    results = []

    for idx in tqdm(indices, desc="Computing Covariance"):
        window = returns_wide.iloc[idx - window_size : idx]   # (window_size, N)

        # Tickers valid throughout the FULL fitting window
        valid_fit = window.columns[window.notna().all(axis=0)].tolist()
        if len(valid_fit) < F + 2:
            continue

        # Tickers valid in the last EMP_WINDOW rows (for empirical cov)
        emp_window = window[valid_fit].iloc[-EMP_WINDOW:]
        valid_emp = emp_window.columns[emp_window.notna().all(axis=0)].tolist()
        if len(valid_emp) < 2:
            continue

        today = returns_wide.iloc[idx][valid_emp]
        if today.isna().any():
            continue

        R_fit = window[valid_fit].values.astype(np.float64)      # (window_size, n_fit)
        try:
            fitted = ppca_model.fit(R_fit, F, valid_fit)
        except Exception as e:
            print(f"\nWarning (fit) at {returns_wide.index[idx]}: {e}")
            continue

        # Predicted cov for the emp-ticker submatrix
        ticker_to_idx = {t: i for i, t in enumerate(valid_fit)}
        emp_idx = [ticker_to_idx[t] for t in valid_emp]
        W_sub = fitted.W[emp_idx, :]                              # (n_emp, F)
        cov_pred = W_sub @ W_sub.T + fitted.sigma2 * np.eye(len(valid_emp))
        cov_pred *= returns_std ** 2                              # denormalise

        # Empirical cov (denormalised)
        R_emp = emp_window[valid_emp].values.astype(np.float64) * returns_std  # real returns
        cov_emp = np.cov(R_emp, rowvar=False)

        # MSE over upper triangle (including diagonal)
        mask = np.triu(np.ones_like(cov_pred, dtype=bool))
        mse = float(np.mean((cov_pred[mask] - cov_emp[mask]) ** 2))

        results.append({
            "date":     returns_wide.index[idx],
            "mse_cov":  mse,
            "n_stocks": len(valid_emp),
        })

    df = pd.DataFrame(results)
    if not df.empty:
        print(f"\n  Dates processed: {len(df)}")
        print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        print(f"  Mean MSE: {df['mse_cov'].mean():.6f}")
        print(f"  Std MSE:  {df['mse_cov'].std():.6f}")
    else:
        print("\nWarning: No covariance metrics computed")
    return df


def save_cov_results(cov_df: pd.DataFrame, output_dir: Path) -> None:
    """Save covariance results to CSV."""
    output_path = output_dir / "metrics" / "covariance_results.csv"
    cov_df.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")


def plot_cov_metrics(cov_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot covariance MSE over time."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(cov_df["date"], cov_df["mse_cov"], alpha=0.7, linewidth=1.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("MSE")
    ax.set_title("PPCA — Covariance Prediction MSE Over Time")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = output_dir / "plots" / "covariance_mse.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {output_path}")
