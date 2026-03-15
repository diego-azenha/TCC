"""Negative Log-Likelihood metrics for PPCA evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from PPCA import model as ppca_model


def compute_nll_metrics(
    returns_wide: pd.DataFrame,
    eval_indices: List[int],
    F: int,
    window_size: int,
    mode: str,
    returns_std: float,
) -> pd.DataFrame:
    """Compute NLL_joint and NLL_ind for each evaluation date.

    For each date at positional index ``idx``:
    1. Fit PPCA on rows ``[idx-window_size : idx]`` (using only tickers with
       a full window of non-NaN values).
    2. Evaluate ``log p(r_today)`` via the Woodbury identity.
    3. Record ``nll_joint = -log_p`` and ``nll_ind = nll_joint / n_stocks``.

    Parameters
    ----------
    returns_wide  : (T, N) normalised-returns DataFrame
    eval_indices  : positional indices in the split (from loader.get_eval_indices)
    F             : number of PPCA factors
    window_size   : look-back window for fitting (rows before today)
    mode          : ``'debug'`` (first 50 dates) or ``'paper'`` (all)
    returns_std   : scalar used for normalisation (kept for API consistency)

    Returns
    -------
    pd.DataFrame  columns: [date, nll_joint, nll_ind, n_stocks]
    """
    print("\n" + "=" * 80)
    print("COMPUTING NEGATIVE LOG-LIKELIHOOD METRICS")
    print("=" * 80)

    max_dates = 50 if mode == "debug" else None
    if max_dates:
        print(f"Debug mode: Processing first {max_dates} dates")

    indices = eval_indices[:max_dates] if max_dates else eval_indices
    results = []

    for idx in tqdm(indices, desc="Computing NLL"):
        window = returns_wide.iloc[idx - window_size : idx]   # (window_size, N)

        # Keep only tickers with a full window of valid returns
        valid_tickers = window.columns[window.notna().all(axis=0)].tolist()
        if len(valid_tickers) < F + 2:
            continue   # not enough stocks to fit F factors

        # Today's returns for the same valid tickers
        today = returns_wide.iloc[idx][valid_tickers]
        if today.isna().any():
            continue

        R = window[valid_tickers].values.astype(np.float64)    # (window_size, n)
        r = today.values.astype(np.float64)                    # (n,)

        try:
            fitted = ppca_model.fit(R, F, valid_tickers)
            log_p = ppca_model.log_prob(fitted, r)
            nll_joint = -log_p
            n_stocks = len(valid_tickers)
            results.append({
                "date":      returns_wide.index[idx],
                "nll_joint": nll_joint,
                "nll_ind":   nll_joint / n_stocks,
                "n_stocks":  n_stocks,
            })
        except Exception as e:
            print(f"\nWarning at {returns_wide.index[idx]}: {e}")

    df = pd.DataFrame(results)
    if not df.empty:
        print(f"\n  Dates processed: {len(df)}")
        print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        print(f"  Mean NLL_joint: {df['nll_joint'].mean():.4f}")
        print(f"  Std NLL_joint:  {df['nll_joint'].std():.4f}")
        print(f"  Mean NLL_ind:   {df['nll_ind'].mean():.4f}")
    else:
        print("\nWarning: No NLL metrics computed")
    return df


def save_nll_results(nll_df: pd.DataFrame, output_dir: Path) -> None:
    """Save NLL results to CSV."""
    output_path = output_dir / "metrics" / "nll_timeseries.csv"
    nll_df.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")


def plot_nll_timeseries(nll_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot NLL_joint and NLL_ind over time."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(nll_df["date"], nll_df["nll_joint"], alpha=0.7, linewidth=1.5)
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("NLL Joint")
    axes[0].set_title("PPCA — Negative Log-Likelihood (Joint) Over Time")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(nll_df["date"], nll_df["nll_ind"], color="orange", alpha=0.7, linewidth=1.5)
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("NLL per Stock")
    axes[1].set_title("PPCA — NLL per Stock Over Time")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "plots" / "nll_timeseries.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {output_path}")
