"""Data loading and splitting utilities for PPCA evaluation.

Loads prices from the same CSV used by NeuralFactors, computes log returns,
pivots to wide format (date × ticker), normalises by training-period std,
and provides positional indices for each evaluation split.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


# =============================================================================
# Public API
# =============================================================================

def load_returns_wide(
    data_dir: str | Path,
    train_end: str = "2018-12-31",
    val_end: str = "2022-12-31",
) -> Tuple[pd.DataFrame, float]:
    """Load prices, compute normalised log returns in wide format.

    Processing steps
    ----------------
    1. Read ``data/cleaned/fechamentos_ibx.csv`` (semicolon-delimited,
       comma decimals, date column ``DATES`` in DD/MM/YYYY).
    2. Compute log returns: log(price_t / price_{t-1}).
    3. Replace ±Inf / NaN with NaN (handles zero / missing prices).
    4. Pivot to wide format: rows = trading dates, columns = tickers.
    5. Compute ``returns_std`` = std of all finite returns in the
       **training period** (replicates NeuralFactors normalisation).
    6. Divide every return by ``returns_std``.

    Parameters
    ----------
    data_dir  : root data directory (contains ``cleaned/`` subfolder)
    train_end : last date of training period (inclusive, ISO format)
    val_end   : last date of validation period (inclusive, ISO format)

    Returns
    -------
    returns_wide : pd.DataFrame  shape (T, N), index = date, cols = tickers
                   Values are normalised log returns.  Missing entries are NaN.
    returns_std  : float  std used for normalisation (denormalise by * this)
    """
    prices_path = Path(data_dir) / "cleaned" / "fechamentos_ibx.csv"
    if not prices_path.exists():
        raise FileNotFoundError(f"Prices file not found: {prices_path}")

    # ------------------------------------------------------------------ load
    df = pd.read_csv(
        prices_path,
        sep=";",
        decimal=",",
        parse_dates=["DATES"],
        dayfirst=True,
    )
    df.rename(columns={"DATES": "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    df.set_index("date", inplace=True)

    # Ensure numeric (column values might be strings with comma already removed)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.replace(",", ".", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --------------------------------------------------- log returns (wide)
    prices_wide = df  # (T_prices, N)
    log_returns = np.log(prices_wide / prices_wide.shift(1))
    log_returns.replace([np.inf, -np.inf], np.nan, inplace=True)
    log_returns = log_returns.iloc[1:].copy()   # drop first NaN row

    # --------------------------------------------------- returns_std from train
    train_end_dt = pd.to_datetime(train_end)
    train_mask = log_returns.index <= train_end_dt
    train_vals = log_returns[train_mask].values.flatten()
    train_vals = train_vals[np.isfinite(train_vals)]
    if len(train_vals) == 0:
        raise RuntimeError("No valid training returns found — check data file and train_end.")
    returns_std = float(np.std(train_vals))

    # --------------------------------------------------- normalise
    returns_wide = log_returns / returns_std

    return returns_wide, returns_std


def get_eval_indices(
    returns_wide: pd.DataFrame,
    split: str,
    train_end: str = "2018-12-31",
    val_end: str = "2022-12-31",
    window_size: int = 252,
) -> list[int]:
    """Return positional row indices for dates in *split* that have a full window.

    A date at position ``idx`` is included only if
    ``idx >= window_size`` (so rows ``idx-window_size : idx`` all exist).

    Parameters
    ----------
    returns_wide : wide returns DataFrame (output of :func:`load_returns_wide`)
    split        : one of ``'train'``, ``'val'``, ``'test'``
    train_end    : last date of training split (ISO)
    val_end      : last date of validation split (ISO)
    window_size  : number of prior rows required

    Returns
    -------
    list of int  (positional indices into ``returns_wide``)
    """
    train_end_dt = pd.to_datetime(train_end)
    val_end_dt   = pd.to_datetime(val_end)
    dates = returns_wide.index  # DatetimeIndex

    if split == "train":
        mask = dates <= train_end_dt
    elif split == "val":
        mask = (dates > train_end_dt) & (dates <= val_end_dt)
    elif split == "test":
        mask = dates > val_end_dt
    else:
        raise ValueError(f"split must be 'train', 'val', or 'test'; got '{split}'.")

    indices = [
        i for i, in_split in enumerate(mask)
        if in_split and i >= window_size
    ]
    return indices


def setup_output_dirs(output_dir: str | Path, experiment_name: str) -> Path:
    """Create result subdirectories and return the experiment root Path."""
    root = Path(output_dir) / experiment_name
    (root / "metrics").mkdir(parents=True, exist_ok=True)
    (root / "plots").mkdir(parents=True, exist_ok=True)
    print(f"Output directories created under: {root}")
    return root
