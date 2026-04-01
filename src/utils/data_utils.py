"""Data preprocessing utilities for NeuralFactors training.

Functions for loading parquets, computing returns, building lookback windows,
and splitting data according to paper specifications (Section 5).

All data inputs are parquet files produced by the external processing pipeline:
- x_ts.parquet: long format [date, ticker, 38 normalized features]
- x_static.parquet: [ticker, one-hot sector columns] (no date column)
- prices.parquet: long format [date, ticker, close] (raw adjusted prices)
- normalization_stats.json: returns_std, feature_order, fundamental/index stats
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
import torch


def discover_feature_dims(x_ts_path: str, x_static_path: str) -> Tuple[int, int]:
    """Discover feature dimensions from parquet files.
    
    Args:
        x_ts_path: Path to time-series features parquet
        x_static_path: Path to static features parquet (no date column)
        
    Returns:
        Tuple of (d_ts, d_static) - feature dimensions
    """
    df_ts = pd.read_parquet(x_ts_path, engine='pyarrow')
    df_static = pd.read_parquet(x_static_path, engine='pyarrow')
    
    # x_ts has [date, ticker, feature_1, ...]; x_static has [ticker, feature_1, ...]
    d_ts = len([col for col in df_ts.columns if col not in ['date', 'ticker']])
    d_static = len([col for col in df_static.columns if col not in ['ticker']])
    
    return d_ts, d_static


def load_normalization_stats(stats_path: str) -> dict:
    """Load normalization statistics from JSON file.

    Args:
        stats_path: Path to normalization_stats.json

    Returns:
        Dict with keys: returns_std, feature_order, d_ts, fundamental_stats,
        index_stats, train_period, n_tickers_total, n_tickers_train
    """
    with open(stats_path, 'r') as f:
        return json.load(f)


def load_parquets(
    x_ts_path: str,
    x_static_path: str,
    prices_path: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """Load parquet files and ensure proper datetime indexing.
    
    Args:
        x_ts_path: Path to time-series features parquet (long: date, ticker, features)
        x_static_path: Path to static features parquet (ticker, one-hot sectors — no date)
        prices_path: Optional path to prices parquet (long: date, ticker, close)
        
    Returns:
        Tuple of (df_ts, df_static, df_prices)
    """
    df_ts = pd.read_parquet(x_ts_path, engine='pyarrow')
    df_static = pd.read_parquet(x_static_path, engine='pyarrow')
    
    # Ensure date column is datetime (x_ts has date; x_static does NOT)
    df_ts['date'] = pd.to_datetime(df_ts['date'])
    
    df_prices = None
    if prices_path:
        df_prices = pd.read_parquet(prices_path, engine='pyarrow')
        df_prices['date'] = pd.to_datetime(df_prices['date'])
    
    return df_ts, df_static, df_prices


def split_by_date(
    df: pd.DataFrame,
    train_end: str = '2018-12-31',
    val_end: str = '2022-12-31'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataframe by date ranges.
    
    Default splits for IBX data (2005-2025):
    Training: 2005-01-01 to 2018-12-31 (14 years)
    Validation: 2019-01-01 to 2022-12-31 (4 years)
    Test: 2023-01-01 to 2025-11-04 (2.8 years)
    
    Args:
        df: DataFrame with 'date' column
        train_end: End date for training set (inclusive)
        val_end: End date for validation set (inclusive)
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    train_end_dt = pd.to_datetime(train_end)
    val_end_dt = pd.to_datetime(val_end)
    
    train_df = df[df['date'] <= train_end_dt].copy()
    val_df = df[(df['date'] > train_end_dt) & (df['date'] <= val_end_dt)].copy()
    test_df = df[df['date'] > val_end_dt].copy()
    
    return train_df, val_df, test_df


def compute_returns(prices_df: pd.DataFrame, log_returns: bool = True) -> pd.DataFrame:
    """Compute returns from long-format prices.
    
    Args:
        prices_df: DataFrame with columns [date, ticker, close]
        log_returns: If True, compute log returns; else simple returns
        
    Returns:
        DataFrame with columns [date, ticker, return]
    """
    df = prices_df.sort_values(['ticker', 'date']).copy()

    if log_returns:
        with np.errstate(divide='ignore', invalid='ignore'):
            df['return'] = df.groupby('ticker')['close'].transform(
                lambda s: np.log(s / s.shift(1))
            )
    else:
        df['return'] = df.groupby('ticker')['close'].transform(
            lambda s: s / s.shift(1) - 1.0
        )

    # Drop the first observation per ticker (NaN from shift)
    df = df.dropna(subset=['return']).copy()

    # Replace ±Inf with NaN (happens when price is 0 or negative)
    df['return'] = df['return'].replace([np.inf, -np.inf], np.nan)

    return df[['date', 'ticker', 'return']].reset_index(drop=True)


def normalize_returns(
    returns_df: pd.DataFrame,
    std_value: Optional[float] = None,
    compute_std_from_data: bool = False
) -> Tuple[pd.DataFrame, float]:
    """Normalize returns by dividing by standard deviation.
    
    Args:
        returns_df: Long-format DataFrame with columns [date, ticker, return]
        std_value: Standard deviation to divide by (if None, computed from data)
        compute_std_from_data: If True, compute std from data ignoring std_value
        
    Returns:
        Tuple of (normalized_returns_df, std_used)
    """
    if compute_std_from_data or std_value is None:
        all_returns = returns_df['return'].dropna().values
        std_used = float(np.std(all_returns))
    else:
        std_used = std_value
    
    normalized_df = returns_df.copy()
    normalized_df['return'] = returns_df['return'] / std_used
    
    return normalized_df, std_used


def build_lookback_tensor(
    df_group: pd.DataFrame,
    feature_cols: list,
    lookback: int = 256
) -> Optional[torch.Tensor]:
    """Build lookback window tensor for a single stock.
    
    Args:
        df_group: DataFrame for single ticker, sorted by date
        feature_cols: List of feature column names
        lookback: Number of timesteps to look back
        
    Returns:
        Tensor of shape [lookback, d_ts] or None if insufficient data
    """
    if len(df_group) < lookback:
        return None
    
    # Get last 'lookback' rows
    window_df = df_group.iloc[-lookback:]
    
    # Extract features as tensor
    feature_values = window_df[feature_cols].values
    tensor = torch.tensor(feature_values, dtype=torch.float32)
    
    return tensor


def get_universe_at_date(df: pd.DataFrame, date: pd.Timestamp) -> list:
    """Get list of tickers available at a specific date.
    
    Args:
        df: DataFrame with 'date' and 'ticker' columns
        date: Date to query
        
    Returns:
        List of tickers
    """
    tickers = df[df['date'] == date]['ticker'].unique().tolist()
    return tickers


def compute_returns_std_from_train(
    prices_df: pd.DataFrame,
    train_end: str = '2018-12-31'
) -> float:
    """Compute returns standard deviation from training period.
    
    Args:
        prices_df: Long-format DataFrame with columns [date, ticker, close]
        train_end: End date for training period
        
    Returns:
        Standard deviation of returns in training period
    """
    if 'date' not in prices_df.columns:
        raise ValueError("prices_df must have a 'date' column")
    
    # Filter to training period BEFORE computing returns
    train_end_dt = pd.to_datetime(train_end)
    train_prices = prices_df[prices_df['date'] <= train_end_dt].copy()
    
    if len(train_prices) == 0:
        raise ValueError(f"No data found before {train_end}. Check date range.")
    
    # Compute returns on training data (long format)
    returns_df = compute_returns(train_prices, log_returns=True)
    
    # Compute std across all valid returns
    all_returns = returns_df['return'].dropna().values
    all_returns = all_returns[np.isfinite(all_returns)]
    
    if len(all_returns) == 0:
        raise ValueError("No valid returns found after computation. Check data quality.")
    
    std_value = float(np.std(all_returns))
    
    if np.isnan(std_value) or std_value == 0:
        raise ValueError(f"Invalid returns std computed: {std_value}. Check data quality.")
    
    return std_value


__all__ = [
    "discover_feature_dims",
    "load_normalization_stats",
    "load_parquets",
    "split_by_date",
    "compute_returns",
    "normalize_returns",
    "build_lookback_tensor",
    "get_universe_at_date",
    "compute_returns_std_from_train",
]
