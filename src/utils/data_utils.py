"""Data preprocessing utilities for NeuralFactors training.

Functions for loading parquets, computing returns, building lookback windows,
and splitting data according to paper specifications (Section 5).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
import torch


def discover_feature_dims(x_ts_path: str, x_static_path: str) -> Tuple[int, int]:
    """Discover feature dimensions from parquet files.
    
    Args:
        x_ts_path: Path to time-series features parquet
        x_static_path: Path to static features parquet
        
    Returns:
        Tuple of (d_ts, d_static) - feature dimensions
    """
    df_ts = pd.read_parquet(x_ts_path, engine='pyarrow')
    df_static = pd.read_parquet(x_static_path, engine='pyarrow')
    
    # Exclude 'date' and 'ticker' columns
    d_ts = len([col for col in df_ts.columns if col not in ['date', 'ticker']])
    d_static = len([col for col in df_static.columns if col not in ['date', 'ticker']])
    
    return d_ts, d_static


def load_parquets(
    x_ts_path: str,
    x_static_path: str,
    prices_path: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """Load parquet files and ensure proper datetime indexing.
    
    Args:
        x_ts_path: Path to time-series features parquet
        x_static_path: Path to static features parquet
        prices_path: Optional path to prices CSV for return calculation
        
    Returns:
        Tuple of (df_ts, df_static, df_prices)
    """
    df_ts = pd.read_parquet(x_ts_path, engine='pyarrow')
    df_static = pd.read_parquet(x_static_path, engine='pyarrow')
    
    # Ensure date columns are datetime
    df_ts['date'] = pd.to_datetime(df_ts['date'])
    df_static['date'] = pd.to_datetime(df_static['date'])
    
    df_prices = None
    if prices_path:
        df_prices = pd.read_csv(
            prices_path, 
            sep=';', 
            decimal=',',  # Handle European decimal format
            parse_dates=['DATES'], 
            dayfirst=True
        )
        df_prices.rename(columns={'DATES': 'date'}, inplace=True)
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
    """Compute returns from prices.
    
    Args:
        prices_df: DataFrame with columns ['date', 'ticker_1', 'ticker_2', ...]
        log_returns: If True, compute log returns; else simple returns
        
    Returns:
        DataFrame with same structure but returns instead of prices
    """
    returns_df = prices_df.copy()
    
    # Get all ticker columns (exclude 'date')
    ticker_cols = [col for col in prices_df.columns if col != 'date']
    
    # Convert string columns to numeric (handles CSV with string values and comma decimals)
    for ticker in ticker_cols:
        # Replace comma with dot for European decimal format
        if returns_df[ticker].dtype == 'object':
            returns_df[ticker] = returns_df[ticker].str.replace(',', '.')
        returns_df[ticker] = pd.to_numeric(returns_df[ticker], errors='coerce')
    
    # Compute returns per ticker
    for ticker in ticker_cols:
        if log_returns:
            # Suppress divide by zero warnings (produces -inf which we filter later)
            with np.errstate(divide='ignore', invalid='ignore'):
                returns_df[ticker] = np.log(returns_df[ticker] / returns_df[ticker].shift(1))
        else:
            returns_df[ticker] = (returns_df[ticker] / returns_df[ticker].shift(1)) - 1.0
    
    # Drop first row (NaN from shift)
    returns_df = returns_df.iloc[1:].reset_index(drop=True)
    
    # [FIX] Replace -Inf/Inf with NaN (happens when price is 0 or negative)
    # These will be handled by masking in the dataset
    returns_df = returns_df.replace([np.inf, -np.inf], np.nan)
    
    return returns_df


def normalize_returns(
    returns_df: pd.DataFrame,
    std_value: Optional[float] = None,
    compute_std_from_data: bool = False
) -> Tuple[pd.DataFrame, float]:
    """Normalize returns by dividing by standard deviation.
    
    Paper Section 5: Returns normalized by dividing by 0.02672357
    (approximately the standard deviation across training period).
    
    Args:
        returns_df: DataFrame with returns
        std_value: Standard deviation to divide by (if None, computed from data)
        compute_std_from_data: If True, compute std from data ignoring std_value
        
    Returns:
        Tuple of (normalized_returns_df, std_used)
    """
    ticker_cols = [col for col in returns_df.columns if col != 'date']
    
    if compute_std_from_data or std_value is None:
        # Compute std across all returns in dataframe
        all_returns = returns_df[ticker_cols].values.flatten()
        all_returns = all_returns[~np.isnan(all_returns)]
        std_used = np.std(all_returns)
    else:
        std_used = std_value
    
    normalized_df = returns_df.copy()
    normalized_df[ticker_cols] = returns_df[ticker_cols] / std_used
    
    return normalized_df, std_used


def melt_to_long_format(df: pd.DataFrame) -> pd.DataFrame:
    """Convert wide format to long format (date, ticker, value).
    
    Args:
        df: DataFrame with columns ['date', 'ticker_1', 'ticker_2', ...]
        
    Returns:
        DataFrame with columns ['date', 'ticker', 'value']
    """
    ticker_cols = [col for col in df.columns if col != 'date']
    
    melted = df.melt(
        id_vars=['date'],
        value_vars=ticker_cols,
        var_name='ticker',
        value_name='value'
    )
    
    return melted


def merge_features_and_returns(
    df_ts: pd.DataFrame,
    df_static: pd.DataFrame,
    returns_df: pd.DataFrame
) -> pd.DataFrame:
    """Merge time-series features, static features, and returns.
    
    Args:
        df_ts: Long format [date, ticker, feature_1, feature_2, ...]
        df_static: Long format [date, ticker, feature_1, ...]
        returns_df: Long format [date, ticker, return]
        
    Returns:
        Merged DataFrame [date, ticker, return, ts_features..., static_features...]
    """
    # Merge time-series features with returns
    merged = pd.merge(
        returns_df,
        df_ts,
        on=['date', 'ticker'],
        how='inner'
    )
    
    # Merge with static features
    merged = pd.merge(
        merged,
        df_static,
        on=['date', 'ticker'],
        how='inner'
    )
    
    return merged


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
        prices_df: DataFrame with prices (must have 'date' column)
        train_end: End date for training period
        
    Returns:
        Standard deviation of returns in training period
    """
    # Ensure date column exists
    if 'date' not in prices_df.columns:
        raise ValueError("prices_df must have a 'date' column")
    
    # Filter to training period BEFORE computing returns
    train_end_dt = pd.to_datetime(train_end)
    train_prices = prices_df[prices_df['date'] <= train_end_dt].copy()
    
    if len(train_prices) == 0:
        raise ValueError(f"No data found before {train_end}. Check date range.")
    
    # Compute returns on training data
    returns_df = compute_returns(train_prices, log_returns=True)
    
    # Compute std across all returns
    ticker_cols = [col for col in returns_df.columns if col != 'date']
    
    all_returns = returns_df[ticker_cols].values.flatten()
    
    # Filter out NaN and Inf values
    all_returns = all_returns[np.isfinite(all_returns)]
    
    if len(all_returns) == 0:
        raise ValueError("No valid returns found after computation. Check data quality.")
    
    std_value = np.std(all_returns)
    
    if np.isnan(std_value) or std_value == 0:
        raise ValueError(f"Invalid returns std computed: {std_value}. Check data quality.")
    
    return std_value


__all__ = [
    "discover_feature_dims",
    "load_parquets",
    "split_by_date",
    "compute_returns",
    "normalize_returns",
    "melt_to_long_format",
    "merge_features_and_returns",
    "build_lookback_tensor",
    "get_universe_at_date",
    "compute_returns_std_from_train",
]
