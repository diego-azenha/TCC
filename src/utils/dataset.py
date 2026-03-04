"""PyTorch Dataset for NeuralFactors model.

Dataset loads parquet files, builds lookback windows, and returns
(S, S_static, r, mask) tuples where each sample is all stocks from one trading day.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple, List
from pathlib import Path

from .data_utils import (
    load_parquets,
    split_by_date,
    compute_returns,
    normalize_returns,
    get_universe_at_date,
    compute_returns_std_from_train,
)


class NeuralFactorsDataset(Dataset):
    """Dataset for NeuralFactors model.
    
    Each sample contains all stocks from a single trading day with lookback windows.
    Returns: (S, S_static, r, mask) where:
    - S: [N, L, d_ts] time-series features with lookback
    - S_static: [N, d_static] static features
    - r: [N] next-day returns
    - mask: [N] boolean mask for valid stocks
    
    Paper Section 5: Train 1996-2013, Val 2014-2018, Test 2019-2023
    """
    
    def __init__(
        self,
        x_ts_path: str,
        x_static_path: str,
        prices_path: str,
        split: str = 'train',
        lookback: int = 256,
        normalize: bool = True,
        returns_std: Optional[float] = None,
        train_end: str = '2018-12-31',
        val_end: str = '2022-12-31',
    ):
        """
        Args:
            x_ts_path: Path to time-series features parquet
            x_static_path: Path to static features parquet
            prices_path: Path to prices CSV
            split: One of 'train', 'val', 'test'
            lookback: Lookback window size (paper: 256)
            normalize: If True, normalize returns by std
            returns_std: Standard deviation for normalization (computed if None)
            train_end: End date for training split
            val_end: End date for validation split
        """
        super().__init__()
        
        self.lookback = lookback
        self.split = split
        self.normalize = normalize
        
        # Load data
        print(f"Loading data for {split} split...")
        df_ts, df_static, df_prices = load_parquets(x_ts_path, x_static_path, prices_path)
        
        # Compute returns
        print("Computing returns...")
        returns_df = compute_returns(df_prices, log_returns=True)
        
        # Compute or use provided returns std
        if normalize:
            if returns_std is None and split == 'train':
                print("Computing returns std from training data...")
                returns_std = compute_returns_std_from_train(df_prices, train_end)
                print(f"Training returns std: {returns_std:.6f}")
            elif returns_std is None:
                raise ValueError("returns_std must be provided for val/test splits")
            
            print(f"Normalizing returns by std={returns_std:.6f}")
            returns_df, self.returns_std = normalize_returns(returns_df, std_value=returns_std)
        else:
            self.returns_std = 1.0
        
        # Convert returns from wide to long format (date, ticker, value)
        from src.utils.data_utils import melt_to_long_format
        returns_df = melt_to_long_format(returns_df)
        returns_df.rename(columns={'value': 'return'}, inplace=True)
        
        # Split data by date
        print(f"Splitting data: train_end={train_end}, val_end={val_end}")
        df_ts_train, df_ts_val, df_ts_test = split_by_date(df_ts, train_end, val_end)
        df_static_train, df_static_val, df_static_test = split_by_date(df_static, train_end, val_end)
        returns_train, returns_val, returns_test = split_by_date(returns_df, train_end, val_end)
        
        # Select appropriate split
        split_map = {
            'train': (df_ts_train, df_static_train, returns_train),
            'val': (df_ts_val, df_static_val, returns_val),
            'test': (df_ts_test, df_static_test, returns_test),
        }
        
        if split not in split_map:
            raise ValueError(f"split must be one of {list(split_map.keys())}, got {split}")
        
        self.df_ts, self.df_static, self.returns_df = split_map[split]
        
        # For lookback, we need data BEFORE the split start date
        # Concatenate with previous split's data
        if split == 'val':
            self.df_ts_full = pd.concat([df_ts_train, self.df_ts], axis=0)
            self.df_static_full = pd.concat([df_static_train, self.df_static], axis=0)
        elif split == 'test':
            self.df_ts_full = pd.concat([df_ts_train, df_ts_val, self.df_ts], axis=0)
            self.df_static_full = pd.concat([df_static_train, df_static_val, self.df_static], axis=0)
        else:
            self.df_ts_full = self.df_ts.copy()
            self.df_static_full = self.df_static.copy()
        
        # Get feature column names (exclude 'date', 'ticker')
        self.ts_feature_cols = [col for col in self.df_ts.columns if col not in ['date', 'ticker']]
        self.static_feature_cols = [col for col in self.df_static.columns if col not in ['date', 'ticker']]
        
        self.d_ts = len(self.ts_feature_cols)
        self.d_static = len(self.static_feature_cols)
        
        # Get unique dates in this split (these are the samples)
        self.dates = sorted(self.returns_df['date'].unique())

        # ── PRE-INDEX: evita filtragem linear no __getitem__ ─────────────────

        # ts: agrupa por ticker, ordena por date, extrai arrays numpy
        # _ts_grouped é temporário — liberado após extrair datas e valores
        _ts_grouped = {
            ticker: grp.sort_values('date').reset_index(drop=True)
            for ticker, grp in self.df_ts_full.groupby('ticker')
        }
        # ticker -> array de datas (para searchsorted O(log N))
        self._ts_dates_cache: dict = {
            ticker: df['date'].values
            for ticker, df in _ts_grouped.items()
        }
        # ticker -> array float32 de features (sem cópia extra no __getitem__)
        self._ts_values_cache: dict = {
            ticker: df[self.ts_feature_cols].values.astype(np.float32)
            for ticker, df in _ts_grouped.items()
        }
        del _ts_grouped  # libera memória intermediária

        # static: set_index vetorizado → zip(keys, rows) sem iterrows()
        _static_indexed = (
            self.df_static_full
            .set_index(['ticker', 'date'])[self.static_feature_cols]
            .astype(np.float32)
        )
        # {(ticker, date): np.array([f1, f2, ...])} — sem loop Python por linha
        self._static_cache: dict = dict(
            zip(list(_static_indexed.index), _static_indexed.values)
        )
        del _static_indexed

        # returns: pivot vetorizado → iterrows() sobre N_datas (~3458) apenas
        _returns_pivot = self.returns_df.pivot(
            index='date', columns='ticker', values='return'
        )
        # {date: {ticker: return_value}} — NaN descartados por dropna()
        self._returns_cache: dict = {
            date: row.dropna().to_dict()
            for date, row in _returns_pivot.iterrows()
        }
        del _returns_pivot
        # ─────────────────────────────────────────────────────────────────────

        print(f"Dataset {split}: {len(self.dates)} trading days, d_ts={self.d_ts}, d_static={self.d_static}")

    def __len__(self) -> int:
        """Number of trading days in split."""
        return len(self.dates)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get all stocks from one trading day.
        
        Args:
            idx: Index of trading day
            
        Returns:
            S: [N, L, d_ts] time-series features with lookback
            S_static: [N, d_static] static features
            r: [N] next-day returns
            mask: [N] boolean mask (True = valid stock)
        """
        target_date = self.dates[idx]

        # O(1) dict lookup em vez de filtrar o DataFrame inteiro
        ticker_returns = self._returns_cache.get(target_date, {})
        tickers = list(ticker_returns.keys())
        N = len(tickers)

        # Initialize tensors
        S = torch.zeros(N, self.lookback, self.d_ts, dtype=torch.float32)
        S_static = torch.zeros(N, self.d_static, dtype=torch.float32)
        r = torch.zeros(N, dtype=torch.float32)
        mask = torch.zeros(N, dtype=torch.bool)

        for i, ticker in enumerate(tickers):
            # O(1) return lookup
            r_value = ticker_returns[ticker]
            if np.isnan(r_value):
                continue
            r[i] = r_value

            # O(1) ts lookup + O(log N) searchsorted por data
            dates_arr = self._ts_dates_cache.get(ticker)
            values_arr = self._ts_values_cache.get(ticker)
            if dates_arr is None:
                continue

            end_idx = int(np.searchsorted(dates_arr, target_date, side='right'))
            if end_idx < self.lookback:
                continue

            ts_values = values_arr[end_idx - self.lookback: end_idx]
            S[i] = torch.from_numpy(np.nan_to_num(ts_values))

            # O(1) static lookup
            static_values = self._static_cache.get((ticker, target_date))
            if static_values is not None:
                S_static[i] = torch.from_numpy(np.nan_to_num(static_values))
                mask[i] = True

        return S, S_static, r, mask


def collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate function for DataLoader.
    
    Paper uses batch_size=1 per date, where each batch contains all stocks from a single day.
    We keep batch and stock dimensions separate: [batch, N, L, d_ts]
    
    Args:
        batch: List of (S, S_static, r, mask) tuples
        
    Returns:
        Stacked tensors: S [batch, N, L, d_ts], S_static [batch, N, d_static], 
                        r [batch, N], mask [batch, N]
        where batch is typically 1 (one trading day) and N is number of stocks that day
    """
    S_list, S_static_list, r_list, mask_list = zip(*batch)
    
    # Stack along batch dimension, keeping stock dimension separate
    S = torch.stack(S_list, dim=0)  # [batch, N, L, d_ts]
    S_static = torch.stack(S_static_list, dim=0)  # [batch, N, d_static]
    r = torch.stack(r_list, dim=0)  # [batch, N]
    mask = torch.stack(mask_list, dim=0)  # [batch, N]
    
    return S, S_static, r, mask


__all__ = [
    "NeuralFactorsDataset",
    "collate_fn",
]
