"""Model and data loading utilities for NeuralFactors evaluation."""

import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.models.lightning_module import NeuralFactorsLightning
from src.utils.dataset import NeuralFactorsDataset, collate_fn
from src.utils.data_utils import compute_returns_std_from_train


def load_model_and_data(checkpoint_path, data_dir, split='test'):
    """Load trained model and dataset.

    Args:
        checkpoint_path: Path to model checkpoint
        data_dir: Data directory containing parquets and cleaned data
        split: Dataset split ('train', 'val', or 'test')

    Returns:
        tuple: (model, dataloader, dataset, returns_std, device)
    """
    print("=" * 80)
    print("LOADING MODEL AND DATA")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print(f"Loading model from {checkpoint_path}...")
    model = NeuralFactorsLightning.load_from_checkpoint(checkpoint_path, strict=False)
    model.eval()
    model = model.to(device)
    print("✓ Model loaded successfully")

    # Load returns_std, lookback and split dates from saved config or recompute
    checkpoint_dir = Path(checkpoint_path).parent
    config_path = checkpoint_dir / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        returns_std  = config['training']['returns_std']
        lookback     = config['model']['lookback']
        train_end    = config['args'].get('train_end', '2018-12-31')
        val_end      = config['args'].get('val_end',   '2022-12-31')
        print(f"✓ Returns std from config: {returns_std:.6f}")
        print(f"✓ Lookback from config: {lookback}")
        print(f"✓ Split dates from config: train_end={train_end}, val_end={val_end}")
    else:
        print("Config not found, computing returns_std from training data...")
        prices_file = Path(data_dir) / "cleaned" / "fechamentos_ibx.csv"
        df_prices = pd.read_csv(prices_file, sep=';', decimal=',',
                                parse_dates=['DATES'], dayfirst=True)
        df_prices.rename(columns={'DATES': 'date'}, inplace=True)
        returns_std = compute_returns_std_from_train(df_prices, train_end="2018-12-31")
        lookback    = model.model_config.lookback
        train_end   = "2018-12-31"
        val_end     = "2022-12-31"
        print(f"✓ Computed returns_std: {returns_std:.6f}")
        print(f"✓ Lookback from model: {lookback}")

    print(f"Loading {split} dataset...")
    x_ts_file = Path(data_dir) / "parquets" / "x_ts.parquet"
    x_static_file = Path(data_dir) / "parquets" / "x_static.parquet"
    prices_file = Path(data_dir) / "cleaned" / "fechamentos_ibx.csv"

    dataset = NeuralFactorsDataset(
        x_ts_path=str(x_ts_file),
        x_static_path=str(x_static_file),
        prices_path=str(prices_file),
        split=split,
        lookback=lookback,
        returns_std=returns_std,
        train_end=train_end,
        val_end=val_end,
    )
    print(f"✓ Dataset loaded: {len(dataset)} dates")

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    return model, dataloader, dataset, returns_std, device


def setup_output_dirs(output_dir, experiment_name):
    """Create evaluation output directory structure.

    Args:
        output_dir: Base output directory
        experiment_name: Experiment name for subdirectory

    Returns:
        Path: Full output directory path
    """
    output_path = Path(output_dir) / experiment_name
    (output_path / "metrics").mkdir(parents=True, exist_ok=True)
    (output_path / "timeseries").mkdir(parents=True, exist_ok=True)
    (output_path / "plots").mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directories created at: {output_path}")
    return output_path
