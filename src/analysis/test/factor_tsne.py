"""Factor exposure t-SNE — cross-sectional snapshot on the test set.

One point per ticker on a single representative trading day.
Points are coloured by economic sector (data/parquets/sectors.parquet).
Well-known index constituents are annotated directly on the plot.

CLI usage:
    python src/analysis/test/factor_tsne.py \\
        --checkpoint checkpoints/new_dataset/last.ckpt \\
        --data_dir data \\
        --output_dir results/evaluation/new_dataset \\
        --snapshot_date 2024-06-15
"""

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*torch.load.*')
warnings.filterwarnings('ignore', message='.*Found keys that are not in the model state dict.*')

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.models.lightning_module import NeuralFactorsLightning
from src.utils.dataset import NeuralFactorsDataset
from src.utils.data_utils import compute_returns_std_from_train


LABEL_TICKERS = {
    'PETR4', 'VALE3', 'ITUB4', 'BBDC4', 'ABEV3',
    'WEGE3', 'RENT3', 'RADL3', 'BBAS3', 'BPAC11',
    'MGLU3', 'LREN3', 'JBSS3', 'GGBR4', 'CSNA3',
    'SUZB3', 'EMBR3', 'AZUL4', 'GOLL4', 'B3SA3',
    'HAPV3', 'PRIO3', 'CSAN3', 'VBBR3', 'ELET3',
}


# ── Shared helpers (inlined so this file is self-contained) ───────────────────

def _load_model_and_test_dataset(checkpoint_path, data_dir):
    """Load model + test dataset. Reads split dates from config.json if present."""
    ckpt_parent = Path(checkpoint_path).parent
    config_json = ckpt_parent / "config.json"
    if not config_json.exists():
        config_json = ckpt_parent.parent / "config.json"

    train_end_date = "2018-12-31"
    val_end_date   = "2022-12-31"
    if config_json.exists():
        with open(config_json) as f:
            cfg = json.load(f)
        train_end_date = cfg.get("args", {}).get("train_end", train_end_date)
        val_end_date   = cfg.get("args", {}).get("val_end",   val_end_date)

    model = NeuralFactorsLightning.load_from_checkpoint(checkpoint_path, strict=False)
    model.eval()
    model = model.cuda() if torch.cuda.is_available() else model
    lookback = model.model_config.lookback

    data_dir = Path(data_dir)
    prices_path = data_dir / "parquets" / "prices.parquet"
    df_prices = pd.read_parquet(prices_path, engine='pyarrow')
    df_prices['date'] = pd.to_datetime(df_prices['date'])
    returns_std = compute_returns_std_from_train(df_prices, train_end=train_end_date)

    dataset = NeuralFactorsDataset(
        x_ts_path=str(data_dir / "parquets" / "x_ts.parquet"),
        x_static_path=str(data_dir / "parquets" / "x_static.parquet"),
        prices_path=str(prices_path),
        split='test',
        lookback=lookback,
        returns_std=returns_std,
        train_end=train_end_date,
        val_end=val_end_date,
    )
    return model, dataset


def _get_snapshot_beta(model, dataset, snapshot_date):
    """Return (beta_valid[N,F], tickers_valid, target_date) for one snapshot day."""
    dates = dataset.dates
    if snapshot_date is not None:
        snap_ts = pd.Timestamp(snapshot_date)
        idx = int(np.argmin([abs((pd.Timestamp(d) - snap_ts).days) for d in dates]))
    else:
        idx = len(dates) // 2
    target_date = dates[idx]

    ticker_returns = dataset._returns_cache.get(target_date, {})
    tickers = list(ticker_returns.keys())

    S, S_static, r, mask = dataset[idx]
    S        = S.unsqueeze(0)
    S_static = S_static.unsqueeze(0)
    r        = r.unsqueeze(0)
    mask     = mask.unsqueeze(0)

    device = next(model.parameters()).device
    S, S_static, r, mask = S.to(device), S_static.to(device), r.to(device), mask.to(device)

    with torch.no_grad():
        alpha, B, sigma, mu_q, log_sigma_q = model.model.encode(S, S_static, r, mask)

    beta       = B.squeeze(0).cpu().numpy()
    valid_mask = mask.squeeze(0).cpu().numpy().astype(bool)
    beta_valid    = beta[valid_mask]
    tickers_valid = [t for t, v in zip(tickers, valid_mask) if v]
    return beta_valid, tickers_valid, target_date


def _load_sector_mapping(data_dir):
    """Return (ticker_to_sector_id, sector_id_to_name) from sectors.parquet."""
    path = Path(data_dir) / "parquets" / "sectors.parquet"
    if path.exists():
        df = pd.read_parquet(path, engine='pyarrow')
        return dict(zip(df['ticker'], df['sector_id'])), dict(zip(df['sector_id'], df['setor_economico']))
    print(f"Warning: sectors.parquet not found at {path}. Sector colours disabled.")
    return {}, {}


# ── Main function ─────────────────────────────────────────────────────────────

def plot_factor_tsne(model, dataset, output_dir, snapshot_date=None, data_dir="data"):
    """t-SNE of factor exposures from a single cross-sectional snapshot."""
    print("Computing factor exposure t-SNE (single-day snapshot)...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    beta_valid, tickers_valid, target_date = _get_snapshot_beta(model, dataset, snapshot_date)
    N_valid = len(tickers_valid)
    print(f"Snapshot date: {str(target_date)[:10]}  |  Valid stocks: {N_valid}")

    if N_valid < 5:
        print("Not enough valid stocks for t-SNE. Skipping.")
        return

    ticker_to_sector_id, sector_id_to_name = _load_sector_mapping(data_dir)
    if ticker_to_sector_id:
        print(f"Loaded sector mapping: {len(ticker_to_sector_id)} tickers, {len(sector_id_to_name)} sectors")

    sector_indices = np.array([
        ticker_to_sector_id.get(t.split('.')[0], -1) for t in tickers_valid
    ])
    unique_sectors = sorted(set(sector_indices))
    cmap = plt.get_cmap('tab10')

    perplexity = min(30, N_valid - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=1000)
    beta_2d = tsne.fit_transform(beta_valid)

    fig, ax = plt.subplots(figsize=(12, 9))
    fig.suptitle(
        f'Factor Exposures — t-SNE Cross-Sectional Snapshot\n{str(target_date)[:10]}',
        fontsize=14, fontweight='bold',
    )

    for s_pos, s_idx in enumerate(unique_sectors):
        mask_s = sector_indices == s_idx
        if not mask_s.any():
            continue
        label = 'Sem setor' if s_idx == -1 else sector_id_to_name.get(s_idx, f'Setor {s_idx}')
        color = (0.6, 0.6, 0.6, 1.0) if s_idx == -1 else cmap(s_pos % 10)
        ax.scatter(
            beta_2d[mask_s, 0], beta_2d[mask_s, 1],
            c=[color], alpha=0.80, s=65,
            label=label, edgecolors='white', linewidths=0.5,
        )

    for i, ticker in enumerate(tickers_valid):
        base = ticker.split('.')[0]
        if base in LABEL_TICKERS or ticker in LABEL_TICKERS:
            ax.annotate(
                base,
                xy=(beta_2d[i, 0], beta_2d[i, 1]),
                xytext=(4, 4), textcoords='offset points',
                fontsize=7.5, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.65, edgecolor='none'),
            )

    ax.set_xlabel('t-SNE Dimensão 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimensão 2', fontsize=12)
    ax.legend(
        title='Setor Econômico',
        bbox_to_anchor=(1.01, 1), loc='upper left',
        fontsize=9, title_fontsize=10, framealpha=0.9,
    )
    ax.grid(True, alpha=0.2, linestyle='--')
    plt.tight_layout()
    out_path = output_dir / 'factor_exposures_tsne.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"t-SNE snapshot saved to: {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Factor exposure t-SNE")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_dir", default="data")
    p.add_argument("--output_dir", default="results/evaluation/neuralfactors")
    p.add_argument("--snapshot_date", default=None, help="YYYY-MM-DD (default: test split midpoint)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model, dataset = _load_model_and_test_dataset(args.checkpoint, args.data_dir)
    plot_factor_tsne(model, dataset, args.output_dir, args.snapshot_date, args.data_dir)

