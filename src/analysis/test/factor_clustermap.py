"""Factor exposure clustermap — cross-sectional snapshot on the test set.

Bi-clustered heatmap of β[N, F] from a single representative trading day.
Rows (stocks) and columns (factors) are both Ward-clustered.
A sector colour strip on the left uses data/parquets/sectors.parquet.
Colorscale is robust (clipped at 2nd/98th pct).

CLI usage:
    python src/analysis/test/factor_clustermap.py \\
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
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import torch
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.models.lightning_module import NeuralFactorsLightning
from src.utils.dataset import NeuralFactorsDataset, collate_fn
from src.utils.data_utils import compute_returns_std_from_train
from torch.utils.data import DataLoader


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _load_model_and_test_dataset(checkpoint_path, data_dir):
    """Load model + test dataset. Reads split dates from config.json if present."""
    import json
    ckpt_parent = Path(checkpoint_path).parent
    config_json = ckpt_parent / "config.json"
    if not config_json.exists():
        config_json = ckpt_parent.parent / "config.json"

    train_end_date = "2018-12-31"
    val_end_date   = "2022-12-31"
    lookback       = 256
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


def get_snapshot_beta(model, dataset, snapshot_date):
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
        alpha, B, sigma, nu, mu_q, L_q = model.model.encode(S, S_static, r, mask)

    beta       = B.squeeze(0).cpu().numpy()
    valid_mask = mask.squeeze(0).cpu().numpy().astype(bool)
    beta_valid    = beta[valid_mask]
    tickers_valid = [t for t, v in zip(tickers, valid_mask) if v]
    return beta_valid, tickers_valid, target_date


def load_sector_mapping(data_dir):
    """Return (ticker_to_sector_id, sector_id_to_name) from sectors.parquet."""
    path = Path(data_dir) / "parquets" / "sectors.parquet"
    if path.exists():
        df = pd.read_parquet(path, engine='pyarrow')
        return dict(zip(df['ticker'], df['sector_id'])), dict(zip(df['sector_id'], df['setor_economico']))
    print(f"Warning: sectors.parquet not found at {path}. Sector colours disabled.")
    return {}, {}


# ── Main function ─────────────────────────────────────────────────────────────

def plot_factor_clustermap(
    model, dataset, output_dir, snapshot_date=None, data_dir="data",
    top_stocks=50, top_factors=20,
):
    """Bi-clustered heatmap of factor exposures from a single cross-sectional snapshot.

    Rows are filtered to the ``top_stocks`` most active names ranked by the L2 norm
    of their β vector (highest total factor-loading magnitude).  Columns are filtered
    to the ``top_factors`` factors with the highest cross-sectional loading variance.
    Both filters remove noise from near-zero rows / dead factors before clustering.
    """
    print("Computing factor exposure clustermap (single-day snapshot)...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    beta_valid, tickers_valid, target_date = get_snapshot_beta(model, dataset, snapshot_date)
    N_valid = len(tickers_valid)
    print(f"Snapshot date: {str(target_date)[:10]}  |  Valid stocks: {N_valid}")

    if N_valid < 5:
        print("Not enough valid stocks for clustermap. Skipping.")
        return

    # ── Column filter: top factors by cross-sectional loading variance ──────
    col_var = beta_valid.var(axis=0)                    # [F]
    top_factor_idx = np.argsort(col_var)[::-1][:top_factors]
    top_factor_idx = np.sort(top_factor_idx)            # keep original order for labelling
    beta_filtered = beta_valid[:, top_factor_idx]       # [N, top_factors]
    print(f"Keeping top {top_factors} factors by cross-sectional variance "
          f"(min kept var={col_var[top_factor_idx].min():.4f})")

    # ── Row filter: top stocks by L2 norm of β (factor-loading magnitude) ───
    l2_norms = np.linalg.norm(beta_filtered, axis=1)    # [N]
    n_keep = min(top_stocks, N_valid)
    top_stock_idx = np.argsort(l2_norms)[::-1][:n_keep]
    top_stock_idx_sorted = top_stock_idx[np.argsort(top_stock_idx)]  # stable order for sectors
    beta_filtered = beta_filtered[top_stock_idx_sorted]
    tickers_filtered = [tickers_valid[i] for i in top_stock_idx_sorted]
    print(f"Keeping top {n_keep} stocks by β L2 norm "
          f"(min kept norm={l2_norms[top_stock_idx_sorted].min():.4f})")

    ticker_to_sector_id, sector_id_to_name = load_sector_mapping(data_dir)

    sector_indices = np.array([
        ticker_to_sector_id.get(t.split('.')[0], -1) for t in tickers_filtered
    ])
    unique_sectors = sorted(set(sector_indices))
    cmap_tab = plt.get_cmap('tab10')
    palette = {
        s_id: (0.65, 0.65, 0.65) if s_id == -1 else cmap_tab(i % 10)
        for i, s_id in enumerate(unique_sectors)
    }

    beta_df = pd.DataFrame(
        beta_filtered,
        index=[t.split('.')[0] for t in tickers_filtered],
        columns=[str(f) for f in top_factor_idx],
    )
    row_colors = pd.Series(
        [palette[s] for s in sector_indices],
        index=beta_df.index,
        name='Setor',
    )

    # Fixed height based on filtered row count; 0.28 in/row is enough to read tickers
    height = max(10, n_keep * 0.28 + 3)
    g = sns.clustermap(
        beta_df,
        method='ward',
        metric='euclidean',
        cmap='RdBu_r',
        center=0,
        robust=True,
        row_colors=row_colors,
        col_cluster=True,
        row_cluster=True,
        xticklabels=True,
        yticklabels=True,
        figsize=(14, height),
        dendrogram_ratio=(0.15, 0.10),
        colors_ratio=0.03,
        cbar_pos=(0.02, 0.85, 0.03, 0.12),
    )
    g.ax_heatmap.set_xlabel('Fatores (top por variância cross-sectional)', fontsize=11)
    g.ax_heatmap.set_ylabel('')
    g.ax_heatmap.tick_params(axis='y', labelsize=8)
    g.ax_heatmap.tick_params(axis='x', labelsize=7, rotation=90)
    g.fig.suptitle(
        f'Factor Exposures — Top {n_keep} Stocks × Top {top_factors} Factors\n{str(target_date)[:10]}',
        fontsize=13, fontweight='bold', y=1.01,
    )

    legend_handles = [
        mpatches.Patch(
            facecolor=palette[s_id],
            label='Sem setor' if s_id == -1 else sector_id_to_name.get(s_id, f'Setor {s_id}'),
        )
        for s_id in unique_sectors
    ]
    g.ax_col_dendrogram.legend(
        handles=legend_handles,
        title='Setor Econômico',
        loc='center',
        ncol=max(1, len(unique_sectors) // 3),
        fontsize=8.5,
        title_fontsize=9,
        framealpha=0.85,
    )

    out_path = output_dir / 'factor_exposures_heatmap.png'
    g.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close('all')
    print(f"Clustermap saved to: {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Factor exposure clustermap")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_dir", default="data")
    p.add_argument("--output_dir", default="results/evaluation/neuralfactors")
    p.add_argument("--snapshot_date", default=None, help="YYYY-MM-DD (default: test split midpoint)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model, dataset = _load_model_and_test_dataset(args.checkpoint, args.data_dir)
    plot_factor_clustermap(model, dataset, args.output_dir, args.snapshot_date, args.data_dir)
