"""Factor Structure — t-SNE + predicted vs empirical correlation matrices.

Left panel:  t-SNE of factor exposures β[N,F] coloured by sector.
             PCA explained-variance ratio of the 2–PC embedding is annotated.

Right panel: Predicted correlation (from B B' + diag(σ²)) vs empirical
             correlation (20-day rolling window) shown as two side-by-side
             (top×40, 40×40) heatmaps (or a difference heatmap).

CLI usage:
    python src/analysis/test/factor_structure.py \\
        --checkpoint checkpoints/simplified_v1/last.ckpt \\
        --data_dir data \\
        --output_dir results/evaluation/simplified_v1/plots \\
        --snapshot_date 2024-06-15
"""

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*torch.load.*')

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.analysis.test.factor_clustermap import (
    _load_model_and_test_dataset,
    get_snapshot_beta,
    load_sector_mapping,
)

# Tickers to label on t-SNE (same set as factor_tsne.py)
LABEL_TICKERS = {
    'PETR4', 'VALE3', 'ITUB4', 'BBDC4', 'ABEV3',
    'WEGE3', 'RENT3', 'RADL3', 'BBAS3', 'BPAC11',
    'MGLU3', 'LREN3', 'JBSS3', 'GGBR4', 'CSNA3',
    'SUZB3', 'EMBR3', 'AZUL4', 'GOLL4', 'B3SA3',
    'HAPV3', 'PRIO3', 'CSAN3', 'VBBR3', 'ELET3',
}


def _build_return_matrix(dataset, snap_idx, tickers_valid, window=20):
    """Return (returns_matrix[window, N_valid], valid_columns) in normalised space."""
    dates = dataset.dates
    start_idx = max(0, snap_idx - window)
    window_dates = dates[start_idx:snap_idx]

    ticker_to_col = {t: i for i, t in enumerate(tickers_valid)}
    ret_matrix = np.full((len(window_dates), len(tickers_valid)), np.nan)
    for d_i, d in enumerate(window_dates):
        day_rets = dataset._returns_cache.get(d, {})
        for ticker, ret_val in day_rets.items():
            if ticker in ticker_to_col:
                ret_matrix[d_i, ticker_to_col[ticker]] = ret_val
    return ret_matrix


def plot_factor_structure(
    model,
    dataset,
    output_dir,
    snapshot_date=None,
    data_dir="data",
    top_stocks=40,
):
    """Two-panel factor structure figure.

    Left:  t-SNE of β coloured by sector + PCA explained-variance annotation.
    Right: Predicted vs empirical (40×40) correlation heatmaps.

    Args:
        model: Trained NeuralFactors (Lightning) model.
        dataset: NeuralFactorsDataset (test split).
        output_dir: Path for saving the PNG.
        snapshot_date: Optional date string.
        data_dir: Root data directory.
        top_stocks: Number of stocks for the correlation panel.
    """
    print("Computing factor structure (t-SNE + correlation)...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Snapshot ──────────────────────────────────────────────────────────────
    dates = dataset.dates
    if snapshot_date is not None:
        snap_ts = pd.Timestamp(snapshot_date)
        snap_idx = int(np.argmin([abs((pd.Timestamp(d) - snap_ts).days) for d in dates]))
    else:
        snap_idx = len(dates) // 2
    target_date = dates[snap_idx]

    beta_valid, tickers_valid, _ = get_snapshot_beta(model, dataset, snapshot_date)
    N_valid = len(tickers_valid)
    print(f"Snapshot date: {str(target_date)[:10]}  |  Valid stocks: {N_valid}")
    if N_valid < 10:
        print("Not enough valid stocks. Skipping factor structure.")
        return

    # Also get alpha/B/sigma for the correlation panel
    device = next(model.parameters()).device
    S, S_static, r_t, mask_t = dataset[snap_idx]
    S = S.unsqueeze(0).to(device)
    S_static = S_static.unsqueeze(0).to(device)
    r_t = r_t.unsqueeze(0).to(device)
    mask_t = mask_t.unsqueeze(0).to(device)

    with torch.no_grad():
        alpha, B, sigma, mu_q, log_sigma_q = model.model.encode(S, S_static, r_t, mask_t)

    sigma_np = sigma.squeeze(0).cpu().numpy()
    mask_np  = mask_t.squeeze(0).cpu().numpy().astype(bool)
    sigma_valid = sigma_np[mask_np]   # [N_valid]

    # ── Sector colours ────────────────────────────────────────────────────────
    ticker_to_sector_id, sector_id_to_name = load_sector_mapping(data_dir)
    sector_indices = np.array([
        ticker_to_sector_id.get(t.split('.')[0], -1) for t in tickers_valid
    ])
    unique_sectors = sorted(set(sector_indices))
    cmap_tab = plt.get_cmap('tab10')
    palette = {
        s_id: (0.65, 0.65, 0.65) if s_id == -1 else cmap_tab(i % 10)
        for i, s_id in enumerate(unique_sectors)
    }
    colors = [palette[s] for s in sector_indices]

    # ── PCA 2-component explained variance ───────────────────────────────────
    pca = PCA(n_components=min(2, beta_valid.shape[1]))
    pca.fit(beta_valid)
    pca_var_explained = float(pca.explained_variance_ratio_.sum() * 100)

    # ── t-SNE ─────────────────────────────────────────────────────────────────
    perplexity = min(30, max(5, N_valid - 1))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    beta_2d = tsne.fit_transform(beta_valid)  # [N_valid, 2]

    # ── Correlation panel ─────────────────────────────────────────────────────
    # Select top_stocks by ||beta_i||^2
    n_corr = min(top_stocks, N_valid)
    l2_sq  = np.sum(beta_valid**2, axis=1)
    corr_idx = np.argsort(l2_sq)[::-1][:n_corr]

    beta_sub  = beta_valid[corr_idx]     # [n_corr, F]
    sigma_sub = sigma_valid[corr_idx]    # [n_corr]

    # Predicted covariance (Σ = BB' + diag(σ²))
    cov_pred = beta_sub @ beta_sub.T + np.diag(sigma_sub**2)
    d_pred   = np.sqrt(np.diag(cov_pred)).clip(min=1e-8)
    corr_pred = cov_pred / np.outer(d_pred, d_pred)
    np.fill_diagonal(corr_pred, 1.0)

    # Empirical correlation from last 20 trading days
    ret_matrix = _build_return_matrix(dataset, snap_idx, tickers_valid, window=20)
    ret_sub    = ret_matrix[:, corr_idx]  # [T, n_corr]
    # Only keep columns with enough non-nan observations
    valid_cols = np.where(np.sum(~np.isnan(ret_sub), axis=0) >= 5)[0]
    if len(valid_cols) >= 4:
        ret_sub_clean = ret_sub[:, valid_cols]
        # Fill remaining NaN with column mean
        col_means = np.nanmean(ret_sub_clean, axis=0)
        for j in range(ret_sub_clean.shape[1]):
            nan_rows = np.isnan(ret_sub_clean[:, j])
            ret_sub_clean[nan_rows, j] = col_means[j]
        corr_emp_sub = np.corrcoef(ret_sub_clean.T)   # [len(valid_cols), len(valid_cols)]
        corr_pred_sub = corr_pred[np.ix_(valid_cols, valid_cols)]
    else:
        corr_emp_sub  = corr_pred  # fallback: both show predicted
        corr_pred_sub = corr_pred
        valid_cols    = np.arange(n_corr)

    n_c = len(valid_cols)
    tickers_corr  = [tickers_valid[corr_idx[j]].split('.')[0] for j in valid_cols]

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 8))
    gs  = gridspec.GridSpec(1, 3, width_ratios=[2, 1.7, 1.7], wspace=0.35)

    ax_tsne   = fig.add_subplot(gs[0])
    ax_pred   = fig.add_subplot(gs[1])
    ax_emp    = fig.add_subplot(gs[2])

    # ── Left: t-SNE ───────────────────────────────────────────────────────────
    for s_id in unique_sectors:
        idx_s = sector_indices == s_id
        name  = sector_id_to_name.get(s_id, 'Unknown') if s_id != -1 else 'Unknown'
        ax_tsne.scatter(
            beta_2d[idx_s, 0], beta_2d[idx_s, 1],
            color=palette[s_id], label=name, s=25, alpha=0.75, linewidths=0,
        )

    # Label known tickers
    for i, ticker in enumerate(tickers_valid):
        short = ticker.split('.')[0]
        if short in LABEL_TICKERS:
            ax_tsne.annotate(
                short,
                xy=(beta_2d[i, 0], beta_2d[i, 1]),
                fontsize=6, alpha=0.85,
                xytext=(3, 3), textcoords='offset points',
            )

    ax_tsne.set_title(
        f't-SNE of factor exposures β — {str(target_date)[:10]}\n'
        f'(PCA 2PC explains {pca_var_explained:.1f}% of β variance)',
        fontsize=11, fontweight='bold',
    )
    ax_tsne.set_xlabel('t-SNE 1', fontsize=10)
    ax_tsne.set_ylabel('t-SNE 2', fontsize=10)
    ax_tsne.legend(fontsize=7, markerscale=1.4, loc='best',
                   framealpha=0.7, ncol=1)
    ax_tsne.grid(True, alpha=0.2)

    # ── Middle: predicted correlation ─────────────────────────────────────────
    im_pred = ax_pred.imshow(corr_pred_sub, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax_pred.set_title(f'Predicted correlation\n(B B\' + diag(σ²))', fontsize=11, fontweight='bold')
    ax_pred.set_xticks([])
    ax_pred.set_yticks([])
    plt.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04)

    # ── Right: empirical correlation ──────────────────────────────────────────
    im_emp = ax_emp.imshow(corr_emp_sub, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax_emp.set_title(f'Empirical correlation\n(20-day rolling, top-{n_c} stocks)', fontsize=11, fontweight='bold')
    ax_emp.set_xticks([])
    ax_emp.set_yticks([])
    plt.colorbar(im_emp, ax=ax_emp, fraction=0.046, pad=0.04)

    fig.suptitle('Factor Structure', fontsize=14, fontweight='bold')
    fig.tight_layout()
    out_path = output_dir / 'factor_structure.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Factor structure saved to: {out_path}")
    print(f"  t-SNE PCA 2-PC: {pca_var_explained:.1f}% of β variance")
    print(f"  Correlation panel: {n_c}×{n_c} stocks")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--output_dir', default='results/evaluation/neuralfactors/plots')
    parser.add_argument('--snapshot_date', default=None)
    parser.add_argument('--top_stocks', type=int, default=40)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    from src.analysis.test.loader import load_model_and_data
    model, dataloader, dataset, returns_std, device = load_model_and_data(
        args.checkpoint, args.data_dir, 'test'
    )
    plot_factor_structure(
        model, dataset, args.output_dir,
        snapshot_date=args.snapshot_date,
        data_dir=args.data_dir,
        top_stocks=args.top_stocks,
    )
