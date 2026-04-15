"""Variance decomposition — stacked bar chart per stock.

For a snapshot date, decomposes each stock's empirical variance into:
  - Factor variance:       sum_f(beta_if^2)          [blue]
  - Idiosyncratic (sigma): sigma_i^2                  [orange]
  - Gap (unexplained):     var_emp - var_factor - sigma_i^2  [red]

If the model is well-calibrated the gap should be ~0.  In practice
the gap dominates (~99%) because sigma is pinned near its floor and
B is near-zero.

CLI usage:
    python src/analysis/test/variance_decomp.py \\
        --checkpoint checkpoints/simplified_v1/last.ckpt \\
        --data_dir data \\
        --output_dir results/evaluation/simplified_v1 \\
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
import torch
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.analysis.test.factor_clustermap import (
    _load_model_and_test_dataset,
    get_snapshot_beta,
)


def plot_variance_decomposition(
    model,
    dataset,
    output_dir,
    snapshot_date=None,
    data_dir="data",
    top_stocks=35,
    returns_std=1.0,
):
    """Stacked-bar variance decomposition for top_stocks stocks.

    Args:
        model: Trained NeuralFactors (Lightning) model.
        dataset: NeuralFactorsDataset (test split).
        output_dir: Path for saving the PNG.
        snapshot_date: Optional date string (nearest test date is used).
        data_dir: Root data directory (unused here, kept for API symmetry).
        top_stocks: How many stocks to display (ordered by empirical variance).
        returns_std: Scalar std used for normalising returns; used so all
                     variances are in normalised (model) space.
    """
    print("Computing variance decomposition (snapshot)...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Snapshot: get beta, alpha, sigma ─────────────────────────────────────
    beta_valid, tickers_valid, target_date = get_snapshot_beta(model, dataset, snapshot_date)
    N_valid = len(tickers_valid)
    if N_valid < 5:
        print("Not enough valid stocks for variance decomposition. Skipping.")
        return

    device = next(model.parameters()).device
    dates = dataset.dates
    if snapshot_date is not None:
        snap_ts = pd.Timestamp(snapshot_date)
        snap_idx = int(np.argmin([abs((pd.Timestamp(d) - snap_ts).days) for d in dates]))
    else:
        snap_idx = len(dates) // 2

    S, S_static, r, mask = dataset[snap_idx]
    S = S.unsqueeze(0).to(device)
    S_static = S_static.unsqueeze(0).to(device)
    r_t = r.unsqueeze(0).to(device)
    mask_t = mask.unsqueeze(0).to(device)

    with torch.no_grad():
        alpha, B, sigma, mu_q, log_sigma_q = model.model.encode(S, S_static, r_t, mask_t)

    alpha_np = alpha.squeeze(0).cpu().numpy()   # [N]
    sigma_np = sigma.squeeze(0).cpu().numpy()   # [N]
    mask_np  = mask_t.squeeze(0).cpu().numpy().astype(bool)

    alpha_valid = alpha_np[mask_np]   # [N_valid]
    sigma_valid = sigma_np[mask_np]   # [N_valid]
    # beta_valid is already [N_valid, F] from get_snapshot_beta

    # ── Empirical variance: last 20 test dates before snapshot ───────────────
    window = 20
    start_idx = max(0, snap_idx - window)
    window_dates = dates[start_idx:snap_idx]

    ticker_to_idx = {t: i for i, t in enumerate(tickers_valid)}
    ret_matrix = np.full((len(window_dates), N_valid), np.nan)
    for d_i, d in enumerate(window_dates):
        day_rets = dataset._returns_cache.get(d, {})
        for ticker, ret_val in day_rets.items():
            if ticker in ticker_to_idx:
                ret_matrix[d_i, ticker_to_idx[ticker]] = ret_val

    # Empirical variance in normalised space (dataset already normalises)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        var_emp = np.nanvar(ret_matrix, axis=0, ddof=1)  # [N_valid]

    # ── Predicted variance components ────────────────────────────────────────
    var_factor = np.sum(beta_valid ** 2, axis=1)   # sum_f beta_if^2  [N_valid]
    var_sigma  = sigma_valid ** 2                   # [N_valid]
    gap        = var_emp - var_factor - var_sigma   # [N_valid]; may be negative

    # ── Select top_stocks by empirical variance ───────────────────────────────
    n_show = min(top_stocks, N_valid)
    order = np.argsort(var_emp)[::-1][:n_show]
    labels = [tickers_valid[i].split('.')[0] for i in order]
    vf = var_factor[order]
    vs = var_sigma[order]
    vg = gap[order]

    # ── Plot horizontal stacked bar ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, max(6, 0.3 * n_show + 2)))
    y = np.arange(n_show)

    bar_h = 0.7
    ax.barh(y, vf, height=bar_h, color='#1f77b4', label='Factor variance  (Σ β²_if)')
    ax.barh(y, vs, height=bar_h, left=vf, color='#ff7f0e', label='Idiosyncratic (σ²_i)')
    # Gap: plot from (vf+vs) with positive extent; negative gap → bar goes left
    ax.barh(y, vg, height=bar_h, left=vf + vs, color='#d62728',
            label='Gap (var_empirical − predicted)')

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Variance (normalised return space)', fontsize=11)
    ax.set_title(
        f'Variance Decomposition — {str(target_date)[:10]}\n'
        f'(Gap ≈ 0 if calibrated; dominates when σ or β are near-zero)',
        fontsize=12, fontweight='bold',
    )
    ax.legend(loc='lower right', fontsize=9)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.grid(True, axis='x', alpha=0.3)
    ax.invert_yaxis()

    # Annotate coverage ratio
    total_pred = (vf + vs).sum()
    total_emp  = var_emp[order].sum()
    coverage   = total_pred / total_emp * 100 if total_emp > 0 else 0.0
    ax.text(0.99, 0.02, f'Predicted / Empirical variance: {coverage:.1f}%',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
            color='darkred', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    fig.tight_layout()
    out_path = output_dir / 'factor_variance_decomp.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Variance decomposition saved to: {out_path}")
    print(f"  Predicted/empirical coverage: {coverage:.1f}%")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--output_dir', default='results/evaluation/neuralfactors/plots')
    parser.add_argument('--snapshot_date', default=None)
    parser.add_argument('--top_stocks', type=int, default=35)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    from src.analysis.test.loader import load_model_and_data
    model, dataloader, dataset, returns_std, device = load_model_and_data(
        args.checkpoint, args.data_dir, 'test'
    )
    plot_variance_decomposition(
        model, dataset, args.output_dir,
        snapshot_date=args.snapshot_date,
        data_dir=args.data_dir,
        top_stocks=args.top_stocks,
        returns_std=returns_std,
    )
