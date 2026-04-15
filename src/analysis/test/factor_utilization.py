"""Factor utilization — dual bar chart: KL per factor + variance explained per factor.

Left panel: KL divergence per factor (sorted descending).
  Bars below the free-bits threshold are red ("on life support").

Right panel: Variance explained per factor = mean_i(beta_if^2).
  Factors in same order as left panel to show correspondence.

CLI usage:
    python src/analysis/test/factor_utilization.py \\
        --checkpoint checkpoints/simplified_v1/last.ckpt \\
        --data_dir data \\
        --output_dir results/evaluation/simplified_v1/plots \\
        --snapshot_date 2024-06-15
"""

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*torch.load.*')

import argparse
import json
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


def plot_factor_utilization(
    model,
    dataset,
    output_dir,
    snapshot_date=None,
    free_bits_lambda=0.1,
):
    """Two-panel bar chart showing which factors are active and useful.

    Args:
        model: Trained NeuralFactors (Lightning) model.
        dataset: NeuralFactorsDataset (test split).
        output_dir: Path for saving the PNG.
        snapshot_date: Optional date string.
        free_bits_lambda: Floor threshold; factors below this are "on life support".
    """
    print("Computing factor utilization (single-day snapshot)...")
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

    S, S_static, r_t, mask_t = dataset[snap_idx]
    device = next(model.parameters()).device
    S = S.unsqueeze(0).to(device)
    S_static = S_static.unsqueeze(0).to(device)
    r_t = r_t.unsqueeze(0).to(device)
    mask_t = mask_t.unsqueeze(0).to(device)

    with torch.no_grad():
        alpha, B, sigma, mu_q, log_sigma_q = model.model.encode(S, S_static, r_t, mask_t)

    mu_q_np       = mu_q.squeeze(0).cpu().numpy()        # [F]
    log_sigma_q_np = log_sigma_q.squeeze(0).cpu().numpy()  # [F]
    sigma_q_np    = np.exp(log_sigma_q_np)               # [F]
    beta_np       = B.squeeze(0).cpu().numpy()            # [N, F]
    mask_np       = mask_t.squeeze(0).cpu().numpy().astype(bool)

    beta_valid = beta_np[mask_np]  # [N_valid, F]
    F = beta_valid.shape[1]

    # ── Per-factor KL (Gaussian analytical) ──────────────────────────────────
    kl_f = 0.5 * (sigma_q_np**2 + mu_q_np**2 - 1.0 - 2.0 * log_sigma_q_np)  # [F]

    # ── Per-factor variance explained = mean_i(beta_if^2) ────────────────────
    var_exp = np.mean(beta_valid**2, axis=0)  # [F]

    # ── Sort by KL descending ─────────────────────────────────────────────────
    order = np.argsort(kl_f)[::-1]
    kl_sorted   = kl_f[order]
    var_sorted  = var_exp[order]
    labels      = [f'f{order[i]}' for i in range(F)]

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, (ax_kl, ax_var) = plt.subplots(1, 2, figsize=(14, max(6, 0.35 * F + 2)))
    y = np.arange(F)
    bar_h = 0.75

    # Left: KL per factor
    colors_kl = ['#d62728' if v < free_bits_lambda else '#1f77b4' for v in kl_sorted]
    ax_kl.barh(y, kl_sorted, height=bar_h, color=colors_kl)
    ax_kl.axvline(free_bits_lambda, color='red', linestyle='--', linewidth=1.2,
                  label=f'free_bits λ={free_bits_lambda}')
    ax_kl.set_yticks(y)
    ax_kl.set_yticklabels(labels, fontsize=8)
    ax_kl.set_xlabel('KL divergence (nats)', fontsize=11)
    ax_kl.set_title(
        f'KL per factor — {str(target_date)[:10]}\n'
        f'(red = on life support; blue = alive)',
        fontsize=11, fontweight='bold',
    )
    ax_kl.legend(fontsize=9)
    ax_kl.invert_yaxis()
    ax_kl.grid(True, axis='x', alpha=0.3)

    n_alive = int(np.sum(kl_sorted >= free_bits_lambda))
    ax_kl.text(0.99, 0.01, f'Active: {n_alive}/{F}',
               transform=ax_kl.transAxes, ha='right', va='bottom', fontsize=9,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    # Right: variance explained (same factor order as left panel)
    ax_var.barh(y, var_sorted, height=bar_h, color='#2ca02c')
    ax_var.set_yticks(y)
    ax_var.set_yticklabels(labels, fontsize=8)
    ax_var.set_xlabel('mean_i(β²_if)  [explained variance per factor]', fontsize=11)
    ax_var.set_title(
        'Variance explained per factor\n(same order as KL panel)',
        fontsize=11, fontweight='bold',
    )
    ax_var.invert_yaxis()
    ax_var.grid(True, axis='x', alpha=0.3)

    fig.suptitle('Factor Utilization', fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    out_path = output_dir / 'factor_utilization.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Factor utilization saved to: {out_path}")
    print(f"  Active factors (KL ≥ λ): {n_alive}/{F}")
    print(f"  Max variance explained by single factor: {var_sorted.max():.6f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--output_dir', default='results/evaluation/neuralfactors/plots')
    parser.add_argument('--snapshot_date', default=None)
    parser.add_argument('--free_bits_lambda', type=float, default=0.1)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    from src.analysis.test.loader import load_model_and_data
    import json
    ckpt_parent = Path(args.checkpoint).parent
    cfg_path = ckpt_parent / 'config.json'
    if not cfg_path.exists():
        cfg_path = ckpt_parent.parent / 'config.json'
    fbl = args.free_bits_lambda
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)
        fbl = cfg.get('training', {}).get('free_bits_lambda', fbl)

    model, dataloader, dataset, returns_std, device = load_model_and_data(
        args.checkpoint, args.data_dir, 'test'
    )
    plot_factor_utilization(
        model, dataset, args.output_dir,
        snapshot_date=args.snapshot_date,
        free_bits_lambda=fbl,
    )
