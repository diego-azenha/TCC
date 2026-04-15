"""Probability Integral Transform (PIT) histogram.

For each (stock, date) in the test set, computes:
    u_it = Φ( (r_it − α_it) / σ_pit_it )
where σ_pit_it = sqrt(σ_it² + ||β_it||²) is the marginal standard deviation
(prior N(0,I), so Var[r|no z] = B B' + diag(σ²)).

If the model is calibrated, {u_it} ~ Uniform(0,1) → flat histogram.
  - U-shape: tails under-estimated (distribution too narrow)
  - ∩-shape:  tails over-estimated (distribution too wide)
  - Left/right skew: systematic mean bias

A Kolmogorov-Smirnov test against U(0,1) is annotated to quantify calibration.

CLI usage:
    python src/analysis/test/pit.py \\
        --checkpoint checkpoints/simplified_v1/last.ckpt \\
        --data_dir data \\
        --output_dir results/evaluation/simplified_v1/plots \\
        --mode debug
"""

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*torch.load.*')

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy import stats
from tqdm import tqdm
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent))


def compute_and_plot_pit(model, dataloader, dataset, output_dir, mode, device):
    """Compute PIT values for all test (stock, date) pairs and plot histogram.

    Args:
        model: Trained NeuralFactors (Lightning) model.
        dataloader: DataLoader over test split.
        dataset: NeuralFactorsDataset (test split).
        output_dir: Path for saving the PNG.
        mode: 'debug' (first 50 dates) or 'paper' (all dates).
        device: torch device.
    """
    print("\n" + "=" * 80)
    print("COMPUTING PIT HISTOGRAM")
    print("=" * 80)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_dates = 50 if mode == 'debug' else None
    all_u = []

    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, desc="PIT")):
            if max_dates is not None and idx >= max_dates:
                break

            S, S_static, r, mask = [x.to(device) for x in batch]

            alpha, B, sigma, mu_q, log_sigma_q = model.model.encode(S, S_static, r, mask)

            # squeeze batch dim (always 1)
            alpha_np = alpha.squeeze(0).cpu().numpy()   # [N]
            B_np     = B.squeeze(0).cpu().numpy()        # [N, F]
            sigma_np = sigma.squeeze(0).cpu().numpy()   # [N]
            r_np     = r.squeeze(0).cpu().numpy()        # [N]
            mask_np  = mask.squeeze(0).cpu().numpy().astype(bool)

            # Marginal std per stock: sqrt(sigma_i^2 + ||beta_i||^2)
            beta_norm_sq = np.sum(B_np**2, axis=1)          # [N]
            sigma_pit    = np.sqrt(sigma_np**2 + beta_norm_sq)  # [N]

            # u_it = Phi( (r_it - alpha_it) / sigma_pit_it )
            valid_idx = np.where(mask_np)[0]
            for i in valid_idx:
                if sigma_pit[i] > 1e-8:
                    z_score = (r_np[i] - alpha_np[i]) / sigma_pit[i]
                    u = float(stats.norm.cdf(z_score))
                    all_u.append(u)

    if len(all_u) < 10:
        print("Not enough PIT observations. Skipping.")
        return

    all_u = np.array(all_u)
    n_obs = len(all_u)
    print(f"  PIT observations: {n_obs:,}")

    # ── KS test against U(0,1) ────────────────────────────────────────────────
    ks_stat, ks_p = stats.kstest(all_u, 'uniform')

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    n_bins = 20
    counts, edges, patches = ax.hist(
        all_u, bins=n_bins, range=(0, 1),
        density=True, color='#1f77b4', alpha=0.75, edgecolor='white', linewidth=0.5,
    )
    # Uniform density reference
    ax.axhline(1.0, color='red', linestyle='--', linewidth=1.5, label='Uniform(0,1) density')

    # Shade deviations from uniform
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    for cnt, xc in zip(counts, bin_centers):
        color = '#d62728' if cnt > 1.0 else '#2ca02c'
        ax.annotate('', xy=(xc, cnt), xytext=(xc, 1.0),
                    arrowprops=dict(arrowstyle='-', color=color, lw=1.0, alpha=0.4))

    ax.set_xlabel('PIT value  u = Φ((r − α) / σ_pit)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_title(
        'Probability Integral Transform (PIT) Histogram\n'
        'Flat = well-calibrated  |  U-shape = tails under-estimated  |  ∩ = over-estimated',
        fontsize=12, fontweight='bold',
    )
    ax.legend(loc='upper center', fontsize=10)

    # Annotation box
    ax.text(
        0.99, 0.97,
        f'n = {n_obs:,}\nKS stat = {ks_stat:.4f}\nKS p-value = {ks_p:.2e}',
        transform=ax.transAxes, ha='right', va='top', fontsize=10,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85),
    )

    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = output_dir / 'pit_histogram.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ PIT histogram saved to: {out_path}")
    print(f"  KS statistic = {ks_stat:.4f}  |  p-value = {ks_p:.2e}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--output_dir', default='results/evaluation/neuralfactors/plots')
    parser.add_argument('--mode', default='paper', choices=['debug', 'paper'])
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    from src.analysis.test.loader import load_model_and_data
    from torch.utils.data import DataLoader
    from src.utils.dataset import collate_fn
    model, dataloader, dataset, returns_std, device = load_model_and_data(
        args.checkpoint, args.data_dir, 'test'
    )
    compute_and_plot_pit(
        model, dataloader, dataset, args.output_dir, args.mode, device
    )
