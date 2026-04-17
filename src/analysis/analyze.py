"""Analysis and visualization script for NeuralFactors model.

Generates plots and metrics based on paper Section 5:
- Factor analysis (Section 5.3)
- Predicted vs actual returns
- Covariance matrix comparison
- Risk analysis (VaR calibration)

Usage:
    python scripts/analyze.py --checkpoint checkpoints/neuralfactors/best.ckpt --data_dir data
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
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))  # project root

from src.analysis.test.loader import load_model_and_data
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze trained NeuralFactors model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="results/training_analysis", help="Output directory for plots")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples for predictions")
    return parser.parse_args()



def plot_loss_curves(log_dir, output_dir, kl_warmup_steps=0):
    """Plot training curves from TensorBoard logs."""
    print("Plotting loss curves...")
    output_dir = Path(output_dir)
    
    try:
        from tensorboard.backend.event_processing import event_accumulator
        
        # Find the latest version directory
        log_path = Path(log_dir)
        if not log_path.exists():
            print(f"Warning: Log directory {log_dir} not found. Skipping loss curves.")
            return
        
        # Pick the highest-numbered version_N directory (most recent training run)
        version_dirs = sorted(
            [d for d in log_path.iterdir() if d.is_dir() and d.name.startswith("version_")],
            key=lambda d: int(d.name.split("_")[1])
        )
        if not version_dirs:
            print(f"Warning: No version_N directories found in {log_dir}")
            return

        version_dir = version_dirs[-1]
        event_files = list(version_dir.glob("events.out.tfevents.*"))
        if not event_files:
            print(f"Warning: No TensorBoard event files found in {version_dir}")
            return

        print(f"Reading TensorBoard logs from: {version_dir}  ({len(version_dirs)} versions found, using latest)")

        # Load events from that specific version directory
        ea = event_accumulator.EventAccumulator(str(version_dir))
        ea.Reload()
        
        # Get available scalars
        scalar_tags = ea.Tags()['scalars']
        
        # Plot loss curves
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Metrics Over Time', fontsize=16, fontweight='bold')
        
        # 1. Training Loss (with smoothing)
        if 'train/loss_step' in scalar_tags:
            train_loss = ea.Scalars('train/loss_step')
            steps = [e.step for e in train_loss]
            values = [e.value for e in train_loss]
            
            vals_arr = np.array(values)
            # Clip y-axis at [5th, 99th] percentile — spikes are noise, not signal
            y_lo = np.percentile(vals_arr, 5)
            y_hi = np.percentile(vals_arr, 99)

            # Plot raw (only within clipped window so it isn't dominated by spikes)
            axes[0, 0].plot(steps, values, alpha=0.25, color='blue', linewidth=0.5, label='Raw')

            # Plot smoothed (rolling mean)
            window = min(100, len(values) // 10)
            if window > 1:
                smoothed = pd.Series(values).rolling(window, center=True).mean()
                axes[0, 0].plot(steps, smoothed, color='blue', linewidth=2, label=f'Smoothed (w={window})')

            axes[0, 0].set_ylim(y_lo - abs(y_lo) * 0.1, y_hi + abs(y_hi) * 0.1)
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Training Loss (5th–99th pct y-axis)')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Encoder health: sigma_q_mean (should stay ~1.0, collapse if → 0)
        sigma_q_tag = 'train/sigma_q_mean'
        if sigma_q_tag in scalar_tags:
            sigma_q = ea.Scalars(sigma_q_tag)
            steps = [e.step for e in sigma_q]
            values = [e.value for e in sigma_q]
            axes[0, 1].plot(steps, values, color='green', alpha=0.6)
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('σ_q mean')
            axes[0, 1].set_title('Posterior scale σ_q mean\n(≈1.0 healthy; → 0 = collapse)')
            axes[0, 1].axhline(y=1.0, color='red', linestyle='--', label='ideal σ_q=1')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Variance competition ZOOM (first 20k steps) — early-training dynamics
        _comp_series_zoom = {}
        for tag, label, color in [
            ('train/alpha_std',      'alpha_std',     'orange'),
            ('train/sigma_mean',     'mean(sigma)',    'steelblue'),
            ('train/beta_norm_mean', 'mean(||β_i||)', 'green'),
        ]:
            if tag in scalar_tags:
                evts = ea.Scalars(tag)
                s = [e.step for e in evts if e.step <= 20_000]
                v = [e.value for e in evts if e.step <= 20_000]
                if s:
                    _comp_series_zoom[label] = {'steps': s, 'vals': v, 'color': color}
        if _comp_series_zoom:
            for label, d in _comp_series_zoom.items():
                axes[1, 0].plot(d['steps'], d['vals'], color=d['color'],
                                linewidth=1.5, label=label, alpha=0.85)
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Value')
            axes[1, 0].set_title('Variance competition (first 20k steps)\n(early dynamics: who wins first?)')
            axes[1, 0].legend(fontsize=9)
            axes[1, 0].grid(True, alpha=0.3)

        # 4. Variance competition: alpha_std, mean(sigma), mean(||beta_i||)
        # Shows which channel dominates: alpha shortcut, noise floor, or true factors
        _comp_series = {}
        for tag, label, color in [
            ('train/alpha_std',      'alpha_std',     'orange'),
            ('train/sigma_mean',     'mean(sigma)',    'steelblue'),
            ('train/beta_norm_mean', 'mean(||β_i||)', 'green'),
        ]:
            if tag in scalar_tags:
                evts = ea.Scalars(tag)
                _comp_series[label] = {
                    'steps': [e.step for e in evts],
                    'vals':  [e.value for e in evts],
                    'color': color,
                }
        if _comp_series:
            for label, d in _comp_series.items():
                axes[1, 1].plot(d['steps'], d['vals'], color=d['color'],
                                linewidth=1.5, label=label, alpha=0.85)
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Value')
            missing = {'train/beta_norm_mean'} - set(scalar_tags)
            note = ' (beta_norm absent: re-train)' if missing else ''
            axes[1, 1].set_title(f'Variance competition{note}\n(alpha_std↑ + sigma/β low → alpha shortcut)')
            axes[1, 1].legend(fontsize=9)
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = output_dir / "training_curves.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved training curves to: {output_path}")

        # ── Loss decomposition panel (2×2) ────────────────────────────────────
        _ld_tags = [
            ('train/L_recon_step',      'L_recon',            '#1f77b4'),
            ('train/L_sigma_step',      'L_sigma',            '#ff7f0e'),
            ('train/kl_divergence_step','KL divergence',      '#2ca02c'),
            ('train/free_bits_penalty', 'Free bits penalty',  '#d62728'),
            ('train/loss_step',         'Total loss',         '#9467bd'),
            ('train/kl_weight',         'kl_weight',          '#ff7f0e'),
        ]
        _ld_data = {}
        for tag, lbl, col in _ld_tags:
            if tag in scalar_tags:
                evts = ea.Scalars(tag)
                _ld_data[tag] = {
                    'steps': [e.step for e in evts],
                    'vals':  [e.value for e in evts],
                    'label': lbl,
                    'color': col,
                }

        _have_ld = any(t in _ld_data for t in [
            'train/L_recon_step', 'train/kl_divergence_step',
            'train/free_bits_penalty', 'train/loss_step',
        ])
        if _have_ld:
            def _ld_plot(ax, tag, smoothing=True):
                if tag not in _ld_data:
                    ax.text(0.5, 0.5, f'{tag}\nnot logged', transform=ax.transAxes,
                            ha='center', va='center', fontsize=9, color='grey')
                    return
                d = _ld_data[tag]
                arr = np.array(d['vals'], dtype=float)
                steps = np.array(d['steps'])
                y_lo = np.nanpercentile(arr, 5)
                y_hi = np.nanpercentile(arr, 95)
                arr_w = np.clip(arr, y_lo, y_hi)
                margin = max(abs(y_hi - y_lo) * 0.05, 1e-9)
                ax.plot(steps, arr_w, alpha=0.2, color=d['color'], linewidth=0.6)
                if smoothing:
                    w = min(100, max(1, len(arr_w) // 10))
                    sm = pd.Series(arr_w).rolling(w, center=True).mean()
                    ax.plot(steps, sm, color=d['color'], linewidth=2, label=d['label'])
                else:
                    ax.plot(steps, arr_w, color=d['color'], linewidth=1.5, label=d['label'])
                ax.set_ylim(y_lo - margin, y_hi + margin)
                if kl_warmup_steps > 0:
                    ax.axvline(kl_warmup_steps, color='grey', linestyle='--',
                               linewidth=1.0, alpha=0.7, label=f'warmup end ({kl_warmup_steps:,})')

            fig2, axes2 = plt.subplots(2, 2, figsize=(15, 10))
            fig2.suptitle('Loss Decomposition', fontsize=16, fontweight='bold')

            # [0,0] L_recon + L_sigma on same axes
            ax00 = axes2[0, 0]
            _ld_plot(ax00, 'train/L_recon_step')
            _ld_plot(ax00, 'train/L_sigma_step')
            ax00.set_title('Reconstruction components\n(L_recon = ELBO recon; L_sigma = σ calibration)', fontsize=11)
            ax00.set_xlabel('Step'); ax00.legend(fontsize=9); ax00.grid(True, alpha=0.3)

            # [0,1] KL divergence
            ax01 = axes2[0, 1]
            _ld_plot(ax01, 'train/kl_divergence_step')
            ax01.set_title('KL divergence\n(rises from ~0 during warmup, stabilises after)', fontsize=11)
            ax01.set_xlabel('Step'); ax01.legend(fontsize=9); ax01.grid(True, alpha=0.3)

            # [1,0] free bits penalty
            ax10 = axes2[1, 0]
            _ld_plot(ax10, 'train/free_bits_penalty')
            ax10.set_title('Free bits penalty\n(→ 0 when all factors are active)', fontsize=11)
            ax10.set_xlabel('Step'); ax10.legend(fontsize=9); ax10.grid(True, alpha=0.3)

            # [1,1] total loss (left y) + kl_weight ramp (right y, twin axis)
            ax11 = axes2[1, 1]
            _ld_plot(ax11, 'train/loss_step')
            ax11.set_title('Total loss + KL weight ramp', fontsize=11)
            ax11.set_xlabel('Step'); ax11.legend(loc='upper right', fontsize=9); ax11.grid(True, alpha=0.3)
            if 'train/kl_weight' in _ld_data:
                ax11b = ax11.twinx()
                d_kw = _ld_data['train/kl_weight']
                ax11b.plot(d_kw['steps'], d_kw['vals'],
                           color='#ff7f0e', linestyle='--', linewidth=1.5,
                           alpha=0.85, label='kl_weight')
                ax11b.set_ylim(-0.05, 1.15)
                ax11b.set_ylabel('kl_weight', fontsize=9, color='#ff7f0e')
                ax11b.tick_params(axis='y', labelcolor='#ff7f0e')
                ax11b.legend(loc='center right', fontsize=9)

            fig2.tight_layout()
            ld_path = output_dir / 'loss_decomposition.png'
            fig2.savefig(ld_path, dpi=300, bbox_inches='tight')
            plt.close(fig2)
            print(f"Saved loss decomposition to: {ld_path}")

    except ImportError:
        print("Warning: tensorboard package not available. Install with: pip install tensorboard")
    except Exception as e:
        print(f"Warning: Could not plot training curves: {e}")


def analyze_factor_exposures(model, dataloader, output_dir, num_batches=50):
    """Analyze and visualize factor exposures (Paper Section 5.3)."""
    print("Analyzing factor exposures...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_beta = []
    all_alpha = []
    all_sigma = []
    stock_names = []
    
    device = next(model.parameters()).device
    if device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            
            S, S_static, r, mask = batch
            S, S_static, r, mask = S.to(device), S_static.to(device), r.to(device), mask.to(device)
            
            # Get factor exposures
            alpha, B, sigma, mu_q, log_sigma_q = model.model.encode(S, S_static, r, mask)
            
            all_beta.append(B.cpu().numpy())  # [1, N, F]
            all_alpha.append(alpha.cpu().numpy())
            all_sigma.append(sigma.cpu().numpy())
    
    # Concatenate all batches
    all_beta = np.concatenate(all_beta, axis=1)[0]  # [N_total, F]
    all_alpha = np.concatenate(all_alpha, axis=1)[0]
    all_sigma = np.concatenate(all_sigma, axis=1)[0]
    
    F = all_beta.shape[1]

    print(f"Factor analysis complete (no training-time plots — use test.py for diagnostics)")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("Factor Exposure Summary Statistics:")
    print("="*60)
    print(f"Beta (Factor Exposures): mean={all_beta.mean():.4f}, std={all_beta.std():.4f}")
    print(f"Alpha (Idiosyncratic): mean={all_alpha.mean():.4f}, std={all_alpha.std():.4f}")
    print(f"Sigma (Scale): mean={all_sigma.mean():.4f}, std={all_sigma.std():.4f}")


def plot_bootstrap_diagnostics(log_dir, output_dir):
    """Plot bootstrap diagnostic metrics: R²(α,r), signal ratio, grad norm ratio.

    These three metrics disambiguate why β gradient is weak:
      - r2_alpha > 0.10  → α is absorbing factor-level variance (Hypothesis A/B)
      - signal_ratio < 0.10 → β·μ_q signal is dominated by α (confirms intervention)
      - grad_norm_ratio < 0.01 → encoder gradient starvation (may need IWAE k)
    """
    print("Plotting bootstrap diagnostics...")
    output_dir = Path(output_dir)

    try:
        from tensorboard.backend.event_processing import event_accumulator
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("  tensorboard not available; skipping bootstrap diagnostics")
        return

    log_dir = Path(log_dir)
    if not log_dir.exists():
        print(f"  Log dir {log_dir} not found; skipping bootstrap diagnostics")
        return

    versions = sorted(log_dir.glob("version_*"), key=lambda p: int(p.name.split("_")[1]))
    if not versions:
        print(f"  No version subdirs found in {log_dir}; skipping")
        return

    print(f"  Reading TensorBoard logs from: {versions[-1]}")
    ea = EventAccumulator(str(versions[-1]))
    ea.Reload()
    available = set(ea.Tags().get('scalars', []))

    diag_tags = {
        'train/r2_alpha':                    ('R²(α, r)',           '#e74c3c', 'want < 0.10'),
        'train/signal_ratio':                ('std(β·μ_q) / std(α)', '#2ecc71', 'want > 0.10'),
        'train/beta_mu_std':                 ('std(β·μ_q)',          '#3498db', ''),
        'train/alpha_std':                   ('std(α)',              '#e67e22', ''),
        'train/grad_norm_ratio_beta_alpha':  ('||∇β|| / ||∇α||',    '#9b59b6', 'want > 0.01'),
    }

    def _load(tag):
        if tag not in available:
            return None, None
        evts = ea.Scalars(tag)
        return np.array([e.step for e in evts]), np.array([e.value for e in evts])

    def _smooth(arr, w=50):
        if len(arr) < w:
            return arr
        return pd.Series(arr).rolling(w, center=True, min_periods=1).mean().values

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Bootstrap Diagnostics\n(VAE β-gradient health vs α dominance)', fontsize=14, fontweight='bold')

    # Panel [0,0]: R²(α, r)
    ax = axes[0, 0]
    steps, vals = _load('train/r2_alpha')
    if steps is not None:
        arr_w = np.clip(vals, np.nanpercentile(vals, 5), np.nanpercentile(vals, 95))
        ax.plot(steps, arr_w, alpha=0.2, color='#e74c3c', linewidth=0.6)
        ax.plot(steps, _smooth(arr_w), color='#e74c3c', linewidth=2, label='R²(α, r)')
        ax.axhline(0.10, color='grey', linestyle='--', linewidth=1.0, alpha=0.7, label='threshold = 0.10')
    else:
        ax.text(0.5, 0.5, 'train/r2_alpha\nnot logged', transform=ax.transAxes,
                ha='center', va='center', fontsize=9, color='grey')
    ax.set_title('R²(α, r) — α share of return variance\n(> 0.10 = α absorbing factor variance)', fontsize=10)
    ax.set_xlabel('Step'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Panel [0,1]: signal ratio
    ax = axes[0, 1]
    steps, vals = _load('train/signal_ratio')
    if steps is not None:
        arr_w = np.clip(vals, np.nanpercentile(vals, 5), np.nanpercentile(vals, 95))
        ax.plot(steps, arr_w, alpha=0.2, color='#2ecc71', linewidth=0.6)
        ax.plot(steps, _smooth(arr_w), color='#2ecc71', linewidth=2, label='std(β·μ_q) / std(α)')
        ax.axhline(0.10, color='grey', linestyle='--', linewidth=1.0, alpha=0.7, label='threshold = 0.10')
    else:
        ax.text(0.5, 0.5, 'train/signal_ratio\nnot logged', transform=ax.transAxes,
                ha='center', va='center', fontsize=9, color='grey')
    ax.set_title('Signal ratio std(β·μ_q) / std(α)\n(< 0.10 = β·z signal drowned by α)', fontsize=10)
    ax.set_xlabel('Step'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Panel [1,0]: β·μ_q std vs α std (magnitude comparison)
    ax = axes[1, 0]
    steps_b, vals_b = _load('train/beta_mu_std')
    steps_a, vals_a = _load('train/alpha_std')
    if steps_b is not None:
        arr_b = np.clip(vals_b, np.nanpercentile(vals_b, 5), np.nanpercentile(vals_b, 95))
        ax.plot(steps_b, arr_b, alpha=0.2, color='#3498db', linewidth=0.6)
        ax.plot(steps_b, _smooth(arr_b), color='#3498db', linewidth=2, label='std(β·μ_q)')
    if steps_a is not None:
        arr_a = np.clip(vals_a, np.nanpercentile(vals_a, 5), np.nanpercentile(vals_a, 95))
        ax.plot(steps_a, arr_a, alpha=0.2, color='#e67e22', linewidth=0.6)
        ax.plot(steps_a, _smooth(arr_a), color='#e67e22', linewidth=2, label='std(α)')
    if steps_b is None and steps_a is None:
        ax.text(0.5, 0.5, 'beta_mu_std / alpha_std\nnot logged', transform=ax.transAxes,
                ha='center', va='center', fontsize=9, color='grey')
    ax.set_title('Magnitude comparison\n(β·μ_q should eventually compete with α)', fontsize=10)
    ax.set_xlabel('Step'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Panel [1,1]: gradient norm ratio ||∇β|| / ||∇α||
    ax = axes[1, 1]
    steps, vals = _load('train/grad_norm_ratio_beta_alpha')
    if steps is not None:
        arr_w = np.clip(vals, np.nanpercentile(vals, 5), np.nanpercentile(vals, 95))
        ax.plot(steps, arr_w, alpha=0.2, color='#9b59b6', linewidth=0.6)
        ax.plot(steps, _smooth(arr_w), color='#9b59b6', linewidth=2, label='||∇β|| / ||∇α||')
        ax.axhline(0.01, color='grey', linestyle='--', linewidth=1.0, alpha=0.7, label='threshold = 0.01')
    else:
        ax.text(0.5, 0.5, 'grad_norm_ratio_beta_alpha\nnot logged', transform=ax.transAxes,
                ha='center', va='center', fontsize=9, color='grey')
    ax.set_title('Gradient norm ratio ||∇β|| / ||∇α||\n(< 0.01 = β gradient starvation)', fontsize=10)
    ax.set_xlabel('Step'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / 'bootstrap_diagnostics.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved bootstrap diagnostics to: {out_path}")





def main():
    args = parse_args()
    
    # Load model and data
    model, dataloader, _, _returns_std, _device = load_model_and_data(
        args.checkpoint,
        args.data_dir,
        args.split,
    )
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"NeuralFactors Model Analysis")
    print(f"{'='*60}\n")

    # Deriva experiment_name a partir do config.json salvo junto ao checkpoint
    # Check parent dir first, then grandparent (checkpoint may be inside an epoch subdir)
    _ckpt_parent = Path(args.checkpoint).parent
    config_json = _ckpt_parent / "config.json"
    if not config_json.exists():
        config_json = _ckpt_parent.parent / "config.json"
    experiment_name = "neuralfactors"  # fallback
    if config_json.exists():
        with open(config_json) as _f:
            _cfg = json.load(_f)
        experiment_name = _cfg.get("args", {}).get("experiment_name", experiment_name)

    # Plot training curves from TensorBoard logs
    log_dir = f"logs/{experiment_name}"
    _kl_warmup_steps = 0
    if config_json.exists():
        _kl_warmup_steps = _cfg.get('training', {}).get('kl_warmup_steps', 0)
    plot_loss_curves(log_dir, output_dir, kl_warmup_steps=_kl_warmup_steps)
    plot_bootstrap_diagnostics(log_dir, output_dir)
    
    # Run analyses
    analyze_factor_exposures(model, dataloader, output_dir)
    
    print(f"\n{'='*60}")
    print(f"Analysis complete! Plots saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
