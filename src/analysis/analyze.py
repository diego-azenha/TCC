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
import seaborn as sns
import torch
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
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



def plot_loss_curves(log_dir, output_dir):
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
        
        # 3. KL per-factor min (should stay > free_bits threshold)
        kl_min_tag = 'train/kl_min_factor'
        if kl_min_tag in scalar_tags:
            kl_min = ea.Scalars(kl_min_tag)
            steps = [e.step for e in kl_min]
            values = [e.value for e in kl_min]
            axes[1, 0].plot(steps, values, color='purple')
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('KL_min factor')
            axes[1, 0].set_title('Min per-factor KL divergence')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Decoder parameters (alpha, sigma)
        if 'train/alpha_mean' in scalar_tags and 'train/sigma_mean' in scalar_tags:
            alpha = ea.Scalars('train/alpha_mean')
            sigma = ea.Scalars('train/sigma_mean')
            steps_alpha = [e.step for e in alpha]
            values_alpha = [e.value for e in alpha]
            steps_sigma = [e.step for e in sigma]
            values_sigma = [e.value for e in sigma]
            
            ax_twin = axes[1, 1].twinx()
            axes[1, 1].plot(steps_alpha, values_alpha, color='orange', label='Alpha (location)')
            ax_twin.plot(steps_sigma, values_sigma, color='brown', label='Sigma (scale)')
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Alpha', color='orange')
            ax_twin.set_ylabel('Sigma', color='brown')
            axes[1, 1].set_title('Decoder Parameters')
            axes[1, 1].legend(loc='upper left')
            ax_twin.legend(loc='upper right')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = output_dir / "training_curves.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved training curves to: {output_path}")
        
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

    # 1. Hierarchical clustering of factors
    plt.figure(figsize=(12, 6))
    linkage_matrix = linkage(all_beta.T, method='ward')
    dendrogram(linkage_matrix)
    plt.xlabel('Factor Index')
    plt.ylabel('Distance')
    plt.title('Hierarchical Clustering of Factors')
    plt.tight_layout()
    plt.savefig(output_dir / 'factor_clustering.png', dpi=300)
    plt.close()
    
    # 3. Distribution of alpha, sigma
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    def _clipped_hist(ax, data, xlabel, title):
        lo, hi = np.percentile(data, 1), np.percentile(data, 99)
        clipped = data[(data >= lo) & (data <= hi)]
        pct_shown = 100 * len(clipped) / max(len(data), 1)
        data_range = hi - lo
        # Near-constant data: bins='auto' produces invisibly thin bars.
        # Use 50 fixed bins and annotate with mean ± std instead.
        near_constant = data_range < 1e-3 * max(abs(lo + hi) / 2, 1e-9)
        bins = 50 if near_constant else 'auto'
        try:
            ax.hist(clipped, bins=bins, alpha=0.7, edgecolor='black')
            if lo < hi:
                ax.set_xlim(lo - data_range * 0.1, hi + data_range * 0.1)
            if near_constant:
                ax.annotate(
                    f"Near-constant: mean={np.mean(data):.5f}, std={np.std(data):.2e}",
                    xy=(0.5, 0.97), xycoords='axes fraction',
                    ha='center', va='top', fontsize=8, color='red',
                )
        except ValueError:
            ax.text(0.5, 0.5, f'All values = {np.mean(data):.4f}',
                    ha='center', va='center', transform=ax.transAxes)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Frequency')
        ax.set_title(f'{title}\n(1st-99th pct, {pct_shown:.0f}% of data)')

    _clipped_hist(axes[0], all_alpha, 'Alpha (Idiosyncratic Return)', 'Distribution of Alpha')
    _clipped_hist(axes[1], all_sigma, 'Sigma (Scale)', 'Distribution of Sigma')

    plt.tight_layout()
    plt.savefig(output_dir / 'decoder_param_distributions.png', dpi=300)
    plt.close()
    
    print(f"Factor analysis plots saved to {output_dir}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("Factor Exposure Summary Statistics:")
    print("="*60)
    print(f"Beta (Factor Exposures): mean={all_beta.mean():.4f}, std={all_beta.std():.4f}")
    print(f"Alpha (Idiosyncratic): mean={all_alpha.mean():.4f}, std={all_alpha.std():.4f}")
    print(f"Sigma (Scale): mean={all_sigma.mean():.4f}, std={all_sigma.std():.4f}")





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
    plot_loss_curves(log_dir, output_dir)
    
    # Run analyses
    analyze_factor_exposures(model, dataloader, output_dir)
    
    print(f"\n{'='*60}")
    print(f"Analysis complete! Plots saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
