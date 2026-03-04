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
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))  # project root

from src.models.lightning_module import NeuralFactorsLightning
from src.utils.dataset import NeuralFactorsDataset, collate_fn
from src.utils.config import get_default_config
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze trained NeuralFactors model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="results/training_analysis", help="Output directory for plots")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples for predictions")
    return parser.parse_args()


def load_model_and_data(checkpoint_path, data_dir, split="test"):
    """Load trained model and dataset."""
    print(f"Loading model from {checkpoint_path}...")
    
    # Load model (strict=False to ignore polyak_model keys in checkpoint)
    model = NeuralFactorsLightning.load_from_checkpoint(checkpoint_path, strict=False)
    model.eval()
    model = model.cuda() if torch.cuda.is_available() else model

    # Read hyperparameters directly from the loaded model so analyze.py is always
    # consistent with whatever settings were used during training.
    lookback   = model.model_config.lookback
    train_end  = model.training_config.checkpoint_dir  # fallback default below
    # training_config doesn't store split dates, so read from config.json if present
    config_json = Path(checkpoint_path).parent / "config.json"
    train_end_date = "2018-12-31"
    val_end_date   = "2022-12-31"
    if config_json.exists():
        import json as _json
        with open(config_json) as _f:
            _cfg = _json.load(_f)
        train_end_date = _cfg.get("args", {}).get("train_end", train_end_date)
        val_end_date   = _cfg.get("args", {}).get("val_end",   val_end_date)
    print(f"Model config: lookback={lookback}, train_end={train_end_date}, val_end={val_end_date}")

    # Load dataset
    print(f"Loading {split} dataset...")
    x_ts_file = Path(data_dir) / "parquets" / "x_ts.parquet"
    x_static_file = Path(data_dir) / "parquets" / "x_static.parquet"
    prices_file = Path(data_dir) / "cleaned" / "fechamentos_ibx.csv"
    
    # Compute returns_std from training data for normalization
    if split in ['val', 'test']:
        import pandas as pd
        from src.utils.data_utils import compute_returns_std_from_train
        df_prices = pd.read_csv(prices_file, sep=';', decimal=',', parse_dates=['DATES'], dayfirst=True)
        df_prices.rename(columns={'DATES': 'date'}, inplace=True)
        returns_std = compute_returns_std_from_train(df_prices, train_end=train_end_date)
        print(f"Returns std from training data: {returns_std:.6f}")
    else:
        returns_std = None
    
    dataset = NeuralFactorsDataset(
        x_ts_path=str(x_ts_file),
        x_static_path=str(x_static_file),
        prices_path=str(prices_file),
        split=split,
        lookback=lookback,
        returns_std=returns_std,
        train_end=train_end_date,
        val_end=val_end_date,
    )
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    return model, dataloader, dataset


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
        
        # Get all event files
        event_files = list(log_path.rglob("events.out.tfevents.*"))
        if not event_files:
            print(f"Warning: No TensorBoard event files found in {log_dir}")
            return
        
        # Use the most recent event file
        event_file = max(event_files, key=lambda p: p.stat().st_mtime)
        print(f"Reading TensorBoard logs from: {event_file}")
        
        # Load events
        ea = event_accumulator.EventAccumulator(str(event_file.parent))
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
            
            # Plot raw
            axes[0, 0].plot(steps, values, alpha=0.3, color='blue', linewidth=0.5, label='Raw')
            
            # Plot smoothed (rolling mean)
            window = min(100, len(values) // 10)
            if window > 1:
                smoothed = pd.Series(values).rolling(window, center=True).mean()
                axes[0, 0].plot(steps, smoothed, color='blue', linewidth=2, label=f'Smoothed (window={window})')
            
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Effective Sample Size (ESS)
        if 'train/ess' in scalar_tags:
            ess = ea.Scalars('train/ess')
            steps = [e.step for e in ess]
            values = [e.value for e in ess]
            axes[0, 1].plot(steps, values, color='green', alpha=0.6)
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('ESS')
            axes[0, 1].set_title('Effective Sample Size (ESS)')
            axes[0, 1].axhline(y=20, color='red', linestyle='--', label='k=20 (num samples)')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Prior Sigma_z (scale parameter)
        if 'train/prior_sigma_z_mean' in scalar_tags:
            sigma = ea.Scalars('train/prior_sigma_z_mean')
            steps = [e.step for e in sigma]
            values = [e.value for e in sigma]
            axes[1, 0].plot(steps, values, color='purple')
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Mean Sigma_z')
            axes[1, 0].set_title('Prior Scale Parameter (Sigma_z)')
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
    all_nu = []
    stock_names = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            
            S, S_static, r, mask = batch
            if torch.cuda.is_available():
                S, S_static, r, mask = S.cuda(), S_static.cuda(), r.cuda(), mask.cuda()
            
            # Get factor exposures
            alpha, B, sigma, nu, mu_q, L_q = model.model.encode(S, S_static, r, mask)
            
            all_beta.append(B.cpu().numpy())  # [1, N, F]
            all_alpha.append(alpha.cpu().numpy())
            all_sigma.append(sigma.cpu().numpy())
            all_nu.append(nu.cpu().numpy())
    
    # Concatenate all batches
    all_beta = np.concatenate(all_beta, axis=1)[0]  # [N_total, F]
    all_alpha = np.concatenate(all_alpha, axis=1)[0]
    all_sigma = np.concatenate(all_sigma, axis=1)[0]
    all_nu = np.concatenate(all_nu, axis=1)[0]
    
    F = all_beta.shape[1]
    
    # 1. Heatmap of factor exposures
    plt.figure(figsize=(14, 10))
    sns.heatmap(all_beta[:100], cmap='RdBu_r', center=0, cbar_kws={'label': 'Factor Exposure'})
    plt.xlabel('Factors')
    plt.ylabel('Stocks')
    plt.title(f'Factor Exposures Heatmap (First 100 Stocks, {F} Factors)')
    plt.tight_layout()
    plt.savefig(output_dir / 'factor_exposures_heatmap.png', dpi=300)
    plt.close()
    
    # 2. Hierarchical clustering of factors
    plt.figure(figsize=(12, 6))
    linkage_matrix = linkage(all_beta.T, method='ward')
    dendrogram(linkage_matrix)
    plt.xlabel('Factor Index')
    plt.ylabel('Distance')
    plt.title('Hierarchical Clustering of Factors')
    plt.tight_layout()
    plt.savefig(output_dir / 'factor_clustering.png', dpi=300)
    plt.close()
    
    # 3. t-SNE visualization of stock embeddings
    if all_beta.shape[0] > 50:  # Only if we have enough stocks
        print("Computing t-SNE of factor exposures...")
        tsne = TSNE(n_components=2, random_state=42)
        beta_2d = tsne.fit_transform(all_beta)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(beta_2d[:, 0], beta_2d[:, 1], alpha=0.5)
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.title('t-SNE of Stock Factor Exposures')
        plt.tight_layout()
        plt.savefig(output_dir / 'factor_exposures_tsne.png', dpi=300)
        plt.close()
    
    # 4. Distribution of alpha, sigma, nu
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].hist(all_alpha, bins=50, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Alpha (Idiosyncratic Return)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Alpha')
    
    axes[1].hist(all_sigma, bins=50, alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Sigma (Scale)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Sigma')
    
    axes[2].hist(all_nu, bins=50, alpha=0.7, edgecolor='black')
    axes[2].set_xlabel('Nu (Degrees of Freedom)')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Distribution of Nu')
    
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
    print(f"Nu (Degrees of Freedom): mean={all_nu.mean():.4f}, std={all_nu.std():.4f}")


def analyze_prior_parameters(model, output_dir):
    """Analyze learned prior parameters."""
    print("Analyzing prior parameters...")
    output_dir = Path(output_dir)
    
    mu_z, sigma_z, nu_z = model.model.prior.get_params()
    mu_z = mu_z.detach().cpu().numpy()
    sigma_z = sigma_z.detach().cpu().numpy()
    nu_z = nu_z.detach().cpu().item()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot sigma_z distribution
    axes[0].hist(sigma_z, bins=30, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Sigma_z (Factor Scale)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'Prior Sigma_z Distribution\nMean: {sigma_z.mean():.3f}, Std: {sigma_z.std():.3f}')
    
    # Plot mu_z distribution
    axes[1].hist(mu_z, bins=30, alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Mu_z (Factor Mean)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'Prior Mu_z Distribution\nMean: {mu_z.mean():.3f}, Std: {mu_z.std():.3f}')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'prior_parameters.png', dpi=300)
    plt.close()
    
    print(f"\nPrior Parameters:")
    print(f"Nu_z (degrees of freedom): {nu_z:.4f}")
    print(f"Sigma_z: mean={sigma_z.mean():.4f}, std={sigma_z.std():.4f}")
    print(f"Mu_z: mean={mu_z.mean():.4f}, std={mu_z.std():.4f}")


def main():
    args = parse_args()
    
    # Load model and data
    model, dataloader, dataset = load_model_and_data(
        args.checkpoint, 
        args.data_dir, 
        args.split
    )
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"NeuralFactors Model Analysis")
    print(f"{'='*60}\n")

    # Deriva experiment_name a partir do config.json salvo junto ao checkpoint
    config_json = Path(args.checkpoint).parent / "config.json"
    experiment_name = "neuralfactors"  # fallback
    if config_json.exists():
        with open(config_json) as _f:
            _cfg = json.load(_f)
        experiment_name = _cfg.get("args", {}).get("experiment_name", experiment_name)

    # Plot training curves from TensorBoard logs
    log_dir = f"logs/{experiment_name}"
    plot_loss_curves(log_dir, output_dir)
    
    # Run analyses
    analyze_prior_parameters(model, output_dir)
    analyze_factor_exposures(model, dataloader, output_dir)
    
    print(f"\n{'='*60}")
    print(f"Analysis complete! Plots saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
