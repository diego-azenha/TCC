"""Training script for NeuralFactors model.

Trains the VAE-based factor model using CIWAE loss with PyTorch Lightning.
Implements all training procedures from paper Section 3.5.

Usage:
    python scripts/train.py --data_dir data --checkpoint_dir checkpoints
"""

import argparse
import json
import os
from pathlib import Path
from dataclasses import asdict
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config import ModelConfig, TrainingConfig, get_default_config
from src.utils.dataset import NeuralFactorsDataset, collate_fn
from src.utils.data_utils import discover_feature_dims, compute_returns_std_from_train
from src.models.lightning_module import NeuralFactorsLightning


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train NeuralFactors model")
    
    # Data paths
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing parquet files"
    )
    parser.add_argument(
        "--x_ts_file",
        type=str,
        default="parquets/x_ts.parquet",
        help="Time-series features parquet file (relative to data_dir)"
    )
    parser.add_argument(
        "--x_static_file",
        type=str,
        default="parquets/x_static.parquet",
        help="Static features parquet file (relative to data_dir)"
    )
    parser.add_argument(
        "--prices_file",
        type=str,
        default="cleaned/fechamentos_ibx.csv",
        help="Prices CSV file (relative to data_dir)"
    )
    
    # Model hyperparameters
    parser.add_argument("--num_factors", type=int, default=64, help="Number of latent factors F")
    parser.add_argument("--hidden_size", type=int, default=256, help="Hidden dimension h")
    parser.add_argument("--lookback", type=int, default=256, help="Lookback window L")
    parser.add_argument("--dropout", type=float, default=0.25, help="Dropout rate")
    
    # Training hyperparameters
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay")
    parser.add_argument("--num_iwae_samples", type=int, default=20, help="k in IWAE loss")
    parser.add_argument("--max_steps", type=int, default=17_290, help="5 epochs (~4.8 hours)")
    parser.add_argument("--val_every_n_steps", type=int, default=10_000, help="Validate only at the end")
    parser.add_argument("--polyak_start_step", type=int, default=8_645, help="Polyak averaging start (halfway)")
    parser.add_argument("--polyak_alpha", type=float, default=0.999, help="Polyak EMA decay")
    
    # Data split dates (adjusted for IBX data 2005-2025)
    parser.add_argument("--train_end", type=str, default="2018-12-31", help="Training end date")
    parser.add_argument("--val_end", type=str, default="2022-12-31", help="Validation end date")
    
    # Checkpointing and logging
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--log_dir", type=str, default="logs", help="TensorBoard log directory")
    parser.add_argument("--experiment_name", type=str, default="neuralfactors", help="Experiment name")
    
    # Device and reproducibility
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs")
    
    # Debugging
    parser.add_argument("--fast_dev_run", action="store_true", help="Fast dev run for debugging")
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seed
    pl.seed_everything(args.seed)
    
    # Setup paths
    data_dir = Path(args.data_dir)
    x_ts_path = data_dir / args.x_ts_file
    x_static_path = data_dir / args.x_static_file
    prices_path = data_dir / args.prices_file
    
    checkpoint_dir = Path(args.checkpoint_dir) / args.experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("NeuralFactors Training")
    print("="*80)
    
    # Discover feature dimensions
    print("\nDiscovering feature dimensions...")
    d_ts, d_static = discover_feature_dims(str(x_ts_path), str(x_static_path))
    print(f"Feature dimensions: d_ts={d_ts}, d_static={d_static}")
    
    # Compute returns std from training data
    print("\nComputing returns normalization std from training data...")
    df_prices_for_std = pd.read_csv(
        str(prices_path), 
        sep=';', 
        decimal=',',  # Handle European decimal format
        parse_dates=['DATES'], 
        dayfirst=True
    )
    df_prices_for_std.rename(columns={'DATES': 'date'}, inplace=True)
    
    returns_std = compute_returns_std_from_train(
        df_prices_for_std,
        train_end=args.train_end
    )
    print(f"Returns std: {returns_std:.6f} (paper reports ~0.02672357)")
    
    # Create model configuration
    print("\nCreating model configuration...")
    model_config = ModelConfig(
        d_ts=d_ts,
        d_static=d_static,
        num_factors=args.num_factors,
        hidden_size=args.hidden_size,
        lookback=args.lookback,
        dropout=args.dropout,
    )
    
    # Create training configuration
    training_config = TrainingConfig(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        num_iwae_samples=args.num_iwae_samples,
        val_every_n_steps=args.val_every_n_steps,
        polyak_start_step=args.polyak_start_step,
        polyak_alpha=args.polyak_alpha,
        checkpoint_dir=str(checkpoint_dir),
        log_dir=args.log_dir,
        seed=args.seed,
        returns_std=returns_std,
    )
    
    # Save configurations
    config_path = checkpoint_dir / "config.json"
    with open(config_path, 'w') as f:
        config_dict = {
            'model': asdict(model_config),
            'training': asdict(training_config),
            'args': vars(args),
        }
        json.dump(config_dict, f, indent=2)
    print(f"Saved configuration to {config_path}")
    
    # Create datasets
    print("\n" + "="*80)
    print("Loading Datasets")
    print("="*80)
    
    print("\nTraining dataset:")
    train_dataset = NeuralFactorsDataset(
        x_ts_path=str(x_ts_path),
        x_static_path=str(x_static_path),
        prices_path=str(prices_path),
        split='train',
        lookback=args.lookback,
        normalize=True,
        returns_std=None,  # Compute from data
        train_end=args.train_end,
        val_end=args.val_end,
    )
    
    print("\nValidation dataset:")
    val_dataset = NeuralFactorsDataset(
        x_ts_path=str(x_ts_path),
        x_static_path=str(x_static_path),
        prices_path=str(prices_path),
        split='val',
        lookback=args.lookback,
        normalize=True,
        returns_std=train_dataset.returns_std,  # Use training std
        train_end=args.train_end,
        val_end=args.val_end,
    )
    
    # Create dataloaders (batch_size=1 as per paper)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=11,  # Parallel data loading for better performance
        collate_fn=collate_fn,
        pin_memory=True if args.gpus > 0 else False,
        persistent_workers=True,  # Keep workers alive between epochs
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=11,  # Parallel data loading for better performance
        collate_fn=collate_fn,
        pin_memory=True if args.gpus > 0 else False,
        persistent_workers=True,  # Keep workers alive between epochs
    )
    
    print(f"\nDataLoaders created:")
    print(f"  Train: {len(train_loader)} batches")
    print(f"  Val: {len(val_loader)} batches")
    
    # Create Lightning module
    print("\n" + "="*80)
    print("Initializing Model")
    print("="*80)
    
    lightning_module = NeuralFactorsLightning(
        model_config=model_config,
        training_config=training_config,
    )
    
    print(f"\nModel parameters: {sum(p.numel() for p in lightning_module.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in lightning_module.parameters() if p.requires_grad):,}")
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='neuralfactors-{step:06d}-{train/loss_epoch:.4f}',
        monitor='train/loss_epoch',
        mode='min',
        save_top_k=3,
        save_last=True,
        every_n_epochs=1,  # Save at end of each epoch
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Setup logger
    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name=args.experiment_name,
    )
    
    # Create trainer
    print("\n" + "="*80)
    print("Starting Training")
    print("="*80)
    
    # Check if GPU is available
    if args.gpus > 0 and torch.cuda.is_available():
        accelerator = 'gpu'
        devices = args.gpus
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        accelerator = 'cpu'
        devices = None
        if args.gpus > 0:
            print("Warning: GPU requested but not available. Using CPU instead.")
        print("Using CPU")
    
    trainer = pl.Trainer(
        max_steps=args.max_steps,
        max_epochs=-1,  # Unlimited epochs - only stop when max_steps is reached
        accelerator=accelerator,
        devices=devices,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        log_every_n_steps=100,
        limit_val_batches=0.0,  # Disable validation during training
        gradient_clip_val=1.0,  # Clip gradients to prevent explosion
        gradient_clip_algorithm='norm',  # Clip by global norm
        deterministic=False,  # Faster training
        fast_dev_run=args.fast_dev_run,
        num_sanity_val_steps=0,  # Skip sanity check - model needs training first before encoder is stable
    )
    
    # Train model
    trainer.fit(
        lightning_module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print(f"\nBest model checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best validation loss: {checkpoint_callback.best_model_score:.4f}")
    
    # Save Polyak model separately if available
    if lightning_module.polyak_model is not None:
        polyak_path = checkpoint_dir / "polyak_model.pt"
        torch.save(lightning_module.polyak_model.state_dict(), polyak_path)
        print(f"Saved Polyak-averaged model to {polyak_path}")
    
    # Run automatic analysis on test set
    if not args.fast_dev_run and checkpoint_callback.best_model_path:
        print("\n" + "="*80)
        print("Running Automatic Analysis")
        print("="*80)
        
        try:
            import subprocess
            
            # Save to src/evaluation/train/
            analysis_dir = Path("src") / "evaluation" / "train" / args.experiment_name
            analysis_dir.mkdir(parents=True, exist_ok=True)
            
            # Run analyze.py script
            analyze_cmd = [
                sys.executable,
                str(Path(__file__).parent / "analyze.py"),
                "--checkpoint", checkpoint_callback.best_model_path,
                "--data_dir", args.data_dir,
                "--output_dir", str(analysis_dir),
                "--split", "test",
            ]
            
            print(f"Running analysis: {' '.join(analyze_cmd)}")
            result = subprocess.run(analyze_cmd, capture_output=False, text=True)
            
            if result.returncode == 0:
                print(f"\nAnalysis complete! Plots saved to: {analysis_dir}")
            else:
                print(f"\nAnalysis failed with return code: {result.returncode}")
        
        except Exception as e:
            print(f"\nWarning: Automatic analysis failed: {e}")
            print("You can run analysis manually with:")
            print(f"  python scripts/analyze.py --checkpoint {checkpoint_callback.best_model_path} --data_dir {args.data_dir}")


if __name__ == "__main__":
    import pandas as pd  # Import here to avoid circular dependency
    import sys
    main()

