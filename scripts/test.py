"""Quantitative evaluation script for NeuralFactors model.

Computes paper metrics on test set:
- Negative Log-Likelihood (NLL_joint, NLL_ind)
- Covariance prediction (MSE)
- Value at Risk (VaR) calibration
- Portfolio backtest with benchmark comparison

Each metric is implemented in its own module under src/analysis/:
  loader.py     - model and data loading
  nll.py        - NLL metrics
  var.py        - VaR calibration
  covariance.py - covariance MSE
  portfolio.py  - backtest and portfolio optimization
  report.py     - summary report generation

Usage:
    # Debug mode (fast, first 50 dates)
    python scripts/test.py --checkpoint checkpoints/neuralfactors/last.ckpt --mode debug

    # Full evaluation (paper mode)
    python scripts/test.py --checkpoint checkpoints/neuralfactors/last.ckpt --mode paper
"""

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*torch.load.*')

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.analysis import (
    load_model_and_data,
    setup_output_dirs,
    compute_nll_metrics,
    save_nll_results,
    plot_nll_timeseries,
    compute_var_metrics,
    save_var_results,
    plot_var_calibration,
    compute_covariance_metrics,
    save_cov_results,
    plot_cov_metrics,
    compute_portfolio_metrics,
    plot_cumulative_returns,
    generate_summary_report,
)


def parse_args():
    parser = argparse.ArgumentParser(description="NeuralFactors Quantitative Evaluation")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Data directory')
    parser.add_argument('--output_dir', type=str, default='results/evaluation',
                        help='Output directory for results')
    parser.add_argument('--experiment_name', type=str, default='neuralfactors',
                        help='Experiment name for organizing outputs')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples for NLL computation (overridden by mode)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--mode', type=str, default='paper',
                        choices=['debug', 'paper'],
                        help='Evaluation mode: debug (fast, 50 dates) or paper (full)')
    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.mode == 'debug':
        num_samples_nll = 10
        num_samples_var = 100
        print("\n" + "=" * 80)
        print("RUNNING IN DEBUG MODE")
        print("=" * 80)
        print("- NLL samples: 10 | VaR samples: 100 | Max dates: 50")
    else:
        num_samples_nll = args.num_samples
        num_samples_var = 1000
        print("\n" + "=" * 80)
        print("RUNNING IN PAPER MODE (FULL EVALUATION)")
        print("=" * 80)
        print(f"- NLL samples: {num_samples_nll} | VaR samples: {num_samples_var} | Max dates: All")

    start_time = time.time()

    try:
        # 1. Load
        model, dataloader, dataset, returns_std, device = load_model_and_data(
            args.checkpoint, args.data_dir, args.split
        )
        output_dir = setup_output_dirs(args.output_dir, args.experiment_name)

        # 2. NLL
        t = time.time()
        nll_df = compute_nll_metrics(model, dataloader, dataset, num_samples_nll, args.mode, device)
        save_nll_results(nll_df, output_dir)
        plot_nll_timeseries(nll_df, output_dir)
        print(f"  Time elapsed: {time.time() - t:.1f}s")

        # 3. Covariance
        t = time.time()
        cov_df = compute_covariance_metrics(model, dataloader, dataset, args.mode, returns_std, device)
        if not cov_df.empty:
            save_cov_results(cov_df, output_dir)
            plot_cov_metrics(cov_df, output_dir)
        print(f"  Time elapsed: {time.time() - t:.1f}s")

        # 4. VaR
        t = time.time()
        var_df = compute_var_metrics(model, dataloader, dataset, num_samples_var, args.mode, returns_std, device)
        if not var_df.empty:
            save_var_results(var_df, output_dir)
            plot_var_calibration(var_df, output_dir)
        print(f"  Time elapsed: {time.time() - t:.1f}s")

        # 5. Portfolio backtest
        t = time.time()
        returns_df, backtest_metrics = compute_portfolio_metrics(
            model, dataset, returns_std, args.mode, device, output_dir
        )
        if not returns_df.empty:
            plot_cumulative_returns(returns_df, output_dir, Path(args.data_dir))
        print(f"  Time elapsed: {time.time() - t:.1f}s")

        # 6. Summary report
        generate_summary_report(output_dir, nll_df, cov_df, var_df, backtest_metrics)

        total = time.time() - start_time
        print("\n" + "=" * 80)
        print("EVALUATION COMPLETE")
        print("=" * 80)
        print(f"Total time: {total:.1f}s ({total / 60:.1f} minutes)")
        print(f"Results saved to: {output_dir}")

    except Exception as e:
        print("\n" + "=" * 80)
        print("ERROR DURING EVALUATION")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
