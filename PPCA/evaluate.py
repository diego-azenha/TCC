"""PPCA quantitative evaluation script.

Computes the same four metrics as scripts/test.py (NeuralFactors) — NLL,
covariance MSE, VaR calibration, portfolio backtest — using a rolling
closed-form PPCA fit (no training step required).

Usage
-----
    # Debug mode (first 50 evaluation dates, fast)
    python PPCA/evaluate.py --mode debug

    # Full paper-mode evaluation
    python PPCA/evaluate.py --mode paper

    # Custom split or factor count
    python PPCA/evaluate.py --split val --num_factors 10 --window_size 126
"""

import argparse
import time
from pathlib import Path
import sys

# Make project root importable regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from PPCA.loader import load_returns_wide, get_eval_indices, setup_output_dirs
from PPCA.analysis import (
    compute_nll_metrics,
    save_nll_results,
    plot_nll_timeseries,
    compute_covariance_metrics,
    save_cov_results,
    plot_cov_metrics,
    compute_var_metrics,
    save_var_results,
    plot_var_calibration,
    compute_portfolio_metrics,
    plot_cumulative_returns,
    generate_summary_report,
)


def parse_args():
    p = argparse.ArgumentParser(description="PPCA Quantitative Evaluation")
    p.add_argument("--data_dir",         type=str, default="data",
                   help="Root data directory")
    p.add_argument("--output_dir",       type=str, default="results/ppca",
                   help="Parent directory for output folders")
    p.add_argument("--experiment_name",  type=str, default="ppca",
                   help="Sub-folder name inside output_dir")
    p.add_argument("--split",            type=str, default="test",
                   choices=["train", "val", "test"],
                   help="Dataset split to evaluate")
    p.add_argument("--num_factors",      type=int, default=12,
                   help="Number of PPCA latent factors F")
    p.add_argument("--window_size",      type=int, default=252,
                   help="Rolling look-back window for fitting (trading days)")
    p.add_argument("--num_samples",      type=int, default=100,
                   help="Samples for VaR / NLL (overridden by --mode)")
    p.add_argument("--mode",             type=str, default="paper",
                   choices=["debug", "paper"],
                   help="'debug': first 50 dates, fast; 'paper': full evaluation")
    p.add_argument("--train_end",        type=str, default="2018-12-31",
                   help="Last date of training split (ISO)")
    p.add_argument("--val_end",          type=str, default="2022-12-31",
                   help="Last date of validation split (ISO)")
    p.add_argument("--seed",             type=int, default=42,
                   help="Random seed for reproducibility")
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    if args.mode == "debug":
        num_samples_var = 100
        print("\n" + "=" * 80)
        print("RUNNING IN DEBUG MODE")
        print("=" * 80)
        print(f"  VaR samples: {num_samples_var}  |  Max dates: 50")
    else:
        num_samples_var = max(args.num_samples, 1000)
        print("\n" + "=" * 80)
        print("RUNNING IN PAPER MODE (FULL EVALUATION)")
        print("=" * 80)
        print(f"  VaR samples: {num_samples_var}  |  Max dates: All")

    print(f"  F = {args.num_factors}  |  Window = {args.window_size} days  |  Split = {args.split}")
    total_start = time.time()

    try:
        # ------------------------------------------------------------------
        # 1. Load data
        # ------------------------------------------------------------------
        print("\n" + "=" * 80)
        print("LOADING DATA")
        print("=" * 80)
        returns_wide, returns_std = load_returns_wide(
            args.data_dir, args.train_end, args.val_end
        )
        print(f"  Returns shape: {returns_wide.shape}  (dates × tickers)")
        print(f"  Returns std (train): {returns_std:.6f}")
        print(f"  Date range: {returns_wide.index[0].date()} to {returns_wide.index[-1].date()}")

        eval_indices = get_eval_indices(
            returns_wide, args.split,
            args.train_end, args.val_end, args.window_size,
        )
        print(f"  Evaluation dates ({args.split}): {len(eval_indices)}")
        if not eval_indices:
            print("  ERROR: No evaluation indices found. Check split/window_size/data.")
            return 1

        output_dir = setup_output_dirs(args.output_dir, args.experiment_name)

        # ------------------------------------------------------------------
        # 2. NLL
        # ------------------------------------------------------------------
        t = time.time()
        nll_df = compute_nll_metrics(
            returns_wide, eval_indices, args.num_factors,
            args.window_size, args.mode, returns_std,
        )
        if not nll_df.empty:
            save_nll_results(nll_df, output_dir)
            plot_nll_timeseries(nll_df, output_dir)
        print(f"  Time elapsed: {time.time() - t:.1f}s")

        # ------------------------------------------------------------------
        # 3. Covariance
        # ------------------------------------------------------------------
        t = time.time()
        cov_df = compute_covariance_metrics(
            returns_wide, eval_indices, args.num_factors,
            args.window_size, args.mode, returns_std,
        )
        if not cov_df.empty:
            save_cov_results(cov_df, output_dir)
            plot_cov_metrics(cov_df, output_dir)
        print(f"  Time elapsed: {time.time() - t:.1f}s")

        # ------------------------------------------------------------------
        # 4. VaR
        # ------------------------------------------------------------------
        t = time.time()
        var_df = compute_var_metrics(
            returns_wide, eval_indices, args.num_factors,
            args.window_size, num_samples_var, args.mode, returns_std, args.seed,
        )
        if not var_df.empty:
            save_var_results(var_df, output_dir)
            plot_var_calibration(var_df, output_dir)
        print(f"  Time elapsed: {time.time() - t:.1f}s")

        # ------------------------------------------------------------------
        # 5. Portfolio backtest
        # ------------------------------------------------------------------
        t = time.time()
        returns_df, backtest_metrics = compute_portfolio_metrics(
            returns_wide, eval_indices, args.num_factors,
            args.window_size, args.mode, returns_std, output_dir,
            data_dir=args.data_dir,
        )
        if not returns_df.empty:
            plot_cumulative_returns(
                returns_df, output_dir,
                Path(args.data_dir),
            )
        print(f"  Time elapsed: {time.time() - t:.1f}s")

        # ------------------------------------------------------------------
        # 6. Summary report
        # ------------------------------------------------------------------
        generate_summary_report(output_dir, nll_df, cov_df, var_df, backtest_metrics)

        total = time.time() - total_start
        print("\n" + "=" * 80)
        print("EVALUATION COMPLETE")
        print("=" * 80)
        print(f"Total time: {total:.1f}s ({total / 60:.1f} min)")
        print(f"Results saved to: {output_dir}")

    except Exception:
        print("\n" + "=" * 80)
        print("ERROR DURING EVALUATION")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
