"""Full pipeline: training followed by evaluation.

Runs train.py then test.py in sequence, wiring the best checkpoint from training
directly into the evaluation step.  All training and evaluation flags are exposed
as CLI arguments so any experiment can be launched with a single command.

Usage examples
--------------
# Standard collapse-fix run (prior LR + free bits, 200k steps, then full evaluation)
python scripts/run.py --experiment_name collapse_fix \\
    --prior_lr_scale 0.1 --free_bits_lambda 0.2

# Quick 20k probe with debug evaluation
python scripts/run.py --experiment_name probe \\
    --prior_lr_scale 0.1 --free_bits_lambda 0.2 \\
    --max_steps 20000 --eval_mode debug

# Skip training, only re-run evaluation on an existing checkpoint
python scripts/run.py --experiment_name collapse_fix --skip_train

# Skip evaluation, only train
python scripts/run.py --experiment_name collapse_fix --skip_eval
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="Full NeuralFactors pipeline: train → evaluate",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Shared ────────────────────────────────────────────────────────────────
    p.add_argument("--experiment_name", type=str, default="neuralfactors",
                   help="Name used for checkpoint and results subdirectories")
    p.add_argument("--data_dir", type=str, default="data",
                   help="Directory containing parquet files")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                   help="Root checkpoint directory")
    p.add_argument("--log_dir", type=str, default="logs",
                   help="TensorBoard log directory")
    p.add_argument("--results_dir", type=str, default="results/evaluation",
                   help="Root results directory for evaluation outputs")
    p.add_argument("--seed", type=int, default=42, help="Random seed")

    # ── Pipeline control ──────────────────────────────────────────────────────
    p.add_argument("--skip_train", action="store_true",
                   help="Skip training; use an existing checkpoint from checkpoint_dir/experiment_name")
    p.add_argument("--skip_eval", action="store_true",
                   help="Skip evaluation after training")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Explicit checkpoint path for evaluation (overrides auto-detection; "
                        "implies --skip_train unless --skip_train is also passed)")

    # ── Training hyperparameters ──────────────────────────────────────────────
    train = p.add_argument_group("Training")
    train.add_argument("--num_factors", type=int, default=16)
    train.add_argument("--hidden_size", type=int, default=256)
    train.add_argument("--lookback", type=int, default=256)
    train.add_argument("--dropout", type=float, default=0.25)
    train.add_argument("--sigma_min", type=float, default=0.1,
                       help="Sigma lower bound via sigmoid (normalised space)")
    train.add_argument("--sigma_max", type=float, default=3.0,
                       help="Sigma upper bound via sigmoid (normalised space)")
    train.add_argument("--alpha_max", type=float, default=3.0,
                       help="Alpha clamp bound in normalised space")
    train.add_argument("--alpha_scale", type=float, default=0.0,
                       help="If > 0, use alpha_scale * tanh(alpha_head) instead of hard clamp; e.g. 0.1")
    train.add_argument("--learning_rate", type=float, default=1e-4)
    train.add_argument("--weight_decay", type=float, default=1e-6)
    train.add_argument("--max_steps", type=int, default=250_000)
    train.add_argument("--val_every_n_steps", type=int, default=100_000)
    train.add_argument("--polyak_start_step", type=int, default=None,
                       help="Polyak start step (default: max_steps // 2)")
    train.add_argument("--polyak_alpha", type=float, default=0.999)
    train.add_argument("--free_bits_lambda", type=float, default=0.1,
                       help="Min KL floor per factor in nats; 0 = disabled")
    train.add_argument("--lambda_sigma", type=float, default=1.0,
                       help="Weight on L_sigma calibration loss (detached sigma)")
    train.add_argument("--sigma_ref_ema", type=float, default=0.99,
                       help="EMA momentum for sigma_ref buffer")
    train.add_argument("--alpha_freeze_steps", type=int, default=None,
                       help="Freeze alpha_head for N steps (default: max_steps // 2; 0 = disabled)")
    train.add_argument("--kl_warmup_steps", type=int, default=0,
                       help="KL annealing warm-up: ramp kl_weight 0→1 over N steps (0 = disabled)")
    train.add_argument("--train_end", type=str, default="2018-12-31")
    train.add_argument("--val_end", type=str, default="2022-12-31")
    train.add_argument("--gpus", type=int, default=1)
    train.add_argument("--fast_dev_run", action="store_true",
                       help="Fast dev run (1 batch train+val); also sets eval_mode=debug")

    # ── Evaluation hyperparameters ────────────────────────────────────────────
    eval_group = p.add_argument_group("Evaluation")
    eval_group.add_argument("--eval_mode", type=str, default="paper",
                            choices=["debug", "paper"],
                            help="debug = fast 50-date pass; paper = full test set")
    eval_group.add_argument("--eval_split", type=str, default="test",
                            choices=["train", "val", "test"])
    eval_group.add_argument("--num_samples", type=int, default=100,
                            help='NLL importance samples (paper mode)')
    eval_group.add_argument('--run_ppca', action='store_true', default=False,
                            help='Run PPCA baseline and cross-model comparison after evaluation (off by default)')
    return p.parse_args()


def find_best_checkpoint(checkpoint_dir: Path) -> Path:
    """Return the Polyak model if present, else the last checkpoint in the dir."""
    polyak = checkpoint_dir / "polyak_model.pt"
    if polyak.exists():
        return polyak

    # Look for step-tagged checkpoints first (Lightning saves these)
    ckpts = sorted(checkpoint_dir.glob("**/*.ckpt"))
    if ckpts:
        # Prefer the one with highest step number
        def _step(p):
            for part in p.parts:
                if "step=" in part:
                    try:
                        return int(part.split("step=")[1].split("-")[0])
                    except ValueError:
                        pass
            return 0
        return max(ckpts, key=_step)

    raise FileNotFoundError(
        f"No checkpoint found in {checkpoint_dir}. "
        "Run without --skip_train to train a model first."
    )


def run_training(args, python: str) -> Path:
    """Invoke train.py and return the checkpoint path for evaluation."""
    script = Path(__file__).parent / "train.py"
    cmd = [
        python, str(script),
        "--experiment_name", args.experiment_name,
        "--data_dir", args.data_dir,
        "--checkpoint_dir", args.checkpoint_dir,
        "--log_dir", args.log_dir,
        "--seed", str(args.seed),
        "--num_factors", str(args.num_factors),
        "--hidden_size", str(args.hidden_size),
        "--lookback", str(args.lookback),
        "--dropout", str(args.dropout),
        "--learning_rate", str(args.learning_rate),
        "--weight_decay", str(args.weight_decay),
        "--max_steps", str(args.max_steps),
        "--val_every_n_steps", str(args.val_every_n_steps),
        "--polyak_alpha", str(args.polyak_alpha),
        "--free_bits_lambda", str(args.free_bits_lambda),
        "--lambda_sigma", str(args.lambda_sigma),
        "--sigma_ref_ema", str(args.sigma_ref_ema),
        "--sigma_min", str(args.sigma_min),
        "--sigma_max", str(args.sigma_max),
        "--alpha_max", str(args.alpha_max),
        "--alpha_scale", str(args.alpha_scale),
        "--train_end", args.train_end,
        "--val_end", args.val_end,
        "--gpus", str(args.gpus),
    ]
    if args.polyak_start_step is not None:
        cmd += ["--polyak_start_step", str(args.polyak_start_step)]
    if args.alpha_freeze_steps is not None:
        cmd += ["--alpha_freeze_steps", str(args.alpha_freeze_steps)]
    if args.kl_warmup_steps > 0:
        cmd += ["--kl_warmup_steps", str(args.kl_warmup_steps)]
    if args.fast_dev_run:
        cmd.append("--fast_dev_run")

    print("\n" + "=" * 80)
    print("PHASE 1 — TRAINING")
    print("=" * 80)
    print("Command:", " ".join(cmd))
    print()

    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\nTraining failed (exit code {result.returncode}). Aborting pipeline.")
        sys.exit(result.returncode)

    print(f"\nTraining completed in {elapsed/3600:.2f}h")

    ckpt_dir = Path(args.checkpoint_dir) / args.experiment_name
    return find_best_checkpoint(ckpt_dir)


def run_evaluation(args, checkpoint: Path, python: str):
    """Invoke test.py on the given checkpoint."""
    script = Path(__file__).parent / "test.py"
    eval_mode = "debug" if args.fast_dev_run else args.eval_mode
    cmd = [
        python, str(script),
        "--checkpoint", str(checkpoint),
        "--data_dir", args.data_dir,
        "--output_dir", args.results_dir,
        "--experiment_name", args.experiment_name,
        "--split", args.eval_split,
        "--mode", eval_mode,
        "--num_samples", str(args.num_samples),
        "--seed", str(args.seed),
    ]
    if args.run_ppca:
        cmd.append("--run_ppca")

    print("\n" + "=" * 80)
    print("PHASE 2 — EVALUATION")
    print("=" * 80)
    print(f"Checkpoint : {checkpoint}")
    print(f"Mode       : {eval_mode}")
    print("Command    :", " ".join(cmd))
    print()

    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\nEvaluation failed (exit code {result.returncode}).")
        sys.exit(result.returncode)

    print(f"\nEvaluation completed in {elapsed/60:.1f}min")


def main():
    args = parse_args()
    python = sys.executable  # same interpreter that launched this script

    # ── Resolve checkpoint ────────────────────────────────────────────────────
    if args.checkpoint:
        # Explicit path provided: skip training unless user also passed --skip_train=False
        checkpoint = Path(args.checkpoint)
        if not checkpoint.exists():
            print(f"Error: specified checkpoint does not exist: {checkpoint}")
            sys.exit(1)
        skip_train = True
    else:
        skip_train = args.skip_train
        checkpoint = None

    # ── Training ──────────────────────────────────────────────────────────────
    if not skip_train:
        checkpoint = run_training(args, python)
    else:
        if checkpoint is None:
            # Auto-detect from checkpoint_dir/experiment_name
            ckpt_dir = Path(args.checkpoint_dir) / args.experiment_name
            checkpoint = find_best_checkpoint(ckpt_dir)
        print(f"\nSkipping training. Using checkpoint: {checkpoint}")

    # ── Evaluation ────────────────────────────────────────────────────────────
    if not args.skip_eval:
        run_evaluation(args, checkpoint, python)
    else:
        print("\nSkipping evaluation (--skip_eval).")

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print(f"Experiment : {args.experiment_name}")
    print(f"Checkpoint : {checkpoint}")
    if not args.skip_eval:
        results_path = Path(args.results_dir) / args.experiment_name
        print(f"Results    : {results_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
