"""Cumulative-returns comparison chart across all evaluated models.

Automatically discovers every model that has already been evaluated and saved
backtest results, then plots their cumulative return series on a single chart
with distinct colours and a legend that includes the Sharpe ratio.

Discovery logic (applied unless ``--no_auto`` is given)
-------------------------------------------------------
1. ``<results_dir>/evaluation/*/timeseries/backtest_returns.csv``   — NeuralFactors runs
2. ``<results_dir>/ppca/*/metrics/portfolio_returns.csv``           — PPCA runs

In both cases the sub-directory name becomes the model alias (e.g.
``neuralfactors``, ``neuralfactors_78epochs``, ``ppca``).

The chart is saved to ``<output_dir>/plots/cumulative_returns_comparison.png``.

Usage examples
--------------
    # Auto-discover everything and generate the chart
    python results/plot_comparison.py

    # Manually specify models (auto-discovery still active unless --no_auto)
    python results/plot_comparison.py \\
        --results "NF v1:results/evaluation/neuralfactors" \\
        --results "PPCA:results/ppca/ppca"

    # Manually select a subset, disable auto-discovery
    python results/plot_comparison.py \\
        --results "NF v1:results/evaluation/neuralfactors" \\
        --no_auto

    # Custom root directories
    python results/plot_comparison.py \\
        --results_dir results \\
        --output_dir results/comparison \\
        --data_dir data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Sub-paths tried (in order) under a model's result directory
_RETURNS_CANDIDATES = [
    "timeseries/backtest_returns.csv",   # NeuralFactors
    "metrics/portfolio_returns.csv",     # PPCA
]

# Ibovespa benchmark path (relative to data_dir)
_IBOV_PATH = "cleaned/ibovespa.csv"


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_models(results_dir: Path) -> list[tuple[str, Path]]:
    """Scan *results_dir* and return every model directory that has results.

    Returns
    -------
    list of (alias, model_dir) tuples, sorted alphabetically by alias.
    """
    found: dict[str, Path] = {}

    # Pattern 1: NeuralFactors — results/evaluation/<name>/timeseries/backtest_returns.csv
    eval_root = results_dir / "evaluation"
    if eval_root.is_dir():
        for candidate in sorted(eval_root.iterdir()):
            if candidate.is_dir():
                csv = candidate / "timeseries" / "backtest_returns.csv"
                if csv.exists():
                    found[candidate.name] = candidate

    # Pattern 2: PPCA — results/ppca/<name>/metrics/portfolio_returns.csv
    ppca_root = results_dir / "ppca"
    if ppca_root.is_dir():
        for candidate in sorted(ppca_root.iterdir()):
            if candidate.is_dir():
                csv = candidate / "metrics" / "portfolio_returns.csv"
                if csv.exists():
                    alias = f"ppca_{candidate.name}" if candidate.name in found else candidate.name
                    found[alias] = candidate

    return sorted(found.items())


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_returns(model_dir: Path) -> pd.DataFrame | None:
    """Load backtest returns from a model directory.

    Tries several sub-paths in order; returns a DataFrame with columns
    ``[date, return]`` or ``None`` if no file is found.
    """
    for rel in _RETURNS_CANDIDATES:
        p = model_dir / rel
        if p.exists():
            df = pd.read_csv(p, parse_dates=["date"])
            df = df.sort_values("date").reset_index(drop=True)
            return df[["date", "return"]]
    return None


def load_ibovespa(data_dir: Path, start_date=None, end_date=None) -> pd.DataFrame | None:
    """Load Ibovespa benchmark daily returns.

    Returns a DataFrame with columns ``[date, return]``, or ``None`` if the
    file does not exist (placeholder — will be wired up when data is ready).
    """
    ibov_path = data_dir / _IBOV_PATH
    if not ibov_path.exists():
        return None
    try:
        df = pd.read_csv(ibov_path, sep=";", decimal=",", parse_dates=["date"])
        df = df.sort_values("date").reset_index(drop=True)
        if start_date is not None:
            df = df[df["date"] >= start_date]
        if end_date is not None:
            df = df[df["date"] <= end_date]
        # Cleaned CSV already contains daily returns in the 'return' column
        return df[["date", "return"]].dropna().reset_index(drop=True)
    except Exception as e:
        print(f"  Warning: could not load Ibovespa data — {e}")
        return None


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _compute_sharpe(returns: np.ndarray, ann: int = 252) -> float:
    arr = np.asarray(returns, dtype=float)
    if arr.std() == 0 or len(arr) < 2:
        return float("nan")
    ann_ret = float((1 + (1 + arr).prod() - 1) ** (ann / len(arr)) - 1)
    ann_vol = float(arr.std() * np.sqrt(ann))
    return ann_ret / ann_vol if ann_vol > 0 else float("nan")


def _compute_total_return(returns: np.ndarray) -> float:
    return float((1 + np.asarray(returns, dtype=float)).prod() - 1)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_cumulative_comparison(
    models_data: list[tuple[str, pd.DataFrame]],
    output_path: Path,
    ibovespa_df: pd.DataFrame | None = None,
) -> None:
    """Generate and save a cumulative-returns comparison chart.

    Parameters
    ----------
    models_data:
        List of ``(alias, returns_df)`` where *returns_df* has columns
        ``[date, return]``.
    output_path:
        Full path (including filename) where the PNG will be saved.
    ibovespa_df:
        Optional benchmark DataFrame with columns ``[date, return]``.
        When ``None`` (benchmark not yet available) the chart is generated
        without a benchmark line.
    """
    if not models_data:
        print("  No model data to plot.")
        return

    # Colour cycle — up to 10 distinct colours
    cmap = plt.get_cmap("tab10")
    colours = [cmap(i % 10) for i in range(len(models_data))]

    fig, ax = plt.subplots(figsize=(14, 7))

    all_dates = []

    for (alias, df), colour in zip(models_data, colours):
        arr = df["return"].values.astype(float)
        cum = (1 + arr).cumprod()
        sharpe = _compute_sharpe(arr)
        total_ret = _compute_total_return(arr)

        label = (
            f"{alias}  "
            f"(Total: {total_ret:+.1%}, "
            f"Sharpe: {sharpe:.2f})"
        )

        ax.plot(df["date"], cum, label=label, linewidth=1.8, alpha=0.9, color=colour)
        all_dates.extend(df["date"].tolist())

    # Ibovespa benchmark (placeholder — not plotted if df is None)
    if ibovespa_df is not None:
        ibov_arr = ibovespa_df["return"].values.astype(float)
        ibov_cum = (1 + ibov_arr).cumprod()
        ibov_total = _compute_total_return(ibov_arr)
        ibov_sharpe = _compute_sharpe(ibov_arr)
        ibov_label = (
            f"Ibovespa  "
            f"(Total: {ibov_total:+.1%}, "
            f"Sharpe: {ibov_sharpe:.2f})"
        )
        ax.plot(
            ibovespa_df["date"], ibov_cum,
            label=ibov_label,
            linewidth=2, alpha=0.9,
            color="#AAAAAA",
        )

    # Reference line at y=1
    ax.axhline(1.0, color="grey", linewidth=0.8, linestyle=":")

    # Formatting
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Cumulative Return (1 = initial investment)", fontsize=12)
    ax.set_title("Cumulative Returns — Model Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.2f}x"))

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Chart saved to: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Plot cumulative-returns comparison across all evaluated models. "
            "Auto-discovers every model directory that contains backtest results "
            "unless --no_auto is provided."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--results", action="append", metavar="ALIAS:DIR", default=[],
        help=(
            "Manually add a model in 'ALIAS:path/to/result/dir' form.\n"
            "Can be repeated. Combined with auto-discovery unless --no_auto is set."
        ),
    )
    p.add_argument(
        "--results_dir", type=str, default="results",
        help="Root directory scanned for auto-discovery (default: results).",
    )
    p.add_argument(
        "--output_dir", type=str, default="results/comparison",
        help="Directory where the plot PNG is saved (default: results/comparison).",
    )
    p.add_argument(
        "--data_dir", type=str, default="data",
        help="Root data directory, used for Ibovespa benchmark lookup (default: data).",
    )
    p.add_argument(
        "--no_auto", action="store_true",
        help="Disable auto-discovery; only use models specified via --results.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    results_root = Path(args.results_dir)
    output_dir   = Path(args.output_dir)
    data_dir     = Path(args.data_dir)

    # ------------------------------------------------------------------
    # 1. Build model list
    # ------------------------------------------------------------------
    models_map: dict[str, Path] = {}

    # Auto-discovery
    if not args.no_auto:
        discovered = discover_models(results_root)
        for alias, path in discovered:
            models_map[alias] = path

    # Manual overrides / additions
    for entry in args.results:
        if ":" not in entry:
            print(f"  ERROR: --results entry must be 'ALIAS:DIR', got: '{entry}'")
            return 1
        alias, dir_str = entry.split(":", 1)
        alias = alias.strip()
        path  = Path(dir_str.strip())
        if not path.exists():
            print(f"  WARNING: result directory not found — {path}")
        models_map[alias] = path  # manual entry overrides discovered one with same alias

    if not models_map:
        if args.no_auto:
            print("  ERROR: --no_auto set but no --results entries given. Nothing to plot.")
        else:
            print(f"  No evaluated models found under '{results_root}'. Run evaluation first.")
        return 1

    # ------------------------------------------------------------------
    # 2. Load returns
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("MODEL DISCOVERY & LOADING")
    print("=" * 70)

    models_data: list[tuple[str, pd.DataFrame]] = []
    for alias, model_dir in sorted(models_map.items()):
        df = load_returns(model_dir)
        if df is None or df.empty:
            print(f"  [{alias}]  WARNING: no backtest returns found in {model_dir}")
            continue
        sharpe = _compute_sharpe(df["return"].values)
        total  = _compute_total_return(df["return"].values)
        print(
            f"  [{alias}]  {len(df)} periods  "
            f"{df['date'].min().date()} → {df['date'].max().date()}  "
            f"Total: {total:+.2%}  Sharpe: {sharpe:.2f}"
        )
        models_data.append((alias, df))

    if not models_data:
        print("  No valid backtest data loaded. Aborting.")
        return 1

    # ------------------------------------------------------------------
    # 3. Ibovespa benchmark (placeholder)
    # ------------------------------------------------------------------
    all_dates = pd.concat([df["date"] for _, df in models_data])
    start_date = all_dates.min()
    end_date   = all_dates.max()

    ibovespa_df = load_ibovespa(data_dir, start_date, end_date)
    if ibovespa_df is None:
        print("\n  Ibovespa benchmark: not available — will be added when data is ready.")
    else:
        print(f"\n  Ibovespa benchmark loaded: {len(ibovespa_df)} periods")

    # ------------------------------------------------------------------
    # 4. Plot
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("GENERATING CHART")
    print("=" * 70)

    output_path = output_dir / "plots" / "cumulative_returns_comparison.png"
    plot_cumulative_comparison(models_data, output_path, ibovespa_df)

    # ------------------------------------------------------------------
    # 5. Summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)

    rows = []
    if ibovespa_df is not None:
        rows.append({
            "Model":        "Ibovespa (benchmark)",
            "Periods":      len(ibovespa_df),
            "Start":        str(ibovespa_df["date"].min().date()),
            "End":          str(ibovespa_df["date"].max().date()),
            "Total Return": f"{_compute_total_return(ibovespa_df['return'].values):+.2%}",
            "Sharpe":       f"{_compute_sharpe(ibovespa_df['return'].values):.2f}",
        })
    for alias, df in models_data:
        arr = df["return"].values
        rows.append({
            "Model":        alias,
            "Periods":      len(df),
            "Start":        str(df["date"].min().date()),
            "End":          str(df["date"].max().date()),
            "Total Return": f"{_compute_total_return(arr):+.2%}",
            "Sharpe":       f"{_compute_sharpe(arr):.2f}",
        })

    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))

    return 0


if __name__ == "__main__":
    exit(main())
