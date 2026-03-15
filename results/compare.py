"""Side-by-side model comparison script.

Reads the standardised CSV outputs produced by each model's evaluation script
and assembles comparison tables (NLL, covariance MSE, VaR, portfolio).

Each model's result directory must contain:
  metrics/nll_timeseries.csv      — columns: date, nll_joint, nll_ind, n_stocks
  metrics/covariance_results.csv  — columns: date, mse_cov, n_stocks
  metrics/var_calibration.csv     — columns: quantile, theoretical, empirical, error
  metrics/portfolio_returns.csv   — columns: date, return

Tables are saved to ``--output_dir`` as CSV and printed to stdout.

Usage
-----
    python results/compare.py \\
        --results "NeuralFactors:results/evaluation/neuralfactors" \\
        --results "PPCA:results/ppca/ppca"

    # Adding a third model later requires no code changes:
    python results/compare.py \\
        --results "NeuralFactors:results/evaluation/neuralfactors" \\
        --results "PPCA:results/ppca/ppca" \\
        --results "FactorAnalysis:results/fa/fa"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# CSV loading helpers
# =============================================================================

def _load(path: Path, filename: str, fallbacks: list[str] | None = None) -> pd.DataFrame:
    candidates = [path / "metrics" / filename]
    for fb in (fallbacks or []):
        candidates.append(path / fb)
    for f in candidates:
        if f.exists():
            return pd.read_csv(f)
    return pd.DataFrame()


def _fmt_pct(v):
    return f"{v:.2%}" if pd.notna(v) else "—"

def _fmt(v, decimals=4):
    return f"{v:.{decimals}f}" if pd.notna(v) else "—"


# =============================================================================
# Metric aggregators
# =============================================================================

def _nll_stats(df: pd.DataFrame) -> dict:
    if df.empty or "nll_joint" not in df.columns:
        return {}
    return {
        "NLL_joint_mean": df["nll_joint"].mean(),
        "NLL_joint_std":  df["nll_joint"].std(),
        "NLL_ind_mean":   df["nll_ind"].mean(),
        "NLL_ind_std":    df["nll_ind"].std(),
        "dates":          len(df),
    }


def _cov_stats(df: pd.DataFrame) -> dict:
    if df.empty or "mse_cov" not in df.columns:
        return {}
    return {
        "CovMSE_mean": df["mse_cov"].mean(),
        "CovMSE_std":  df["mse_cov"].std(),
        "dates":       len(df),
    }


def _var_stats(df: pd.DataFrame) -> dict:
    if df.empty or "error" not in df.columns:
        return {}
    result = {}
    for _, row in df.iterrows():
        q = row["quantile"]
        result[f"q{int(q*100):02d}pct_error"] = row["error"]
        result[f"q{int(q*100):02d}pct_empirical"] = row["empirical"]
    return result


def _portfolio_stats(df: pd.DataFrame) -> dict:
    if df.empty or "return" not in df.columns:
        return {}
    arr = df["return"].values
    ann = 252
    total   = float((1 + arr).prod() - 1)
    ann_ret = float((1 + total) ** (ann / len(arr)) - 1)
    ann_vol = float(arr.std() * np.sqrt(ann))
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan

    cumulative  = (1 + arr).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdown    = (cumulative - running_max) / running_max
    max_dd      = float(drawdown.min())

    return {
        "TotalReturn":      total,
        "AnnReturn":        ann_ret,
        "AnnVol":           ann_vol,
        "Sharpe":           sharpe,
        "MaxDrawdown":      max_dd,
        "periods":          len(arr),
    }


# =============================================================================
# Table printing
# =============================================================================

def _print_table(title: str, df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    if df.empty:
        print("  No data")
        return
    print(df.to_string(index=False))


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare evaluation metrics across models."
    )
    parser.add_argument(
        "--results", action="append", metavar="ALIAS:DIR", required=True,
        help=(
            "Model result entry in the form 'ALIAS:path/to/result/dir'. "
            "Repeat this flag for each model."
        ),
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/comparison",
        help="Directory where comparison CSVs are saved.",
    )
    args = parser.parse_args()

    # Parse --results entries
    models: list[tuple[str, Path]] = []
    for entry in args.results:
        if ":" not in entry:
            parser.error(f"--results must be in 'ALIAS:DIR' format, got: '{entry}'")
        alias, dir_str = entry.split(":", 1)
        models.append((alias.strip(), Path(dir_str.strip())))

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    for alias, path in models:
        status = "(found)" if path.exists() else "(NOT FOUND)"
        print(f"  {alias:<25} {path}  {status}")

    # ------------------------------------------------------------------
    # Collect per-model metric summaries
    # ------------------------------------------------------------------
    nll_rows, cov_rows, var_rows, port_rows, summary_rows = [], [], [], [], []

    for alias, path in models:
        nll_df  = _load(path, "nll_timeseries.csv")
        cov_df  = _load(path, "covariance_results.csv", fallbacks=["metrics/cov_metrics.csv"])
        var_df  = _load(path, "var_calibration.csv")
        port_df = _load(path, "portfolio_returns.csv", fallbacks=["timeseries/backtest_returns.csv"])

        nll_s  = _nll_stats(nll_df)
        cov_s  = _cov_stats(cov_df)
        var_s  = _var_stats(var_df)
        port_s = _portfolio_stats(port_df)

        nll_rows.append({"Model": alias, **nll_s})
        cov_rows.append({"Model": alias, **cov_s})
        var_rows.append({"Model": alias, **var_s})
        port_rows.append({"Model": alias, **port_s})
        summary_rows.append({"Model": alias, **nll_s, **cov_s, **var_s, **port_s})

    # ------------------------------------------------------------------
    # Build DataFrames and print
    # ------------------------------------------------------------------
    nll_out  = pd.DataFrame(nll_rows)
    cov_out  = pd.DataFrame(cov_rows)
    var_out  = pd.DataFrame(var_rows)
    port_out = pd.DataFrame(port_rows)
    sum_out  = pd.DataFrame(summary_rows)

    _print_table("1. NEGATIVE LOG-LIKELIHOOD", nll_out)
    _print_table("2. COVARIANCE MSE", cov_out)
    _print_table("3. VALUE AT RISK CALIBRATION", var_out)
    _print_table("4. PORTFOLIO BACKTEST", port_out)

    # ------------------------------------------------------------------
    # Formatted summary table (mirrors paper Table 2 style)
    # ------------------------------------------------------------------
    fmt_rows = []
    for alias, path in models:
        nll_df  = _load(path, "nll_timeseries.csv")
        cov_df  = _load(path, "covariance_results.csv", fallbacks=["metrics/cov_metrics.csv"])
        var_df  = _load(path, "var_calibration.csv")
        port_df = _load(path, "portfolio_returns.csv", fallbacks=["timeseries/backtest_returns.csv"])
        nll_s   = _nll_stats(nll_df)
        cov_s   = _cov_stats(cov_df)
        var_s   = _var_stats(var_df)
        port_s  = _portfolio_stats(port_df)

        fmt_rows.append({
            "Model":          alias,
            "NLL_joint":      _fmt(nll_s.get("NLL_joint_mean")),
            "NLL_ind":        _fmt(nll_s.get("NLL_ind_mean")),
            "CovMSE":         _fmt(cov_s.get("CovMSE_mean"), 6),
            "VaR_q01_err":    _fmt(var_s.get("q01pct_error")),
            "VaR_q05_err":    _fmt(var_s.get("q05pct_error")),
            "VaR_q10_err":    _fmt(var_s.get("q10pct_error")),
            "Sharpe":         _fmt(port_s.get("Sharpe"), 2),
            "Ann.Ret":        _fmt_pct(port_s.get("AnnReturn")),
            "MaxDrawdown":    _fmt_pct(port_s.get("MaxDrawdown")),
        })

    fmt_df = pd.DataFrame(fmt_rows)
    _print_table("SUMMARY (paper-style, lower NLL/MSE/VaR-err = better)", fmt_df)

    # ------------------------------------------------------------------
    # Save CSVs
    # ------------------------------------------------------------------
    nll_out.to_csv(out / "comparison_nll.csv",       index=False)
    cov_out.to_csv(out / "comparison_cov.csv",       index=False)
    var_out.to_csv(out / "comparison_var.csv",       index=False)
    port_out.to_csv(out / "comparison_portfolio.csv", index=False)
    sum_out.to_csv(out / "comparison_summary.csv",   index=False)
    fmt_df.to_csv(out  / "comparison_formatted.csv", index=False)

    print("\n" + "=" * 80)
    print(f"Results saved to: {out}")
    print("=" * 80)


if __name__ == "__main__":
    main()
