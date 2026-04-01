"""Value at Risk (VaR) calibration metrics for PPCA evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm

from PPCA import model as ppca_model


# =============================================================================
# Statistical tests
# =============================================================================

def kupiec_pof_test(violations: np.ndarray, p: float) -> dict:
    """Kupiec (1995) Proportion of Failures test.

    Tests H0: true violation rate = p.
    """
    T = len(violations)
    n = int(violations.sum())
    p_hat = n / T if T > 0 else 0.0

    if p_hat == 0 or p_hat == 1:
        return {"lr_statistic": float("nan"), "p_value": float("nan"), "reject_5pct": False}

    lr = 2 * (n * np.log(p_hat / p) + (T - n) * np.log((1 - p_hat) / (1 - p)))
    p_value = float(1 - stats.chi2.cdf(lr, df=1))
    return {"lr_statistic": float(lr), "p_value": p_value, "reject_5pct": p_value < 0.05}


def christoffersen_test(violations: np.ndarray) -> dict:
    """Christoffersen (1998) test for independence of violations."""
    T = len(violations)
    if T < 2:
        return {"lr_statistic": float("nan"), "p_value": float("nan"), "reject_5pct": False}

    n00 = n01 = n10 = n11 = 0
    for i in range(T - 1):
        v0, v1 = int(violations[i]), int(violations[i + 1])
        if v0 == 0 and v1 == 0:
            n00 += 1
        elif v0 == 0 and v1 == 1:
            n01 += 1
        elif v0 == 1 and v1 == 0:
            n10 += 1
        else:
            n11 += 1

    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi   = (n01 + n11) / (T - 1) if T > 1 else 0.0

    if pi == 0 or pi == 1 or pi01 == 0 or pi01 == 1:
        return {"lr_statistic": float("nan"), "p_value": float("nan"), "reject_5pct": False}
    if (n10 + n11) == 0 or pi11 == 0 or pi11 == 1:
        return {"lr_statistic": float("nan"), "p_value": float("nan"), "reject_5pct": False}

    log_l0 = (n00 + n10) * np.log(1 - pi) + (n01 + n11) * np.log(pi)
    log_l1 = 0.0
    if n00 > 0:
        log_l1 += n00 * np.log(1 - pi01)
    if n01 > 0:
        log_l1 += n01 * np.log(pi01)
    if n10 > 0:
        log_l1 += n10 * np.log(1 - pi11)
    if n11 > 0:
        log_l1 += n11 * np.log(pi11)

    lr = 2 * (log_l1 - log_l0)
    p_value = float(1 - stats.chi2.cdf(lr, df=1))
    return {"lr_statistic": float(lr), "p_value": p_value, "reject_5pct": p_value < 0.05}


def compute_var_metrics(
    returns_wide: pd.DataFrame,
    eval_indices: List[int],
    F: int,
    window_size: int,
    num_samples: int,
    mode: str,
    returns_std: float,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute VaR calibration: theoretical vs empirical violation rates.

    For each date at positional index ``idx``:
    1. Fit PPCA on rows ``[idx-window_size : idx]``.
    2. Draw ``num_samples`` return vectors from the fitted model.
    3. For each valid stock, compare the sampled quantile to the actual return.

    A "violation" occurs when the actual return falls below the theoretical
    quantile prediction — so the empirical violation rate at quantile q should
    be approximately q if the model is well-calibrated.

    Parameters
    ----------
    returns_wide : (T, N) normalised-returns DataFrame
    eval_indices : positional indices in the split
    F            : number of PPCA factors
    window_size  : look-back window for fitting
    num_samples  : number of samples drawn per date
    mode         : ``'debug'`` or ``'paper'``
    returns_std  : normalisation std (used to denormalise before comparison)
    seed         : random seed for reproducibility

    Returns
    -------
    pd.DataFrame  columns: [quantile, theoretical, empirical, error]
    """
    print("\n" + "=" * 80)
    print("COMPUTING VALUE AT RISK CALIBRATION")
    print("=" * 80)
    print(f"Number of samples per date: {num_samples}")

    quantiles = [0.01, 0.05, 0.10]
    max_dates = 50 if mode == "debug" else None
    if max_dates:
        print(f"Debug mode: Processing first {max_dates} dates")

    rng = np.random.default_rng(seed)
    indices = eval_indices[:max_dates] if max_dates else eval_indices

    all_predictions: list[np.ndarray] = []   # list of (n_valid, num_samples) arrays
    all_actuals: list[np.ndarray] = []       # list of (n_valid,) arrays

    for idx in tqdm(indices, desc="Computing VaR"):
        window = returns_wide.iloc[idx - window_size : idx]
        valid_tickers = window.columns[window.notna().all(axis=0)].tolist()
        if len(valid_tickers) < F + 2:
            continue

        today = returns_wide.iloc[idx][valid_tickers]
        if today.isna().any():
            continue

        R = window[valid_tickers].values.astype(np.float64)
        r_actual_norm = today.values.astype(np.float64)

        try:
            fitted = ppca_model.fit(R, F, valid_tickers)
        except Exception as e:
            print(f"\nWarning (fit) at {returns_wide.index[idx]}: {e}")
            continue

        # Sample: (num_samples, n_valid) in normalised space
        samples_norm = ppca_model.sample(fitted, num_samples, rng)      # (K, n)

        # Denormalise both samples and actuals to real return space
        samples_real = samples_norm * returns_std                        # (K, n)
        r_actual_real = r_actual_norm * returns_std                      # (n,)

        all_predictions.append(samples_real.T)   # (n, K)
        all_actuals.append(r_actual_real)        # (n,)

    if not all_predictions:
        print("\nWarning: No VaR metrics computed")
        return pd.DataFrame()

    predictions = np.concatenate(all_predictions, axis=0)   # (N_total, K)
    actuals = np.concatenate(all_actuals, axis=0)           # (N_total,)

    print(f"\n  Total stock-day observations: {len(actuals)}")
    results = []
    for q in quantiles:
        theoretical_q = np.quantile(predictions, q, axis=1)   # (N_total,)
        violation_flags = (actuals < theoretical_q).astype(int)
        violations = violation_flags.sum()
        empirical_prob = violations / len(actuals)
        error = abs(empirical_prob - q)
        quality = "Good" if error < 0.02 else ("OK" if error < 0.05 else "Poor")

        kupiec = kupiec_pof_test(violation_flags, q)
        christoff = christoffersen_test(violation_flags)

        print(f"  q={q:.2f}: empirical={empirical_prob:.4f}  error={error:.4f}  [{quality}]")
        print(f"        Kupiec LR={kupiec['lr_statistic']:.3f}, p={kupiec['p_value']:.4f}"
              f"  {'REJECT' if kupiec['reject_5pct'] else 'accept'}")
        print(f"        Christoffersen LR={christoff['lr_statistic']:.3f}, p={christoff['p_value']:.4f}"
              f"  {'REJECT' if christoff['reject_5pct'] else 'accept'}")

        results.append({
            "quantile":    q,
            "theoretical": q,
            "empirical":   empirical_prob,
            "error":       error,
            "kupiec_lr":   kupiec["lr_statistic"],
            "kupiec_p":    kupiec["p_value"],
            "kupiec_reject": kupiec["reject_5pct"],
            "christoffersen_lr": christoff["lr_statistic"],
            "christoffersen_p":  christoff["p_value"],
            "christoffersen_reject": christoff["reject_5pct"],
        })

    print(f"\n  VaR Calibration complete")
    return pd.DataFrame(results)


def save_var_results(var_df: pd.DataFrame, output_dir: Path) -> None:
    """Save VaR results to CSV."""
    output_path = output_dir / "metrics" / "var_calibration.csv"
    var_df.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")


def plot_var_calibration(var_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot theoretical vs empirical quantiles (calibration plot)."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(var_df["theoretical"], var_df["empirical"], s=120, zorder=5)

    lo = var_df["theoretical"].min()
    hi = var_df["theoretical"].max()
    ax.plot([lo, hi], [lo, hi], "k--", label="Perfect calibration")

    for _, row in var_df.iterrows():
        ax.annotate(
            f"q={row['quantile']:.2f}",
            (row["theoretical"], row["empirical"]),
            xytext=(6, 6), textcoords="offset points", fontsize=9,
        )

    ax.set_xlabel("Theoretical quantile")
    ax.set_ylabel("Empirical quantile")
    ax.set_title("PPCA — VaR Calibration")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = output_dir / "plots" / "var_calibration.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {output_path}")
