"""Value at Risk (VaR) calibration metrics for NeuralFactors evaluation."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

import src.models.decoder as dec
from scipy import stats


# =============================================================================
# Statistical tests
# =============================================================================

def kupiec_pof_test(violations: np.ndarray, p: float) -> dict:
    """Kupiec (1995) Proportion of Failures test.

    Tests H0: true violation rate = p.

    Parameters
    ----------
    violations : binary array (1 = violation, 0 = no violation)
    p : theoretical quantile level

    Returns
    -------
    dict with lr_statistic, p_value, reject_5pct
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
    """Christoffersen (1998) test for independence of violations.

    Tests H0: violations are independent (no clustering).

    Parameters
    ----------
    violations : binary array (1 = violation, 0 = no violation)

    Returns
    -------
    dict with lr_statistic, p_value, reject_5pct
    """
    T = len(violations)
    if T < 2:
        return {"lr_statistic": float("nan"), "p_value": float("nan"), "reject_5pct": False}

    # Transition counts
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

    # Transition probabilities
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


def compute_var_metrics(model, dataloader, dataset, num_samples, mode, returns_std, device):
    """Compute VaR calibration metrics.

    Args:
        model: Trained model
        dataloader: DataLoader
        dataset: Dataset for dates
        num_samples: Number of samples for predictions
        mode: 'debug' or 'paper'
        returns_std: Returns std for denormalization
        device: torch device

    Returns:
        pd.DataFrame: [quantile, theoretical, empirical, error]
    """
    print("\n" + "=" * 80)
    print("COMPUTING VALUE AT RISK CALIBRATION")
    print("=" * 80)
    print(f"Number of samples: {num_samples}")

    quantiles = [0.01, 0.05, 0.10]
    max_dates = 50 if mode == 'debug' else None
    if max_dates:
        print(f"Debug mode: Processing first {max_dates} dates")

    all_predictions = []
    all_actuals = []

    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, desc="Computing VaR")):
            if max_dates and idx >= max_dates:
                break

            S, S_static, r, mask = batch
            S = S.to(device)
            S_static = S_static.to(device)
            r = r.to(device)
            mask = mask.to(device)

            S_sq = S.squeeze(0)
            S_static_sq = S_static.squeeze(0)

            alpha, B, sigma, nu = model.model.embedder(S_sq, S_static_sq)

            z = model.model.prior.sample(batch_size=1, num_samples=num_samples, device=S.device)

            alpha_b = alpha.unsqueeze(0)
            B_b = B.unsqueeze(0)
            sigma_b = sigma.unsqueeze(0)
            nu_b = nu.unsqueeze(0)

            r_samples = dec.sample_r_given_z(alpha_b, B_b, sigma_b, nu_b, z)  # [1, N, K]
            r_samples = r_samples[0].cpu().numpy() * returns_std          # [N, K]

            r_actual = r[0].cpu().numpy() * returns_std
            mask_np = mask[0].cpu().numpy().astype(bool)

            all_predictions.append(r_samples[mask_np])
            all_actuals.append(r_actual[mask_np])

    predictions = np.concatenate(all_predictions, axis=0)  # [N_total, K]
    actuals = np.concatenate(all_actuals)                  # [N_total]

    results = []
    for q in quantiles:
        theoretical_q = np.quantile(predictions, q, axis=1)
        violation_flags = (actuals < theoretical_q).astype(int)
        violations = violation_flags.sum()
        empirical_prob = violations / len(actuals)
        error = abs(empirical_prob - q)
        quality = 'Good' if error < 0.02 else ('OK' if error < 0.05 else 'Poor')

        kupiec = kupiec_pof_test(violation_flags, q)
        christoff = christoffersen_test(violation_flags)

        print(f"  {q:.2f}: empirical={empirical_prob:.4f}, error={error:.4f} [{quality}]")
        print(f"        Kupiec LR={kupiec['lr_statistic']:.3f}, p={kupiec['p_value']:.4f}"
              f"  {'REJECT' if kupiec['reject_5pct'] else 'accept'}")
        print(f"        Christoffersen LR={christoff['lr_statistic']:.3f}, p={christoff['p_value']:.4f}"
              f"  {'REJECT' if christoff['reject_5pct'] else 'accept'}")

        results.append({
            'quantile': q,
            'theoretical': q,
            'empirical': empirical_prob,
            'error': error,
            'kupiec_lr': kupiec['lr_statistic'],
            'kupiec_p': kupiec['p_value'],
            'kupiec_reject': kupiec['reject_5pct'],
            'christoffersen_lr': christoff['lr_statistic'],
            'christoffersen_p': christoff['p_value'],
            'christoffersen_reject': christoff['reject_5pct'],
        })

    df = pd.DataFrame(results)
    print(f"\n✓ VaR Calibration Complete")
    return df


def save_var_results(var_df, output_dir):
    """Save VaR results to CSV."""
    output_path = output_dir / "metrics" / "var_calibration.csv"
    var_df.to_csv(output_path, index=False)
    print(f"✓ VaR results saved to: {output_path}")


def plot_var_calibration(var_df, output_dir):
    """Plot theoretical vs empirical quantiles (calibration plot)."""
    plt.figure(figsize=(8, 8))
    plt.scatter(var_df['theoretical'], var_df['empirical'], s=100, alpha=0.7)

    min_val = var_df['theoretical'].min()
    max_val = var_df['theoretical'].max()
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect Calibration')

    for _, row in var_df.iterrows():
        plt.annotate(f"{row['quantile']:.2f}",
                     (row['theoretical'], row['empirical']),
                     xytext=(5, 5), textcoords='offset points')

    plt.xlabel('Theoretical Quantile')
    plt.ylabel('Empirical Quantile')
    plt.title('VaR Calibration')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / "plots" / "var_calibration.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ VaR plot saved to: {output_path}")
