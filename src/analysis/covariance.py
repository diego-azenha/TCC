"""Covariance prediction metrics for NeuralFactors evaluation."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

import src.models.decoder as decoder


def compute_covariance_metrics(model, dataloader, dataset, mode, returns_std, device):
    """Compare predicted vs empirical rolling covariance.

    Args:
        model: Trained model
        dataloader: DataLoader
        dataset: Dataset for dates
        mode: 'debug' or 'paper'
        returns_std: Returns std for denormalization
        device: torch device

    Returns:
        pd.DataFrame: [date, mse_cov, n_stocks]
    """
    print("\n" + "=" * 80)
    print("COMPUTING COVARIANCE METRICS")
    print("=" * 80)

    window_size = 20
    max_dates = 50 if mode == 'debug' else None
    print(f"Rolling window size: {window_size} days")
    if max_dates:
        print(f"Debug mode: Processing first {max_dates} dates")

    results = []
    returns_history = []
    model.eval()

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, desc="Computing Covariance")):
            if max_dates and idx >= max_dates:
                break

            S, S_static, r, mask = batch
            S = S.to(device)
            S_static = S_static.to(device)
            r = r.to(device)
            mask = mask.to(device)

            r_np = r[0].cpu().numpy() * returns_std
            mask_np = mask[0].cpu().numpy().astype(bool)
            returns_history.append((r_np, mask_np))

            if len(returns_history) <= window_size:
                continue

            # Predicted covariance
            alpha, B, sigma, nu, mu_q, L_q = model.model.encode(S, S_static, r, mask)
            mu_z, Sigma_z = model.model.prior.to_normal_params()
            cov_pred = decoder.marginal_covariance(B[0], Sigma_z, sigma[0])
            if cov_pred.dim() == 3:
                cov_pred = cov_pred[0]
            cov_pred = cov_pred.cpu().numpy() * (returns_std ** 2)

            # Empirical covariance from rolling window
            valid_now = mask_np
            n_stocks = valid_now.sum()
            if n_stocks < 2:
                continue

            returns_matrix = []
            for r_hist, mask_hist in returns_history[-window_size:]:
                valid_both = valid_now & mask_hist
                r_valid = np.where(valid_both, r_hist, np.nan)
                returns_matrix.append(r_valid[valid_now])

            returns_matrix = np.array(returns_matrix)
            valid_cols = ~np.isnan(returns_matrix).any(axis=0)
            if valid_cols.sum() < 2:
                continue

            returns_matrix = returns_matrix[:, valid_cols]

            try:
                cov_emp = np.cov(returns_matrix, rowvar=False)
                all_valid_idx = np.where(valid_now)[0]
                final_idx = all_valid_idx[valid_cols]
                cov_pred_sub = cov_pred[np.ix_(final_idx, final_idx)]
                mse = np.mean((cov_pred_sub - cov_emp) ** 2)

                results.append({
                    'date': dataset.dates[idx],
                    'mse_cov': mse,
                    'n_stocks': returns_matrix.shape[1],
                })
            except Exception as e:
                print(f"\nWarning at {dataset.dates[idx]}: {e}")

    df = pd.DataFrame(results)
    if len(df) > 0:
        print(f"\n✓ Covariance Computation Complete")
        print(f"  Dates processed: {len(df)}")
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"  Mean MSE: {df['mse_cov'].mean():.6f}")
        print(f"  Std MSE:  {df['mse_cov'].std():.6f}")
    else:
        print("\nWarning: No covariance metrics computed")
    return df


def save_cov_results(cov_df, output_dir):
    """Save covariance results to CSV."""
    if len(cov_df) > 0:
        output_path = output_dir / "metrics" / "cov_metrics.csv"
        cov_df.to_csv(output_path, index=False)
        print(f"✓ Covariance results saved to: {output_path}")


def plot_cov_metrics(cov_df, output_dir):
    """Plot covariance MSE timeseries."""
    if len(cov_df) == 0:
        return

    plt.figure(figsize=(12, 6))
    plt.plot(cov_df['date'], cov_df['mse_cov'], alpha=0.7, linewidth=1.5)
    plt.xlabel('Date')
    plt.ylabel('MSE')
    plt.title('Covariance Prediction Error Over Time')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / "plots" / "cov_mse_timeseries.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Covariance plot saved to: {output_path}")
