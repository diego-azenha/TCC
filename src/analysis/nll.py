"""Negative Log-Likelihood (NLL) metrics for NeuralFactors evaluation."""

import pandas as pd
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm


def compute_nll_metrics(model, dataloader, dataset, num_samples, mode, device):
    """Compute NLL_joint and NLL_ind on dataset.

    Args:
        model: Trained NeuralFactors model
        dataloader: DataLoader for iteration
        dataset: Dataset for accessing dates
        num_samples: Number of IWAE samples
        mode: 'debug' (first 50 dates) or 'paper' (all dates)
        device: torch device

    Returns:
        pd.DataFrame: [date, nll_joint, nll_ind, log_p_r_z, kl, n_stocks]
    """
    print("\n" + "=" * 80)
    print("COMPUTING NEGATIVE LOG-LIKELIHOOD METRICS")
    print("=" * 80)

    max_dates = 50 if mode == 'debug' else None
    if max_dates:
        print(f"Debug mode: Processing first {max_dates} dates")

    results = []
    model.eval()

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, desc="Computing NLL")):
            if max_dates and idx >= max_dates:
                break

            S, S_static, r, mask = batch
            S = S.to(device)
            S_static = S_static.to(device)
            r = r.to(device)
            mask = mask.to(device)

            output = model.model.compute_iwae_loss(S, S_static, r, num_samples, mask)

            nll_joint = output['loss'].item()
            log_p_r_z = output['log_likelihood'].item()
            kl = output['kl_divergence'].item()
            n_stocks = mask.sum().item()
            nll_ind = nll_joint / max(n_stocks, 1)

            results.append({
                'date': dataset.dates[idx],
                'nll_joint': nll_joint,
                'nll_ind': nll_ind,
                'log_p_r_z': log_p_r_z,
                'kl': kl,
                'n_stocks': n_stocks,
            })

    df = pd.DataFrame(results)
    print(f"\n✓ NLL Computation Complete")
    print(f"  Dates processed: {len(df)}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Mean NLL_joint: {df['nll_joint'].mean():.4f}")
    print(f"  Std NLL_joint:  {df['nll_joint'].std():.4f}")
    print(f"  Mean NLL_ind:   {df['nll_ind'].mean():.4f}")
    print(f"  Mean log p(r|z):{df['log_p_r_z'].mean():.4f}")
    print(f"  Mean KL:        {df['kl'].mean():.4f}")
    return df


def save_nll_results(nll_df, output_dir):
    """Save NLL results to CSV."""
    output_path = output_dir / "metrics" / "nll_timeseries.csv"
    nll_df.to_csv(output_path, index=False)
    print(f"✓ NLL results saved to: {output_path}")


def plot_nll_timeseries(nll_df, output_dir):
    """Plot NLL_joint and KL divergence over time."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(nll_df['date'], nll_df['nll_joint'], alpha=0.7, linewidth=1.5)
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('NLL Joint')
    axes[0].set_title('Negative Log-Likelihood (Joint) Over Time')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(nll_df['date'], nll_df['kl'], color='orange', alpha=0.7, linewidth=1.5)
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('KL Divergence')
    axes[1].set_title('KL Divergence Over Time')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "plots" / "nll_timeseries.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ NLL plot saved to: {output_path}")
