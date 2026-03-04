"""Summary report generation for NeuralFactors evaluation."""

import pandas as pd
from pathlib import Path


def generate_summary_report(output_dir, nll_df, cov_df, var_df, backtest_metrics):
    """Generate and save a human-readable summary of all evaluation metrics.

    Args:
        output_dir: Output directory (Path)
        nll_df: NLL results DataFrame
        cov_df: Covariance results DataFrame
        var_df: VaR results DataFrame
        backtest_metrics: Dictionary of backtest metrics
    """
    print("\n" + "=" * 80)
    print("GENERATING SUMMARY REPORT")
    print("=" * 80)

    lines = []
    lines.append("=" * 80)
    lines.append("NEURALFACTORS MODEL EVALUATION SUMMARY")
    lines.append("=" * 80)
    lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # 1. NLL
    lines.append("-" * 80)
    lines.append("1. NEGATIVE LOG-LIKELIHOOD")
    lines.append("-" * 80)
    if not nll_df.empty:
        lines.append(f"Dates processed: {len(nll_df)}")
        lines.append(f"Date range: {nll_df['date'].min()} to {nll_df['date'].max()}")
        lines.append(f"Mean NLL_joint:   {nll_df['nll_joint'].mean():.4f} ± {nll_df['nll_joint'].std():.4f}")
        lines.append(f"Mean NLL_ind:     {nll_df['nll_ind'].mean():.4f} ± {nll_df['nll_ind'].std():.4f}")
        lines.append(f"Mean log p(r|z):  {nll_df['log_p_r_z'].mean():.4f}")
        lines.append(f"Mean KL:          {nll_df['kl'].mean():.4f}")
    else:
        lines.append("No NLL metrics computed")
    lines.append("")

    # 2. Covariance
    lines.append("-" * 80)
    lines.append("2. COVARIANCE PREDICTION")
    lines.append("-" * 80)
    if not cov_df.empty:
        lines.append(f"Dates processed: {len(cov_df)}")
        lines.append(f"Date range: {cov_df['date'].min()} to {cov_df['date'].max()}")
        lines.append(f"Mean MSE: {cov_df['mse_cov'].mean():.6f}")
        lines.append(f"Std MSE:  {cov_df['mse_cov'].std():.6f}")
        lines.append(f"Min MSE:  {cov_df['mse_cov'].min():.6f}")
        lines.append(f"Max MSE:  {cov_df['mse_cov'].max():.6f}")
    else:
        lines.append("No covariance metrics computed")
    lines.append("")

    # 3. VaR
    lines.append("-" * 80)
    lines.append("3. VALUE AT RISK CALIBRATION")
    lines.append("-" * 80)
    if not var_df.empty:
        lines.append(f"{'Quantile':<12} {'Theoretical':<15} {'Empirical':<15} {'Error':<10} Quality")
        lines.append("-" * 70)
        for _, row in var_df.iterrows():
            q = row['quantile']
            err = row['error']
            quality = 'Good' if err < 0.02 else ('OK' if err < 0.05 else 'Poor')
            lines.append(f"{q:<12.2f} {row['theoretical']:<15.4f} {row['empirical']:<15.4f} {err:<10.4f} {quality}")
    else:
        lines.append("No VaR metrics computed")
    lines.append("")

    # 4. Portfolio
    lines.append("-" * 80)
    lines.append("4. PORTFOLIO BACKTEST (MIN-VARIANCE)")
    lines.append("-" * 80)
    if backtest_metrics:
        lines.append(f"Total Return:       {backtest_metrics['total_return']:.2%}")
        lines.append(f"Annualized Return:  {backtest_metrics['annualized_return']:.2%}")
        lines.append(f"Annualized Vol:     {backtest_metrics['annualized_vol']:.2%}")
        lines.append(f"Sharpe Ratio:       {backtest_metrics['sharpe_ratio']:.2f}")
        lines.append(f"Max Drawdown:       {backtest_metrics['max_drawdown']:.2%}")
        if 'benchmark_total_return' in backtest_metrics:
            lines.append("")
            lines.append("Benchmark Comparison (Ibovespa):")
            lines.append(f"  Ann. Return:      {backtest_metrics['benchmark_annualized_return']:.2%}")
            lines.append(f"  Sharpe:           {backtest_metrics['benchmark_sharpe']:.2f}")
            lines.append(f"  Excess Return:    {backtest_metrics['excess_return']:.2%}")
            lines.append(f"  Information Ratio:{backtest_metrics['information_ratio']:.2f}")
    else:
        lines.append("No backtest metrics computed")
    lines.append("")

    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)

    report_text = "\n".join(lines)
    report_path = Path(output_dir) / "evaluation_summary.txt"
    with open(report_path, 'w') as f:
        f.write(report_text)

    print(f"\n✓ Summary report saved to: {report_path}")
    print("\n" + report_text)
