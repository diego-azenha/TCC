"""NeuralFactors analysis package."""

from .loader import load_model_and_data, setup_output_dirs
from .nll import compute_nll_metrics, save_nll_results, plot_nll_timeseries
from .var import compute_var_metrics, save_var_results, plot_var_calibration
from .covariance import compute_covariance_metrics, save_cov_results, plot_cov_metrics
from .portfolio import (
    compute_portfolio_metrics,
    plot_cumulative_returns,
    optimize_portfolio,
    compute_max_drawdown,
    load_ibovespa_returns,
)
from .report import generate_summary_report

__all__ = [
    "load_model_and_data",
    "setup_output_dirs",
    "compute_nll_metrics",
    "save_nll_results",
    "plot_nll_timeseries",
    "compute_var_metrics",
    "save_var_results",
    "plot_var_calibration",
    "compute_covariance_metrics",
    "save_cov_results",
    "plot_cov_metrics",
    "compute_portfolio_metrics",
    "plot_cumulative_returns",
    "optimize_portfolio",
    "compute_max_drawdown",
    "load_ibovespa_returns",
    "generate_summary_report",
]
