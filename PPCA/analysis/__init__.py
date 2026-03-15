"""PPCA analysis sub-package."""

from .nll import compute_nll_metrics, save_nll_results, plot_nll_timeseries
from .covariance import compute_covariance_metrics, save_cov_results, plot_cov_metrics
from .var import compute_var_metrics, save_var_results, plot_var_calibration
from .portfolio import (
    compute_portfolio_metrics,
    plot_cumulative_returns,
    optimize_portfolio,
    compute_max_drawdown,
    load_ibovespa_returns,
)
from .report import generate_summary_report

__all__ = [
    "compute_nll_metrics",
    "save_nll_results",
    "plot_nll_timeseries",
    "compute_covariance_metrics",
    "save_cov_results",
    "plot_cov_metrics",
    "compute_var_metrics",
    "save_var_results",
    "plot_var_calibration",
    "compute_portfolio_metrics",
    "plot_cumulative_returns",
    "optimize_portfolio",
    "compute_max_drawdown",
    "load_ibovespa_returns",
    "generate_summary_report",
]
