"""
Utility modules for the geospatial neural adapter package.
"""

from .experiment import log_covariance_and_basis

# General experiment helper utilities
from .experiment_helpers import (
    clear_gpu_memory,
    compute_ols_coefficients,
    create_experiment_config,
    get_device_info,
    predict_ols,
    print_experiment_summary,
)

# Model caching utilities (for hyperparameter optimization)
from .model_cache import ModelCache

# Model factory utilities (for model creation)
from .model_factory import create_basis_model, create_fresh_models, create_trend_model

__all__ = [
    # Model caching (for tuning)
    "ModelCache",
    # Model factory
    "create_fresh_models",
    "create_trend_model",
    "create_basis_model",
    # Experiment helpers
    "clear_gpu_memory",
    "compute_ols_coefficients",
    "create_experiment_config",
    "get_device_info",
    "predict_ols",
    "print_experiment_summary",
    # Existing utilities
    "log_covariance_and_basis",
]
