"""
General experiment helper utilities.

This module provides helper functions for experiments including OLS computation,
memory management, and experiment configuration.
"""

from typing import Dict, Tuple

import torch


def predict_ols(X_val: torch.Tensor, w: torch.Tensor, b: float) -> torch.Tensor:
    """
    OLS prediction function.

    Args:
        X_val: Input features tensor
        w: OLS coefficients
        b: OLS bias term

    Returns:
        OLS predictions
    """
    # Ensure all tensors are on the same device
    device = X_val.device
    w = w.to(device)
    b_tensor = torch.tensor(b, device=device, dtype=torch.float32)

    flat = X_val.reshape(-1, X_val.shape[-1]).float()
    return (flat @ w + b_tensor).reshape(X_val.shape[0], X_val.shape[1])


def clear_gpu_memory() -> None:
    """
    Clear GPU memory and cache.

    This is useful for preventing out-of-memory errors during
    long-running experiments with multiple model instances.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def compute_ols_coefficients(
    X_train: torch.Tensor, y_train: torch.Tensor, device: torch.device = None
) -> Tuple[torch.Tensor, float]:
    """
    Compute OLS coefficients using least squares.

    Args:
        X_train: Training features
        y_train: Training targets
        device: Device to use for computation

    Returns:
        Tuple of (coefficients, bias)
    """
    if device is None:
        device = X_train.device

    X_flat = X_train.reshape(-1, X_train.shape[-1]).double().to(device)
    y_flat = y_train.reshape(-1, 1).double().to(device)

    # Use 'gels' driver for CUDA compatibility
    driver = "gels" if device.type == "cuda" else "gelsy"

    coef = torch.linalg.lstsq(
        torch.cat([X_flat, torch.ones_like(y_flat)], 1), y_flat, driver=driver
    ).solution.squeeze()

    return coef[:-1].float(), float(coef[-1])


def create_experiment_config(
    n_trials_per_seed: int = None,
    n_dataset_seeds: int = 10,
    seed_range_start: int = 1,
    seed_range_end: int = 11,
) -> Dict:
    """
    Create experiment configuration dictionary.

    Args:
        n_trials_per_seed: Number of Optuna trials per dataset seed
        n_dataset_seeds: Number of different dataset seeds to test
        seed_range_start: Starting seed number
        seed_range_end: Ending seed number (exclusive)

    Returns:
        Experiment configuration dictionary
    """
    if n_trials_per_seed is None:
        n_trials_per_seed = 20 if torch.cuda.is_available() else 50

    return {
        "n_trials_per_seed": n_trials_per_seed,
        "n_dataset_seeds": n_dataset_seeds,
        "seed_range_start": seed_range_start,
        "seed_range_end": seed_range_end,
    }


def print_experiment_summary(config: Dict) -> None:
    """
    Print experiment configuration summary.

    Args:
        config: Experiment configuration dictionary
    """
    total_experiments = config["n_trials_per_seed"] * (
        config["seed_range_end"] - config["seed_range_start"]
    )

    print("Experiment Configuration:")
    print(f"  Trials per seed: {config['n_trials_per_seed']}")
    print(
        f"  Dataset seeds: {config['seed_range_start']} to {config['seed_range_end']-1}"
    )
    print(f"  Total experiments: {total_experiments}")
    print(f"  Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")


def get_device_info() -> Dict[str, str]:
    """
    Get information about the current device setup.

    Returns:
        Dictionary with device information
    """
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        return {
            "device": "cuda",
            "device_name": device_name,
            "memory_gb": f"{memory_gb:.1f}",
            "device_count": torch.cuda.device_count(),
        }
    else:
        return {
            "device": "cpu",
            "device_name": "CPU",
            "memory_gb": "N/A",
            "device_count": 0,
        }
