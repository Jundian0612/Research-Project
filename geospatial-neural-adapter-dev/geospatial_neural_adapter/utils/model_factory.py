"""
Model factory utilities for creating and initializing models.

This module provides utilities for creating trend and basis models
with proper initialization and configuration.
"""

from typing import List, Tuple

import torch

from geospatial_neural_adapter.models.spatial_basis_learner import SpatialBasisLearner
from geospatial_neural_adapter.models.trend_model import TrendModel


def create_fresh_models(
    device: torch.device,
    p_dim: int,
    n_locations: int,
    latent_dim: int,
    w_ols: torch.Tensor,
    b_ols: float,
    hidden_layer_sizes: List[int] = None,
    dropout_rate: float = 0.1,
) -> Tuple[TrendModel, SpatialBasisLearner]:
    """
    Create fresh trend and basis models with OLS initialization.

    Args:
        device: Device to place models on
        p_dim: Number of continuous features
        n_locations: Number of spatial locations
        latent_dim: Latent dimension for basis
        w_ols: OLS coefficients for initialization
        b_ols: OLS bias for initialization
        hidden_layer_sizes: Hidden layer sizes for trend model
        dropout_rate: Dropout rate for trend model

    Returns:
        Tuple of (trend_model, basis_model)
    """
    if hidden_layer_sizes is None:
        hidden_layer_sizes = [256, 64]

    trend = TrendModel(
        num_continuous_features=p_dim,
        hidden_layer_sizes=hidden_layer_sizes,
        n_locations=n_locations,
        init_weight=w_ols,
        init_bias=b_ols,
        freeze_init=True,
        dropout_rate=dropout_rate,
    ).to(device)

    basis = SpatialBasisLearner(num_locations=n_locations, latent_dim=latent_dim).to(
        device
    )

    return trend, basis


def create_trend_model(
    device: torch.device,
    p_dim: int,
    n_locations: int,
    w_ols: torch.Tensor,
    b_ols: float,
    hidden_layer_sizes: List[int] = None,
    dropout_rate: float = 0.1,
    freeze_init: bool = True,
) -> TrendModel:
    """
    Create a trend model with OLS initialization.

    Args:
        device: Device to place model on
        p_dim: Number of continuous features
        n_locations: Number of spatial locations
        w_ols: OLS coefficients for initialization
        b_ols: OLS bias for initialization
        hidden_layer_sizes: Hidden layer sizes
        dropout_rate: Dropout rate
        freeze_init: Whether to freeze initial weights

    Returns:
        Initialized trend model
    """
    if hidden_layer_sizes is None:
        hidden_layer_sizes = [256, 64]

    trend = TrendModel(
        num_continuous_features=p_dim,
        hidden_layer_sizes=hidden_layer_sizes,
        n_locations=n_locations,
        init_weight=w_ols,
        init_bias=b_ols,
        freeze_init=freeze_init,
        dropout_rate=dropout_rate,
    ).to(device)

    return trend


def create_basis_model(
    device: torch.device,
    n_locations: int,
    latent_dim: int,
) -> SpatialBasisLearner:
    """
    Create a spatial basis learner model.

    Args:
        device: Device to place model on
        n_locations: Number of spatial locations
        latent_dim: Latent dimension for basis

    Returns:
        Initialized basis model
    """
    basis = SpatialBasisLearner(num_locations=n_locations, latent_dim=latent_dim).to(
        device
    )

    return basis
