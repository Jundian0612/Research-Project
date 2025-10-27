"""
Synthetic data generation functions for geospatial neural adapter.
"""

from typing import Optional, Tuple

import numpy as np


def _generate_spatial_basis(locations: np.ndarray) -> np.ndarray:
    """
    Generate spatial basis using Gaussian kernel.

    Args:
        locations: Spatial locations (N,)

    Returns:
        spatial_basis: Normalized spatial basis (N, 1)
    """
    # Generate spatial basis (Gaussian kernel)
    spatial_basis = np.exp(-(locations**2))[:, None]
    spatial_basis /= np.linalg.norm(spatial_basis)
    return spatial_basis


def generate_combined_synthetic_data(
    location: np.ndarray,
    n_samples: int,
    noise_std: float,
    eigenvalue: float,
    global_mean: float = 50.0,
    feature_noise_std: float = 0.0,  # Additional feature noise
    non_linear_strength: float = 0.0,  # Non-linear relationships
    correlation_strength: float = 1.0,  # Strength of feature-target correlations
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic data with combined trend and spatial components.

    Args:
        location: Spatial locations (N,)
        n_samples: Number of samples
        noise_std: Standard deviation of noise
        eigenvalue: Eigenvalue for spatial correlation
        global_mean: Global mean
        seed: Random seed

    Returns:
        cat_features: Categorical features (T, N, 0)
        cont_features: Continuous features (T, N, p)
        targets: Target values (T, N)
    """
    if seed is not None:
        np.random.seed(seed)

    N = len(location)
    p = 3  # Number of continuous features

    # Generate continuous features with meaningful patterns
    cont_features = np.random.randn(n_samples, N, p)

    # Add spatial and temporal patterns to features to create correlations
    spatial_pattern = np.sin(location * np.pi / 5)  # Spatial wave pattern
    temporal_trend = np.linspace(0, 2 * np.pi, n_samples)

    # Feature 1: Spatial pattern correlation
    cont_features[:, :, 0] += (
        np.outer(temporal_trend, spatial_pattern) * 0.5 * correlation_strength
    )

    # Feature 2: Temporal trend correlation
    cont_features[:, :, 1] += (
        np.outer(temporal_trend, np.ones(N)) * 0.3 * correlation_strength
    )

    # Feature 3: Interaction term
    cont_features[:, :, 2] += (
        cont_features[:, :, 0] * cont_features[:, :, 1] * 0.2 * correlation_strength
    )

    # ADD FEATURE NOISE
    if feature_noise_std > 0:
        cont_features += np.random.randn(*cont_features.shape) * feature_noise_std

    # Generate spatial basis using shared function
    spatial_basis = _generate_spatial_basis(location)

    # Generate spatial weights
    spatial_weights = np.random.randn(n_samples, 1) * np.sqrt(eigenvalue)

    # Generate target values with feature correlations
    targets = np.zeros((n_samples, N))
    for t in range(n_samples):
        # Trend component
        trend = global_mean + np.random.randn(N) * 0.1

        # Spatial component
        spatial = spatial_weights[t] * spatial_basis.flatten()

        # Feature-based component with optional non-linear relationships
        feature_component = (
            0.3 * cont_features[t, :, 0]
            + 0.4 * cont_features[t, :, 1]  # Feature 1 weight
            + 0.2 * cont_features[t, :, 2]  # Feature 2 weight  # Feature 3 weight
        ) * correlation_strength

        # ADD NON-LINEAR RELATIONSHIPS
        if non_linear_strength > 0:
            feature_component += (
                non_linear_strength * cont_features[t, :, 0] ** 2
                + non_linear_strength  # Quadratic term
                * 0.5
                * cont_features[t, :, 0]
                * cont_features[t, :, 1]
                + non_linear_strength  # Interaction
                * 0.3
                * np.sin(cont_features[t, :, 2])  # Sinusoidal term
            )

        # Noise
        noise = np.random.randn(N) * noise_std

        targets[t] = trend + spatial + feature_component + noise

    # No categorical features for now
    cat_features = np.zeros((n_samples, N, 0), dtype=np.int64)

    return cat_features, cont_features, targets


def generate_time_synthetic_data(
    locs: np.ndarray,
    n_time_steps: int,
    noise_std: float,
    eigenvalue: float,
    eta_rho: float = 0.8,
    f_rho: float = 0.6,
    global_mean: float = 50.0,
    feature_noise_std: float = 0.0,  # Additional feature noise
    non_linear_strength: float = 0.0,  # Non-linear relationships
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic temporal data with spatial correlation and meaningful feature-target relationships.

    Args:
        locs: Spatial locations (N,)
        n_time_steps: Number of time steps
        noise_std: Standard deviation of noise
        eigenvalue: Eigenvalue for spatial correlation
        eta_rho: AR coefficient for spatial weights (t)
        f_rho: AR coefficient for backbone drift (t)
        global_mean: Global mean
        seed: Random seed

    Returns:
        cat: Categorical features (T, N, 0)
        cont: Continuous features (T, N, p)
        y: Target values (T, N)
    """
    if seed is not None:
        np.random.seed(seed)

    N = len(locs)
    p = 3  # Number of continuous features

    # Generate base continuous features
    cont = np.random.randn(n_time_steps, N, p)

    # Add meaningful patterns to features
    spatial_pattern = np.sin(locs * np.pi / 5)  # Spatial wave pattern
    temporal_trend = np.linspace(0, 2 * np.pi, n_time_steps)

    # Feature 1: Spatial pattern correlation
    cont[:, :, 0] += np.outer(temporal_trend, spatial_pattern) * 0.5

    # Feature 2: Temporal trend correlation
    cont[:, :, 1] += np.outer(temporal_trend, np.ones(N)) * 0.3

    # Feature 3: Interaction term
    cont[:, :, 2] += cont[:, :, 0] * cont[:, :, 1] * 0.2

    # ADD FEATURE NOISE
    if feature_noise_std > 0:
        cont += np.random.randn(*cont.shape) * feature_noise_std

    # Generate spatial basis using shared function
    spatial_basis = _generate_spatial_basis(locs)

    # Generate temporal components
    spatial_weights = np.zeros((n_time_steps, 1))  # Spatial weights
    trend_drift = np.zeros((n_time_steps, 1))  # Backbone drift

    # Initialize with random values
    spatial_weights[0] = np.random.randn(1)
    trend_drift[0] = np.random.randn(1)

    # Generate AR processes
    for t in range(1, n_time_steps):
        spatial_weights[t] = eta_rho * spatial_weights[t - 1] + np.random.randn(1) * 0.1
        trend_drift[t] = f_rho * trend_drift[t - 1] + np.random.randn(1) * 0.1

    # Generate target values with feature correlations
    y = np.zeros((n_time_steps, N))
    for t in range(n_time_steps):
        # Trend component
        trend = global_mean + trend_drift[t] * np.ones(N)

        # Spatial component
        spatial = eigenvalue * spatial_weights[t] * spatial_basis.flatten()

        # Feature-based component with optional non-linear relationships
        feature_component = (
            0.3 * cont[t, :, 0]
            + 0.4 * cont[t, :, 1]  # Feature 1 weight
            + 0.2 * cont[t, :, 2]  # Feature 2 weight  # Feature 3 weight
        )

        # ADD NON-LINEAR RELATIONSHIPS
        if non_linear_strength > 0:
            feature_component += (
                non_linear_strength * cont[t, :, 0] ** 2
                + non_linear_strength  # Quadratic term
                * 0.5
                * cont[t, :, 0]
                * cont[t, :, 1]
                + non_linear_strength  # Interaction
                * 0.3
                * np.sin(cont[t, :, 2])  # Sinusoidal term
            )

        # Noise
        noise = np.random.randn(N) * noise_std

        y[t] = trend + spatial + feature_component + noise

    # No categorical features for now
    cat = np.zeros((n_time_steps, N, 0), dtype=np.int64)

    return cat, cont, y
