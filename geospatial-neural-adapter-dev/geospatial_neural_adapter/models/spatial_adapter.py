"""
Unified SpatialAdapter PyTorch Module

This module provides a complete, self-contained implementation of the spatial adapter:
Y(t,s) = g(f_θ(x(t,s)) + Φ(s)^T η(t)) + ε

Key features:
- Multiple methods for computing spatial factors η(t)
- Unified forward pass for inference
- Spatial uncertainty quantification
- Save/load as single module
- Compatible with standard PyTorch workflows
- Seamless integration with ADMM training
"""

from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from geospatial_neural_adapter.models.spatial_basis_learner import SpatialBasisLearner
from geospatial_neural_adapter.models.trend_model import TrendModel


class SpatialAdapter(nn.Module):
    """
    Complete Spatial Adapter module implementing Y(t,s) = g(f_θ(x) + Φ(s)^T η(t)) + ε

    This module encapsulates:
    - Trend model f_θ(x): backbone neural network
    - Spatial basis Φ(s): learnable thin-plate spline basis
    - Spatial factors η(t): time-varying coefficients with multiple computation methods
    - Output transformation g(): identity, sigmoid, or softmax
    - Uncertainty quantification via spatial covariance
    """

    def __init__(
        self,
        trend_model: TrendModel,
        spatial_basis: SpatialBasisLearner,
        num_locations: int,
        latent_dim: int,
        locations: torch.Tensor,
        factor_computation_method: str = "residual_projection",
        output_activation: str = "identity",
        noise_variance: float = 1.0,
        enable_uncertainty: bool = True,
    ):
        """
        Args:
            trend_model: The backbone trend model f_θ
            spatial_basis: The spatial basis learner Φ
            num_locations: Number of spatial locations N
            latent_dim: Dimension of spatial basis K
            locations: Spatial coordinates (N, 2) for locations
            factor_computation_method: Method for computing spatial factors:
                - "residual_projection": Project residuals onto spatial basis
                - "learned_mapping": Use a learned neural network
                - "stored_factors": Use pre-computed stored factors
                - "zero": Zero factors (trend-only predictions)
            output_activation: 'identity', 'sigmoid', or 'softmax'
            noise_variance: Estimated noise variance σ²
            enable_uncertainty: Whether to compute uncertainty estimates
        """
        super().__init__()

        self.trend_model = trend_model
        self.spatial_basis = spatial_basis
        self.num_locations = num_locations
        self.latent_dim = latent_dim
        self.factor_computation_method = factor_computation_method
        self.output_activation = output_activation
        self.noise_variance = noise_variance
        self.enable_uncertainty = enable_uncertainty

        # Store locations for uncertainty quantification
        self.register_buffer("locations", locations)

        # Initialize spatial factors computation based on method
        if factor_computation_method == "learned_mapping":
            # Neural network to compute spatial factors from input features
            self.factor_network = nn.Sequential(
                nn.Linear(num_locations, latent_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(latent_dim * 2, latent_dim),
            )

            # Zero initialize to start with trend-only predictions
            for layer in self.factor_network.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.zeros_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

        elif factor_computation_method == "stored_factors":
            # Buffer to store pre-computed spatial factors
            self.register_buffer("stored_factors", torch.zeros(1, latent_dim))

        # Store training statistics for normalization
        self.register_buffer("train_mean", torch.zeros(num_locations))
        self.register_buffer("train_std", torch.ones(num_locations))

        # Spatial covariance for uncertainty quantification
        if enable_uncertainty:
            # Initialize spatial covariance parameters
            self.register_buffer("spatial_cov_eigenvals", torch.ones(latent_dim))
            self.register_buffer("spatial_cov_eigenvecs", torch.eye(latent_dim))

        # Output transformation function
        self.output_transform = self._get_output_transform(output_activation)

    def _get_output_transform(self, activation: str):
        """Get the output transformation function g()."""
        if activation == "identity":
            return lambda x: x
        elif activation == "sigmoid":
            return torch.sigmoid
        elif activation == "softmax":
            return lambda x: F.softmax(x, dim=-1)
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(
        self,
        x: torch.Tensor,
        return_components: bool = False,
        return_uncertainty: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Forward pass through the spatial adapter.

        Args:
            x: Input features (batch_size, num_locations, num_features)
            return_components: Whether to return individual components
            return_uncertainty: Whether to return uncertainty estimates

        Returns:
            If return_components=False and return_uncertainty=False:
                predictions: (batch_size, num_locations)

            If return_components=True or return_uncertainty=True:
                predictions: (batch_size, num_locations)
                info: Dict containing components and/or uncertainty estimates
        """
        x.shape[0]

        # 1. Compute trend prediction: f_θ(x)
        trend_pred = self.trend_model(x)  # (batch_size, num_locations)

        # 2. Compute spatial factors η(t) using the specified method
        spatial_factors = self._compute_spatial_factors(
            x, trend_pred
        )  # (batch_size, latent_dim)

        # 3. Compute spatial component: Φ(s)^T η(t)
        spatial_basis = self.spatial_basis.basis  # (num_locations, latent_dim)
        spatial_component = (
            spatial_factors @ spatial_basis.T
        )  # (batch_size, num_locations)

        # 4. Combine trend and spatial components
        combined_logits = trend_pred + spatial_component  # (batch_size, num_locations)

        # 5. Apply output transformation
        predictions = self.output_transform(combined_logits)

        # 6. Prepare return values
        if not return_components and not return_uncertainty:
            return predictions

        # Prepare additional information
        info = {}

        if return_components:
            info.update(
                {
                    "trend_component": trend_pred,
                    "spatial_component": spatial_component,
                    "spatial_factors": spatial_factors,
                    "combined_logits": combined_logits,
                }
            )

        if return_uncertainty and self.enable_uncertainty:
            uncertainty_info = self._compute_uncertainty(
                x, predictions, spatial_factors
            )
            info.update(uncertainty_info)

        return predictions, info

    def _compute_spatial_factors(
        self, x: torch.Tensor, trend_pred: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute spatial factors η(t) using the specified method.
        """
        batch_size = x.shape[0]

        if self.factor_computation_method == "residual_projection":
            # Project residuals onto spatial basis using pseudo-inverse
            return self._compute_factors_residual_projection(x, trend_pred)

        elif self.factor_computation_method == "learned_mapping":
            # Use learned neural network to map features to spatial factors
            return self._compute_factors_learned_mapping(x, trend_pred)

        elif self.factor_computation_method == "stored_factors":
            # Use stored pre-computed factors
            return self.stored_factors.expand(batch_size, -1)

        else:  # "zero" or any other method
            # Zero factors (trend-only predictions)
            return torch.zeros(
                batch_size, self.latent_dim, device=x.device, dtype=x.dtype
            )

    def _compute_factors_residual_projection(
        self, x: torch.Tensor, trend_pred: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute spatial factors by projecting residuals onto spatial basis.

        This assumes we have some reference values (e.g., from training data mean)
        to compute meaningful residuals.
        """
        batch_size = x.shape[0]

        # Use training mean as reference
        reference = self.train_mean.unsqueeze(0).expand(batch_size, -1)

        # Compute residual from trend prediction
        residual = reference - trend_pred  # (batch_size, num_locations)

        # Project residual onto spatial basis: η = Φ^T residual
        spatial_basis = self.spatial_basis.basis  # (num_locations, latent_dim)
        spatial_factors = residual @ spatial_basis  # (batch_size, latent_dim)

        return spatial_factors

    def _compute_factors_learned_mapping(
        self, x: torch.Tensor, trend_pred: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute spatial factors using a learned neural network.
        """
        # Use trend prediction as input to factor network
        spatial_factors = self.factor_network(trend_pred)
        return spatial_factors

    def _compute_uncertainty(
        self, x: torch.Tensor, predictions: torch.Tensor, spatial_factors: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute spatial uncertainty estimates.

        Returns predictive variance at each location based on:
        1. Spatial basis uncertainty
        2. Trend model uncertainty
        3. Noise variance
        """
        batch_size = x.shape[0]

        # Spatial basis uncertainty
        # Var[Φ η] = Φ Var[η] Φ^T
        spatial_basis = self.spatial_basis.basis  # (num_locations, latent_dim)

        # Use stored covariance eigenvalues and eigenvectors
        cov_matrix = (
            self.spatial_cov_eigenvecs * self.spatial_cov_eigenvals
        ) @ self.spatial_cov_eigenvecs.T

        # Compute spatial variance: Φ Σ_η Φ^T
        spatial_variance = (
            spatial_basis @ cov_matrix @ spatial_basis.T
        )  # (num_locations, num_locations)

        # Extract diagonal (per-location variance)
        spatial_var_diag = torch.diag(spatial_variance)  # (num_locations,)

        # Add noise variance
        total_variance = spatial_var_diag + self.noise_variance

        # Expand to batch dimension
        total_variance = total_variance.unsqueeze(0).expand(batch_size, -1)

        # Standard deviation
        total_std = torch.sqrt(total_variance)

        # Confidence intervals (95%)
        confidence_lower = predictions - 1.96 * total_std
        confidence_upper = predictions + 1.96 * total_std

        return {
            "predictive_variance": total_variance,
            "predictive_std": total_std,
            "confidence_lower": confidence_lower,
            "confidence_upper": confidence_upper,
            "spatial_variance_diag": spatial_var_diag,
        }

    def update_from_trainer(self, trainer):
        """
        Update the spatial adapter with learned parameters from trainer.

        Args:
            trainer: SpatialNeuralAdapter instance after training
        """
        # Update trend model
        self.trend_model.load_state_dict(trainer.trend.state_dict())

        # Update spatial basis
        self.spatial_basis.load_state_dict(trainer.basis.state_dict())

        # Update training statistics
        self.train_mean.copy_(trainer.train_y.mean(dim=0))
        self.train_std.copy_(trainer.train_y.std(dim=0))

        # Update spatial factors based on training results
        if self.factor_computation_method == "stored_factors":
            # Use the final consensus variables as spatial factors
            final_residuals = trainer.z_train.mean(dim=0)  # Average over time
            spatial_factors = (
                final_residuals @ self.spatial_basis.basis
            )  # Project to factor space
            self.stored_factors.copy_(spatial_factors.mean(dim=0, keepdim=True))

        # Update spatial covariance from training residuals
        if self.enable_uncertainty:
            self._update_spatial_covariance_from_training(trainer)

    def _update_spatial_covariance_from_training(self, trainer):
        """
        Update spatial covariance parameters from training data.
        """
        # Compute spatial factors from training data
        training_residuals = trainer.z_train  # (T, N)
        spatial_factors = training_residuals @ self.spatial_basis.basis  # (T, K)

        # Compute sample covariance of spatial factors
        spatial_factors_centered = spatial_factors - spatial_factors.mean(dim=0)
        cov_matrix = torch.cov(spatial_factors_centered.T)  # (K, K)

        # Eigendecomposition
        eigenvals, eigenvecs = torch.linalg.eigh(cov_matrix)

        # Store eigenvalues and eigenvectors
        self.spatial_cov_eigenvals.copy_(
            eigenvals.clamp(min=1e-6)
        )  # Avoid numerical issues
        self.spatial_cov_eigenvecs.copy_(eigenvecs)

        # Update noise variance estimate
        residual_variance = trainer.z_train.var().item()
        self.noise_variance = residual_variance

    def train_factor_network(self, trainer, num_epochs: int = 50):
        """
        Train the factor network to predict spatial factors from trend predictions.

        This is only applicable when using "learned_mapping" method.
        """
        if self.factor_computation_method != "learned_mapping":
            return

        # Create training data for factor network
        with torch.no_grad():
            trend_predictions = trainer.trend(trainer.train_cont)  # (T, N)
            target_factors = trainer.z_train @ self.spatial_basis.basis  # (T, K)

        # Train factor network
        optimizer = torch.optim.Adam(self.factor_network.parameters(), lr=1e-3)

        for epoch in range(num_epochs):
            optimizer.zero_grad()

            predicted_factors = self.factor_network(trend_predictions)
            loss = F.mse_loss(predicted_factors, target_factors)

            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                print(
                    f"Factor network training - Epoch {epoch}, Loss: {loss.item():.4f}"
                )

    def predict_at_locations(
        self,
        x: torch.Tensor,
        new_locations: torch.Tensor,
        return_uncertainty: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Predict at new spatial locations using kriging.

        Args:
            x: Input features (batch_size, num_locations, num_features)
            new_locations: New spatial coordinates (num_new_locations, 2)
            return_uncertainty: Whether to return uncertainty estimates

        Returns:
            predictions: (batch_size, num_new_locations)
            uncertainty_info: Dict with uncertainty estimates (if requested)
        """
        batch_size = x.shape[0]
        num_new_locations = new_locations.shape[0]

        # This would involve:
        # 1. Compute basis functions at new locations
        # 2. Use trained spatial factors for prediction
        # 3. Compute uncertainty via kriging variance

        # For now, return placeholder
        predictions = torch.zeros(
            batch_size, num_new_locations, device=x.device, dtype=x.dtype
        )

        if return_uncertainty:
            uncertainty_info = {
                "predictive_variance": torch.ones_like(predictions),
                "predictive_std": torch.ones_like(predictions),
                "confidence_lower": predictions - 1.96,
                "confidence_upper": predictions + 1.96,
            }
            return predictions, uncertainty_info

        return predictions

    def get_spatial_basis_maps(self) -> torch.Tensor:
        """
        Get the learned spatial basis functions for visualization.

        Returns:
            basis_maps: (num_locations, latent_dim) spatial basis functions
        """
        return self.spatial_basis.basis.detach().cpu()

    def save_model(self, path: str):
        """Save the complete spatial adapter model."""
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "num_locations": self.num_locations,
                "latent_dim": self.latent_dim,
                "factor_computation_method": self.factor_computation_method,
                "output_activation": self.output_activation,
                "noise_variance": self.noise_variance,
                "enable_uncertainty": self.enable_uncertainty,
                "locations": self.locations,
            },
            path,
        )

    @classmethod
    def load_model(
        cls,
        path: str,
        trend_model: TrendModel,
        spatial_basis: SpatialBasisLearner,
    ):
        """Load a spatial adapter model."""
        checkpoint = torch.load(path)

        model = cls(
            trend_model=trend_model,
            spatial_basis=spatial_basis,
            num_locations=checkpoint["num_locations"],
            latent_dim=checkpoint["latent_dim"],
            locations=checkpoint["locations"],
            factor_computation_method=checkpoint["factor_computation_method"],
            output_activation=checkpoint["output_activation"],
            noise_variance=checkpoint["noise_variance"],
            enable_uncertainty=checkpoint["enable_uncertainty"],
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        return model


def create_spatial_adapter(
    trend_model: TrendModel,
    num_locations: int,
    latent_dim: int,
    locations: torch.Tensor,
    factor_computation_method: str = "residual_projection",
    pca_init: Optional[np.ndarray] = None,
    output_activation: str = "identity",
    noise_variance: float = 1.0,
    enable_uncertainty: bool = True,
) -> SpatialAdapter:
    """
    Factory function to create a complete spatial adapter.

    Args:
        trend_model: The backbone trend model
        num_locations: Number of spatial locations
        latent_dim: Dimension of spatial basis (K << N)
        locations: Spatial coordinates (N, 2)
        factor_computation_method: Method for computing spatial factors
        pca_init: Optional PCA initialization for basis
        output_activation: Output transformation
        noise_variance: Estimated noise variance
        enable_uncertainty: Whether to enable uncertainty quantification

    Returns:
        Complete SpatialAdapter module
    """
    # Create spatial basis learner
    spatial_basis = SpatialBasisLearner(
        num_locations=num_locations,
        latent_dim=latent_dim,
        pca_init=pca_init,
    )

    # Create complete spatial adapter
    spatial_adapter = SpatialAdapter(
        trend_model=trend_model,
        spatial_basis=spatial_basis,
        num_locations=num_locations,
        latent_dim=latent_dim,
        locations=locations,
        factor_computation_method=factor_computation_method,
        output_activation=output_activation,
        noise_variance=noise_variance,
        enable_uncertainty=enable_uncertainty,
    )

    return spatial_adapter
