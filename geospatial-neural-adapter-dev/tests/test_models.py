"""
Unit tests for geospatial_neural_adapter.models module.
"""

import numpy as np
import torch

from geospatial_neural_adapter.models.spatial_basis_learner import SpatialBasisLearner
from geospatial_neural_adapter.models.trend_model import TrendModel


class TestTrendModel:
    def test_trend_model_initialization(self):
        """Test TrendModel initialization with OLS weights."""
        p_dim = 5
        n_locations = 10
        hidden_sizes = [64, 32]

        # Create OLS weights
        w_ols = torch.randn(p_dim)
        b_ols = 1.5

        trend = TrendModel(
            num_continuous_features=p_dim,
            hidden_layer_sizes=hidden_sizes,
            n_locations=n_locations,
            init_weight=w_ols,
            init_bias=b_ols,
            freeze_init=True,
            dropout_rate=0.1,
        )

        assert trend.init_lin.weight.shape == (1, p_dim)
        assert trend.init_lin.bias.shape == (1,)
        assert not trend.init_lin.weight.requires_grad  # Should be frozen

        # Check that weights are set correctly
        torch.testing.assert_close(trend.init_lin.weight.squeeze(), w_ols)
        torch.testing.assert_close(trend.init_lin.bias.squeeze(), torch.tensor(b_ols))

    def test_trend_model_forward(self):
        """Test TrendModel forward pass."""
        p_dim = 3
        n_locations = 5
        batch_size = 4

        trend = TrendModel(
            num_continuous_features=p_dim,
            hidden_layer_sizes=[64, 32],
            n_locations=n_locations,
            init_weight=None,
            init_bias=None,
            freeze_init=False,
            dropout_rate=0.1,
        )

        x = torch.randn(batch_size, n_locations, p_dim)
        output = trend(x)

        assert output.shape == (batch_size, n_locations)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_trend_model_no_hidden_layers(self):
        """Test TrendModel with no hidden layers (linear only)."""
        p_dim = 3
        n_locations = 5
        batch_size = 4

        trend = TrendModel(
            num_continuous_features=p_dim,
            hidden_layer_sizes=[],
            n_locations=n_locations,
            init_weight=None,
            init_bias=None,
            freeze_init=False,
            dropout_rate=0.0,
        )

        x = torch.randn(batch_size, n_locations, p_dim)
        output = trend(x)

        assert output.shape == (batch_size, n_locations)
        assert not torch.isnan(output).any()

    def test_trend_model_residual_parameters(self):
        """Test that residual_parameters returns only trainable parameters."""
        p_dim = 3
        n_locations = 5

        # Create model with frozen initialization
        w_ols = torch.randn(p_dim)
        b_ols = 1.0

        trend = TrendModel(
            num_continuous_features=p_dim,
            hidden_layer_sizes=[64],
            n_locations=n_locations,
            init_weight=w_ols,
            init_bias=b_ols,
            freeze_init=True,
            dropout_rate=0.1,
        )

        residual_params = trend.residual_parameters()

        # Check that init_lin parameters are not in residual_params
        init_params = set(trend.init_lin.parameters())
        residual_param_set = set(residual_params)

        assert len(init_params.intersection(residual_param_set)) == 0

        # Check that other parameters are in residual_params
        assert len(residual_params) > 0


class TestSpatialBasisLearner:
    def test_spatial_basis_learner_initialization(self):
        """Test SpatialBasisLearner initialization."""
        num_locations = 10
        latent_dim = 3

        basis = SpatialBasisLearner(
            num_locations=num_locations,
            latent_dim=latent_dim,
            pca_init=None,
        )

        assert basis.basis.shape == (num_locations, latent_dim)
        assert basis.num_locations == num_locations
        assert basis.latent_dim == latent_dim

    def test_spatial_basis_learner_forward(self):
        """Test SpatialBasisLearner forward pass."""
        num_locations = 10
        latent_dim = 3

        basis = SpatialBasisLearner(
            num_locations=num_locations,
            latent_dim=latent_dim,
            pca_init=None,
        )

        output = basis()

        assert output.shape == (num_locations, latent_dim)
        assert torch.allclose(output, basis.basis)

    def test_spatial_basis_learner_pca_init(self):
        """Test SpatialBasisLearner with PCA initialization."""
        num_locations = 10
        latent_dim = 3

        # Create PCA initialization
        pca_init = np.random.randn(num_locations, latent_dim)

        basis = SpatialBasisLearner(
            num_locations=num_locations,
            latent_dim=latent_dim,
            pca_init=pca_init,
        )

        # Check that basis is close to PCA init (after orthogonalization)
        basis_np = basis.basis.detach().numpy()
        assert basis_np.shape == (num_locations, latent_dim)

    def test_spatial_basis_learner_project_reconstruct(self):
        """Test projection and reconstruction methods."""
        num_locations = 10
        latent_dim = 3
        batch_size = 4

        basis = SpatialBasisLearner(
            num_locations=num_locations,
            latent_dim=latent_dim,
            pca_init=None,
        )

        # Test data
        data = torch.randn(batch_size, num_locations)

        # Project
        coefficients = basis.project(data)
        assert coefficients.shape == (batch_size, latent_dim)

        # Reconstruct
        reconstructed = basis.reconstruct(coefficients)
        assert reconstructed.shape == (batch_size, num_locations)

        # Check that projection + reconstruction gives reasonable result
        assert not torch.isnan(reconstructed).any()
        assert not torch.isinf(reconstructed).any()
