"""
Unit tests for geospatial_neural_adapter.experiment module.
"""

from unittest.mock import Mock

import numpy as np
import pytest
import torch
from torch.utils.tensorboard import SummaryWriter

from geospatial_neural_adapter.experiment import (
    log_covariance_and_basis,
    plot_tau_heatmap,
)
from geospatial_neural_adapter.models.spatial_basis_learner import SpatialBasisLearner
from geospatial_neural_adapter.models.trend_model import TrendModel


class TestLogCovarianceAndBasis:
    """Test log_covariance_and_basis function."""

    @pytest.fixture
    def mock_writer(self):
        """Create a mock TensorBoard writer."""
        return Mock(spec=SummaryWriter)

    @pytest.fixture
    def sample_models(self):
        """Create sample trend and basis models."""
        trend = TrendModel(
            num_continuous_features=3,
            hidden_layer_sizes=[16],
            w_ols=torch.randn(3),
            b_ols=1.0,
            dropout_rate=0.1,
        )
        # Use latent_dim=1 to avoid shape mismatch
        basis = SpatialBasisLearner(num_locations=5, latent_dim=1)
        return trend, basis

    @pytest.fixture
    def sample_data(self):
        """Create sample validation data."""
        val_cont = torch.randn(10, 5, 3)  # (T, N, features)
        val_y = torch.randn(10, 5)  # (T, N)
        locs = np.linspace(-3, 3, 5)
        config = {
            "eigenvalue": 4.0,
            "noise_std": 0.1,
        }
        return val_cont, val_y, locs, config

    def test_log_covariance_and_basis_basic(
        self, mock_writer, sample_models, sample_data
    ):
        """Test basic functionality of log_covariance_and_basis."""
        trend, basis = sample_models
        val_cont, val_y, locs, config = sample_data

        # Call the function
        log_covariance_and_basis(
            writer=mock_writer,
            tag="test_tag",
            step=0,
            trend_best=trend,
            basis_best=basis,
            val_cont=val_cont,
            val_y=val_y,
            locs=locs,
            config=config,
            tau1=1.0,
            tau2=0.5,
            best_val=0.1,
        )

        # Check that writer methods were called
        assert mock_writer.add_image.called
        # Should be called 3 times: overall covariances, spatial covariances, basis
        assert mock_writer.add_image.call_count == 3

    def test_log_covariance_and_basis_different_parameters(
        self, mock_writer, sample_models, sample_data
    ):
        """Test with different tau parameters."""
        trend, basis = sample_models
        val_cont, val_y, locs, config = sample_data

        log_covariance_and_basis(
            writer=mock_writer,
            tag="test_tag",
            step=1,
            trend_best=trend,
            basis_best=basis,
            val_cont=val_cont,
            val_y=val_y,
            locs=locs,
            config=config,
            tau1=0.0,
            tau2=0.0,
            best_val=0.05,
        )

        assert mock_writer.add_image.called

    def test_log_covariance_and_basis_device_handling(
        self, mock_writer, sample_models, sample_data
    ):
        """Test that function works with different devices."""
        trend, basis = sample_models
        val_cont, val_y, locs, config = sample_data

        # Move models to CPU explicitly
        trend = trend.cpu()
        basis = basis.cpu()
        val_cont = val_cont.cpu()
        val_y = val_y.cpu()

        log_covariance_and_basis(
            writer=mock_writer,
            tag="test_tag",
            step=0,
            trend_best=trend,
            basis_best=basis,
            val_cont=val_cont,
            val_y=val_y,
            locs=locs,
            config=config,
            tau1=1.0,
            tau2=0.5,
            best_val=0.1,
        )

        assert mock_writer.add_image.called


class TestPlotTauHeatmap:
    """Test plot_tau_heatmap function."""

    @pytest.fixture
    def mock_writer(self):
        """Create a mock TensorBoard writer."""
        return Mock(spec=SummaryWriter)

    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame for testing."""
        import pandas as pd

        data = {
            "tau1": [0.1, 0.1, 1.0, 1.0],
            "tau2": [0.1, 1.0, 0.1, 1.0],
            "rmse": [0.5, 0.3, 0.4, 0.2],
        }
        return pd.DataFrame(data)

    def test_plot_tau_heatmap_basic(self, mock_writer, sample_dataframe):
        """Test basic functionality of plot_tau_heatmap."""
        tau1_list = [0.1, 1.0]
        tau2_list = [0.1, 1.0]

        plot_tau_heatmap(
            writer=mock_writer,
            df=sample_dataframe,
            tau1_list=tau1_list,
            tau2_list=tau2_list,
            tag="test_heatmap",
            global_step=0,
        )

        # Check that writer method was called
        assert mock_writer.add_figure.called
        assert mock_writer.add_figure.call_count == 1

    def test_plot_tau_heatmap_different_sizes(self, mock_writer):
        """Test with different list sizes."""
        import pandas as pd

        # Create larger dataset
        tau1_vals = [0.1, 0.5, 1.0]
        tau2_vals = [0.1, 0.5, 1.0]
        data = {
            "tau1": [t1 for t1 in tau1_vals for t2 in tau2_vals],
            "tau2": [t2 for t1 in tau1_vals for t2 in tau2_vals],
            "rmse": np.random.rand(len(tau1_vals) * len(tau2_vals)),
        }
        df = pd.DataFrame(data)

        plot_tau_heatmap(
            writer=mock_writer,
            df=df,
            tau1_list=tau1_vals,
            tau2_list=tau2_vals,
            tag="test_heatmap",
            global_step=1,
        )

        assert mock_writer.add_figure.called

    def test_plot_tau_heatmap_custom_tag(self, mock_writer, sample_dataframe):
        """Test with custom tag."""
        tau1_list = [0.1, 1.0]
        tau2_list = [0.1, 1.0]

        plot_tau_heatmap(
            writer=mock_writer,
            df=sample_dataframe,
            tau1_list=tau1_list,
            tau2_list=tau2_list,
            tag="custom_heatmap_tag",
            global_step=5,
        )

        # Check that the correct tag was used
        call_args = mock_writer.add_figure.call_args
        assert call_args[0][0] == "custom_heatmap_tag"
        # Check that global_step is passed correctly as keyword argument
        assert call_args[1]["global_step"] == 5  # global_step


class TestExperimentIntegration:
    """Integration tests for experiment functions."""

    def test_log_covariance_and_basis_with_real_models(self):
        """Test with real model instances."""
        # Create real models
        trend = TrendModel(
            num_continuous_features=2,
            hidden_layer_sizes=[8],
            w_ols=torch.randn(2),
            b_ols=0.5,
            dropout_rate=0.1,
        )
        # Use latent_dim=1 to avoid shape issues
        basis = SpatialBasisLearner(num_locations=4, latent_dim=1)

        # Create real data
        val_cont = torch.randn(5, 4, 2)
        val_y = torch.randn(5, 4)
        locs = np.linspace(-2, 2, 4)
        config = {"eigenvalue": 2.0, "noise_std": 0.1}

        # Create mock writer
        mock_writer = Mock(spec=SummaryWriter)

        # Test the function
        log_covariance_and_basis(
            writer=mock_writer,
            tag="integration_test",
            step=0,
            trend_best=trend,
            basis_best=basis,
            val_cont=val_cont,
            val_y=val_y,
            locs=locs,
            config=config,
            tau1=1.0,
            tau2=0.5,
            best_val=0.15,
        )

        # Verify it completed without errors
        assert mock_writer.add_image.called
