"""
Unit tests for geospatial_neural_adapter.metrics module.
"""

import numpy as np
import torch

from geospatial_neural_adapter.metrics import (
    compute_metrics,
    frobenius_norm,
    fusion_score,
)


class TestComputeMetrics:
    """Test compute_metrics function."""

    def test_compute_metrics_perfect_prediction(self):
        """Test compute_metrics with perfect predictions."""
        y_true = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        y_pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        rmse, mae, r2 = compute_metrics(y_true, y_pred)

        assert rmse == 0.0
        assert mae == 0.0
        assert r2 == 1.0

    def test_compute_metrics_with_error(self):
        """Test compute_metrics with prediction errors."""
        y_true = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        y_pred = torch.tensor([[1.5, 2.5], [2.5, 3.5]])

        rmse, mae, r2 = compute_metrics(y_true, y_pred)

        assert rmse > 0.0
        assert mae > 0.0
        assert r2 < 1.0

    def test_compute_metrics_different_shapes(self):
        """Test compute_metrics with different tensor shapes."""
        y_true = torch.randn(10, 5)
        y_pred = torch.randn(10, 5)

        rmse, mae, r2 = compute_metrics(y_true, y_pred)

        assert isinstance(rmse, float)
        assert isinstance(mae, float)
        assert isinstance(r2, float)
        assert rmse >= 0.0
        assert mae >= 0.0


class TestFusionScore:
    """Test fusion_score function."""

    def test_fusion_score_rmse_only(self):
        """Test fusion_score when only RMSE is provided."""
        rmse = 0.5
        result = fusion_score(rmse, None, None)
        assert result == rmse

    def test_fusion_score_with_projection_gap(self):
        """Test fusion_score with projection gap."""
        rmse = 0.5
        proj_gap = 0.1
        p = 10
        expected = rmse + (proj_gap / p)
        result = fusion_score(rmse, proj_gap, p)
        assert result == expected

    def test_fusion_score_zero_p(self):
        """Test fusion_score when p is zero."""
        rmse = 0.5
        proj_gap = 0.1
        p = 0
        result = fusion_score(rmse, proj_gap, p)
        assert result == rmse


class TestFrobeniusNorm:
    """Test frobenius_norm function."""

    def test_frobenius_norm_identical_matrices(self):
        """Test frobenius_norm with identical matrices."""
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[1, 2], [3, 4]])
        result = frobenius_norm(A, B)
        assert result == 0.0

    def test_frobenius_norm_different_matrices(self):
        """Test frobenius_norm with different matrices."""
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[2, 3], [4, 5]])
        result = frobenius_norm(A, B)
        assert result > 0.0

    def test_frobenius_norm_large_matrices(self):
        """Test frobenius_norm with larger matrices."""
        A = np.random.randn(10, 10)
        B = np.random.randn(10, 10)
        result = frobenius_norm(A, B)
        assert isinstance(result, float)
        assert result >= 0.0
