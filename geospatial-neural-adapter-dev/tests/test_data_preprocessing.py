"""
Unit tests for geospatial_neural_adapter.data.preprocessing module.
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset

from geospatial_neural_adapter.data.preprocessing import prepare_all


class TestPrepareAll:
    """Test prepare_all function."""

    def test_data_splitting(self):
        """Test that data is correctly split into train/val/test."""
        # Create dummy data
        T, N = 100, 10
        cat = np.zeros((T, N, 1), dtype=np.int64)
        cont = np.random.randn(T, N, 5).astype(np.float64)
        y = np.random.randn(T, N).astype(np.float64)

        train, val, test = prepare_all(cat, cont, y)

        # Check that we get TensorDatasets
        assert isinstance(train, TensorDataset)
        assert isinstance(val, TensorDataset)
        assert isinstance(test, TensorDataset)

        # Check shapes
        train_cat, train_cont, train_y = train.tensors
        val_cat, val_cont, val_y = val.tensors
        test_cat, test_cont, test_y = test.tensors

        # Default ratios: 0.7, 0.15, 0.15
        expected_train = int(T * 0.7)
        expected_val = int(T * 0.15)

        assert train_cat.shape[0] == expected_train
        assert val_cat.shape[0] == expected_val
        assert test_cat.shape[0] == T - expected_train - expected_val

        # Check data types
        assert train_cat.dtype == torch.int64
        assert train_cont.dtype == torch.float32
        assert train_y.dtype == torch.float32

    def test_custom_ratios(self):
        """Test data splitting with custom ratios."""
        T, N = 100, 10
        cat = np.zeros((T, N, 1), dtype=np.int64)
        cont = np.random.randn(T, N, 5).astype(np.float64)
        y = np.random.randn(T, N).astype(np.float64)

        train, val, test = prepare_all(cat, cont, y, train_ratio=0.6, val_ratio=0.2)

        train_cat, _, _ = train.tensors
        val_cat, _, _ = val.tensors
        test_cat, _, _ = test.tensors

        expected_train = int(T * 0.6)
        expected_val = int(T * 0.2)

        assert train_cat.shape[0] == expected_train
        assert val_cat.shape[0] == expected_val
        assert test_cat.shape[0] == T - expected_train - expected_val
