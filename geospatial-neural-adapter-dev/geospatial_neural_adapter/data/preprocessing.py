from typing import Any, Dict, Tuple

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class DataPreprocessor:
    """
    Data preprocessor with normalization and denormalization capabilities.

    This class handles:
    - Feature and target scaling
    - Train/validation/test splits
    - Data conversion to tensors
    - Denormalization for predictions
    """

    def __init__(
        self,
        feature_scaler_type: str = "standard",
        target_scaler_type: str = "standard",
        fit_on_train_only: bool = True,
    ):
        """
        Initialize the preprocessor.

        Args:
            feature_scaler_type: Type of scaler for features ("standard" or "minmax")
            target_scaler_type: Type of scaler for targets ("standard" or "minmax")
            fit_on_train_only: Whether to fit scalers only on training data
        """
        self.feature_scaler_type = feature_scaler_type
        self.target_scaler_type = target_scaler_type
        self.fit_on_train_only = fit_on_train_only

        # Initialize scalers
        if feature_scaler_type == "standard":
            self.feature_scaler = StandardScaler()
        elif feature_scaler_type == "minmax":
            self.feature_scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown feature scaler type: {feature_scaler_type}")

        if target_scaler_type == "standard":
            self.target_scaler = StandardScaler()
        elif target_scaler_type == "minmax":
            self.target_scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown target scaler type: {target_scaler_type}")

        self.is_fitted = False
        self.scaler_stats = {}

    def fit_scalers(self, cont_features: np.ndarray, targets: np.ndarray) -> None:
        """
        Fit scalers on the provided data.

        Args:
            cont_features: Continuous features (T, N, F)
            targets: Target values (T, N)
        """
        # Reshape for fitting
        T, N, F = cont_features.shape
        cont_reshaped = cont_features.reshape(-1, F)
        target_reshaped = targets.reshape(-1, 1)

        # Fit scalers
        self.feature_scaler.fit(cont_reshaped)
        self.target_scaler.fit(target_reshaped)

        # Store statistics
        self.scaler_stats = {
            "feature_mean": self.feature_scaler.mean_
            if hasattr(self.feature_scaler, "mean_")
            else None,
            "feature_scale": self.feature_scaler.scale_
            if hasattr(self.feature_scaler, "scale_")
            else None,
            "feature_min": self.feature_scaler.min_
            if hasattr(self.feature_scaler, "min_")
            else None,
            "feature_max": self.feature_scaler.max_
            if hasattr(self.feature_scaler, "max_")
            else None,
            "target_mean": self.target_scaler.mean_
            if hasattr(self.target_scaler, "mean_")
            else None,
            "target_scale": self.target_scaler.scale_
            if hasattr(self.target_scaler, "scale_")
            else None,
            "target_min": self.target_scaler.min_
            if hasattr(self.target_scaler, "min_")
            else None,
            "target_max": self.target_scaler.max_
            if hasattr(self.target_scaler, "max_")
            else None,
        }

        self.is_fitted = True

    def transform_data(
        self, cont_features: np.ndarray, targets: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform data using fitted scalers.

        Args:
            cont_features: Continuous features (T, N, F)
            targets: Target values (T, N)

        Returns:
            cont_features_scaled: Scaled continuous features (T, N, F)
            targets_scaled: Scaled targets (T, N)
        """
        if not self.is_fitted:
            raise ValueError("Scalers must be fitted before transforming data")

        # Reshape for transformation
        T, N, F = cont_features.shape
        cont_reshaped = cont_features.reshape(-1, F)
        target_reshaped = targets.reshape(-1, 1)

        # Transform
        cont_scaled = self.feature_scaler.transform(cont_reshaped)
        target_scaled = self.target_scaler.transform(target_reshaped)

        # Reshape back
        cont_scaled = cont_scaled.reshape(T, N, F)
        target_scaled = target_scaled.reshape(T, N)

        return cont_scaled, target_scaled

    def inverse_transform_targets(self, targets_scaled: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled targets back to original scale.

        Args:
            targets_scaled: Scaled targets (any shape)

        Returns:
            targets_original: Targets in original scale (same shape)
        """
        if not self.is_fitted:
            raise ValueError("Scalers must be fitted before inverse transforming")

        # Reshape for inverse transformation
        original_shape = targets_scaled.shape
        target_reshaped = targets_scaled.reshape(-1, 1)

        # Inverse transform
        target_original = self.target_scaler.inverse_transform(target_reshaped)

        # Reshape back
        target_original = target_original.reshape(original_shape)

        return target_original

    def get_scaler_info(self) -> Dict[str, Any]:
        """
        Get information about the fitted scalers.

        Returns:
            Dictionary with scaler statistics
        """
        if not self.is_fitted:
            return {"error": "Scalers not fitted yet"}

        return self.scaler_stats


def prepare_all(
    cat_features: np.ndarray,
    cont_features: np.ndarray,
    targets: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[
    torch.utils.data.TensorDataset,
    torch.utils.data.TensorDataset,
    torch.utils.data.TensorDataset,
]:
    """
    Prepare train/validation/test datasets.

    Args:
        cat_features: Categorical features (T, N, C)
        cont_features: Continuous features (T, N, p)
        targets: Target values (T, N)
        train_ratio: Training data ratio
        val_ratio: Validation data ratio

    Returns:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
    """
    T = cat_features.shape[0]

    # Split indices
    train_end = int(T * train_ratio)
    val_end = int(T * (train_ratio + val_ratio))

    # Split data
    train_cat = cat_features[:train_end]
    train_cont = cont_features[:train_end]
    train_targets = targets[:train_end]

    val_cat = cat_features[train_end:val_end]
    val_cont = cont_features[train_end:val_end]
    val_targets = targets[train_end:val_end]

    test_cat = cat_features[val_end:]
    test_cont = cont_features[val_end:]
    test_targets = targets[val_end:]

    # Convert to tensors
    train_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(train_cat).float(),
        torch.from_numpy(train_cont).float(),
        torch.from_numpy(train_targets).float(),
    )

    val_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(val_cat).float(),
        torch.from_numpy(val_cont).float(),
        torch.from_numpy(val_targets).float(),
    )

    test_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(test_cat).float(),
        torch.from_numpy(test_cont).float(),
        torch.from_numpy(test_targets).float(),
    )

    return train_dataset, val_dataset, test_dataset


def prepare_all_with_scaling(
    cat_features: np.ndarray,
    cont_features: np.ndarray,
    targets: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    feature_scaler_type: str = "standard",
    target_scaler_type: str = "standard",
    fit_on_train_only: bool = True,
) -> Tuple[
    torch.utils.data.TensorDataset,
    torch.utils.data.TensorDataset,
    torch.utils.data.TensorDataset,
    DataPreprocessor,
]:
    """
    Prepare train/validation/test datasets with automatic scaling.

    Args:
        cat_features: Categorical features (T, N, C)
        cont_features: Continuous features (T, N, p)
        targets: Target values (T, N)
        train_ratio: Training data ratio
        val_ratio: Validation data ratio
        feature_scaler_type: Type of scaler for features
        target_scaler_type: Type of scaler for targets
        fit_on_train_only: Whether to fit scalers only on training data

    Returns:
        train_dataset: Training dataset (scaled)
        val_dataset: Validation dataset (scaled)
        test_dataset: Test dataset (scaled)
        preprocessor: Fitted preprocessor for denormalization
    """
    T = cat_features.shape[0]

    # Split indices
    train_end = int(T * train_ratio)
    val_end = int(T * (train_ratio + val_ratio))

    # Split data
    train_cat = cat_features[:train_end]
    train_cont = cont_features[:train_end]
    train_targets = targets[:train_end]

    val_cat = cat_features[train_end:val_end]
    val_cont = cont_features[train_end:val_end]
    val_targets = targets[train_end:val_end]

    test_cat = cat_features[val_end:]
    test_cont = cont_features[val_end:]
    test_targets = targets[val_end:]

    # Initialize preprocessor
    preprocessor = DataPreprocessor(
        feature_scaler_type=feature_scaler_type,
        target_scaler_type=target_scaler_type,
        fit_on_train_only=fit_on_train_only,
    )

    # Fit scalers on training data
    preprocessor.fit_scalers(train_cont, train_targets)

    # Transform all datasets
    train_cont_scaled, train_targets_scaled = preprocessor.transform_data(
        train_cont, train_targets
    )
    val_cont_scaled, val_targets_scaled = preprocessor.transform_data(
        val_cont, val_targets
    )
    test_cont_scaled, test_targets_scaled = preprocessor.transform_data(
        test_cont, test_targets
    )

    # Convert to tensors
    train_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(train_cat).float(),
        torch.from_numpy(train_cont_scaled).float(),
        torch.from_numpy(train_targets_scaled).float(),
    )

    val_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(val_cat).float(),
        torch.from_numpy(val_cont_scaled).float(),
        torch.from_numpy(val_targets_scaled).float(),
    )

    test_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(test_cat).float(),
        torch.from_numpy(test_cont_scaled).float(),
        torch.from_numpy(test_targets_scaled).float(),
    )

    return train_dataset, val_dataset, test_dataset, preprocessor


def denormalize_predictions(
    predictions_scaled: np.ndarray, preprocessor: DataPreprocessor
) -> np.ndarray:
    """
    Denormalize scaled predictions back to original scale.

    Args:
        predictions_scaled: Scaled predictions (any shape)
        preprocessor: Fitted preprocessor

    Returns:
        predictions_original: Predictions in original scale (same shape)
    """
    return preprocessor.inverse_transform_targets(predictions_scaled)
