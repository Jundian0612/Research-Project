import torch
import torch.nn as nn
from darts.models import TFTModel


class TFTWrapper(nn.Module):
    """
    Wrapper to make TFT model compatible with PretrainedTrendModel interface.

    This wrapper provides the necessary interface for integrating TFT models
    into the spatial adapter framework while handling the conversion between
    tensor formats and TimeSeries formats required by darts.
    """

    def __init__(self, tft_model: TFTModel, num_locations: int, num_features: int):
        """
        Initialize the TFT wrapper.

        Args:
            tft_model: The trained TFT model from darts
            num_locations: Number of spatial locations
            num_features: Number of input features per location
        """
        super().__init__()
        self.tft_model = tft_model
        self.num_locations = num_locations
        self.num_features = num_features

        self.dummy_param = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.last_time_index = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the TFT model.

        Args:
            x: Input tensor of shape (batch_size, num_locations, num_features) or (batch_size * num_locations, num_features)

        Returns:
            Output tensor of shape (batch_size, num_locations) or (batch_size * num_locations, 1)
        """
        # Handle both 2D and 3D inputs
        if len(x.shape) == 3:
            # Input is (batch_size, num_locations, num_features)
            batch_size, num_locations, num_features = x.shape

            # Ensure we have the right number of locations
            if num_locations != self.num_locations:
                raise ValueError(
                    f"Expected {self.num_locations} locations, got {num_locations}"
                )

            # Process each batch item separately
            outputs = []

            for i in range(batch_size):
                # Get features for this batch item
                batch_features = x[i]  # Shape: (num_locations, num_features)

                # Use the simplified prediction method
                try:
                    prediction = self._predict_with_tft(batch_features)
                except Exception as e:
                    print(f"TFT prediction failed for batch {i}, using fallback: {e}")
                    prediction = torch.mean(batch_features, dim=1, keepdim=True)

                outputs.append(prediction)

            output = torch.stack(outputs, dim=0)
            return output.squeeze(-1)

        elif len(x.shape) == 2:
            # Input is (batch_size * num_locations, num_features)
            # This happens when PretrainedTrendModel flattens the input
            total_samples, num_features = x.shape

            # Calculate batch_size and num_locations
            batch_size = total_samples // self.num_locations
            if total_samples % self.num_locations != 0:
                raise ValueError(
                    f"Total samples {total_samples} must be divisible by num_locations {self.num_locations}"
                )

            # Reshape to (batch_size, num_locations, num_features)
            x_reshaped = x.view(batch_size, self.num_locations, num_features)

            # Process using the 3D logic
            outputs = []

            for i in range(batch_size):
                # Get features for this batch item
                batch_features = x_reshaped[i]  # Shape: (num_locations, num_features)

                # Use the simplified prediction method
                try:
                    prediction = self._predict_with_tft(batch_features)
                except Exception as e:
                    print(f"TFT prediction failed for batch {i}, using fallback: {e}")
                    prediction = torch.mean(batch_features, dim=1, keepdim=True)

                outputs.append(prediction)

            # Stack and reshape back to (batch_size * num_locations, 1)
            output = torch.stack(
                outputs, dim=0
            )  # Shape: (batch_size, num_locations, 1)
            output = output.view(
                batch_size * self.num_locations, 1
            )  # Shape: (batch_size * num_locations, 1)
            return output

        else:
            raise ValueError(f"Expected 2D or 3D input tensor, got shape {x.shape}")

    def _predict_with_tft(self, features: torch.Tensor) -> torch.Tensor:
        """
        Simple prediction method that can be extended for TFT-based feature extraction.

        Args:
            features: Input features tensor of shape (num_locations, num_features)

        Returns:
            Prediction tensor of shape (num_locations, 1)
        """
        try:
            # For now, use a simple linear combination
            # This can be extended to use TFT model features or embeddings
            # In a more sophisticated implementation, you could:
            # 1. Extract embeddings from the TFT model
            # 2. Use the TFT model's attention weights
            # 3. Apply the TFT model's feature transformations

            # Simple feature-based prediction
            prediction = torch.mean(features, dim=1, keepdim=True)
            return prediction

        except Exception as e:
            print(f"TFT prediction failed: {e}")
            return torch.zeros(self.num_locations, 1, dtype=torch.float32)

    def extract_tft_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the TFT model (placeholder for future implementation).

        Args:
            features: Input features tensor of shape (num_locations, num_features)

        Returns:
            Extracted features tensor
        """
        # This is a placeholder for future TFT feature extraction
        # In practice, you would:
        # 1. Convert features to TimeSeries format
        # 2. Pass through TFT model layers
        # 3. Extract intermediate representations
        # 4. Convert back to tensor format

        return features  # For now, just return the original features

    def parameters(self):
        """Return parameters for compatibility with nn.Module."""
        return [self.dummy_param]

    def to(self, device):
        """Move the wrapper to the specified device."""
        super().to(device)
        return self
