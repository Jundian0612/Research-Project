from typing import Callable, Optional

import torch
import torch.nn as nn


class PretrainedTrendModel(nn.Module):
    """
    Wrapper for integrating pretrained models into the Spatial Adapter framework.

    This wrapper allows you to use any pretrained model as the backbone f_θ
    in the spatial adapter equation: Y(t,s) = g(f_θ(x(t,s)) + Φ(s)^T η(t)) + ε
    """

    def __init__(
        self,
        pretrained_model: nn.Module,
        input_shape: tuple,
        output_shape: tuple,
        freeze_backbone: bool = True,
        add_residual_head: bool = True,
        residual_hidden_dim: int = 128,
        dropout_rate: float = 0.1,
        input_adapter: Optional[Callable] = None,
        output_adapter: Optional[Callable] = None,
    ):
        """
        Args:
            pretrained_model: The pretrained model to wrap
            input_shape: Expected input shape (batch_size, ..., num_features)
            output_shape: Expected output shape (batch_size, num_locations)
            freeze_backbone: Whether to freeze the pretrained model weights
            add_residual_head: Whether to add a trainable residual head
            residual_hidden_dim: Hidden dimension for the residual head
            dropout_rate: Dropout rate for the residual head
            input_adapter: Optional function to transform inputs before backbone
            output_adapter: Optional function to transform outputs after backbone
        """
        super().__init__()

        self.pretrained_model = pretrained_model
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.freeze_backbone = freeze_backbone
        self.input_adapter = input_adapter
        self.output_adapter = output_adapter

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.pretrained_model.parameters():
                param.requires_grad = False

        # Add residual head for fine-tuning
        if add_residual_head:
            # Determine the output dimension of the backbone
            backbone_out_dim = self._get_backbone_output_dim()
            target_dim = output_shape[-1]  # num_locations

            self.residual_head = nn.Sequential(
                nn.Linear(backbone_out_dim, residual_hidden_dim),
                nn.LayerNorm(residual_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(residual_hidden_dim, target_dim),
            )

            # Zero-initialize the residual head
            for m in self.residual_head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        else:
            self.residual_head = None

    def _get_backbone_output_dim(self) -> int:
        """Determine output dimension of the backbone model."""
        # Create a dummy input to determine output shape
        dummy_input = torch.randn(1, *self.input_shape[1:])

        self.pretrained_model.eval()
        with torch.no_grad():
            try:
                output = self.pretrained_model(dummy_input)
                if isinstance(output, tuple):
                    output = output[0]  # Take first element if tuple
                result: int = int(output.shape[-1])
                return result
            except Exception as e:
                raise ValueError(f"Could not determine backbone output dimension: {e}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the pretrained model + optional residual head.

        Args:
            x: Input tensor of shape (batch_size, ..., num_features)

        Returns:
            Output tensor of shape (batch_size, num_locations)
        """
        x.shape[0]

        # Apply input adapter if provided
        if self.input_adapter:
            x = self.input_adapter(x)

        # Reshape for backbone if needed
        if len(x.shape) == 3:  # (batch_size, num_locations, num_features)
            original_shape = x.shape
            x = x.view(-1, x.shape[-1])  # Flatten for backbone
            should_reshape = True
        else:
            should_reshape = False
            original_shape = None

        # Forward through pretrained backbone
        backbone_output = self.pretrained_model(x)

        # Handle different output types
        if isinstance(backbone_output, tuple):
            backbone_output = backbone_output[0]  # Take first element

        # Reshape back if needed
        if should_reshape and original_shape is not None:
            # If we had 3D input, ensure we get proper 2D output (batch_size, num_locations)
            if len(backbone_output.shape) == 1:
                # If backbone outputs 1D, reshape to (batch_size, num_locations)
                backbone_output = backbone_output.view(
                    original_shape[0], original_shape[1]
                )
            elif (
                len(backbone_output.shape) == 2
                and backbone_output.shape[0] == original_shape[0] * original_shape[1]
            ):
                # If backbone outputs (batch_size * num_locations, 1), reshape properly
                backbone_output = backbone_output.view(
                    original_shape[0], original_shape[1]
                )

        # Apply output adapter if provided
        if self.output_adapter:
            backbone_output = self.output_adapter(backbone_output)

        # Add residual head if present
        if self.residual_head is not None:
            # Always pass the backbone output directly to residual head
            # The residual head is designed to match the backbone output shape
            residual = self.residual_head(backbone_output)

            # For spatial adapter, we need output shape (batch_size, num_locations)
            if len(backbone_output.shape) == 3:
                # If backbone outputs (batch_size, num_locations, 1), squeeze last dim
                if backbone_output.shape[-1] == 1:
                    backbone_output = backbone_output.squeeze(-1)
                else:
                    # If backbone outputs (batch_size, num_locations, features), take mean
                    backbone_output = backbone_output.mean(dim=-1)

            output = backbone_output + residual
        else:
            output = backbone_output

        # Ensure output shape matches expected format
        if len(output.shape) == 3 and output.shape[-1] == 1:
            output = output.squeeze(-1)

        return output

    def residual_parameters(self):
        """Return only the trainable parameters (for optimizer)."""
        params = []

        # Add backbone parameters if not frozen
        if not self.freeze_backbone:
            params.extend(self.pretrained_model.parameters())

        # Add residual head parameters if present
        if self.residual_head is not None:
            params.extend(self.residual_head.parameters())

        return [p for p in params if p.requires_grad]

    def unfreeze_backbone(self):
        """Unfreeze the backbone for fine-tuning."""
        self.freeze_backbone = False
        for param in self.pretrained_model.parameters():
            param.requires_grad = True

    def freeze_backbone_layers(self, num_layers: int):
        """Freeze the first num_layers of the backbone."""
        if hasattr(self.pretrained_model, "layers"):
            for i, layer in enumerate(self.pretrained_model.layers):
                if i < num_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
                else:
                    for param in layer.parameters():
                        param.requires_grad = True


def create_pretrained_trend_model(
    pretrained_model: nn.Module,
    input_shape: tuple,
    output_shape: tuple,
    model_type: str = "custom",
    **kwargs,
) -> PretrainedTrendModel:
    """
    Factory function to create a pretrained trend model with common adapters.

    Args:
        pretrained_model: The pretrained model
        input_shape: Expected input shape
        output_shape: Expected output shape
        model_type: Type of model ("tabnet", "transformer", "mlp", "custom")
        **kwargs: Additional arguments for PretrainedTrendModel

    Returns:
        PretrainedTrendModel instance
    """

    # Define common input/output adapters
    input_adapters = {
        "tabnet": lambda x: x.view(-1, x.shape[-1]),
        "transformer": lambda x: x,
        "mlp": lambda x: x.view(-1, x.shape[-1]),
    }

    output_adapters = {
        "tabnet": lambda x: x.squeeze(-1) if len(x.shape) > 2 else x,
        "transformer": lambda x: x.mean(dim=1) if len(x.shape) > 2 else x,
        "mlp": lambda x: x,
    }

    # Set adapters based on model type
    if model_type in input_adapters:
        kwargs.setdefault("input_adapter", input_adapters[model_type])
    if model_type in output_adapters:
        kwargs.setdefault("output_adapter", output_adapters[model_type])

    return PretrainedTrendModel(
        pretrained_model=pretrained_model,
        input_shape=input_shape,
        output_shape=output_shape,
        **kwargs,
    )
