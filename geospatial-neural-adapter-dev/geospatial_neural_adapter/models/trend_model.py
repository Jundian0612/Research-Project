from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim


class TrendModel(nn.Module):
    """
    Trend = frozen OLS in the first Linear + zero-init residual MLP on top.
    """

    def __init__(
        self,
        num_continuous_features: int,
        hidden_layer_sizes: list[int],
        n_locations: int,
        init_weight: Optional[torch.Tensor] = None,
        init_bias: Optional[float] = None,
        freeze_init: bool = True,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        F = num_continuous_features

        # 1) Initial linear layer - output single value per input
        self.init_lin = nn.Linear(F, 1, bias=True)
        if init_weight is not None and init_bias is not None:
            with torch.no_grad():
                if init_weight.dim() == 2:
                    w_global = init_weight.mean(dim=0)
                else:
                    w_global = init_weight
                self.init_lin.weight.copy_(w_global.view(1, F))
                self.init_lin.bias.copy_(torch.tensor([init_bias]))
            if freeze_init:
                for p in self.init_lin.parameters():
                    p.requires_grad = False

        # 2) Residual MLP
        if hidden_layer_sizes:
            # Simplified residual blocks for better learning
            blocks = []
            for i, hidden_size in enumerate(hidden_layer_sizes):
                if i == 0:
                    # First layer: F -> hidden_size
                    blocks.extend(
                        [
                            nn.Linear(F, hidden_size, bias=True),
                            nn.LayerNorm(hidden_size),
                            nn.GELU(),
                            nn.Dropout(dropout_rate),
                        ]
                    )
                else:
                    # Subsequent layers: previous_size -> current_size
                    prev_size = hidden_layer_sizes[i - 1]
                    blocks.extend(
                        [
                            nn.Linear(prev_size, hidden_size, bias=True),
                            nn.LayerNorm(hidden_size),
                            nn.GELU(),
                            nn.Dropout(dropout_rate),
                        ]
                    )
            self.res_blocks = nn.Sequential(*blocks)
            self.res_out = nn.Linear(hidden_layer_sizes[-1], 1)  # Output single value
        else:
            self.res_blocks = None
            self.res_out = None

        # 3) Proper initialization for residual components
        for m in self.modules():
            if isinstance(m, nn.Linear) and m is not self.init_lin:
                # Use Xavier/Glorot initialization for better gradient flow
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, P, F = x.shape
        flat = x.view(-1, F)
        out = self.init_lin(flat)
        if self.res_blocks is not None and self.res_out is not None:
            # Simplified residual path
            h = self.res_blocks(flat)
            h = self.res_out(h)
            out = out + h
        return out.view(B, -1)

    def _residual_forward(self, h):
        """Residual forward pass for gradient checkpointing."""
        return self.res_blocks(h)

    def residual_parameters(self):
        """
        Return only the trainable (residual-MLP) parameters.
        """
        return [p for p in self.parameters() if p.requires_grad]


def train_trend_model(
    model: nn.Module,
    continuous_data: torch.Tensor,  # Float [T, N, F]
    train_labels: torch.Tensor,  # Float [T, N]
    categorical_data: torch.Tensor | None = None,  # Long [T, N, C] or None / empty
    *,
    num_epochs: int = 10,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    device: str = "cpu",
):
    """
    Generic trainer for the TrendModel.

    • If `categorical_data` is None (or has shape[-1]==0), the model is
      called in *continuous-only* mode:   `model(continuous, None)`.
    • Otherwise, the usual mixed call is used:
      `model(continuous, categorical)`.

    Returns
    -------
    model : nn.Module         (with trained weights on *device*)
    loss_history : list[float]
    """
    import logging

    from torch import optim
    from torch.utils.data import DataLoader, TensorDataset

    LOGGER = logging.getLogger(__name__)

    model = model.to(device)
    continuous_data = continuous_data.to(device)
    train_labels = train_labels.to(device)

    if categorical_data is not None and categorical_data.numel() > 0:
        categorical_data = categorical_data.to(device)
        dataset = TensorDataset(continuous_data, categorical_data, train_labels)
        use_cats = True
    else:
        categorical_data = None
        dataset = TensorDataset(continuous_data, train_labels)  # 2-tuple
        use_cats = False

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loss_history: list[float] = []

    for epoch in range(num_epochs):
        model.train()
        running = 0.0

        for batch in loader:
            if use_cats:
                batch_cont, batch_cat, batch_lbl = batch
                preds = model(batch_cont, batch_cat)  # mixed mode
            else:
                batch_cont, batch_lbl = batch
                preds = model(batch_cont)  # continuous-only mode

            loss = criterion(preds, batch_lbl)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()

        epoch_loss = running / len(loader)
        loss_history.append(epoch_loss)
        LOGGER.info("Epoch %d/%d | loss %.4f", epoch + 1, num_epochs, epoch_loss)

    return model, loss_history


def train_trend_model(
    model: nn.Module,
    train_cont: torch.Tensor,
    train_targets: torch.Tensor,
    val_cont: torch.Tensor,
    val_targets: torch.Tensor,
    num_epochs: int = 100,
    learning_rate: float = 1e-3,
    device: str = "cpu",
    patience: int = 20,
):
    """
    Train the improved model with modern optimization techniques.

    Features:
    - AdamW optimizer with weight decay
    - Cosine annealing learning rate scheduler
    - Gradient clipping for stability
    - Early stopping to prevent overfitting
    """

    model = model.to(device)
    train_cont = train_cont.to(device)
    train_targets = train_targets.to(device)
    val_cont = val_cont.to(device)
    val_targets = val_targets.to(device)

    # Use AdamW optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Use cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Loss function
    criterion = nn.MSELoss()

    # Early stopping
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # Training
        model.train()
        optimizer.zero_grad()

        train_preds = model(train_cont)
        train_loss = criterion(train_preds, train_targets)

        train_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = model(val_cont)
            val_loss = criterion(val_preds, val_targets)

        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        if (epoch + 1) % 20 == 0:
            print(
                f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss.item():.4f} | Val Loss: {val_loss.item():.4f}"
            )

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_losses, val_losses
