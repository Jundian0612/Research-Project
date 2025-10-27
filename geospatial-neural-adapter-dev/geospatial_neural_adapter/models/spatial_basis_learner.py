from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from geospatial_neural_adapter.logger import setup_logger

logger = setup_logger("spatial_basis_learner")


def compute_smoothing_loss(
    embedding: torch.Tensor, lambda_smooth: float, omega: torch.Tensor = None
) -> torch.Tensor:
    """Compute smoothing loss for spatial basis."""
    if omega is None:
        omega = torch.eye(
            embedding.size(0), device=embedding.device, dtype=embedding.dtype
        )
    elif isinstance(omega, np.ndarray):
        omega = torch.tensor(omega, device=embedding.device, dtype=embedding.dtype)

    assert omega.size(0) == omega.size(1), "Omega must be square (p x p)."
    assert omega.size(0) == embedding.size(0), (
        f"Omega's dimensions ({omega.size(0)}x{omega.size(1)}) must match "
        f"the number of rows in embedding ({embedding.size(0)})."
    )

    return lambda_smooth * torch.sum(embedding * (omega @ embedding))


def compute_lasso_loss(embedding: torch.Tensor, lambda_l1: float) -> torch.Tensor:
    """Compute L1 regularization loss."""
    return lambda_l1 * torch.sum(torch.abs(embedding))


def compute_reconstruction_loss(
    residuals: torch.Tensor,
    reconstructed: torch.Tensor,
) -> torch.Tensor:
    """Compute reconstruction loss using Frobenius norm."""
    frob_norm_sq = torch.norm(residuals - reconstructed, p="fro") ** 2
    mse_loss = frob_norm_sq / residuals.numel()
    return mse_loss


class SpatialBasisLearner(nn.Module):
    """
    Learns an orthonormal spatial basis Φ ∈ ℝ^{N×K} by explicit SVD‐retraction.
    All ops (init, retraction, projection) happen on the same device as `self.basis`.
    """

    def __init__(
        self,
        num_locations: int,
        latent_dim: int,
        pca_init: Optional[np.ndarray] = None,
    ):
        super().__init__()
        if latent_dim > num_locations:
            raise ValueError("latent_dim must be ≤ num_locations")

        # 1) Initialize a raw basis matrix (on CPU for now)
        if pca_init is not None:
            if pca_init.shape != (num_locations, latent_dim):
                raise ValueError("pca_init must have shape (N, K)")
            B = torch.tensor(pca_init, dtype=torch.float32)
        else:
            B = torch.empty(num_locations, latent_dim, dtype=torch.float32)
            nn.init.orthogonal_(B)  # random orthonormal init

        # 2) Store as a plain Parameter.
        # Lightning will later move this to GPU when you call `trainer.fit(model,…)`
        self.basis = nn.Parameter(B)

    @torch.no_grad()
    def _retract(self):
        """
        Re-project `self.basis` onto the Stiefel manifold (orthonormal columns)
        via a thin SVD. This all stays on the current device (CPU or CUDA).
        """
        # M is the same device as self.basis
        M = self.basis
        # economic SVD: U is [N×K] with orthonormal columns
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        # overwrite in-place so the optimizer sees the change
        self.basis.copy_(U)

    def forward(self, residuals: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: project residuals onto the spatial basis.

        Args:
            residuals: Input residuals [batch×N]

        Returns:
            Reconstructed residuals [batch×N]
        """
        # 1) re-orthonormalize before use
        self._retract()

        # 2) form projector P = Φ Φᵀ
        P = self.basis @ self.basis.T  # [N×N] on same device

        # 3) apply to residuals
        return residuals @ P  # [batch×N] @ [N×N] → [batch×N]

    def get_basis(self) -> torch.Tensor:
        """Get the current spatial basis."""
        return self.basis

    def project(self, data: torch.Tensor) -> torch.Tensor:
        """
        Project data onto the spatial basis.

        Args:
            data: Data to project (batch_size, N)

        Returns:
            Projected data (batch_size, K)
        """
        return data @ self.basis

    def reconstruct(self, coefficients: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct data from spatial coefficients.

        Args:
            coefficients: Spatial coefficients (batch_size, K)

        Returns:
            Reconstructed data (batch_size, N)
        """
        return coefficients @ self.basis.T

    @torch.no_grad()
    def reinit_from_pca(self, pca_basis: Union[np.ndarray, torch.Tensor]):
        """
        Overwrite with a new PCA basis (e.g. at end of warm-up),
        then immediately re-orthonormalize on the current device.
        """
        if isinstance(pca_basis, np.ndarray):
            pca_basis = torch.from_numpy(pca_basis).to(self.basis.device).float()
        else:
            pca_basis = pca_basis.to(self.basis.device).float()

        if pca_basis.shape != self.basis.shape:
            raise ValueError("Shape mismatch when re-initialising basis")

        # copy & retract in-place
        self.basis.copy_(pca_basis)
        self._retract()


def train_spatial_basis(
    model: SpatialBasisLearner,
    targets: torch.Tensor,
    epochs: int = 10,
    lr: float = 1e-3,
    omega: torch.Tensor = None,
    tau1: float = 1e-4,
    tau2: float = 1e-4,
    verbose: bool = False,
):
    """
    Train the spatial basis learner with denoising capabilities.

    Args:
        model: SpatialBasisLearner instance
        targets: Target data to denoise [batch×N]
        epochs: Number of training epochs
        lr: Learning rate
        omega: Smoothing matrix (optional)
        tau1: Smoothing regularization weight
        tau2: L1 regularization weight
        verbose: Whether to print training progress
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()

        reconstructed = model(targets)
        Phi = model.basis

        reconstruction_loss = compute_reconstruction_loss(
            targets,
            reconstructed,
        )
        smoothing_loss = compute_smoothing_loss(Phi, lambda_smooth=tau1, omega=omega)
        lasso_loss = compute_lasso_loss(Phi, lambda_l1=tau2)
        total_loss = reconstruction_loss + smoothing_loss + lasso_loss

        total_loss.backward()
        optimizer.step()

        if verbose:
            logger.info(
                f"Epoch {epoch + 1}/{epochs}, Total Loss: {total_loss.item():.6f}, "
                f"Reconstruction Loss: {reconstruction_loss.item():.6f}, "
                f"Smoothing Loss: {smoothing_loss.item():.6f}, "
                f"Lasso Loss: {lasso_loss.item():.6f}"
            )

    return total_loss.item()
