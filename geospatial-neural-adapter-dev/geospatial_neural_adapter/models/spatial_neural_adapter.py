import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from geospatial_neural_adapter.cpp_extensions import spatial_utils
from geospatial_neural_adapter.logger import setup_logger
from geospatial_neural_adapter.models.spatial_basis_learner import SpatialBasisLearner
from geospatial_neural_adapter.models.trend_model import TrendModel

logger = setup_logger("spatial_neural_adapter")


@dataclass
class ADMMConfig:
    """Configuration for ADMM optimization parameters."""

    rho: float = 5.0  # Base ADMM penalty parameter
    dual_momentum: float = 0.2  # Dual variable momentum
    max_iters: int = 300  # Maximum ADMM iterations
    min_outer: int = 100  # Minimum outer iterations before convergence check
    tol: float = 1e-4  # Convergence tolerance


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""

    lr_mu: float = 1e-3  # Learning rate for trend parameters
    batch_size: int = 128  # Batch size for theta step
    pretrain_epochs: int = 5  # Default pretraining epochs
    use_mixed_precision: bool = False  # Whether to use mixed precision


@dataclass
class BasisConfig:
    """Configuration for spatial basis learning."""

    phi_every: int = 5  # Update basis every N iterations
    phi_freeze: int = 100  # Stop updating basis after N iterations
    matrix_reg: float = 1e-6  # Matrix regularization for basis update
    irl1_max_iters: int = 10  # IRL₁ maximum iterations
    irl1_eps: float = 1e-6  # IRL₁ epsilon
    irl1_tol: float = 5e-4  # IRL₁ inner tolerance


@dataclass
class SpatialNeuralAdapterConfig:
    """Complete configuration for SpatialNeuralAdapter."""

    admm: ADMMConfig = None
    training: TrainingConfig = None
    basis: BasisConfig = None

    def __post_init__(self):
        """Initialize default configs if not provided."""
        if self.admm is None:
            self.admm = ADMMConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.basis is None:
            self.basis = BasisConfig()

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for backward compatibility."""
        config_dict = {}

        # ADMM parameters
        config_dict.update(
            {
                "rho": self.admm.rho,
                "dual_momentum": self.admm.dual_momentum,
                "max_iters": self.admm.max_iters,
                "min_outer": self.admm.min_outer,
                "tol": self.admm.tol,
            }
        )

        # Training parameters
        config_dict.update(
            {
                "lr_mu": self.training.lr_mu,
                "batch_size": self.training.batch_size,
                "pretrain_epochs": self.training.pretrain_epochs,
                "use_mixed_precision": self.training.use_mixed_precision,
            }
        )

        # Basis parameters
        config_dict.update(
            {
                "phi_every": self.basis.phi_every,
                "phi_freeze": self.basis.phi_freeze,
                "matrix_reg": self.basis.matrix_reg,
                "irl1_max_iters": self.basis.irl1_max_iters,
                "irl1_eps": self.basis.irl1_eps,
                "irl1_tol": self.basis.irl1_tol,
            }
        )

        return config_dict

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SpatialNeuralAdapterConfig":
        """Create config from dictionary."""
        admm = ADMMConfig(
            rho=config_dict.get("rho", 5.0),
            dual_momentum=config_dict.get("dual_momentum", 0.2),
            max_iters=config_dict.get("max_iters", 300),
            min_outer=config_dict.get("min_outer", 100),
            tol=config_dict.get("tol", 1e-4),
        )

        training = TrainingConfig(
            lr_mu=config_dict.get("lr_mu", 1e-3),
            batch_size=config_dict.get("batch_size", 128),
            pretrain_epochs=config_dict.get("pretrain_epochs", 5),
            use_mixed_precision=config_dict.get("use_mixed_precision", False),
        )

        basis = BasisConfig(
            phi_every=config_dict.get("phi_every", 5),
            phi_freeze=config_dict.get("phi_freeze", 100),
            matrix_reg=config_dict.get("matrix_reg", 1e-6),
            irl1_max_iters=config_dict.get("irl1_max_iters", 10),
            irl1_eps=config_dict.get("irl1_eps", 1e-6),
            irl1_tol=config_dict.get("irl1_tol", 5e-4),
        )

        return cls(admm=admm, training=training, basis=basis)

    def log_config(self) -> None:
        """Log current configuration."""
        logger.info("SpatialNeuralAdapterConfig:")
        logger.info(f"  ADMM Config:")
        logger.info(f"    rho: {self.admm.rho}")
        logger.info(f"    dual_momentum: {self.admm.dual_momentum}")
        logger.info(f"    max_iters: {self.admm.max_iters}")
        logger.info(f"    min_outer: {self.admm.min_outer}")
        logger.info(f"    tol: {self.admm.tol}")
        logger.info(f"  Training Config:")
        logger.info(f"    lr_mu: {self.training.lr_mu}")
        logger.info(f"    batch_size: {self.training.batch_size}")
        logger.info(f"    pretrain_epochs: {self.training.pretrain_epochs}")
        logger.info(f"    use_mixed_precision: {self.training.use_mixed_precision}")
        logger.info(f"  Basis Config:")
        logger.info(f"    phi_every: {self.basis.phi_every}")
        logger.info(f"    phi_freeze: {self.basis.phi_freeze}")
        logger.info(f"    matrix_reg: {self.basis.matrix_reg}")
        logger.info(f"    irl1_max_iters: {self.basis.irl1_max_iters}")
        logger.info(f"    irl1_eps: {self.basis.irl1_eps}")
        logger.info(f"    irl1_tol: {self.basis.irl1_tol}")


class SpatialNeuralAdapter:
    """
    Simplified ADMM-BCD implementation for academic research:
      θ-step : neural trend (ResMLP)
      Φ-step : spatial basis (PCA / IRL₁ / coordinate)
      Z-step : consensus + low-rank projection
      u      : dual variables

    Args:
        trend: Trend model
        basis: Spatial basis learner
        train_loader: Training data loader
        val_cont: Validation continuous features
        val_y: Validation targets
        locs: Spatial locations
        config: Configuration object or dictionary
        device: Device to run on
        writer: TensorBoard writer
        tau1: First regularization parameter
        tau2: Second regularization parameter
    """

    def __init__(
        self,
        trend: TrendModel,
        basis: SpatialBasisLearner,
        train_loader: DataLoader,
        val_cont: torch.Tensor,
        val_y: torch.Tensor,
        locs: np.ndarray,
        config: Union[SpatialNeuralAdapterConfig, Dict[str, Any]],
        device: torch.device,
        writer: SummaryWriter,
        tau1: float = 0.0,
        tau2: float = 0.0,
    ):
        self.device = device
        self.writer = writer
        self.tau1, self.tau2 = tau1, tau2

        # Convert dict to config if needed
        if isinstance(config, dict):
            self.config = SpatialNeuralAdapterConfig.from_dict(config)
        else:
            self.config = config

        # Simple mixed precision setup (fixed deprecated GradScaler)
        self.use_mixed_precision = self.config.training.use_mixed_precision
        if self.use_mixed_precision:
            self.scaler = torch.amp.GradScaler("cuda")
            logger.info("Mixed precision training enabled")
        else:
            self.scaler = None
            logger.info("Mixed precision training disabled")

        # Data tensors
        self.trend = trend.to(device)
        self.basis = basis.to(device)
        self.train_loader = train_loader
        self.val_cont, self.val_y = val_cont.to(device), val_y.to(device)

        # Get training data
        _, train_cont, train_y = train_loader.dataset.tensors
        self.train_cont, self.train_y = train_cont.to(device), train_y.to(device)

        # ADMM hyper-parameters
        self._rho_base = float(self.config.admm.rho * torch.std(self.train_y).item())
        self.rho = self._rho_base
        self.beta = float(self.config.admm.dual_momentum)
        self.max_iters = int(self.config.admm.max_iters)

        # Smoothing penalty matrix
        omega = spatial_utils.smoothing_penalty_matrix(locs)
        self.omega = torch.as_tensor(omega, dtype=torch.float32, device=device)

        # Consensus and dual variables
        T_train, N = self.train_y.shape
        T_val = self.val_y.shape[0]
        self.z_train = torch.zeros(T_train, N, device=device)
        self.u_train = torch.zeros_like(self.z_train)
        self.z_val = torch.zeros(T_val, N, device=device)
        self.u_val = torch.zeros_like(self.z_val)

        # Optimizer
        self.opt_mu = optim.AdamW(
            self.trend.residual_parameters(), lr=self.config.training.lr_mu
        )

        # Statistics
        self.best_val = float("inf")
        self.global_iter = 0
        self.y_mean = self.train_y.mean().item()
        self.y_std = self.train_y.std(unbiased=False).item() + 1e-12

        # Validate configuration
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        # All validation is now handled by the dataclass structure

    def log_config(self) -> None:
        """Log current configuration."""
        logger.info("SpatialNeuralAdapter Configuration:")
        logger.info(f"  ADMM Config:")
        logger.info(f"    rho: {self.config.admm.rho}")
        logger.info(f"    dual_momentum: {self.config.admm.dual_momentum}")
        logger.info(f"    max_iters: {self.config.admm.max_iters}")
        logger.info(f"    tol: {self.config.admm.tol}")
        logger.info(f"  Training Config:")
        logger.info(f"    lr_mu: {self.config.training.lr_mu}")
        logger.info(f"    batch_size: {self.config.training.batch_size}")
        logger.info(
            f"    use_mixed_precision: {self.config.training.use_mixed_precision}"
        )
        logger.info(f"  Basis Config:")
        logger.info(f"    phi_every: {self.config.basis.phi_every}")
        logger.info(f"    phi_freeze: {self.config.basis.phi_freeze}")
        logger.info(f"    matrix_reg: {self.config.basis.matrix_reg}")

    def pretrain_trend(self, epochs: Optional[int] = None) -> None:
        """Warm-up trend parameters on plain MSE."""
        if epochs is None:
            epochs = int(self.config.training.pretrain_epochs)

        self.trend.train()
        for _ in range(epochs):
            for _, x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                loss = F.mse_loss(self.trend(x), y)
                self.opt_mu.zero_grad()
                loss.backward()
                self.opt_mu.step()

    @torch.no_grad()
    def init_basis_dense(self) -> None:
        """Initialize basis with τ₂ = 0 warm-start."""
        # Residual matrix
        X = torch.cat([xb for _, xb, _ in self.train_loader]).to(self.device)
        Y = torch.cat([yb for _, _, yb in self.train_loader]).to(self.device)
        R = Y - self.trend(X)  # T × N

        C = R.T @ R  # N × N (symmetric)
        M = 0.5 * (C - self.tau1 * self.omega + (C - self.tau1 * self.omega).T)

        # Eigendecomposition
        _, V = torch.linalg.eigh(M)  # ascending eigenvalues
        K = self.basis.basis.shape[1]
        self.basis.basis.data.copy_(V[:, -K:])  # keep top-K

        # Consensus warm-start
        self.z_train.copy_(R)
        self.u_train.zero_()
        self.z_val.copy_(self.val_y - self.trend(self.val_cont))
        self.u_val.zero_()

    def run(self) -> float:
        """Run the ADMM optimization."""
        tol = float(self.config.admm.tol)
        min_outer = int(self.config.admm.min_outer)

        # Progress bar
        outer = trange(1, self.max_iters + 1, desc="ADMM", dynamic_ncols=True)

        for _ in outer:
            z_prev = self.z_train.clone()

            # ADMM steps
            delta_phi = self._phi_step()
            self._theta_step()
            self._z_step(val=False)
            self._z_step(val=True)

            # Residuals
            r_pri = self.train_y - self.trend(self.train_cont)
            pri = F.mse_loss(self.z_train, r_pri).sqrt().item()
            dua = (self.rho * (self.z_train - z_prev)).pow(2).mean().sqrt().item()

            # Progress display
            res = max(delta_phi, pri)
            outer.set_postfix(
                res=f"{res:.2e}",
                pri=f"{pri:.2e}",
                dua=f"{dua:.2e}",
                rho=f"{self.rho:.2e}",
            )

            # Validation and logging
            val_result = self._validate()
            rmse, *_ = val_result
            if self.writer is not None:
                self.writer.add_scalar("val/rmse_admm", rmse, self.global_iter)
                self.writer.add_scalar("train/rho", self.rho, self.global_iter)
                self.writer.add_scalar("train/primal_residual", pri, self.global_iter)
                self.writer.add_scalar("train/dual_residual", dua, self.global_iter)

            self.best_val = min(self.best_val, rmse)

            # Convergence check
            if (self.global_iter + 1) >= min_outer and res < tol:
                outer.write(f"Converged (res={res:.2e} < ε={tol})")
                break

            self.global_iter += 1

        logger.info(f"Training completed in {self.global_iter} iterations")
        return self.best_val

    def _theta_loss(self, yb, xb, zb, ub):
        """Compute theta-step loss."""
        sigma = self.y_std
        r = (yb - self.trend(xb)) / sigma  # residual
        cons = (zb + ub) / sigma  # consensus target
        return 0.5 * self.rho * F.mse_loss(cons, r)

    def _theta_step(self) -> None:
        """Update trend parameters."""
        batch_size = self.config.training.batch_size
        self.trend.train()

        T = self.train_y.size(0)
        perm = torch.randperm(T, device=self.device)

        for start in range(0, T, batch_size):
            idx = perm[start : start + batch_size]

            xb = self.train_cont[idx]
            yb = self.train_y[idx]
            zb = self.z_train[idx]
            ub = self.u_train[idx]

            loss = self._theta_loss(yb, xb, zb, ub)
            self.opt_mu.zero_grad()

            if self.use_mixed_precision and self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.opt_mu)
                self.scaler.update()
            else:
                loss.backward()
                self.opt_mu.step()

    @torch.no_grad()
    def _phi_step(self) -> float:
        """Update spatial basis."""
        phi_every = int(self.config.basis.phi_every)
        freeze_after = int(self.config.basis.phi_freeze)

        if (self.global_iter % phi_every) or (self.global_iter >= freeze_after):
            return 0.0

        K = self.basis.basis.shape[1]
        old = self.basis.basis.data.clone()

        # Centered data
        Zc = self.z_train - self.z_train.mean(0, keepdim=True)

        # Covariance matrix
        C = Zc.T @ Zc
        M = 0.5 * (C - self.tau1 * self.omega + (C - self.tau1 * self.omega).T)

        # Add regularization
        matrix_reg = float(self.config.basis.matrix_reg)
        M += (
            matrix_reg
            * torch.trace(M).item()
            / M.size(0)
            * torch.eye(M.size(0), device=M.device)
        )

        if self.tau2 == 0.0:  # PCA
            _, V = torch.linalg.eigh(M)
            new_phi = V[:, -K:]
        else:  # IRL₁ surrogate
            _, V = torch.linalg.eigh(M)
            Phi = V[:, -K:]  # warm start
            alpha = 1.0 / (2.0 * torch.linalg.norm(M, 2))
            Lmax = int(self.config.basis.irl1_max_iters)
            eps_irls = float(self.config.basis.irl1_eps)
            tol_inner = float(self.config.basis.irl1_tol)

            for t in range(Lmax):
                Phi_prev = Phi.clone()
                G = 2 * (M @ Phi)
                Y = Phi + alpha * G
                W = 1.0 / (Phi.abs() + eps_irls)
                Phi = torch.sign(Y) * torch.clamp(Y.abs() - alpha * self.tau2 * W, 0.0)
                U, _, Vt = torch.linalg.svd(Phi, full_matrices=False)
                Phi = U @ Vt
                if t >= 2 and torch.norm(Phi - Phi_prev) < tol_inner:
                    break
            new_phi = Phi

        # Update basis
        self.basis.basis.data.copy_(new_phi)
        norm_value = float(torch.norm(new_phi - old, p="fro").item())
        return norm_value

    @torch.no_grad()
    def _z_step(self, *, val: bool = False) -> None:
        """Update consensus variables."""
        if val:
            R = self.val_y - self.trend(self.val_cont)
            z, u = self.z_val, self.u_val
        else:
            R = self.train_y - self.trend(self.train_cont)
            z, u = self.z_train, self.u_train

        # Woodbury coefficients
        a1 = self.rho / (self.rho + 2.0)
        a2 = 2.0 / (self.rho + 2.0)

        Phi = self.basis.basis
        Res = R - u

        # Projection
        V = Res @ Phi
        PRes = V @ Phi.T

        z.copy_(a1 * Res + a2 * PRes)

        # Dual update with momentum
        u_prev = u.clone()
        u.add_(z - R)
        if self.beta:
            u.add_(self.beta * (u - u_prev))

    @torch.no_grad()
    def _validate(self) -> Tuple[float, float, float]:
        """Compute validation metrics."""
        self.trend.eval()
        self.basis.eval()

        mu = self.trend(self.val_cont)
        basis_proj = (self.z_val @ self.basis.basis) @ self.basis.basis.T
        y_hat = mu + basis_proj

        rmse = math.sqrt(F.mse_loss(y_hat, self.val_y).item())
        return rmse, 0.0, rmse

    @torch.no_grad()
    def predict(
        self, cont_features: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict full output by summing trend prediction plus basis projection
        of residual (y_true - trend_pred).

        Args:
            cont_features: Input features for trend (batch_size × p)
            y_true: True targets (batch_size × N), needed to compute residual

        Returns:
            y_pred: Full prediction (batch_size × N)
        """
        self.trend.eval()
        self.basis.eval()

        trend_pred = self.trend(cont_features)  # shape: (batch_size, N)
        residual = y_true - trend_pred  # residual: (batch_size, N)

        # Project residual into basis subspace
        basis_proj = (
            residual @ self.basis.basis
        ) @ self.basis.basis.T  # (batch_size, N)

        y_pred = trend_pred + basis_proj
        return y_pred
