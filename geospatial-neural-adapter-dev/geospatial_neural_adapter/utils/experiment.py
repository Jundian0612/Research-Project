import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

from geospatial_neural_adapter.models.spatial_basis_learner import SpatialBasisLearner
from geospatial_neural_adapter.models.spatial_neural_adapter import SpatialNeuralAdapter
from geospatial_neural_adapter.models.trend_model import TrendModel


def log_covariance_and_basis(
    writer: SummaryWriter,
    tag: str,
    step: int,
    trend_best: TrendModel,
    basis_best: SpatialBasisLearner,
    val_cont: torch.Tensor,
    val_y: torch.Tensor,
    locs: np.ndarray,
    config: dict,
    tau1: float,
    tau2: float,
    best_val: float,
) -> None:
    """Log covariance and basis information to tensorboard."""
    # Create a simple trend model for logging
    p_dim = val_cont.shape[-1]
    trend_model = TrendModel(
        num_continuous_features=p_dim,
        hidden_layer_sizes=[256, 64],
        n_locations=config["n_locations"],
        init_weight=None,
        init_bias=None,
        freeze_init=False,
        dropout_rate=0.1,
    )

    # Log the basis
    basis_data = basis_best.basis.detach().cpu().numpy()
    writer.add_histogram(f"{tag}/basis_hist", basis_data, step)

    # Log some basic statistics
    writer.add_scalar(f"{tag}/basis_norm", np.linalg.norm(basis_data), step)
    writer.add_scalar(f"{tag}/best_val", best_val, step)
    writer.add_scalar(f"{tag}/tau1", tau1, step)
    writer.add_scalar(f"{tag}/tau2", tau2, step)

    # Log covariance if available
    try:
        with torch.no_grad():
            trend_pred = trend_best(val_cont)
            residuals = val_y - trend_pred
            residuals_np = residuals.cpu().numpy()

            # Compute empirical covariance
            emp_cov = residuals_np.T @ residuals_np / residuals_np.shape[0]

            # Log covariance statistics
            writer.add_scalar(f"{tag}/emp_cov_trace", np.trace(emp_cov), step)
            writer.add_scalar(f"{tag}/emp_cov_norm", np.linalg.norm(emp_cov), step)

    except Exception as e:
        print(f"Warning: Could not compute covariance for logging: {e}")


def sweep_tau(
    train_loader,
    val_cont,
    val_y,
    locs,
    config,
    device,
    writer,
    tau1_list,
    tau2_list,
    pretrain_epochs: int = 5,
    *,  # ⇢ force keyword use for new flags
    log_tag_base: str = "TauSweep",
    log_heatmap: bool = True,
):
    """
    Two–level warm-started sweep over tau1 and tau2.
    Returns a DataFrame with columns [tau1, tau2, rmse].
    """
    records = []
    prev_trend_tau1, prev_basis_tau1 = None, None
    step = 0

    for tau1 in tau1_list:
        prev_trend_tau2, prev_basis_tau2 = prev_trend_tau1, prev_basis_tau1
        for tau2 in tau2_list:
            print(f"→ grid τ₁={tau1:.2g}, τ₂={tau2:.2g} …", end="")
            trend = TrendModel(
                num_continuous_features=val_cont.shape[-1],
                hidden_layer_sizes=config["hidden_layer_sizes"],
                n_locations=config["n_locations"],
                init_weight=config.get("w_ols"),
                init_bias=config.get("b_ols"),
                dropout_rate=config.get("dropout_rate", 0.1),
            ).to(device)
            basis = SpatialBasisLearner(
                num_locations=config["n_locations"],
                latent_dim=config["latent_dim"],
            ).to(device)
            if prev_trend_tau2 is not None:
                trend.load_state_dict(prev_trend_tau2)
            if prev_basis_tau2 is not None:
                basis.load_state_dict(prev_basis_tau2)
            trainer = SpatialNeuralAdapter(
                trend=trend,
                basis=basis,
                train_loader=train_loader,
                val_cont=val_cont,
                val_y=val_y,
                locs=locs,
                config=config,
                device=device,
                writer=writer,
                tau1=tau1,
                tau2=tau2,
            )
            if prev_basis_tau2 is None:
                trainer.init_basis_dense()
            trainer.pretrain_trend(epochs=pretrain_epochs)
            best_rmse: float = trainer.run()
            print(f" RMSE={best_rmse:.4f}")
            writer.add_scalar(
                f"{log_tag_base}/RMSE/tau1={tau1:.2g}_tau2={tau2:.2g}",
                best_rmse,
                step,
            )
            log_covariance_and_basis(
                writer=writer,
                tag=f"{log_tag_base}/Cov/tau1={tau1:.2g}_tau2={tau2:.2g}",
                step=step,
                trend_best=trainer.trend,
                basis_best=trainer.basis,
                val_cont=val_cont,
                val_y=val_y,
                locs=locs,
                config=config,
                tau1=tau1,
                tau2=tau2,
                best_val=best_rmse,
            )
            prev_trend_tau2 = trainer.trend.state_dict()
            prev_basis_tau2 = trainer.basis.state_dict()
            if tau2 == 0.0:
                prev_trend_tau1 = prev_trend_tau2
                prev_basis_tau1 = prev_basis_tau2
            records.append({"tau1": tau1, "tau2": tau2, "rmse": best_rmse})
            step += 1
    df = pd.DataFrame(records)
    if log_heatmap:
        plot_tau_heatmap(
            writer,
            df,
            tau1_list=tau1_list,
            tau2_list=tau2_list,
            tag=f"{log_tag_base}/Heatmap",
            global_step=step,
        )
    return df


def plot_tau_heatmap(
    writer: SummaryWriter,
    df,
    tau1_list,
    tau2_list,
    tag: str = "tau_heatmap",
    global_step: int = 0,
):
    """
    Build a heatmap from `df` (with columns tau1, tau2, rmse)
    and record it into TensorBoard under `tag` at `global_step`.
    """
    pivot = df.pivot(index="tau1", columns="tau2", values="rmse")
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(
        pivot.values,
        origin="lower",
        extent=[min(tau2_list), max(tau2_list), min(tau1_list), max(tau1_list)],
        aspect="auto",
    )
    ax.set_xlabel("τ₂")
    ax.set_ylabel("τ₁")
    ax.set_title("Validation RMSE")
    fig.colorbar(im, ax=ax, label="RMSE")
    plt.tight_layout()
    writer.add_figure(tag, fig, global_step=global_step)
    plt.close(fig)
