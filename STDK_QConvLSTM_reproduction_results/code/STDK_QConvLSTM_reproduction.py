#!/usr/bin/env python3
"""Reproduce the paper's QConvLSTM experiment on its 100 x 500 simulation.

References
----------
Nag, Sun, and Reich (2023), Spatio-temporal DeepKriging for Interpolation
and Probabilistic Forecasting, Spatial Statistics 100773.
https://github.com/pratiknag/Space-Time.DeepKriging

The upstream CONV_LSTM notebook depends on files that are not committed
(`training_data.csv` and `model_real.h5`).  This standalone implementation
therefore reconstructs both published stages:

1. Space-Time.DeepKriging (STDK) fits the first 495 time points and produces
   an 8 x 8 regular neighbourhood around a requested target location.
2. Median, lower, and upper STDK models form quantile-specific local grids.
3. Three quantile ConvLSTMs use the final five historical grids to directly
   forecast all final five times, following the multi-output CONV_LSTM notebook.

The script downloads the two small simulation files from the authors' GitHub
repository when they are not already present locally.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset


RAW_ROOT = "https://raw.githubusercontent.com/pratiknag/Space-Time.DeepKriging/main/synthetic_ds"
LOC_FILE = "LOC_50000_univariate_spacetime_matern_stationary_1"
VALUE_FILE = "Z1_50000_univariate_spacetime_matern_stationary_1"
BUNDLE_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    seed: int = 41
    train_times: int = 495
    horizon: int = 5
    # 50kSimulation-space-time_DeepKriging.ipynb forecasting: n_steps = 5.
    lookback: int = 5
    grid_size: int = 8
    neighbourhood_radius: float = 0.2
    target_index: int = 0
    stdk_epochs: int = 350
    stdk_batch_size: int = 512
    stdk_lr: float = 1e-3
    stdk_patience: int = 30
    # Keras activity_regularizer=regularizers.l2(1e-5) on Dense layers 4 and 5.
    stdk_activity_l2: float = 1e-5
    stdk_regularization_profile: str = "github_dense4_5_kernel_bias_activity"
    qconv_epochs: int = 25
    qconv_batch_size: int = 5
    qconv_lr: float = 1e-3
    qconv_patience: int = 5
    conv_filters: int = 64
    # Default to the three-block architecture actually released in
    # CONV_LSTM.ipynb.  The simulation-specific 5-step quantile coupling is
    # still a reconstruction because its training_data.csv/model_real.h5
    # bridge was not published.
    qconv_profile: str = "github_3block"
    # Author notebooks use different validation fractions:
    # 50K STDK: train_test_split(..., test_size=0.1)
    # CONV_LSTM: 95% training / 5% validation.
    stdk_validation_fraction: float = 0.10
    qconv_validation_fraction: float = 0.05
    add_paper_nonstationary_mean: bool = True
    stdk_profile: str = "repository"
    validation_mode: str = "repository_random"
    normalization: str = "none"
    grid_boundary: str = "shift"
    forecast_grid_update: str = "none_paper_direct5"
    # Paper Eq. (7): lambda controls the deviation and is chosen proportional
    # to half the training-response range. Set in main after loading the data.
    quantile_lambda: float = 1.0
    # Post-hoc interval calibration chosen only on chronological validation.
    # It rescales q05/q95 deviations around q50 and therefore leaves MSPE unchanged.
    interval_scale: float = 1.0
    forecast_mode: str = "paper_eq7_direct5"
    convlstm_implementation: str = "keras_compatible_pytorch"


def experiment_tag(cfg: Config) -> str:
    qconv_tag = "kerascompat" if cfg.qconv_profile == "github_3block" else "paper3x3"
    tag = f"{cfg.stdk_profile}_{qconv_tag}_papereq7_nsteps{cfg.lookback}"
    if (
        cfg.grid_size != 8
        or not math.isclose(cfg.neighbourhood_radius, 0.2)
        or cfg.grid_boundary != "shift"
    ):
        radius_tag = f"{cfg.neighbourhood_radius:g}".replace(".", "p")
        tag += f"_grid{cfg.grid_size}_radius{radius_tag}_{cfg.grid_boundary}"
    if cfg.train_times != 495:
        tag += f"_train{cfg.train_times}_validate{cfg.horizon}"
    if not math.isclose(cfg.interval_scale, 1.0):
        scale_tag = f"{cfg.interval_scale:g}".replace(".", "p")
        tag += f"_intervalscale{scale_tag}"
    return tag


def parameter_provenance(cfg: Config) -> dict[str, dict[str, str]]:
    """Record which settings are published and which remain reproduction assumptions."""
    recurrent_note = (
        "OFFICIAL_REPO"
        if cfg.qconv_profile == "github_3block"
        else "INFERRED_RECURRENT_REALIZATION"
    )
    return {
        "simulation_LOC_Z1": {"source": "OFFICIAL_REPO", "value": "synthetic_ds stationary_1"},
        "nonstationary_mean": {"source": "PAPER", "value": "Nag et al. Section 3.2"},
        "train_test_split": {
            "source": "PAPER" if cfg.train_times == 495 else "VALIDATION_DESIGN",
            "value": (
                "495 observed / 5 future"
                if cfg.train_times == 495
                else f"1--{cfg.train_times} train / {cfg.train_times + 1}--{cfg.train_times + cfg.horizon} chronological validation"
            ),
        },
        "stdk_basis_profile": {
            "source": "PAPER" if cfg.stdk_profile == "paper" else "OFFICIAL_REPO",
            "value": cfg.stdk_profile,
        },
        "stdk_validation_fraction": {"source": "OFFICIAL_REPO", "value": str(cfg.stdk_validation_fraction)},
        "qconv_validation_fraction": {"source": "OFFICIAL_REPO", "value": str(cfg.qconv_validation_fraction)},
        "quantile_levels": {"source": "PAPER", "value": "0.05, 0.50, 0.95"},
        "quantile_constraint": {"source": "PAPER", "value": "Eq. (7)"},
        "quantile_lambda": {"source": "INFERRED_FROM_PAPER", "value": str(cfg.quantile_lambda)},
        "interval_scale": {
            "source": "CHRONOLOGICAL_VALIDATION_CALIBRATION",
            "value": str(cfg.interval_scale),
        },
        "qconv_profile": {"source": recurrent_note, "value": cfg.qconv_profile},
        "paper_input_convolution": {"source": "PAPER", "value": "valid 3x3, 64 maps"},
        "grid_size": {"source": "INFERRED", "value": str(cfg.grid_size)},
        "neighbourhood_radius": {"source": "INFERRED", "value": str(cfg.neighbourhood_radius)},
        "grid_boundary": {"source": "INFERRED", "value": cfg.grid_boundary},
        "lookback": {"source": "OFFICIAL_REPO_QLSTM", "value": str(cfg.lookback)},
        "forecast_mode": {"source": "INFERRED", "value": cfg.forecast_mode},
        "recursive_forecast": {"source": "NOT_USED", "value": "author QConv future-grid rule unavailable"},
        "test_tuned_lambda": {"source": "NOT_USED", "value": "lambda not selected on final five test times"},
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_data(data_dir: Path) -> tuple[Path, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = (data_dir / LOC_FILE, data_dir / VALUE_FILE)
    for path in paths:
        if not path.exists():
            url = f"{RAW_ROOT}/{path.name}"
            print(f"Downloading {url}")
            urllib.request.urlretrieve(url, path)
    return paths


def load_simulation(data_dir: Path, add_mean: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loc_path, value_path = ensure_data(data_dir)
    locations = np.loadtxt(loc_path, delimiter=",", dtype=np.float32)
    values_flat = np.loadtxt(value_path, dtype=np.float32)
    if locations.shape != (50000, 3) or values_flat.shape != (50000,):
        raise ValueError(f"Unexpected source shapes: locations={locations.shape}, values={values_flat.shape}")

    times = np.unique(locations[:, 2])
    coords = locations[:100, :2].copy()
    if len(times) != 500 or not np.allclose(locations[:, :2], np.tile(coords, (500, 1))):
        raise ValueError("Expected 100 fixed spatial locations repeated over 500 time points")
    values = values_flat.reshape(500, 100)
    if not np.array_equal(times, np.arange(1, 501, dtype=times.dtype)):
        raise AssertionError("Expected ordered source times 1..500")
    if values.shape != (500, 100) or coords.shape != (100, 2):
        raise AssertionError(
            f"Pipeline data shapes are invalid: values={values.shape}, coords={coords.shape}"
        )

    if add_mean:
        # Space-Time.DeepKriging/GpGp_example.R:
        # nonstat_z = z + 2*sin(15*(t/1000-.9))*cos(-37*(t/1000-.9)^4)
        #             + (t/1000-.9)/2.
        u = times / 1000.0 - 0.9
        mean_t = 2.0 * np.sin(15.0 * u) * np.cos(-37.0 * u**4) + u / 2.0
        values = values + mean_t[:, None].astype(np.float32)
    return coords, times.astype(np.float32), values.astype(np.float32)


def profile_basis_settings(profile: str):
    if profile == "repository":
        # Space-Time.DeepKriging/50kSimulation-space-time_DeepKriging.ipynb
        return (5, 9, 11), (70, 250, 410), (0.2, 0.09, 0.009)
    if profile == "paper":
        # Nag et al. (2023), Section 3.1: G=25,81,144 and H=10,15,45.
        return (5, 9, 12), (10, 15, 45), None
    raise ValueError(f"Unknown STDK profile: {profile}")


def wendland_basis(coords: np.ndarray, resolutions, profile: str) -> np.ndarray:
    outputs = []
    for resolution in resolutions:
        axis = np.linspace(0.0, 1.0, resolution, dtype=np.float32)
        xx, yy = np.meshgrid(axis, axis)
        knots = np.column_stack((xx.ravel(), yy.ravel()))
        # The notebook uses 2.5/m. The paper defines bandwidth as 2.5 times
        # adjacent-anchor spacing; linspace(0, 1, m) has spacing 1/(m-1).
        theta = 2.5 / (resolution - 1 if profile == "paper" else resolution)
        d = np.linalg.norm(coords[:, None, :] - knots[None, :, :], axis=2) / theta
        outputs.append(np.where(d <= 1.0, (1.0 - d) ** 6 * (35.0 * d**2 + 18.0 * d + 3.0) / 3.0, 0.0))
    return np.concatenate(outputs, axis=1).astype(np.float32)


def temporal_basis(times_normalized: np.ndarray, counts, std_values) -> np.ndarray:
    outputs = []
    for level, count in enumerate(counts):
        knots = np.linspace(0.0, 1.0, count, dtype=np.float32)
        # Repository uses explicit std values; paper says adjacent-knot spacing.
        kappa = float(knots[1] - knots[0]) if std_values is None else float(std_values[level])
        outputs.append(np.exp(-0.5 * ((times_normalized[:, None] - knots[None, :]) / kappa) ** 2))
    return np.concatenate(outputs, axis=1).astype(np.float32)


def make_stdk_features(coords: np.ndarray, time_indices: np.ndarray, total_times: int, profile: str) -> np.ndarray:
    spatial_resolutions, temporal_counts, temporal_std = profile_basis_settings(profile)
    spatial = wendland_basis(coords, spatial_resolutions, profile)
    # Both original notebooks encode time as 1..K divided by K.
    time_normalized = (time_indices.astype(np.float32) + 1.0) / float(total_times)
    temporal = temporal_basis(time_normalized, temporal_counts, temporal_std)
    return np.concatenate(
        (np.tile(temporal[:, None, :], (1, len(coords), 1)), np.tile(spatial[None, :, :], (len(time_indices), 1, 1))),
        axis=2,
    ).reshape(len(time_indices) * len(coords), -1).astype(np.float32)


class STDK(nn.Module):
    """Paper architecture: 8 x 100 nodes, 4 x 50 nodes, scalar output."""

    def __init__(self, input_dim: int):
        super().__init__()
        dims = [input_dim] + [100] * 8 + [50] * 4 + [1]
        layers = []
        self.regularized_linears = []
        for i in range(len(dims) - 1):
            linear = nn.Linear(dims[i], dims[i + 1])
            # The 50K notebook explicitly requests He-uniform only on layer 1;
            # Keras Dense defaults to Glorot-uniform for the remaining layers.
            if i == 0:
                nn.init.kaiming_uniform_(linear.weight, nonlinearity="relu")
            else:
                nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            if i in (3, 4):
                self.regularized_linears.append(linear)
            layers.append(linear)
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def forward_with_activity_penalty(
        self, x: torch.Tensor, activity_l2: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return prediction and Keras-equivalent Dense activity penalty."""
        penalty = x.new_zeros(())
        regularize_next_activation = False
        for layer in self.network:
            x = layer(x)
            if isinstance(layer, nn.Linear):
                regularize_next_activation = any(
                    layer is candidate for candidate in self.regularized_linears
                )
            elif isinstance(layer, nn.ReLU) and regularize_next_activation:
                # Keras divides an activity-regularizer loss by the input batch
                # size so its relative strength does not change with batch size.
                penalty = penalty + float(activity_l2) * x.square().sum() / x.shape[0]
                regularize_next_activation = False
        return x, penalty


def pinball(prediction: torch.Tensor, target: torch.Tensor, quantile: float) -> torch.Tensor:
    error = target - prediction
    return torch.maximum(quantile * error, (quantile - 1.0) * error).mean()


def fit_stdk(features: np.ndarray, targets: np.ndarray, cfg: Config, device: torch.device) -> STDK:
    count = len(features)
    n_val = max(1, int(cfg.stdk_validation_fraction * count))
    if cfg.validation_mode == "chronological":
        train_idx, val_idx = np.arange(0, count - n_val), np.arange(count - n_val, count)
    else:
        generator = np.random.default_rng(cfg.seed)
        order = generator.permutation(count)
        val_idx, train_idx = order[:n_val], order[n_val:]
    train_ds = TensorDataset(torch.from_numpy(features[train_idx]), torch.from_numpy(targets[train_idx, None]))
    val_x = torch.from_numpy(features[val_idx]).to(device)
    val_y = torch.from_numpy(targets[val_idx, None]).to(device)
    loader = DataLoader(train_ds, batch_size=cfg.stdk_batch_size, shuffle=True)

    model = STDK(features.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.stdk_lr)
    best_loss, best_state, wait = math.inf, None, 0
    for epoch in range(1, cfg.stdk_epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction, activity_penalty = model.forward_with_activity_penalty(
                xb, cfg.stdk_activity_l2
            )
            loss = pinball(prediction, yb, 0.5) + activity_penalty
            # 50K notebook applies L1/L2 kernel and L2 bias penalties to its
            # fourth and fifth 100-unit layers (despite the paper saying first two).
            for layer in model.regularized_linears:
                loss = loss + 1e-5 * layer.weight.abs().sum()
                loss = loss + 1e-4 * layer.weight.square().sum()
                loss = loss + 1e-4 * layer.bias.square().sum()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(pinball(model(val_x), val_y, 0.5))
        if epoch == 1 or epoch % 10 == 0:
            print(f"STDK epoch={epoch:03d} val_pinball={val_loss:.6f}")
        if val_loss < best_loss - 1e-6:
            best_loss, wait = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg.stdk_patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def constrained_quantile(raw: torch.Tensor, median: torch.Tensor, quantile: float, quantile_lambda: float):
    """Paper Eq. (7): median +/- lambda * |tau - 0.5| * sigmoid(raw)."""
    deviation = float(quantile_lambda) * abs(float(quantile) - 0.5) * torch.sigmoid(raw)
    return median - deviation if quantile < 0.5 else median + deviation


def fit_stdk_tail(
    features: np.ndarray,
    targets: np.ndarray,
    median_model: STDK,
    quantile: float,
    cfg: Config,
    device: torch.device,
) -> STDK:
    count = len(features)
    n_val = max(1, int(cfg.stdk_validation_fraction * count))
    if cfg.validation_mode == "chronological":
        train_idx, val_idx = np.arange(0, count - n_val), np.arange(count - n_val, count)
    else:
        order = np.random.default_rng(cfg.seed).permutation(count)
        val_idx, train_idx = order[:n_val], order[n_val:]
    train_ds = TensorDataset(torch.from_numpy(features[train_idx]), torch.from_numpy(targets[train_idx, None]))
    loader = DataLoader(train_ds, batch_size=cfg.stdk_batch_size, shuffle=True)
    val_x = torch.from_numpy(features[val_idx]).to(device)
    val_y = torch.from_numpy(targets[val_idx, None]).to(device)
    raw_model = STDK(features.shape[1]).to(device)
    optimizer = torch.optim.Adam(raw_model.parameters(), lr=cfg.stdk_lr)
    median_model.eval()
    for parameter in median_model.parameters():
        parameter.requires_grad_(False)
    best_loss, best_state, wait = math.inf, None, 0
    for epoch in range(1, cfg.stdk_epochs + 1):
        raw_model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                median = median_model(xb)
            raw, activity_penalty = raw_model.forward_with_activity_penalty(
                xb, cfg.stdk_activity_l2
            )
            prediction = constrained_quantile(raw, median, quantile, cfg.quantile_lambda)
            loss = pinball(prediction, yb, quantile) + activity_penalty
            # The tail networks use the same regularized Dense layers as the
            # median network in the author's 50K notebook.
            for layer in raw_model.regularized_linears:
                loss = loss + 1e-5 * layer.weight.abs().sum()
                loss = loss + 1e-4 * layer.weight.square().sum()
                loss = loss + 1e-4 * layer.bias.square().sum()
            loss.backward()
            optimizer.step()
        raw_model.eval()
        with torch.no_grad():
            val_prediction = constrained_quantile(
                raw_model(val_x), median_model(val_x), quantile, cfg.quantile_lambda
            )
            val_loss = float(pinball(val_prediction, val_y, quantile))
        if epoch == 1 or epoch % 10 == 0:
            print(f"STDK q={quantile:.2f} epoch={epoch:03d} val_pinball={val_loss:.6f}")
        if val_loss < best_loss - 1e-6:
            best_loss, wait = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in raw_model.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg.stdk_patience:
                break
    if best_state is not None:
        raw_model.load_state_dict(best_state)
    for parameter in median_model.parameters():
        parameter.requires_grad_(True)
    return raw_model


def regular_neighbourhood(center: np.ndarray, size: int, radius: float, boundary: str) -> np.ndarray:
    def axis(value):
        lower, upper = float(value - radius), float(value + radius)
        if boundary == "shift":
            if lower < 0.0:
                upper, lower = upper - lower, 0.0
            if upper > 1.0:
                lower, upper = lower - (upper - 1.0), 1.0
        points = np.linspace(lower, upper, size, dtype=np.float32)
        return np.clip(points, 0.0, 1.0) if boundary == "clip" else points

    x, y = axis(center[0]), axis(center[1])
    xx, yy = np.meshgrid(x, y)
    return np.column_stack((xx.ravel(), yy.ravel())).astype(np.float32)


def predict_neighbourhood_frames(
    model: STDK, grid: np.ndarray, n_times: int, device: torch.device, batch_size: int,
    profile: str, total_times: int,
) -> np.ndarray:
    features = make_stdk_features(grid, np.arange(n_times), total_times, profile)
    parts = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            xb = torch.from_numpy(features[start : start + batch_size]).to(device)
            parts.append(model(xb).cpu().numpy().reshape(-1))
    return np.concatenate(parts).reshape(n_times, int(math.sqrt(len(grid))), -1).astype(np.float32)


def predict_stdk_quantile_frames(
    median_model: STDK,
    tail_model: STDK | None,
    quantile: float,
    grid: np.ndarray,
    n_times: int,
    device: torch.device,
    batch_size: int,
    profile: str,
    quantile_lambda: float,
    total_times: int,
) -> np.ndarray:
    median = predict_neighbourhood_frames(
        median_model, grid, n_times, device, batch_size, profile, total_times
    )
    if quantile == 0.5:
        return median
    raw = predict_neighbourhood_frames(
        tail_model, grid, n_times, device, batch_size, profile, total_times
    )
    deviation = (
        float(quantile_lambda) * abs(float(quantile) - 0.5)
        / (1.0 + np.exp(-raw))
    )
    return (median - deviation if quantile < 0.5 else median + deviation).astype(np.float32)


class WindowDataset(Dataset):
    """Historical local grids paired with a direct multi-step target vector."""

    def __init__(
        self, frames: np.ndarray, target: np.ndarray, median_frames: np.ndarray,
        lookback: int, horizon: int, final_start: int,
    ):
        if (
            frames.ndim != 3
            or frames.shape[0] != final_start
            or frames.shape[1] != frames.shape[2]
            or median_frames.shape != frames.shape
        ):
            raise AssertionError(
                f"Expected aligned (time,r,r) frames; frames={frames.shape}, median={median_frames.shape}"
            )
        if target.shape != (final_start,):
            raise AssertionError(f"Expected target shape {(final_start,)}, received {target.shape}")
        if lookback + horizon > final_start:
            raise AssertionError("lookback + horizon exceeds observed training interval")
        self.frames = torch.from_numpy(frames[:, None, :, :])
        self.median_frames = torch.from_numpy(median_frames[:, None, :, :])
        self.target = torch.from_numpy(target)
        self.starts = np.arange(0, final_start - lookback - horizon + 1)
        self.lookback = lookback
        self.horizon = horizon
        last_training_target = int(self.starts[-1] + lookback + horizon - 1)
        if last_training_target >= final_start:
            raise AssertionError(
                f"Training window leaks beyond observed time {final_start - 1}: {last_training_target}"
            )

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int):
        start = int(self.starts[index])
        stop = start + self.lookback
        return (
            self.frames[start:stop],
            self.target[stop : stop + self.horizon],
            self.median_frames[start:stop],
        )


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        # Keras ConvLSTM2D keeps the input and recurrent kernels separate.
        self.input_conv = nn.Conv2d(
            input_channels, 4 * hidden_channels, kernel_size,
            padding=padding, bias=False,
        )
        self.recurrent_conv = nn.Conv2d(
            hidden_channels, 4 * hidden_channels, kernel_size,
            padding=padding, bias=False,
        )
        self.bias = nn.Parameter(torch.zeros(4 * hidden_channels))
        nn.init.xavier_uniform_(self.input_conv.weight)
        # Orthogonal initialization on the flattened recurrent kernel is the
        # PyTorch equivalent of Keras recurrent_initializer="orthogonal".
        nn.init.orthogonal_(self.recurrent_conv.weight)
        # Keras gate order is input, forget, candidate, output; the second
        # block therefore receives unit_forget_bias=True.
        with torch.no_grad():
            self.bias[hidden_channels : 2 * hidden_channels].fill_(1.0)

    def forward(self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]):
        h, c = state
        gates = self.input_conv(x) + self.recurrent_conv(h) + self.bias[None, :, None, None]
        i, f, g, o = gates.chunk(4, dim=1)
        # Keras ConvLSTM2D in the original notebook sets activation="relu"
        # while retaining the default sigmoid recurrent activation.
        i, f, o, g = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o), torch.relu(g)
        c = f * c + i * g
        h = o * torch.relu(c)
        return h, c


class QConvLSTM(nn.Module):
    """CONV_LSTM.ipynb architecture with direct multi-step output."""

    def __init__(self, filters: int, lookback: int, grid_size: int, horizon: int):
        super().__init__()
        # Match CONV_LSTM.ipynb: three 64-filter ConvLSTM blocks with
        # 5x5, 3x3, and 1x1 kernels. The paper specifically describes the
        # spatial 3x3/64-filter convolution; the surrounding blocks come from
        # the authors' released notebook.
        self.cells = nn.ModuleList(
            (
                ConvLSTMCell(1, filters, kernel_size=5),
                ConvLSTMCell(filters, filters, kernel_size=3),
                ConvLSTMCell(filters, filters, kernel_size=1),
            )
        )
        # Original notebook applies BatchNorm after its first two blocks.
        # Keras BatchNormalization(axis=-1) on a returned sequence computes
        # moments over batch, time, height and width. BatchNorm3d on B,C,T,H,W
        # is the exact layout equivalent. Keras momentum=.99 maps to PyTorch
        # momentum=.01 because their update definitions are reversed.
        self.batch_norms = nn.ModuleList(
            (
                nn.BatchNorm3d(filters, eps=1e-3, momentum=0.01),
                nn.BatchNorm3d(filters, eps=1e-3, momentum=0.01),
            )
        )
        # Original notebook flattens the full returned sequence; no pooling.
        self.head = nn.Linear(lookback * filters * grid_size * grid_size, horizon)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        batch, steps, _, height, width = sequence.shape
        layer_sequence = sequence
        for layer, cell in enumerate(self.cells):
            state = (
                sequence.new_zeros((batch, cell.hidden_channels, height, width)),
                sequence.new_zeros((batch, cell.hidden_channels, height, width)),
            )
            outputs = []
            for step in range(steps):
                state = cell(layer_sequence[:, step], state)
                outputs.append(state[0])
            layer_sequence = torch.stack(outputs, dim=1)
            if layer < 2:
                layer_sequence = self.batch_norms[layer](
                    layer_sequence.permute(0, 2, 1, 3, 4)
                ).permute(0, 2, 1, 3, 4)
        return self.head(layer_sequence.flatten(1))


class Paper3x3ConvLSTMCell(nn.Module):
    """Single paper-described valid 3x3/64-map ConvLSTM layer.

    The paper specifies a valid 3x3 input convolution, hence r x r becomes
    (r-2) x (r-2). It does not specify recurrent padding; same padding is the
    minimal recurrent realization that keeps the hidden-state shape valid.
    """

    def __init__(self, input_channels: int, hidden_channels: int):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.input_conv = nn.Conv2d(
            input_channels, 4 * hidden_channels, kernel_size=3,
            padding=0, bias=False,
        )
        self.recurrent_conv = nn.Conv2d(
            hidden_channels, 4 * hidden_channels, kernel_size=3,
            padding=1, bias=False,
        )
        self.bias = nn.Parameter(torch.zeros(4 * hidden_channels))
        nn.init.xavier_uniform_(self.input_conv.weight)
        nn.init.orthogonal_(self.recurrent_conv.weight)
        with torch.no_grad():
            self.bias[hidden_channels : 2 * hidden_channels].fill_(1.0)

    def forward(self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]):
        h, c = state
        gates = self.input_conv(x) + self.recurrent_conv(h) + self.bias[None, :, None, None]
        i, f, g, o = gates.chunk(4, dim=1)
        i, f, o, g = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o), torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


class Paper3x3QConvLSTM(nn.Module):
    """Paper-profile QConvLSTM with one valid 3x3, 64-map layer."""

    def __init__(self, filters: int, lookback: int, grid_size: int, horizon: int):
        super().__init__()
        if grid_size < 3:
            raise ValueError("paper_3x3 requires grid_size >= 3")
        self.cell = Paper3x3ConvLSTMCell(1, filters)
        self.spatial_size = grid_size - 2
        self.head = nn.Linear(
            lookback * filters * self.spatial_size * self.spatial_size, horizon
        )
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        batch, steps, channels, height, width = sequence.shape
        if channels != 1 or height != width:
            raise AssertionError(f"Expected B,T,1,r,r input; received {tuple(sequence.shape)}")
        spatial = height - 2
        state = (
            sequence.new_zeros((batch, self.cell.hidden_channels, spatial, spatial)),
            sequence.new_zeros((batch, self.cell.hidden_channels, spatial, spatial)),
        )
        outputs = []
        for step in range(steps):
            state = self.cell(sequence[:, step], state)
            outputs.append(state[0])
        hidden_sequence = torch.stack(outputs, dim=1)
        expected = (batch, steps, self.cell.hidden_channels, spatial, spatial)
        if tuple(hidden_sequence.shape) != expected:
            raise AssertionError(
                f"paper_3x3 hidden shape {tuple(hidden_sequence.shape)} != {expected}"
            )
        return self.head(hidden_sequence.flatten(1))


def build_qconvlstm(cfg: Config) -> nn.Module:
    if cfg.qconv_profile == "github_3block":
        return QConvLSTM(cfg.conv_filters, cfg.lookback, cfg.grid_size, cfg.horizon)
    if cfg.qconv_profile == "paper_3x3":
        return Paper3x3QConvLSTM(
            cfg.conv_filters, cfg.lookback, cfg.grid_size, cfg.horizon
        )
    raise ValueError(f"Unknown QConvLSTM profile: {cfg.qconv_profile}")


def fit_qconvlstm(
    dataset: WindowDataset, quantile: float, cfg: Config, device: torch.device,
    median_model: nn.Module | None = None,
) -> tuple[nn.Module, dict[str, float | int]]:
    n_val = max(1, int(cfg.qconv_validation_fraction * len(dataset)))
    n_train = len(dataset) - n_val
    if cfg.validation_mode == "chronological":
        train_ds = torch.utils.data.Subset(dataset, range(n_train))
        val_ds = torch.utils.data.Subset(dataset, range(n_train, len(dataset)))
    else:
        train_ds, val_ds = torch.utils.data.random_split(
            dataset, (n_train, n_val), generator=torch.Generator().manual_seed(cfg.seed)
        )
    train_loader = DataLoader(train_ds, batch_size=cfg.qconv_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.qconv_batch_size)
    model = build_qconvlstm(cfg).to(device)
    if median_model is not None:
        median_model.eval()
        for parameter in median_model.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.qconv_lr)
    best_loss, best_epoch, best_state, wait = math.inf, 0, None, 0
    for epoch in range(1, cfg.qconv_epochs + 1):
        model.train()
        for xb, yb, median_xb in train_loader:
            xb, yb, median_xb = xb.to(device), yb.to(device), median_xb.to(device)
            optimizer.zero_grad(set_to_none=True)
            raw = model(xb)
            if median_model is None:
                prediction = raw
            else:
                with torch.no_grad():
                    median = median_model(median_xb)
                prediction = constrained_quantile(raw, median, quantile, cfg.quantile_lambda)
            loss = pinball(prediction, yb, quantile)
            loss.backward()
            optimizer.step()
        model.eval()
        losses = []
        with torch.no_grad():
            for xb, yb, median_xb in val_loader:
                raw = model(xb.to(device))
                prediction = raw if median_model is None else constrained_quantile(
                    raw, median_model(median_xb.to(device)), quantile, cfg.quantile_lambda
                )
                losses.append(float(pinball(prediction, yb.to(device), quantile)))
        val_loss = float(np.mean(losses))
        print(f"QConvLSTM q={quantile:.2f} epoch={epoch:03d} val_pinball={val_loss:.6f}")
        if val_loss < best_loss - 1e-6:
            best_loss, best_epoch, wait = val_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg.qconv_patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    validation_predictions, validation_truth = [], []
    with torch.no_grad():
        for xb, yb, median_xb in val_loader:
            raw = model(xb.to(device))
            prediction = raw if median_model is None else constrained_quantile(
                raw, median_model(median_xb.to(device)), quantile, cfg.quantile_lambda
            )
            validation_predictions.append(prediction.cpu())
            validation_truth.append(yb.cpu())
    val_prediction = torch.cat(validation_predictions)
    val_truth = torch.cat(validation_truth)
    validation_summary = {
        "best_epoch": best_epoch,
        "pinball": float(pinball(val_prediction, val_truth, quantile)),
        "MSPE": float(torch.mean((val_prediction - val_truth) ** 2)),
        "window_count": len(val_ds),
        "prediction_count": int(val_truth.numel()),
    }
    if median_model is not None:
        for parameter in median_model.parameters():
            parameter.requires_grad_(True)
    return model, validation_summary


def evaluate(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    lower, median, upper = prediction[:, 0], prediction[:, 1], prediction[:, 2]
    return {
        "MSPE": float(np.mean((median - truth) ** 2)),
        "RMSE": float(np.sqrt(np.mean((median - truth) ** 2))),
        "MAE": float(np.mean(np.abs(median - truth))),
        "MPIW_90": float(np.mean(upper - lower)),
        "coverage_90": float(np.mean((truth >= lower) & (truth <= upper))),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=BUNDLE_ROOT / "data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to a profile-specific results folder",
    )
    parser.add_argument("--target-index", type=int, default=0, choices=range(100), metavar="0..99")
    parser.add_argument(
        "--target-indices", type=int, nargs="+", choices=range(100), metavar="0..99",
        help="Run a selected station subset for a pilot; compatible results are reused by a later --all-targets run",
    )
    parser.add_argument("--all-targets", action="store_true", help="Run all 100 locations and aggregate Table 2 metrics")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument(
        "--train-times", type=int, default=495,
        help="Observed prefix used for training; the author reproduction uses 495",
    )
    parser.add_argument("--stdk-epochs", type=int, default=350)
    parser.add_argument("--qconv-epochs", type=int, default=25)
    parser.add_argument(
        "--qconv-profile", choices=("github_3block", "paper_3x3"),
        default="github_3block",
        help="Released notebook architecture or the paper's explicit valid 3x3/64-map profile",
    )
    parser.add_argument(
        "--no-stdk-activity-regularization", action="store_true",
        help="Ablation only: disable the GitHub STDK Dense activity L2 penalty",
    )
    parser.add_argument("--no-paper-mean", action="store_true", help="Use repository's stationary values without Section 3.2 mean")
    parser.add_argument("--stdk-profile", choices=("repository", "paper"), default="repository")
    parser.add_argument("--validation-mode", choices=("repository_random", "chronological"), default="repository_random")
    parser.add_argument("--normalization", choices=("none", "train_global"), default="none")
    parser.add_argument(
        "--grid-size", type=int, default=8,
        help="Number of rows and columns in each local spatial grid",
    )
    parser.add_argument(
        "--neighbourhood-radius", type=float, default=0.2,
        help="Half-width of the normalized local spatial neighbourhood",
    )
    parser.add_argument("--grid-boundary", choices=("shift", "clip", "allow_outside"), default="shift")
    parser.add_argument(
        "--quantile-lambda", type=float, default=None,
        help="Paper Eq. (7) lambda; default is half the training-response range",
    )
    parser.add_argument(
        "--interval-scale", type=float, default=1.0,
        help="Validation-calibrated post-hoc scale for q05/q95 deviations around q50",
    )
    parser.add_argument("--overwrite", action="store_true", help="Re-run target JSON files that already exist")
    parser.add_argument("--quick", action="store_true", help="Smoke test with two epochs per stage")
    return parser.parse_args()


def forecast_target(
    target_index: int,
    stdk_models: dict[float, STDK],
    coords: np.ndarray,
    values_model_scale: np.ndarray,
    values_raw: np.ndarray,
    cfg: Config,
    device: torch.device,
    output_dir: Path,
    inverse_scale,
) -> dict:
    seed_everything(cfg.seed + target_index)
    grid = regular_neighbourhood(
        coords[target_index], cfg.grid_size, cfg.neighbourhood_radius, cfg.grid_boundary
    )
    total_times = len(values_raw)
    median_stdk = stdk_models[0.5]
    frames, target_stdk = {}, {}
    for q in (0.05, 0.5, 0.95):
        tail = None if q == 0.5 else stdk_models[q]
        frames[q] = predict_stdk_quantile_frames(
            median_stdk, tail, q, grid, cfg.train_times, device,
            cfg.stdk_batch_size, cfg.stdk_profile, cfg.quantile_lambda,
            total_times,
        )
        target_stdk[q] = predict_stdk_quantile_frames(
            median_stdk, tail, q, coords[target_index : target_index + 1],
            cfg.train_times, device, cfg.stdk_batch_size, cfg.stdk_profile,
            cfg.quantile_lambda, total_times,
        )[:, 0, 0]
        expected_frames = (cfg.train_times, cfg.grid_size, cfg.grid_size)
        if frames[q].shape != expected_frames:
            raise AssertionError(
                f"STDK q={q:.2f} frames {frames[q].shape} != {expected_frames}"
            )
        if target_stdk[q].shape != (cfg.train_times,):
            raise AssertionError(
                f"STDK q={q:.2f} target series shape {target_stdk[q].shape} is invalid"
            )

    datasets = {
        q: WindowDataset(
            frames[q], target_stdk[q], frames[0.5], cfg.lookback,
            cfg.horizon, cfg.train_times,
        )
        for q in (0.05, 0.5, 0.95)
    }
    median_model, median_validation = fit_qconvlstm(
        datasets[0.5], 0.5, cfg, device
    )
    models = {0.5: median_model}
    validation = {0.5: median_validation}
    for q in (0.05, 0.95):
        models[q], validation[q] = fit_qconvlstm(
            datasets[q], q, cfg, device, models[0.5]
        )

    start = cfg.train_times - cfg.lookback
    inputs = {
        q: torch.from_numpy(frames[q][start : cfg.train_times, None, :, :][None]).to(device)
        for q in (0.05, 0.5, 0.95)
    }
    expected_input = (1, cfg.lookback, 1, cfg.grid_size, cfg.grid_size)
    for q, tensor in inputs.items():
        if tuple(tensor.shape) != expected_input:
            raise AssertionError(
                f"QConvLSTM q={q:.2f} input {tuple(tensor.shape)} != {expected_input}"
            )
    with torch.no_grad():
        median_prediction = models[0.5](inputs[0.5])
        lower_prediction = constrained_quantile(
            models[0.05](inputs[0.05]), median_prediction, 0.05,
            cfg.quantile_lambda,
        )
        upper_prediction = constrained_quantile(
            models[0.95](inputs[0.95]), median_prediction, 0.95,
            cfg.quantile_lambda,
        )
    expected_output = (1, cfg.horizon)
    for name, tensor in (
        ("q05", lower_prediction), ("q50", median_prediction), ("q95", upper_prediction)
    ):
        if tuple(tensor.shape) != expected_output:
            raise AssertionError(
                f"QConvLSTM {name} output {tuple(tensor.shape)} != {expected_output}"
            )
    predictions_scaled = torch.stack(
        (lower_prediction, median_prediction, upper_prediction), dim=-1
    ).squeeze(0).cpu().numpy()

    prediction = inverse_scale(np.asarray(predictions_scaled, dtype=np.float32))
    if not math.isclose(cfg.interval_scale, 1.0):
        median = prediction[:, 1].copy()
        prediction[:, 0] = median - cfg.interval_scale * (median - prediction[:, 0])
        prediction[:, 2] = median + cfg.interval_scale * (prediction[:, 2] - median)
    truth = values_raw[cfg.train_times : cfg.train_times + cfg.horizon, target_index]
    if truth.shape != (cfg.horizon,):
        raise AssertionError(f"Forecast truth shape {truth.shape} is invalid")
    if not np.all(prediction[:, 0] <= prediction[:, 1]) or not np.all(
        prediction[:, 1] <= prediction[:, 2]
    ):
        raise AssertionError("Quantile crossing detected after inverse scaling")
    metrics = evaluate(prediction, truth)
    rows = [
        {
            "target_index": target_index,
            "lead": lead + 1,
            "time_index": cfg.train_times + lead + 1,
            "truth": float(truth[lead]),
            "q05": float(prediction[lead, 0]),
            "q50": float(prediction[lead, 1]),
            "q95": float(prediction[lead, 2]),
        }
        for lead in range(cfg.horizon)
    ]
    report = {
        "model": "STDK + QConvLSTM",
        "source": "Space-Time.DeepKriging notebooks plus explicitly recorded reproduction assumptions",
        "config": asdict(cfg),
        "parameter_provenance": parameter_provenance(cfg),
        "target_index": target_index,
        "target_coordinate": coords[target_index].tolist(),
        "evaluation": {
            "role": "final_test" if cfg.train_times == 495 else "chronological_validation",
            "train_time_indices": [1, cfg.train_times],
            "forecast_time_indices": [
                cfg.train_times + 1, cfg.train_times + cfg.horizon
            ],
            "truth_source": "original simulation observations",
        },
        "metrics": metrics,
        "validation_metrics": {
            "q50_MSPE_model_scale": validation[0.5]["MSPE"],
            "q50_pinball_model_scale": validation[0.5]["pinball"],
            "q05_pinball_model_scale": validation[0.05]["pinball"],
            "q95_pinball_model_scale": validation[0.95]["pinball"],
            "details": {f"q{int(q * 100):02d}": validation[q] for q in (0.05, 0.5, 0.95)},
            "note": "Internal early-stopping validation against the STDK-reconstructed training target; use evaluation.role=chronological_validation metrics for end-to-end model selection.",
        },
        "forecasts": rows,
        "assumptions": {
            "quantiles": "paper Eq. (7): q50 +/- lambda*abs(q-0.5)*sigmoid(raw); no post-hoc sorting",
            "interval_calibration": f"q05/q95 deviations around q50 multiplied by {cfg.interval_scale}; q50 is unchanged",
            "quantile_grids": "separate STDK q05/q50/q95 local grids",
            "future_grid": "none; the final five historical grids directly produce five forecasts",
            "grid_boundary": cfg.grid_boundary,
            "se_definition": "unknown in public sources; multiple candidates reported in aggregate JSON",
        },
    }
    stem = f"qconvlstm_target{target_index:03d}_seed{cfg.seed}_{experiment_tag(cfg)}"
    (output_dir / f"{stem}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (output_dir / f"{stem}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    torch.save(
        {f"q{int(q * 100):02d}": model.state_dict() for q, model in models.items()},
        output_dir / f"{stem}.pt",
    )
    del frames, datasets, models
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


def aggregate_reports(reports: list[dict], cfg: Config, output_dir: Path) -> dict:
    forecasts = [row for report in reports for row in report["forecasts"]]
    truth = np.array([row["truth"] for row in forecasts])
    q05 = np.array([row["q05"] for row in forecasts])
    q50 = np.array([row["q50"] for row in forecasts])
    q95 = np.array([row["q95"] for row in forecasts])
    squared_error = (truth - q50) ** 2
    width = q95 - q05
    location_mspe = np.array([r["metrics"]["MSPE"] for r in reports])
    location_mpiw = np.array([r["metrics"]["MPIW_90"] for r in reports])
    n_locations = len(reports)
    metrics = {
        "MSPE": float(squared_error.mean()),
        "MPIW": float(width.mean()),
        "Coverage_percent": float(100.0 * np.mean((truth >= q05) & (truth <= q95))),
        "MSPE_SD_across_locations": float(location_mspe.std(ddof=1)) if n_locations > 1 else None,
        "MSPE_SE_across_locations": float(location_mspe.std(ddof=1) / np.sqrt(n_locations)) if n_locations > 1 else None,
        "MPIW_SD_across_locations": float(location_mpiw.std(ddof=1)) if n_locations > 1 else None,
        "MPIW_SE_across_locations": float(location_mpiw.std(ddof=1) / np.sqrt(n_locations)) if n_locations > 1 else None,
        "MSPE_SE_across_all_predictions": float(squared_error.std(ddof=1) / np.sqrt(len(squared_error))) if len(squared_error) > 1 else None,
        "MPIW_SE_across_all_predictions": float(width.std(ddof=1) / np.sqrt(len(width))) if len(width) > 1 else None,
    }
    validation_mspes = np.array([
        report["validation_metrics"]["q50_MSPE_model_scale"]
        for report in reports
        if "validation_metrics" in report
    ])
    validation_summary = {
        "q50_MSPE_model_scale_mean_across_locations": (
            float(validation_mspes.mean()) if len(validation_mspes) else None
        ),
        "q50_MSPE_model_scale_SD_across_locations": (
            float(validation_mspes.std(ddof=1)) if len(validation_mspes) > 1 else None
        ),
        "completed_locations_with_validation_metrics": int(len(validation_mspes)),
        "selection_rule": "Internal early-stopping diagnostic only; compare end-to-end profiles with evaluation.role=chronological_validation metrics against original observations.",
    }
    paper = {"MSPE": 0.267, "MSPE_SE": 0.219, "MPIW": 1.462, "MPIW_SE": 0.126, "Coverage_percent": 90.39}
    is_final_test = cfg.train_times == 495
    aggregate = {
        "completed_locations": n_locations,
        "prediction_count": len(forecasts),
        "evaluation": {
            "role": "final_test" if is_final_test else "chronological_validation",
            "train_time_indices": [1, cfg.train_times],
            "forecast_time_indices": [
                cfg.train_times + 1, cfg.train_times + cfg.horizon
            ],
            "truth_source": "original simulation observations",
        },
        "metrics": metrics,
        "validation_summary": validation_summary,
        "paper_table2": paper,
        "differences_for_unambiguous_metrics": (
            {
                "MSPE": metrics["MSPE"] - paper["MSPE"],
                "MPIW": metrics["MPIW"] - paper["MPIW"],
                "Coverage_percent": metrics["Coverage_percent"] - paper["Coverage_percent"],
            }
            if is_final_test else None
        ),
        "paper_comparison_note": (
            "Valid only for final times 496--500."
            if is_final_test
            else "Not computed: this output is chronological validation, not the paper's final test period."
        ),
        "se_warning": "The paper/repository do not identify the SE aggregation; candidate SD/SE definitions are all retained above.",
        "config": asdict(cfg),
        "parameter_provenance": parameter_provenance(cfg),
    }
    (output_dir / f"table2_aggregate_seed{cfg.seed}_{experiment_tag(cfg)}.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    with (output_dir / f"table2_forecasts_seed{cfg.seed}_{experiment_tag(cfg)}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=forecasts[0].keys())
        writer.writeheader()
        writer.writerows(forecasts)
    return aggregate


def main() -> None:
    args = parse_args()
    if args.all_targets and args.target_indices is not None:
        raise ValueError("--all-targets and --target-indices cannot be used together")
    if args.target_indices is not None and len(set(args.target_indices)) != len(args.target_indices):
        raise ValueError("--target-indices must be unique")
    if args.grid_size < 1 or (args.qconv_profile == "paper_3x3" and args.grid_size < 3):
        raise ValueError("--grid-size must be >=1 (and >=3 for paper_3x3)")
    if args.neighbourhood_radius <= 0:
        raise ValueError("--neighbourhood-radius must be positive")
    if args.train_times < 2 * 5 or args.train_times + 5 > 500:
        raise ValueError("--train-times must leave one five-step horizon within the 500 time points")
    if args.interval_scale <= 0:
        raise ValueError("--interval-scale must be positive")
    if args.output_dir is None:
        folder = (
            "16_full_paper_profile_pilot"
            if args.qconv_profile == "paper_3x3" and args.stdk_profile == "paper"
            else "22_selected_repository_paper3x3_final"
            if args.qconv_profile == "paper_3x3" and args.stdk_profile == "repository"
            else "06_stdkval10_qconvval05_current"
            if args.qconv_profile == "github_3block"
            else "09_paper_3x3_stdkval10_qconvval05"
        )
        args.output_dir = BUNDLE_ROOT / "raw_results" / folder
    cfg = Config(
        seed=args.seed,
        train_times=args.train_times,
        interval_scale=args.interval_scale,
        target_index=args.target_index,
        stdk_epochs=2 if args.quick else args.stdk_epochs,
        qconv_epochs=2 if args.quick else args.qconv_epochs,
        qconv_profile=args.qconv_profile,
        convlstm_implementation=(
            "keras_compatible_pytorch"
            if args.qconv_profile == "github_3block"
            else "paper_valid3x3_pytorch_inferred_recurrence"
        ),
        stdk_activity_l2=0.0 if args.no_stdk_activity_regularization else 1e-5,
        add_paper_nonstationary_mean=not args.no_paper_mean,
        stdk_profile=args.stdk_profile,
        validation_mode=args.validation_mode,
        normalization=args.normalization,
        grid_size=args.grid_size,
        neighbourhood_radius=args.neighbourhood_radius,
        grid_boundary=args.grid_boundary,
    )
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    coords, _, values_raw = load_simulation(args.data_dir, cfg.add_paper_nonstationary_mean)
    train_time_idx = np.arange(cfg.train_times)

    if cfg.normalization == "train_global":
        y_mean = float(values_raw[: cfg.train_times].mean())
        y_std = float(values_raw[: cfg.train_times].std()) or 1.0
        values_model_scale = ((values_raw - y_mean) / y_std).astype(np.float32)
        inverse_scale = lambda x: x * y_std + y_mean
    else:
        y_mean, y_std = 0.0, 1.0
        values_model_scale = values_raw
        inverse_scale = lambda x: x
    training_range = float(np.ptp(values_model_scale[: cfg.train_times]))
    cfg.quantile_lambda = (
        float(args.quantile_lambda) if args.quantile_lambda is not None
        else training_range / 2.0
    )
    if cfg.quantile_lambda <= 0:
        raise ValueError("quantile lambda must be positive")
    print(
        f"paper Eq.(7) lambda={cfg.quantile_lambda:.6f} "
        f"(training range={training_range:.6f})"
    )
    stdk_x = make_stdk_features(coords, train_time_idx, len(values_raw), cfg.stdk_profile)
    stdk_y = values_model_scale[: cfg.train_times].reshape(-1)

    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stdk_path = args.output_dir / f"stdk_quantiles_seed{cfg.seed}_{cfg.stdk_profile}_kerascompat_papereq7.pt"
    checkpoint = None
    if stdk_path.exists() and not args.overwrite:
        candidate = torch.load(stdk_path, map_location=device, weights_only=False)
        saved_cfg = candidate.get("config", {})
        compatibility_keys = (
            "seed", "stdk_profile", "normalization", "train_times",
            "quantile_lambda", "add_paper_nonstationary_mean",
            "validation_mode", "stdk_validation_fraction", "stdk_activity_l2",
            "stdk_regularization_profile",
            "stdk_epochs", "stdk_batch_size", "stdk_lr", "stdk_patience",
        )
        if all(saved_cfg.get(key) == getattr(cfg, key) for key in compatibility_keys):
            checkpoint = candidate
            print(f"Reusing compatible STDK checkpoint: {stdk_path.name}")
        else:
            print(f"Ignoring incompatible STDK checkpoint: {stdk_path.name}")

    if checkpoint is not None:
        stdk_median = STDK(stdk_x.shape[1]).to(device)
        stdk_lower = STDK(stdk_x.shape[1]).to(device)
        stdk_upper = STDK(stdk_x.shape[1]).to(device)
        stdk_median.load_state_dict(checkpoint["state_dict_q50"])
        stdk_lower.load_state_dict(checkpoint["state_dict_q05_raw"])
        stdk_upper.load_state_dict(checkpoint["state_dict_q95_raw"])
    else:
        print("\n===== STDK q=0.50 =====")
        stdk_median = fit_stdk(stdk_x, stdk_y, cfg, device)
        print("\n===== STDK q=0.05 (median-centred) =====")
        stdk_lower = fit_stdk_tail(stdk_x, stdk_y, stdk_median, 0.05, cfg, device)
        print("\n===== STDK q=0.95 (median-centred) =====")
        stdk_upper = fit_stdk_tail(stdk_x, stdk_y, stdk_median, 0.95, cfg, device)
        torch.save(
            {
                "state_dict_q05_raw": stdk_lower.state_dict(),
                "state_dict_q50": stdk_median.state_dict(),
                "state_dict_q95_raw": stdk_upper.state_dict(),
                "config": asdict(cfg),
                "normalization": {"mean": y_mean, "std": y_std},
            },
            stdk_path,
        )
    stdk_models = {0.05: stdk_lower, 0.5: stdk_median, 0.95: stdk_upper}
    if args.target_indices is not None:
        targets = args.target_indices
    else:
        targets = range(100) if args.all_targets else (cfg.target_index,)
    reports = []
    for target_index in targets:
        stem = f"qconvlstm_target{target_index:03d}_seed{cfg.seed}_{experiment_tag(cfg)}"
        report_path = args.output_dir / f"{stem}.json"
        if report_path.exists() and not args.overwrite:
            saved_report = json.loads(report_path.read_text(encoding="utf-8"))
            saved_cfg = saved_report.get("config", {})
            target_compatibility_keys = (
                "seed", "stdk_profile", "normalization", "train_times",
                "quantile_lambda", "convlstm_implementation", "add_paper_nonstationary_mean",
                "interval_scale",
                "stdk_validation_fraction", "qconv_validation_fraction",
                "stdk_activity_l2", "validation_mode", "grid_size",
                "stdk_regularization_profile",
                "neighbourhood_radius", "grid_boundary", "lookback", "horizon",
                "stdk_epochs", "stdk_batch_size", "stdk_lr", "stdk_patience",
                "qconv_epochs", "qconv_batch_size", "qconv_lr", "qconv_patience",
                "conv_filters", "qconv_profile", "forecast_mode", "forecast_grid_update",
            )
            if all(
                (
                    saved_cfg.get(key, "github_3block")
                    if key == "qconv_profile"
                    else saved_cfg.get(key)
                ) == getattr(cfg, key)
                for key in target_compatibility_keys
            ):
                print(f"Reusing completed target {target_index}: {report_path.name}")
                reports.append(saved_report)
                continue
            print(f"Ignoring incompatible completed target {target_index}: {report_path.name}")
        print(f"\n===== target {target_index}/99 =====")
        reports.append(
            forecast_target(
                target_index, stdk_models, coords, values_model_scale, values_raw, cfg,
                device, args.output_dir, inverse_scale,
            )
        )
    aggregate = aggregate_reports(reports, cfg, args.output_dir)
    aggregate["elapsed_seconds_current_invocation"] = time.time() - started
    (args.output_dir / f"table2_aggregate_seed{cfg.seed}_{experiment_tag(cfg)}.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2))
    print(f"Saved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
