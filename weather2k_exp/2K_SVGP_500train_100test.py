#!/usr/bin/env python3
"""SVGP baseline trained on 500 stations and evaluated on 100 held-out stations."""

import ast
import contextlib
import copy
import gc
import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset

try:
    import gpytorch
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "This script requires gpytorch. Install it in the SVGP environment before running."
    ) from exc

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):
        return iterable


# ============================================================
# Weather2K loader
# ============================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
cwd = os.getcwd()

candidate_paths = [
    os.environ.get("WEATHER2K_NPY", ""),
    os.path.join(script_dir, "../Josh's Weather2K/Weather2K/weather2k.npy"),
    os.path.join(script_dir, "../../Josh's Weather2K/Weather2K/weather2k.npy"),
    os.path.join(cwd, "Josh's Weather2K/Weather2K/weather2k.npy"),
    os.path.join(cwd, "../Josh's Weather2K/Weather2K/weather2k.npy"),
]

dataset_path = None
for candidate_path in candidate_paths:
    if not candidate_path:
        continue
    absolute_path = os.path.abspath(candidate_path)
    if os.path.isfile(absolute_path):
        dataset_path = absolute_path
        break

if dataset_path is None:
    raise FileNotFoundError(f"Dataset not found in any of: {candidate_paths}")

print(f"Dataset found: {dataset_path}")

data = np.load(dataset_path)
data = np.transpose(data, (0, 2, 1))

loc = data[:, 0, :3]
data = data[:, :, 3:]

date_start = datetime(2017, 1, 1)
date_end = datetime(2021, 8, 31, 23, 59)

time_list = []
current = date_start
while current <= date_end:
    time_list.append(current)
    current += timedelta(hours=3)

time_index = pd.to_datetime(np.array(time_list))

nloc = data.shape[0]
ntime = data.shape[1]
nvar = data.shape[2]

weather_var_names = [
    "air_pressure",
    "air_temperature",
    "relative_humidity",
    "wind_speed",
    "wind_direction",
    "precipitation",
    "solar_radiation",
    "dew_point_temperature",
    "cloud_cover",
    "visibility",
]

if len(weather_var_names) != nvar:
    raise ValueError(
        f"Variable name count = {len(weather_var_names)}, but Weather2K has {nvar} variables."
    )

target_var_name = "air_temperature"
if target_var_name not in weather_var_names:
    raise ValueError(f"{target_var_name} not found in weather_var_names")

var_idx = weather_var_names.index(target_var_name)
y_all = data[:, :, var_idx].astype(np.float32)

lat = loc[:, 0]
lon = loc[:, 1]
gg = np.column_stack([lon, lat]).astype(np.float32)

print(f"Using target variable: {target_var_name}")


# ============================================================
# Utilities
# ============================================================
def _fmt_time(s):
    m, sec = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h:d}h{m:02d}m{sec:02d}s"


def _json_dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _cleanup_torch_cache() -> None:
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _silent_call(func, *args, **kwargs):
    with open(os.devnull, "w") as fnull:
        with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
            return func(*args, **kwargs)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return {
        "RMSE": float(np.sqrt(mse)),
        "MSE": float(mse),
        "MAE": float(mae),
        "R2": float(
            r2_score(
                np.asarray(y_true).reshape(-1),
                np.asarray(y_pred).reshape(-1),
            )
        ),
    }


class FeatureDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return {"X": self.X[idx], "y": self.y[idx]}


class FeatureOnlyDataset(Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return {"X": self.X[idx]}


def collate_train(batch):
    return {
        "X": torch.stack([b["X"] for b in batch]),
        "y": torch.stack([b["y"] for b in batch]),
    }


def collate_predict(batch):
    return {"X": torch.stack([b["X"] for b in batch])}


def build_flat_inputs(y_matrix, coords, time_idx_subset, t_norm_all):
    coords = np.asarray(coords, dtype=np.float32)
    time_idx_subset = np.asarray(time_idx_subset, dtype=int)
    y_matrix = np.asarray(y_matrix, dtype=np.float32)
    t_values = np.asarray(t_norm_all[time_idx_subset], dtype=np.float32).reshape(-1, 1)

    n_sites = coords.shape[0]
    n_times = len(time_idx_subset)

    coords_flat = np.tile(coords, (n_times, 1)).astype(np.float32)
    t_flat = np.repeat(t_values, n_sites, axis=0).astype(np.float32)
    y_flat = y_matrix[:, time_idx_subset].T.reshape(-1).astype(np.float32)
    X_flat = np.concatenate([coords_flat, t_flat], axis=1).astype(np.float32)

    return X_flat, y_flat


def normalize_coords(coords):
    coords = np.asarray(coords, dtype=np.float32)
    min_vals = coords.min(axis=0)
    max_vals = coords.max(axis=0)
    denom = np.where((max_vals - min_vals) < 1e-12, 1.0, max_vals - min_vals)
    return ((coords - min_vals) / denom).astype(np.float32)


def build_svgp_config():
    return {
        # Defaults use the validation-selected SVGP configuration from
        # svgp_tuning_current (seeds 41 and 42; test metrics were not used).
        "epochs": int(os.environ.get("SVGP_EPOCHS", "500")),
        "lr": float(os.environ.get("SVGP_LR", "1e-3")),
        "weight_decay": float(os.environ.get("SVGP_WEIGHT_DECAY", "0.0")),
        "batch_size": int(os.environ.get("SVGP_BATCH_SIZE", "4096")),
        "patience": int(os.environ.get("SVGP_PATIENCE", "30")),
        "num_inducing": int(os.environ.get("SVGP_NUM_INDUCING", "1024")),
        "init_noise": float(os.environ.get("SVGP_INIT_NOISE", "0.1")),
        "variational_jitter": float(
            os.environ.get("SVGP_VARIATIONAL_JITTER", "1e-2")
        ),
        "kernel": os.environ.get("SVGP_KERNEL", "matern_periodic").strip().lower(),
        "matern_nu": float(os.environ.get("SVGP_MATERN_NU", "1.5")),
        "period_length": float(
            os.environ.get("SVGP_PERIOD_LENGTH", str(8.0 / 999.0))
        ),
    }


class SVGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, config):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
            jitter_val=float(config["variational_jitter"]),
        )
        super().__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean()
        kernel_name = str(config["kernel"])
        if kernel_name == "rbf":
            base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=3)
        elif kernel_name == "matern":
            base_kernel = gpytorch.kernels.MaternKernel(
                nu=float(config["matern_nu"]),
                ard_num_dims=3,
            )
        elif kernel_name == "matern_periodic":
            spatial_kernel = gpytorch.kernels.MaternKernel(
                nu=float(config["matern_nu"]),
                ard_num_dims=2,
                active_dims=(0, 1),
            )
            temporal_matern = gpytorch.kernels.MaternKernel(
                nu=float(config["matern_nu"]),
                active_dims=(2,),
            )
            temporal_periodic = gpytorch.kernels.PeriodicKernel(
                active_dims=(2,),
            )
            temporal_periodic.initialize(
                period_length=float(config["period_length"])
            )
            base_kernel = spatial_kernel * (
                temporal_matern + temporal_periodic
            )
        else:
            raise ValueError(
                f"Unsupported SVGP_KERNEL={kernel_name!r}; "
                "choose rbf, matern, or matern_periodic"
            )
        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)

    def forward(self, x):
        mean = self.mean_module(x)
        cov = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, cov)


def stable_svgp_call(model, x):
    """Use the configured Kzz jitter for Cholesky fallback as well."""
    jitter = float(model.variational_strategy.jitter_val or 1e-6)
    with (
        gpytorch.settings.cholesky_jitter(
            float_value=jitter,
            double_value=jitter,
            half_value=jitter,
        ),
        gpytorch.settings.cholesky_max_tries(3),
    ):
        return model(x)


def train_svgp_model(model, likelihood, train_loader, val_loader, device, config):
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(likelihood.parameters()),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=int(config["num_data"]))

    best_state = None
    best_likelihood_state = None
    best_val_loss = float("inf")
    best_epoch = 0
    patience = int(config["patience"])
    patience_counter = 0

    epoch_bar = tqdm(range(int(config["epochs"])), desc="SVGP epochs", unit="epoch")
    for epoch in epoch_bar:
        model.train()
        likelihood.train()
        train_loss = 0.0

        train_bar = tqdm(
            train_loader,
            desc=f"epoch {epoch + 1:03d} train",
            leave=False,
            unit="batch",
            total=len(train_loader),
        )
        for batch in train_bar:
            optimizer.zero_grad(set_to_none=True)
            X_batch = batch["X"].to(device)
            y_batch = batch["y"].to(device)
            output = stable_svgp_call(model, X_batch)
            loss = -mll(output, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())
            train_bar.set_postfix(loss=f"{float(loss.item()):.6f}")

        train_loss /= max(1, len(train_loader))

        model.eval()
        likelihood.eval()
        # Match STDK/DLinear+FRK model selection: validation block 4+5 is
        # evaluated by prediction error, not by a second training ELBO. Using
        # ELBO here would add the training KL term once per validation batch.
        val_squared_error = 0.0
        val_count = 0
        val_bar = tqdm(
            val_loader,
            desc=f"epoch {epoch + 1:03d} val",
            leave=False,
            unit="batch",
            total=len(val_loader),
        )
        with torch.no_grad():
            for batch in val_bar:
                X_batch = batch["X"].to(device)
                y_batch = batch["y"].to(device)
                y_pred = likelihood(stable_svgp_call(model, X_batch)).mean
                squared_error = torch.sum((y_pred - y_batch) ** 2)
                val_squared_error += float(squared_error.item())
                val_count += int(y_batch.numel())
                running_rmse = np.sqrt(val_squared_error / max(1, val_count))
                val_bar.set_postfix(rmse=f"{running_rmse:.6f}")

        val_loss = float(np.sqrt(val_squared_error / max(1, val_count)))

        if val_loss < best_val_loss - 1e-12:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            best_likelihood_state = copy.deepcopy(likelihood.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        print(
            f"Epoch {epoch + 1:03d}/{int(config['epochs'])} "
            f"train_neg_elbo={train_loss:.6f} val_4plus5_rmse_scaled={val_loss:.6f}",
            flush=True,
        )

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1} (patience={patience})")
            break

        epoch_bar.set_postfix(
            train_neg_elbo=f"{train_loss:.6f}",
            val_4plus5_rmse=f"{val_loss:.6f}",
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    if best_likelihood_state is not None:
        likelihood.load_state_dict(best_likelihood_state)

    training_summary = {
        "best_epoch": int(best_epoch),
        "epochs_ran": int(epoch + 1),
        "best_val_rmse_scaled": float(best_val_loss),
    }
    return model, likelihood, training_summary


def predict_svgp(model, likelihood, X, batch_size, device):
    if len(X) == 0:
        return np.empty((0,), dtype=np.float32)

    dataset = FeatureOnlyDataset(torch.from_numpy(X))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_predict)

    preds = []
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for batch in tqdm(loader, desc="predict", unit="batch", total=len(loader), leave=False):
            X_batch = batch["X"].to(device)
            pred = likelihood(stable_svgp_call(model, X_batch)).mean
            preds.append(pred.cpu().numpy())

    return np.concatenate(preds, axis=0).reshape(-1)


# ============================================================
# Main settings
# ============================================================
_env = os.environ
EXPERIMENT_SCENARIO = _env.get("EXPERIMENT_SCENARIO", "spatiotemp_100x150")
N_SAMPLE_TARGET = int(_env.get("N_SAMPLE_TARGET", "600"))
N_TRAIN_TARGET = int(_env.get("N_TRAIN_TARGET", "100"))
N_UNKNOWN_PRIMARY_TARGET = int(_env.get("N_UNKNOWN_PRIMARY_TARGET", "400"))
N_UNKNOWN_EVAL_TARGET = int(_env.get("N_UNKNOWN_EVAL_TARGET", "100"))
N_SUPERVISED_TRAIN_TARGET = N_TRAIN_TARGET + N_UNKNOWN_PRIMARY_TARGET
N_LAST = int(_env.get("N_LAST", "1000"))
TIME_TRAIN_LEN = int(_env.get("TIME_TRAIN_LEN", "700"))
TIME_VAL_LEN = int(_env.get("TIME_VAL_LEN", "150"))
TIME_TEST_LEN = int(_env.get("TIME_TEST_LEN", "150"))
SEED_LIST = ast.literal_eval(_env.get("SEED_LIST", str(list(range(41, 46)))))
RESULT_SUFFIX = _env.get("RESULT_SUFFIX", "")

print(
    f"\n===== SVGP run ({EXPERIMENT_SCENARIO}) "
    f"sample={N_SAMPLE_TARGET} obs={N_TRAIN_TARGET} "
    f"supervised_train={N_SUPERVISED_TRAIN_TARGET} "
    f"unknown_primary={N_UNKNOWN_PRIMARY_TARGET} unknown_eval={N_UNKNOWN_EVAL_TARGET} "
    f"time={TIME_TRAIN_LEN}/{TIME_VAL_LEN}/{TIME_TEST_LEN} n_last={N_LAST} ====="
)

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

SVGP_CONFIG = build_svgp_config()
SAVE_DIR = Path(script_dir)
RESULT_DIR = Path(
    _env.get("WEATHER2K_OUTPUT_DIR", str(SAVE_DIR))
).expanduser().resolve()
suffix = f"_{RESULT_SUFFIX}" if RESULT_SUFFIX else ""
RESULT_PATH = RESULT_DIR / f"2K_svgp_metrics{suffix}.json"
TIME_STRIDE = int(os.environ.get("TIME_STRIDE", "1"))


def run_single_seed_experiment(sample_seed: int) -> dict:
    n_sample = min(N_SAMPLE_TARGET, nloc)

    np.random.seed(sample_seed)
    sample_idx_global = np.random.choice(nloc, size=n_sample, replace=False)
    sample_idx_global = np.sort(sample_idx_global)

    y_sample_full = y_all[sample_idx_global, :]
    coords_sample_full = gg[sample_idx_global, :]

    print(f"Sampled {n_sample} stations")
    print(
        f"Sampled {n_sample} stations (with {N_TRAIN_TARGET} for training, "
        f"{N_UNKNOWN_PRIMARY_TARGET} unknown400, {N_UNKNOWN_EVAL_TARGET} unknown100)"
    )

    # Select the held-out target100 first so it is identical across models and
    # independent of the DLinear obs/unobs space split.
    np.random.seed(sample_seed + 1)
    n_unknown_eval = min(N_UNKNOWN_EVAL_TARGET, n_sample)
    sample_idx_unknown_eval = np.sort(
        np.random.choice(n_sample, size=n_unknown_eval, replace=False)
    )
    sample_idx_train500 = np.setdiff1d(
        np.arange(n_sample), sample_idx_unknown_eval
    )

    n_train = min(N_TRAIN_TARGET, len(sample_idx_train500))
    np.random.seed(sample_seed)
    sample_idx_train = np.sort(
        np.random.choice(sample_idx_train500, size=n_train, replace=False)
    )
    sample_idx_unknown_primary = np.setdiff1d(
        sample_idx_train500, sample_idx_train
    )
    sample_idx_unknown = np.sort(
        np.concatenate([sample_idx_unknown_primary, sample_idx_unknown_eval])
    )

    if len(sample_idx_unknown_primary) != N_UNKNOWN_PRIMARY_TARGET:
        print(f"[Warning] unknown primary size is {len(sample_idx_unknown_primary)}, expected {N_UNKNOWN_PRIMARY_TARGET}.")

    y_train_full = y_sample_full[sample_idx_train, :]
    y_unknown_full = y_sample_full[sample_idx_unknown, :]
    y_unknown_primary_full = y_sample_full[sample_idx_unknown_primary, :]
    y_unknown_eval_full = y_sample_full[sample_idx_unknown_eval, :]

    print(f"SVGP train station count: {len(sample_idx_train500)} (obs100 + unobs400)")
    print(f"SVGP unknown station count: {len(sample_idx_unknown)}")
    print(f"SVGP unknown400 station count: {len(sample_idx_unknown_primary)}")
    print(f"SVGP unknown100 station count: {len(sample_idx_unknown_eval)}")
    print(f"\n===== SVGP seed {sample_seed} =====")
    print("Preparing SVGP dataset...")

    time_idx = np.arange(0, y_train_full.shape[1], TIME_STRIDE)
    ts_df = pd.DataFrame(
        y_train_full[:, time_idx].T,
        index=time_index[time_idx],
        columns=[f"cell_{i}" for i in range(y_train_full.shape[0])],
    ).astype("float32")

    print("Total time steps (original):", ntime)
    print("TRAIN_SIZE:", len(sample_idx_train500))
    print("UNKNOWN_SIZE:", len(sample_idx_unknown))
    print("UNKNOWN400_SIZE:", len(sample_idx_unknown_primary))
    print("UNKNOWN100_SIZE:", len(sample_idx_unknown_eval))
    print("ts_df shape (before subsetting):", ts_df.shape)

    if len(ts_df) > N_LAST:
        ts_df = ts_df.iloc[-N_LAST:]
        time_idx = time_idx[-N_LAST:]
        print(f"Subsetting to last {N_LAST} timepoints")
    else:
        print(f"Requested last {N_LAST} timepoints but only {len(ts_df)} available; using all available time steps")

    print("ts_df shape (used):", ts_df.shape)

    train_y_matrix = y_train_full[:, time_idx].astype(np.float32)
    unknown_y_matrix = y_unknown_full[:, time_idx].astype(np.float32)
    unknown_primary_y_matrix = y_unknown_primary_full[:, time_idx].astype(np.float32)
    unknown_eval_y_matrix = y_unknown_eval_full[:, time_idx].astype(np.float32)
    full_y_matrix = y_sample_full[:, time_idx].astype(np.float32)
    train500_y_matrix = full_y_matrix[sample_idx_train500, :]

    coords_full_norm = normalize_coords(coords_sample_full)
    coords_train_norm = coords_full_norm[sample_idx_train, :]
    coords_unknown_norm = coords_full_norm[sample_idx_unknown, :]
    coords_unknown_primary_norm = coords_full_norm[sample_idx_unknown_primary, :]
    coords_unknown_eval_norm = coords_full_norm[sample_idx_unknown_eval, :]
    coords_train500_norm = coords_full_norm[sample_idx_train500, :]

    n_full = coords_full_norm.shape[0]
    n_train = coords_train_norm.shape[0]
    n_unknown = coords_unknown_norm.shape[0]
    n_unknown_primary = coords_unknown_primary_norm.shape[0]
    n_unknown_eval = coords_unknown_eval_norm.shape[0]
    n_time_used = len(ts_df)

    expected_time_len = TIME_TRAIN_LEN + TIME_VAL_LEN + TIME_TEST_LEN
    if n_time_used < expected_time_len:
        raise ValueError(
            f"Current time points {n_time_used} are insufficient for {TIME_TRAIN_LEN}/{TIME_VAL_LEN}/{TIME_TEST_LEN} split"
        )

    cut_train = TIME_TRAIN_LEN
    cut_val = TIME_TRAIN_LEN + TIME_VAL_LEN

    train_time_idx = np.arange(0, cut_train)
    val_time_idx = np.arange(cut_train, cut_val)
    test_time_idx = np.arange(cut_val, n_time_used)
    if len(test_time_idx) != TIME_TEST_LEN:
        test_time_idx = test_time_idx[:TIME_TEST_LEN]

    if EXPERIMENT_SCENARIO == "space_extrap_fixed850":
        eval_time_idx = np.arange(0, cut_val)
        eval_time_label = "SPACE_FIXED850"
    else:
        eval_time_idx = test_time_idx
        eval_time_label = "TIME_TEST150"

    print(f"Total time steps (used): {n_time_used}")
    print(f"Train len: {len(train_time_idx)}, Val len: {len(val_time_idx)}, Test len: {len(test_time_idx)}")
    print(f"Eval window: {eval_time_label}, Eval len: {len(eval_time_idx)}")
    print("SVGP train block: 1+2 = Train700 x (obs100 + unobs400)")
    print("SVGP validation block: 4+5 = Val150 x (obs100 + unobs400)")

    t_norm_all = np.linspace(0.0, 1.0, n_time_used, dtype=np.float32)

    train_y_raw = train_y_matrix[:, train_time_idx].reshape(-1)
    y_mean = float(np.mean(train_y_raw))
    y_std = float(np.std(train_y_raw, ddof=0))
    if y_std < 1e-12:
        y_std = 1.0

    def to_std(y):
        return ((y - y_mean) / y_std).astype(np.float32)

    def to_raw(y_std_arr):
        return y_std_arr * y_std + y_mean

    X_train, y_train = build_flat_inputs(
        train500_y_matrix, coords_train500_norm, train_time_idx, t_norm_all
    )
    X_val_obs, y_val_obs = build_flat_inputs(train_y_matrix, coords_train_norm, val_time_idx, t_norm_all)
    X_val_unobs400, y_val_unobs400 = build_flat_inputs(
        unknown_primary_y_matrix, coords_unknown_primary_norm, val_time_idx, t_norm_all
    )
    X_val = np.concatenate([X_val_obs, X_val_unobs400], axis=0)
    y_val = np.concatenate([y_val_obs, y_val_unobs400], axis=0)

    X_test_train, y_test_train = build_flat_inputs(train_y_matrix, coords_train_norm, eval_time_idx, t_norm_all)
    X_test_unknown, y_test_unknown = build_flat_inputs(unknown_y_matrix, coords_unknown_norm, eval_time_idx, t_norm_all)
    X_test_unknown_primary, y_test_unknown_primary = build_flat_inputs(
        unknown_primary_y_matrix, coords_unknown_primary_norm, eval_time_idx, t_norm_all
    )
    X_test_unknown_eval, y_test_unknown_eval = build_flat_inputs(
        unknown_eval_y_matrix, coords_unknown_eval_norm, eval_time_idx, t_norm_all
    )
    X_test_full, y_test_full = build_flat_inputs(full_y_matrix, coords_full_norm, eval_time_idx, t_norm_all)

    y_train = to_std(y_train)
    y_val = to_std(y_val)
    y_test_train = to_std(y_test_train)
    y_test_unknown = to_std(y_test_unknown)
    y_test_unknown_primary = to_std(y_test_unknown_primary)
    y_test_unknown_eval = to_std(y_test_unknown_eval)
    y_test_full = to_std(y_test_full)

    train_dataset = FeatureDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = FeatureDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    g = torch.Generator()
    g.manual_seed(sample_seed + 1000)

    batch_size = int(SVGP_CONFIG["batch_size"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_train,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_train,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device)

    torch.manual_seed(sample_seed)
    np.random.seed(sample_seed)

    num_inducing = min(int(SVGP_CONFIG["num_inducing"]), len(train_dataset))
    inducing_idx = torch.randperm(len(train_dataset))[:num_inducing]
    inducing_points = train_dataset.X[inducing_idx].clone().to(device)

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    likelihood.initialize(noise=float(SVGP_CONFIG["init_noise"]))
    model = SVGPModel(inducing_points, SVGP_CONFIG).to(device)

    train_start = time.time()
    model, likelihood, training_summary = _silent_call(
        train_svgp_model,
        model,
        likelihood,
        train_loader,
        val_loader,
        device,
        {**SVGP_CONFIG, "num_data": len(train_dataset)},
    )
    elapsed = time.time() - train_start

    pred_test_train_std = _silent_call(
        predict_svgp,
        model,
        likelihood,
        X_test_train,
        batch_size,
        device,
    )
    pred_test_unknown_std = _silent_call(
        predict_svgp,
        model,
        likelihood,
        X_test_unknown,
        batch_size,
        device,
    )
    pred_test_unknown_primary_std = _silent_call(
        predict_svgp,
        model,
        likelihood,
        X_test_unknown_primary,
        batch_size,
        device,
    )
    pred_test_unknown_eval_std = _silent_call(
        predict_svgp,
        model,
        likelihood,
        X_test_unknown_eval,
        batch_size,
        device,
    )
    pred_test_full_std = _silent_call(
        predict_svgp,
        model,
        likelihood,
        X_test_full,
        batch_size,
        device,
    )

    eval_len = len(eval_time_idx)
    pred_test_train = to_raw(pred_test_train_std).reshape(eval_len, n_train)
    pred_test_unknown = to_raw(pred_test_unknown_std).reshape(eval_len, n_unknown)
    pred_test_unknown_primary = to_raw(pred_test_unknown_primary_std).reshape(eval_len, n_unknown_primary)
    pred_test_unknown_eval = to_raw(pred_test_unknown_eval_std).reshape(eval_len, n_unknown_eval)
    pred_test_full = to_raw(pred_test_full_std).reshape(eval_len, n_full)

    test_true_train = train_y_matrix[:, eval_time_idx].T.astype(np.float32)
    test_true_unknown = unknown_y_matrix[:, eval_time_idx].T.astype(np.float32)
    test_true_unknown_primary = unknown_primary_y_matrix[:, eval_time_idx].T.astype(np.float32)
    test_true_unknown_eval = unknown_eval_y_matrix[:, eval_time_idx].T.astype(np.float32)
    test_true_full = full_y_matrix[:, eval_time_idx].T.astype(np.float32)

    def safe_metrics(y_true, y_pred):
        if y_true.size == 0 or y_pred.size == 0:
            return {"RMSE": float("nan"), "MSE": float("nan"), "MAE": float("nan"), "R2": float("nan")}
        return _compute_metrics(y_true, y_pred)

    train_metric = safe_metrics(test_true_train, pred_test_train)
    unknown_metric = safe_metrics(test_true_unknown, pred_test_unknown)
    unknown_primary_metric = safe_metrics(test_true_unknown_primary, pred_test_unknown_primary)
    unknown_eval_metric = safe_metrics(test_true_unknown_eval, pred_test_unknown_eval)
    full_metric = safe_metrics(test_true_full, pred_test_full)

    empty_metric = {"RMSE": float("nan"), "MSE": float("nan"), "MAE": float("nan"), "R2": float("nan")}
    train700_metric = empty_metric
    val150_metric = empty_metric
    target_time150_metric = full_metric
    if EXPERIMENT_SCENARIO == "time_extrap_fixed500":
        eval500_idx = np.sort(np.concatenate([sample_idx_train, sample_idx_unknown_primary]))
        eval500_y_matrix = full_y_matrix[eval500_idx, :]
        coords_eval500_norm = coords_full_norm[eval500_idx, :]
        X_train700, y_train700 = build_flat_inputs(eval500_y_matrix, coords_eval500_norm, train_time_idx, t_norm_all)
        X_val150, y_val150 = build_flat_inputs(eval500_y_matrix, coords_eval500_norm, val_time_idx, t_norm_all)
        pred_train700_std = _silent_call(predict_svgp, model, likelihood, X_train700, batch_size, device)
        pred_val150_std = _silent_call(predict_svgp, model, likelihood, X_val150, batch_size, device)
        pred_train700 = to_raw(pred_train700_std).reshape(len(train_time_idx), len(eval500_idx))
        pred_val150 = to_raw(pred_val150_std).reshape(len(val_time_idx), len(eval500_idx))
        true_train700 = eval500_y_matrix[:, train_time_idx].T.astype(np.float32)
        true_val150 = eval500_y_matrix[:, val_time_idx].T.astype(np.float32)
        train700_metric = safe_metrics(true_train700, pred_train700)
        val150_metric = safe_metrics(true_val150, pred_val150)
        target_time150_metric = safe_metrics(test_true_full[:, eval500_idx], pred_test_full[:, eval500_idx])

    def prefixed(prefix, metric):
        return {
            f"{prefix}_RMSE": metric["RMSE"],
            f"{prefix}_MSE": metric["MSE"],
            f"{prefix}_MAE": metric["MAE"],
            f"{prefix}_R2": metric["R2"],
        }

    if EXPERIMENT_SCENARIO == "time_extrap_fixed500":
        metrics = {
            **prefixed("Train700", train700_metric),
            **prefixed("Val150", val150_metric),
            **prefixed("Target_Time150", target_time150_metric),
        }
        result_sections = {
            "Train700": train700_metric,
            "Val150": val150_metric,
            "Target_Time150": target_time150_metric,
        }
    elif EXPERIMENT_SCENARIO == "space_extrap_fixed850":
        metrics = {
            **prefixed("Train100", train_metric),
            **prefixed("Val400", unknown_primary_metric),
            **prefixed("Target_Space100", unknown_eval_metric),
        }
        result_sections = {
            "Train100": train_metric,
            "Val400": unknown_primary_metric,
            "Target_Space100": unknown_eval_metric,
        }
    elif EXPERIMENT_SCENARIO == "spatiotemp_100x150":
        metrics = prefixed("Target_ST100x150", unknown_eval_metric)
        result_sections = {
            "Target_ST100x150": unknown_eval_metric,
        }
    else:
        metrics = prefixed("Target", full_metric)
        result_sections = {"Target": full_metric}

    print(f"Training elapsed={_fmt_time(elapsed)}")
    print(pd.DataFrame([{"Model": "SVGP", "Split": f"TEST_seed_{sample_seed}", **metrics}]).to_string(index=False))

    payload = {
        "seed": int(sample_seed),
        "Model": "SVGP",
        "Split": "TEST",
        "results": result_sections,
        "elapsed_seconds": float(elapsed),
        "params_used": SVGP_CONFIG,
        "training_summary": training_summary,
        "sampling_info": {
            "full_sample_size": int(N_SAMPLE_TARGET),
            "observed_sample_size": int(N_TRAIN_TARGET),
            "train_sample_size": int(N_SUPERVISED_TRAIN_TARGET),
            "unknown_primary_sample_size": int(N_UNKNOWN_PRIMARY_TARGET),
            "unknown_eval_sample_size": int(N_UNKNOWN_EVAL_TARGET),
            "sample_seed": int(sample_seed),
            "time_stride": int(TIME_STRIDE),
            "n_last_timepoints": int(N_LAST),
            "experiment_scenario": EXPERIMENT_SCENARIO,
            "time_train_len": int(TIME_TRAIN_LEN),
            "time_val_len": int(TIME_VAL_LEN),
            "time_test_len": int(TIME_TEST_LEN),
            "eval_time_label": eval_time_label,
            "eval_time_len": int(eval_len),
            "sample_idx_global": sample_idx_global.tolist(),
            "sample_idx_train": sample_idx_train.tolist(),
            "sample_idx_train500": sample_idx_train500.tolist(),
            "sample_idx_unknown": sample_idx_unknown.tolist(),
            "sample_idx_unknown_primary": sample_idx_unknown_primary.tolist(),
            "sample_idx_unknown_eval": sample_idx_unknown_eval.tolist(),
        },
        "n_full": int(n_full),
        "n_train": int(n_train),
        "n_unknown": int(n_unknown),
        "n_unknown_primary": int(n_unknown_primary),
        "n_unknown_eval": int(n_unknown_eval),
        "n_time_used": int(n_time_used),
    }

    del model
    del likelihood
    gc.collect()
    _cleanup_torch_cache()

    return {"seed": int(sample_seed), "metrics": metrics, "payload": payload}


seed_runs = []
for sample_seed in SEED_LIST:
    seed_runs.append(run_single_seed_experiment(sample_seed))

metrics_df = pd.DataFrame([run["metrics"] for run in seed_runs])
mean_metrics = metrics_df.mean(numeric_only=True).to_dict()
std_metrics = metrics_df.std(numeric_only=True, ddof=0).to_dict()
elapsed_mean = float(np.mean([run["payload"]["elapsed_seconds"] for run in seed_runs]))

summary_metrics = pd.DataFrame(
    [
        {
            "Model": "SVGP",
            "Split": "TEST_MEAN",
            **{key: float(value) for key, value in mean_metrics.items()},
        }
    ]
)

print("\n===== SVGP TEST RESULTS (seed 41~45 mean) =====")
print(summary_metrics.to_string(index=False))
summary_metrics_std = pd.DataFrame(
    [
        {
            "Model": "SVGP",
            "Split": "TEST_STD",
            **{key: float(value) for key, value in std_metrics.items()},
        }
    ]
)
print("\n===== SVGP TEST RESULTS (seed 41~45 std) =====")
print(summary_metrics_std.to_string(index=False))
print(f"Mean training elapsed={_fmt_time(elapsed_mean)}")

payload = {
    "Model": "SVGP",
    "Split": "TEST_MEAN",
    "seed_range": [int(SEED_LIST[0]), int(SEED_LIST[-1])],
    "seed_list": [int(seed) for seed in SEED_LIST],
    "mean_metrics": {key: float(value) for key, value in mean_metrics.items()},
    "std_metrics": {key: float(value) for key, value in std_metrics.items()},
    "avg_elapsed_seconds": float(elapsed_mean),
    "params_used": SVGP_CONFIG,
    "sampling_info": {
        "full_sample_size": int(N_SAMPLE_TARGET),
        "observed_sample_size": int(N_TRAIN_TARGET),
        "train_sample_size": int(N_SUPERVISED_TRAIN_TARGET),
        "unknown_primary_sample_size": int(N_UNKNOWN_PRIMARY_TARGET),
        "unknown_eval_sample_size": int(N_UNKNOWN_EVAL_TARGET),
        "time_stride": int(TIME_STRIDE),
        "n_last_timepoints": int(N_LAST),
        "experiment_scenario": EXPERIMENT_SCENARIO,
        "time_train_len": int(TIME_TRAIN_LEN),
        "time_val_len": int(TIME_VAL_LEN),
        "time_test_len": int(TIME_TEST_LEN),
    },
    "seed_runs": seed_runs,
}

_json_dump(payload, RESULT_PATH)
print(f"Saved metrics: {RESULT_PATH.resolve()}")
print("Repo SVGP run completed.")
