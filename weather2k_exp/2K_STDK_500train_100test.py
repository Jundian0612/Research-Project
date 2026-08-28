"""Two-stage STDK interpolation + shared univariate LSTM forecasting.

STDK forms complete spatial histories; one LSTM shared across stations performs
autoregressive temporal forecasting. The benchmark remains train500/test100.
"""

import importlib.util
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
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):
        return iterable


# ============================================================
# Weather2K 載入器
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
        f"變數名稱數量 = {len(weather_var_names)}，但資料中的氣象變數數量 = {nvar}，請確認 Weather2K 變數順序"
    )

# 目前實驗目標：氣溫
target_var_name = "air_temperature"
if target_var_name not in weather_var_names:
    raise ValueError(f"{target_var_name} 不在 weather_var_names 中")

var_idx = weather_var_names.index(target_var_name)
var_data = data[:, :, var_idx].astype(np.float32)
y_all = var_data

lat = loc[:, 0]
lon = loc[:, 1]
gg = np.column_stack([lon, lat])

print(f"✓ 確認使用 {target_var_name} 資料")


# ============================================================
# 工具函式
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
    rmse = float(np.sqrt(mse))
    r2 = r2_score(
        np.asarray(y_true).reshape(-1),
        np.asarray(y_pred).reshape(-1),
    )
    return {
        "rmse": float(rmse),
        "mse": float(mse),
        "mae": float(mae),
        "r2": float(r2),
    }


class DictDataset(Dataset):
    def __init__(self, X, coords, t, y):
        self.X = X
        self.coords = coords
        self.t = t
        self.y = y

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return {
            "X": self.X[idx],
            "coords": self.coords[idx],
            "t": self.t[idx],
            "y": self.y[idx],
        }


def collate_fn(batch):
    return {
        "X": torch.stack([b["X"] for b in batch]),
        "coords": torch.stack([b["coords"] for b in batch]),
        "t": torch.stack([b["t"] for b in batch]),
        "y": torch.stack([b["y"] for b in batch]),
    }


def build_stdk_model_config():
    return {
        "p_covariates": 0,
        "regression_type": "mean",
        "epochs": 350,
        "lr": 1.0e-3,
        "weight_decay": 1.0e-4,
        "batch_size": 512,
        "patience": 30,
        "k_spatial_centers": [25, 81, 121],
        "k_temporal_centers": [10, 15, 45],
        "hidden_dims": [256, 256, 128],
        "dropout": 0.1,
        "layernorm": True,
        "spatial_learnable": False,
        "spatial_init_method": "uniform",
        "spatial_basis_function": "wendland",
        "gradient_damping": False,
        "damping_threshold": 0.0,
        "damping_strength": 1.0,
        "temporal_bandwidth_factor": 2.5,
        "use_delta_reparameterization": False,
    }


def train_stdk_model(model, train_loader, val_loader, device, config):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )

    best_state = None
    best_val_loss = float("inf")
    patience = int(config.get("patience", 30))
    patience_counter = 0

    epoch_bar = tqdm(range(int(config["epochs"])), desc="STDK epochs", unit="epoch")

    for epoch in epoch_bar:
        model.train()
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
            y_pred = model(
                batch["X"].to(device),
                batch["coords"].to(device),
                batch["t"].to(device),
            )
            loss = criterion(y_pred, batch["y"].to(device))
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())
            train_bar.set_postfix(loss=f"{float(loss.item()):.6f}")

        train_loss /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        val_bar = tqdm(
            val_loader,
            desc=f"epoch {epoch + 1:03d} val",
            leave=False,
            unit="batch",
            total=len(val_loader),
        )
        with torch.no_grad():
            for batch in val_bar:
                y_pred = model(
                    batch["X"].to(device),
                    batch["coords"].to(device),
                    batch["t"].to(device),
                )
                loss = criterion(y_pred, batch["y"].to(device))
                val_loss += float(loss.item())
                val_bar.set_postfix(loss=f"{float(loss.item()):.6f}")

        val_loss /= max(1, len(val_loader))

        if val_loss < best_val_loss - 1e-12:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        print(
            f"Epoch {epoch + 1:03d}/{int(config['epochs'])} "
            f"train_mse={train_loss:.6f} val_mse={val_loss:.6f}",
            flush=True,
        )

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1} (patience={patience})")
            break

        epoch_bar.set_postfix(train_mse=f"{train_loss:.6f}", val_mse=f"{val_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def predict_stdk(model, X, coords, t, batch_size, device):
    if len(X) == 0:
        return np.empty((0,), dtype=np.float32)

    dataset = DictDataset(X, coords, t, torch.zeros((len(X), 1), dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    preds = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="predict", unit="batch", total=len(loader), leave=False):
            pred = model(
                batch["X"].to(device),
                batch["coords"].to(device),
                batch["t"].to(device),
            )
            preds.append(pred.cpu().numpy())

    return np.concatenate(preds, axis=0).reshape(-1)


def build_flat_inputs(y_matrix, coords, time_idx_subset, t_norm_all):
    coords = np.asarray(coords, dtype=np.float32)
    time_idx_subset = np.asarray(time_idx_subset, dtype=int)
    y_matrix = np.asarray(y_matrix, dtype=np.float32)
    t_values = np.asarray(t_norm_all[time_idx_subset], dtype=np.float32).reshape(-1, 1)

    n_sites = coords.shape[0]
    n_times = len(time_idx_subset)

    coords_flat = np.tile(coords, (n_times, 1)).astype(np.float32)
    t_flat = np.repeat(t_values, n_sites, axis=0).astype(np.float32)
    y_flat = y_matrix[:, time_idx_subset].T.reshape(-1, 1).astype(np.float32)
    X_flat = np.empty((n_sites * n_times, 0), dtype=np.float32)

    return X_flat, coords_flat, t_flat, y_flat


def predict_stdk_matrix_chunked(
    model, y_matrix, coords, time_indices, t_norm_all, batch_size, device, time_chunk_size=50
):
    """Predict a time-by-station matrix without materializing every time point."""
    time_indices = np.asarray(time_indices, dtype=int)
    parts = []
    for start in range(0, len(time_indices), int(time_chunk_size)):
        chunk = time_indices[start : start + int(time_chunk_size)]
        X, coords_flat, t_flat, _ = build_flat_inputs(y_matrix, coords, chunk, t_norm_all)
        pred = _silent_call(
            predict_stdk,
            model,
            torch.from_numpy(X),
            torch.from_numpy(coords_flat),
            torch.from_numpy(t_flat),
            batch_size,
            device,
        ).reshape(len(chunk), len(coords))
        parts.append(pred)
        del X, coords_flat, t_flat
    return np.concatenate(parts, axis=0)


def normalize_coords(coords):
    coords = np.asarray(coords, dtype=np.float32)
    min_vals = coords.min(axis=0)
    max_vals = coords.max(axis=0)
    denom = np.where((max_vals - min_vals) < 1e-12, 1.0, max_vals - min_vals)
    return ((coords - min_vals) / denom).astype(np.float32)


class SharedUnivariateLSTM(torch.nn.Module):
    """One univariate LSTM shared by every station sequence."""

    def __init__(self, hidden_size=50, num_layers=1):
        super().__init__()
        self.lstm = torch.nn.LSTM(1, hidden_size, num_layers=num_layers, batch_first=True)
        self.head = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.head(output[:, -1, :])


def make_shared_lstm_windows(series_time_station, lookback):
    values = np.asarray(series_time_station, dtype=np.float32)
    x, y = [], []
    for station in range(values.shape[1]):
        sequence = values[:, station]
        for start in range(0, len(sequence) - lookback):
            x.append(sequence[start : start + lookback])
            y.append(sequence[start + lookback])
    return np.asarray(x, dtype=np.float32)[..., None], np.asarray(y, dtype=np.float32)[:, None]


def forecast_shared_lstm(model, history_time_station, n_steps, lookback, device, batch_size):
    history = np.asarray(history_time_station, dtype=np.float32).copy()
    predictions = []
    model.eval()
    with torch.no_grad():
        for _ in range(int(n_steps)):
            windows = history[-lookback:, :].T[..., None]
            step_parts = []
            for start in range(0, len(windows), batch_size):
                xb = torch.from_numpy(windows[start : start + batch_size]).to(device)
                step_parts.append(model(xb).cpu().numpy().reshape(-1))
            step = np.concatenate(step_parts)
            predictions.append(step)
            history = np.vstack([history, step[None, :]])
    return np.asarray(predictions, dtype=np.float32)


def train_shared_lstm(train_series, val_truth, val_station_idx, device, config, seed):
    lookback = int(config["lookback"])
    x, y = make_shared_lstm_windows(train_series, lookback)
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    generator = torch.Generator().manual_seed(seed + 3000)
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    torch.manual_seed(seed + 4000)
    model = SharedUnivariateLSTM(config["hidden_size"], config["num_layers"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["lr"]))
    criterion = torch.nn.MSELoss()
    best_state, best_val, wait, best_epoch = None, float("inf"), 0, 0
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        val_pred = forecast_shared_lstm(
            model, train_series, len(val_truth), lookback, device, int(config["predict_batch_size"])
        )[:, val_station_idx]
        val_rmse = float(np.sqrt(mean_squared_error(val_truth, val_pred)))
        if val_rmse < best_val - 1e-6:
            best_val, best_epoch, wait = val_rmse, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
        print(f"SharedLSTM epoch {epoch:03d} val_rmse_scaled={val_rmse:.6f} best={best_val:.6f}")
        if wait >= int(config["patience"]):
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "best_val_rmse_scaled": best_val}


def _build_stdk_dataset(y_values, time_stride=3):
    time_idx = np.arange(0, y_values.shape[1], time_stride)
    ts_df = pd.DataFrame(
        y_values[:, time_idx].T,
        index=time_index[time_idx],
        columns=[f"cell_{i}" for i in range(y_values.shape[0])],
    ).astype("float32")
    return ts_df, time_idx


def load_stdk_create_model(script_dir: str):
    spatial_adapter_root = Path(script_dir).resolve().parent / "spatial-adapter"
    module_path = spatial_adapter_root / "examples" / "baselines" / "stdk" / "st_interp.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"Cannot find STDK baseline file: {module_path}")

    spec = importlib.util.spec_from_file_location("stdk_st_interp", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load STDK baseline module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.create_model


# ============================================================
# 主要設定
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
    f"\n===== STDK run ({EXPERIMENT_SCENARIO}) "
    f"sample={N_SAMPLE_TARGET} obs={N_TRAIN_TARGET} "
    f"supervised_train={N_SUPERVISED_TRAIN_TARGET} "
    f"unknown_primary={N_UNKNOWN_PRIMARY_TARGET} unknown_eval={N_UNKNOWN_EVAL_TARGET} "
    f"time={TIME_TRAIN_LEN}/{TIME_VAL_LEN}/{TIME_TEST_LEN} n_last={N_LAST} ====="
)

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

STDK_CONFIG = build_stdk_model_config()
LSTM_CONFIG = {
    "mode": "shared_univariate",
    "lookback": int(_env.get("STDK_LSTM_LOOKBACK", "5")),
    "hidden_size": int(_env.get("STDK_LSTM_HIDDEN_SIZE", "50")),
    "num_layers": int(_env.get("STDK_LSTM_NUM_LAYERS", "1")),
    "epochs": int(_env.get("STDK_LSTM_EPOCHS", "120")),
    "batch_size": int(_env.get("STDK_LSTM_BATCH_SIZE", "1024")),
    "predict_batch_size": int(_env.get("STDK_LSTM_PREDICT_BATCH_SIZE", "1024")),
    "lr": float(_env.get("STDK_LSTM_LR", "0.001")),
    "patience": int(_env.get("STDK_LSTM_PATIENCE", "15")),
    "loss": "MSE",
}
SAVE_DIR = Path(script_dir)
suffix = f"_{RESULT_SUFFIX}" if RESULT_SUFFIX else ""
RESULT_PATH = SAVE_DIR / f"2K_stdk_metrics{suffix}.json"
TIME_STRIDE = int(os.environ.get("TIME_STRIDE", "1"))


def run_single_seed_experiment(sample_seed: int) -> dict:
    n_sample = min(N_SAMPLE_TARGET, nloc)

    np.random.seed(sample_seed)
    sample_idx_global = np.random.choice(nloc, size=n_sample, replace=False)
    sample_idx_global = np.sort(sample_idx_global)

    y_sample_full = y_all[sample_idx_global, :]
    coords_sample_full = gg[sample_idx_global, :]

    print(f"抽樣 {n_sample} 個測站")
    print(
        f"抽樣 {n_sample} 個測站（其中 {N_TRAIN_TARGET} 個進行訓練，"
        f"{N_UNKNOWN_PRIMARY_TARGET} 個 unknown400，{N_UNKNOWN_EVAL_TARGET} 個 unknown100）"
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
    coords_train_full = coords_sample_full[sample_idx_train, :]
    y_unknown_full = y_sample_full[sample_idx_unknown, :]
    y_unknown_primary_full = y_sample_full[sample_idx_unknown_primary, :]
    y_unknown_eval_full = y_sample_full[sample_idx_unknown_eval, :]
    coords_unknown_full = coords_sample_full[sample_idx_unknown, :]
    coords_unknown_primary_full = coords_sample_full[sample_idx_unknown_primary, :]
    coords_unknown_eval_full = coords_sample_full[sample_idx_unknown_eval, :]

    print(f"STDK 訓練站點數: {len(sample_idx_train500)} (obs100 + unobs400)")
    print(f"STDK unknown 總站點數: {len(sample_idx_unknown)}")
    print(f"STDK unknown400 站點數: {len(sample_idx_unknown_primary)}")
    print(f"STDK unknown100 站點數: {len(sample_idx_unknown_eval)}")
    print(f"\n===== STDK seed {sample_seed} =====")
    print("Preparing STDK dataset...")

    ts_df, used_time_idx = _build_stdk_dataset(y_train_full, time_stride=TIME_STRIDE)
    print("Total time steps (original):", ntime)
    print("TRAIN_SIZE:", len(sample_idx_train500))
    print("UNKNOWN_SIZE:", len(sample_idx_unknown))
    print("UNKNOWN400_SIZE:", len(sample_idx_unknown_primary))
    print("UNKNOWN100_SIZE:", len(sample_idx_unknown_eval))
    print("ts_df shape (before subsetting):", ts_df.shape)

    if len(ts_df) > N_LAST:
        ts_df = ts_df.iloc[-N_LAST:]
        used_time_idx = used_time_idx[-N_LAST:]
        print(f"Subsetting to last {N_LAST} timepoints")
    else:
        print(f"Requested last {N_LAST} timepoints but only {len(ts_df)} available; using all available time steps")

    print("ts_df shape (used):", ts_df.shape)

    train_y_matrix = y_train_full[:, used_time_idx].astype(np.float32)
    unknown_y_matrix = y_unknown_full[:, used_time_idx].astype(np.float32)
    unknown_primary_y_matrix = y_unknown_primary_full[:, used_time_idx].astype(np.float32)
    unknown_eval_y_matrix = y_unknown_eval_full[:, used_time_idx].astype(np.float32)
    full_y_matrix = y_sample_full[:, used_time_idx].astype(np.float32)
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

    # 時間切分固定為最後 1000 個時間點中的 700/150/150；
    # 純空間外推只使用前 850 個時間點做評估，避免把時間外推混進來。
    expected_time_len = TIME_TRAIN_LEN + TIME_VAL_LEN + TIME_TEST_LEN
    if n_time_used < expected_time_len:
        raise ValueError(
            f"目前時間點數 {n_time_used} 不足以做 {TIME_TRAIN_LEN}/{TIME_VAL_LEN}/{TIME_TEST_LEN} 切分"
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
    print("STDK train block: 1+2 = Train700 x (obs100 + unobs400)")
    print("STDK validation block: 4+5 = Val150 x (obs100 + unobs400)")

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

    X_train, coords_train, t_train, y_train = build_flat_inputs(
        train500_y_matrix, coords_train500_norm, train_time_idx, t_norm_all
    )
    # 對齊 DLinear+FRK 的可用監督資訊：Train700 使用 obs100 +
    # unobs400 共 500 站，validation 使用相同 500 站的 Val150。
    X_val_obs, coords_val_obs, t_val_obs, y_val_obs = build_flat_inputs(
        train_y_matrix, coords_train_norm, val_time_idx, t_norm_all
    )
    X_val_unobs400, coords_val_unobs400, t_val_unobs400, y_val_unobs400 = build_flat_inputs(
        unknown_primary_y_matrix, coords_unknown_primary_norm, val_time_idx, t_norm_all
    )
    X_val = np.concatenate([X_val_obs, X_val_unobs400], axis=0)
    coords_val = np.concatenate([coords_val_obs, coords_val_unobs400], axis=0)
    t_val = np.concatenate([t_val_obs, t_val_unobs400], axis=0)
    y_val = np.concatenate([y_val_obs, y_val_unobs400], axis=0)

    y_train = to_std(y_train)
    y_val = to_std(y_val)

    train_dataset = DictDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(coords_train),
        torch.from_numpy(t_train),
        torch.from_numpy(y_train),
    )
    val_dataset = DictDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(coords_val),
        torch.from_numpy(t_val),
        torch.from_numpy(y_val),
    )

    g = torch.Generator()
    g.manual_seed(sample_seed + 1000)

    batch_size = int(STDK_CONFIG["batch_size"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device)

    torch.manual_seed(sample_seed)
    np.random.seed(sample_seed)

    create_model = load_stdk_create_model(script_dir)

    model = create_model(STDK_CONFIG, train_coords=coords_train).to(device)

    train_start = time.time()
    model = _silent_call(
        train_stdk_model,
        model,
        train_loader,
        val_loader,
        device,
        STDK_CONFIG,
    )
    stdk_elapsed = time.time() - train_start

    # The flattened training/validation tensors are no longer needed after
    # fitting STDK. Release them before constructing interpolation inputs.
    del train_loader, val_loader, train_dataset, val_dataset
    del X_train, coords_train, t_train, y_train
    del X_val, coords_val, t_val, y_val
    del X_val_obs, coords_val_obs, t_val_obs, y_val_obs
    del X_val_unobs400, coords_val_unobs400, t_val_unobs400, y_val_unobs400
    gc.collect()
    _cleanup_torch_cache()

    # Stage 1: STDK interpolation forms a complete Train700 spatial time series.
    pred_hist_std = predict_stdk_matrix_chunked(
        model,
        full_y_matrix,
        coords_full_norm,
        train_time_idx,
        t_norm_all,
        batch_size,
        device,
        time_chunk_size=int(_env.get("STDK_PRED_TIME_CHUNK", "50")),
    )
    gc.collect()
    _cleanup_torch_cache()

    # Stage 2: one shared univariate LSTM forecasts every station.
    val_truth_scaled = to_std(full_y_matrix[sample_idx_train500][:, val_time_idx].T)
    lstm_model, lstm_summary = train_shared_lstm(
        pred_hist_std,
        val_truth_scaled,
        sample_idx_train500,
        device,
        LSTM_CONFIG,
        sample_seed,
    )
    pred_future_std = forecast_shared_lstm(
        lstm_model,
        pred_hist_std,
        TIME_VAL_LEN + TIME_TEST_LEN,
        LSTM_CONFIG["lookback"],
        device,
        LSTM_CONFIG["predict_batch_size"],
    )
    pred_val_full = to_raw(pred_future_std[:TIME_VAL_LEN])
    pred_test_full = to_raw(pred_future_std[TIME_VAL_LEN:])
    del pred_future_std
    gc.collect()
    _cleanup_torch_cache()
    elapsed = time.time() - train_start

    # Pure spatial interpolation uses STDK directly over Train700+Val150.
    space_idx = np.arange(0, cut_val)
    pred_space_full = to_raw(predict_stdk_matrix_chunked(
        model,
        full_y_matrix,
        coords_full_norm,
        space_idx,
        t_norm_all,
        batch_size,
        device,
        time_chunk_size=int(_env.get("STDK_PRED_TIME_CHUNK", "50")),
    ))
    gc.collect()
    _cleanup_torch_cache()

    if EXPERIMENT_SCENARIO == "space_extrap_fixed850":
        pred_eval_full = pred_space_full
    else:
        pred_eval_full = pred_test_full
    eval_len = int(pred_eval_full.shape[0])
    pred_test_train = pred_eval_full[:, sample_idx_train]
    pred_test_unknown = pred_eval_full[:, sample_idx_unknown]
    pred_test_unknown_primary = pred_eval_full[:, sample_idx_unknown_primary]
    pred_test_unknown_eval = pred_eval_full[:, sample_idx_unknown_eval]

    test_true_train = train_y_matrix[:, eval_time_idx].T.astype(np.float32)
    test_true_unknown = unknown_y_matrix[:, eval_time_idx].T.astype(np.float32)
    test_true_unknown_primary = unknown_primary_y_matrix[:, eval_time_idx].T.astype(np.float32)
    test_true_unknown_eval = unknown_eval_y_matrix[:, eval_time_idx].T.astype(np.float32)
    test_true_full = full_y_matrix[:, eval_time_idx].T.astype(np.float32)

    def safe_metrics(y_true, y_pred):
        if y_true.size == 0 or y_pred.size == 0:
            return {"RMSE": float("nan"), "MSE": float("nan"), "MAE": float("nan"), "R2": float("nan")}
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
        # 時間外推情境沒有 Unknown400/Unknown100；
        # 這裡只看固定 500 空間點，也就是圖上的 1+2、4+5、7+8。
        eval500_idx = np.sort(np.concatenate([sample_idx_train, sample_idx_unknown_primary]))
        eval500_y_matrix = full_y_matrix[eval500_idx, :]
        pred_train700 = to_raw(pred_hist_std[:, eval500_idx])
        pred_val150 = pred_val_full[:, eval500_idx]
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
    print(pd.DataFrame([{"Model": "STDK+SharedLSTM", "Split": f"TEST_seed_{sample_seed}", **metrics}]).to_string(index=False))

    payload = {
        "seed": int(sample_seed),
        "Model": "STDK+SharedLSTM",
        "Split": "TEST",
        "results": result_sections,
        "elapsed_seconds": float(elapsed),
        "params_used": STDK_CONFIG,
        "lstm_params_used": LSTM_CONFIG,
        "lstm_training_summary": lstm_summary,
        "stdk_elapsed_seconds": float(stdk_elapsed),
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

    del model, lstm_model
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
            "Model": "STDK+SharedLSTM",
            "Split": "TEST_MEAN",
            **{key: float(value) for key, value in mean_metrics.items()},
        }
    ]
)

print("\n===== STDK TEST RESULTS (seed 41~45 mean) =====")
print(summary_metrics.to_string(index=False))
summary_metrics_std = pd.DataFrame(
    [
        {
            "Model": "STDK+SharedLSTM",
            "Split": "TEST_STD",
            **{key: float(value) for key, value in std_metrics.items()},
        }
    ]
)
print("\n===== STDK TEST RESULTS (seed 41~45 std) =====")
print(summary_metrics_std.to_string(index=False))
print(f"Mean training elapsed={_fmt_time(elapsed_mean)}")

payload = {
    "Model": "STDK+SharedLSTM",
    "Split": "TEST_MEAN",
    "seed_range": [int(SEED_LIST[0]), int(SEED_LIST[-1])],
    "seed_list": [int(seed) for seed in SEED_LIST],
    "mean_metrics": {key: float(value) for key, value in mean_metrics.items()},
    "std_metrics": {key: float(value) for key, value in std_metrics.items()},
    "avg_elapsed_seconds": float(elapsed_mean),
    "params_used": STDK_CONFIG,
    "lstm_params_used": LSTM_CONFIG,
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

print("Repo STDK run completed.")
