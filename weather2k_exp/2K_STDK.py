import importlib.util
import contextlib
import copy
import gc
import json
import logging
import os
import sys
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

#
# 這個檔案 (2K_STDK.py)
# - 目的：在 Weather2K 資料上執行 STDK 基準 (使用 spatial-adapter/examples/baselines/stdk/st_interp.py)
# - 功能摘要：讀取 weather2k.npy → 抽樣站點 → 建立 STDK 所需的平坦輸入 → 訓練模型 → 對全部 500 個站點做預測並輸出評估指標
# - 註解語言：繁體中文（函式與主要區段已逐一註解）
#


def _fmt_time(s):
    m, sec = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h:d}h{m:02d}m{sec:02d}s"


# 將 Python 秒數轉成人可讀的時:分:秒 字串，供訓練耗時顯示使用


def _json_dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# 將 Python 物件以 pretty JSON 寫入指定路徑，會建立父目錄（如需要）


def _cleanup_torch_cache() -> None:
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# 嘗試清理 CUDA 快取（若可用），避免記憶體殘留影響後續執行


def _silent_call(func, *args, **kwargs):
    with open(os.devnull, "w") as fnull:
        with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
            return func(*args, **kwargs)


# 在不想要外部函式輸出到 stdout/stderr 時使用的包裝器（靜音呼叫）


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    return {
        "rmse": float(rmse),
        "mse": float(mse),
        "mae": float(mae),
        "r2": float(r2),
    }


# 計算常用回歸評估指標（MSE/MAE/RMSE/R2），回傳字典格式


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


# 自訂 Dataset：輸入為已平坦化的 X, coords, t 與對應的 y
# DataLoader 會使用 collate_fn 將 batch 聚合回 tensor


def collate_fn(batch):
    return {
        "X": torch.stack([b["X"] for b in batch]),
        "coords": torch.stack([b["coords"] for b in batch]),
        "t": torch.stack([b["t"] for b in batch]),
        "y": torch.stack([b["y"] for b in batch]),
    }


# 自訂 collate 函式，將 list-of-dict 轉為 batch tensors


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


# 建立 STDK 的固定配置參數（依據使用者提供的圖片/預設值）


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


# 訓練迴圈（含 early stopping 與 epoch/mini-batch 的 tqdm 進度條）
# 回傳最佳驗證集權重載入後的模型


def predict_stdk(model, X, coords, t, batch_size, device):
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


# 使用模型對輸入做批次預測，回傳一維 numpy 預測向量（對應於輸入順序）


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


# 將 (n_sites, n_times) 的 y_matrix 轉成 STDK 所需的平坦化輸入格式：
# - X_flat: (n_sites*n_times, 0)（目前無 exogenous covariates）
# - coords_flat: 每個時間點重複所有站點的座標
# - t_flat: 每個站點在每個時間的時間編碼
# - y_flat: 對應的觀測值，作為訓練目標


def normalize_coords(coords):
    coords = np.asarray(coords, dtype=np.float32)
    min_vals = coords.min(axis=0)
    max_vals = coords.max(axis=0)
    denom = np.where((max_vals - min_vals) < 1e-12, 1.0, max_vals - min_vals)
    return ((coords - min_vals) / denom).astype(np.float32)


# 將經緯度座標正規化到 [0,1] 範圍，避免數值尺度差異影響模型


def _build_stdk_dataset(y_values, time_stride=3):
    time_idx = np.arange(0, y_values.shape[1], time_stride)
    ts_df = pd.DataFrame(
        y_values[:, time_idx].T,
        index=time_index[time_idx],
        columns=[f"cell_{i}" for i in range(y_values.shape[0])],
    ).astype("float32")
    return ts_df, time_idx


# 根據 time_stride 建立時間序列資料表（DataFrame），並回傳對應的時間索引陣列
# ts_df 的 shape 為 (n_downsampled_times, n_sites)


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


# 動態從 workspace 的 spatial-adapter 範例載入 STDK baseline 的 create_model()
# 這樣在 IDE 與執行環境路徑不同時仍可正確匯入


# ============================================================
# 讀 Weather2K NPY 檔案
# 這一段完全照 2K_DLinear_FRK 的寫法
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

time_list = np.array(time_list)
time_index = pd.to_datetime(time_list)

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
# 抽樣：全球 1866 個測站中抽 500，再從 500 中抽 100
# ============================================================
N_SAMPLE_TARGET = 500
N_STDK = 100
N_LAST = 1000
SEED_LIST = list(range(41, 46))

print("\n===== STDK fixed-parameter run (500 -> 100 sampling) =====")

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

STDK_CONFIG = build_stdk_model_config()
SAVE_DIR = Path(".")
RESULT_PATH = SAVE_DIR / "2K_stdk_metrics.json"
TIME_STRIDE = 1  # 使用完整時間序列，不跳過任何時間點


def run_single_seed_experiment(sample_seed: int) -> dict:
    n_sample = min(N_SAMPLE_TARGET, nloc)

    np.random.seed(sample_seed)
    sample_idx_global = np.random.choice(nloc, size=n_sample, replace=False)
    sample_idx_global = np.sort(sample_idx_global)

    y_sample_full = y_all[sample_idx_global, :]
    coords_sample_full = gg[sample_idx_global, :]

    print(f"抽樣 {n_sample} 個測站")
    print(f"抽樣 {n_sample} 個測站（將從中再抽取 {N_STDK} 個進行 STDK）")

    n_stdk = min(N_STDK, n_sample)
    np.random.seed(sample_seed)
    sample_idx_local = np.random.choice(n_sample, size=n_stdk, replace=False)
    sample_idx_local = np.sort(sample_idx_local)

    y_sample = y_sample_full[sample_idx_local, :]
    coords_sample = coords_sample_full[sample_idx_local, :]
    n_sample = y_sample.shape[0]

    print(f"STDK 使用樣本數: {n_sample}")
    print(f"\n===== STDK seed {sample_seed} =====")
    print("Preparing STDK dataset...")

    ts_df, used_time_idx = _build_stdk_dataset(y_sample, time_stride=TIME_STRIDE)
    print("Total time steps (original):", ntime)
    print("SAMPLE_SIZE:", n_sample)
    print("ts_df shape (before subsetting):", ts_df.shape)

    if len(ts_df) > N_LAST:
        ts_df = ts_df.iloc[-N_LAST:]
        used_time_idx = used_time_idx[-N_LAST:]
        print(f"Subsetting to last {N_LAST} timepoints")
    else:
        print(f"Requested last {N_LAST} timepoints but only {len(ts_df)} available; using all available time steps")

    print("ts_df shape (used):", ts_df.shape)

    full_y_matrix = y_sample_full[:, used_time_idx].astype(np.float32)
    obs_y_matrix = y_sample_full[sample_idx_local, :][:, used_time_idx].astype(np.float32)

    coords_full_norm = normalize_coords(coords_sample_full)
    coords_obs_norm = coords_full_norm[sample_idx_local, :]

    n_full = coords_full_norm.shape[0]
    n_obs = coords_obs_norm.shape[0]
    n_time_used = len(ts_df)

    train_frac, val_frac, test_frac = 0.7, 0.15, 0.15
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("train_frac + val_frac + test_frac 必須等於 1")

    cut_train = int(n_time_used * train_frac)
    cut_val = int(n_time_used * (train_frac + val_frac))

    train_time_idx = np.arange(0, cut_train)
    val_time_idx = np.arange(cut_train, cut_val)
    test_time_idx = np.arange(cut_val, n_time_used)

    print(f"Total time steps (used): {n_time_used}")
    print(f"Train len: {len(train_time_idx)}, Val len: {len(val_time_idx)}, Test len: {len(test_time_idx)}")

    t_norm_all = np.linspace(0.0, 1.0, n_time_used, dtype=np.float32)

    train_y_raw = obs_y_matrix[:, train_time_idx].reshape(-1)
    y_mean = float(np.mean(train_y_raw))
    y_std = float(np.std(train_y_raw, ddof=0))
    if y_std < 1e-12:
        y_std = 1.0

    def to_std(y):
        return ((y - y_mean) / y_std).astype(np.float32)

    def to_raw(y_std_arr):
        return y_std_arr * y_std + y_mean

    X_train, coords_train, t_train, y_train = build_flat_inputs(
        obs_y_matrix, coords_obs_norm, train_time_idx, t_norm_all
    )
    X_val, coords_val, t_val, y_val = build_flat_inputs(
        obs_y_matrix, coords_obs_norm, val_time_idx, t_norm_all
    )
    X_test_obs, coords_test_obs, t_test_obs, y_test_obs = build_flat_inputs(
        obs_y_matrix, coords_obs_norm, test_time_idx, t_norm_all
    )
    X_test_full, coords_test_full, t_test_full, y_test_full = build_flat_inputs(
        full_y_matrix, coords_full_norm, test_time_idx, t_norm_all
    )

    y_train = to_std(y_train)
    y_val = to_std(y_val)
    y_test_obs = to_std(y_test_obs)
    y_test_full = to_std(y_test_full)

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
    elapsed = time.time() - train_start

    pred_test_obs_std = _silent_call(
        predict_stdk,
        model,
        torch.from_numpy(X_test_obs),
        torch.from_numpy(coords_test_obs),
        torch.from_numpy(t_test_obs),
        batch_size,
        device,
    )
    pred_test_full_std = _silent_call(
        predict_stdk,
        model,
        torch.from_numpy(X_test_full),
        torch.from_numpy(coords_test_full),
        torch.from_numpy(t_test_full),
        batch_size,
        device,
    )

    pred_test_obs = to_raw(pred_test_obs_std).reshape(len(test_time_idx), n_obs)
    pred_test_full = to_raw(pred_test_full_std).reshape(len(test_time_idx), n_full)

    test_true_obs = obs_y_matrix[:, test_time_idx].T.astype(np.float32)
    test_true_full = full_y_matrix[:, test_time_idx].T.astype(np.float32)

    unknown_idx = np.setdiff1d(np.arange(n_full), sample_idx_local)
    pred_test_unknown = pred_test_full[:, unknown_idx]
    test_true_unknown = test_true_full[:, unknown_idx]

    full_mse = mean_squared_error(test_true_full, pred_test_full)
    full_mae = mean_absolute_error(test_true_full, pred_test_full)
    full_rmse = float(np.sqrt(full_mse))
    full_r2 = r2_score(test_true_full, pred_test_full)

    unknown_mse = mean_squared_error(test_true_unknown, pred_test_unknown)
    unknown_mae = mean_absolute_error(test_true_unknown, pred_test_unknown)
    unknown_rmse = float(np.sqrt(unknown_mse))
    unknown_r2 = r2_score(test_true_unknown, pred_test_unknown)

    metrics = {
        "Full500_RMSE": float(full_rmse),
        "Full500_MSE": float(full_mse),
        "Full500_MAE": float(full_mae),
        "Full500_R2": float(full_r2),
        "Unknown400_RMSE": float(unknown_rmse),
        "Unknown400_MSE": float(unknown_mse),
        "Unknown400_MAE": float(unknown_mae),
        "Unknown400_R2": float(unknown_r2),
    }

    print(f"Training elapsed={_fmt_time(elapsed)}")
    print(pd.DataFrame([{"Model": "STDK", "Split": f"TEST_seed_{sample_seed}", **metrics}]).to_string(index=False))

    payload = {
        "seed": int(sample_seed),
        "Model": "STDK",
        "Split": "TEST",
        "Full500": {
            "RMSE": float(full_rmse),
            "MSE": float(full_mse),
            "MAE": float(full_mae),
            "R2": float(full_r2),
        },
        "Unknown400": {
            "RMSE": float(unknown_rmse),
            "MSE": float(unknown_mse),
            "MAE": float(unknown_mae),
            "R2": float(unknown_r2),
        },
        "elapsed_seconds": float(elapsed),
        "params_used": STDK_CONFIG,
        "sampling_info": {
            "full_sample_size": int(N_SAMPLE_TARGET),
            "stdk_sample_size": int(N_STDK),
            "sample_seed": int(sample_seed),
            "time_stride": int(TIME_STRIDE),
            "n_last_timepoints": int(N_LAST),
            "sample_idx_global": sample_idx_global.tolist(),
            "sample_idx_local": sample_idx_local.tolist(),
            "train_frac": train_frac,
            "val_frac": val_frac,
            "test_frac": test_frac,
        },
        "n_full": int(n_full),
        "n_obs": int(n_obs),
        "n_unknown": int(len(unknown_idx)),
        "n_time_used": int(n_time_used),
    }

    del model
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
            "Model": "STDK",
            "Split": "TEST_MEAN",
            **{key: float(value) for key, value in mean_metrics.items()},
        }
    ]
)

print("\n===== STDK TEST RESULTS (seed 41~45 average) =====")
print(summary_metrics.to_string(index=False))
print(f"Average training elapsed={_fmt_time(elapsed_mean)}")

payload = {
    "Model": "STDK",
    "Split": "TEST_MEAN",
    "seed_range": [int(SEED_LIST[0]), int(SEED_LIST[-1])],
    "seed_list": [int(seed) for seed in SEED_LIST],
    "mean_metrics": {key: float(value) for key, value in mean_metrics.items()},
    "std_metrics": {key: float(value) for key, value in std_metrics.items()},
    "avg_elapsed_seconds": float(elapsed_mean),
    "params_used": STDK_CONFIG,
    "sampling_info": {
        "full_sample_size": int(N_SAMPLE_TARGET),
        "stdk_sample_size": int(N_STDK),
        "time_stride": int(TIME_STRIDE),
        "n_last_timepoints": int(N_LAST),
    },
    "seed_runs": seed_runs,
}

_json_dump(payload, RESULT_PATH)
print(f"Saved metrics: {RESULT_PATH.resolve()}")

print("Repo STDK run completed.")