import contextlib
import gc
import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
import torch

from darts import TimeSeries
from darts.models import DLinearModel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 讀 Weather2K NPY 檔案
# 這一段完全照 2K_STDK 的寫法
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

# 1. 讀 NPY 檔案 (location, feature, time)
data = np.load(dataset_path)

# 2. 轉置成 (location, time, feature)
data = np.transpose(data, (0, 2, 1))

# 3. 分離位置資訊與氣象變數
loc = data[:, 0, :3]  # (1866, 3)
data = data[:, :, 3:]  # (1866, 13632, 10)

# 4. 生成時間序列（2017-01-01 到 2021-08-31，每 3 小時）
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

# 氣象變數名稱
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

# 使用氣溫
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
n_sample = min(N_SAMPLE_TARGET, nloc)

DATA_SEED = 42
np.random.seed(DATA_SEED)
sample_idx_global = np.random.choice(nloc, size=n_sample, replace=False)
sample_idx_global = np.sort(sample_idx_global)

y_sample_full = y_all[sample_idx_global, :]
coords_sample_full = gg[sample_idx_global, :]

print(f"抽樣 {n_sample} 個測站")
print(f"抽樣 {n_sample} 個測站（將從中再抽取 100 個進行 DLinear）")

N_STDK = 100
n_stdk = min(N_STDK, n_sample)
np.random.seed(DATA_SEED)
sample_idx_local = np.random.choice(n_sample, size=n_stdk, replace=False)
sample_idx_local = np.sort(sample_idx_local)

y_sample = y_sample_full[sample_idx_local, :]
coords_sample = coords_sample_full[sample_idx_local, :]
n_sample = y_sample.shape[0]

print(f"DLinear 使用樣本數: {n_sample}")

# Tune/sample constants for reproducibility and reporting
TUNE_SAMPLE_SIZE = N_STDK
 


def _fmt_time(s):
    m, sec = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h:d}h{m:02d}m{sec:02d}s"


def _json_dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _json_load(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _cleanup_torch_cache() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _silent_call(func, *args, **kwargs):
    with open(os.devnull, "w") as fnull:
        with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
            return func(*args, **kwargs)


def _ts_to_2d(ts: TimeSeries) -> np.ndarray:
    arr = np.asarray(ts.all_values(copy=False))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.float64, copy=False)


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


def _build_model_kwargs(in_len, out_len, ksize, n_epochs, bs, lr, wd, const_init, random_state=42):
    return {
        "input_chunk_length": int(in_len),
        "output_chunk_length": int(out_len),
        "kernel_size": int(ksize),
        "n_epochs": int(n_epochs),
        "random_state": int(random_state),
        "use_reversible_instance_norm": bool(USE_REVIN),
        "batch_size": int(bs),
        "optimizer_kwargs": {
            "lr": float(lr),
            "weight_decay": float(wd),
        },
        "const_init": bool(const_init),
        "pl_trainer_kwargs": PL_TRAINER_KWARGS,
        "log_tensorboard": False,
        "save_checkpoints": False,
    }


# ============================================================
# DLinear + Optuna 設定
# ============================================================
print("\n===== DLinear hyperparameter search (500 -> 100 sampling) =====")

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

RUN_SEARCH = True  # 是否執行超參數搜尋，若 False 則直接讀取 BEST_PARAMS_PATH 的參數重訓並評估
N_TRIALS = 500
TIMEOUT_SECONDS = None

USE_COVARIATES = False
USE_REVIN = True
RANDOM_STATE = 42

PL_TRAINER_KWARGS = {
    "enable_progress_bar": False,
    "logger": False,
    "enable_checkpointing": False,
    "enable_model_summary": False,
}

SAVE_DIR = Path(script_dir)
BEST_RESULT_PATH = SAVE_DIR / "2K_best_dlinear_result_500to100.json"
BEST_PARAMS_PATH = SAVE_DIR / "2K_best_dlinear_params_500to100.json"
RERUN_METRICS_PATH = SAVE_DIR / "2K_best_dlinear_rerun_metrics_500to100.json"
FRK_METRICS_PATH = SAVE_DIR / "dlinear_autofrk_test_100to500_metrics.json"

HAS_CACHED_PARAMS = BEST_PARAMS_PATH.exists()
if HAS_CACHED_PARAMS:
    print(f"Found cached params: {BEST_PARAMS_PATH}")
    if BEST_RESULT_PATH.exists():
        print(f"Found cached result: {BEST_RESULT_PATH}")
    else:
        print(f"Cached params exist but result file is missing: {BEST_RESULT_PATH}")
else:
    print("No cached params found; Optuna search will run.")

SEARCH_SPACE = {
    "INPUT_CHUNK_LENGTH": [24, 36, 48],
    "OUTPUT_CHUNK_LENGTH": [12, 24],
    "KERNEL_SIZE": [15, 25],
    "N_EPOCHS": [30, 50],
    "BATCH_SIZE": [16, 32, 64],
    "LR": [1e-4, 2e-4, 3e-4],
    "WEIGHT_DECAY": [0.0, 1e-5],
    "CONST_INIT": [True, False],
}

keys = list(SEARCH_SPACE.keys())


# ============================================================
# 依照 STDK 的方式建立資料表與切分
# ============================================================
print("Preparing repo DLinear dataset...")
TIME_STRIDE = 1   # 使用完整時間序列，不跳過任何時間點
EVAL_SEEDS = list(range(41, 46))


def _build_darts_dataset(y_values, time_stride=TIME_STRIDE):
    time_idx = np.arange(0, y_values.shape[1], time_stride)
    ts_df = pd.DataFrame(
        y_values[:, time_idx].T,
        index=time_index[time_idx],
        columns=[f"cell_{i}" for i in range(y_values.shape[0])],
    ).astype("float32")
    return ts_df, time_idx


ts_df, used_time_idx = _build_darts_dataset(y_sample, time_stride=TIME_STRIDE)

print("Total time steps (original):", ntime)
print("SAMPLE_SIZE:", n_sample)
print("ts_df shape (before subsetting):", ts_df.shape)

# Use only the last N timepoints as requested
N_LAST = 1000
if len(ts_df) > N_LAST:
    ts_df = ts_df.iloc[-N_LAST:]
    used_time_idx = used_time_idx[-N_LAST:]
    print(f"Subsetting to last {N_LAST} timepoints")
else:
    print(f"Requested last {N_LAST} timepoints but only {len(ts_df)} available; using all available time steps")

print("ts_df shape (used):", ts_df.shape)

month_values = time_index.month.astype("float32")
month_df = pd.DataFrame({"month": month_values}, index=time_index)
month_ts = TimeSeries.from_dataframe(month_df.astype("float32"))

train_frac, val_frac, test_frac = 0.7, 0.15, 0.15
if not np.isclose(train_frac + val_frac + test_frac, 1.0):
    raise ValueError("train_frac + val_frac + test_frac 必須等於 1")

cut_train = int(len(ts_df) * train_frac)
cut_val = int(len(ts_df) * (train_frac + val_frac))

train_raw = TimeSeries.from_dataframe(ts_df.iloc[:cut_train])
trainval_raw = TimeSeries.from_dataframe(ts_df.iloc[:cut_val])
test_raw = TimeSeries.from_dataframe(ts_df.iloc[cut_val:])
val_raw = TimeSeries.from_dataframe(ts_df.iloc[cut_train:cut_val])

trainval_ts = TimeSeries.from_dataframe(ts_df.iloc[:cut_val])
train_ts = TimeSeries.from_dataframe(ts_df.iloc[:cut_train])
val_ts = TimeSeries.from_dataframe(ts_df.iloc[cut_train:cut_val])
test_ts = TimeSeries.from_dataframe(ts_df.iloc[cut_val:])

month_trainval = month_ts[:cut_val]
month_train = month_ts[:cut_train]
month_val = month_ts[cut_train:cut_val]
month_test = month_ts[cut_val:]

print(f"Train len (raw): {len(train_raw)}, Val len (raw): {len(val_raw)}, Test len (raw): {len(test_raw)}")

train_df = ts_df.iloc[:cut_train]
mean_vec = train_df.mean(axis=0)
std_vec = train_df.std(axis=0).replace(0.0, 1.0)

ts_df_scaled = (ts_df - mean_vec) / std_vec
series_scaled = TimeSeries.from_dataframe(ts_df_scaled)
trainval_scaled = TimeSeries.from_dataframe(ts_df_scaled.iloc[:cut_val])
train_scaled = TimeSeries.from_dataframe(ts_df_scaled.iloc[:cut_train])
val_scaled = TimeSeries.from_dataframe(ts_df_scaled.iloc[cut_train:cut_val])
test_scaled = TimeSeries.from_dataframe(ts_df_scaled.iloc[cut_val:])

T_train, T_val, T_test = len(train_scaled), len(val_scaled), len(test_scaled)
print("T_train:", T_train, "T_val:", T_val, "T_test:", T_test)

val_true_raw = ts_df.to_numpy(dtype=np.float32)[cut_train:cut_val]
test_true_raw = ts_df.to_numpy(dtype=np.float32)[cut_val:]


def inverse_scale(x):
    return x * std_vec.values + mean_vec.values


sampling_info = {
    "full_sample_size": int(N_SAMPLE_TARGET),
    "tune_sample_size": int(TUNE_SAMPLE_SIZE),
    "sample_seed": int(DATA_SEED),
    "time_stride": int(TIME_STRIDE),
    "sample_idx_global": sample_idx_global.tolist(),
    "sample_idx_local": sample_idx_local.tolist(),
}

# 記錄實際使用的最後時間點數（若有套用子集）
try:
    sampling_info["n_last_timepoints"] = int(N_LAST)
except NameError:
    sampling_info["n_last_timepoints"] = None


# ============================================================
# Optuna 搜尋
# ============================================================
best_params = None
best_val_metrics = None

if HAS_CACHED_PARAMS:
    best_params = _json_load(BEST_PARAMS_PATH)
    if BEST_RESULT_PATH.exists():
        cached_result = _json_load(BEST_RESULT_PATH)
        best_val_metrics = {
            "rmse_val": float(cached_result.get("rmse_val", float("nan"))),
            "mse_val": float(cached_result.get("mse_val", float("nan"))),
            "mae_val": float(cached_result.get("mae_val", float("nan"))),
            "r2_val": float(cached_result.get("r2_val", float("nan"))),
        }
    print("\n===== LOADED CACHED DLinear PARAMS =====")
    print(f"Using params from: {BEST_PARAMS_PATH}")
    if best_val_metrics is not None:
        print(f"Cached VAL RMSE: {best_val_metrics['rmse_val']:.6f}")

elif RUN_SEARCH:

    def objective(trial):
        print(f"[Trial {trial.number + 1}/{N_TRIALS}]", flush=True)

        model_kwargs = _build_model_kwargs(
            in_len=trial.suggest_categorical("INPUT_CHUNK_LENGTH", SEARCH_SPACE["INPUT_CHUNK_LENGTH"]),
            out_len=trial.suggest_categorical("OUTPUT_CHUNK_LENGTH", SEARCH_SPACE["OUTPUT_CHUNK_LENGTH"]),
            ksize=trial.suggest_categorical("KERNEL_SIZE", SEARCH_SPACE["KERNEL_SIZE"]),
            n_epochs=trial.suggest_categorical("N_EPOCHS", SEARCH_SPACE["N_EPOCHS"]),
            bs=trial.suggest_categorical("BATCH_SIZE", SEARCH_SPACE["BATCH_SIZE"]),
            lr=trial.suggest_categorical("LR", SEARCH_SPACE["LR"]),
            wd=trial.suggest_categorical("WEIGHT_DECAY", SEARCH_SPACE["WEIGHT_DECAY"]),
            const_init=trial.suggest_categorical("CONST_INIT", SEARCH_SPACE["CONST_INIT"]),
        )

        model = None
        pred_val = None

        try:
            model = DLinearModel(**model_kwargs)

            fit_kwargs = {"verbose": False}
            pred_kwargs = {"verbose": False, "show_warnings": False}

            if USE_COVARIATES:
                fit_kwargs["past_covariates"] = month_train
                fit_kwargs["future_covariates"] = month_train
                pred_kwargs["past_covariates"] = month_trainval
                pred_kwargs["future_covariates"] = month_trainval

            _silent_call(model.fit, series=train_scaled, **fit_kwargs)
            pred_val = _silent_call(model.predict, n=len(val_scaled), **pred_kwargs)

            pred_val_raw = inverse_scale(_ts_to_2d(pred_val))
            val_metrics = _compute_metrics(val_true_raw, pred_val_raw)

            trial.set_user_attr("rmse_val", val_metrics["rmse"])
            trial.set_user_attr("mse_val", val_metrics["mse"])
            trial.set_user_attr("mae_val", val_metrics["mae"])
            trial.set_user_attr("r2_val", val_metrics["r2"])

            return float(val_metrics["rmse"])
        finally:
            del model
            del pred_val
            gc.collect()
            _cleanup_torch_cache()

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0),
    )
    study.optimize(
        objective,
        n_trials=N_TRIALS,
        timeout=TIMEOUT_SECONDS,
        gc_after_trial=True,
        show_progress_bar=False,
    )

    best_trial = study.best_trial
    best_params = dict(best_trial.params)
    best_val_metrics = {
        "rmse_val": float(best_trial.user_attrs["rmse_val"]),
        "mse_val": float(best_trial.user_attrs["mse_val"]),
        "mae_val": float(best_trial.user_attrs["mae_val"]),
        "r2_val": float(best_trial.user_attrs["r2_val"]),
    }

    print("\n===== SEARCH FINISHED =====")
    print(f"Best trial number: {best_trial.number}")
    print(f"Best validation RMSE: {best_val_metrics['rmse_val']:.6f}")

    best_payload = {
        "rmse_val": best_val_metrics["rmse_val"],
        "mse_val": best_val_metrics["mse_val"],
        "mae_val": best_val_metrics["mae_val"],
        "r2_val": best_val_metrics["r2_val"],
        "params": best_params,
        "best_val_rmse": float(best_trial.value),
        "best_trial_number": int(best_trial.number),
        "sampling_info": sampling_info,
        "n_trials": int(N_TRIALS),
        "search_space": SEARCH_SPACE,
    }
    params_payload = dict(best_params)
    params_payload["sampling_info"] = sampling_info

    _json_dump(best_payload, BEST_RESULT_PATH)
    _json_dump(params_payload, BEST_PARAMS_PATH)

    print("\nSaved:")
    print(BEST_RESULT_PATH)
    print(BEST_PARAMS_PATH)

else:
    raise RuntimeError("RUN_SEARCH = False is not supported in this version. Set it to True for Optuna tuning.")


def _to_2d_np(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return arr.astype(np.float64, copy=False)


def _run_seeded_dlinear_and_frk(sample_seed: int, best_params: dict) -> dict:
    import torch

    try:
        from autoFRK import AutoFRK
    except ImportError as exc:
        raise ImportError(
            "Cannot import autoFRK / torch for the FRK stage. Make sure the package shown in the screenshot is installed."
        ) from exc

    n_sample = min(N_SAMPLE_TARGET, nloc)

    np.random.seed(sample_seed)
    sample_idx_global_seed = np.random.choice(nloc, size=n_sample, replace=False)
    sample_idx_global_seed = np.sort(sample_idx_global_seed)

    y_sample_full_seed = y_all[sample_idx_global_seed, :]
    coords_sample_full_seed = gg[sample_idx_global_seed, :]

    print(f"\n===== DLinear + autoFRK seed {sample_seed} =====")
    print(f"抽樣 {n_sample} 個測站")
    print(f"抽樣 {n_sample} 個測站（將從中再抽取 100 個進行 DLinear）")

    n_stdk = min(N_STDK, n_sample)
    np.random.seed(sample_seed)
    sample_idx_local_seed = np.random.choice(n_sample, size=n_stdk, replace=False)
    sample_idx_local_seed = np.sort(sample_idx_local_seed)

    y_sample_seed = y_sample_full_seed[sample_idx_local_seed, :]
    coords_sample_seed = coords_sample_full_seed[sample_idx_local_seed, :]
    n_sample_seed = y_sample_seed.shape[0]

    print(f"DLinear 使用樣本數: {n_sample_seed}")

    ts_df_seed, used_time_idx_seed = _build_darts_dataset(y_sample_seed, time_stride=TIME_STRIDE)
    print("Total time steps (original):", ntime)
    print("SAMPLE_SIZE:", n_sample_seed)
    print("ts_df shape (before subsetting):", ts_df_seed.shape)

    if len(ts_df_seed) > N_LAST:
        ts_df_seed = ts_df_seed.iloc[-N_LAST:]
        used_time_idx_seed = used_time_idx_seed[-N_LAST:]
        print(f"Subsetting to last {N_LAST} timepoints")
    else:
        print(f"Requested last {N_LAST} timepoints but only {len(ts_df_seed)} available; using all available time steps")

    print("ts_df shape (used):", ts_df_seed.shape)

    month_values = time_index.month.astype("float32")
    month_df = pd.DataFrame({"month": month_values}, index=time_index)
    month_ts = TimeSeries.from_dataframe(month_df.astype("float32"))

    cut_train_seed = int(len(ts_df_seed) * train_frac)
    cut_val_seed = int(len(ts_df_seed) * (train_frac + val_frac))

    train_raw_seed = TimeSeries.from_dataframe(ts_df_seed.iloc[:cut_train_seed])
    val_raw_seed = TimeSeries.from_dataframe(ts_df_seed.iloc[cut_train_seed:cut_val_seed])
    test_raw_seed = TimeSeries.from_dataframe(ts_df_seed.iloc[cut_val_seed:])

    trainval_ts_seed = TimeSeries.from_dataframe(ts_df_seed.iloc[:cut_val_seed])
    train_ts_seed = TimeSeries.from_dataframe(ts_df_seed.iloc[:cut_train_seed])
    test_ts_seed = TimeSeries.from_dataframe(ts_df_seed.iloc[cut_val_seed:])

    month_trainval_seed = month_ts[:cut_val_seed]
    month_train_seed = month_ts[:cut_train_seed]
    month_test_seed = month_ts[cut_val_seed:]

    # 在 seed 重訓階段不使用 val 進行訓練；val 僅保留作為 test 時間對齊間隔
    print(
        f"Train len (raw): {len(train_raw_seed)}, "
        f"Val gap len (raw): {len(val_raw_seed)}, "
        f"Test len (raw): {len(test_raw_seed)}"
    )

    train_df_seed = ts_df_seed.iloc[:cut_train_seed]
    mean_vec_seed = train_df_seed.mean(axis=0)
    std_vec_seed = train_df_seed.std(axis=0).replace(0.0, 1.0)

    ts_df_scaled_seed = (ts_df_seed - mean_vec_seed) / std_vec_seed
    series_scaled_seed = TimeSeries.from_dataframe(ts_df_scaled_seed)
    trainval_scaled_seed = TimeSeries.from_dataframe(ts_df_scaled_seed.iloc[:cut_val_seed])
    train_scaled_seed = TimeSeries.from_dataframe(ts_df_scaled_seed.iloc[:cut_train_seed])
    test_scaled_seed = TimeSeries.from_dataframe(ts_df_scaled_seed.iloc[cut_val_seed:])

    T_train_seed, T_test_seed = len(train_scaled_seed), len(test_scaled_seed)
    print("T_train:", T_train_seed, "T_test:", T_test_seed)

    test_true_raw_seed = ts_df_seed.to_numpy(dtype=np.float32)[cut_val_seed:]

    def inverse_scale_seed(x):
        return x * std_vec_seed.values + mean_vec_seed.values

    model_best_kwargs = _build_model_kwargs(
        in_len=best_params["INPUT_CHUNK_LENGTH"],
        out_len=best_params["OUTPUT_CHUNK_LENGTH"],
        ksize=best_params["KERNEL_SIZE"],
        n_epochs=best_params["N_EPOCHS"],
        bs=best_params["BATCH_SIZE"],
        lr=best_params["LR"],
        wd=best_params["WEIGHT_DECAY"],
        const_init=best_params["CONST_INIT"],
        random_state=sample_seed,
    )

    model_best = DLinearModel(**model_best_kwargs)

    fit_kwargs_best = {"verbose": False}
    pred_kwargs_best = {"verbose": False, "show_warnings": False}

    if USE_COVARIATES:
        fit_kwargs_best["past_covariates"] = month_train_seed
        fit_kwargs_best["future_covariates"] = month_train_seed
        pred_kwargs_best["past_covariates"] = month_ts
        pred_kwargs_best["future_covariates"] = month_ts

    train_start = time.time()
    _silent_call(model_best.fit, series=trainval_scaled_seed, **fit_kwargs_best)
    pred_best = _silent_call(model_best.predict, n=len(test_scaled_seed), **pred_kwargs_best)
    elapsed = time.time() - train_start

    pred_best_raw = inverse_scale_seed(_ts_to_2d(pred_best))
    y_test_true_best = test_true_raw_seed.reshape(-1, n_sample_seed)
    y_test_pred_best = pred_best_raw.reshape(-1, n_sample_seed)

    mse_best = mean_squared_error(y_test_true_best, y_test_pred_best)
    mae_best = mean_absolute_error(y_test_true_best, y_test_pred_best)
    rmse_best = float(np.sqrt(mse_best))
    r2_best = r2_score(y_test_true_best, y_test_pred_best)

    print(f"Training elapsed={_fmt_time(elapsed)}")
    print(
        pd.DataFrame(
            {
                "Model": [f"DLINEAR(best)_seed_{sample_seed}"],
                "Split": ["TEST"],
                "RMSE": [rmse_best],
                "MSE": [float(mse_best)],
                "MAE": [float(mae_best)],
                "R2": [float(r2_best)],
            }
        ).to_string(index=False)
    )

    import torch
    from autoFRK import AutoFRK

    y_sample_full_arr = np.asarray(y_sample_full_seed, dtype=np.float64)
    coords_sample_full_arr = np.asarray(coords_sample_full_seed, dtype=np.float64)
    sample_idx_global_arr = np.asarray(sample_idx_global_seed, dtype=int)
    sample_idx_local_arr = np.asarray(sample_idx_local_seed, dtype=int)

    N_full = y_sample_full_arr.shape[0]
    N_obs = sample_idx_local_arr.shape[0]

    if N_full != 500:
        print(f"[Warning] full sample size is {N_full}, not 500.")

    if N_obs != 100:
        print(f"[Warning] observed sample size is {N_obs}, not 100.")

    unknown_idx = np.setdiff1d(np.arange(N_full), sample_idx_local_arr)
    N_unknown = len(unknown_idx)

    coords_obs = coords_sample_full_arr[sample_idx_local_arr, :]
    coords_full = coords_sample_full_arr
    loc_obs = torch.from_numpy(coords_obs).to(dtype=torch.float64)
    loc_full = torch.from_numpy(coords_full).to(dtype=torch.float64)

    Y_obs_pred_test = np.asarray(pred_best_raw, dtype=np.float64)
    if Y_obs_pred_test.ndim != 2:
        raise ValueError(f"pred_best_raw shape must be 2D, got {Y_obs_pred_test.shape}")

    T_test, pred_n = Y_obs_pred_test.shape
    if pred_n != len(sample_idx_local_arr):
        raise ValueError(
            f"pred_best_raw has {pred_n} columns, but sample_idx_local has {len(sample_idx_local_arr)} entries"
        )

    Y_true_full_test = y_sample_full_arr[:, used_time_idx_seed[cut_val_seed:]].T.astype(np.float64)
    if Y_true_full_test.shape != (T_test, N_full):
        raise ValueError(
            f"Full test truth shape mismatch: {Y_true_full_test.shape} != {(T_test, N_full)}"
        )

    Yhat_full_test = np.zeros((T_test, N_full), dtype=np.float64)
    Yhat_unknown_test = np.zeros((T_test, N_unknown), dtype=np.float64)

    afrk = AutoFRK(dtype=torch.float64, device="cpu")

    for t in range(T_test):
        y_obs_t = Y_obs_pred_test[t, :]

        fit_obj = _silent_call(
            afrk.forward,
            data=torch.from_numpy(y_obs_t.reshape(N_obs, 1)).to(dtype=torch.float64),
            loc=loc_obs,
            method="EM",
            maxit=50,
            tolerance=1e-6,
            n_neighbor=3,
            tps_method="spherical_fast",
        )

        pred_full = _silent_call(
            afrk.predict,
            obj=fit_obj,
            newloc=loc_full,
            se_report=False,
            tps_method="spherical_fast",
        )

        y_full_t = _to_2d_np(pred_full["pred.value"]).reshape(-1)
        if y_full_t.shape[0] != N_full:
            raise ValueError(
                f"FRK prediction at step {t + 1} has length {y_full_t.shape[0]}, expected {N_full}"
            )

        Yhat_full_test[t, :] = y_full_t
        Yhat_unknown_test[t, :] = y_full_t[unknown_idx]

        del fit_obj, pred_full
        gc.collect()
        _cleanup_torch_cache()

    full_mse = mean_squared_error(Y_true_full_test, Yhat_full_test)
    full_mae = mean_absolute_error(Y_true_full_test, Yhat_full_test)
    full_rmse = float(np.sqrt(full_mse))
    full_r2 = r2_score(Y_true_full_test, Yhat_full_test)

    unknown_true = Y_true_full_test[:, unknown_idx]
    unknown_mse = mean_squared_error(unknown_true, Yhat_unknown_test)
    unknown_mae = mean_absolute_error(unknown_true, Yhat_unknown_test)
    unknown_rmse = float(np.sqrt(unknown_mse))
    unknown_r2 = r2_score(unknown_true, Yhat_unknown_test)

    print(
        pd.DataFrame(
            {
                "Model": [f"DLINEAR + autoFRK_seed_{sample_seed}"],
                "Split": ["TEST"],
                "Full500_RMSE": [full_rmse],
                "Full500_MSE": [float(full_mse)],
                "Full500_MAE": [float(full_mae)],
                "Full500_R2": [float(full_r2)],
                "Unknown400_RMSE": [unknown_rmse],
                "Unknown400_MSE": [float(unknown_mse)],
                "Unknown400_MAE": [float(unknown_mae)],
                "Unknown400_R2": [float(unknown_r2)],
            }
        ).to_string(index=False)
    )

    del model_best, pred_best
    gc.collect()
    _cleanup_torch_cache()

    return {
        "seed": int(sample_seed),
        "elapsed_seconds": float(elapsed),
        "dlinear_metrics": {
            "RMSE": float(rmse_best),
            "MSE": float(mse_best),
            "MAE": float(mae_best),
            "R2": float(r2_best),
        },
        "frk_metrics": {
            "Full500_RMSE": float(full_rmse),
            "Full500_MSE": float(full_mse),
            "Full500_MAE": float(full_mae),
            "Full500_R2": float(full_r2),
            "Unknown400_RMSE": float(unknown_rmse),
            "Unknown400_MSE": float(unknown_mse),
            "Unknown400_MAE": float(unknown_mae),
            "Unknown400_R2": float(unknown_r2),
        },
        "sampling_info": {
            "full_sample_size": int(N_SAMPLE_TARGET),
            "tune_sample_size": int(TUNE_SAMPLE_SIZE),
            "sample_seed": int(sample_seed),
            "time_stride": int(TIME_STRIDE),
            "sample_idx_global": sample_idx_global_arr.tolist(),
            "sample_idx_local": sample_idx_local_arr.tolist(),
            "n_last_timepoints": int(N_LAST),
        },
    }


seed_runs = []
for sample_seed in EVAL_SEEDS:
    seed_runs.append(_run_seeded_dlinear_and_frk(sample_seed, best_params))

dlinear_metrics_df = pd.DataFrame([run["dlinear_metrics"] for run in seed_runs])
dlinear_mean_metrics = dlinear_metrics_df.mean(numeric_only=True).to_dict()
dlinear_std_metrics = dlinear_metrics_df.std(numeric_only=True, ddof=0).to_dict()
dlinear_elapsed_mean = float(np.mean([run["elapsed_seconds"] for run in seed_runs]))

frk_metrics_df = pd.DataFrame([run["frk_metrics"] for run in seed_runs])
frk_mean_metrics = frk_metrics_df.mean(numeric_only=True).to_dict()
frk_std_metrics = frk_metrics_df.std(numeric_only=True, ddof=0).to_dict()

print("\n===== DLinear best TEST RESULTS (seed 41~45 mean) =====")
print(
    pd.DataFrame(
        [
            {
                "Model": "DLINEAR(best)",
                "Split": "TEST_MEAN",
                **{key: float(value) for key, value in dlinear_mean_metrics.items()},
            }
        ]
    ).to_string(index=False)
)
print("\n===== DLinear best TEST RESULTS (seed 41~45 std) =====")
print(
    pd.DataFrame(
        [
            {
                "Model": "DLINEAR(best)",
                "Split": "TEST_STD",
                **{key: float(value) for key, value in dlinear_std_metrics.items()},
            }
        ]
    ).to_string(index=False)
)
print(f"Mean training elapsed={_fmt_time(dlinear_elapsed_mean)}")

print("\n===== DLinear + autoFRK TEST RESULTS (seed 41~45 mean) =====")
print(
    pd.DataFrame(
        [
            {
                "Model": "DLINEAR + autoFRK",
                "Split": "TEST_MEAN",
                **{key: float(value) for key, value in frk_mean_metrics.items()},
            }
        ]
    ).to_string(index=False)
)
print("\n===== DLinear + autoFRK TEST RESULTS (seed 41~45 std) =====")
print(
    pd.DataFrame(
        [
            {
                "Model": "DLINEAR + autoFRK",
                "Split": "TEST_STD",
                **{key: float(value) for key, value in frk_std_metrics.items()},
            }
        ]
    ).to_string(index=False)
)

rerun_payload = {
    "Model": "DLINEAR(best)",
    "Split": "TEST_MEAN",
    "seed_range": [int(EVAL_SEEDS[0]), int(EVAL_SEEDS[-1])],
    "seed_list": [int(seed) for seed in EVAL_SEEDS],
    "mean_metrics": {key: float(value) for key, value in dlinear_mean_metrics.items()},
    "std_metrics": {key: float(value) for key, value in dlinear_std_metrics.items()},
    "avg_elapsed_seconds": float(dlinear_elapsed_mean),
    "params_used": best_params,
    "sampling_info": {
        "full_sample_size": int(N_SAMPLE_TARGET),
        "tune_sample_size": int(TUNE_SAMPLE_SIZE),
        "time_stride": int(TIME_STRIDE),
        "n_last_timepoints": int(N_LAST),
    },
    "seed_runs": seed_runs,
}

frk_payload = {
    "Model": "DLINEAR + autoFRK",
    "Split": "TEST_MEAN",
    "seed_range": [int(EVAL_SEEDS[0]), int(EVAL_SEEDS[-1])],
    "seed_list": [int(seed) for seed in EVAL_SEEDS],
    "mean_metrics": {key: float(value) for key, value in frk_mean_metrics.items()},
    "std_metrics": {key: float(value) for key, value in frk_std_metrics.items()},
    "params_used": best_params,
    "sampling_info": {
        "full_sample_size": int(N_SAMPLE_TARGET),
        "tune_sample_size": int(TUNE_SAMPLE_SIZE),
        "time_stride": int(TIME_STRIDE),
        "n_last_timepoints": int(N_LAST),
    },
    "seed_runs": seed_runs,
}

_json_dump(rerun_payload, RERUN_METRICS_PATH)
_json_dump(frk_payload, FRK_METRICS_PATH)

print(f"\nSaved re-run metrics: {RERUN_METRICS_PATH.resolve()}")
print(f"Saved FRK metrics: {FRK_METRICS_PATH.resolve()}")
print("Repo DLinear + autoFRK run completed.")