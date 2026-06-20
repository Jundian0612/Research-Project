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
import torch.nn.functional as F

from darts import TimeSeries
from darts.models import DLinearModel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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

# 目前實驗目標：氣壓
target_var_name = "air_pressure"
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
# 抽樣（可由環境變數覆寫）
# ============================================================
_env = os.environ
EXPERIMENT_SCENARIO = _env.get("EXPERIMENT_SCENARIO", "spatiotemp_100x150")


def _scenario_default(key, fallback):
    defaults = {
        "time_extrap_fixed500": {
            "N_SAMPLE_TARGET": "600",
            "N_STDK": "100",
            "N_LAST": "1000",
        },
        "space_extrap_fixed850": {
            "N_SAMPLE_TARGET": "600",
            "N_STDK": "100",
            "N_LAST": "1000",
        },
        "spatiotemp_100x150": {
            "N_SAMPLE_TARGET": "600",
            "N_STDK": "100",
            "N_LAST": "1000",
        },
    }
    return defaults.get(EXPERIMENT_SCENARIO, {}).get(key, fallback)


N_SAMPLE_TARGET = int(_env.get("N_SAMPLE_TARGET", _scenario_default("N_SAMPLE_TARGET", "600")))
N_STDK = int(_env.get("N_STDK", _scenario_default("N_STDK", "100")))
DATA_SEED = int(_env.get("DATA_SEED", "42"))
N_LAST = int(_env.get("N_LAST", _scenario_default("N_LAST", "1000")))
TIME_TRAIN_LEN = int(_env.get("TIME_TRAIN_LEN", "700"))
TIME_VAL_LEN = int(_env.get("TIME_VAL_LEN", "150"))
TIME_TEST_LEN = int(_env.get("TIME_TEST_LEN", "150"))
TIME_STRIDE = int(_env.get("TIME_STRIDE", "1"))

n_sample = min(N_SAMPLE_TARGET, nloc)
np.random.seed(DATA_SEED)
sample_idx_global = np.random.choice(nloc, size=n_sample, replace=False)
sample_idx_global = np.sort(sample_idx_global)

y_sample_full = y_all[sample_idx_global, :]
coords_sample_full = gg[sample_idx_global, :]

print(f"抽樣 {n_sample} 個測站")
print(f"抽樣 {n_sample} 個測站（將從中再抽取 {N_STDK} 個進行 DLinear）")

n_stdk = min(N_STDK, n_sample)
np.random.seed(DATA_SEED)
sample_idx_local = np.random.choice(n_sample, size=n_stdk, replace=False)
sample_idx_local = np.sort(sample_idx_local)

y_sample = y_sample_full[sample_idx_local, :]
coords_sample = coords_sample_full[sample_idx_local, :]
n_sample = y_sample.shape[0]

print(f"DLinear 使用樣本數: {n_sample}")
TUNE_SAMPLE_SIZE = N_STDK


# ============================================================
# 輔助函式
# ============================================================
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


def _suppress_autofrk_logs() -> None:
    class _NoAutoFRKFilter(logging.Filter):
        def filter(self, record):
            return not record.name.startswith("autoFRK")

    # autoFRK 會自己掛 INFO handler，這裡直接關掉相關 logger，
    # 並在 root handler 補 filter，避免綠色 INFO 訊息洗版。
    for logger_name in ("autoFRK", "autoFRK.utils", "autoFRK.utils.logger"):
        logger_obj = logging.getLogger(logger_name)
        logger_obj.disabled = True
        logger_obj.setLevel(logging.CRITICAL + 1)
        logger_obj.propagate = False
        for handler in logger_obj.handlers:
            handler.setLevel(logging.CRITICAL + 1)
            if not any(isinstance(flt, _NoAutoFRKFilter) for flt in handler.filters):
                handler.addFilter(_NoAutoFRKFilter())

    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if not any(isinstance(flt, _NoAutoFRKFilter) for flt in handler.filters):
            handler.addFilter(_NoAutoFRKFilter())


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
    return {"rmse": float(rmse), "mse": float(mse), "mae": float(mae), "r2": float(r2)}



def _build_model_kwargs(
    in_len,
    out_len,
    ksize,
    n_epochs,
    bs,
    lr,
    wd,
    const_init,
    random_state=42,
    loss_fn=None,
    save_checkpoints=False,
    model_name=None,
    work_dir=None,
):
    kwargs = {
        "input_chunk_length": int(in_len),
        "output_chunk_length": int(out_len),
        "kernel_size": int(ksize),
        "n_epochs": int(n_epochs),
        "random_state": int(random_state),
        "use_reversible_instance_norm": bool(USE_REVIN),
        "batch_size": int(bs),
        "optimizer_kwargs": {"lr": float(lr), "weight_decay": float(wd)},
        "const_init": bool(const_init),
        "pl_trainer_kwargs": PL_TRAINER_KWARGS,
        "log_tensorboard": False,
        "save_checkpoints": bool(save_checkpoints),
    }
    if model_name is not None:
        kwargs["model_name"] = model_name
    if work_dir is not None:
        kwargs["work_dir"] = str(work_dir)
    if loss_fn is not None:
        kwargs["loss_fn"] = loss_fn
    return kwargs



def _build_darts_dataset(y_values, time_stride=1):
    time_idx = np.arange(0, y_values.shape[1], time_stride)
    ts_df = pd.DataFrame(
        y_values[:, time_idx].T,
        index=time_index[time_idx],
        columns=[f"cell_{i}" for i in range(y_values.shape[0])],
    ).astype("float32")
    return ts_df, time_idx


class FRKConsistencyLoss(torch.nn.Module):
    """混合式 loss = 監督式 MSE + 時序差分正則 + FRK 空間一致性正則項。

    這個 loss 仍以監督式 MSE 為主，但不再只看最後一步的 batch 平均，
    而是先對整段 forecast window 做加權摘要，再用 FRK 把空間場拉回
    更合理的外推形狀。額外的 temporal 項用來抑制輸出窗內的抖動。
    """

    def __init__(
        self,
        obs_coords,
        full_coords,
        obs_idx_in_full,
        frk_weight=0.01,
        temporal_weight=0.05,
        obs_recon_weight=1.0,
        unobs_mean_weight=0.0,
        unobs_std_weight=0.0,
        apply_every=1,
    ):
        super().__init__()
        self.obs_coords = np.asarray(obs_coords, dtype=np.float64)
        self.full_coords = np.asarray(full_coords, dtype=np.float64)
        self.obs_idx_in_full = np.asarray(obs_idx_in_full, dtype=int)
        self.unobs_idx_in_full = np.setdiff1d(np.arange(self.full_coords.shape[0]), self.obs_idx_in_full)
        self.frk_weight = float(frk_weight)
        self.temporal_weight = float(temporal_weight)
        self.obs_recon_weight = float(obs_recon_weight)
        self.unobs_mean_weight = float(unobs_mean_weight)
        self.unobs_std_weight = float(unobs_std_weight)
        self.apply_every = max(int(apply_every), 1)
        self._call_count = 0

    @staticmethod
    def _weighted_horizon_summary(pred_tensor: torch.Tensor) -> torch.Tensor:
        """Collapse [batch, horizon, series] to a single spatial vector.

        Later forecast steps receive a slightly larger weight because they are
        the ones that dominate the downstream test windows after rollout.
        """
        if pred_tensor.ndim != 3:
            raise ValueError(f"Expected pred tensor with 3 dims, got {tuple(pred_tensor.shape)}")

        horizon_weights = torch.linspace(
            1.0,
            2.0,
            steps=pred_tensor.shape[1],
            device=pred_tensor.device,
            dtype=pred_tensor.dtype,
        )
        horizon_weights = horizon_weights / horizon_weights.sum()
        weighted_horizon = (pred_tensor * horizon_weights.view(1, -1, 1)).sum(dim=1)
        return weighted_horizon.mean(dim=0)

    def _frk_predict_full(self, obs_vector: np.ndarray) -> np.ndarray:
        _suppress_autofrk_logs()
        try:
            from autoFRK import AutoFRK
        except ImportError as exc:
            raise ImportError(
                "Cannot import autoFRK. Install the package used by the existing DLinear + FRK script."
            ) from exc
        try:
            afrk = AutoFRK(dtype=torch.float64, device="cpu")
            y_obs_t = torch.from_numpy(obs_vector.reshape(-1, 1)).to(dtype=torch.float64)
            fit_obj = afrk.forward(
                data=y_obs_t,
                loc=torch.from_numpy(self.obs_coords).to(dtype=torch.float64),
                method="EM",
                maxit=20,
                tolerance=1e-6,
                n_neighbor=FRK_TRAIN_N_NEIGHBOR,
                tps_method="spherical_fast",
            )
            pred_full = afrk.predict(
                obj=fit_obj,
                newloc=torch.from_numpy(self.full_coords).to(dtype=torch.float64),
                se_report=False,
                tps_method="spherical_fast",
            )
            pred_value = pred_full["pred.value"]
            if isinstance(pred_value, torch.Tensor):
                pred_value = pred_value.detach().cpu().numpy()
            return np.asarray(pred_value, dtype=np.float64).reshape(-1)
        except Exception as exc:
            # If FRK fitting fails (e.g. too few locations), fall back to a conservative
            # constant field equal to the mean of observed values to avoid crashing.
            print(f"[Warning] autoFRK failed in _frk_predict_full: {exc}; falling back to mean-field.")
            mean_val = float(np.mean(obs_vector))
            return np.full((self.full_coords.shape[0],), fill_value=mean_val, dtype=np.float64)

    def forward(self, pred, target):
        base_loss = F.mse_loss(pred, target)
        if pred.ndim == 3 and pred.shape[1] > 1:
            pred_step_diff = pred[:, 1:, :] - pred[:, :-1, :]
            target_step_diff = target[:, 1:, :] - target[:, :-1, :]
            temporal_loss = F.mse_loss(pred_step_diff, target_step_diff)
            base_loss = base_loss + self.temporal_weight * temporal_loss

        if self.frk_weight <= 0.0:
            return base_loss

        self._call_count += 1
        if self._call_count % self.apply_every != 0:
            return base_loss

        with torch.no_grad():
            pred_detached = pred.detach().float().cpu()
            pred_summary = self._weighted_horizon_summary(pred_detached).numpy().astype(np.float64)
            frk_full = self._frk_predict_full(pred_summary)
            frk_obs = frk_full[self.obs_idx_in_full]
            frk_unobs = frk_full[self.unobs_idx_in_full]

        pred_summary_t = self._weighted_horizon_summary(pred)
        frk_obs_t = torch.as_tensor(frk_obs, dtype=pred.dtype, device=pred.device)

        obs_recon_loss = F.mse_loss(pred_summary_t, frk_obs_t)
        frk_loss = self.obs_recon_weight * obs_recon_loss
        if frk_unobs.size > 0:
            frk_unobs_mean_t = torch.as_tensor(float(np.mean(frk_unobs)), dtype=pred.dtype, device=pred.device)
            frk_unobs_std_t = torch.as_tensor(float(np.std(frk_unobs)), dtype=pred.dtype, device=pred.device)
            unobs_mean_loss = F.mse_loss(pred_summary_t.mean(), frk_unobs_mean_t)
            unobs_std_loss = F.mse_loss(pred_summary_t.std(unbiased=False), frk_unobs_std_t)
            frk_loss = (
                frk_loss
                + self.unobs_mean_weight * unobs_mean_loss
                + self.unobs_std_weight * unobs_std_loss
            )
        return base_loss + self.frk_weight * frk_loss


# ============================================================
# DLinear + 既有參數
# ============================================================
print(
    f"\n===== DLinear hybrid FRK loss run ({EXPERIMENT_SCENARIO}) "
    f"space={N_SAMPLE_TARGET}->{N_STDK} time={TIME_TRAIN_LEN}/{TIME_VAL_LEN}/{TIME_TEST_LEN} ====="
)

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
_suppress_autofrk_logs()

RUN_SEARCH = False
N_TRIALS = 500
TIMEOUT_SECONDS = None

USE_COVARIATES = False
USE_REVIN = True
RANDOM_STATE = 42

FRK_LOSS_WEIGHT = 0.01
FRK_OBS_RECON_WEIGHT = 1.0
FRK_UNOBS_MEAN_WEIGHT = 0.0
FRK_UNOBS_STD_WEIGHT = 0.0
FRK_LOSS_APPLY_EVERY = 1
FRK_TRAIN_N_NEIGHBOR = 3
FRK_TEST_N_NEIGHBOR = 5

# 以 DLinear+FRK 後的 500 站 validation RMSE 作為 early stopping 準則
VAL500_PATIENCE = 30
VAL500_MIN_DELTA = 1e-6

PL_TRAINER_KWARGS = {
    "enable_progress_bar": False,
    "logger": False,
    "enable_model_summary": False,
}

SAVE_DIR = Path(script_dir)
_env = os.environ
RESULT_SUFFIX = _env.get("RESULT_SUFFIX", "")
suffix = f"_{RESULT_SUFFIX}" if RESULT_SUFFIX else ""
BASE_PARAMS_PATH = SAVE_DIR / "2K_best_dlinear_params_500to100.json"
BEST_RESULT_PATH = SAVE_DIR / f"2K_best_dlinear_frkloss_result_500to100{suffix}.json"
BEST_PARAMS_PATH = SAVE_DIR / f"2K_best_dlinear_frkloss_params_500to100{suffix}.json"
RERUN_METRICS_PATH = SAVE_DIR / f"2K_best_dlinear_frkloss_rerun_metrics_500to100{suffix}.json"
FRK_METRICS_PATH = SAVE_DIR / f"dlinear_autofrk_frkloss_test_100to500_metrics{suffix}.json"

if BASE_PARAMS_PATH.exists():
    print(f"Using existing DLinear params: {BASE_PARAMS_PATH}")
elif BEST_PARAMS_PATH.exists():
    print(f"Using cached hybrid params: {BEST_PARAMS_PATH}")
else:
    print("No parameter file found; Optuna search may be needed.")

SEARCH_SPACE = {
    "INPUT_CHUNK_LENGTH": [24, 36, 48],
    "OUTPUT_CHUNK_LENGTH": [12, 24],
    "KERNEL_SIZE": [15, 25],
    "N_EPOCHS": [350],
    "BATCH_SIZE": [16, 32, 64],
    "LR": [1e-4, 2e-4, 3e-4],
    "WEIGHT_DECAY": [0.0, 1e-5],
    "CONST_INIT": [True, False],
}

print("Preparing repo DLinear dataset...")
EVAL_SEEDS = eval(_env.get("SEED_LIST", str(list(range(41, 46)))))

ts_df, used_time_idx = _build_darts_dataset(y_sample, time_stride=TIME_STRIDE)
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

month_values = time_index.month.astype("float32")
month_df = pd.DataFrame({"month": month_values}, index=time_index)
month_ts = TimeSeries.from_dataframe(month_df.astype("float32"))

# 時間切分固定為最後 1000 個時間點中的 700/150/150。
# 純空間外推情境會在最後的 FRK 指標中額外看前 850 個時間點。
expected_time_len = TIME_TRAIN_LEN + TIME_VAL_LEN + TIME_TEST_LEN
if len(ts_df) < expected_time_len:
    raise ValueError(
        f"目前時間點數 {len(ts_df)} 不足以做 {TIME_TRAIN_LEN}/{TIME_VAL_LEN}/{TIME_TEST_LEN} 切分"
    )

cut_train = TIME_TRAIN_LEN
cut_val = TIME_TRAIN_LEN + TIME_VAL_LEN

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
    "experiment_scenario": EXPERIMENT_SCENARIO,
    "full_sample_size": int(N_SAMPLE_TARGET),
    "tune_sample_size": int(TUNE_SAMPLE_SIZE),
    "sample_seed": int(DATA_SEED),
    "time_stride": int(TIME_STRIDE),
    "time_train_len": int(TIME_TRAIN_LEN),
    "time_val_len": int(TIME_VAL_LEN),
    "time_test_len": int(TIME_TEST_LEN),
    "sample_idx_global": sample_idx_global.tolist(),
    "sample_idx_local": sample_idx_local.tolist(),
    "n_last_timepoints": int(N_LAST),
}


def _make_frk_loss(obs_coords, full_coords, obs_idx_in_full):
    return FRKConsistencyLoss(
        obs_coords=obs_coords,
        full_coords=full_coords,
        obs_idx_in_full=obs_idx_in_full,
        frk_weight=FRK_LOSS_WEIGHT,
        obs_recon_weight=FRK_OBS_RECON_WEIGHT,
        unobs_mean_weight=FRK_UNOBS_MEAN_WEIGHT,
        unobs_std_weight=FRK_UNOBS_STD_WEIGHT,
        apply_every=FRK_LOSS_APPLY_EVERY,
    )


best_params = None
best_val_metrics = None

if BASE_PARAMS_PATH.exists():
    best_params = _json_load(BASE_PARAMS_PATH)
    best_params.pop("sampling_info", None)
    print("\n===== LOADED EXISTING DLinear PARAMS =====")
    print(f"Using params from: {BASE_PARAMS_PATH}")

elif BEST_PARAMS_PATH.exists():
    best_params = _json_load(BEST_PARAMS_PATH)
    best_params.pop("sampling_info", None)
    print("\n===== LOADED CACHED HYBRID PARAMS =====")
    print(f"Using params from: {BEST_PARAMS_PATH}")

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
            loss_fn=_make_frk_loss(coords_sample, coords_sample_full, sample_idx_local),
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
    study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT_SECONDS, gc_after_trial=True, show_progress_bar=False)

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
    raise RuntimeError("No parameter file found. Place 2K_best_dlinear_params_500to100.json in the folder or enable RUN_SEARCH.")


def _to_2d_np(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return arr.astype(np.float64, copy=False)


def _frk_extrapolate_matrix(y_obs_matrix, coords_obs, coords_full, maxit=50, n_neighbor=FRK_TEST_N_NEIGHBOR):
    from autoFRK import AutoFRK

    _suppress_autofrk_logs()
    y_obs_matrix = np.asarray(y_obs_matrix, dtype=np.float64)
    if y_obs_matrix.ndim != 2:
        raise ValueError(f"y_obs_matrix must be 2D, got {y_obs_matrix.shape}")

    n_time, n_obs = y_obs_matrix.shape
    n_full = coords_full.shape[0]

    loc_obs = torch.from_numpy(np.asarray(coords_obs, dtype=np.float64)).to(dtype=torch.float64)
    loc_full = torch.from_numpy(np.asarray(coords_full, dtype=np.float64)).to(dtype=torch.float64)
    yhat_full = np.zeros((n_time, n_full), dtype=np.float64)

    afrk = AutoFRK(dtype=torch.float64, device="cpu")
    for t in range(n_time):
        y_obs_t = y_obs_matrix[t, :]
        try:
            fit_obj = _silent_call(
                afrk.forward,
                data=torch.from_numpy(y_obs_t.reshape(n_obs, 1)).to(dtype=torch.float64),
                loc=loc_obs,
                method="EM",
                maxit=int(maxit),
                tolerance=1e-6,
                n_neighbor=int(n_neighbor),
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
            if y_full_t.shape[0] != n_full:
                raise ValueError(
                    f"FRK prediction at step {t + 1} has length {y_full_t.shape[0]}, expected {n_full}"
                )
            yhat_full[t, :] = y_full_t
            del fit_obj, pred_full
        except Exception as exc:
            # If FRK fails for this time slice (e.g. degenerate geometry or too few
            # neighbors), fall back to a constant mean field to avoid crashing.
            print(f"[Warning] autoFRK failed at time {t}: {exc}; using mean-field fallback.")
            mean_val = float(np.mean(y_obs_t))
            yhat_full[t, :] = mean_val

        gc.collect()
        _cleanup_torch_cache()

    return yhat_full


def _compute_val500_rmse(
    model,
    n_val,
    pred_kwargs,
    inverse_scale_fn,
    coords_obs,
    coords_full,
    y_true_full_val,
    val500_idx,
):
    pred_val = _silent_call(model.predict, n=n_val, **pred_kwargs)
    y_obs_pred_val = np.asarray(inverse_scale_fn(_ts_to_2d(pred_val)), dtype=np.float64)
    yhat_full_val = _frk_extrapolate_matrix(
        y_obs_pred_val,
        coords_obs=coords_obs,
        coords_full=coords_full,
        maxit=20,
        n_neighbor=FRK_TRAIN_N_NEIGHBOR,
    )
    y_true_sub = y_true_full_val[:, val500_idx]
    y_pred_sub = yhat_full_val[:, val500_idx]
    mask = np.isfinite(y_true_sub) & np.isfinite(y_pred_sub)
    if not np.any(mask):
        print("[Warning] No finite values for val500 RMSE computation; returning large error")
        return float(1e6)
    y_true_flat = y_true_sub.flatten()[mask.flatten()]
    y_pred_flat = y_pred_sub.flatten()[mask.flatten()]
    rmse_val500 = float(np.sqrt(mean_squared_error(y_true_flat, y_pred_flat)))
    return rmse_val500


def _compute_metrics_safe(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size == 0 or y_pred.size == 0:
        return {"RMSE": float("nan"), "MSE": float("nan"), "MAE": float("nan"), "R2": float("nan")}
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return {"RMSE": float("nan"), "MSE": float("nan"), "MAE": float("nan"), "R2": float("nan")}
    y_true_mask = y_true.flatten()[mask.flatten()]
    y_pred_mask = y_pred.flatten()[mask.flatten()]
    mse = mean_squared_error(y_true_mask, y_pred_mask)
    mae = mean_absolute_error(y_true_mask, y_pred_mask)
    return {
        "RMSE": float(np.sqrt(mse)),
        "MSE": float(mse),
        "MAE": float(mae),
        "R2": float(r2_score(y_true_mask, y_pred_mask)),
    }


def _predict_dlinear_then_frk(model, n_steps, pred_kwargs, inverse_scale_fn, coords_obs, coords_full):
    pred_obs = _silent_call(model.predict, n=n_steps, **pred_kwargs)
    y_obs_pred = np.asarray(inverse_scale_fn(_ts_to_2d(pred_obs)), dtype=np.float64)
    yhat_full = _frk_extrapolate_matrix(
        y_obs_pred,
        coords_obs=coords_obs,
        coords_full=coords_full,
        maxit=50,
        n_neighbor=FRK_TEST_N_NEIGHBOR,
    )
    return y_obs_pred, yhat_full


class DifferentiableDLinear(torch.nn.Module):
    """簡化版 DLinear，讓空間 surrogate loss 可以真的反向傳回時間模型。"""

    def __init__(self, input_len, output_len, n_series, kernel_size=15, const_init=True):
        super().__init__()
        self.input_len = int(input_len)
        self.output_len = int(output_len)
        self.n_series = int(n_series)
        self.kernel_size = int(kernel_size)
        self.linear_trend = torch.nn.ModuleList(
            [torch.nn.Linear(self.input_len, self.output_len) for _ in range(self.n_series)]
        )
        self.linear_seasonal = torch.nn.ModuleList(
            [torch.nn.Linear(self.input_len, self.output_len) for _ in range(self.n_series)]
        )
        if const_init:
            for layer in list(self.linear_trend) + list(self.linear_seasonal):
                torch.nn.init.constant_(layer.weight, 1.0 / self.input_len)
                torch.nn.init.zeros_(layer.bias)

    def _moving_average(self, x):
        # x: [batch, input_len, n_series]
        pad = (self.kernel_size - 1) // 2
        x_ch = x.transpose(1, 2)
        trend = F.avg_pool1d(x_ch, kernel_size=self.kernel_size, stride=1, padding=pad)
        if trend.shape[-1] != self.input_len:
            trend = trend[..., : self.input_len]
        return trend.transpose(1, 2)

    def forward(self, x):
        trend = self._moving_average(x)
        seasonal = x - trend
        outs = []
        for i in range(self.n_series):
            y_i = self.linear_trend[i](trend[:, :, i]) + self.linear_seasonal[i](seasonal[:, :, i])
            outs.append(y_i.unsqueeze(-1))
        return torch.cat(outs, dim=-1)


class DifferentiableSpatialSurrogate(torch.nn.Module):
    """固定 RBF 權重的 FRK-like surrogate；可微分，但不是 autoFRK EM。"""

    def __init__(self, obs_coords, full_coords, obs_idx, unobs_idx, bandwidth=None):
        super().__init__()
        obs_coords = np.asarray(obs_coords, dtype=np.float32)
        full_coords = np.asarray(full_coords, dtype=np.float32)
        obs_idx = np.asarray(obs_idx, dtype=int)
        unobs_idx = np.asarray(unobs_idx, dtype=int)
        coord_min = full_coords.min(axis=0, keepdims=True)
        coord_max = full_coords.max(axis=0, keepdims=True)
        denom = np.where((coord_max - coord_min) < 1e-12, 1.0, coord_max - coord_min)
        obs_norm = (obs_coords - coord_min) / denom
        unobs_norm = (full_coords[unobs_idx] - coord_min) / denom
        d2 = ((unobs_norm[:, None, :] - obs_norm[None, :, :]) ** 2).sum(axis=-1)
        if bandwidth is None:
            positive_dist = np.sqrt(d2[d2 > 1e-12])
            bandwidth = float(np.median(positive_dist)) if positive_dist.size else 0.25
        bandwidth = max(float(bandwidth), 1e-3)
        weights = np.exp(-d2 / (2.0 * bandwidth * bandwidth))
        weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
        self.obs_idx = obs_idx
        self.unobs_idx = unobs_idx
        self.register_buffer("weights_unobs_obs", torch.as_tensor(weights, dtype=torch.float32))

    def predict_unobs(self, pred_obs):
        # pred_obs: [batch, horizon, n_obs]
        return torch.einsum("uo,bho->bhu", self.weights_unobs_obs, pred_obs)

    def predict_full(self, pred_obs, n_full):
        pred_full = pred_obs.new_zeros((pred_obs.shape[0], pred_obs.shape[1], int(n_full)))
        pred_full[:, :, self.obs_idx.tolist()] = pred_obs
        pred_full[:, :, self.unobs_idx.tolist()] = self.predict_unobs(pred_obs)
        return pred_full


class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, x, y_obs, y_unobs):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y_obs = torch.as_tensor(y_obs, dtype=torch.float32)
        self.y_unobs = torch.as_tensor(y_unobs, dtype=torch.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y_obs[idx], self.y_unobs[idx]


def _make_windows(obs_matrix, unobs_matrix, start_idx, end_idx, input_len, output_len):
    x_list, y_obs_list, y_unobs_list = [], [], []
    last_start = int(end_idx) - int(input_len) - int(output_len) + 1
    for s in range(int(start_idx), max(int(start_idx), last_start)):
        x_list.append(obs_matrix[s : s + input_len, :])
        y_obs_list.append(obs_matrix[s + input_len : s + input_len + output_len, :])
        y_unobs_list.append(unobs_matrix[s + input_len : s + input_len + output_len, :])
    if not x_list:
        raise ValueError("無法建立 DLinear 訓練視窗，請檢查 input/output 長度與時間切分")
    return np.stack(x_list), np.stack(y_obs_list), np.stack(y_unobs_list)


def _autoregressive_predict(model, history_obs, n_steps, device):
    model.eval()
    history = torch.as_tensor(history_obs, dtype=torch.float32, device=device).unsqueeze(0)
    preds = []
    with torch.no_grad():
        while sum(p.shape[1] for p in preds) < n_steps:
            x = history[:, -model.input_len :, :]
            y = model(x)
            preds.append(y)
            history = torch.cat([history, y], dim=1)
    pred = torch.cat(preds, dim=1)[:, :n_steps, :]
    return pred.squeeze(0).cpu().numpy()


def _metric_from_raw_scaled(y_true_scaled, y_pred_scaled, y_mean, y_std):
    return _compute_metrics_safe(y_true_scaled * y_std + y_mean, y_pred_scaled * y_std + y_mean)


def _run_seeded_diff_dlinear_frk(sample_seed: int, best_params: dict) -> dict:
    n_sample = min(N_SAMPLE_TARGET, nloc)
    np.random.seed(sample_seed)
    sample_idx_global_seed = np.random.choice(nloc, size=n_sample, replace=False)
    sample_idx_global_seed = np.sort(sample_idx_global_seed)

    y_sample_full_seed = y_all[sample_idx_global_seed, :]
    coords_sample_full_seed = gg[sample_idx_global_seed, :]

    n_obs = min(N_STDK, n_sample)
    np.random.seed(sample_seed)
    sample_idx_local_seed = np.random.choice(n_sample, size=n_obs, replace=False)
    sample_idx_local_seed = np.sort(sample_idx_local_seed)

    all_unknown_idx = np.setdiff1d(np.arange(n_sample), sample_idx_local_seed)
    np.random.seed(sample_seed + 1)
    n_eval_holdout = min(100, len(all_unknown_idx))
    unobs_eval_idx = np.sort(np.random.choice(all_unknown_idx, size=n_eval_holdout, replace=False))
    unobs_primary_idx = np.setdiff1d(all_unknown_idx, unobs_eval_idx)
    val500_idx = np.sort(np.concatenate([sample_idx_local_seed, unobs_primary_idx]))

    _, used_time_idx_seed = _build_darts_dataset(y_sample_full_seed[sample_idx_local_seed, :], time_stride=TIME_STRIDE)
    if len(used_time_idx_seed) > N_LAST:
        used_time_idx_seed = used_time_idx_seed[-N_LAST:]

    cut_train_seed = TIME_TRAIN_LEN
    cut_val_seed = TIME_TRAIN_LEN + TIME_VAL_LEN
    expected_time_len_seed = TIME_TRAIN_LEN + TIME_VAL_LEN + TIME_TEST_LEN
    if len(used_time_idx_seed) < expected_time_len_seed:
        raise ValueError(
            f"目前時間點數 {len(used_time_idx_seed)} 不足以做 "
            f"{TIME_TRAIN_LEN}/{TIME_VAL_LEN}/{TIME_TEST_LEN} 切分"
        )

    y_full_raw = y_sample_full_seed[:, used_time_idx_seed].T.astype(np.float32)
    y_mean = float(np.mean(y_full_raw[:cut_train_seed, sample_idx_local_seed]))
    y_std = float(np.std(y_full_raw[:cut_train_seed, sample_idx_local_seed], ddof=0))
    if y_std < 1e-12:
        y_std = 1.0
    y_full = ((y_full_raw - y_mean) / y_std).astype(np.float32)
    y_obs = y_full[:, sample_idx_local_seed]
    y_unobs_primary = y_full[:, unobs_primary_idx]

    use_in = int(best_params.get("INPUT_CHUNK_LENGTH", 24))
    use_out = int(best_params.get("OUTPUT_CHUNK_LENGTH", 12))
    if use_in + use_out > cut_train_seed:
        use_out = max(1, min(6, int(cut_train_seed * 0.3)))
        use_in = max(1, cut_train_seed - use_out)

    x_train, y_train_obs, y_train_unobs = _make_windows(
        obs_matrix=y_obs,
        unobs_matrix=y_unobs_primary,
        start_idx=0,
        end_idx=cut_train_seed,
        input_len=use_in,
        output_len=use_out,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(sample_seed)
    model = DifferentiableDLinear(
        input_len=use_in,
        output_len=use_out,
        n_series=n_obs,
        kernel_size=best_params.get("KERNEL_SIZE", 15),
        const_init=best_params.get("CONST_INIT", True),
    ).to(device)
    surrogate_unobs_idx = all_unknown_idx
    primary_pos_in_surrogate = np.array(
        [int(np.where(surrogate_unobs_idx == idx)[0][0]) for idx in unobs_primary_idx],
        dtype=int,
    )
    spatial = DifferentiableSpatialSurrogate(
        obs_coords=coords_sample_full_seed[sample_idx_local_seed, :],
        full_coords=coords_sample_full_seed,
        obs_idx=sample_idx_local_seed,
        unobs_idx=surrogate_unobs_idx,
    ).to(device)

    dataset = WindowDataset(x_train, y_train_obs, y_train_unobs)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(best_params.get("BATCH_SIZE", 32)),
        shuffle=True,
        generator=torch.Generator().manual_seed(sample_seed + 2000),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(best_params.get("LR", 3e-4)),
        weight_decay=float(best_params.get("WEIGHT_DECAY", 0.0)),
    )
    obs_loss_weight = float(os.environ.get("DIFF_FRK_OBS_LOSS_WEIGHT", "1.0"))
    spatial_loss_weight = float(os.environ.get("DIFF_FRK_LOSS_WEIGHT", "1.0"))
    max_train_epochs = int(best_params.get("N_EPOCHS", 350))
    best_state_dict = None
    best_val_rmse = float("inf")
    best_epoch = 0
    wait = 0
    train_start = time.time()
    loss_history = []
    PRINT_EPOCH_LOSS = os.environ.get("PRINT_EPOCH_LOSS", "0") == "1"

    history_train = y_obs[:cut_train_seed, :]
    truth_val_full = y_full[cut_train_seed:cut_val_seed, :]
    truth_test_full = y_full[cut_val_seed : cut_val_seed + TIME_TEST_LEN, :]

    for epoch_idx in range(1, max_train_epochs + 1):
        model.train()
        epoch_loss_obs = 0.0
        epoch_loss_unobs = 0.0
        epoch_loss_total = 0.0
        epoch_batches = 0
        for xb, yb_obs, yb_unobs in loader:
            xb = xb.to(device)
            yb_obs = yb_obs.to(device)
            yb_unobs = yb_unobs.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred_obs = model(xb)
            pred_unobs = spatial.predict_unobs(pred_obs)[:, :, primary_pos_in_surrogate.tolist()]
            loss_obs = F.mse_loss(pred_obs, yb_obs)
            loss_unobs = F.mse_loss(pred_unobs, yb_unobs)
            loss = obs_loss_weight * loss_obs + spatial_loss_weight * loss_unobs
            loss.backward()
            optimizer.step()
            epoch_loss_obs += float(loss_obs.detach().cpu().item())
            epoch_loss_unobs += float(loss_unobs.detach().cpu().item())
            epoch_loss_total += float(loss.detach().cpu().item())
            epoch_batches += 1

        epoch_loss_obs /= max(1, epoch_batches)
        epoch_loss_unobs /= max(1, epoch_batches)
        epoch_loss_total /= max(1, epoch_batches)

        pred_val_obs = _autoregressive_predict(model, history_train, TIME_VAL_LEN, device)
        pred_val_obs_t = torch.as_tensor(pred_val_obs, dtype=torch.float32, device=device).unsqueeze(0)
        pred_val_full = spatial.predict_full(pred_val_obs_t, n_sample).squeeze(0).detach().cpu().numpy()
        val_metric = _metric_from_raw_scaled(
            truth_val_full[:, val500_idx],
            pred_val_full[:, val500_idx],
            y_mean,
            y_std,
        )
        val_rmse = val_metric["RMSE"]
        improved = (best_val_rmse - val_rmse) > VAL500_MIN_DELTA
        if improved:
            best_val_rmse = val_rmse
            best_epoch = epoch_idx
            wait = 0
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        loss_history.append(
            {
                "epoch": int(epoch_idx),
                "loss_obs_scaled": float(epoch_loss_obs),
                "loss_unobs_scaled": float(epoch_loss_unobs),
                "weighted_loss_obs_scaled": float(obs_loss_weight * epoch_loss_obs),
                "weighted_loss_unobs_scaled": float(spatial_loss_weight * epoch_loss_unobs),
                "loss_total_scaled": float(epoch_loss_total),
                "val_4plus5_rmse_raw": float(val_rmse),
                "is_best": bool(improved),
            }
        )
        if PRINT_EPOCH_LOSS:
            print(
                f"[seed {sample_seed}] diff-DLinear epoch {epoch_idx}/{max_train_epochs} "
                f"loss_obs={epoch_loss_obs:.6f} loss_unobs={epoch_loss_unobs:.6f} "
                f"alpha_obs={obs_loss_weight:.3f} lambda_unobs={spatial_loss_weight:.3f} "
                f"loss_total={epoch_loss_total:.6f} val_4plus5_rmse={val_rmse:.6f} "
                f"best={best_val_rmse:.6f} wait={wait}/{VAL500_PATIENCE}"
            )
        if wait >= VAL500_PATIENCE:
            print(f"[seed {sample_seed}] early stopping differentiable FRK surrogate at epoch {epoch_idx}")
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    pred_val_obs = _autoregressive_predict(model, history_train, TIME_VAL_LEN, device)
    pred_val_full = spatial.predict_full(
        torch.as_tensor(pred_val_obs, dtype=torch.float32, device=device).unsqueeze(0),
        n_sample,
    ).squeeze(0).detach().cpu().numpy()
    pred_300_obs = _autoregressive_predict(model, history_train, TIME_VAL_LEN + TIME_TEST_LEN, device)
    pred_test_obs = pred_300_obs[TIME_VAL_LEN:, :]
    pred_test_full = spatial.predict_full(
        torch.as_tensor(pred_test_obs, dtype=torch.float32, device=device).unsqueeze(0),
        n_sample,
    ).squeeze(0).detach().cpu().numpy()

    # 1+2：1 是真實 obs100，2 透過可微分 surrogate 的固定權重外推。
    train_obs_t = torch.as_tensor(y_obs[:cut_train_seed, :], dtype=torch.float32, device=device).unsqueeze(0)
    train_full_pred = spatial.predict_full(train_obs_t, n_sample).squeeze(0).detach().cpu().numpy()
    train700_metric = _metric_from_raw_scaled(
        y_full[:cut_train_seed, val500_idx],
        train_full_pred[:, val500_idx],
        y_mean,
        y_std,
    )
    val150_metric = _metric_from_raw_scaled(
        truth_val_full[:, val500_idx],
        pred_val_full[:, val500_idx],
        y_mean,
        y_std,
    )
    target_time150_metric = _metric_from_raw_scaled(
        truth_test_full[:, val500_idx],
        pred_test_full[:, val500_idx],
        y_mean,
        y_std,
    )

    fixed850_truth = y_full[:cut_val_seed, :]
    fixed850_obs = np.vstack([y_obs[:cut_train_seed, :], pred_val_obs]).astype(np.float32)
    fixed850_full_pred = spatial.predict_full(
        torch.as_tensor(fixed850_obs, dtype=torch.float32, device=device).unsqueeze(0),
        n_sample,
    ).squeeze(0).detach().cpu().numpy()
    space_train100_metric = _metric_from_raw_scaled(
        fixed850_truth[:, sample_idx_local_seed],
        fixed850_full_pred[:, sample_idx_local_seed],
        y_mean,
        y_std,
    )
    space_val400_metric = _metric_from_raw_scaled(
        fixed850_truth[:, unobs_primary_idx],
        fixed850_full_pred[:, unobs_primary_idx],
        y_mean,
        y_std,
    )
    space_target100_metric = _metric_from_raw_scaled(
        fixed850_truth[:, unobs_eval_idx],
        fixed850_full_pred[:, unobs_eval_idx],
        y_mean,
        y_std,
    )
    target_st_metric = _metric_from_raw_scaled(
        truth_test_full[:, unobs_eval_idx],
        pred_test_full[:, unobs_eval_idx],
        y_mean,
        y_std,
    )

    def prefixed(prefix, metric):
        return {
            f"{prefix}_RMSE": metric["RMSE"],
            f"{prefix}_MSE": metric["MSE"],
            f"{prefix}_MAE": metric["MAE"],
            f"{prefix}_R2": metric["R2"],
        }

    if EXPERIMENT_SCENARIO == "time_extrap_fixed500":
        frk_metrics = {
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
        frk_metrics = {
            **prefixed("Train100", space_train100_metric),
            **prefixed("Val400", space_val400_metric),
            **prefixed("Target_Space100", space_target100_metric),
        }
        result_sections = {
            "Train100": space_train100_metric,
            "Val400": space_val400_metric,
            "Target_Space100": space_target100_metric,
        }
    elif EXPERIMENT_SCENARIO == "spatiotemp_100x150":
        frk_metrics = prefixed("Target_ST100x150", target_st_metric)
        result_sections = {"Target_ST100x150": target_st_metric}
    else:
        frk_metrics = prefixed("Target", target_time150_metric)
        result_sections = {"Target": target_time150_metric}

    elapsed = time.time() - train_start
    print(f"Training elapsed={_fmt_time(elapsed)}")
    print(
        pd.DataFrame(
            [{"Model": f"DLINEAR + differentiable_FRK_seed_{sample_seed}", "Split": "TEST", **frk_metrics}]
        ).to_string(index=False)
    )

    dlinear_metric = _metric_from_raw_scaled(
        truth_test_full[:, sample_idx_local_seed],
        pred_test_obs,
        y_mean,
        y_std,
    )

    return {
        "seed": int(sample_seed),
        "elapsed_seconds": float(elapsed),
        "dlinear_metrics": dlinear_metric,
        "frk_metrics": frk_metrics,
        "results": result_sections,
        "loss_summary": {
            "scale": "standardized training scale except val_4plus5_rmse_raw",
            "alpha_obs": float(obs_loss_weight),
            "lambda_unobs": float(spatial_loss_weight),
            "best_epoch": int(best_epoch),
            "best_val_4plus5_rmse_raw": float(best_val_rmse),
            "first_epoch": loss_history[0] if loss_history else None,
            "best_epoch_record": loss_history[best_epoch - 1] if best_epoch > 0 and best_epoch <= len(loss_history) else None,
            "last_epoch": loss_history[-1] if loss_history else None,
        },
        "loss_history": loss_history,
        "sampling_info": {
            "experiment_scenario": EXPERIMENT_SCENARIO,
            "full_sample_size": int(N_SAMPLE_TARGET),
            "tune_sample_size": int(N_STDK),
            "sample_seed": int(sample_seed),
            "time_stride": int(TIME_STRIDE),
            "time_train_len": int(TIME_TRAIN_LEN),
            "time_val_len": int(TIME_VAL_LEN),
            "time_test_len": int(TIME_TEST_LEN),
            "sample_idx_global": sample_idx_global_seed.tolist(),
            "sample_idx_local": sample_idx_local_seed.tolist(),
            "sample_idx_unobs_primary": unobs_primary_idx.tolist(),
            "sample_idx_unobs_eval": unobs_eval_idx.tolist(),
            "n_last_timepoints": int(N_LAST),
            "spatial_surrogate": "differentiable_rbf_frk_like",
            "diff_frk_loss_weight": float(spatial_loss_weight),
        },
    }



def _run_seeded_dlinear_and_frk(sample_seed: int, best_params: dict) -> dict:
    _suppress_autofrk_logs()

    n_sample = min(N_SAMPLE_TARGET, nloc)

    np.random.seed(sample_seed)
    sample_idx_global_seed = np.random.choice(nloc, size=n_sample, replace=False)
    sample_idx_global_seed = np.sort(sample_idx_global_seed)

    y_sample_full_seed = y_all[sample_idx_global_seed, :]
    coords_sample_full_seed = gg[sample_idx_global_seed, :]

    print(f"\n===== DLinear + autoFRK hybrid-loss seed {sample_seed} =====")
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

    month_values_seed = time_index.month.astype("float32")
    month_df_seed = pd.DataFrame({"month": month_values_seed}, index=time_index)
    month_ts_seed = TimeSeries.from_dataframe(month_df_seed.astype("float32"))

    # 每個 seed 都使用同一個明確時間切分：700/150/150。
    expected_time_len_seed = TIME_TRAIN_LEN + TIME_VAL_LEN + TIME_TEST_LEN
    if len(ts_df_seed) < expected_time_len_seed:
        raise ValueError(
            f"目前時間點數 {len(ts_df_seed)} 不足以做 "
            f"{TIME_TRAIN_LEN}/{TIME_VAL_LEN}/{TIME_TEST_LEN} 切分"
        )

    cut_train_seed = TIME_TRAIN_LEN
    cut_val_seed = TIME_TRAIN_LEN + TIME_VAL_LEN

    train_raw_seed = TimeSeries.from_dataframe(ts_df_seed.iloc[:cut_train_seed])
    val_raw_seed = TimeSeries.from_dataframe(ts_df_seed.iloc[cut_train_seed:cut_val_seed])
    test_raw_seed = TimeSeries.from_dataframe(ts_df_seed.iloc[cut_val_seed:])

    trainval_ts_seed = TimeSeries.from_dataframe(ts_df_seed.iloc[:cut_val_seed])
    train_ts_seed = TimeSeries.from_dataframe(ts_df_seed.iloc[:cut_train_seed])
    test_ts_seed = TimeSeries.from_dataframe(ts_df_seed.iloc[cut_val_seed:])

    month_trainval_seed = month_ts_seed[:cut_val_seed]
    month_train_seed = month_ts_seed[:cut_train_seed]

    print(
        f"Train len (raw): {len(train_raw_seed)}, Val len (raw): {len(val_raw_seed)}, Test len (raw): {len(test_raw_seed)}"
    )

    train_df_seed = ts_df_seed.iloc[:cut_train_seed]
    mean_vec_seed = train_df_seed.mean(axis=0)
    std_vec_seed = train_df_seed.std(axis=0).replace(0.0, 1.0)

    ts_df_scaled_seed = (ts_df_seed - mean_vec_seed) / std_vec_seed
    train_scaled_seed = TimeSeries.from_dataframe(ts_df_scaled_seed.iloc[:cut_train_seed])
    val_scaled_seed = TimeSeries.from_dataframe(ts_df_scaled_seed.iloc[cut_train_seed:cut_val_seed])
    test_scaled_seed = TimeSeries.from_dataframe(ts_df_scaled_seed.iloc[cut_val_seed:])

    T_train_seed, T_test_seed = len(train_scaled_seed), len(test_scaled_seed)
    print("T_train:", T_train_seed, "T_test:", T_test_seed)

    test_true_raw_seed = ts_df_seed.to_numpy(dtype=np.float32)[cut_val_seed:]

    def inverse_scale_seed(x):
        return x * std_vec_seed.values + mean_vec_seed.values

    y_sample_full_arr = np.asarray(y_sample_full_seed, dtype=np.float64)
    coords_sample_full_arr = np.asarray(coords_sample_full_seed, dtype=np.float64)
    sample_idx_global_arr = np.asarray(sample_idx_global_seed, dtype=int)
    sample_idx_local_arr = np.asarray(sample_idx_local_seed, dtype=int)

    n_full = y_sample_full_arr.shape[0]
    n_obs = sample_idx_local_arr.shape[0]
    if n_full != 600:
        print(f"[Warning] full sample size is {n_full}, expected 600.")
    if n_obs != 100:
        print(f"[Warning] observed sample size is {n_obs}, not 100.")

    # 把剩下的未觀測站分成主要 unobs + 評估 holdout (最多 100)
    all_unknown_idx = np.setdiff1d(np.arange(n_full), sample_idx_local_arr)
    if len(all_unknown_idx) != (n_full - n_obs):
        raise RuntimeError("Unexpected unknown index split")

    np.random.seed(sample_seed + 1)
    # 若 unknown 不足 100，則降為可用數量（可能為 0）
    n_eval_holdout = min(100, len(all_unknown_idx))
    if n_eval_holdout > 0:
        unobs_eval_idx = np.random.choice(all_unknown_idx, size=n_eval_holdout, replace=False)
        unobs_eval_idx = np.sort(unobs_eval_idx)
        unobs_primary_idx = np.setdiff1d(all_unknown_idx, unobs_eval_idx)
    else:
        unobs_eval_idx = np.array([], dtype=int)
        unobs_primary_idx = all_unknown_idx

    # val500 = observed + primary unobs (size may be <500 if sample <600)
    val500_idx = np.sort(np.concatenate([sample_idx_local_arr, unobs_primary_idx]))
    if len(val500_idx) != 500:
        print(f"[Warning] val500 size is {len(val500_idx)}, expected 500.")

    # 三個情境都維持 DLinear+FRK：DLinear 只吃 100 個 train 空間點，
    # FRK 會把預測場外推到 400 個 validation 空間點與 100 個 target 空間點。
    seed_loss_fn = _make_frk_loss(coords_sample_seed, coords_sample_full_seed, sample_idx_local_seed)
    # 若時間序列太短，動態調整 input/output 長度以符合最小樣本要求
    desired_in = int(best_params.get("INPUT_CHUNK_LENGTH", 24))
    desired_out = int(best_params.get("OUTPUT_CHUNK_LENGTH", 12))
    min_series_len = min(len(train_scaled_seed), len(val_scaled_seed), len(test_scaled_seed))
    if (desired_in + desired_out) > min_series_len:
        # 選擇保守回退值，使 input+output <= min_series_len
        fallback_out = max(1, int(max(1, min(6, int(min_series_len * 0.3)))))
        fallback_in = max(1, min_series_len - fallback_out)
        print(
            f"[Warning] time series too short for desired in+out ({desired_in}+{desired_out} > {min_series_len}),"
            f" falling back to in={fallback_in}, out={fallback_out} for this seed"
        )
        use_in, use_out = fallback_in, fallback_out
    else:
        use_in, use_out = desired_in, desired_out

    model_best_kwargs = _build_model_kwargs(
        in_len=use_in,
        out_len=use_out,
        ksize=best_params.get("KERNEL_SIZE", 15),
        n_epochs=1,
        bs=best_params.get("BATCH_SIZE", 32),
        lr=best_params.get("LR", 3e-4),
        wd=best_params.get("WEIGHT_DECAY", 0.0),
        const_init=best_params.get("CONST_INIT", True),
        random_state=sample_seed,
        loss_fn=seed_loss_fn,
        save_checkpoints=False,
    )

    model_best = DLinearModel(**model_best_kwargs)
    fit_kwargs_best = {"verbose": False}
    pred_kwargs_best = {"verbose": False, "show_warnings": False}

    fit_kwargs_best["val_series"] = val_scaled_seed
    if USE_COVARIATES:
        fit_kwargs_best["past_covariates"] = month_train_seed
        fit_kwargs_best["future_covariates"] = month_train_seed
        fit_kwargs_best["val_past_covariates"] = month_val
        fit_kwargs_best["val_future_covariates"] = month_val
        pred_kwargs_best["past_covariates"] = month_train_seed
        pred_kwargs_best["future_covariates"] = month_train_seed

    coords_obs = coords_sample_full_arr[sample_idx_local_arr, :]
    coords_full = coords_sample_full_arr
    y_true_full_val = y_sample_full_arr[:, used_time_idx_seed[cut_train_seed:cut_val_seed]].T.astype(np.float64)

    train_start = time.time()
    max_train_epochs = int(best_params["N_EPOCHS"])
    wait = 0
    best_val500_rmse = float("inf")
    best_state_dict = None
    best_epoch_idx = 1

    # 每個 epoch：先訓練 DLinear，再以 DLinear+FRK(500站) 的 val RMSE 當作選模準則
    for epoch_idx in range(1, max_train_epochs + 1):
        _silent_call(model_best.fit, series=train_scaled_seed, epochs=1, **fit_kwargs_best)

        val500_rmse = _compute_val500_rmse(
            model=model_best,
            n_val=len(val_scaled_seed),
            pred_kwargs=pred_kwargs_best,
            inverse_scale_fn=inverse_scale_seed,
            coords_obs=coords_obs,
            coords_full=coords_full,
            y_true_full_val=y_true_full_val,
            val500_idx=val500_idx,
        )

        improved = (best_val500_rmse - val500_rmse) > VAL500_MIN_DELTA
        if improved:
            best_val500_rmse = val500_rmse
            best_epoch_idx = epoch_idx
            wait = 0
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model_best.model.state_dict().items()
            }
        else:
            wait += 1

        print(
            f"[seed {sample_seed}] epoch {epoch_idx}/{max_train_epochs} "
            f"val500_rmse={val500_rmse:.6f} best={best_val500_rmse:.6f} wait={wait}/{VAL500_PATIENCE}"
        )

        if wait >= VAL500_PATIENCE:
            print(f"[seed {sample_seed}] early stopping on val500 criterion at epoch {epoch_idx}")
            break

    if best_state_dict is not None:
        model_best.model.load_state_dict(best_state_dict)
    else:
        print(f"[Warning] seed {sample_seed}: no improved val500 checkpoint captured; using latest weights")

    val150_metric = {"RMSE": float("nan"), "MSE": float("nan"), "MAE": float("nan"), "R2": float("nan")}
    y_obs_pred_val = None
    yhat_full_val = None
    if EXPERIMENT_SCENARIO in ("time_extrap_fixed500", "space_extrap_fixed850"):
        # Val150 = 圖上的 4+5：
        # 4 由 DLinear 預測 100 個 train 空間點，5 由 FRK 外推 400 個 val 空間點。
        y_obs_pred_val, yhat_full_val = _predict_dlinear_then_frk(
            model=model_best,
            n_steps=len(val_scaled_seed),
            pred_kwargs=pred_kwargs_best,
            inverse_scale_fn=inverse_scale_seed,
            coords_obs=coords_obs,
            coords_full=coords_full,
        )
    if EXPERIMENT_SCENARIO == "time_extrap_fixed500":
        val150_metric = _compute_metrics_safe(y_true_full_val[:, val500_idx], yhat_full_val[:, val500_idx])

    # 最終測試維持嚴格切分：Train700 只訓練、Val150 只選模、Target_Time150 才是 final test。
    # 因此不 refit 1+4，直接使用 4+5 combined loss 選到的最佳 DLinear 權重預測 7+8。

    pred_best = _silent_call(model_best.predict, n=len(test_scaled_seed), **pred_kwargs_best)
    elapsed = time.time() - train_start

    pred_best_raw = inverse_scale_seed(_ts_to_2d(pred_best))
    y_test_true_best = test_true_raw_seed.reshape(-1, n_sample_seed)
    y_test_pred_best = pred_best_raw.reshape(-1, n_sample_seed)

    y_true_flat = y_test_true_best.flatten()
    y_pred_flat = y_test_pred_best.flatten()
    mask_test = np.isfinite(y_true_flat) & np.isfinite(y_pred_flat)
    if not np.any(mask_test):
        print("[Warning] No finite values for final TEST metrics; returning large errors")
        mse_best = float(1e6)
        mae_best = float(1e6)
        rmse_best = float(1e3)
        r2_best = float(-1e6)
    else:
        y_true_mask = y_true_flat[mask_test]
        y_pred_mask = y_pred_flat[mask_test]
        mse_best = mean_squared_error(y_true_mask, y_pred_mask)
        mae_best = mean_absolute_error(y_true_mask, y_pred_mask)
        rmse_best = float(np.sqrt(mse_best))
        try:
            r2_best = float(r2_score(y_true_mask, y_pred_mask))
        except Exception:
            r2_best = float("nan")

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

    Y_obs_pred_test = np.asarray(pred_best_raw, dtype=np.float64)
    if Y_obs_pred_test.ndim != 2:
        raise ValueError(f"pred_best_raw shape must be 2D, got {Y_obs_pred_test.shape}")

    T_test, pred_n = Y_obs_pred_test.shape
    if pred_n != len(sample_idx_local_arr):
        raise ValueError(f"pred_best_raw has {pred_n} columns, but sample_idx_local has {len(sample_idx_local_arr)} entries")

    Y_true_full_test = y_sample_full_arr[:, used_time_idx_seed[cut_val_seed:]].T.astype(np.float64)
    if Y_true_full_test.shape != (T_test, n_full):
        raise ValueError(f"Full test truth shape mismatch: {Y_true_full_test.shape} != {(T_test, n_full)}")

    Yhat_full_test = _frk_extrapolate_matrix(
        Y_obs_pred_test,
        coords_obs=coords_obs,
        coords_full=coords_full,
        maxit=50,
        n_neighbor=FRK_TEST_N_NEIGHBOR,
    )
    Yhat_unknown_primary = Yhat_full_test[:, unobs_primary_idx]
    Yhat_unknown_eval = Yhat_full_test[:, unobs_eval_idx]

    full_metric = _compute_metrics_safe(Y_true_full_test, Yhat_full_test)
    # Target_Time150 = 圖上的 7+8：最後 150 時間點、前 500 空間點。
    target_time150_metric = _compute_metrics_safe(Y_true_full_test[:, val500_idx], Yhat_full_test[:, val500_idx])
    unknown_primary_true = Y_true_full_test[:, unobs_primary_idx]
    unknown_primary_metric = _compute_metrics_safe(unknown_primary_true, Yhat_unknown_primary)
    unknown_eval_true = Y_true_full_test[:, unobs_eval_idx]
    unknown_eval_metric = _compute_metrics_safe(unknown_eval_true, Yhat_unknown_eval)

    # 固定前 850 個時間點做空間外推，但仍強制使用 DLinear+FRK：
    # 1 使用 obs100 真實值，4 使用 DLinear 預測值，接著 FRK 外推到 2+3+5+6。
    empty_metric = {"RMSE": float("nan"), "MSE": float("nan"), "MAE": float("nan"), "R2": float("nan")}
    space_train100_metric = empty_metric
    space_val400_metric = empty_metric
    space_target100_metric = empty_metric
    if EXPERIMENT_SCENARIO == "space_extrap_fixed850" and n_obs < n_full:
        fixed850_idx = np.arange(0, cut_val_seed)
        Y_obs_train700 = y_sample_seed[:, used_time_idx_seed[:cut_train_seed]].T.astype(np.float64)
        if y_obs_pred_val is None:
            raise RuntimeError("space_extrap_fixed850 需要 DLinear 的 Val150 預測值，但 y_obs_pred_val 為 None")
        Y_obs_fixed850 = np.vstack([Y_obs_train700, y_obs_pred_val]).astype(np.float64)
        Y_true_fixed850 = y_sample_full_arr[:, used_time_idx_seed[fixed850_idx]].T.astype(np.float64)
        Yhat_fixed850 = _frk_extrapolate_matrix(
            Y_obs_fixed850,
            coords_obs=coords_obs,
            coords_full=coords_full,
            maxit=50,
            n_neighbor=FRK_TEST_N_NEIGHBOR,
        )
        space_train100_metric = _compute_metrics_safe(Y_true_fixed850[:, sample_idx_local_arr], Yhat_fixed850[:, sample_idx_local_arr])
        space_val400_metric = _compute_metrics_safe(Y_true_fixed850[:, unobs_primary_idx], Yhat_fixed850[:, unobs_primary_idx])
        space_target100_metric = _compute_metrics_safe(Y_true_fixed850[:, unobs_eval_idx], Yhat_fixed850[:, unobs_eval_idx])

    train700_metric = empty_metric
    if EXPERIMENT_SCENARIO == "time_extrap_fixed500":
        # Train700 = 圖上的 1+2：
        # 1 是 100 個 obs 空間點的真實值，2 是由 FRK 外推到 400 個 unobs 空間點。
        Y_obs_train700 = y_sample_seed[:, used_time_idx_seed[:cut_train_seed]].T.astype(np.float64)
        Y_true_full_train700 = y_sample_full_arr[:, used_time_idx_seed[:cut_train_seed]].T.astype(np.float64)
        Yhat_full_train700 = _frk_extrapolate_matrix(
            Y_obs_train700,
            coords_obs=coords_obs,
            coords_full=coords_full,
            maxit=50,
            n_neighbor=FRK_TEST_N_NEIGHBOR,
        )
        train700_metric = _compute_metrics_safe(
            Y_true_full_train700[:, val500_idx],
            Yhat_full_train700[:, val500_idx],
        )

    def prefixed(prefix, metric):
        return {
            f"{prefix}_RMSE": metric["RMSE"],
            f"{prefix}_MSE": metric["MSE"],
            f"{prefix}_MAE": metric["MAE"],
            f"{prefix}_R2": metric["R2"],
        }

    if EXPERIMENT_SCENARIO == "time_extrap_fixed500":
        frk_metrics = {
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
        frk_metrics = {
            **prefixed("Train100", space_train100_metric),
            **prefixed("Val400", space_val400_metric),
            **prefixed("Target_Space100", space_target100_metric),
        }
        result_sections = {
            "Train100": space_train100_metric,
            "Val400": space_val400_metric,
            "Target_Space100": space_target100_metric,
        }
    elif EXPERIMENT_SCENARIO == "spatiotemp_100x150":
        frk_metrics = prefixed("Target_ST100x150", unknown_eval_metric)
        result_sections = {
            "Target_ST100x150": unknown_eval_metric,
        }
    else:
        frk_metrics = prefixed("Target", full_metric)
        result_sections = {"Target": full_metric}

    print(
        pd.DataFrame(
            {
                "Model": [f"DLINEAR + autoFRK_seed_{sample_seed}"],
                "Split": ["TEST"],
                **frk_metrics,
            }
        ).to_string(index=False)
    )

    del model_best, pred_best
    gc.collect()
    _cleanup_torch_cache()

    return {
        "seed": int(sample_seed),
        "elapsed_seconds": float(elapsed),
        "dlinear_metrics": {"RMSE": float(rmse_best), "MSE": float(mse_best), "MAE": float(mae_best), "R2": float(r2_best)},
        "frk_metrics": frk_metrics,
        "results": result_sections,
        "sampling_info": {
            "experiment_scenario": EXPERIMENT_SCENARIO,
            "full_sample_size": int(N_SAMPLE_TARGET),
            "tune_sample_size": int(TUNE_SAMPLE_SIZE),
            "sample_seed": int(sample_seed),
            "time_stride": int(TIME_STRIDE),
            "time_train_len": int(TIME_TRAIN_LEN),
            "time_val_len": int(TIME_VAL_LEN),
            "time_test_len": int(TIME_TEST_LEN),
            "sample_idx_global": sample_idx_global_arr.tolist(),
            "sample_idx_local": sample_idx_local_arr.tolist(),
            "n_last_timepoints": int(N_LAST),
        },
    }


seed_runs = []
for sample_seed in EVAL_SEEDS:
    seed_runs.append(_run_seeded_diff_dlinear_frk(sample_seed, best_params))

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
        [{"Model": "DLINEAR(best)", "Split": "TEST_MEAN", **{key: float(value) for key, value in dlinear_mean_metrics.items()}}]
    ).to_string(index=False)
)
print("\n===== DLinear best TEST RESULTS (seed 41~45 std) =====")
print(
    pd.DataFrame(
        [{"Model": "DLINEAR(best)", "Split": "TEST_STD", **{key: float(value) for key, value in dlinear_std_metrics.items()}}]
    ).to_string(index=False)
)
print(f"Mean training elapsed={_fmt_time(dlinear_elapsed_mean)}")

print("\n===== DLinear + differentiable FRK surrogate TEST RESULTS (seed 41~45 mean) =====")
print(
    pd.DataFrame(
        [{"Model": "DLINEAR + differentiable_FRK", "Split": "TEST_MEAN", **{key: float(value) for key, value in frk_mean_metrics.items()}}]
    ).to_string(index=False)
)
print("\n===== DLinear + differentiable FRK surrogate TEST RESULTS (seed 41~45 std) =====")
print(
    pd.DataFrame(
        [{"Model": "DLINEAR + differentiable_FRK", "Split": "TEST_STD", **{key: float(value) for key, value in frk_std_metrics.items()}}]
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
        "frk_loss_weight": float(FRK_LOSS_WEIGHT),
        "frk_loss_apply_every": int(FRK_LOSS_APPLY_EVERY),
    },
    "seed_runs": seed_runs,
}

frk_payload = {
    "Model": "DLINEAR + differentiable_FRK",
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
        "frk_loss_weight": float(FRK_LOSS_WEIGHT),
        "frk_loss_apply_every": int(FRK_LOSS_APPLY_EVERY),
        "spatial_surrogate": "differentiable_rbf_frk_like",
        "diff_frk_obs_loss_weight": float(os.environ.get("DIFF_FRK_OBS_LOSS_WEIGHT", "1.0")),
        "diff_frk_loss_weight": float(os.environ.get("DIFF_FRK_LOSS_WEIGHT", "1.0")),
    },
    "seed_runs": seed_runs,
}

_json_dump(rerun_payload, RERUN_METRICS_PATH)
_json_dump(frk_payload, FRK_METRICS_PATH)

print(f"\nSaved re-run metrics: {RERUN_METRICS_PATH.resolve()}")
print(f"Saved FRK metrics: {FRK_METRICS_PATH.resolve()}")
print("Repo DLinear + autoFRK run completed.")
