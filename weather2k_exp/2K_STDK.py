import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time

import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

repo_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "spatial-adapter"))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

try:
    from examples.baselines.stdk.st_interp import create_model
except ImportError as exc:
    raise ImportError(
        "Cannot import baseline STDK from spatial-adapter. Make sure the repo is cloned at:\n"
        f"{repo_root}\n"
        "and contains examples/baselines/stdk/st_interp.py"
    ) from exc


# ================= 讀 Weather2K NPY 檔案 =================

# 候選路徑
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
    "visibility"
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


# ===== 從全球1866個測站中隨機抽樣500個格點 =====

N_SAMPLE_TARGET = 500
n_sample = min(N_SAMPLE_TARGET, nloc)  # nloc = 1866

np.random.seed(42)  # 為了可重現性

# 隨機選擇 n_sample 個測站的索引
sample_idx_global = np.random.choice(
    nloc,
    size=n_sample,
    replace=False
)

# 排序以保持一致性
sample_idx_global = np.sort(sample_idx_global)

# 抽樣後資料 (cell, time)
y_sample = y_all[sample_idx_global, :]  # (n_sample, ntime)

# 抽樣後座標
coords_sample = gg[sample_idx_global, :]  # (n_sample, 2)

print(f"抽樣 {n_sample} 個測站")


# 只保留最少輸出：抽樣 500 個測站，接著從中抽取 100 個進行 STDK
print(f"抽樣 {n_sample} 個測站（將從中再抽取 100 個進行 STDK）")

# 從 500 個抽樣中再抽取 100 個做 STDK
N_STDK = 100
n_stdk = min(N_STDK, n_sample)
np.random.seed(42)
sample_idx_local = np.random.choice(n_sample, size=n_stdk, replace=False)
sample_idx_local = np.sort(sample_idx_local)

# 將 y_sample 與 coords_sample 縮小為 STDK 使用的 100 個點
y_sample = y_sample[sample_idx_local, :]
coords_sample = coords_sample[sample_idx_local, :]
n_sample = y_sample.shape[0]

print(f"STDK 使用樣本數: {n_sample}")


def _fmt_time(s):
    m, sec = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h:d}h{m:02d}m{sec:02d}s"


def _minmax_scale(values):
    values = values.astype(np.float32)
    minimum = values.min(axis=0, keepdims=True)
    maximum = values.max(axis=0, keepdims=True)
    scale = np.maximum(maximum - minimum, 1e-12)
    return (values - minimum) / scale


def _build_repo_stdk_dataset(y_values, coords_values, time_stride=24):
    time_idx = np.arange(0, y_values.shape[1], time_stride)
    time_values = time_idx.astype(np.float32) / max(y_values.shape[1] - 1, 1)

    coords_norm = _minmax_scale(coords_values)
    coords_train = np.repeat(coords_norm, len(time_idx), axis=0).astype(np.float32)
    time_train = np.tile(time_values, coords_values.shape[0]).reshape(-1, 1).astype(np.float32)
    y_train = y_values[:, time_idx].reshape(-1, 1).astype(np.float32)

    return coords_train, time_train, y_train, time_idx


print("Preparing repo STDK dataset...")
TIME_STRIDE = 24
coords_train, t_train, y_train, used_time_idx = _build_repo_stdk_dataset(
    y_sample, coords_sample, time_stride=TIME_STRIDE
)

rng = np.random.default_rng(42)
all_indices = np.arange(len(y_train))
rng.shuffle(all_indices)
n_total = len(all_indices)
train_end = int(n_total * 0.7)
val_end = train_end + int(n_total * 0.15)
train_idx = all_indices[:train_end]
val_idx = all_indices[train_end:val_end]
test_idx = all_indices[val_end:]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_config = {
    "regression_type": "mean",
    "p_covariates": 0,
    "k_spatial_centers": [25, 81, 121],
    "k_temporal_centers": [10, 15, 45],
    "hidden_dims": [256, 256, 128],
    "dropout": 0.0,
    "layernorm": False,
    "spatial_learnable": False,
    "spatial_init_method": "uniform",
    "spatial_basis_function": "wendland",
    "gradient_damping": False,
    "use_delta_reparameterization": False,
}

model = create_model(model_config, train_coords=coords_train)
model.to(device)

X_train = torch.zeros((len(train_idx), 0), dtype=torch.float32, device=device)
coords_train_tensor = torch.from_numpy(coords_train[train_idx]).to(device)
t_train_tensor = torch.from_numpy(t_train[train_idx]).to(device)
y_train_tensor = torch.from_numpy(y_train[train_idx]).to(device)

X_val = torch.zeros((len(val_idx), 0), dtype=torch.float32, device=device)
coords_val_tensor = torch.from_numpy(coords_train[val_idx]).to(device)
t_val_tensor = torch.from_numpy(t_train[val_idx]).to(device)
y_val_tensor = torch.from_numpy(y_train[val_idx]).to(device)

# Test split tensors
X_test = torch.zeros((len(test_idx), 0), dtype=torch.float32, device=device)
coords_test_tensor = torch.from_numpy(coords_train[test_idx]).to(device)
t_test_tensor = torch.from_numpy(t_train[test_idx]).to(device)
y_test_tensor = torch.from_numpy(y_train[test_idx]).to(device)

train_loader = DataLoader(
    TensorDataset(X_train, coords_train_tensor, t_train_tensor, y_train_tensor),
    batch_size=512,
    shuffle=True,
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = torch.nn.MSELoss()
epochs = 350
patience = 30

print(
    f"Training repo STDK on {len(train_idx):,} train samples / {len(val_idx):,} val samples / {len(test_idx):,} test samples "
    f"(time stride={TIME_STRIDE})"
)
train_start = time.time()
best_val_loss = float("inf")
best_state_dict = None
patience_counter = 0
for epoch in range(1, epochs + 1):
    model.train()
    epoch_loss = 0.0
    for batch_X, batch_coords, batch_t, batch_y in train_loader:
        optimizer.zero_grad(set_to_none=True)
        predictions = model(batch_X, batch_coords, batch_t)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch_y.size(0)

    epoch_loss /= len(train_idx)

    model.eval()
    with torch.no_grad():
        val_predictions = model(X_val, coords_val_tensor, t_val_tensor)
        val_loss = criterion(val_predictions, y_val_tensor).item()

    elapsed = time.time() - train_start
    avg_epoch = elapsed / epoch
    remaining = avg_epoch * (epochs - epoch)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
    print(
        f"Epoch {epoch}/{epochs} train_loss={epoch_loss:.4f} val_loss={val_loss:.4f} "
        f"elapsed={_fmt_time(elapsed)} eta={_fmt_time(remaining)} patience={patience_counter}/{patience}"
    )

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch} (best val_loss={best_val_loss:.4f})")
        break

if best_state_dict is not None:
    model.load_state_dict(best_state_dict)

model.eval()
with torch.no_grad():
    preview = model(X_val[:5], coords_val_tensor[:5], t_val_tensor[:5])

with torch.no_grad():
    val_predictions = model(X_val, coords_val_tensor, t_val_tensor).detach().cpu().numpy().reshape(-1)
    val_targets = y_val_tensor.detach().cpu().numpy().reshape(-1)

mse = mean_squared_error(val_targets, val_predictions)
rmse = float(np.sqrt(mse))
mae = mean_absolute_error(val_targets, val_predictions)
r2 = r2_score(val_targets, val_predictions)

print(f"Preview prediction shape: {tuple(preview.shape)}")
print(f"RMSE: {rmse:.4f}")
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R^2: {r2:.4f}")

# --- Test set evaluation ---
with torch.no_grad():
    test_predictions = model(X_test, coords_test_tensor, t_test_tensor).detach().cpu().numpy().reshape(-1)
    test_targets = y_test_tensor.detach().cpu().numpy().reshape(-1)

test_mse = mean_squared_error(test_targets, test_predictions)
test_rmse = float(np.sqrt(test_mse))
test_mae = mean_absolute_error(test_targets, test_predictions)
test_r2 = r2_score(test_targets, test_predictions)

print("--- Test set metrics ---")
print(f"TEST RMSE: {test_rmse:.4f}")
print(f"TEST MSE: {test_mse:.4f}")
print(f"TEST MAE: {test_mae:.4f}")
print(f"TEST R^2: {test_r2:.4f}")
print("Repo STDK run completed.")


