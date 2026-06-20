# Weather2K DLinear+FRK alpha/lambda sweep results

本資料夾整理這次 Weather2K 實驗的輸出結果。實驗目標是比較 DLinear+FRK 與 STDK，並掃描 hybrid loss 裡兩個權重：

```text
total_loss = alpha_obs * loss_obs + lambda_unobs * loss_unobs
```

## 實驗設定

時間使用最後 1000 個時間點，切成：

```text
Train700 / Val150 / Target150
```

空間抽 600 個空間點，切成：

```text
obs100 / unobs400 / target100
```

三個外推情境：

| 情境 | 最終評估 target | 意義 |
| --- | --- | --- |
| `time_extrap_fixed500` | `Target_Time150` | 固定前 500 個空間點，預測最後 150 個時間點，也就是 block 7+8 |
| `space_extrap_fixed850` | `Target_Space100` | 固定前 850 個時間點，預測最後 100 個空間點，也就是 block 3+6 |
| `spatiotemp_100x150` | `Target_ST100x150` | 同時預測最後 100 個空間點與最後 150 個時間點，也就是 block 9 |

## 掃描權重

DLinear+FRK 掃描：

```text
alpha_obs = 0.5, 1.0, 1.5, 2.0
lambda_unobs = 0.5, 1.0, 1.5, 2.0
```

所以每個情境有 16 組 DLinear+FRK。STDK 不吃 alpha/lambda，所以每個情境只有一組 baseline。

## 資料夾內容

| 資料夾 | 內容 |
| --- | --- |
| `tables/` | 每個情境整理好的一張總表，包含每個 seed 結果與 mean +/- std |
| `dlinear_frk_json/` | DLinear+FRK 每組 alpha/lambda 的完整 JSON 結果 |
| `dlinear_best_json/` | 同一批實驗中純 DLinear best 的 JSON 參考結果 |
| `stdk_json/` | STDK 三個情境的 JSON baseline 結果 |
| `loss_plots/` | 已產生的 seed 41 loss 圖範例 |

目前檔案數：

```text
tables: 6
dlinear_frk_json: 48
dlinear_best_json: 48
stdk_json: 3
loss_plots: 15
```

## 主要看哪個檔案

優先看 `tables/` 裡的三個 CSV：

```text
tables/results_time_extrap_fixed500.csv
tables/results_space_extrap_fixed850.csv
tables/results_spatiotemp_100x150.csv
```

每個 CSV 只有一個情境，裡面同時包含：

```text
row_type = seed      單一 seed 結果
row_type = mean_std  五個 seed 的 mean +/- std
```

欄位說明：

| 欄位 | 說明 |
| --- | --- |
| `scenario` | 實驗情境 |
| `target` | 該情境真正拿來比較的 target |
| `model` | 模型名稱，主要比較 `DLINEAR + differentiable_FRK` 與 `STDK` |
| `alpha_obs` | `loss_obs` 的權重，STDK 為空 |
| `lambda_unobs` | `loss_unobs` 的權重，STDK 為空 |
| `row_type` | `seed` 或 `mean_std` |
| `seed` | seed 編號，mean/std 列為 `mean +/- std` |
| `RMSE`, `MSE`, `MAE`, `R2` | 評估指標 |

## JSON 結果怎麼看

DLinear+FRK 的 JSON 位於：

```text
dlinear_frk_json/
```

檔名格式：

```text
dlinear_autofrk_frkloss_test_100to500_metrics_{scenario_suffix}_alpha{alpha}_lam{lambda}.json
```

例如：

```text
dlinear_autofrk_frkloss_test_100to500_metrics_time500_alpha1p0_lam1p5.json
```

JSON 裡面包含：

```text
seed_runs      每個 seed 的完整結果與 loss_history
mean_metrics   多 seed 平均
std_metrics    多 seed 標準差
```

## 重新產生完整實驗

在 conda env `geospatial-neural-adapter` 裡執行：

```bash
cd /home/jundian/Research-Project/weather2k_exp
python experiments_runner.py
```

只先跑單一 seed 測試：

```bash
SEED_LIST='[41]' python experiments_runner.py
```

只跑 DLinear+FRK，不跑 STDK：

```bash
RUN_STDK=0 python experiments_runner.py
```

自訂掃描權重：

```bash
ALPHA_LIST='0.5,1,1.5,2' LAMBDA_LIST='0.5,1,1.5,2' python experiments_runner.py
```

## 產生 loss 圖

使用：

```bash
python plot_dlinear_loss.py dlinear_autofrk_frkloss_test_100to500_metrics_time500_alpha1p0_lam1p0.json --seed 41
```

圖中會包含：

```text
loss_obs_scaled
loss_unobs_scaled
weighted_loss_obs_scaled
weighted_loss_unobs_scaled
loss_total_scaled
val_4plus5_rmse_raw
```

其中：

```text
weighted_loss_obs_scaled = alpha_obs * loss_obs_scaled
weighted_loss_unobs_scaled = lambda_unobs * loss_unobs_scaled
loss_total_scaled = weighted_loss_obs_scaled + weighted_loss_unobs_scaled
```

