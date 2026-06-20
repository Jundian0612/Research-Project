# Weather2K selected setting results

本資料夾整理這次指定設定的實驗結果。

## 指定設定

```text
space split = 100/400/100
DLinear+FRK: alpha_obs = 1.0, lambda_unobs = 1.0
DLinear+FRK max epochs = 350
STDK max epochs = 350
patience = 30
```

空間切分代表：

```text
train_space / unobs_space / target_space = 100 / 400 / 100
```

## 三個外推情境

| 情境 | Target | 說明 |
| --- | --- | --- |
| `time_extrap_fixed500` | `Target_Time150` | 固定 train+unobs 空間點做時間外推 |
| `space_extrap_fixed850` | `Target_Space100` | 固定 Train700+Val150 時間點做空間外推 |
| `spatiotemp_100x150` | `Target_ST100x150` | 同時做時間與空間外推 |

## Mean +/- std 摘要

| 情境 | Target | Model | RMSE | MSE | MAE | R2 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Scenario 1 時間外推 | `Target_Time150` | DLinear+FRK | 4.563634 +/- 0.064969 | 20.830975 +/- 0.592398 | 3.457598 +/- 0.059638 | 0.300342 +/- 0.014878 |
| Scenario 1 時間外推 | `Target_Time150` | STDK | 4.708012 +/- 0.174915 | 22.195976 +/- 1.646368 | 3.683134 +/- 0.105122 | -0.816687 +/- 0.239503 |
| Scenario 2 空間外推 | `Target_Space100` | DLinear+FRK | 4.579210 +/- 0.361900 | 21.100140 +/- 3.409306 | 3.503097 +/- 0.185494 | 0.496449 +/- 0.041450 |
| Scenario 2 空間外推 | `Target_Space100` | STDK | 4.758942 +/- 0.338879 | 22.762369 +/- 3.266255 | 3.709924 +/- 0.223225 | 0.124536 +/- 0.123484 |
| Scenario 3 時空外推 | `Target_ST100x150` | DLinear+FRK | 4.718386 +/- 0.364429 | 22.395980 +/- 3.493627 | 3.576065 +/- 0.209345 | 0.245076 +/- 0.056131 |
| Scenario 3 時空外推 | `Target_ST100x150` | STDK | 4.742824 +/- 0.349888 | 22.616801 +/- 3.338056 | 3.740498 +/- 0.230877 | -0.892704 +/- 0.446342 |

## 資料夾內容

| 資料夾 | 檔案數 | 內容 |
| --- | ---: | --- |
| `tables/` | 6 | 三個情境的 CSV/MD 總表，包含每個 seed 與 mean +/- std |
| `dlinear_frk_json/` | 3 | DLinear+FRK 三個情境的完整 JSON |
| `dlinear_best_json/` | 3 | 同一批實驗中純 DLinear best 的 JSON 參考結果 |
| `stdk_json/` | 3 | STDK 三個情境的 JSON baseline |
| `loss_plots/` | 3 | seed 41 的三個情境 loss 圖 |
| `params/` | 1 | DLinear 使用的 best params 備份 |

## 主要看哪個檔案

優先看：

```text
tables/results_time_extrap_fixed500.csv
tables/results_space_extrap_fixed850.csv
tables/results_spatiotemp_100x150.csv
```

每個表都包含：

```text
row_type = seed      每個 seed 的結果
row_type = mean_std  多 seed 的 mean +/- std
```

## Loss 圖

`loss_plots/` 裡有 seed 41 的三張 loss 圖：

```text
loss_plots/dlinear_autofrk_frkloss_test_100to500_metrics_time500_s100u400e100_alpha1p0_lam1p0_seed41_loss.png
loss_plots/dlinear_autofrk_frkloss_test_100to500_metrics_space850_s100u400e100_alpha1p0_lam1p0_seed41_loss.png
loss_plots/dlinear_autofrk_frkloss_test_100to500_metrics_st_100x150_s100u400e100_alpha1p0_lam1p0_seed41_loss.png
```

圖中包含：

```text
loss_obs_scaled
loss_unobs_scaled
weighted_loss_obs_scaled
weighted_loss_unobs_scaled
loss_total_scaled
val_4plus5_rmse_raw
```

## 重新產生

在 conda env `geospatial-neural-adapter` 裡執行：

```bash
cd /home/jundian/Research-Project/weather2k_exp
SPACE_SPLITS='100/400/100' ALPHA_LIST='1.0' LAMBDA_LIST='1.0' python experiments_runner.py
```

只跑 DLinear+FRK，不跑 STDK：

```bash
SPACE_SPLITS='100/400/100' ALPHA_LIST='1.0' LAMBDA_LIST='1.0' RUN_STDK=0 python experiments_runner.py
```

## 備註

本資料夾內有備份 `params/2K_best_dlinear_params_500to100.json`。根目錄下的參數檔也會保留，不會刪除。

