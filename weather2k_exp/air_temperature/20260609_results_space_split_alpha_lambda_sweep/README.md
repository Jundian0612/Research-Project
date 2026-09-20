# Weather2K space split + alpha/lambda sweep results

本資料夾整理這次完整重複實驗的輸出結果。這批實驗同時掃描空間切分與 DLinear+FRK hybrid loss 的兩個係數：

```text
total_loss = alpha_obs * loss_obs + lambda_unobs * loss_unobs
```

## 實驗設定

時間固定使用最後 1000 個時間點：

```text
Train700 / Val150 / Target150
```

空間總數固定為 600，掃描四組切分：

```text
100/400/100
200/300/100
300/200/100
400/100/100
```

其中三個數字分別代表：

```text
train_space / unobs_space / target_space
```

DLinear+FRK 掃描 loss 係數：

```text
alpha_obs = 0.5, 1.0, 1.5, 2.0
lambda_unobs = 0.5, 1.0, 1.5, 2.0
```

## 三個外推情境

| 情境 | Target | 說明 |
| --- | --- | --- |
| `time_extrap_fixed500` | `Target_Time150` | 固定 train+unobs 空間點做時間外推 |
| `space_extrap_fixed850` | `Target_Space100` | 固定 Train700+Val150 時間點做空間外推 |
| `spatiotemp_100x150` | `Target_ST100x150` | 同時做時間與空間外推 |

## 資料夾內容

| 資料夾 | 檔案數 | 內容 |
| --- | ---: | --- |
| `tables/` | 6 | 三個情境的 CSV/MD 總表 |
| `dlinear_frk_json/` | 192 | DLinear+FRK 在 3 個情境 x 4 個空間切分 x 16 組係數下的完整 JSON |
| `dlinear_best_json/` | 192 | 同一批實驗中純 DLinear best 的 JSON 參考結果 |
| `stdk_json/` | 12 | STDK 在 3 個情境 x 4 個空間切分下的 JSON baseline |
| `loss_plots/` | 3 | `space_split=400/100/100, alpha=2.0, lambda=0.5, seed=41` 的三張 loss 圖 |
| `params/` | 1 | DLinear 使用的 best params 備份 |

## 主要看哪個檔案

優先看 `tables/` 裡的三個 CSV：

```text
tables/results_time_extrap_fixed500.csv
tables/results_space_extrap_fixed850.csv
tables/results_spatiotemp_100x150.csv
```

每個表格都同時包含：

```text
row_type = seed      每個 seed 的結果
row_type = mean_std  多 seed 的 mean +/- std
```

重要欄位：

| 欄位 | 說明 |
| --- | --- |
| `space_split` | 空間切分，格式為 train/unobs/target |
| `n_train_space` | DLinear/STDK 訓練用的空間點數 |
| `n_unobs_space` | FRK loss / 中間 unobs 空間點數 |
| `n_target_space` | 最終空間外推評估點數 |
| `model` | `DLINEAR + differentiable_FRK` 或 `STDK` |
| `alpha_obs` | `loss_obs` 的權重，STDK 為空 |
| `lambda_unobs` | `loss_unobs` 的權重，STDK 為空 |
| `seed` | seed 編號，mean/std 列為 `mean +/- std` |
| `RMSE`, `MSE`, `MAE`, `R2` | 評估指標 |

## 指定結果摘要

條件：

```text
space split = 400/100/100
DLinear+FRK: alpha = 2.0, lambda = 0.5
STDK: 同一個 space split baseline
```

| 情境 | Target | Model | RMSE | MSE | MAE | R2 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Scenario 1 時間外推 | `Target_Time150` | DLinear+FRK | 4.573234 +/- 0.017747 | 20.914789 +/- 0.162517 | 3.578237 +/- 0.016609 | 0.290838 +/- 0.019886 |
| Scenario 1 時間外推 | `Target_Time150` | STDK | 5.051728 +/- 0.741156 | 26.069266 +/- 7.410794 | 4.014788 +/- 0.581784 | -1.186002 +/- 0.645903 |
| Scenario 2 空間外推 | `Target_Space100` | DLinear+FRK | 4.805526 +/- 0.434415 | 23.281794 +/- 4.209898 | 3.661747 +/- 0.297959 | 0.463520 +/- 0.052753 |
| Scenario 2 空間外推 | `Target_Space100` | STDK | 4.629088 +/- 0.316859 | 21.528854 +/- 2.960816 | 3.596656 +/- 0.181845 | 0.162751 +/- 0.112914 |
| Scenario 3 時空外推 | `Target_ST100x150` | DLinear+FRK | 5.443358 +/- 0.380288 | 29.774768 +/- 4.193071 | 4.207182 +/- 0.279170 | 0.030479 +/- 0.053474 |
| Scenario 3 時空外推 | `Target_ST100x150` | STDK | 5.472112 +/- 0.647061 | 30.362700 +/- 7.018909 | 4.324541 +/- 0.578898 | -1.556014 +/- 0.658507 |

## Loss 圖

`loss_plots/` 裡有這組條件的 seed 41 loss 圖：

```text
space split = 400/100/100
alpha = 2.0
lambda = 0.5
seed = 41
```

檔案：

```text
loss_plots/dlinear_autofrk_frkloss_test_100to500_metrics_time500_s400u100e100_alpha2p0_lam0p5_seed41_loss.png
loss_plots/dlinear_autofrk_frkloss_test_100to500_metrics_space850_s400u100e100_alpha2p0_lam0p5_seed41_loss.png
loss_plots/dlinear_autofrk_frkloss_test_100to500_metrics_st_100x150_s400u100e100_alpha2p0_lam0p5_seed41_loss.png
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

## 檔名規則

DLinear+FRK JSON：

```text
dlinear_autofrk_frkloss_test_100to500_metrics_{scenario}_s{train}u{unobs}e{target}_alpha{alpha}_lam{lambda}.json
```

例如：

```text
dlinear_autofrk_frkloss_test_100to500_metrics_time500_s400u100e100_alpha2p0_lam0p5.json
```

STDK JSON：

```text
2K_stdk_metrics_{scenario}_s{train}u{unobs}e{target}.json
```

## 重新產生

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

自訂空間切分或係數：

```bash
SPACE_SPLITS='400/100/100' ALPHA_LIST='2.0' LAMBDA_LIST='0.5' python experiments_runner.py
```

## 備註

本資料夾內有備份 `params/2K_best_dlinear_params_500to100.json`。根目錄下的參數檔也保留，不會刪除。

