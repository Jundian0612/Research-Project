# Weather2K space split sweep results

本資料夾整理這次空間切分重複實驗的輸出結果。這批實驗固定 loss 權重，不掃描 alpha/lambda：

```text
total_loss = 1.0 * loss_obs + 1.0 * loss_unobs
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

## 三個外推情境

| 情境 | target | 說明 |
| --- | --- | --- |
| `time_extrap_fixed500` | `Target_Time150` | 固定 train+unobs 空間點做時間外推 |
| `space_extrap_fixed850` | `Target_Space100` | 固定 Train700+Val150 時間點做空間外推 |
| `spatiotemp_100x150` | `Target_ST100x150` | 同時做時間與空間外推 |

## 資料夾內容

| 資料夾 | 檔案數 | 內容 |
| --- | ---: | --- |
| `tables/` | 6 | 三個情境的 CSV/MD 總表 |
| `dlinear_frk_json/` | 12 | DLinear+FRK 在 3 個情境 x 4 個空間切分下的完整 JSON |
| `dlinear_best_json/` | 12 | 同一批實驗中純 DLinear best 的 JSON 參考結果 |
| `stdk_json/` | 12 | STDK 在 3 個情境 x 4 個空間切分下的 JSON baseline |
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
| `alpha_obs` | 固定為 `1.0`，STDK 為空 |
| `lambda_unobs` | 固定為 `1.0`，STDK 為空 |
| `seed` | seed 編號，mean/std 列為 `mean +/- std` |
| `RMSE`, `MSE`, `MAE`, `R2` | 評估指標 |

## 檔名規則

DLinear+FRK JSON 檔名包含情境、空間切分與固定 loss 權重：

```text
dlinear_autofrk_frkloss_test_100to500_metrics_{scenario}_s{train}u{unobs}e{target}_alpha1p0_lam1p0.json
```

例如：

```text
dlinear_autofrk_frkloss_test_100to500_metrics_time500_s200u300e100_alpha1p0_lam1p0.json
```

STDK JSON 檔名格式：

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

自訂空間切分：

```bash
SPACE_SPLITS='100/400/100,200/300/100' python experiments_runner.py
```

## 備註

本資料夾內有備份 `params/2K_best_dlinear_params_500to100.json`。按照這次整理要求，根目錄下的參數檔也會保留，不會刪除。

