# Weather2K selected setting results

本資料夾整理這次指定設定的實驗結果。

## 指定設定

```text
space split = 400/100/100
DLinear+FRK: alpha_obs = 2.0, lambda_unobs = 0.5
DLinear+FRK max epochs = 350
STDK max epochs = 350
patience = 30
```

空間切分代表：

```text
train_space / unobs_space / target_space = 400 / 100 / 100
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
| Scenario 1 時間外推 | `Target_Time150` | DLinear+FRK | 3.849525 +/- 0.046606 | 14.821017 +/- 0.360537 | 2.972261 +/- 0.054955 | 0.497497 +/- 0.017214 |
| Scenario 1 時間外推 | `Target_Time150` | STDK | 5.051728 +/- 0.741156 | 26.069266 +/- 7.410794 | 4.014788 +/- 0.581784 | -1.186002 +/- 0.645903 |
| Scenario 2 空間外推 | `Target_Space100` | DLinear+FRK | 4.711301 +/- 0.447390 | 22.396512 +/- 4.252365 | 3.591688 +/- 0.302258 | 0.484301 +/- 0.055447 |
| Scenario 2 空間外推 | `Target_Space100` | STDK | 4.629088 +/- 0.316859 | 21.528854 +/- 2.960816 | 3.596656 +/- 0.181845 | 0.162751 +/- 0.112914 |
| Scenario 3 時空外推 | `Target_ST100x150` | DLinear+FRK | 5.094318 +/- 0.464516 | 26.167853 +/- 4.806550 | 3.881862 +/- 0.342328 | 0.151315 +/- 0.080302 |
| Scenario 3 時空外推 | `Target_ST100x150` | STDK | 5.472112 +/- 0.647061 | 30.362700 +/- 7.018909 | 4.324541 +/- 0.578898 | -1.556014 +/- 0.658507 |

## 資料夾內容

| 資料夾 | 檔案數 | 內容 |
| --- | ---: | --- |
| `tables/` | 6 | 三個情境的 CSV/MD 總表，包含每個 seed 與 mean +/- std |
| `dlinear_frk_json/` | 3 | DLinear+FRK 三個情境的完整 JSON |
| `dlinear_best_json/` | 3 | 同一批實驗中純 DLinear best 的 JSON 參考結果 |
| `stdk_json/` | 3 | STDK 三個情境的 JSON baseline |
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

## 重新產生

在 conda env `geospatial-neural-adapter` 裡執行：

```bash
cd /home/jundian/Research-Project/weather2k_exp
SPACE_SPLITS='400/100/100' ALPHA_LIST='2.0' LAMBDA_LIST='0.5' python experiments_runner.py
```

只跑 DLinear+FRK，不跑 STDK：

```bash
SPACE_SPLITS='400/100/100' ALPHA_LIST='2.0' LAMBDA_LIST='0.5' RUN_STDK=0 python experiments_runner.py
```

## 備註

本資料夾內有備份 `params/2K_best_dlinear_params_500to100.json`。根目錄下的參數檔也會保留，不會刪除。

