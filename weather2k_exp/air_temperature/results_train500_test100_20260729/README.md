# Weather2K air-temperature results: train500 / test100

本資料夾整理 2026-07-28 完成、於 2026-07-29 歸檔的氣溫實驗結果。

## 實驗基準

```text
sample stations = 600
supervised train stations = 500 (obs100 + unobs400)
held-out spatial test stations = 100
time split = 700 train / 150 validation / 150 test
seeds = 41, 42, 43, 44, 45
max epochs = 350
patience = 30
DLinear+FRK alpha_obs = 1.0
DLinear+FRK lambda_unobs = 1.0
```

三個模型皆使用 `air_temperature`，並在相同的站點抽樣、時間切分與 final targets 下比較。

## 與前一版的差異

前一版指：

```text
../results_s100u400e100_alpha1_lam1_20260609/
```

兩版都使用相同的氣溫資料、seeds 41–45、600 個抽樣站、100/400/100
空間角色、700/150/150 時間切分，以及 DLinear+FRK 的
`alpha_obs=1.0`、`lambda_unobs=1.0`。主要差異如下：

| 項目 | 前一版（20260609） | 本版（20260729） |
| --- | --- | --- |
| STDK Train700 | 只直接使用 obs100 | 直接使用 obs100 + unobs400，共 500 站 |
| STDK Val150 | obs100 + unobs400，共 500 站 | obs100 + unobs400，共 500 站 |
| DLinear+FRK Train700 | obs100 的 DLinear loss + unobs400 的 FRK spatial loss | 相同，未改變 |
| SVGP | 未包含在原始兩模型比較中 | 加入比較，直接使用 500 站訓練 |
| JSON metadata | `train_sample_size=100` | `observed_sample_size=100`、`train_sample_size=500` |
| 新結果檔名 | `s100u400e100` | `s100u400e100_train500_test100` |

因此，DLinear+FRK 的兩版結果完全相同；本版的主要實驗變化是讓 STDK
也取得與 DLinear+FRK 相同範圍的 500 站 Train700 監督，並加入 SVGP。

### DLinear+FRK mean 變化

| 情境 | 前一版 RMSE | 本版 RMSE | 差異 |
| --- | ---: | ---: | ---: |
| 時間外推 | 4.563634 +/- 0.064969 | 4.563634 +/- 0.064969 | 0 |
| 空間外推 | 4.579210 +/- 0.361900 | 4.579210 +/- 0.361900 | 0 |
| 時空外推 | 4.718386 +/- 0.364429 | 4.718386 +/- 0.364429 | 0 |

### STDK mean 變化

| 情境 | 前一版 RMSE | 本版 RMSE | mean RMSE 變化 |
| --- | ---: | ---: | ---: |
| 時間外推 | 4.708012 +/- 0.174915 | 4.729141 +/- 0.662610 | +0.021129 |
| 空間外推 | 4.758942 +/- 0.338879 | 4.493263 +/- 0.168650 | -0.265679 |
| 時空外推 | 4.742824 +/- 0.349888 | 4.964020 +/- 0.597754 | +0.221196 |

RMSE 變化為「本版 mean − 前一版 mean」；負值代表改善。STDK 增加至
500 站訓練後，空間外推明顯改善，但時間與時空外推沒有改善，且兩者的
seed 間標準差增大。這表示新增空間監督主要提升空間擬合，現有 STDK
超參數尚未針對 500 站訓練重新最佳化。

## Mean +/- std

| 情境 | Target | Model | RMSE | MSE | MAE | R2 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 時間外推 | `Target_Time150` | SVGP | 5.334396 +/- 0.401271 | 28.616802 +/- 4.362317 | 4.172473 +/- 0.315521 | -1.118279 +/- 0.301377 |
| 時間外推 | `Target_Time150` | DLinear+FRK | **4.563634 +/- 0.064969** | **20.830975 +/- 0.592398** | **3.457598 +/- 0.059638** | **0.300342 +/- 0.014878** |
| 時間外推 | `Target_Time150` | STDK | 4.729141 +/- 0.662610 | 22.803824 +/- 6.633971 | 3.785485 +/- 0.525705 | -0.918841 +/- 0.623904 |
| 空間外推 | `Target_Space100` | SVGP | 4.875658 +/- 0.198158 | 23.811306 +/- 1.959704 | 3.820010 +/- 0.093201 | 0.082190 +/- 0.045644 |
| 空間外推 | `Target_Space100` | DLinear+FRK | 4.579210 +/- 0.361900 | 21.100140 +/- 3.409306 | **3.503097 +/- 0.185494** | **0.496449 +/- 0.041450** |
| 空間外推 | `Target_Space100` | STDK | **4.493263 +/- 0.168650** | **20.217852 +/- 1.526021** | 3.525435 +/- 0.101153 | 0.222523 +/- 0.026736 |
| 時空外推 | `Target_ST100x150` | SVGP | 5.308034 +/- 0.674596 | 28.630308 +/- 7.402294 | 4.137914 +/- 0.487242 | -1.066771 +/- 0.428056 |
| 時空外推 | `Target_ST100x150` | DLinear+FRK | **4.718386 +/- 0.364429** | **22.395980 +/- 3.493627** | **3.576065 +/- 0.209345** | **0.245076 +/- 0.056131** |
| 時空外推 | `Target_ST100x150` | STDK | 4.964020 +/- 0.597754 | 24.998804 +/- 6.234479 | 3.930796 +/- 0.524206 | -0.997851 +/- 0.567075 |

粗體為各情境、各指標的最佳 mean。RMSE、MSE、MAE 越低越好，R2 越高越好。

## 結果摘要

- 時間外推：DLinear+FRK 四個指標皆最佳。
- 空間外推：STDK 的 RMSE/MSE 最佳；DLinear+FRK 的 MAE/R2 最佳。
- 時空外推：DLinear+FRK 四個指標皆最佳。
- SVGP 在三個情境皆未取得最佳 mean。

## 資料夾內容

| 資料夾 | 檔案數 | 內容 |
| --- | ---: | --- |
| `tables/` | 6 | 三個情境的 CSV/Markdown 完整比較表 |
| `dlinear_frk_json/` | 3 | DLinear+FRK 完整結果 |
| `dlinear_best_json/` | 3 | DLinear best/rerun 完整結果 |
| `stdk_json/` | 3 | STDK 完整結果 |
| `svgp_json/` | 3 | SVGP 完整結果 |
| `params/` | 1 | DLinear best parameters 備份 |

每個模型與情境均包含 seeds 41–45，以及 mean/std 統計。

## 建議優先查看

```text
tables/results_time_extrap_fixed500.csv
tables/results_space_extrap_fixed850.csv
tables/results_spatiotemp_100x150.csv
```
