# Weather2K DLinear+FRK lambda sweep results

本資料夾整理 `weather2k_exp` 根目錄下這批 DLinear+FRK 輸出檔。這批結果是較早的 **lambda-only sweep**，也就是只調整：

```text
total_loss = loss_obs + lambda_unobs * loss_unobs
```

不是後來的 alpha/lambda 4x4 sweep。後來那批已整理在：

```text
../results_dlinear_frk_alpha_lambda_sweep_20260608/
```

## 實驗情境

時間使用最後 1000 個時間點：

```text
Train700 / Val150 / Target150
```

空間抽 600 個空間點：

```text
obs100 / unobs400 / target100
```

三個情境：

| 情境 suffix | 情境 | target |
| --- | --- | --- |
| `time500` | 固定前 500 個空間點做時間外推 | `Target_Time150` |
| `space850` | 固定前 850 個時間點做空間外推 | `Target_Space100` |
| `st_100x150` | 同時做時間和空間外推 | `Target_ST100x150` |

## 掃描權重

這批檔案包含：

```text
lambda_unobs = 0.5, 1.0, 1.5, 2.0
```

另外也保留沒有 lambda suffix 的原始輸出檔，例如：

```text
dlinear_autofrk_frkloss_test_100to500_metrics_time500.json
```

## 資料夾內容

| 資料夾 | 檔案數 | 內容 |
| --- | ---: | --- |
| `dlinear_frk_json/` | 15 | DLinear+FRK 的完整 JSON 結果 |
| `dlinear_best_json/` | 15 | 同一批實驗中純 DLinear best 的 JSON 參考結果 |
| `params/` | 1 | DLinear 使用的 best params |

## JSON 怎麼看

DLinear+FRK 主要看：

```text
dlinear_frk_json/
```

檔名格式：

```text
dlinear_autofrk_frkloss_test_100to500_metrics_{scenario_suffix}_lam{lambda}.json
```

例如：

```text
dlinear_autofrk_frkloss_test_100to500_metrics_time500_lam1p0.json
```

JSON 裡通常包含：

```text
seed_runs      每個 seed 的結果與 loss history
mean_metrics   多 seed 平均
std_metrics    多 seed 標準差
```

## 注意

這個資料夾沒有 `results_*.csv` 總表，因為目前根目錄下這批輸出只有 JSON 檔。若需要表格化比較，建議以後優先使用：

```text
../results_dlinear_frk_alpha_lambda_sweep_20260608/tables/
```

那批表格已經包含每個 seed 與 mean +/- std。

