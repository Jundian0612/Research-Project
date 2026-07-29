# Legacy Weather2K air-temperature JSON results

本資料夾封存原先散落在 `weather2k_exp/` 根目錄、產生於 2026-07-25 的舊版氣溫實驗 JSON。

## 舊版實驗角色

```text
space split = obs100 / unobs400 / target100
time split = 700 train / 150 validation / 150 test
seeds = 41, 42, 43, 44, 45
DLinear+FRK alpha_obs = 1.0
DLinear+FRK lambda_unobs = 1.0
```

這批結果的 STDK training dataset 僅直接使用 obs100；它不同於目前
`results_train500_test100_20260729/` 中 STDK 與 SVGP 直接使用 500 站訓練的比較基準。

## 資料夾內容

| 資料夾 | 檔案數 | 內容 |
| --- | ---: | --- |
| `dlinear_best_json/` | 3 | DLinear best/rerun 三個情境 |
| `dlinear_frk_json/` | 3 | DLinear+FRK 三個情境 |
| `stdk_json/` | 3 | 舊版 STDK 三個情境 |

這些檔案僅供歷史追蹤，不會被目前的 `experiments_runner.py` 或三個模型程式讀取。

根目錄的 `2K_best_dlinear_params_500to100.json` 是目前 DLinear+FRK 執行時需要的參數檔，因此不在此封存資料夾內。
