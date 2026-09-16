# Weather2K air-temperature experiments

此目錄是 Weather2K 氣溫實驗的結果根目錄。每個實驗保留自己的 `README.md`，並依產生日期與實驗用途分開保存。

## 目前流程

- `qconvlstm_location_specific_tuning_20260915/`：最新逐目標位置 QConvLSTM 正式調參；使用 seeds 41/42 的 Val150，最佳設定為 grid 11、radius 0.3、filters 32、weight decay 0。
- `qconvlstm_obs100_ema_tuning_20260912/`：採用 batch-average validation 與 EMA 的最新 STDK+Q 調參結果。
- `STDK_QConvLSTM_formal_ema_all3_rtx2000_20260912/`：使用最新流程執行三種情境的正式結果目錄。
- `three_baselines_onefit_seed41_rtx2000_20260912/`：SVGP、純 STDK、DLinear+FRK 在相同 seed 41 下，各訓練一次並評估三種情境；含四模型比較與舊分別訓練對照。
- `STDK_onefit_aligned_seed41_rtx2000_20260913/`：對齊 pinned Spatial-adapter STDK 訓練規則後重新執行的純 STDK seed 41 三情境結果。
- `dlinear_frk_tuning_current/`、`qconvlstm_tuning_current/`、`svgp_tuning_current/`：程式使用的調參續跑目錄；根層檔案位置不可任意搬動。

## 目錄命名

- `*_formal_*`：正式或準正式評估。
- `*_tuning_*`、`*_tuning_current`：validation 選參過程與續跑狀態。
- `*_reprocheck_*`、`*_repro_check_*`：重現性或實作對齊檢查。
- `results_*`：早期 baseline、切分或超參數實驗封存。
- 名稱中的日期採 `YYYYMMDD`；`rtx2000` 表示執行裝置。

## 實驗內部結構

整理過的實驗依內容使用以下子目錄：

- `metrics/`：JSON 評估結果。
- `forecasts/`：逐筆預測 CSV。
- `tables/`：比較表與 Markdown 表格。
- `params/`：固定參數或試驗參數。
- `summary/`：調參摘要與最佳參數。
- `trials/`：各次調參試驗。
- `logs/`：執行紀錄。
- `metadata/` 或 `provenance/`：程式雜湊、環境及追溯資料。

各資料夾根層的 `README.md` 說明該次實驗。歷史結果只重新分類，沒有刪除或重新計算；比較模型時仍須依各 README 記載的資料切分、標準化、程式版本與評估流程判斷是否可直接比較。
