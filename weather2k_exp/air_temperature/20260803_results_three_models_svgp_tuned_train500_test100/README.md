# Weather2K 氣溫三模型正式實驗（2026-08-03）

本資料夾整理 Weather2K 氣溫資料的 DLinear+FRK、STDK 與調參後
SVGP 三模型正式比較結果。檔案原先輸出於 `weather2k_exp/` 根目錄，
現統一收納至 `air_temperature/`，以免與氣壓及相對濕度實驗混淆。

## 實驗設定

- 目標變數：air temperature
- 空間設定：500 個 supervised training stations、100 個 held-out target stations
- 訓練站組成：100 observed + 400 supervised-unobserved
- 時間切分：Train700 / Val150 / Test150
- seeds：41--45
- 正式情境：時間外推、空間外推、時空外推
- DLinear+FRK：沿用先前 Optuna 保存的 DLinear 最佳參數；空間切分固定為 100 observed + 400 supervised-unobserved + 100 target；loss 權重為 `alpha_obs=1.0`、`lambda_unobs=1.0`
- STDK：採用 spatial-adapter baseline 架構及對齊 `kaust.yaml` 的主要訓練參數，未另行調參
- SVGP：採用調參後的 Matérn-periodic kernel、1024 inducing points、learning rate 0.001、500 epochs
- 評估：所有預測還原至原始氣溫尺度後計算 RMSE、MSE、MAE 與 flatten global R²

## 資料夾內容

- `model_metrics_json/`：DLinear+FRK、STDK 與 SVGP 的逐模型完整輸出 JSON。
- `comparison_tables/`：三個正式情境的整合比較表，包含 CSV 與 Markdown 版本。
- `legacy_parameters/`：本次實驗曾使用、現已被聯合調參流程取代的歷史參數檔。

檔名中的情境縮寫：

- `time500`：時間外推。
- `space850`：空間外推。
- `st_100x150`：時空外推。

## 本次實驗所用程式與參數

以下檔案保留在 `weather2k_exp/` 根目錄，供正式執行或後續調參使用：

- `2K_DLinear_FRK_hybridloss.py`
- `2K_STDK_500train_100test.py`
- `2K_SVGP_500train_100test.py`
- `experiments_runner.py`
- `tune_svgp_hyperparams.py`
- `tune_dlinear_and_frk_hyperparams.py`
- `run_svgp_500train_100test.sh`
- `2K_best_dlinear_and_frk_params_500to100.json`（執行新 tuner 後產生）

舊的 `2K_best_dlinear_params_500to100.json` 只包含先前 Optuna 找到的
DLinear 參數，未包含 FRK 參數，現已移至 `legacy_parameters/` 保存。
正式程式改為讀取聯合調參產生的
`2K_best_dlinear_and_frk_params_500to100.json`；在執行新 tuner 前，
此檔案尚不存在，因此不能直接啟動正式 DLinear+FRK 實驗。

## 保存原則

本資料夾是已完成的氣溫正式實驗結果，不是目前程式的執行目錄。
後續新實驗可能在根目錄產生同名輸出；請勿覆寫或刪除本資料夾中的
既有結果。氣壓與相對濕度結果應分別保存於 `air_pressure/` 與
`relative_humidity/`。
