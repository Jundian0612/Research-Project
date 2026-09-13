# Weather2K q50 Stage 2 capacity tuning（新裝置）

執行日期：2026-09-08 至 2026-09-09。4 configurations × seeds 41、42，共 8 runs 已完成，
累計 elapsed_seconds 約 2.16 小時。這是 validation-only 歷史結果，不是正式 Test150 或五-seed 結論。

## 實驗來源與限制

本輪沿用 [舊裝置 Stage 1](../STDK_QConvLSTM_Weather2K_q50_interface_tuning_20260908/README.md)
選出的 grid=5、radius=0.1，但 Stage 2 改在新裝置執行。
兩台的 Python、PyTorch、CUDA runtime、NumPy、scikit-learn 與 GPU 不同。
不能用兩階段 RMSE 的高低直接判斷調參改善；也尚未驗證舊 Stage 1 最佳 interface 在新環境仍然最佳。

固定流程：air_temperature、train500 / strict held-out100、Train700 / Val150 / Test150、
Spatial-adapter STDK、direct 5-to-5 blocks、github_3block、q50 only。
LR=0.0001、batch size=64、max epochs=40、patience=10。
僅使用 train500 的 Train700 訓練和 chronological Val150 選參，未計算 Test150 或 held-out100 指標，未保存 checkpoint。

## 結果

| Trial | Filters | Weight decay | Mean standardized Val RMSE | Std（2 seeds） |
|---:|---:|---:|---:|---:|
| 0 | 32 | 0 | 0.669337 | 0.026920 |
| **1** | **32** | **1e-5** | **0.668526** | **0.024742** |
| 2 | 64 | 0 | 0.681591 | 0.034084 |
| 3 | 64 | 1e-5 | 0.683398 | 0.033849 |

本輪最佳 trial 1，seed 41 / 42 RMSE 分別為 0.643784 / 0.693268。
filters=32 的兩個 weight decay 結果接近，不能據此聲稱正則化效果已穩定成立。

## 環境紀錄

下表由使用者於兩台 terminal 查詢並提供截圖，並非完整 pip lockfile，也不是訓練程序自動記錄的環境快照。

| 項目 | 舊裝置 Stage 1 | 新裝置 Stage 2 |
|---|---|---|
| Python | 3.10.13 | 3.12.3 |
| PyTorch | 2.7.1+cu126 | 2.14.0+cu130 |
| CUDA runtime | 12.6 | 13.0 |
| NumPy | 1.26.4 | 2.4.6 |
| scikit-learn | 1.7.2 | 1.9.0 |
| GPU | GTX 1050 Ti | RTX 2000 Ada Generation |

兩邊診斷程序回報 deterministic algorithms、cuDNN benchmark、cuDNN deterministic 均為 False。
資料與三份模型程式 SHA256 已由兩台輸出核對相同。
seed 42 同設定跨裝置 RMSE 有明顯差異；[新裝置重跑檢查](../STDK_QConvLSTM_Weather2K_seed42_repro_check_20260909/README.md)
只出現小幅波動，原因尚未完全定位。

## 檔案與後續

- `trials/`：16 份原始 trial params、validation 與 summary JSON。
- `summary/`：本輪總表、最佳 capacity 與 tuning 參數。
- `inputs_and_params/`：執行所用 Stage 1 interface 與完成後正式參數的快照。
- `provenance/`：原 runner 產生的說明、封存搬移清單及 SHA256。
- `logs/stage2_capacity.log`：本機執行 log；受 `.gitignore` 的 `*.log` 規則忽略，不會隨一般 git add 上傳。

原 runner 說明中的「formal two-stage」是自動文字，實際解讀以本文件的跨環境限制為準。
根目錄 `qconvlstm_interface_best_params.json` 與 `2K_best_stdk_qconvlstm_params_500to100.json`
保留為 operational copies，後者目前仍是本輪最佳值，不表示已完成新裝置一致環境的兩階段調參。

下一步在新装置固定環境，以新的 output-dir 重跑 Stage 1 → Stage 2。
本封存不應被覆寫。若新 Stage 1 最佳 interface 相同，需確認完整設定及環境一致後才考慮沿用本輪 Stage 2。
搬走根目錄 trial reports 後，runner 的預設 resume 不會自動搜尋這個封存資料夾。
本次僅整理結果，未啟動新訓練、改參數或 push。
