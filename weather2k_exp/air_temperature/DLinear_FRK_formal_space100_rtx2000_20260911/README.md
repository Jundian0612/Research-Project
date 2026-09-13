# DLinear+FRK Space100 正式評估

此資料夾保存最新版 DLinear+FRK 在 `100/400/100` 空間切分下的 Space100 正式結果。
時間資料固定取最後 1000 點並切成 Train700、Val150、Test150；模型參數固定讀取
`params/locked_params.json`，不依正式測試結果重新選參。

`params/locked_params.json` 與 `weather2k_exp/air_temperature/dlinear_frk_tuning_current/best_params.json`
內容相同。選定參數包含 `DIFF_FRK_OBS_LOSS_WEIGHT=2.0` 與
`DIFF_FRK_LOSS_WEIGHT=1.5`。`metadata/code_and_params.sha256` 記錄執行程式與鎖定參數的指紋。

`metrics/seed41/` 是從 `four_models_seed41_20260831/json/` 收進來的既有最新版正式結果。
seeds 42–45 使用 `weather2k_exp/experiments_runner.py` 執行，合併結果直接寫入本資料夾。

runner 僅啟用 `space_extrap_fixed850`、DLinear+FRK 與 STDK+QConvLSTM，其他 baseline
不會重跑。STDK+QConvLSTM 的輸出另指定到它原本的正式資料夾。

## 整理後目錄

- `logs/`：執行紀錄
- `metadata/`：程式與環境追溯資訊
- `metrics/`：JSON 評估結果
- `params/`：固定或試驗參數
- `tables/`：彙整表格

本次整理只調整檔案位置，沒有刪除或重新計算結果。
