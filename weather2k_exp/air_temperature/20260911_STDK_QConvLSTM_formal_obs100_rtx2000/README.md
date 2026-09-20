# STDK + QConvLSTM obs100 對齊版正式評估

> **歷史結果（pre-EMA）**：本資料夾保留 2026-09-11 當時的正式輸出，供追溯與比較。
> 當時前段 STDK 未使用原 repo 的 EMA，validation 使用逐點加權平均。
> 2026-09-12 起的現行流程已改成原 repo 的 EMA 與 batch-average validation；因此這裡的
> locked params 和 seeds 結果不得與新版 run 混合彙總，也不能當成新版正式結果續跑。

狀態：歷史參數曾鎖定；正式 seed 41 已於 2026-09-11 完成並通過當時的輸出完整性檢查。

參數來源是相鄰的 `20260910_qconvlstm_obs100_tuning`：grid=5、radius=0.1、filters=32、
weight decay=0、LR=0.0001、batch=64、max epochs=40、patience=10。
正式執行固定讀取 `params/locked_params.json`，不因 seed41 Test150 表現重新選參。
`metadata/code_and_params.sha256` 記錄啟動前的模型程式與參數指紋。

## 為何同時輸出 fitted STDK

每個正式 run 本來就會訓練 Spatial-adapter STDK 來產生 QConvLSTM grids。
程式會直接評估記憶體中這一組已 fitted STDK，不會重新訓練第二套 STDK，額外成本主要是推論與存檔。

這個配對對照與 STDK+Q 共用同一 seed、測站、時間、標準化及 fitted STDK，因此能直接回答
「接上 QConvLSTM 後是否改善」。以前的純 STDK 結果仍有效並保留，但來自舊執行環境；
部分歷史版本的前處理或程式軌跡不同，適合做外部基準，不能取代此配對消融。

正式 JSON 會包含 `results`（STDK+Q）和 `fitted_stdk_baseline.results`；另輸出六列 comparison CSV，
以及兩模型各三份逐點預測。共同 point metrics 為原始氣溫尺度的 RMSE/MSE/MAE/global R²；
兩者也會計算 q05/q95 的 MPIW與coverage。

## 執行 seed 41

在可見 RTX 2000 Ada GPU 的 terminal 中：

```bash
tmux new -s weather2k-formal-obs100
cd /home/user/Research-Project
source .venv-wsl/bin/activate
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
python -m pip freeze > weather2k_exp/air_temperature/20260911_STDK_QConvLSTM_formal_obs100_rtx2000/metadata/pip-freeze.txt
set -o pipefail
PYTHONUNBUFFERED=1 python -u weather2k_exp/2K_STDK_QConvLSTM.py \
  --params-file weather2k_exp/air_temperature/20260911_STDK_QConvLSTM_formal_obs100_rtx2000/params/locked_params.json \
  --seed 41 --forecast-mode block5to5 --prediction-mode direct \
  --output-dir weather2k_exp/air_temperature/20260911_STDK_QConvLSTM_formal_obs100_rtx2000 \
  2>&1 | tee -a weather2k_exp/air_temperature/20260911_STDK_QConvLSTM_formal_obs100_rtx2000/logs/seed41.log
```

不要加入 `--validation-only`、`--q50-only` 或 `--smoke-test`。看到
`seed=41 device=cuda train=500 heldout=100` 才是正式 GPU run。
可用 Ctrl+b、放開、再按 d 離開 tmux；回來用 `tmux attach -t weather2k-formal-obs100`。

完成後先檢查 JSON、六列比較表、三情境筆數與 truth/index 對齊。流程正確就保留 seed41，
用相同程式、參數及環境補跑42–45，不依 seed41 Test結果改參數。

## 補跑 seeds 42–45

使用 `weather2k_exp/experiments_runner.py`，並以環境變數指定 seeds 42–45、此處的
`params/locked_params.json`、`block5to5` 與 `direct`。每個 seed 的三種情境會直接寫入本資料夾；
完成後再統一整理 logs、metrics 與 forecasts。

## Seed 41 正式結果

執行時間為 3387.31 秒（約 56 分 27 秒）。正式輸出包含 JSON、六列比較表，
以及兩個模型各三份逐點預測 CSV；Time150、Space100、ST100x150 的列數分別為
75,000、85,000、15,000，均符合預期。

| 情境 | 模型 | RMSE | MAE | R² | MPIW 90 | Coverage 90 |
|---|---|---:|---:|---:|---:|---:|
| Time150 | fitted STDK | 4.2766 | 3.4223 | 0.3968 | 13.9472 | 0.9167 |
| Time150 | STDK+QConvLSTM | 4.1908 | 3.3512 | 0.4207 | 11.6450 | 0.8211 |
| Space100 | fitted STDK | 4.3235 | 3.4265 | 0.5508 | 12.8285 | 0.8928 |
| Space100 | STDK+QConvLSTM | 4.2841 | 3.3479 | 0.5590 | 12.3242 | 0.8667 |
| ST100x150 | fitted STDK | 4.2533 | 3.4263 | 0.3699 | 14.0297 | 0.9217 |
| ST100x150 | STDK+QConvLSTM | 4.3056 | 3.4612 | 0.3544 | 12.1606 | 0.8259 |

Seed 41 中，STDK+QConvLSTM 在 Time150 與 Space100 的 RMSE、MAE 和 R² 略優於
同次 fitted STDK，但在 ST100x150 略差。三個情境的預測區間都較窄，coverage 也較低，
其中 Time150 與 ST100x150 明顯低於名目 90%。這是單一 seed 的初步結果；最終比較需完成
seeds 41–45，再報告跨 seed 平均值與變異。

## 整理後目錄

- `forecasts/`：逐筆預測輸出
- `logs/`：執行紀錄
- `metadata/`：程式與環境追溯資訊
- `metrics/`：JSON 評估結果
- `params/`：固定或試驗參數
- `tables/`：彙整表格

本次整理只調整檔案位置，沒有刪除或重新計算結果。
