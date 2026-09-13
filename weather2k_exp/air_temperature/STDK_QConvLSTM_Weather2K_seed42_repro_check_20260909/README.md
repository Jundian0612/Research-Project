# Weather2K seed 42 同機重跑檢查

日期：2026-09-09。新裝置 RTX 2000 Ada、原 `.venv-wsl` 環境。
目的是檢查同機原流程的波動，沒有更改 deterministic 設定或套件。
此結果僅 validation-only q50，不是正式測試或新增 tuning trial。

## 固定設定與結果

設定及 split JSON 與舊 Stage 1 trial 0、新 Stage 2 trial 2 完全一致：
seed=42、direct 5-to-5、grid=5、radius=0.1、filters=64、weight decay=0、
LR=0.0001、batch=64、epochs=40、patience=10、STDK epochs=350。

| 執行 | STDK best validation MSE | 最終 q50 standardized Val RMSE |
|---|---:|---:|
| 舊裝置 Stage 1 trial 0 | 0.390047714 | 0.621448080 |
| 新裝置 Stage 2 trial 2 | 0.468663568 | 0.715675121 |
| 本次新裝置重跑 | 0.468663568 | 0.716010470 |

本次耗時 1107.73 秒，約 18.46 分鐘。新裝置兩次最終 RMSE 差 0.00033535（約 0.047%），
STDK best validation MSE 完全相同。這一次同機檢查觀察到的波動不足以解釋跨裝置約 15% 的 RMSE 差距，
但不是同機結果永遠一致的保證，也未能將原因單獨歸於 GPU 或某個套件版本。

完整數值保存在 `metrics/shared_qconvlstm_block5to5_seed42_validation.json`。
原路徑為 `weather2k_exp/repro_check_seed42_run1/`，已移到本目錄。
SHA256 搬移核對紀錄見 [Stage 2 archive manifest](../STDK_QConvLSTM_Weather2K_q50_capacity_tuning_20260909/provenance/archive_manifest.json)。

## 當次重跑命令

下列 output-dir 是原始執行位置；記錄供追溯，不需為封存再次執行。

```bash
PYTHONUNBUFFERED=1 python -u weather2k_exp/2K_STDK_QConvLSTM.py \
  --seed 42 --forecast-mode block5to5 --prediction-mode direct \
  --validation-only --q50-only --stdk-epochs 350 \
  --grid-size 5 --neighbourhood-radius 0.1 --conv-filters 64 \
  --qconv-weight-decay 0 --qconv-lr 0.0001 \
  --qconv-batch-size 64 --qconv-epochs 40 --qconv-patience 10 \
  --output-dir weather2k_exp/repro_check_seed42_run1
```

## 整理後目錄

- `metrics/`：JSON 評估結果

本次整理只調整檔案位置，沒有刪除或重新計算結果。
