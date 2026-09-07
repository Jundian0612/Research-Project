# Weather2K Spatial-adapter STDK + QConvLSTM tuning

日期：2026-09-03 至 2026-09-04

## 目的與狀態

這個資料夾保存 `air_temperature` 的正式第一階段 hyperparameter tuning 結果。模型使用 Spatial-adapter STDK 建立 local quantile grids，再以作者 released notebook 對應的 shared three-block QConvLSTM 執行 `5 -> 5` 預測。

本次 tuning 已完成，共執行 seed 41 的四個 trials。選參只使用 train500 stations 的 chronological Val150；沒有計算 Test150 指標，也沒有使用 held-out100 真值，因此不是 test-set tuning。所有 trial 均未寫入 model checkpoint。

## 固定設定

- Weather2K variable：`air_temperature`
- 空間切分：train500 / held-out100
- 時間切分：Train700 / Val150 / Test150
- STDK backend：`spatial_adapter`
- QConvLSTM input：過去 5 張 `8 x 8` local grids
- QConvLSTM output：直接預測未來 5 點
- neighbourhood radius：`0.2`
- QConvLSTM profile：`github_3block`
- filters：`64`
- selection metric：standardized Val150 q50 RMSE

## 結果

| Trial | Learning rate | Batch size | Max epochs | Patience | Val q50 RMSE | Mean quantile pinball | Elapsed |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.001 | 64 | 25 | 5 | 0.666094 | 0.131968 | 3.44 h |
| 1 | 0.0003 | 64 | 40 | 10 | 0.661208 | 0.130774 | 6.30 h |
| 2 | 0.0001 | 64 | 40 | 10 | **0.658933** | **0.128473** | 6.31 h |
| 3 | 0.0003 | 32 | 40 | 10 | 0.671510 | 0.131919 | 6.67 h |

最佳為 trial 2。相較 trial 0，Val q50 RMSE 約降低 1.08%，mean quantile pinball 約降低 2.65%。四個 trials 累計約 22.71 小時。

正式參數：

```text
QCONV_LR=0.0001
QCONV_BATCH_SIZE=64
QCONV_EPOCHS=40
QCONV_PATIENCE=10
CONV_FILTERS=64
```

## 檔案

- `qconvlstm_tuning_summary.csv`：四個 trials 的比較表。
- `qconvlstm_tuning_best_params.json`：最佳 trial 與正式參數的封存副本。
- `qconvlstm_tune_trialXXXX_params.json`：各 trial 輸入參數。
- `qconvlstm_tune_trialXXXX_seed41_validation.json`：各 trial 完整 validation report。
- `qconvlstm_tune_trialXXXX_summary.json`：各 trial 摘要。

模型正式執行仍需讀取的 `2K_best_stdk_qconvlstm_params_500to100.json` 特意保留在 `weather2k_exp/` 根目錄；其內容與本資料夾的最佳參數副本一致。

## 解讀限制

這些數值只代表 seed 41 的 Val150 選參表現，不能視為三情境正式結果，也不能與舊 STDK backend 的 QConvLSTM Test 結果直接比較。下一步應固定 trial 2，先完成新版 seed 41 的時間、空間及時空外推，再執行 seeds 41 至 45；不能依 Test150 結果繼續調參。
