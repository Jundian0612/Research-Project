# Weather2K Spatial-adapter STDK + Residual QConvLSTM

日期：2026-09-06

## 狀態與目的

這是 `air_temperature` 的正式 seed 41 residual ablation。它保留 paper-aligned direct 版本的資料切分、Spatial-adapter STDK、local grids、shared three-block QConvLSTM 與 `5 -> 5` blocks，但將預測定義改為：

```text
最終 quantile 預測
= 目標座標與目標時間的 direct Spatial-adapter STDK q50 預測
+ QConvLSTM 從過去五張 STDK grids 學得的 residual quantile
```

Residual 設計不是作者原始公開流程，因此應標示為 `Spatial-adapter STDK + Residual QConvLSTM`，不能當作 exact paper reproduction。這是 single-seed pilot，不是五-seed正式結論。

## 資料與防止洩漏

- Seed：41
- Weather2K variable：`air_temperature`
- Stations：固定抽樣 600，train500 / strict held-out100
- 時間切分：Train700 / chronological Val150 / Test150
- Test150 與 held-out100 真值只用於最終評分
- held-out100 真值沒有進入 STDK、QConvLSTM、early stopping 或選參
- 沒有寫入 model checkpoint

本次 sampled600、train500 與 held-out100 均已確認和純 STDK、direct QConvLSTM seed 41 相同。

## 模型與參數

- STDK backend：`spatial_adapter`
- QConvLSTM input：過去 5 張 `8 x 8` STDK local grids
- QConvLSTM output：未來 5 點的 residual quantiles
- Forecast mode：`block5to5`
- QConvLSTM profile：`github_3block`
- Neighbourhood radius：0.2
- Filters：64
- Learning rate：0.0001
- Batch size：64
- Maximum epochs：40
- Patience：10
- 總執行時間：約 7.19 小時

為了和 direct 版本作固定設定比較，本次沿用先前 validation-only tuning 選出的 Trial 2 optimization parameters，沒有使用本次 Test150 重新調參。

q50 residual head 採 zero initialization，因此 epoch 0 的最終中位數預測等同 direct STDK。q50 checkpoint 以 Val150 MSE 選擇，並包含 epoch 0 作為可退回的候選；這只保證 validation 可以拒絕無效修正，不保證 Test150 一定優於 STDK。

## 三情境結果

| 情境 | RMSE | MSE | MAE | R2 | MPIW 90% | Coverage 90% |
|---|---:|---:|---:|---:|---:|---:|
| Target_Time150 | 4.210097 | 17.724920 | 3.362721 | 0.415386 | 11.965261 | 0.832667 |
| Target_Space100 | 4.307046 | 18.550648 | 3.353528 | 0.554260 | 11.801197 | 0.859459 |
| Target_ST100x150 | 4.214765 | 17.764242 | 3.375084 | 0.381312 | 12.098861 | 0.823800 |

## Validation 結果

| Quantile | Standardized RMSE | Pinball loss | Checkpoint selection |
|---|---:|---:|---|
| q50 | 0.642921 | 0.255542 | Val MSE，包含 zero-residual epoch 0 |
| q05 | 0.922274 | 0.061940 | Val pinball |
| q95 | 1.157412 | 0.068214 | Val pinball |

同次 fitted Spatial-adapter STDK q50 的 standardized Val RMSE 為 0.660127；direct QConvLSTM 為 0.660286。Residual q50 在 Val150 也有改善，不只是 Test150 指標較低。

## 與 direct QConvLSTM 比較

| 情境 | Direct RMSE | Residual RMSE | Residual 相對改善 |
|---|---:|---:|---:|
| 時間外推 | 4.387748 | **4.210097** | 4.05% |
| 空間外推 | 4.418732 | **4.307046** | 2.53% |
| 時空外推 | 4.339629 | **4.214765** | 2.88% |

## 與純 STDK 比較

| 情境 | 純 STDK RMSE | Residual RMSE | Residual 相對改善 |
|---|---:|---:|---:|
| 時間外推 | 4.313021 | **4.210097** | 2.39% |
| 空間外推 | 4.318465 | **4.307046** | 0.26% |
| 時空外推 | 4.307308 | **4.214765** | 2.15% |

Residual seed 41 的三個 point RMSE 都優於 direct QConvLSTM 與純 STDK，但空間外推只改善 0.26%，必須用更多 seeds 判斷是否穩定。

## 區間預測限制

Residual 的 point forecast 較好，但 90% coverage 只有 82.38% 至 85.95%，低於理想的 90%，也低於 direct 版本的 86.15% 至 87.01%。因此目前只能判定 residual 中位數預測較好，不能宣稱 uncertainty estimation 全面較好。若要校準區間，只能使用 Train700/Val150，不能根據本次 Test coverage 選參。

## 空間外推起始邊界

- time 0 至 4 使用 direct Spatial-adapter STDK warm-up。
- 從 time 5 起，以合法的 STDK grids 0 至 4 預測 5 至 9。
- 之後持續 `5 -> 5` 到 time 849。
- 不使用不存在的 `-5..-1` 時間，也不讀取 held-out100 真值。

## 檔案

- `shared_residual_qconvlstm_block5to5_seed41.json`：完整設定、split、validation 與三情境指標。
- `shared_residual_qconvlstm_block5to5_seed41_time_forecasts.csv`：train500 的 Test150，共 75,000 筆。
- `shared_residual_qconvlstm_block5to5_seed41_space_forecasts.csv`：held-out100 的 Fixed850，共 85,000 筆。
- `shared_residual_qconvlstm_block5to5_seed41_forecasts.csv`：held-out100 的 Test150，共 15,000 筆。

## 下一步

若把 residual 視為候選正式模型，應固定目前流程與參數補跑 seeds 42 至 45，再合併本資料夾的 seed 41 計算五-seed mean ± std。不可依本次 seed 41 Test 指標繼續修改或挑選超參數。若研究報告需要正式比較 direct 與 residual，direct 版本也必須完成相同五個 seeds。
