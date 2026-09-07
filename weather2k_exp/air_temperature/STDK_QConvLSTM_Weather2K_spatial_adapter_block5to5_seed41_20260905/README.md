# Weather2K Spatial-adapter STDK + Shared QConvLSTM

日期：2026-09-05

## 狀態

這是 `air_temperature` 的正式 seed 41 single-seed 結果。模型以與純 STDK baseline 相同的 Spatial-adapter STDK 建立 local quantile grids，再使用作者 released notebook 對應的 shared three-block QConvLSTM，以 `5 -> 5` blocks 延伸完成 Weather2K 的 150-step 評估。

這不是五-seed結論。結果沒有用於重新選參；hyperparameters 已事先由 Train700/chronological Val150 的 validation-only tuning 決定。Test150 與 held-out100 真值只用於最後評分，沒有進入訓練或選參。沒有寫入 model checkpoint。

## 實驗設定

- Seed：41
- Weather2K variable：`air_temperature`
- Stations：固定抽樣 600，train500 / strict held-out100
- 時間切分：Train700 / Val150 / Test150
- STDK backend：`spatial_adapter`
- QConvLSTM input：過去 5 張 `8 x 8` STDK local grids
- QConvLSTM output：直接預測未來 5 點
- Forecast mode：`block5to5`
- QConvLSTM profile：`github_3block`
- Neighbourhood radius：0.2
- Filters：64
- Learning rate：0.0001
- Batch size：64
- Maximum epochs：40
- Patience：10
- 總執行時間：約 6.44 小時

正式參數來自相鄰的 tuning 封存資料夾：

```text
../STDK_QConvLSTM_Weather2K_spatial_adapter_tuning_20260904/
```

## 三情境結果

| 情境 | RMSE | MSE | MAE | R2 | MPIW 90% | Coverage 90% |
|---|---:|---:|---:|---:|---:|---:|
| Target_Time150 | 4.387748 | 19.252337 | 3.418592 | 0.365008 | 12.354848 | 0.861453 |
| Target_Space100 | 4.418732 | 19.525188 | 3.444825 | 0.530844 | 12.306922 | 0.866094 |
| Target_ST100x150 | 4.339629 | 18.832382 | 3.466058 | 0.344111 | 12.669021 | 0.870133 |

## 與相同 seed 41 純 STDK 比較

兩者使用相同 sampled600、held-out100、時間切分及 Spatial-adapter STDK 設定。

| 情境 | 純 STDK RMSE | STDK+QConvLSTM RMSE | QConvLSTM 相對差異 |
|---|---:|---:|---:|
| 時間外推 | 4.313021 | 4.387748 | +1.73% |
| 空間外推 | 4.318465 | 4.418732 | +2.32% |
| 時空外推 | 4.307308 | 4.339629 | +0.75% |

目前 QConvLSTM 三個情境都略輸純 STDK，但差距已縮小到約 1% 至 2%。同次正式執行中，Spatial-adapter STDK q50 的 standardized Val RMSE 為 0.660127，QConvLSTM q50 為 0.660286，顯示 QConvLSTM 在 validation 階段也大致只追平 STDK，沒有取得額外增益。

這個結果不代表程式又使用了舊 STDK backend。主要限制是目前 QConvLSTM 以過去 STDK grids 重新預測未來值，而不是保留 STDK 的目標時間直接預測再學 residual，因此組合後不保證優於 STDK。

## 空間外推起始邊界

Weather2K time 0 前沒有合法的五張歷史 grids，因此不再使用虛構的 `-5..-1`：

- time 0 至 4 使用 direct Spatial-adapter STDK warm-up。
- 從 time 5 起，以 STDK grids 0 至 4 交給 QConvLSTM 預測 5 至 9。
- 之後持續使用合法的 `5 -> 5` blocks 到 time 849。

此處沒有讀取 held-out100 真值。

## 檔案

- `shared_qconvlstm_block5to5_seed41.json`：完整設定、split、validation 與三情境指標。
- `shared_qconvlstm_block5to5_seed41_time_forecasts.csv`：train500 的 Test150，共 75,000 筆預測。
- `shared_qconvlstm_block5to5_seed41_space_forecasts.csv`：held-out100 的 Fixed850，共 85,000 筆預測。
- `shared_qconvlstm_block5to5_seed41_forecasts.csv`：held-out100 的 Test150，共 15,000 筆時空預測。

## 下一步

若目標是完成 paper-aligned Weather2K 評估，應固定目前參數補跑 seeds 42 至 45，再與本資料夾的 seed 41 合併計算五-seed mean ± std，不可依本次 Test150 結果繼續調參。若目標是讓組合模型穩定保留純 STDK 能力，應另建 residual/skip-connection ablation，不能覆蓋本結果或稱為作者原始流程。
