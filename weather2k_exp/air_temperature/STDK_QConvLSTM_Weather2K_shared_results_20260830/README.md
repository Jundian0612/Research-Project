# STDK + Shared QConvLSTM：歷史 seed 41 結果

> **狀態：歷史 pilot，不是目前正式的 5→5 Weather2K 結果。**

本資料夾保存最早將 STDK+QConvLSTM 改為 Weather2K shared model 時產生的
seed 41 結果。保留它是為了追蹤方法演變與核對舊數值，不應與目前正式
`block5to5` 結果混合計算。

## 實驗設定

| 項目 | 設定 |
|---|---:|
| 變數 | air temperature |
| Seed | 41 |
| Weather2K 時間範圍 | 最後 1,000 點 |
| 空間抽樣 | 600 stations |
| 訓練／保留 stations | train500 / held-out100 |
| 時間切分 | Train700 / Val150 / Test150 |
| QConvLSTM | 三個 shared quantile models：q05、q50、q95 |
| 輸入／輸出 | **直接 5→150** |
| Local grid | 8×8，radius 0.2 |
| STDK | 350 epochs，batch 512 |
| QConvLSTM | 25 epochs，batch 64 |
| Checkpoint | 未寫入 |
| Held-out100 真值 | 僅供最後評估，不參與訓練 |

這個歷史版本的 QConvLSTM validation 是訓練 windows 內部切分；每個
quantile 實際使用 259,350 個 training windows。它不是目前正式版本所用的
完整 chronological Val150 block evaluation。

## 結果

這次輸出只記錄 held-out100 在 Test150 的時空外推，共 15,000 個預測：

| 指標 | Seed 41 |
|---|---:|
| RMSE | 5.796536 |
| MSE | 33.599834 |
| MAE | 4.508700 |
| MPIW 90 | 7.234217 |
| Coverage 90 | 47.23% |
| 執行時間 | 22,489 秒，約 6 小時 15 分 |

Coverage 明顯低於名目 90%，代表此歷史版本的 prediction interval 過窄。

## 檔案

- `shared_qconvlstm_seed41.json`
  - 完整設定、抽樣 station indices、train/held-out split、validation pinball、
    最終指標與耗時。
- `shared_qconvlstm_seed41_forecasts.csv`
  - held-out100 × Test150 的逐點預測。
  - 欄位為 `seed, heldout_local, lead, truth, q05, q50, q95`。

## 與目前正式版本的差異

| 項目 | 本資料夾歷史版本 | 目前正式版本 |
|---|---|---|
| Forecast horizon | 直接 5→150 | 5→5 contiguous blocks |
| Validation | training windows 內部切分 | chronological Val150 |
| 評估輸出 | 只有時空外推 | 時間、空間、時空三種情境 |
| 正式用途 | 否，僅供歷史比較 | 是 |

目前正式結果位於相鄰的
`STDK_QConvLSTM_Weather2K_results_20260901/`，其中 README 另有 seed 41 A/B 比較。
兩個資料夾的結果不可直接合併成 multi-seed mean ± std。
