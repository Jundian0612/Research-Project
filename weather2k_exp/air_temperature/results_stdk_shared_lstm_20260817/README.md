# Weather2K 氣溫實驗：STDK + Shared LSTM（2026-08-17 整理）

> **方法來源警告：** 本資料夾中標記為 `STDK+SharedLSTM` 的時間外推與時空外推，是本專案自行建立的 Weather2K deterministic baseline，不是 Nag、Sun 與 Reich 論文中的 QLSTM 或 QConvLSTM 流程，也不能視為 paper reproduction。它使用跨站共享的單變量 LSTM、MSE loss 與 point forecast；沒有 quantile/pinball loss、QConvLSTM local spatial grids、q05/q50/q95、MPIW 或 coverage。此警告只針對 `STDK+SharedLSTM` 列，不影響同資料夾中的 SVGP 與 DLinear+FRK 結果。

## 實驗目的

本資料夾保存目前以 Weather2K 氣溫資料比較三個模型的輸出：

- SVGP
- STDK + Shared Univariate LSTM
- DLinear + differentiable FRK

共同設定為 seeds 41--45、500 個訓練站與 100 個最終測試站，正式指標均在還原至原始氣溫尺度後計算；R2 使用 flatten global R2。

## 目前完成狀態

| 情境 | 評估目標 | SVGP | STDK+SharedLSTM | DLinear+FRK | 狀態 |
|---|---|---:|---:|---:|---|
| 時間外推 | Target_Time150 | 5 seeds | 5 seeds | 5 seeds | 完成 |
| 空間外推 | Target_Space100 | 5 seeds | 無 | 無 | 未完成，不可做三模型比較 |
| 時空外推 | Target_ST100x150 | 5 seeds | 5 seeds | 5 seeds | 完成 |

## 正式結果（mean +/- std）

### 時間外推：Target_Time150

| 模型 | RMSE | MAE | R2 |
|---|---:|---:|---:|
| SVGP | 3.397356 +/- 0.108090 | 2.670727 +/- 0.059190 | 0.611587 +/- 0.022806 |
| DLinear+FRK | 4.100121 +/- 0.229395 | 3.119959 +/- 0.093619 | 0.432938 +/- 0.062753 |
| STDK+SharedLSTM | 5.086180 +/- 0.292234 | 4.080373 +/- 0.249410 | 0.128087 +/- 0.090278 |

目前排序為 SVGP、DLinear+FRK、STDK+SharedLSTM。

### 時空外推：Target_ST100x150

| 模型 | RMSE | MAE | R2 |
|---|---:|---:|---:|
| SVGP | 3.434645 +/- 0.138020 | 2.696188 +/- 0.107460 | 0.600822 +/- 0.031693 |
| DLinear+FRK | 4.063101 +/- 0.211935 | 3.161477 +/- 0.196029 | 0.441065 +/- 0.054758 |
| STDK+SharedLSTM | 5.231944 +/- 0.346062 | 4.190971 +/- 0.350014 | 0.072986 +/- 0.108130 |

目前排序同樣為 SVGP、DLinear+FRK、STDK+SharedLSTM。

### 空間外推：Target_Space100（暫存結果）

| 模型 | RMSE | MAE | R2 |
|---|---:|---:|---:|
| SVGP | 3.228834 +/- 0.038906 | 2.480098 +/- 0.026375 | 0.750156 +/- 0.009765 |

此情境缺少 STDK+SharedLSTM 與 DLinear+FRK，不能據此判定三模型排名。

## STDK + Shared LSTM 流程

目前 STDK 採兩階段流程：

1. STDK 根據空間座標與時間基底建立 Train700 的站點歷史。
2. 一個跨站共享的單變量 LSTM 以歷史序列遞迴預測 Val150 與 Test150。

LSTM 設定為 lookback 5、hidden size 50、1 層、Adam、learning rate 0.001、最多 120 epochs、early-stopping patience 15。STDK 與 LSTM 已改成分階段釋放中間資料，STDK 預測預設以 50 個時間點為一批。

時間外推的 STDK+SharedLSTM 結果呈現長距離遞迴誤差累積：Train700 RMSE 為 4.250597、Val150 RMSE 為 4.405250，Target_Time150 RMSE 增至 5.086180。純時間外推中所有 500 站其實都有真實 Train700，因此後續可測試直接將真實歷史交給 LSTM，避免先對已觀測站做 STDK 重建而引入額外誤差。

## DLinear+FRK 與 SVGP 設定摘要

- DLinear+FRK 使用已調整的參數檔，正式輸出中的 hybrid loss 權重為 alpha=2.0、lambda=1.5。
- SVGP 使用 Matérn-periodic kernel、1024 inducing points、learning rate 0.001、500 epochs。

## 檔案結構

- `tables/`：各情境的 CSV 與 Markdown 彙整表，包含逐 seed 與 mean/std。
- `json/`：各模型原始執行輸出與逐 seed 詳細資訊。

同一情境中的 `2K_best_dlinear...json` 與 `dlinear_autofrk...json` 是 DLinear+FRK 執行流程產生的不同層級輸出，均予保留以便追溯。

## 注意事項

- 此資料夾是截至 2026-08-17 的結果快照，不包含尚未完成的空間情境結果。
- 根目錄保留模型程式、runner、tuning 程式及目前最佳參數 JSON；本次只移動實驗輸出。
- 之後重跑實驗時，runner 會在根目錄重新產生同名 JSON、CSV 與 Markdown，不會覆寫本資料夾中的快照。
