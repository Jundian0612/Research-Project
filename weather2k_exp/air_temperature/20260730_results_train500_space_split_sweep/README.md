# Weather2K 氣溫三模型實驗（2026-07-30）

本資料夾保存固定相同 train500/test100、時間切分 700/150/150、seeds 41–45 的完整比較結果。

## 實驗設定

- 目標變數：`air_temperature`
- 每個 seed 固定相同的 500 個訓練站與 100 個測試站
- DLinear+FRK space split：`100/400/100`、`200/300/100`、`300/200/100`、`400/100/100`
- STDK：每個情境只使用合併後的 train500 執行一次
- SVGP：GitHub 固定設定結果（RBF、1024 inducing points、LR 0.01、500 epochs）
- 情境：時間外推、空間外推、時空外推

## 目錄

- `tables/`：三個情境的 CSV 與 Markdown 彙總表
- `svgp_json/`：3 個 SVGP 完整結果
- `stdk_json/`：3 個 STDK 完整結果
- `dlinear_best_json/`：12 個 DLinear 最佳模型／重跑紀錄
- `dlinear_frk_json/`：12 個 DLinear+FRK 指標結果
- `params/`：實驗使用的 DLinear 參數備份；根目錄原檔仍保留供程式執行

## Mean ± std 重點

| 情境 | 最佳 RMSE 模型 | RMSE |
|---|---|---:|
| 時間外推 | DLinear+FRK `400/100/100` | 3.948688 ± 0.209956 |
| 空間外推 | STDK train500/test100 | 4.368006 ± 0.085527 |
| 時空外推 | DLinear+FRK `300/200/100` | 4.883509 ± 0.164614 |

SVGP hyperparameter tuning 的後續產物位於相鄰的 `svgp_tuning_current/`，不屬於本次固定 GitHub 設定結果。
