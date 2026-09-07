# Weather2K 四模型 seed 41 比較

## 執行狀態

本次 runner 已完成，時間、空間與時空三個情境均有下列四個模型的 seed 41 結果：

- SVGP
- 純 STDK
- DLinear + differentiable FRK
- STDK + SharedQConvLSTM

這是 **single-seed pilot**，不是五-seed正式結論。表格中的 `mean +/- std` 對 SVGP、STDK 與 DLinear+FRK 等同 seed 41 本身，`std=0` 只是因為只有一個 seed，不能解讀為模型穩定。SharedQConvLSTM 的 single-seed summary 則直接顯示單一數值。

## 共同實驗標準

| 項目 | 設定 |
|---|---|
| Weather2K 變數 | air temperature |
| Seeds | 只有 41 |
| 抽樣測站 | 600 |
| 受監督訓練測站 | train500 |
| 完全 held-out 測站 | 100 |
| 時間切分 | Train700 / Val150 / Test150 |
| 時間外推 target | train500 × Test150 |
| 空間外推 target | held-out100 × Fixed850 |
| 時空外推 target | held-out100 × Test150 |
| DLinear+FRK | alpha=2.0, lambda=1.5 |

held-out100 的真值不參與模型訓練或 validation，只在最終評估時計分。

## Seed 41 結果

### 時間外推：Target\_Time150

| 模型 | RMSE | MSE | MAE | R2 |
|---|---:|---:|---:|---:|
| **SVGP** | **3.320592** | **11.026330** | **2.603338** | **0.636323** |
| DLinear+FRK | 3.791848 | 14.378115 | 2.937695 | 0.525773 |
| STDK | 4.313021 | 18.602154 | 3.508868 | 0.386453 |
| STDK+SharedQConvLSTM | 6.035994 | 36.433228 | 4.768288 | -0.201662 |

### 空間外推：Target\_Space100

| 模型 | RMSE | MSE | MAE | R2 |
|---|---:|---:|---:|---:|
| **DLinear+FRK** | **3.095808** | **9.584029** | **2.241228** | **0.769712** |
| SVGP | 3.251627 | 10.573078 | 2.510879 | 0.745947 |
| STDK | 4.318465 | 18.649139 | 3.430412 | 0.551894 |
| STDK+SharedQConvLSTM | 4.711214 | 22.195538 | 3.467410 | 0.466680 |

### 時空外推：Target\_ST100x150

| 模型 | RMSE | MSE | MAE | R2 |
|---|---:|---:|---:|---:|
| **SVGP** | **3.255117** | **10.595787** | **2.551797** | **0.630973** |
| DLinear+FRK | 3.735412 | 13.953302 | 2.920093 | 0.514038 |
| STDK | 4.307308 | 18.552900 | 3.526602 | 0.353845 |
| STDK+SharedQConvLSTM | 5.859388 | 34.332424 | 4.607038 | -0.195720 |

## STDK+SharedQConvLSTM 補充結果

SharedQConvLSTM 維持 direct `5 -> 150`。它只訓練一次，再由同一 fitted model 評估三種情境。空間 Fixed850 以150步為 block；第一個 block 的輸入使用 STDK-only 相對時間座標 `-5..-1`，不讀取 held-out responses。

| 情境 | MPIW 90% | Coverage 90% |
|---|---:|---:|
| 時間外推 | 9.184864 | 52.25% |
| 空間外推 | 8.975844 | 72.14% |
| 時空外推 | 8.947696 | 52.61% |

三個情境的 coverage 都低於目標90%，尤其時間與時空外推約只有52%。因此目前問題不只是 point forecast RMSE，quantile interval 也尚未校準。

## 執行時間

| 模型 | 約略時間 |
|---|---:|
| DLinear+FRK | 每個情境約3.1–3.2分鐘 |
| 純 STDK | 每個情境約3.4分鐘 |
| SVGP | 每個情境約22.2分鐘 |
| STDK+SharedQConvLSTM | 一次訓練及三情境評估約3小時24分鐘 |

SharedQConvLSTM 沒有寫 checkpoint。

## 結果解讀

- SVGP 在時間與時空外推最佳。
- DLinear+FRK 在空間外推最佳，RMSE 比 SVGP 約低4.8%。
- SharedQConvLSTM 在時間與時空外推的 R2 為負，尚未優於簡單平均基準。
- SharedQConvLSTM 的 RMSE 相較各情境最佳模型約高52%–82%，目前不適合直接照相同設定擴展到 seeds 42–45。

## 建議下一步

1. 先凍結 Test150 與 held-out100，不用本表的 final-test數值選超參數。
2. 只用 Train700/Val150 比較 SharedQConvLSTM 的 lookback、forecast block、learning rate、batch size與 grid設定。
3. 優先比較 direct `5 -> 150` 與較合理的 `48 -> 12`、`168 -> 24`；若使用 block forecast，需維持同一套無 held-out truth洩漏規則。
4. 使用 Val150 校準 q05/q95 interval，使90% coverage接近目標，再鎖定設定。
5. 設定鎖定後才執行 seeds 42–45，計算正式五-seed mean與 standard deviation。
6. DLinear+FRK 的空間結果值得優先補成五 seeds，以確認 seed 41 的優勢是否穩定。

## 檔案結構

```text
four_models_seed41_20260831/
├── README.md
├── tables/      # 三情境的四模型 CSV 與 Markdown 比較表
├── json/        # 各模型完整 metrics、設定、split 與 provenance
└── forecasts/   # SharedQConvLSTM 三情境逐點預測
```

詳細比較請直接查看：

- `tables/results_time_extrap_fixed500.md`
- `tables/results_space_extrap_fixed850.md`
- `tables/results_spatiotemp_100x150.md`
- `json/shared_qconvlstm_seed41.json`

