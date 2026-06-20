# Relative Humidity Experiments

這個資料夾保存 Weather2K `relative_humidity` 欄位的 DLinear+FRK 與 STDK 實驗程式、輸出表格、metrics JSON 和 loss 圖。

## 實驗設定

- 資料變數：`relative_humidity`
- 時間切分：最後 1000 個時間點，`700 / 150 / 150`
- 空間切分：`100 / 400 / 100`
- DLinear+FRK loss：`total_loss = 1.0 * loss_obs + 1.0 * loss_unobs`
- DLinear+FRK epoch：350，patience：30
- STDK epoch：350，patience：30
- seeds：41, 42, 43, 44, 45

## 三個情境

| 情境 | Target | 說明 |
| --- | --- | --- |
| Scenario 1 | `Target_Time150` | 固定 500 個空間點，做時間外推 |
| Scenario 2 | `Target_Space100` | 固定 850 個時間點，做空間外推 |
| Scenario 3 | `Target_ST100x150` | 同時做時間與空間外推 |

## Mean ± Std 結果

| Scenario | Target | Model | RMSE | MSE | MAE | R2 |
| --- | --- | --- | --- | --- | --- | --- |
| Scenario 1 時間外推 | Target_Time150 | DLinear+FRK | 4.583476 ± 0.107513 | 21.019811 ± 0.994539 | 3.483393 ± 0.089556 | 0.301264 ± 0.029809 |
| Scenario 1 時間外推 | Target_Time150 | STDK | 4.795577 ± 0.307557 | 23.092152 ± 3.000536 | 3.796201 ± 0.266459 | -0.719267 ± 0.200792 |
| Scenario 2 空間外推 | Target_Space100 | DLinear+FRK | 4.566119 ± 0.356918 | 20.976836 ± 3.353607 | 3.497670 ± 0.182452 | 0.499414 ± 0.040966 |
| Scenario 2 空間外推 | Target_Space100 | STDK | 4.764922 ± 0.287510 | 22.787146 ± 2.750736 | 3.763019 ± 0.228146 | 0.136045 ± 0.111037 |
| Scenario 3 時空外推 | Target_ST100x150 | DLinear+FRK | 4.731012 ± 0.395092 | 22.538576 ± 3.838604 | 3.596632 ± 0.230498 | 0.248625 ± 0.067074 |
| Scenario 3 時空外推 | Target_ST100x150 | STDK | 4.839967 ± 0.545047 | 23.722356 ± 5.437703 | 3.870999 ± 0.460068 | -0.789362 ± 0.422237 |

## 檔案說明

- `2K_DLinear_FRK_hybridloss.py`：`relative_humidity` 版 DLinear+FRK 實驗程式。
- `2K_STDK_500train_100test.py`：`relative_humidity` 版 STDK 實驗程式。
- `experiments_runner.py`：三個情境的批次執行與表格整理。
- `plot_dlinear_loss.py`：從 DLinear+FRK metrics JSON 畫 loss 圖。
- `results_*.csv` / `results_*.md`：每個 seed 與 mean±std 的結果表格。
- `dlinear_autofrk_frkloss_test_100to500_metrics_*.json`：DLinear+FRK 每個情境的完整 metrics 與 loss history。
- `2K_stdk_metrics_*.json`：STDK 每個情境的完整 metrics。
- `loss_relative_humidity_*.png`：DLinear+FRK seed 41 的 loss 曲線圖。

## 執行方式

在 `weather2k_exp` 根目錄執行：

```bash
python experiments_runner.py
```

若要重畫 loss 圖，可以使用：

```bash
/home/jundian/installers/yes/envs/geospatial-neural-adapter/bin/python plot_dlinear_loss.py dlinear_autofrk_frkloss_test_100to500_metrics_time500_s100u400e100_alpha1p0_lam1p0.json --seed 41
```

## 備註

先前 EDA 顯示 `relative_humidity` 欄位的數值範圍有負值，和一般相對濕度單位直覺不同；後續解讀時建議再確認 Weather2K 原始欄位順序或單位定義。
