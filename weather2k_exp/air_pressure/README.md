# Air Pressure Experiments

這個資料夾保存 Weather2K 氣壓變數 (`air_pressure`) 的 DLinear+FRK 與 STDK 實驗程式、輸出表格、metrics JSON 和 loss 圖。

## 實驗設定

- 資料變數：`air_pressure`
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
| Scenario 1 時間外推 | Target_Time150 | DLinear+FRK | 69.814082 ± 2.651173 | 4881.034814 ± 373.314654 | 54.644330 ± 2.733566 | 0.253899 ± 0.070287 |
| Scenario 1 時間外推 | Target_Time150 | STDK | 35.018666 ± 5.739181 | 1259.245190 ± 441.926290 | 24.510091 ± 4.242273 | -266.874356 ± 119.010504 |
| Scenario 2 空間外推 | Target_Space100 | DLinear+FRK | 73.108890 ± 12.986239 | 5513.552158 ± 2113.821410 | 53.694949 ± 7.508796 | 0.187335 ± 0.034414 |
| Scenario 2 空間外推 | Target_Space100 | STDK | 37.228017 ± 11.201756 | 1511.404626 ± 812.348934 | 22.778145 ± 6.356095 | -177.977872 ± 111.649038 |
| Scenario 3 時空外推 | Target_ST100x150 | DLinear+FRK | 71.484609 ± 9.396577 | 5198.344933 ± 1460.384398 | 59.246223 ± 6.643926 | 0.221437 ± 0.077361 |
| Scenario 3 時空外推 | Target_ST100x150 | STDK | 39.655869 ± 11.575013 | 1706.568921 ± 960.143135 | 27.863321 ± 7.886325 | -392.696350 ± 297.968425 |

## 檔案說明

- `2K_DLinear_FRK_hybridloss.py`：氣壓版 DLinear+FRK 實驗程式。
- `2K_STDK_500train_100test.py`：氣壓版 STDK 實驗程式。
- `experiments_runner.py`：三個情境的批次執行與表格整理。
- `plot_dlinear_loss.py`：從 DLinear+FRK metrics JSON 畫 loss 圖。
- `results_*.csv` / `results_*.md`：每個 seed 與 mean±std 的結果表格。
- `dlinear_autofrk_frkloss_test_100to500_metrics_*.json`：DLinear+FRK 每個情境的完整 metrics 與 loss history。
- `2K_stdk_metrics_*.json`：STDK 每個情境的完整 metrics。
- `loss_*.png`：DLinear+FRK seed 41 的 loss 曲線圖。

## 執行方式

在 `weather2k_exp` 根目錄執行：

```bash
python experiments_runner.py
```

若要重畫 loss 圖，可以使用：

```bash
/home/jundian/installers/yes/envs/geospatial-neural-adapter/bin/python plot_dlinear_loss.py dlinear_autofrk_frkloss_test_100to500_metrics_time500_s100u400e100_alpha1p0_lam1p0.json --seed 41
```
