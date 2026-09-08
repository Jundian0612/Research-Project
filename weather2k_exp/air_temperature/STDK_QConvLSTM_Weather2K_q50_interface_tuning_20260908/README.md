# Weather2K STDK+QConvLSTM q50 interface tuning

日期：2026-09-06 至 2026-09-08

## 狀態

Stage 1 interface tuning 已完成。這是 `air_temperature` 的 validation-only
結果，不是正式 Test150 結果，也不是五-seed test mean ± std。

所有 trials 僅使用 train500 stations 的 Train700 訓練，並以相同 train500
stations 的 chronological Val150 選參。Test150 指標未計算，held-out100
真值未用於訓練或選參。沒有寫入 model checkpoint。

## 固定設定

- Tuning seeds：41、42
- Station split：train500 / strict held-out100
- Time split：Train700 / Val150 / Test150
- STDK backend：Spatial-adapter
- Prediction mode：direct
- Forecast mode：5-to-5 blocks
- Tuning quantile：q50 only
- QConvLSTM profile：作者 released `github_3block`
- Learning rate：0.0001
- Batch size：64
- Maximum epochs：40
- Patience：10
- Filters：64
- Weight decay：0

## 搜尋範圍

- Grid size：5、8、11
- Neighbourhood radius：0.1、0.2、0.3
- 總計：9 configurations × 2 seeds = 18 runs
- 18 runs 累計實際執行時間：約 33.82 小時

## 結果

| Trial | Grid size | Radius | Mean q50 Val RMSE | Std |
|---:|---:|---:|---:|---:|
| 0 | **5** | **0.1** | **0.632992** | 0.011544 |
| 1 | 8 | 0.1 | 0.633275 | 0.010949 |
| 2 | 11 | 0.1 | 0.638386 | 0.017623 |
| 3 | 5 | 0.2 | 0.649627 | 0.011096 |
| 4 | 8 | 0.2 | 0.656352 | 0.013701 |
| 5 | 11 | 0.2 | 0.651165 | 0.018895 |
| 6 | 5 | 0.3 | 0.659193 | 0.005299 |
| 7 | 8 | 0.3 | 0.653686 | 0.000138 |
| 8 | 11 | 0.3 | 0.654239 | 0.011052 |

最佳設定為 `grid_size=5`、`neighbourhood_radius=0.1`。相較原本使用的
`grid_size=8`、`radius=0.2`（trial 4），mean q50 Val RMSE 從 0.656352
降至 0.632992，validation 改善約 3.56%。Trial 0 與 trial 1 的差距很小，
因此結果主要支持較小的 radius；grid size 5 相對 grid size 8 的優勢仍弱。

## 檔案

```text
STDK_QConvLSTM_Weather2K_q50_interface_tuning_20260908/
├── README.md
├── summary/
│   ├── qconvlstm_interface_best_params.json
│   └── qconvlstm_interface_summary.csv
└── trials/
    ├── qconvlstm_interface_trial0000_params.json
    ├── qconvlstm_interface_trial0000_seed41_validation.json
    ├── qconvlstm_interface_trial0000_seed42_validation.json
    ├── qconvlstm_interface_trial0000_summary.json
    └── ... trial0008 ...
```

根目錄的 `weather2k_exp/qconvlstm_interface_best_params.json` 是供 Stage 2
續跑讀取的 operational copy，不是重複的歷史結果。Stage 2 完成後才會更新
正式參數檔 `weather2k_exp/2K_best_stdk_qconvlstm_params_500to100.json`。

## 下一步

以 Stage 1 最佳 interface 固定 `grid_size=5`、`radius=0.1`，再執行 Stage 2
搜尋 filters 32/64 與 weight decay 0/1e-5：

```bash
/home/jundian/installers/yes/envs/geospatial-neural-adapter/bin/python \
  weather2k_exp/tune_stdk_qconvlstm_hyperparams.py \
  --stage capacity \
  --seeds 41 42
```

Stage 2 完成以前不應執行正式五-seed test。
