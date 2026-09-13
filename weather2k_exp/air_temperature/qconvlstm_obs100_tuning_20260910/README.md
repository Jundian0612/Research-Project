# STDK + QConvLSTM obs100 對齊版調參

執行日期：2026-09-10。狀態：Stage 1 與 Stage 2 全部完成，僅使用 validation 選參。

本輪將標準化改成與 SVGP、DLinear+FRK、獨立 STDK 一致：每個 seed 使用相同的
obs100 × Train700 計算一組全域 mean/std；模型訓練仍使用 train500 × Train700，
validation 使用 train500 × Val150。Test150 與 held-out100 真值沒有參與選參。

## 搜尋結果

- 26 runs：Stage 1 9組 × 2 seeds；Stage 2 4組 × 2 seeds。
- 原始 reports 的 elapsed time 累計約 9.77 小時。
- Stage 1 最佳：grid=5、radius=0.1。
- Stage 2 最佳：filters=32、weight decay=0。
- Seed 41 / 42 standardized q50 Val RMSE：0.646992 / 0.651852。
- 平均：0.649422 ± 0.002430（ddof=0）。

| Filters | Weight decay | Mean Val RMSE |
|---:|---:|---:|
| **32** | **0** | **0.649422** |
| 32 | 1e-5 | 0.649578 |
| 64 | 0 | 0.655724 |
| 64 | 1e-5 | 0.654832 |

前兩組只差 0.000156，weight decay 的排名優勢很弱；依預先採用的最低平均 RMSE 規則選0。
這是 q50 validation-only 結果，不是正式 Test150 或五-seed結果。

## 固定流程

- Spatial-adapter STDK → local grids → direct QConvLSTM。
- block5to5、作者公開 notebook 架構的 `github_3block` PyTorch 重現版。
- LR=0.0001、batch=64、max epochs=40、patience=10。
- q50 訓練用 pinball loss；epoch checkpoint 用 Val MSE；trial 用 seeds41/42 平均 standardized Val RMSE。
- held-out100 只供後續正式評分。

## 資料夾

```text
qconvlstm_obs100_tuning_20260910/
├── README.md
├── trials/interface/
├── trials/capacity/
├── summary/
├── logs/run.log
└── provenance/QCONVLSTM_TUNING_README.md
```

`summary/qconvlstm_tuning_best_params.json` 是本輪最終參數，已複製到相鄰正式實驗資料夾的
`locked_params.json`。分類後不要再把本資料夾當作 tuner output-dir；runner 不會自動搜尋子資料夾。
若需新增調參，使用新的 output-dir。

舊 train500-normalization 結果仍保留作方法敏感度與研究歷程，不能混入本輪正式五-seed統計。
