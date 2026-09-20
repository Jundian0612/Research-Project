# Tuned SVGP results before global-R² unification（2026-08-01）

本資料夾保存 SVGP hyperparameter tuning 後、R² 計算方式統一前的三個正式情境結果。

## 模型設定

- Target：`air_temperature`
- Kernel：`matern_periodic`
- Learning rate：`0.001`
- Inducing points：`1024`
- Variational jitter：`0.01`
- Max epochs：`500`
- Patience：`30`
- Seeds：`41–45`
- 空間切分：固定相同 train500/test100
- 時間切分：Train700／Val150／Test150

## 結果摘要

| 情境 | RMSE（mean ± std） | MAE（mean ± std） |
|---|---:|---:|
| 時間外推 | 3.397356 ± 0.108090 | 2.670727 ± 0.059190 |
| 空間外推 | 3.228834 ± 0.038906 | 2.480098 ± 0.026375 |
| 時空外推 | 3.434645 ± 0.138020 | 2.696188 ± 0.107460 |

## 重要限制

這一版 SVGP 的 R² 是將每個測站的 R² 做 uniform average，尚未使用後來統一的 flatten global R²。因此：

- RMSE、MSE、MAE仍可作為有效結果保存。
- 此資料夾內的 R² 不應與 DLinear+FRK 的 global R²直接比較。
- 下一次三模型正式重跑會全部使用 flatten global R²，並產生新的主結果。

## 目錄

- `tables/`：三個情境的 CSV 與 Markdown 表格
- `svgp_json/`：三個情境的完整 SVGP JSON

SVGP tuning 的最佳參數與 trial 紀錄仍保存在相鄰的 `svgp_tuning_current/`，本次整理沒有移動或刪除它們。
