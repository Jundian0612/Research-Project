# Root legacy files archive（2026-08-01）

本資料夾保存原先散落在 `weather2k_exp/` 根目錄、目前三模型實驗與 SVGP tuning 不會呼叫的舊檔案。

## 封存內容

- `2K_DLinear_FRK_500.ipynb`：早期 notebook 實驗
- `2K_DLinear_FRK.py`：早期 standalone DLinear+FRK 程式
- `2K_STDK.py`：早期 standalone STDK 程式；目前版本直接載入 `spatial-adapter/examples/baselines/stdk/st_interp.py`
- `plot_dlinear_loss.py`：早期 loss 繪圖工具
- `weather2k_eda.py`：獨立 EDA 工具

這些檔案沒有永久刪除；只是從根目錄移到此處，必要時仍可復原。資料夾內檔案不應由清理程序刪除。
