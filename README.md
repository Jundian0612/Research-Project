# 全球氣候長期時空結構特性與預測方法之研究

本 repository 保存氣候資料時間序列與時空預測方法的研究程式、實驗設定及重現紀錄。研究從規則網格上的長期氣溫時間序列出發，逐步加入空間統計、未知位置推估與機率預測，建立能同時處理時間外推、空間外推及時空外推的比較流程。

目前的研究發展分成兩個相連階段：

1. `project/`：以月氣溫資料探索單點與多點時間序列模型，並將時間預測殘差交由 FRK 處理空間結構。
2. `weather2k_exp/`：將前一階段的方法延伸到 Weather2K 測站資料，統一比較 DLinear+FRK、SVGP、STDK 與 STDK+QConvLSTM。

## Research direction

氣候資料同時包含長期趨勢、季節性、測站間的空間關聯，以及未來時間與未知位置的不確定性。本專案依序處理這些問題：

```text
氣候網格／測站資料
        │
        ├─ 時間序列結構
        │    SARIMA · AutoARIMA · VARIMA · DLinear
        │
        ├─ 時間模型延伸到空間
        │    DLinear + FRK
        │
        └─ 完整時空模型比較
             SVGP · STDK · STDK + QConvLSTM
                      │
                      └─ q05 / q50 / q95 機率預測
```

研究重點包括：

- 長期氣溫序列的趨勢、季節性與自相關結構；
- 傳統統計模型與深度時間序列模型的預測差異；
- 如何由已觀測位置推估未知位置；
- 時間、空間與時空外推在共同切分和評分規則下的比較；
- 分位數預測區間的準確度、寬度與覆蓋率；
- 不同裝置與軟體環境下的可重現性。

## From time series to spatiotemporal models

### Time-series foundation

[`project/`](project/) 保存研究早期使用 NetCDF 月資料進行的分析。主要變數為近地表氣溫 `T2M`，並從美國本土範圍抽取規則網格進行建模。

| Method | Research role |
|---|---|
| SARIMA | 描述單一位置的季節性與時間相依 |
| AutoARIMA | 為不同位置選擇各自的時間序列階數 |
| VARIMA | 探索多位置序列的聯合時間動態 |
| DLinear | 以分解式線性網路建立長期時間預測 baseline |
| DLinear + FRK | 以 DLinear 負責時間預測，再用 FRK 表示殘差的空間結構 |

相關 notebook 包括 [`SARIMA.ipynb`](project/SARIMA.ipynb)、[`AutoARIMA.ipynb`](project/AutoARIMA.ipynb)、[`VARIMA.ipynb`](project/VARIMA.ipynb)、[`DLinear.ipynb`](project/DLinear.ipynb) 與 [`DLinear_FRK_500.ipynb`](project/DLinear_FRK_500.ipynb)。這一區保留研究演進與探索性分析，部分 notebook 的環境及資料路徑仍反映當時的執行設定。

### Weather2K spatiotemporal extension

[`weather2k_exp/`](weather2k_exp/) 將上述時間模型延伸到測站層級的時空問題，並納入下列比較與延伸模型：

| Model | Role |
|---|---|
| DLinear + differentiable FRK | 延續 `project/` 的時間模型加空間殘差路線 |
| SVGP | Gaussian process 時空比較基準 |
| STDK | 使用 Spatial-adapter repository 中的 Space-Time DeepKriging baseline |
| STDK + QConvLSTM | 由 STDK 產生局部網格，再以 QConvLSTM 建立分位數預測 |

目前 Weather2K 比較固定使用共同的資料抽樣、Train／Validation／Test 時間切分、標準化來源、seeds 與評分規則。每個模型在每個 seed 只訓練一次，再以同一 fitted model 評估：

- 已知位置的未來時間預測；
- 未知位置的空間推估；
- 未知位置與未來時間同時發生的時空外推。

完整比較規則見 [Weather2K comparison audit](docs/WEATHER2K_COMPARISON_AUDIT_20260910.md)。STDK 與 QConvLSTM 的實作來源及改編範圍見 [STDK and QConvLSTM source audit](docs/STDK_QCONVLSTM_SOURCE_AUDIT_20260913.md)。

## Repository structure

| Path | Contents |
|---|---|
| [`project/`](project/) | 氣候網格資料的時間序列研究、DLinear+FRK 發展過程與歷史 notebook |
| [`weather2k_exp/`](weather2k_exp/) | Weather2K 時間、空間、時空實驗程式與結果根目錄 |
| [`docs/`](docs/) | 安裝、資料切分、模型公平比較及來源稽核文件 |
| [`spatial-adapter/`](spatial-adapter/) | 固定版本的 Spatial-adapter Git submodule |
| [`STDK_QConvLSTM_reproduction_results/`](STDK_QConvLSTM_reproduction_results/) | Space-Time.DeepKriging／QConvLSTM 重建程式與限制說明 |
| [`geospatial-neural-adapter-dev/`](geospatial-neural-adapter-dev/) | 相關地理空間模型開發內容 |
| [`Josh's Weather2K/`](<Josh's Weather2K/>) | Weather2K dataset submodule 與資料位置 |

## Getting started

Clone repository 並初始化 submodules：

```bash
git clone https://github.com/Jundian0612/Research-Project.git
cd Research-Project
git submodule update --init --recursive
```

Weather2K 實驗建議使用獨立 Python environment：

```bash
python3 -m venv .venv-wsl
source .venv-wsl/bin/activate
python -m pip install --upgrade pip
python -m pip install -r weather2k_exp/requirements-core.txt
python -m pip install -r weather2k_exp/requirements-baselines.txt
```

PyTorch 應依作業系統、GPU 與 CUDA 版本使用官方指令安裝。新裝置設定、Weather2K 資料位置、tmux 與裝置間同步方式見 [Weather2K setup guide](docs/SETUP_WEATHER2K.md)。

`project/` 中的 notebook 涉及 Xarray、NetCDF、Darts、Statsmodels、PyTorch、Optuna、Cartopy 與空間分析套件。它們是研究歷程的一部分，執行前應依 notebook 的 import 與資料路徑建立對應環境。

## Reproducibility

正式實驗保存下列資訊：

1. 資料來源、時間範圍、空間位置及 train／validation／test 切分；
2. 標準化統計量的估計範圍；
3. 模型設定、選參指標與 seeds；
4. Python、套件、CUDA 與 GPU 環境；
5. 程式、資料與參數檔的 SHA-256；
6. validation 選參結果與 final test 結果的明確區分。

原始大型資料、虛擬環境與模型 checkpoint 不放入 Git。可保留的指標、表格、參數、來源雜湊及必要逐筆預測，依各實驗資料夾的 README 分類保存。

## Documentation

- [Weather2K environment and data setup](docs/SETUP_WEATHER2K.md)
- [Weather2K comparison audit](docs/WEATHER2K_COMPARISON_AUDIT_20260910.md)
- [Weather2K observed100 alignment](docs/WEATHER2K_OBS100_ALIGNMENT_20260910.md)
- [STDK alignment with Spatial-adapter](docs/STDK_SPATIAL_ADAPTER_ALIGNMENT_20260913.md)
- [STDK and QConvLSTM source audit](docs/STDK_QCONVLSTM_SOURCE_AUDIT_20260913.md)
- [Repository hygiene](docs/REPOSITORY_HYGIENE.md)

## References

- [Weather2K dataset](https://huggingface.co/datasets/BUPT-PRIS-727/Weather2K)
- [Spatial-adapter](https://github.com/STLABTW/spatial-adapter)
- [Space-Time.DeepKriging](https://github.com/pratiknag/Space-Time.DeepKriging)
- [Space-Time.DeepKriging paper](https://arxiv.org/abs/2306.11472)
