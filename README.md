# Weather2K Spatiotemporal Forecasting

本專案研究 Weather2K 測站氣溫的時間、空間與時空外推，並在一致的資料切分與評分規則下比較下列方法：

- Sparse Variational Gaussian Process（SVGP）
- DLinear + differentiable FRK surrogate
- Spatial-adapter Space-Time DeepKriging（STDK）
- Spatial-adapter STDK + QConvLSTM

Repository 保存實驗程式、固定參數、重現文件與可追蹤的結果產物。原始 Weather2K 資料與 Python 環境不納入 Git 版本控制。

## Methods

| Model | Role in this project |
| --- | --- |
| SVGP | 時空 Gaussian process baseline |
| DLinear + FRK | DLinear 時間預測搭配 FRK 空間代理模型 |
| Spatial-adapter STDK | 以 Spatial-adapter 實作建立的 STDK baseline |
| STDK + QConvLSTM | 先由 Spatial-adapter STDK 產生局部網格，再以 QConvLSTM 輸出 q05、q50、q95 |

Spatial-adapter 以 Git submodule 固定在 commit `2aea188f3b8d92f948663b6705f0a22850d6e4ee`，方便追蹤 STDK 前段所依據的實作版本。

## Benchmark protocol

四個模型採用共同的 Weather2K 評估骨架：

| Item | Setting |
| --- | --- |
| Time split | 最後 1000 個時間點切成 Train700 / Val150 / Test150 |
| Station split | 每個 seed 抽樣 600 站；train500 與 strict held-out100 |
| Training roles | train500 內含 observed100 與 unobserved400 |
| Normalization | 僅使用 observed100 × Train700 估計統計量 |
| Seeds | 正式實驗使用 41–45 |

每個模型、每個 seed 只訓練一次，再以同一個 fitted model 評估三種情境：

| Scenario | Evaluation data |
| --- | --- |
| Time150 | train500 測站的 Test150 |
| Space100 | held-out100 測站的前 850 個時間點 |
| ST100×150 | held-out100 測站的 Test150 |

Val150 用於選參與 checkpoint selection；Test150 與 held-out100 的真值只用於最終評分。共同指標為 RMSE、MSE、MAE、R²；可輸出分位數的模型另外計算 90% coverage 與 MPIW。

## Installation

```bash
git clone https://github.com/Jundian0612/Research-Project.git
cd Research-Project
git submodule update --init --recursive spatial-adapter

python3 -m venv .venv-wsl
source .venv-wsl/bin/activate
python -m pip install --upgrade pip
```

請先依照作業系統與 CUDA 版本，從 [PyTorch 官方安裝頁](https://pytorch.org/get-started/locally/) 安裝 PyTorch，再安裝本專案套件：

```bash
python -m pip install -r weather2k_exp/requirements-core.txt
python -m pip install -r weather2k_exp/requirements-baselines.txt
```

確認 GPU：

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

完整的新裝置安裝、資料同步及 tmux 操作方式見 [Weather2K setup guide](docs/SETUP_WEATHER2K.md)。

## Data

實驗程式預設讀取：

```text
Josh's Weather2K/Weather2K/weather2k.npy
```

若要從 Weather2K submodule 取得資料：

```bash
git submodule update --init "Josh's Weather2K/Weather2K"
git -C "Josh's Weather2K/Weather2K" lfs pull
```

確認檔案不是 Git LFS pointer，並記錄資料雜湊：

```bash
python -c "import numpy as np; p=\"Josh's Weather2K/Weather2K/weather2k.npy\"; x=np.load(p, mmap_mode='r'); print(x.shape, x.dtype)"
sha256sum "Josh's Weather2K/Weather2K/weather2k.npy"
```

## Quick start

先以 smoke test 檢查 STDK + QConvLSTM 流程：

```bash
python -u weather2k_exp/tune_stdk_qconvlstm_hyperparams.py \
  --stage all \
  --smoke-test \
  --seeds 41 \
  --max-interface-trials 1 \
  --max-capacity-trials 1 \
  --output-dir /tmp/weather2k-smoke
```

執行 seed 41 的 SVGP、DLinear + FRK 與純 STDK；每個模型訓練一次並評估三種情境：

```bash
OUT=weather2k_exp/air_temperature/baselines_onefit_seed41
mkdir -p "$OUT"

SEED_LIST='[41]' \
RUN_STDK=1 \
RUN_SVGP=1 \
RUN_DLINEAR=1 \
RUN_QCONVLSTM=0 \
WEATHER2K_OUTPUT_DIR="$OUT" \
PYTHONUNBUFFERED=1 \
python -u weather2k_exp/experiments_runner.py \
  2>&1 | tee "$OUT/run.log"
```

執行 seed 41 的 STDK + QConvLSTM：

```bash
OUT=weather2k_exp/air_temperature/stdk_qconvlstm_seed41
mkdir -p "$OUT"

PYTHONUNBUFFERED=1 python -u weather2k_exp/2K_STDK_QConvLSTM.py \
  --params-file weather2k_exp/2K_best_stdk_qconvlstm_params_500to100.json \
  --seed 41 \
  --forecast-mode block5to5 \
  --prediction-mode direct \
  --output-dir "$OUT" \
  2>&1 | tee "$OUT/run.log"
```

正式實驗耗時較長，建議在 tmux 中執行。輸出資料夾會保存 log、參數、環境資訊、指標與預測檔案，之後再整理成封存結果。

## Project structure

| Path | Contents |
| --- | --- |
| [`weather2k_exp/`](weather2k_exp/) | Weather2K 的模型、調參與實驗 runner |
| [`spatial-adapter/`](spatial-adapter/) | 固定版本的 Spatial-adapter submodule |
| [`STDK_QConvLSTM_reproduction_results/`](STDK_QConvLSTM_reproduction_results/) | STDK–QConvLSTM 重現程式與報告 |
| [`docs/`](docs/) | 安裝、比較規則、實作對齊與 repository 文件 |
| [`project/`](project/) | 先前研究程式與實驗內容 |
| [`Josh's Weather2K/`](<Josh's Weather2K/>) | Weather2K dataset submodule 與資料位置 |

## Reproducibility

正式實驗應保存以下資訊：

1. 使用的 seed、資料切分與固定超參數。
2. Python、PyTorch、CUDA 與主要套件版本。
3. GPU 型號與程式、參數、資料的 SHA-256。
4. 完整 terminal log、逐 seed 指標及預測輸出。
5. 明確區分 validation 選參結果與最終 test 結果。

比較規則與目前已確認的限制見 [Weather2K comparison audit](docs/WEATHER2K_COMPARISON_AUDIT_20260910.md)；STDK 與上游 Spatial-adapter 的對齊情況見 [STDK alignment notes](docs/STDK_SPATIAL_ADAPTER_ALIGNMENT_20260913.md)。

## References

- [Spatial-adapter](https://github.com/STLABTW/spatial-adapter)
- [Spatial-adapter experiment examples](https://github.com/STLABTW/spatial-adapter/tree/main/examples/experiments)
- [Space-Time DeepKriging paper](https://arxiv.org/abs/2306.11472)
- [Space-Time.DeepKriging repository](https://github.com/pratiknag/Space-Time.DeepKriging)
- [Weather2K dataset](https://huggingface.co/datasets/BUPT-PRIS-727/Weather2K)
