# Weather2K STDK + QConvLSTM：新裝置完整兩階段調參

執行日期：2026-09-09 至 2026-09-10。狀態：**Stage 1 與 Stage 2 全部完成，validation-only**。
共 13 configurations × seeds 41、42 = 26 runs，原始 reports 的 elapsed_seconds 累計約 **11.30 小時**。
尚未以本輪參數完成正式五-seed Test150 評估。

## 目的與模型

先前 Stage 1 在舊裝置、Stage 2 在新裝置執行，相同設定出現跨環境差異。
本輪在新裝置固定環境，重新執行完整兩階段搜尋，避免把兩台的 validation 分數直接混用。

模型流程為 **Spatial-adapter STDK → local grids → direct QConvLSTM**。
STDK 載入 `spatial-adapter/examples/baselines/stdk/st_interp.py`，
QConvLSTM 使用本專案 `STDK_QConvLSTM_reproduction_results/code/STDK_QConvLSTM_reproduction.py`
中的 `github_3block` PyTorch 重現版（依作者公開 notebook 架構），不是 residual 版本。
本模型是搭配 Spatial-adapter 並適配 Weather2K 的組合，不應標示為作者整套流程的完全重現。

此處 Stage 1 / Stage 2 指「兩階段調參」，不是前段 STDK / 後段 QConvLSTM。

## 資料切分與固定設定

| 項目 | 設定 |
|---|---|
| 變數 | air_temperature |
| Tuning seeds | 41、42 |
| 測站 | 抽樣 600，train500 / strict held-out100 |
| 時間 | Train700 / chronological Val150 / Test150 |
| 訓練及選參 | train500 的 Train700 訓練、Val150 選參 |
| 模式 | direct，過去 5 張 grids → 未來 5 點，block5to5 |
| Quantile | q50 only |
| Learning rate / batch size | 0.0001 / 64 |
| Max QConvLSTM epochs / patience | 40 / 10 |
| STDK epochs | 350 |

Test150 與 held-out100 真值未用於本輪訓練或選參，沒有計算其最終指標，也沒有寫入模型 checkpoint。
組態以兩個 seeds 的平均 standardized q50 Val RMSE 選擇；
direct QConvLSTM 的 epoch checkpoint 仍以 Val pinball loss 選擇，STDK q50 以 Val MSE 選擇。

## Stage 1：interface 搜尋

固定 filters=64、weight decay=0，比較 grid 5/8/11 × radius 0.1/0.2/0.3。

| Trial | Grid | Radius | Mean Val RMSE | Std（2 seeds） |
|---:|---:|---:|---:|---:|
| 0 | 5 | 0.1 | 0.682631 | 0.033684 |
| 1 | 8 | 0.1 | 0.680044 | 0.034349 |
| **2** | **11** | **0.1** | **0.676611** | **0.032484** |
| 3 | 5 | 0.2 | 0.680321 | 0.015102 |
| 4 | 8 | 0.2 | 0.694264 | 0.022260 |
| 5 | 11 | 0.2 | 0.692996 | 0.025267 |
| 6 | 5 | 0.3 | 0.686126 | 0.019514 |
| 7 | 8 | 0.3 | 0.690975 | 0.024361 |
| 8 | 11 | 0.3 | 0.699009 | 0.030084 |

最佳 interface 為 grid=11、radius=0.1。這是兩個 seeds 的 validation 排名，不代表已證明對所有 seeds 穩定最佳。

## Stage 2：capacity 搜尋

固定新 Stage 1 選出的 grid=11、radius=0.1。

| Trial | Filters | Weight decay | Mean Val RMSE | Std（2 seeds） |
|---:|---:|---:|---:|---:|
| 0 | 32 | 0 | 0.689634 | 0.036760 |
| 1 | 32 | 1e-5 | 0.687520 | 0.037989 |
| **2** | **64** | **0** | **0.676611** | **0.032484** |
| 3 | 64 | 1e-5 | 0.682641 | 0.030802 |

Stage 2 最佳仍是 filters=64、weight decay=0，沒有取得超越 Stage 1 最佳組態的改善。

## 本輪選出的參數與解讀

```json
{
  "GRID_SIZE": 11,
  "NEIGHBOURHOOD_RADIUS": 0.1,
  "CONV_FILTERS": 64,
  "QCONV_WEIGHT_DECAY": 0.0,
  "QCONV_LR": 0.0001,
  "QCONV_BATCH_SIZE": 64,
  "QCONV_EPOCHS": 40,
  "QCONV_PATIENCE": 10
}
```

平均 Val RMSE **0.676611 ± 0.032484**。Std 是兩個 seeds 的離散程度，不是信賴區間。
本輪完成時已更新 `weather2k_exp/2K_best_stdk_qconvlstm_params_500to100.json`；
封存整理時已核對其內容與 `summary/qconvlstm_tuning_best_params.json` 一致。

**本輪選出的設定不等於所有歷史候選中的最低 RMSE。**
前次在同一新裝置，grid=5、radius=0.1、filters=32、weight decay=1e-5 曾得到 0.668526。
兩階段搜尋先以 filters=64 選 grid，再固定 grid 選 filters，可能漏掉 grid 與 capacity 的交互影響。
因此應稱為「本輪既定兩階段流程選出的設定」，並保留前次結果如實報告。
本次整理不新增候選比較、不重新挑選參數，也不宣告正式測試已完成。

## 環境與跨裝置檢查

以下由使用者 terminal 截圖提供，不是訓練程序自動保存的完整 lockfile：

| 項目 | 本輪新裝置 |
|---|---|
| Python | 3.12.3 |
| PyTorch / CUDA runtime | 2.14.0+cu130 / 13.0 |
| NumPy / scikit-learn | 2.4.6 / 1.9.0 |
| GPU | NVIDIA RTX 2000 Ada Generation |
| 環境 | `.venv-wsl` |

診斷程序回報 deterministic algorithms、cuDNN benchmark、cuDNN deterministic 均為 False。
兩台資料與三份模型程式的 SHA256 相同，但套件版本與 GPU 不同。
同機 seed 42 檢查只觀察到小幅 RMSE 波動；跨裝置差異原因尚未被單獨定位，不能歸因於 GPU「性能較好或較差」。

相關封存：

- [舊裝置 Stage 1](../STDK_QConvLSTM_Weather2K_q50_interface_tuning_20260908/README.md)
- [新裝置前次 Stage 2](../STDK_QConvLSTM_Weather2K_q50_capacity_tuning_20260909/README.md)
- [Seed 42 同機重跑檢查](../STDK_QConvLSTM_Weather2K_seed42_repro_check_20260909/README.md)

## 資料夾結構

```text
qconvlstm_retune_rtx2000_20260909/
├── README.md
├── trials/
│   ├── interface/    # 36 個 JSON：9 組 × params、2 seeds、summary
│   └── capacity/     # 16 個 JSON：4 組 × params、2 seeds、summary
├── summary/          # 2 個 CSV 與 3 個最佳參數 JSON
├── logs/run.log
└── provenance/
    ├── QCONVLSTM_TUNING_README.md  # runner 原始簡短說明
    ├── archive_manifest.json      # 59 個原始檔案的新舊位置與 SHA256
    └── incomplete_formal_seed41_20260910/ # 舊規則中止的 seed41 紀錄
```

59 個原始檔案均保留，搬移後已核對 SHA256。`run.log` 仍受 `*.log` 忽略規則保護，
一般 git add 不會上傳；Markdown、summary、trial JSON 與 manifest 可加入版本控制。
本次封存沒有修改根目錄 operational 參數，也沒有 commit / push。

## 執行方式與封存後注意事項

原始執行命令（只供追溯）：

```bash
PYTHONUNBUFFERED=1 python -u weather2k_exp/tune_stdk_qconvlstm_hyperparams.py \
  --stage all --seeds 41 42 \
  --output-dir weather2k_exp/air_temperature/qconvlstm_retune_rtx2000_20260909
```

**分類後不要再把本目錄直接當 runner 的 output-dir。**
runner 只在 output-dir 根層尋找 resume reports，不會自動搜尋 `trials/` 或 `summary/`，
重新使用原指令可能重新訓練並生成另一組結果。後續實驗應使用全新的 output-dir。
先前依根層最佳參數檔偵測完成的監看指令也不適用於此封存結構；本輪已完成。

下一步先確定並記錄四模型共同的選參與評估規則；若採用本輪既定兩階段選參結果，
固定上述參數及環境後，才跑 seeds 41–45 的完整 q05/q50/q95 三情境正式評估。
SVGP、DLinear+FRK、純 STDK 必須使用相同資料切分及可用資訊規則，
純 STDK 應與組合模型前段設定一致。任何追加候選選參仍只能用 validation。
