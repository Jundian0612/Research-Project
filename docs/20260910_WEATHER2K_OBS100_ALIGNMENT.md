# STDK+Q 與 SVGP 的前處理／選參對齊

> 2026-09-12 更新：STDK+Q 前段 STDK 與獨立 STDK 現已依 pinned spatial-adapter
> trainer 啟用 EMA，並使用 batch-average validation loss 選 checkpoint。此變更會改變
> fitted STDK 與 QConvLSTM grids；2026-09-10/11 的既有 tuning/formal 輸出列為
> pre-EMA 歷史結果，現行流程必須重新調參。

> 2026-09-15 評估流程更新：SVGP、純 STDK、DLinear+FRK 與 STDK+Q 恢復為
> 「每個 seed、每個情境分別訓練一個模型實例」。三次執行仍使用相同 Train700、
> Val150、train500／held-out100 與標準化規則；情境差異位於最終評估區塊。
> 2026-09-12 產生的 one-fit 輸出保留為歷史對照，不與新流程統計合併。

2026-09-10：已修改新執行的流程，既有 train500-normalization 結果仍保留。
檢查時原正式 seed41 只有 log，尚無 JSON 或比較 CSV，不能認定已完成。
已啟動的單 seed Python 程序使用啟動時載入的程式；本次未中止它。
即使它之後完成，也屬於舊規則，不可混入新規則的五-seed 統計。

## 修改內容

- 固定 train500/heldout100 後，以 `RandomState(seed)` 從 train500 選100站，與 SVGP 的 obs100 索引相同。
- 使用這100站的 Train700 計算全域 μ、σ（ddof=0，σ<1e-12 則設1），實際訓練及 validation 仍涵蓋500站。
- JSON 保存 normalization source、obs 索引、μ、σ及 protocol config。
- 2026-09-15 起，STDK q50 與 QConvLSTM q50 都以 pinball loss 訓練及選 checkpoint；q05/q95 同樣按各自 pinball loss 選擇。
- QConvLSTM 的 q05/q50/q95 訓練與 Val150 labels 改為同一 fitted STDK 的對應 quantile series；Weather2K 真值只用於最後情境評分。
- trial objective 保持 seeds41/42 平均 standardized Val q50 RMSE，和 SVGP 的尺度一致。
- fitted STDK 對照共用新 normalization。未更改其模型架構與主要超參數。

## 搜尋差異與尺度

| 模型 | Trial objective | 搜尋 |
|---|---|---|
| SVGP | seeds41/42 平均 best standardized Val RMSE | 先 kernel×LR，再 inducing points，實際也是兩階段 |
| STDK+Q 新規則 | seeds41/42 平均 standardized Val q50 RMSE | 先 grid×radius，再 filters×weight decay |
| DLinear+FRK | seeds41/42 平均原始尺度 Val train500 RMSE | 30-trial Optuna 聯合搜尋 DLinear 與空間 loss／FRK 參數 |
| 純 STDK | 無額外 trial search；Val MSE 選 epoch | 固定架構／主要參數 baseline |

同一 seed 的 RMSE_raw = σ × RMSE_scaled，因此同 seed 排名不因尺度改变。
跨 seed 求平均時，raw 與 scaled 的權重不同，可能選出不同組態；不能把不同尺度的 Val 數值直接作模型排行榜。
SVGP 與 Q 的 early stopping loss 依各自模型定義，不再強行對齊；跨模型最終仍比較相同情境的原始尺度指標。
不同模型搜尋不同類型的超參數是合理的，但需公開範圍及預算；本次不變更 SVGP 或 DLinear 的搜尋。

## 避免混用

新的 tuner 預設 output-dir：`weather2k_exp/air_temperature/20260915_qconvlstm_location_specific_tuning`。
新版 resume／調參相容性要求 config 包含 `normalization_source=obs100_train700`、`stdk_q50_loss=pinball`、`qconv_training_target=quantile_specific_fitted_stdk`、`qconv_model_scope=location_specific` 與 `q50_checkpoint_selection=pinball`。舊的共享模型或 MSE／真值標籤輸出不會被新版誤用。

QConvLSTM 現在依論文的目標位置 `s_0` 定義執行：Time150 對 train500 各自訓練三個 quantile 模型；Space100 與 ST100×150 對 held-out100 各自訓練三個 quantile 模型。各站只使用自己的 fitted-STDK Train700/Val150 序列與 local grids，站與站之間不共享 QConvLSTM 權重。
capacity-only 也會檢查 interface best 的 protocol 及 seeds，拒絕舊結果。
正式模型拒絕缺少上述標記的最佳參數檔；舊 locked_params 保留為歷史證據，不能啟動新規則正式測試。
新兩階段 tuning 完成後才更新正式參數。此時尚未有新規則的正式最佳值。

## 新調參指令

在 GPU 可用的 `.venv-wsl`、獨立 tmux session 中，確認舊訓練已完成或已自行停止後執行：

```bash
mkdir -p weather2k_exp/air_temperature/20260915_qconvlstm_location_specific_tuning
set -o pipefail
PYTHONUNBUFFERED=1 python -u weather2k_exp/tune_stdk_qconvlstm_hyperparams.py \
  --stage all --seeds 41 42 \
  --output-dir weather2k_exp/air_temperature/20260915_qconvlstm_location_specific_tuning \
  2>&1 | tee -a weather2k_exp/air_temperature/20260915_qconvlstm_location_specific_tuning/run.log
```

本次驗證：缺少新 protocol 的舊參數檔會被拒絕；CPU 小規模 q05/q50/q95 端到端流程及新版 tuner q50 流程通過，direct checkpoint 使用 pinball 且不含 residual epoch0。
未啟動正式 GPU 調參、未刪除舊結果或 push。
