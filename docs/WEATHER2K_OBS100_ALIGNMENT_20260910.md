# STDK+Q 與 SVGP 的前處理／選參對齊

> 2026-09-12 更新：STDK+Q 前段 STDK 與獨立 STDK 現已依 pinned spatial-adapter
> trainer 啟用 EMA，並使用 batch-average validation loss 選 checkpoint。此變更會改變
> fitted STDK 與 QConvLSTM grids；2026-09-10/11 的既有 tuning/formal 輸出列為
> pre-EMA 歷史結果，現行流程必須重新調參。

> 2026-09-12 評估流程更新：SVGP、純 STDK、DLinear+FRK 與 STDK+Q 現在都採用
> 「每個 seed／每組超參數只訓練一次，再用同一 fitted model 評估 Time150、
> Space100、ST100x150」。模型 JSON 同時保存三個 target；`experiments_runner.py`
> 只負責把同一份結果拆成三張比較表。過去每個情境分開重訓的輸出保留為歷史結果，
> 不與此流程的新結果合併。

2026-09-10：已修改新執行的流程，既有 train500-normalization 結果仍保留。
檢查時原正式 seed41 只有 log，尚無 JSON 或比較 CSV，不能認定已完成。
已啟動的單 seed Python 程序使用啟動時載入的程式；本次未中止它。
即使它之後完成，也屬於舊規則，不可混入新規則的五-seed 統計。

## 修改內容

- 固定 train500/heldout100 後，以 `RandomState(seed)` 從 train500 選100站，與 SVGP 的 obs100 索引相同。
- 使用這100站的 Train700 計算全域 μ、σ（ddof=0，σ<1e-12 則設1），實際訓練及 validation 仍涵蓋500站。
- JSON 保存 normalization source、obs 索引、μ、σ及 protocol config。
- direct q50 epoch 改以 Val MSE 選 checkpoint，與最小 Val RMSE 等價；不新增 epoch0 候選。
- QConvLSTM 訓練仍用 quantile pinball loss，q05/q95 epoch 仍按 pinball 選擇。SVGP 則以 variational ELBO 訓練；不將兩種模型訓練 loss 強行改成一樣。
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
SVGP 的 early stopping 及 Q 的 q50 epoch 現在同樣以 RMSE/MSE 排名；訓練目標仍各自遵循模型定義。
不同模型搜尋不同類型的超參數是合理的，但需公開範圍及預算；本次不變更 SVGP 或 DLinear 的搜尋。

## 避免混用

新的 tuner 預設 output-dir：`weather2k_exp/air_temperature/qconvlstm_obs100_tuning_20260910`。
resume 要求 config 包含 `normalization_source=obs100_train700`、`q50_checkpoint_selection=mse`。
capacity-only 也會檢查 interface best 的 protocol 及 seeds，拒絕舊結果。
正式模型拒絕缺少上述標記的最佳參數檔；舊 locked_params 保留為歷史證據，不能啟動新規則正式測試。
新兩階段 tuning 完成後才更新正式參數。此時尚未有新規則的正式最佳值。

## 新調參指令

在 GPU 可用的 `.venv-wsl`、獨立 tmux session 中，確認舊訓練已完成或已自行停止後執行：

```bash
mkdir -p weather2k_exp/air_temperature/qconvlstm_obs100_tuning_20260910
set -o pipefail
PYTHONUNBUFFERED=1 python -u weather2k_exp/tune_stdk_qconvlstm_hyperparams.py \
  --stage all --seeds 41 42 \
  --output-dir weather2k_exp/air_temperature/qconvlstm_obs100_tuning_20260910 \
  2>&1 | tee -a weather2k_exp/air_temperature/qconvlstm_obs100_tuning_20260910/run.log
```

本次驗證：seeds41–45 obs100 索引對照歷史 SVGP JSON 一致；新 report 可 resume，缺少新 protocol 的 report 被拒絕；
CPU 小規模兩階段 q50 流程通過，direct checkpoint 使用 MSE 且不含 residual epoch0。
未啟動正式 GPU 調參、未刪除舊結果或 push。
