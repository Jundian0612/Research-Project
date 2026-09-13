# STDK 與 STDK+QConvLSTM 來源稽核（2026-09-13）

## 結論

目前的純 STDK **直接使用** `spatial-adapter` 固定版本中的 STDK 模型，並讓本地訓練迴圈對齊該版本的 optimizer、EMA、validation aggregation、early stopping 與 checkpoint 規則。因此它是「以 Spatial-adapter repository 的 STDK baseline 為基礎，套用本研究 Weather2K 切分」；它不是該 repository 完整的兩階段 Spatial Adapter，因為沒有執行 residual Spatial Adapter stage。

目前的 STDK+QConvLSTM 確實以同一個 STDK baseline 當前段，再接入由 Nag et al. 論文與公開 notebook 重建的 QConvLSTM。不過作者沒有公開 Table 2 的完整橋接程式與必要檔案，因此 Weather2K 版本包含明確的研究改編。較準確的名稱是：

> Spatial-adapter-repository STDK + paper/notebook-inspired shared direct QConvLSTM, adapted and partially tuned for Weather2K.

不宜稱為「完整 Spatial Adapter + QConvLSTM」或「作者 QConvLSTM 的精確重現」。

## 固定來源版本

- Spatial-adapter repository：<https://github.com/STLABTW/spatial-adapter>
- 本地 submodule commit：`2aea188f3b8d92f948663b6705f0a22850d6e4ee`（tag `v0.6.0`）
- QConvLSTM 論文：Nag et al., *Space-Time.DeepKriging: A Deep Learning Based Spatio-Temporal Modeling Approach*，<https://arxiv.org/abs/2306.11472>
- QConvLSTM repository：<https://github.com/pratiknag/Space-Time.DeepKriging>

## 純 STDK 對照

| 部分 | 目前實作 | 來源與判定 |
|---|---|---|
| 模型類別 | `STInterpMLP` / `create_model` | 直接動態載入 `spatial-adapter/examples/baselines/stdk/st_interp.py`，屬直接沿用 |
| 空間 basis | centers 25/81/121、uniform、fixed、Wendland | `STInterpMLP` defaults，直接沿用 |
| 時間 basis | centers 10/15/45、Gaussian、bandwidth factor 2.5 | `STInterpMLP` defaults，直接沿用 |
| MLP | hidden 256/256/128、dropout 0.1、LayerNorm | `STInterpMLP` defaults，直接沿用 |
| optimizer | AdamW，lr 1e-3，weight decay 1e-4 | repository Weather2K `first_stage` config，對齊 |
| 訓練 | batch 512、最多 350 epochs、patience 30 | repository Weather2K `first_stage` config，對齊 |
| validation | 每個 validation batch 的 MSE 再取平均 | repository `Trainer` active branch，對齊 |
| EMA | 啟用；decay = `1 - 1/(10*batches_per_epoch)` | repository `Trainer`，對齊 |
| checkpoint | validation loss 最低者 | repository `Trainer`，對齊 |
| 訓練迴圈 | 本地重新寫出同一 active branch | 沒有直接建立 repository `Trainer`；行為對齊，但不是逐行直接呼叫 |
| Spatial Adapter residual stage | 未執行 | repository 的完整 Weather2K experiment 有 Stage 2；目前純 STDK 只取 Stage 1 |

上游 `examples/experiments/weather2k/experiment.py` 明確定義：Stage 1 是 STDK，Stage 2 才是 Spatial Adapter on residuals。因此本研究檔名中的「純 STDK」是正確的；將它稱作完整「Spatial Adapter」則不正確。

## 本研究 Weather2K 設計

下列不是上游 Weather2K production experiment 的原設定，而是四模型比較共同使用或本研究保留的設計：

- 最後 1000 時點切成 Train700 / Val150 / Test150；
- 每個 seed 從資料抽 600 站，再分成 train500 與完全 held-out100；
- train500 內保留 obs100 / unobs400 的角色；
- 以 obs100 × Train700 計算目標平均與標準差；
- 座標與時間映射到 `[0,1]`；
- 每個模型、每個 seed 訓練一次，再評估 Time150、Space100、ST100×150；
- seeds 41–45 與共同 station split；
- held-out100 真值不參與訓練、validation 或選參。

Spatial-adapter repository 的 production Weather2K config 使用 `space_ratio_keep=0.1`、80/10/10 時間比例與 30 replications 等設計。直接複製這些項目會改變本研究的四模型 benchmark，因此目前只對齊 STDK 模型與訓練方法，資料實驗設計仍採本研究規則。

## STDK+QConvLSTM 來源對照

| 部分 | 目前 Weather2K 實作 | 來源 |
|---|---|---|
| Q 的前段 | 同一 `spatial-adapter` STDK q50；另訓練 q05/q95 STDK | q50 來自 Spatial-adapter STDK；tails 是本研究為機率預測加上的延伸 |
| Q 輸入概念 | 目標位置周圍的 STDK q05/q50/q95 局部網格序列 | 論文 QConvLSTM 方法 |
| Q 網路主要結構 | 5×5、3×3、1×1 三個 ConvLSTM blocks，64 filters，前兩層 BatchNorm，flatten 後輸出 5 步 | 公開 `CONV_LSTM.ipynb` 的重建；論文文字只明確指定 3×3 convolution 與 64 maps |
| recurrent 初始化與 activation | Keras-compatible PyTorch 重建 | 公開 notebook 行為的重建 |
| quantile ordering | `q50 ± lambda*abs(q-0.5)*sigmoid(raw)` | 論文 Eq. (7) |
| direct 輸出 | QConvLSTM 直接產生 q05/q50/q95 | 論文型式；目前 formal default 是 `prediction_mode=direct` |
| residual 模式 | 程式保留可選的 STDK + learned residual 與 validation fallback | 本研究的實驗性延伸，不是論文原流程，也不是目前 formal default |
| lookback / horizon | 5 → 5 | 論文 simulation 最後 5 點預測與公開 forecasting 程式所採的 5-frame 設定；固定而未在最終 grid search 中搜尋 |
| grid size / radius | 11×11 / 0.1 | Weather2K validation 調參結果；不是作者固定值 |
| 邊界網格 | shift 回 `[0,1]` | 公開材料未說明；reproduction assumption |
| Q 訓練標籤 | train500 的 Weather2K 真值 | Weather2K 改編。論文風險函數以 STDK interpolated series 近似未知站真值 |
| Q 模型共享方式 | 500 個訓練站共享一個 QConvLSTM | Weather2K 計算與泛化改編。論文說明 forecast every site separately |
| Test150 產生方式 | 將 5→5 模型分 block 套用到 150 步 | Weather2K 改編。論文公開實驗只直接評估最後 5 步 |
| 三情境 | 同一 fitted STDK/Q 模型評估三種 mask/位置/時間 | 本研究公平比較設計 |

### Q 訓練目標的關鍵差異

論文先定義 STDK 於未知位置形成的插值序列 `X^NN`，再說未知位置沒有真實的 `Z` 可供訓練，所以用 `X^NN` 近似風險函數中的真值。QConvLSTM 同理使用局部 STDK 插值網格 `X^NN_CONV`。

目前程式則以 STDK local grids 作為輸入，但 `qconv_train_targets` 與 `val_targets` 是 train500 的 Weather2K 真值。這是合理且可清楚說明的 supervised adaptation，卻不是論文該段方法的逐字實作。它會讓 Q 學習從 STDK 網格修正到實際觀測，而非只模仿 STDK 插值序列。

## Weather2K 調參實際範圍

最終相容調參資料夾：`weather2k_exp/air_temperature/qconvlstm_obs100_ema_tuning_20260912/`

選參資料與規則：

- seeds 41、42；
- 只用 train500 的 chronological Val150；
- 指標為兩個 seeds 的 standardized q50 validation RMSE 平均；
- Test150 與 held-out100 真值均未用於選參；
- Stage 1：grid size `{5,8,11}` × radius `{0.1,0.2,0.3}`，共 9 組；
- Stage 2：filters `{32,64}` × weight decay `{0,1e-5}`，共 4 組；
- 最終選到 grid 11、radius 0.1、64 filters、weight decay 1e-5；
- 最終 mean/std standardized q50 Val150 RMSE：0.67220296 / 0.05268962。

固定參數為 lr 1e-4、batch 64、最多 40 epochs、patience 10、lookback 5、horizon 5、`github_3block`。其中 lr/batch/epochs/patience 源自 2026-09-04 的早期 Weather2K seed41 四組試驗；在 2026-09-12 最終 obs100 + EMA protocol 中是固定值，沒有重新搜尋。lookback/horizon/profile 依研究流程固定，也沒有搜尋。

因此「已依 Weather2K 調參」是正確的，但應限定為：

> 已用 Weather2K Val150、seeds 41–42 調整 QConvLSTM 的 grid/radius、filters/weight decay，並沿用早期 Weather2K 試驗選出的 learning rate 等參數；不是所有 STDK 與 QConvLSTM 參數的完整重新搜尋。

q05/q95 也沒有各自選參；formal experiment 沿用由 q50 validation 選出的設定。

## 尚無法宣稱精確重現的原因

作者公開資料缺少 Table 2 完整 QConvLSTM quantile 程式、`training_data.csv`、`model_real.h5`、確切 lambda、random seed、boundary handling 與部分架構橋接細節。專案中的 reproduction 程式已將這些標為 assumptions。因此現有 STDK+Q 應視為可追溯的 Weather2K adaptation，而不是 exact author-code reproduction。

## 對應檔案

- 純 STDK：`weather2k_exp/2K_STDK_500train_100test.py`
- STDK+QConvLSTM：`weather2k_exp/2K_STDK_QConvLSTM.py`
- QConvLSTM 重建元件：`STDK_QConvLSTM_reproduction_results/code/STDK_QConvLSTM_reproduction.py`
- 最終參數：`weather2k_exp/2K_best_stdk_qconvlstm_params_500to100.json`
- 最終調參紀錄：`weather2k_exp/air_temperature/qconvlstm_obs100_ema_tuning_20260912/`
- STDK 對齊稽核：`docs/STDK_SPATIAL_ADAPTER_ALIGNMENT_20260913.md`
- QConvLSTM 重建限制：`STDK_QConvLSTM_reproduction_results/docs/QCONVLSTM_REPRODUCTION.md`
