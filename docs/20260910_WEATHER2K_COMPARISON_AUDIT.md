# Weather2K 四模型歷史結果與比較流程稽核

> 2026-09-13 STDK 與 pinned spatial-adapter 的最新逐項對齊狀態，見
> [STDK 對齊紀錄](20260913_STDK_SPATIAL_ADAPTER_ALIGNMENT.md)。

後續更新：2026-09-15 起 STDK+Q 新執行改用 q50 pinball checkpoint，並以 quantile-specific fitted STDK series 作為 QConvLSTM 訓練標籤；舊 q50 MSE／測站真值標籤結果保留為歷史版本。詳見
[對齊紀錄](20260910_WEATHER2K_OBS100_ALIGNMENT.md)。下文保留修改前的稽核發現與歷史結果限制。

日期：2026-09-10。範圍：本機已同步的 `weather2k_exp/air_temperature` JSON、README、目前四模型及 tuner 原始碼。
沒有登入舊裝置檢查未同步檔案；歷史檔案通常未記錄 hostname、GPU、完整依賴和程式 hash，
因此「已有五-seed 結果」可以確認，不能僅憑資料夾日期斷言每個檔案在哪台 GPU 產生。
目前原始碼不等於所有歷史執行時的完整快照；下列程式層級發現須與歷史 JSON 證據分開解讀。

## 已完成的 seeds 41–45

| 歷史資料夾 | 模型與範圍 | 可否當目前四模型正式五-seed 結果 |
|---|---|---|
| `20260803_results_three_models_svgp_tuned_train500_test100` | SVGP、純 STDK、DLinear+FRK，三情境均有 41–45 | 有三個 baseline 的歷史完整結果；DLinear 是舊 alpha=1/lambda=1 參數 |
| `20260817_results_stdk_shared_lstm` | SVGP 三情境；DLinear+FRK alpha=2/lambda=1.5 與 STDK+SharedLSTM 有時間、時空五-seed | SharedLSTM 不是純 STDK 或 QConvLSTM；DLinear 此資料夾缺空間情境 |
| `20260830_STDK_QConvLSTM_Weather2K_495to5_pilot` | QConvLSTM 五-seed、q05/q50/q95，但僅 100 站、最後 500 時點、495→5 | 不可；測站、時間、評分目標都不同，也不是目前 Spatial-adapter location-specific 版本 |
| `20260831_four_models_seed41` | 四模型三情境，只有 seed 41 | 不可當五-seed；QConvLSTM 還是早期 direct5to150 |
| `20260905_STDK_QConvLSTM_Weather2K_spatial_adapter_block5to5_seed41` | 目前這類 Spatial-adapter + direct QConvLSTM 的三情境 seed 41 | 沒有此版本 seeds 42–45 正式結果 |
| `20260906_STDK_Residual_QConvLSTM_Weather2K_block5to5_seed41` | Residual 三情境 seed 41 | 獨立 ablation，不可混入 direct 結果 |
| `20260909_qconvlstm_retune_rtx2000` | 新環境兩階段 q50 validation，seeds 41、42 | 不是正式 Test150；沒有新參數的五-seed 正式結果 |

更早的 6–7 月資料夾亦有五-seed 輸出，但一些版本 STDK 僅用 obs100 訓練；不可只憑同樣 seeds 混用。

## 切分：已核對的結果

直接解析 8/03 三模型 9 份主要 metrics JSON，共 45 個逐 seed entries：
SVGP 與 STDK 的 sampled600、train500、heldout100、obs100、unobs400 索引及時間 metadata 相同。
DLinear 使用不同欄位名，映射後同樣完全相同：

- `sample_idx_local` = STDK `sample_idx_train`（obs100）
- `sample_idx_unobs_primary` = `sample_idx_unknown_primary`（400）
- `sample_idx_unobs_eval` = `sample_idx_unknown_eval`（heldout100）
- obs100 與 unobs400 的聯集為 train500。

8/17 DLinear 的兩個五-seed 情境、8/31 DLinear 的三個 seed41 情境亦與上述索引一致。
9/05 QConvLSTM seed41，以及新重跑 interface trial0 seeds41/42 的 `sampled_global/train_local/heldout_local`
也與 baseline 記錄一致。尚不存在的 QConvLSTM seeds43–45 正式結果當然無法逐檔核對。

共同評分範圍是：最後 1000 時點，Train700 / Val150 / Test150；
時間 train500×Test150（75000 點），空間 heldout100×前850（85000 點），時空 heldout100×Test150（15000 點）。
這是 JSON 索引一致性的確認，不能替代不同歷史執行時資料 hash 的完整證明。

## 評分與可用資訊

8/03 README 明確記錄原始氣溫尺度的 RMSE/MSE/MAE 與 flatten global R²；目前四模型的主要目標評分也採此定義。
**8/01 `20260801_results_svgp_tuned_stationwise_r2` 使用 stationwise average R²，不能混入 global R² 表格。**
目前 baseline 與 QConvLSTM 五-seed summary 使用 ddof=0；495→5 pilot 用 ddof=1，表格必須明確統一。
q50 是中位數輸出，MSE 訓練的 baseline 是均值目標，RMSE 可以比較，但須說明估計目標不同。
歷史三模型主要輸出沒有一致的 q05/q95 區間，因此不能視為已完成四模型 coverage/MPIW 比較。

目前程式的預測資訊流程：

- STDK/SVGP：Train700 的 train500 responses 訓練，Val150 選 checkpoint，以目標座標與時間預測。
- DLinear+FRK：obs100 時序作輸入、unobs400 透過空間 loss 監督，Val150 在 train500 評估；
  從 Train700 歷史遞迴預測 300 點，取後150作 Test150，沒有在測試起點換入 Val150 真值。
- QConvLSTM：5→5 各 block 的歷史是已 fitted STDK 產生的 grids，不是持續讀取新的測站真值；
  空間最前5點使用 STDK warm-up。分段形式本身不表示獲得更多測試期真值。

DLinear 空間前700的 obs100 使用真實訓練值作空間推估，而 QConvLSTM 使用 STDK 估計場。
可用訓練資料範圍相同，但輸入表示與如何使用訓練資料不同；不要稱為每個模型的數值輸入完全相同。

## 必須更正：純 STDK 與組合前段並非完全相同訓練流程

目前 `2K_STDK_500train_100test.py` 先把 `y_train_full` 設為 obs100，
再以 `train_y_matrix[:, train_time_idx]` 的均值/標準差縮放，實際模型 training targets 仍包含 train500。
`2K_SVGP_500train_100test.py` 同樣由 obs100 計算縮放；DLinear 也由 obs100 Train700 計算縮放。

但 `2K_STDK_QConvLSTM.py:run_seed` 使用 **train500×Train700** 計算均值/標準差。
這不是測試洩漏，卻是不同的前處理，會影響模型訓練與標準化 validation 數值。
前段網路架構、LR、epochs、basis 等主要 config 相同，不代表 fitted STDK 可以視為相同模型。
QConvLSTM 前段還另外設定初始化及 DataLoader seed，不能僅憑相同 seed 號碼保證訓練軌跡相同。

因此先前「與純 STDK 完全相同」的說法過強，應改為「採相同 Spatial-adapter 架構與主要超參數」。
最可靠的 added-value 比較是在每個正式 QConvLSTM run 中，直接對該次 fitted q50 STDK 做 baseline 評分，
或讓獨立 STDK 使用相同的前處理、訓練函式及選 checkpoint 規則。尚未在本次稽核修改程式。

## Validation 選參：哪些有證據

| 模型 | 現有證據 | 限制 |
|---|---|---|
| SVGP | `svgp_tuning_current`，seeds41/42 平均 best Val RMSE（scaled）；tuner 讀 training_summary | 與其他模型 raw-scale 平均不是相同的跨seed加權；搜尋預算不同 |
| DLinear+FRK | `dlinear_frk_tuning_current`，30-trial 聯合 Optuna，以 seeds41/42 的 train500 Val RMSE（raw）選參 | 8/03 五-seed使用更早 DLinear-only 參數及 alpha=1/lambda=1，不能冠上後來聯合調參標籤 |
| 純 STDK | 固定 Spatial-adapter baseline 設定，Val MSE early stopping | 沒有與其他模型同等的超參數搜尋；固定基準可接受，但需明示 |
| QConvLSTM | 新裝置兩階段 26 runs，僅 q50 validation；trial 以 scaled RMSE，direct epoch 以 pinball 選擇 | 追加搜尋較多，非相同調參預算；前次另一組 Val RMSE 更低，需固定並交代最終選擇規則 |

另外目前 SVGP/DLinear tuner 呼叫的正式腳本仍會產生 Test 指標；tuner 的 objective 只讀 validation 欄位，
未見以 Test 選最佳 trial 的程式，但這不等於「調參時完全沒計算 Test」。
QConvLSTM 則有明確 `--validation-only --q50-only` 路徑。
也無法單憑程式保證歷史人工決策完全未受已看到的 Test 分數影響，報告應如實交代探索歷史。

## 下一步

1. 保留歷史結果，沒有必要把所有五-seed 都當作未做過。
2. 正式新增 QConvLSTM 五-seed 前，先處理「純 STDK 前段一致」：優先保存同次 fitted STDK 的 baseline 分數。
3. 明訂最後參數與 validation 指標、調參預算、原始尺度 global R² 和 std 定義。
4. 若要用最新 DLinear 聯合參數做三情境五-seed，目前可見 8/17 缺空間五-seed，8/31 是 single seed；需補齊適用版本。
5. 新 QConvLSTM seeds41–45 正式結果仍需跑；不能用 495→5 pilot 或 SharedLSTM 結果代替。
6. 使用新環境作正式比較時，baseline 環境差異應記錄；若研究重点是 STDK 加 Q 的增益，同環境同次 fitted STDK 尤其重要。

本次僅讀取、核對及寫入稽核文件，未改模型、既有結果、參數、Git index 或啟動訓練。
