# 舊 train500 標準化規則的未完成 seed 41

這是 `20260909_qconvlstm_retune_rtx2000` 調參後準備的正式 seed 41，
使用 train500 × Train700 計算 mean/std，參數為 grid=11、radius=0.1、filters=64、weight decay=0。

執行在完成前停止：`seed41.log` 最後停在產生 local grids，沒有正式 JSON、比較表或逐點預測，
因此不是可報告的正式結果，也不能與後來 obs100 對齊版 seeds 合併。

保留內容：

- `locked_params.json`：當時參數快照。
- `code_and_params_sha256.json`：當時程式與參數指紋，內含搬移前路徑。
- `seed41.log`：未完成執行紀錄；log 受 `.gitignore` 忽略。

本資料夾只供追溯。不要從此處恢復或重新執行；目前正式流程位於
`20260911_STDK_QConvLSTM_formal_obs100_rtx2000`。
