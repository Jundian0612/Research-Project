# Standalone STDK reproduction check

> **Historical result (pre-EMA).** These seed 41 outputs were generated on
> 2026-09-11 before the Weather2K standalone STDK loop enabled the pinned
> spatial-adapter repository's exponential moving average (EMA). They remain
> useful for traceability, but must not be combined with post-2026-09-12 runs.

The current standalone STDK uses batch-average validation MSE and selects the
best checkpoint from EMA parameters. A current-protocol comparison requires a
new run.

## 整理後目錄

- `logs/`：執行紀錄
- `metrics/`：JSON 評估結果
- `tables/`：彙整表格

本次整理只調整檔案位置，沒有刪除或重新計算結果。
