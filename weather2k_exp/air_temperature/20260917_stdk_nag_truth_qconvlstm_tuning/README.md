# STDK+QConvLSTM tuning

Status: formal two-stage validation-only tuning

Seeds: [41, 42]

Every trial trains q50 only. Selection uses the mean chronological Val150 q50 RMSE against observed train500-station truth for independently fitted held-out target-location models using nearest train500 proxies. Q validation-only trials do not use Test150 or held-out100 truth. Pure-STDK checkpoint preparation may write its own final metrics, but Q tuning does not use them.

The front STDK is the exact pure-STDK MSE/EMA checkpoint; QConvLSTM tuning does not refit it.

Stage 1 searches grid size and neighbourhood radius. Stage 2 uses the best interface and searches learning rate, filters, and weight decay. The best epoch is selected separately by truth-based Val150 pinball loss in every Q fit.

Best stage: capacity

Best trial: 5

Best mean RMSE: 0.58907961
