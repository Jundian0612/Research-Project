# STDK+QConvLSTM tuning

Status: formal two-stage validation-only tuning

Seeds: [41, 42]

Every trial trains q50 only. Selection uses the mean chronological Val150 q50 RMSE on train500 stations. Test150 and held-out100 truth are not computed or read.

Stage 1 searches grid size and neighbourhood radius. Stage 2 uses the best interface and searches filters and weight decay.

Best stage: capacity

Best trial: 2

Best mean RMSE: 0.67661150
