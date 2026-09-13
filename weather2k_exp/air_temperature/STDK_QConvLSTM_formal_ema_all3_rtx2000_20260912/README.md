# STDK+QConvLSTM formal EMA seed 41

Status: prepared; seed 41 has not yet been launched.

This run locks the validation-selected parameters from
`qconvlstm_obs100_ema_tuning_20260912`. The front Spatial-adapter STDK uses
EMA and batch-average validation. One fitted STDK+QConvLSTM instance evaluates
`Target_Time150`, `Target_Space100`, and `Target_ST100x150`. The same fitted
q05/q50/q95 STDK models are also reported as the paired STDK baseline.

Selection used seeds 41 and 42 Val150 only. Test150 and held-out100 truth were
not used for parameter selection.
