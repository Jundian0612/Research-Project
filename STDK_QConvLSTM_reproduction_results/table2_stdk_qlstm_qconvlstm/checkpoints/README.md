# QLSTM checkpoints

- `qlstm_locations/current/`: the selected quantile-specific q05/q50/q95
  fitted-STDK input and paper Eq. (7) run (100 locations x seeds 41--45).
- `qlstm_locations/archive/linear_heads/`: superseded independent linear-head
  QLSTM run.
- `qlstm_locations/archive/paper_eq7_shared_input/`: superseded and incomplete
  Eq. (7) run that did not use quantile-specific fitted-STDK inputs.

Only `current/` is eligible for resume by
`code/compare_stdk_qlstm_qconvlstm.py`.
