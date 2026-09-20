# STDK+QConvLSTM obs100 EMA tuning

Status: complete. The two-stage validation-only tuning finished on 2026-09-12.

## Selected configuration

- tuning seeds: 41, 42
- selection metric: mean standardized q50 Val150 RMSE
- mean / standard deviation: 0.67220296 / 0.05268962
- grid size: 11
- neighbourhood radius: 0.1
- Conv filters: 64
- QConvLSTM weight decay: 1e-5
- QConvLSTM learning rate: 1e-4
- batch size: 64
- maximum epochs / patience: 40 / 10
- front STDK validation: batch-average
- front STDK EMA: enabled

Test150 and held-out100 truth were not read or used during selection.

## Directory layout

- `qconvlstm_tuning_best_params.json`: final parameters selected after both stages
- `stage1_interface/`: nine grid-size/radius trials, per-seed validation reports, summary table and stage winner
- `stage2_capacity/`: four filters/weight-decay trials, per-seed validation reports, summary table and stage winner
- `logs/run.log`: complete terminal output
- `metadata/code.sha256`: hashes captured before this tuning run

The root-level `weather2k_exp/2K_best_stdk_qconvlstm_params_500to100.json`
is an identical copy of the selected formal parameters. Existing pre-EMA formal
runs must not be mixed with this protocol.
