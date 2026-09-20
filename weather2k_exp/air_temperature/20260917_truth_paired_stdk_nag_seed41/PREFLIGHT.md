# Seed 41 paired STDK and QConvLSTM preflight

The selected Q parameters are copied into `locked_q_params.json`; the launcher
checks its SHA-256 and the source-file snapshot from the completed tuning run
before each scenario. Pure STDK is fitted separately for each scenario, and
the Q stage must load that scenario's exact `model_best.pt`. The Q loader
checks the checkpoint SHA-256, station and time splits, normalization, model
recipe, source hashes, and a prediction probe. `verify_seed41.py` repeats the
checkpoint SHA and pure-STDK RMSE comparison after each formal Q run.

Test-truth selection audit: Q training labels use only supervised train500
Train700 temperatures. Q checkpoint and early stopping use those stations'
Val150 temperatures; the selected learning rate and other settings came from
the completed Val150-only tuning run. For held-out100 targets, Q uses the
nearest train500 station's labels. The final target truth is accessed only
after the Q predictions are complete, for metrics and forecast CSVs.

An independent reduced-data perturbation check changed all held-out target
truth and the supervised stations' Test block while keeping Train and Val
unchanged. Q's q50 per-location and aggregate validation reports remained
identical. A three-scenario reduced run also confirmed that Q's loaded
checkpoint SHA and fitted-STDK RMSE match pure STDK for each scenario.

`run_seed41.sh` runs the three full scenarios in the order ST100×150,
Space100, Time150, validating each result before continuing. `exit_code` is
written at completion; `0` indicates all three passed. No test result is
used to change Q hyperparameters during this run.
