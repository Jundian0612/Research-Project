# DLinear+FRK hyperparameter tuning

Optuna jointly searches DLinear model parameters and the FRK-related parameters
used by the current differentiable spatial pipeline. Selection uses only the
mean best Val150 RMSE over supervised stations 4+5 and seeds [41, 42].
Held-out target100 test metrics are not read for model selection.

## Best result

- Trial: `16`
- Mean validation RMSE: `3.449546`
- Formal parameter file: `2K_best_dlinear_and_frk_params_500to100.json`

## Search protocol

- Space split: 100 observed / 400 supervised-unobserved / 100 held-out target
- Time split: Train700 / Val150 / Test150
- Objective: mean best validation 4+5 RMSE on the original temperature scale
- Optuna trials: 30
