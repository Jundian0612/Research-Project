# Scenario summary: time_extrap_fixed500

| scenario | target | space_split | n_train_space | n_unobs_space | n_target_space | model | alpha_obs | lambda_unobs | row_type | seed | RMSE | MSE | MAE | R2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| time_extrap_fixed500 | Target_Time150 | 100/400/100 | 100 | 400 | 100 | SVGP |  |  | seed | 41 | 3.248717 | 10.554163 | 2.573845 | 0.651896 |
| time_extrap_fixed500 | Target_Time150 | 100/400/100 | 100 | 400 | 100 | SVGP |  |  | mean_std | mean +/- std | 3.248717 +/- 0.000000 | 10.554163 +/- 0.000000 | 2.573845 +/- 0.000000 | 0.651896 +/- 0.000000 |
| time_extrap_fixed500 | Target_Time150 | 100/400/100 | 100 | 400 | 100 | STDK |  |  | seed | 41 | 4.293592 | 18.434929 | 3.377500 | 0.391968 |
| time_extrap_fixed500 | Target_Time150 | 100/400/100 | 100 | 400 | 100 | STDK |  |  | mean_std | mean +/- std | 4.293592 +/- 0.000000 | 18.434929 +/- 0.000000 | 3.377500 +/- 0.000000 | 0.391968 +/- 0.000000 |
| time_extrap_fixed500 | Target_Time150 | 100/400/100 | 100 | 400 | 100 | DLINEAR + differentiable_FRK | 2.0 | 1.5 | seed | 41 | 3.791848 | 14.378115 | 2.937695 | 0.525773 |
| time_extrap_fixed500 | Target_Time150 | 100/400/100 | 100 | 400 | 100 | DLINEAR + differentiable_FRK | 2.0 | 1.5 | mean_std | mean +/- std | 3.791848 +/- 0.000000 | 14.378115 +/- 0.000000 | 2.937695 +/- 0.000000 | 0.525773 +/- 0.000000 |
