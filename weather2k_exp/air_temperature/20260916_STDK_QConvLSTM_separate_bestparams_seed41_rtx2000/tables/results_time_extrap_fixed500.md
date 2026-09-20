# Scenario summary: time_extrap_fixed500

| scenario | target | space_split | n_train_space | n_unobs_space | n_target_space | model | alpha_obs | lambda_unobs | row_type | seed | RMSE | MSE | MAE | R2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| time_extrap_fixed500 | Target_Time150 | 100/400/100 | 100 | 400 | 100 | STDK |  |  | seed | 41 | 4.293592 | 18.434929 | 3.377500 | 0.391968 |
| time_extrap_fixed500 | Target_Time150 | 100/400/100 | 100 | 400 | 100 | STDK |  |  | mean_std | mean +/- std | 4.293592 +/- 0.000000 | 18.434929 +/- 0.000000 | 3.377500 +/- 0.000000 | 0.391968 +/- 0.000000 |
| time_extrap_fixed500 | Target_Time150 | 100/400/100 | 100 | 400 | 100 | STDK+LocationSpecificQConvLSTM(block5to5) |  |  | seed | 41 | 4.831717 | 23.345486 | 3.797750 | 0.230006 |
| time_extrap_fixed500 | Target_Time150 | 100/400/100 | 100 | 400 | 100 | STDK+LocationSpecificQConvLSTM(block5to5) |  |  | mean_std | mean +/- std | 4.831717 | 23.345486 | 3.797750 | 0.230006 |
