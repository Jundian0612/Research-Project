# Scenario summary: time_extrap_fixed500

| scenario | target | space_split | n_train_space | n_unobs_space | n_target_space | model | alpha_obs | lambda_unobs | row_type | seed | RMSE | MSE | MAE | R2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| time_extrap_fixed500 | Target_Time150 | 100/400/100 | 100 | 400 | 100 | SVGP |  |  | seed | 41 | 3.320592 | 11.026330 | 2.603338 | 0.636323 |
| time_extrap_fixed500 | Target_Time150 | 100/400/100 | 100 | 400 | 100 | SVGP |  |  | mean_std | mean +/- std | 3.320592 +/- 0.000000 | 11.026330 +/- 0.000000 | 2.603338 +/- 0.000000 | 0.636323 +/- 0.000000 |
| time_extrap_fixed500 | Target_Time150 | 100/400/100 | 100 | 400 | 100 | STDK |  |  | seed | 41 | 4.313021 | 18.602154 | 3.508868 | 0.386453 |
| time_extrap_fixed500 | Target_Time150 | 100/400/100 | 100 | 400 | 100 | STDK |  |  | mean_std | mean +/- std | 4.313021 +/- 0.000000 | 18.602154 +/- 0.000000 | 3.508868 +/- 0.000000 | 0.386453 +/- 0.000000 |
| time_extrap_fixed500 | Target_Time150 | 100/400/100 | 100 | 400 | 100 | STDK+SharedQConvLSTM |  |  | seed | 41 | 6.035994 | 36.433228 | 4.768288 | -0.201662 |
| time_extrap_fixed500 | Target_Time150 | 100/400/100 | 100 | 400 | 100 | STDK+SharedQConvLSTM |  |  | mean_std | mean +/- std | 6.035994 | 36.433228 | 4.768288 | -0.201662 |
| time_extrap_fixed500 | Target_Time150 | 100/400/100 | 100 | 400 | 100 | DLINEAR + differentiable_FRK | 2.0 | 1.5 | seed | 41 | 3.791848 | 14.378115 | 2.937695 | 0.525773 |
| time_extrap_fixed500 | Target_Time150 | 100/400/100 | 100 | 400 | 100 | DLINEAR + differentiable_FRK | 2.0 | 1.5 | mean_std | mean +/- std | 3.791848 +/- 0.000000 | 14.378115 +/- 0.000000 | 2.937695 +/- 0.000000 | 0.525773 +/- 0.000000 |
