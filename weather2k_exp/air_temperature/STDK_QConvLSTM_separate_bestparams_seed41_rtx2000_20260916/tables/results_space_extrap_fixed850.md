# Scenario summary: space_extrap_fixed850

| scenario | target | space_split | n_train_space | n_unobs_space | n_target_space | model | alpha_obs | lambda_unobs | row_type | seed | RMSE | MSE | MAE | R2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| space_extrap_fixed850 | Target_Space100 | 100/400/100 | 100 | 400 | 100 | STDK |  |  | seed | 41 | 5.369811 | 28.834875 | 4.174917 | 0.307148 |
| space_extrap_fixed850 | Target_Space100 | 100/400/100 | 100 | 400 | 100 | STDK |  |  | mean_std | mean +/- std | 5.369811 +/- 0.000000 | 28.834875 +/- 0.000000 | 4.174917 +/- 0.000000 | 0.307148 +/- 0.000000 |
| space_extrap_fixed850 | Target_Space100 | 100/400/100 | 100 | 400 | 100 | STDK+LocationSpecificQConvLSTM(block5to5) |  |  | seed | 41 | 5.691370 | 32.391693 | 4.396953 | 0.221684 |
| space_extrap_fixed850 | Target_Space100 | 100/400/100 | 100 | 400 | 100 | STDK+LocationSpecificQConvLSTM(block5to5) |  |  | mean_std | mean +/- std | 5.691370 | 32.391693 | 4.396953 | 0.221684 |
