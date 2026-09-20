# Scenario summary: space_extrap_fixed850

| scenario | target | space_split | n_train_space | n_unobs_space | n_target_space | model | alpha_obs | lambda_unobs | row_type | seed | RMSE | MSE | MAE | R2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| space_extrap_fixed850 | Target_Space100 | 100/400/100 | 100 | 400 | 100 | STDK |  |  | seed | 42 | 4.662348 | 21.737488 | 3.677712 | 0.491042 |
| space_extrap_fixed850 | Target_Space100 | 100/400/100 | 100 | 400 | 100 | STDK |  |  | mean_std | mean +/- std | 4.662348 +/- 0.000000 | 21.737488 +/- 0.000000 | 3.677712 +/- 0.000000 | 0.491042 +/- 0.000000 |
| space_extrap_fixed850 | Target_Space100 | 100/400/100 | 100 | 400 | 100 | STDK+NagQConvLSTM-truth(block5to5) |  |  | seed | 42 | 4.601807 | 21.176632 | 3.547852 | 0.504174 |
| space_extrap_fixed850 | Target_Space100 | 100/400/100 | 100 | 400 | 100 | STDK+NagQConvLSTM-truth(block5to5) |  |  | mean_std | mean +/- std | 4.601807 | 21.176632 | 3.547852 | 0.504174 |
