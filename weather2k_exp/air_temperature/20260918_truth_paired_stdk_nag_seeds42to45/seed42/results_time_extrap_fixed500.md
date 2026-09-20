# Scenario summary: time_extrap_fixed500

| scenario | target | space_split | n_train_space | n_unobs_space | n_target_space | model | alpha_obs | lambda_unobs | row_type | seed | RMSE | MSE | MAE | R2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| time_extrap_fixed500 | Target_Time150 | 100/400/100 | 100 | 400 | 100 | STDK |  |  | seed | 42 | 4.379718 | 19.181931 | 3.500806 | 0.351851 |
| time_extrap_fixed500 | Target_Time150 | 100/400/100 | 100 | 400 | 100 | STDK |  |  | mean_std | mean +/- std | 4.379718 +/- 0.000000 | 19.181931 +/- 0.000000 | 3.500806 +/- 0.000000 | 0.351851 +/- 0.000000 |
| time_extrap_fixed500 | Target_Time150 | 100/400/100 | 100 | 400 | 100 | STDK+NagQConvLSTM-truth(block5to5) |  |  | seed | 42 | 4.488406 | 20.145786 | 3.557790 | 0.319283 |
| time_extrap_fixed500 | Target_Time150 | 100/400/100 | 100 | 400 | 100 | STDK+NagQConvLSTM-truth(block5to5) |  |  | mean_std | mean +/- std | 4.488406 | 20.145786 | 3.557790 | 0.319283 |
