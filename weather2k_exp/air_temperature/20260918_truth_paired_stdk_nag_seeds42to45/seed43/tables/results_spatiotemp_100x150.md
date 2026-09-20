# Scenario summary: spatiotemp_100x150

| scenario | target | space_split | n_train_space | n_unobs_space | n_target_space | model | alpha_obs | lambda_unobs | row_type | seed | RMSE | MSE | MAE | R2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| spatiotemp_100x150 | Target_ST100x150 | 100/400/100 | 100 | 400 | 100 | STDK |  |  | seed | 43 | 4.971008 | 24.710922 | 4.000491 | 0.215744 |
| spatiotemp_100x150 | Target_ST100x150 | 100/400/100 | 100 | 400 | 100 | STDK |  |  | mean_std | mean +/- std | 4.971008 +/- 0.000000 | 24.710922 +/- 0.000000 | 4.000491 +/- 0.000000 | 0.215744 +/- 0.000000 |
| spatiotemp_100x150 | Target_ST100x150 | 100/400/100 | 100 | 400 | 100 | STDK+NagQConvLSTM-truth(block5to5) |  |  | seed | 43 | 4.851893 | 23.540867 | 3.850941 | 0.252878 |
| spatiotemp_100x150 | Target_ST100x150 | 100/400/100 | 100 | 400 | 100 | STDK+NagQConvLSTM-truth(block5to5) |  |  | mean_std | mean +/- std | 4.851893 | 23.540867 | 3.850941 | 0.252878 |
