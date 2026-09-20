# Scenario summary: spatiotemp_100x150

> **重要：** 下表的 `STDK+SharedLSTM` 是自訂 Weather2K deterministic shared-LSTM baseline，不符合論文的 QLSTM／QConvLSTM probabilistic forecasting 流程，不能標示為 paper reproduction。其 LSTM 使用 MSE point forecast，沒有 quantile loss、local spatial grids 或 prediction intervals；SVGP 與 DLinear+FRK 列不受此註記影響。

| scenario | target | space_split | n_train_space | n_unobs_space | n_target_space | model | alpha_obs | lambda_unobs | row_type | seed | RMSE | MSE | MAE | R2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| spatiotemp_100x150 | Target_ST100x150 | 100/400/100 | 100 | 400 | 100 | SVGP |  |  | seed | 41 | 3.255117 | 10.595787 | 2.551797 | 0.630973 |
| spatiotemp_100x150 | Target_ST100x150 | 100/400/100 | 100 | 400 | 100 | SVGP |  |  | seed | 42 | 3.618696 | 13.094957 | 2.847178 | 0.575805 |
| spatiotemp_100x150 | Target_ST100x150 | 100/400/100 | 100 | 400 | 100 | SVGP |  |  | seed | 43 | 3.396999 | 11.539604 | 2.630534 | 0.633765 |
| spatiotemp_100x150 | Target_ST100x150 | 100/400/100 | 100 | 400 | 100 | SVGP |  |  | seed | 44 | 3.567776 | 12.729025 | 2.789110 | 0.552826 |
| spatiotemp_100x150 | Target_ST100x150 | 100/400/100 | 100 | 400 | 100 | SVGP |  |  | seed | 45 | 3.334638 | 11.119809 | 2.662321 | 0.610741 |
| spatiotemp_100x150 | Target_ST100x150 | 100/400/100 | 100 | 400 | 100 | SVGP |  |  | mean_std | mean +/- std | 3.434645 +/- 0.138020 | 11.815837 +/- 0.950720 | 2.696188 +/- 0.107460 | 0.600822 +/- 0.031693 |
| spatiotemp_100x150 | Target_ST100x150 | 100/400/100 | 100 | 400 | 100 | STDK+SharedLSTM |  |  | seed | 41 | 5.147553 | 26.497297 | 4.123987 | 0.077160 |
| spatiotemp_100x150 | Target_ST100x150 | 100/400/100 | 100 | 400 | 100 | STDK+SharedLSTM |  |  | seed | 42 | 5.863966 | 34.386097 | 4.825294 | -0.113896 |
| spatiotemp_100x150 | Target_ST100x150 | 100/400/100 | 100 | 400 | 100 | STDK+SharedLSTM |  |  | seed | 43 | 5.109985 | 26.111946 | 4.063796 | 0.171279 |
| spatiotemp_100x150 | Target_ST100x150 | 100/400/100 | 100 | 400 | 100 | STDK+SharedLSTM |  |  | seed | 44 | 4.810845 | 23.144228 | 3.755657 | 0.186936 |
| spatiotemp_100x150 | Target_ST100x150 | 100/400/100 | 100 | 400 | 100 | STDK+SharedLSTM |  |  | seed | 45 | 5.227371 | 27.325413 | 4.186123 | 0.043450 |
| spatiotemp_100x150 | Target_ST100x150 | 100/400/100 | 100 | 400 | 100 | STDK+SharedLSTM |  |  | mean_std | mean +/- std | 5.231944 +/- 0.346062 | 27.492996 +/- 3.724337 | 4.190971 +/- 0.350014 | 0.072986 +/- 0.108130 |
| spatiotemp_100x150 | Target_ST100x150 | 100/400/100 | 100 | 400 | 100 | DLINEAR + differentiable_FRK | 2.0 | 1.5 | seed | 41 | 3.735412 | 13.953302 | 2.920093 | 0.514038 |
| spatiotemp_100x150 | Target_ST100x150 | 100/400/100 | 100 | 400 | 100 | DLINEAR + differentiable_FRK | 2.0 | 1.5 | seed | 42 | 4.321001 | 18.671050 | 3.435287 | 0.395174 |
| spatiotemp_100x150 | Target_ST100x150 | 100/400/100 | 100 | 400 | 100 | DLINEAR + differentiable_FRK | 2.0 | 1.5 | seed | 43 | 4.041931 | 16.337210 | 3.025545 | 0.481502 |
| spatiotemp_100x150 | Target_ST100x150 | 100/400/100 | 100 | 400 | 100 | DLINEAR + differentiable_FRK | 2.0 | 1.5 | seed | 44 | 3.956749 | 15.655865 | 3.080821 | 0.450005 |
| spatiotemp_100x150 | Target_ST100x150 | 100/400/100 | 100 | 400 | 100 | DLINEAR + differentiable_FRK | 2.0 | 1.5 | seed | 45 | 4.260412 | 18.151110 | 3.345639 | 0.364604 |
| spatiotemp_100x150 | Target_ST100x150 | 100/400/100 | 100 | 400 | 100 | DLINEAR + differentiable_FRK | 2.0 | 1.5 | mean_std | mean +/- std | 4.063101 +/- 0.211935 | 16.553708 +/- 1.711721 | 3.161477 +/- 0.196029 | 0.441065 +/- 0.054758 |
