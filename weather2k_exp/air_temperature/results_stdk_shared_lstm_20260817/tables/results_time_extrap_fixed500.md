# Scenario summary: time_extrap_fixed500

> **重要：** 下表的 `STDK+SharedLSTM` 是自訂 Weather2K deterministic shared-LSTM baseline，不符合論文的 QLSTM／QConvLSTM probabilistic forecasting 流程，不能標示為 paper reproduction。其 LSTM 使用 MSE point forecast，沒有 quantile loss、local spatial grids 或 prediction intervals；SVGP 與 DLinear+FRK 列不受此註記影響。

| scenario | target | space_split | n_train_space | n_unobs_space | n_target_space | model | alpha_obs | lambda_unobs | row_type | seed | RMSE | MSE | MAE | R2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| time_extrap_fixed500 | Target_Time150 | 100/400/100 | 100 | 400 | 100 | SVGP |  |  | seed | 41 | 3.320592 | 11.026330 | 2.603338 | 0.636323 |
| time_extrap_fixed500 | Target_Time150 | 100/400/100 | 100 | 400 | 100 | SVGP |  |  | seed | 42 | 3.347205 | 11.203783 | 2.659564 | 0.621429 |
| time_extrap_fixed500 | Target_Time150 | 100/400/100 | 100 | 400 | 100 | SVGP |  |  | seed | 43 | 3.612039 | 13.046826 | 2.772180 | 0.568870 |
| time_extrap_fixed500 | Target_Time150 | 100/400/100 | 100 | 400 | 100 | SVGP |  |  | seed | 44 | 3.359349 | 11.285225 | 2.625392 | 0.611529 |
| time_extrap_fixed500 | Target_Time150 | 100/400/100 | 100 | 400 | 100 | SVGP |  |  | seed | 45 | 3.347595 | 11.206392 | 2.693164 | 0.619785 |
| time_extrap_fixed500 | Target_Time150 | 100/400/100 | 100 | 400 | 100 | SVGP |  |  | mean_std | mean +/- std | 3.397356 +/- 0.108090 | 11.553711 +/- 0.751357 | 2.670727 +/- 0.059190 | 0.611587 +/- 0.022806 |
| time_extrap_fixed500 | Target_Time150 | 100/400/100 | 100 | 400 | 100 | STDK+SharedLSTM |  |  | seed | 41 | 5.265092 | 27.721195 | 4.216572 | 0.085683 |
| time_extrap_fixed500 | Target_Time150 | 100/400/100 | 100 | 400 | 100 | STDK+SharedLSTM |  |  | seed | 42 | 5.410460 | 29.273075 | 4.412415 | 0.010876 |
| time_extrap_fixed500 | Target_Time150 | 100/400/100 | 100 | 400 | 100 | STDK+SharedLSTM |  |  | seed | 43 | 5.145081 | 26.471863 | 4.081404 | 0.125242 |
| time_extrap_fixed500 | Target_Time150 | 100/400/100 | 100 | 400 | 100 | STDK+SharedLSTM |  |  | seed | 44 | 4.551763 | 20.718542 | 3.655849 | 0.286806 |
| time_extrap_fixed500 | Target_Time150 | 100/400/100 | 100 | 400 | 100 | STDK+SharedLSTM |  |  | seed | 45 | 5.058503 | 25.588449 | 4.035625 | 0.131825 |
| time_extrap_fixed500 | Target_Time150 | 100/400/100 | 100 | 400 | 100 | STDK+SharedLSTM |  |  | mean_std | mean +/- std | 5.086180 +/- 0.292234 | 25.954625 +/- 2.896586 | 4.080373 +/- 0.249410 | 0.128087 +/- 0.090278 |
| time_extrap_fixed500 | Target_Time150 | 100/400/100 | 100 | 400 | 100 | DLINEAR + differentiable_FRK | 2.0 | 1.5 | seed | 41 | 3.791848 | 14.378115 | 2.937695 | 0.525773 |
| time_extrap_fixed500 | Target_Time150 | 100/400/100 | 100 | 400 | 100 | DLINEAR + differentiable_FRK | 2.0 | 1.5 | seed | 42 | 4.012013 | 16.096249 | 3.124288 | 0.456115 |
| time_extrap_fixed500 | Target_Time150 | 100/400/100 | 100 | 400 | 100 | DLINEAR + differentiable_FRK | 2.0 | 1.5 | seed | 43 | 4.500206 | 20.251857 | 3.182866 | 0.330781 |
| time_extrap_fixed500 | Target_Time150 | 100/400/100 | 100 | 400 | 100 | DLINEAR + differentiable_FRK | 2.0 | 1.5 | seed | 44 | 4.087299 | 16.706010 | 3.179756 | 0.424930 |
| time_extrap_fixed500 | Target_Time150 | 100/400/100 | 100 | 400 | 100 | DLINEAR + differentiable_FRK | 2.0 | 1.5 | seed | 45 | 4.109241 | 16.885859 | 3.175189 | 0.427090 |
| time_extrap_fixed500 | Target_Time150 | 100/400/100 | 100 | 400 | 100 | DLINEAR + differentiable_FRK | 2.0 | 1.5 | mean_std | mean +/- std | 4.100121 +/- 0.229395 | 16.863618 +/- 1.911653 | 3.119959 +/- 0.093619 | 0.432938 +/- 0.062753 |
