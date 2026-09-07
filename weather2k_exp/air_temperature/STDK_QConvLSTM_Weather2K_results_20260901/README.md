# Weather2K STDK + Shared QConvLSTM results

## Formal configuration

The formal Weather2K version uses shared **5-to-5 block forecasting**:

```text
five historical 8x8 quantile STDK grids
        -> shared QConvLSTM
        -> five future values
        -> repeat contiguous blocks until the evaluation horizon is covered
```

The model keeps the Weather2K comparison protocol fixed:

- last 1,000 hourly observations;
- Train700 / chronological Val150 / Test150;
- 600 sampled stations split into train500 and strictly held-out100;
- one shared QConvLSTM for each of q05, q50, and q95;
- held-out100 responses are unavailable to STDK and QConvLSTM training;
- no model checkpoint is written.

The optional `direct5to150` mode is retained only to preserve the seed-41 pilot
comparison. It is not the formal default.

## Seed 41 pilot comparison

| Scenario | Direct 5-to-150 RMSE | Block 5-to-5 RMSE | Relative RMSE improvement |
|---|---:|---:|---:|
| Time extrapolation (train500, Test150) | 5.718542 | **4.697431** | 17.86% |
| Spatial extrapolation (held-out100, first 850) | 4.538310 | **4.306494** | 5.11% |
| Spatiotemporal extrapolation (held-out100, Test150) | 5.729354 | **4.657887** | 18.70% |

| Scenario | Block RMSE | Block MAE | Block R2 | Block MPIW90 | Block coverage90 |
|---|---:|---:|---:|---:|---:|
| Time | 4.697431 | 3.671449 | 0.272211 | 10.583788 | 72.82% |
| Space | 4.306494 | 3.337283 | 0.554375 | 12.115602 | 87.68% |
| Spatiotemporal | 4.657887 | 3.626551 | 0.244381 | 10.553513 | 74.45% |

Runtime on the local machine was approximately 3 h 10 min for direct 5-to-150
and 3 h 37 min for block 5-to-5. These are single-seed pilot results, not
five-seed mean and standard deviation estimates.

## Files

- `shared_qconvlstm_block5to5_seed41.json`: formal seed-41 configuration,
  validation diagnostics, metrics, split indices, and elapsed time.
- `shared_qconvlstm_block5to5_seed41_time_forecasts.csv`: train500 Test150.
- `shared_qconvlstm_block5to5_seed41_space_forecasts.csv`: held-out100 first 850.
- `shared_qconvlstm_block5to5_seed41_forecasts.csv`: held-out100 Test150.
- `shared_qconvlstm_direct5to150_seed41*`: archived pilot comparison only.

## Interpretation and limitations

Block 5-to-5 improves all three seed-41 point forecasts, but interval coverage
is still below the nominal 90% target for time and spatiotemporal extrapolation.
The shared model is a necessary Weather2K adaptation: a true per-target model
cannot be trained for strictly held-out stations without using their responses.
Therefore this experiment preserves the paper's STDK quantile grids,
quantile-specific QConvLSTM training, pinball loss, constrained quantiles, and
5-to-5 horizon, but it is not an exact per-target reproduction.

Model changes must be selected using Train700/Val150 only. Test150 and
held-out100 metrics are final evaluation data and must not be used for tuning.
