# Weather2K STDK + Shared QConvLSTM tuned seed-41 result

This directory archives the formal seed-41 run completed on 2026-09-03 after
the validation-only stage-1 QConvLSTM tuning. It is a Weather2K adaptation of
the public STDK + QConvLSTM workflow, not an exact rerun of the paper's
simulation experiment.

## Status

- Status: completed single-seed diagnostic
- Seed: 41
- Variable: `air_temperature`
- Checkpoints written: no
- Runtime: 27,853.96 seconds (approximately 7 h 44 min)
- Intended use: assess the validation-selected parameters before committing to
  a five-seed run
- This directory is not a five-seed mean/standard-deviation result.

## Data and evaluation protocol

- Weather2K window: final 1,000 three-hourly observations
- Time split: Train700 / chronological Val150 / Test150
- Spatial sample: 600 stations
- Model training: train500 stations
- Strict spatial holdout: held-out100 stations
- Forecast rule: shared 5-to-5 QConvLSTM applied in contiguous blocks
- Targets:
  - `Target_Time150`: train500 at Test150
  - `Target_Space100`: held-out100 over the first 850 times
  - `Target_ST100x150`: held-out100 at Test150
- Held-out100 responses were used only for final scoring, not for training or
  hyperparameter selection.

## Validation-selected parameters

Trial 3 was selected from four candidates using seed-41 train500
chronological Val150 q50 RMSE on the standardized training scale. Test150 and
held-out100 responses were not used during tuning.

| Parameter | Value |
|---|---:|
| QConvLSTM learning rate | 0.0003 |
| Batch size | 32 |
| Maximum epochs | 40 |
| Early-stopping patience | 10 |
| ConvLSTM filters | 64 |
| Tuning q50 validation RMSE | 0.660230 |

The paper-aligned components retained by this adaptation include the
repository STDK feature profile, q05/q50/q95 models, 8x8 local grids with
radius 0.2, the released three-block 64-filter QConvLSTM, pinball loss, and the
median-centred non-crossing quantile constraint.

## Seed-41 result

| Scenario | RMSE | MSE | MAE | R2 | MPIW90 | Coverage90 |
|---|---:|---:|---:|---:|---:|---:|
| Time | 4.843033 | 23.454967 | 3.813621 | 0.226395 | 9.540838 | 67.83% |
| Space | 4.275669 | 18.281343 | 3.303386 | 0.560731 | 10.750368 | 82.90% |
| Spatiotemporal | 4.872827 | 23.744442 | 3.815538 | 0.173035 | 9.487796 | 68.05% |

The nominal interval is 90%. Coverage remains below nominal in every
scenario, especially for time and spatiotemporal extrapolation.

## Comparison with the pre-tuning block5to5 seed-41 baseline

The baseline is archived in the adjacent dated directory
`STDK_QConvLSTM_Weather2K_results_20260901/`.

| Scenario | Baseline RMSE | Tuned RMSE | Relative change |
|---|---:|---:|---:|
| Time | 4.697431 | 4.843033 | 3.10% worse |
| Space | 4.306494 | 4.275669 | 0.72% better |
| Spatiotemporal | 4.657887 | 4.872827 | 4.62% worse |

The tuned parameters therefore did not produce a general seed-41 test
improvement. They slightly improved spatial RMSE, but worsened temporal and
spatiotemporal RMSE and reduced interval coverage. This Test150 observation
must not be used to select another already-tested trial; doing so would leak
test information into model selection.

## Important interpretation limits

1. The standalone Weather2K pure-STDK baseline and the STDK stage inside this
   pipeline use different architectures, losses, bases, and validation
   procedures. Their published scores are not a controlled STDK-versus-
   STDK+QConvLSTM ablation.
2. The shared QConvLSTM directly replaces the STDK point forecast; it is not a
   residual correction with an identity fallback. Adding it therefore does
   not guarantee lower test RMSE.
3. Repeated 5-to-5 evaluation uses STDK-generated grids for each block. Scalar
   QConvLSTM outputs are not fed back as complete future spatial grids.
4. The first spatial block currently constructs history at relative indices
   -5 through -1. This boundary should be corrected before treating the
   spatial result as definitive, although it affects only the first 5 of 850
   evaluated times.
5. The next diagnostic should compare QConvLSTM against the q50 output of the
   exact same fitted internal STDK stage before launching a costly five-seed
   experiment.

## Files

- `shared_qconvlstm_block5to5_seed41.json`: configuration, split indices,
  validation diagnostics, three-scenario metrics, and runtime
- `shared_qconvlstm_block5to5_seed41_time_forecasts.csv`: train500 x Test150
- `shared_qconvlstm_block5to5_seed41_space_forecasts.csv`: held-out100 x first850
- `shared_qconvlstm_block5to5_seed41_forecasts.csv`: held-out100 x Test150
