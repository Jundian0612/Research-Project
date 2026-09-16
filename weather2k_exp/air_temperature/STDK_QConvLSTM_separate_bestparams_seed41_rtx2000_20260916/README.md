# Pure STDK vs location-specific STDK+QConvLSTM (seed 41)

Status: completed on 2026-09-16. All six independent model/scenario fits
finished successfully on CUDA.

This formal comparison independently fits each model for each Weather2K
evaluation scenario:

- `time_extrap_fixed500` (Time150)
- `space_extrap_fixed850` (Space100)
- `spatiotemp_100x150` (ST100x150)

Both models use seed 41 and the same 500 supervised stations, held-out 100
stations, Train700, Val150 and Test150 definitions. STDK+QConvLSTM reads
`weather2k_exp/2K_best_stdk_qconvlstm_params_500to100.json`, selected from the
location-specific seeds 41/42 Val150 tuning completed on 2026-09-16. Its formal
forecast uses paper-aligned 5-to-5 blocks, direct prediction and independent
q05/q50/q95 QConvLSTM models at every target location.

The experiment is launched through `weather2k_exp/experiments_runner.py` with
SVGP and DLinear+FRK disabled.

## Seed 41 test results

All RMSE values below are in the original air-temperature scale. Lower is
better. `Pure STDK` is the separately trained baseline from
`2K_STDK_500train_100test.py`. `Fitted STDK` is the front model trained within
the STDK+Q run; its predictions are recorded by that run. These are distinct
model instances, although the sampled stations, train500 and held-out100
indices match exactly for every scenario.

| Scenario | Pure STDK RMSE | Fitted STDK RMSE | STDK+Q RMSE | Q minus pure STDK | Q vs pure STDK |
| --- | ---: | ---: | ---: | ---: | ---: |
| Time150 | 4.293592 | 4.359954 | 4.831717 | +0.538125 | +12.53% |
| Space100 | 5.369811 | 4.788476 | 5.691370 | +0.321559 | +5.99% |
| ST100x150 | 4.322475 | 4.413481 | 4.730592 | +0.408117 | +9.44% |

The paired within-run comparison also worsened after Q: +0.471763 RMSE
(+10.82%) for Time150, +0.902894 (+18.86%) for Space100, and +0.317111
(+7.19%) for ST100x150. The corresponding model-specific MSE, MAE, R2,
90% interval width and coverage are in `tables/*_comparison.csv` and
`tables/results_*.csv`. Pure STDK does not report interval metrics in the
runner table.

The q50-only tuning selected parameters by scaled Val150 RMSE for held-out
locations. Formal test predictions train q05/q50/q95 per target location and
evaluate three different targets. A good validation selection therefore does
not imply that adding Q must improve Test150 error. One seed does not establish
the mean effect across seeds; seed 42-45 runs would be needed for that claim.

## Contents

- `metrics/`: raw model JSON reports, including splits and protocol metadata.
- `forecasts/`: per-point pure fitted-STDK and STDK+Q predictions from the Q
  run. The standalone pure STDK run produced metric JSON but no per-point CSV.
- `tables/`: runner summaries, paired within-run comparisons and
  `seed41_comparison.csv` with RMSE deltas.
- `params/`: the formal tuned parameter snapshot.
- `metadata/`: source/parameter SHA256 and the Git commit used for this run.
- `logs/run.log`: full console output, retained locally and ignored by Git.

The run has finished and its files have been moved into this archive structure.
The runner should use a fresh output directory for any future run.
The code/protocol comparison and error diagnostics are in
`docs/WEATHER2K_STDK_QCONVLSTM_SEED41_DIAGNOSTIC_20260916.md`.
