# Weather2K seed 41: pure STDK, fitted STDK and STDK+QConvLSTM

This audit uses the completed, separately fitted Time150, Space100 and
ST100x150 results in
`weather2k_exp/air_temperature/20260916_STDK_QConvLSTM_separate_bestparams_seed41_rtx2000/`.
The local comparison target is the project's author-data reconstruction in
`STDK_QConvLSTM_reproduction_results/code/STDK_QConvLSTM_reproduction.py`.
That reconstruction is itself best-effort: the authors did not release the
complete Table 2 STDK-to-QConvLSTM data bridge.

## Two STDK instances

The independently fitted pure STDK uses
`weather2k_exp/2K_STDK_500train_100test.py`. Its q50 model trains with MSE
and selects the lowest Val150 MSE checkpoint. The STDK+Q script trains a new
q50 STDK instance with q=0.5 pinball loss and selects the lowest Val150
pinball checkpoint. It also fits constrained q05/q95 STDK models. Consequently
its `fitted_stdk_baseline` means predictions from the same front-stage models
that produced Q's grids, rather than predictions from the independent pure
STDK run.

Both Weather2K STDK paths use the pinned Spatial-adapter `STInterpMLP` model,
the same 256/256/128 hidden layers and spatial/temporal bases, AdamW with
learning rate 0.001 and weight decay 0.0001, batch size 512, up to 350 epochs,
batch-mean validation, EMA, and patience 30. For all three scenarios the
recorded sampled600, train500 and held-out100 station IDs are exactly equal.
Both scripts standardize with the obs100 x Train700 mean and population
standard deviation. The independent q50 selected epoch 3 (MSE 0.507316),
whereas the Q front q50 selected epoch 4 (pinball 0.266746). These losses
have different scales and should not be compared numerically.

| Scenario | Pure STDK RMSE | Same-run fitted STDK RMSE | Q RMSE | Q versus fitted STDK |
| --- | ---: | ---: | ---: | ---: |
| Time150 | 4.293592 | 4.359954 | 4.831717 | +10.82% |
| Space100 | 5.369811 | 4.788476 | 5.691370 | +18.86% |
| ST100x150 | 4.322475 | 4.413481 | 4.730592 | +7.19% |

The within-run comparison isolates the added Q stage more cleanly than the
independent pure-STDK comparison. Q worsens all three seed-41 test RMSEs even
against its own front model. Relative to the independent pure baseline, Q's
RMSE is respectively 12.53%, 5.99% and 9.44% higher on these three targets.

## Agreement with the author-data reconstruction

| Component | Shared mechanism | Remaining difference |
| --- | --- | --- |
| STDK-to-Q bridge | q05/q50/q95 fitted-STDK series and local grids at each target position; no held-out truth used as Q training label | The author-data reconstruction uses its Keras-compatible 8x100+4x50 Dense STDK refit. Weather2K uses Spatial-adapter STInterpMLP, AdamW, EMA and fixed chronological Val150. |
| Q scope and quantiles | An independent QConvLSTM per target and quantile; q50 first, q05/q95 constrained by the same Eq. (7) implementation; pinball loss | Weather2K's location initialization seed formula differs. |
| Q network | Weather2K imports the reconstruction's `build_qconvlstm` implementation with three 5x5/3x3/1x1 ConvLSTM blocks, BatchNorm and a flattened five-output head | Weather2K tuning selected 32 filters; the reconstruction used 64. |
| Grid | The same shifted-boundary `regular_neighbourhood` function | Weather2K tuned an 11x11 grid and radius 0.3; the reconstruction assumed 8x8 and 0.2. Neither grid choice is a fully released Table 2 setting. |
| Q optimization | Adam, pinball loss, checkpoint by validation pinball | Weather2K: learning rate 0.0001, batch 64, at most 40 epochs, patience 10, chronological Val150 pseudo-label validation. Reconstruction: 0.001, batch 5, 25 epochs, patience 5, 5% random window validation. |
| Forecast | Direct 5-to-5 head | Reconstruction evaluates one final five-step block, using only the preceding five STDK grids. Weather2K applies it repeatedly to 150 test times; each later block uses newly generated STDK grids at its preceding times. Space100 additionally uses direct STDK for time indices 0--4 because no earlier five frames exist. This does not feed Q's prior predictions back into Q. |
| Data and model selection | Neither code uses the final test responses to fit Q | Weather2K's q50-only outer tuner selects hyperparameters using mean scaled RMSE against fitted-STDK Val150 pseudo labels at held-out locations. Formal evaluation trains all three quantiles. The reconstruction's internal Q validation also uses fitted-STDK pseudo labels, but its author-data protocol and data are different. |

Thus the Weather2K Q stage shares the reconstruction's architectural core,
quantile coupling and per-location idea; it is not numerically or procedurally
identical outside the Weather2K data split. In particular, the front STDK
backend and optimizer/EMA, Q capacity, training schedule and validation rule
differ. The missing author bridge means even the author-data reconstruction
cannot be called an exact execution of the original Table 2 code.

## What the saved forecasts show

The Q and fitted-STDK forecast CSVs have identical station/time/truth keys.
Paired station RMSE improves with Q at 188/500 Time150 stations, 6/100
Space100 stations and 40/100 ST100x150 stations. For Space100, the first five
time points exactly equal the fitted STDK warm-up. After Q begins predicting,
its RMSE over times 5--149 is 7.322 versus fitted STDK's 5.428. In Time150,
Q's RMSE over the final 50 time points is 5.299 versus fitted STDK's 4.487;
ST100x150's corresponding values are 5.045 versus 4.426. This is a
block-by-block performance observation, not evidence of recursive Q-error
accumulation: subsequent blocks receive STDK-generated grids.

The most direct mechanism supported by the code is that Q learns to reproduce
fitted-STDK pseudo targets and then *replaces* the fitted-STDK point forecast.
Its Val150 score measures agreement with those pseudo targets, not error
against held-out Weather2K truth. The Q approximation can therefore worsen
truth RMSE even when pseudo-label validation improves. Repeating five-step
forecasts over a 150-step target and transferring q50-only, held-out-location
tuning to all three formal scenarios may add further mismatch; their separate
causal contributions have not been isolated. No station-index or target-time
misalignment was found in the saved results or the inspected window builders.

Before claiming a general effect, investigate direct five-step and longer
block diagnostics with a fixed fitted STDK, then assess additional seeds.
Using observed held-out100 responses to choose new parameters would change
the test protocol and must not be done. Residual Q or a validation fallback
could be tested as separately named methods, not as exact paper reproduction.
