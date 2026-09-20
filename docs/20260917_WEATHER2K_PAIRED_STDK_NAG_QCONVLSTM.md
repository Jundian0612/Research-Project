# Paired Spatial Adapter STDK and Nag-style QConvLSTM on Weather2K

## Comparison being run

For each seed and evaluation scenario, the pure STDK script fits the pinned
Spatial Adapter `STInterpMLP` using its `Trainer`. It saves the selected
best/EMA `model_best.pt` and directly evaluates that model. The STDK+Q script
loads **that same checkpoint**, freezes its parameters, forms location-specific
STDK grids, and fits Nag-style three-block QConvLSTM quantile models using
observed Weather2K temperatures at supervised training stations. The two
reported point forecasts therefore share the exact first-stage weights.

| Model | Train700 truth | Val150 truth | Final target truth |
| --- | --- | --- | --- |
| Pure Spatial Adapter STDK | Mean regression with MSE against train500 truth | MSE against the same 500 stations' truth selects the best/EMA checkpoint | Test targets only |
| STDK+QConvLSTM | The same frozen STDK; Q uses local grids and observed training-station truth with pinball loss | Observed training-station truth selects each Q checkpoint by pinball loss; q50 truth RMSE is available to select Q hyperparameters | Test targets only |
| SVGP | Train700 truth at train500 stations fits the GP | Val150 truth at those 500 stations selects the epoch; previous tuning selected kernel and inducing settings | Test targets only |
| DLinear+FRK | DLinear takes obs100 inputs; loss compares both obs100 predictions and spatial-surrogate predictions at unobs400 against their Train700 truth | Val150 truth at all 500 stations selects epoch and previous tuning selected loss/config settings | Test targets only |

For Time150, every Q model uses its target station's own Train700 and Val150
truth, because those 500 stations are supervised. For Space100 and ST100×150,
the 100 target stations are strictly held out. Each target-specific Q model is
instead fitted and validated at its **nearest train500 station**, using that
station's observed truth and grids from the frozen STDK. The fitted Q model is
then applied to grids centred on the held-out target. This deterministic
nearest-station transfer is a Weather2K adaptation, not a released Nag-paper
procedure. Held-out100 truth never selects or trains either stage.

```text
Train700 truth (train500) -> Spatial Adapter STDK, mean regression + MSE
Val150 truth (train500)   -> select best/EMA STDK checkpoint
                                |
                                +-> pure STDK: evaluate three scenarios
                                |
                                +-> freeze the same STDK checkpoint
                                    -> local STDK grids at supervised source sites
Train700 truth (sources)          -> train target-specific Q with pinball loss
Val150 truth (sources)            -> select Q checkpoint and tune Q settings
                                    -> apply Q at each scenario's target sites
Final target truth                -> evaluation only
```

Both arms use the same 600 sampled stations, train500/strict held-out100 split,
last 1,000 times divided into Train700/Val150/Test150, normalization fitted
from obs100 × Train700, and scenario targets:

| Scenario | Final target |
| --- | --- |
| Time150 | train500 × Test150 |
| Space100 | held-out100 × the first 850 times |
| ST100×150 | held-out100 × Test150 |

Each scenario is a separate training invocation, as in the other Weather2K
models. The checkpoint directory is keyed by scenario and seed. The Q loader
checks station IDs, time split, normalization, model configuration, source
hashes, checkpoint SHA-256, and a train/validation prediction probe before it
fits Q. Its saved STDK target RMSE must also match the pure-STDK report.

## Sources and deliberate adaptations

- STDK architecture and training loop: pinned
  `spatial-adapter/examples/baselines/stdk/st_interp.py` and `trainer.py`.
  Weather2K's data split, normalization, and three evaluation targets replace
  the upstream experiment's data protocol.
- QConvLSTM implementation: `build_qconvlstm`, local-neighbourhood grid builder,
  pinball loss, and median-centred non-crossing quantiles from
  `STDK_QConvLSTM_reproduction_results/code/STDK_QConvLSTM_reproduction.py`,
  which reconstructs the published Nag notebook/paper flow. A separate Q model
  is fitted for every target station and quantile; five historical STDK grids
  produce five future predictions.
- One deterministic mean-STDK checkpoint does **not** supply the paper's
  quantile-specific q05/q50/q95 STDK models. Here all Q quantiles receive grids
  from the same frozen STDK. Q targets are observed temperatures at available
  train500 stations, and q05/q50/q95 are learned with pinball loss and the
  non-crossing constraint. For the held-out targets, the nearest supervised
  station supplies Q's labels. This is a paired adaptation, not an exact Table
  2 reproduction.
- Nag's short five-step forecast is repeated in contiguous blocks to cover
  Weather2K's 150-step test. Later blocks receive fresh STDK grids, not earlier
  Q outputs. Space100 uses direct STDK for times 0–4, where five prior grids do
  not exist. Grid size and radius are reconstruction choices, not fully released
  Table 2 settings.

Only point metrics are comparable between pure STDK and STDK+Q. Pure STDK has
no q05/q95 prediction interval; Q's interval width and coverage are reported
separately. The current parameter file
`weather2k_exp/2K_stdk_nag_qconvlstm_params.json` is marked
`untuned_truth_supervised_paired_pilot`; older tuning for a separately fitted
pinball-STDK front is not transferable to this checkpoint-sharing workflow.
Use observed train500 Val150 truth to tune this new Q stage before making a
formal multi-seed claim.

The selected Q epoch's weights are held in memory for the run and used for
inference; this workflow currently saves validation scores and forecasts but
does not write a separate Q weight file for every target station.

The current `tune_stdk_qconvlstm_hyperparams.py` is checkpoint-compatible and
selects q50 grid, radius, learning rate, filters, and weight decay by mean
**observed train500 Val150** RMSE. Each Q fit selects its epoch using Val150
pinball loss. For held-out spatial targets, this score measures temporal validation at their
nearest supervised source stations; it is not a direct estimate of spatial
transfer error. A nested spatial holdout within the 500 supervised stations
would be needed for spatially targeted hyperparameter selection. Never use the
strict held-out100 or Test150 responses to select Q hyperparameters.

The full GPU sweep completed on 2026-09-17 at 19:17 Taipei time with exit
code 0. It evaluated nine grid/radius candidates and eight
learning-rate/filter/weight-decay candidates, each on seeds 41 and 42. The
selected q50 validation RMSE on standardized observed train500 Val150 truth
was **0.589080 ± 0.001383** across the two seeds (seed 41: 0.590463; seed 42:
0.587697). Stage 1 selected a 5×5 grid and radius 0.3; stage 2 selected
learning rate 0.001, 32 filters, and weight decay 0.00001. Batch size 5,
maximum 25 epochs, and patience 5 were fixed during this sweep. Full reports
are under `weather2k_exp/air_temperature/20260917_stdk_nag_truth_qconvlstm_tuning/`;
the selected settings were copied to
`weather2k_exp/2K_stdk_nag_qconvlstm_params.json`.

The sweep fitted q50 only. The three-quantile, three-scenario formal run was
subsequently completed for seeds 41–45; see the
[five-seed result table](../weather2k_exp/air_temperature/20260918_truth_paired_stdk_nag_seeds42to45/tables/five_seed_summary.md).
The validation score is measured
at nearest supervised source stations for held-out spatial targets, not on
strict held-out100 or Test150 truth. The run's `code.sha256` pins the source
version used by all trials.

## Run a paired seed

In the GPU-enabled environment, from the repository root:

```bash
bash weather2k_exp/air_temperature/20260917_truth_paired_stdk_nag_seed41/run_seed41.sh
```

The launcher fits pure STDK and then Q for Time150, Space100, and ST100×150.
Inspect the archived `tables/results_<scenario>.csv`, paired Q
`tables/*_comparison.csv`, `metrics/verified_<scenario>.json`, and
`checkpoints/<scenario>_seed41/metadata.json`. This run can take much longer
than pure STDK because the Q stage fits one model per target station and
quantile. Do not merge its metrics with earlier Q runs that used STDK
pseudo-labels or trained a different STDK front model.
