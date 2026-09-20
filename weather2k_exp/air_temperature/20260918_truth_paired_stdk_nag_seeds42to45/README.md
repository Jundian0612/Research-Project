# Fixed-parameter paired STDK / STDK+Q, seeds 42–45

This batch extends the completed seed-41 run in the neighboring
`20260917_truth_paired_stdk_nag_seed41` directory. Each seed and scenario fits
pure Spatial Adapter STDK, selects its EMA checkpoint using Train700/Val150,
then loads and freezes that exact checkpoint for the Nag-style q05/q50/q95
QConvLSTM stage. The eight Q hyperparameters are copied unchanged from the
completed Val150-only tuning run into `locked_q_params.json`; SHA-256 checks
guard the parameters and source files before every scenario. Test truth is
used only for final scoring.

Run progress is in `run.log`. A completed scenario has
`seedNN/metrics/verified_SCENARIO.json`, which checks the pure/Q checkpoint path and
SHA-256, fitted-STDK RMSE, fixed parameters, split, and all three quantile
validation reports. `exit_code=0` means all twelve seed/scenario pairs passed
and `summarize_five_seeds.py` wrote [`tables/five_seed_summary.md`](tables/five_seed_summary.md),
`.csv`, and `.json`.
The summary includes seed 41 and uses population SD (`ddof=0`), consistent
with existing repository result tables.

Q hyperparameters were selected using Val150 results from seeds 41 and 42.
Consequently seed 42 is included in validation-based hyperparameter selection;
seeds 43–45 were not. No Test150 or held-out100 test metrics were used in that
selection. The five-seed table describes repeatability of the locked protocol,
not an entirely untouched five-seed hyperparameter-selection holdout.

Each `seedNN/` keeps its checkpoint at the original path. `metrics/` contains
the pure-STDK and Q reports plus verification; `tables/` holds scenario tables
and paired comparisons; `forecasts/` holds raw-scale predictions. This batch's
`tables/` contains the combined 41–45 results, while `metrics/` contains the
two independent audit reports. See [the protocol audit](20260920_PROTOCOL_AUDIT.md)
for the checks and their limits. The launcher organizes completed runner
outputs using `weather2k_exp/organize_paired_stdk_results.py`. Checkpoint `.pt`
files and `run.log` remain local under `.gitignore`.
