# STDK + QConvLSTM reproduction results

This directory contains the curated outputs of the 100-location, five-step
STDK + QConvLSTM reproduction. The experiment uses five training seeds
(41--45), and each seed evaluates 100 spatial locations at forecast times
496--500 (500 predictions per seed).

## Main result

| Metric | Five-seed mean | Sample SD | Paper Table 2 | Difference |
|---|---:|---:|---:|---:|
| MSPE | 0.329245 | 0.025844 | 0.267 | +0.062245 |
| MPIW | 1.734538 | 0.039369 | 1.462 | +0.272538 |
| Coverage (%) | 86.000 | 1.822 | 90.39 | -4.390 |

Seed-level results:

| Seed | MSPE | MPIW | Coverage (%) |
|---:|---:|---:|---:|
| 41 | 0.305722 | 1.711418 | 86.8 |
| 42 | 0.326305 | 1.739726 | 85.8 |
| 43 | 0.348131 | 1.738061 | 83.8 |
| 44 | 0.362395 | 1.794288 | 85.0 |
| 45 | 0.303671 | 1.689198 | 88.6 |

The sample SD is calculated across the five complete seed-level metrics
(`ddof=1`). It is not claimed to reproduce the paper's unspecified SE
aggregation rule.

## Directory layout

```text
STDK_QConvLSTM_reproduction_results/
├── README.md
├── STDK_QConvLSTM_results_report.tex    # LaTeX comparison/provenance report
├── code/
│   ├── STDK_QConvLSTM_reproduction.py  # main experiment
│   ├── compare_stdk_qlstm_qconvlstm.py # three-model Table 2 comparison
│   ├── run_qconvlstm_multiseed.py      # five-seed runner/summary
│   ├── run_qconv_profile_validation.py # archived profile diagnostic
│   └── calibrate_qconvlstm_interval.py # archived calibration utility
├── docs/
│   └── QCONVLSTM_REPRODUCTION.md       # detailed provenance notes
├── data/                               # released 100 x 500 simulation files
├── summary/
│   ├── table2_multiseed_...json   # machine-readable five-seed summary
│   ├── table2_multiseed_...csv    # one row per seed
│   └── table2_multiseed_...md     # compact comparison table
├── per_seed/
│   ├── table2_aggregate_seed*.json  # configuration and metrics per seed
│   └── table2_forecasts_seed*.csv   # 500 forecasts per seed
├── stdk_vs_qconvlstm/               # paired refit-STDK versus QConvLSTM
│   ├── checkpoints/
│   ├── per_seed/
│   ├── summary/
│   ├── logs/
│   └── metadata/
└── table2_stdk_qlstm_qconvlstm/     # three-model comparison
    ├── checkpoints/qlstm_locations/
    │   ├── current/                  # selected quantile-specific checkpoints
    │   └── archive/                  # superseded QLSTM variants
    ├── per_seed/
    ├── summary/
    ├── logs/
    └── metadata/
```

The selected QConvLSTM forecasts and aggregate configurations remain in
`per_seed/`; the main selected five-seed table remains in `summary/`. The two
comparison directories keep checkpoints, per-seed outputs, summaries, logs
and reproducibility metadata separate so reruns can resume without mixing
artifacts with final tables.

## Experiment configuration

- Data: the authors' `LOC_50000_univariate_spacetime_matern_stationary_1`
  and `Z1_50000_univariate_spacetime_matern_stationary_1` files.
- Field: the paper's nonstationary temporal mean is added to the released
  stationary realization.
- Final split: times 1--495 are observed; times 496--500 are the final test.
- STDK basis profile: released repository settings (spatial 5/9/11 grids;
  temporal 70/250/410 Gaussian bases with standard deviations
  0.2/0.09/0.009).
- STDK network: eight 100-unit and four 50-unit ReLU layers.
- STDK training: Adam 0.001, batch size 512, at most 350 epochs, patience 30,
  and 10% random internal validation.
- Local frames: an 8 x 8 STDK grid over target +/-0.2.
- QConvLSTM profile: the released three-block architecture with 64 filters
  and 5x5, 3x3, and 1x1 kernels; BatchNorm follows the first two blocks.
- Forecast reconstruction: five historical grids directly produce five
  future target values.
- Quantiles: 0.05, 0.50, and 0.95 with the paper Eq. (7) median-centred
  non-crossing constraint.
- QConvLSTM training: Adam 0.001, batch size 5, at most 25 epochs, patience 5,
  and 5% random internal validation.
- Response normalization: none.
- Interval post-scaling: none (`interval_scale=1.0`).

## Reproducing the run

Run one complete seed:

```bash
/home/jundian/installers/yes/envs/geospatial-neural-adapter/bin/python -u \
  STDK_QConvLSTM_reproduction_results/code/STDK_QConvLSTM_reproduction.py \
  --all-targets \
  --stdk-profile repository \
  --qconv-profile github_3block \
  --validation-mode repository_random \
  --normalization none \
  --seed 41
```

Run and summarize all five seeds:

```bash
/home/jundian/installers/yes/envs/geospatial-neural-adapter/bin/python -u \
  STDK_QConvLSTM_reproduction_results/code/run_qconvlstm_multiseed.py \
  --seeds 41 42 43 44 45 \
  --stdk-profile repository \
  --qconv-profile github_3block \
  --validation-mode repository_random \
  --normalization none
```

Compatible completed location files are reused automatically. Do not add
`--overwrite` unless every compatible model should be retrained.

## Comparing STDK before and after QConvLSTM

The curated results retain the completed QConvLSTM forecasts but not the raw
STDK checkpoints. The following command deterministically refits the exact
seed-specific STDK stage, evaluates q05/q50/q95 on times 496--500, and pairs
those 500 predictions with the saved QConvLSTM predictions:

```bash
.venv-wsl/bin/python -u \
  STDK_QConvLSTM_reproduction_results/code/compare_refit_stdk_vs_qconvlstm.py \
  --seeds 41 42 43 44 45
```

Outputs are written to `stdk_vs_qconvlstm/`. Because the original raw
checkpoint is unavailable, reports call this a deterministic STDK refit rather
than claiming that the exact historical in-memory model was restored.

## Three-model Table 2 comparison

The three-model comparison adds the paper's scalar QLSTM control to the paired
STDK and QConvLSTM results:

```bash
.venv-wsl/bin/python -u \
  STDK_QConvLSTM_reproduction_results/code/compare_stdk_qlstm_qconvlstm.py \
  --seeds 41 42 43 44 45
```

For every seed it evaluates the same 100 locations at times 496--500 and
writes both metrics and 1,500 prediction rows (three models x 500 targets).
QLSTM combines the released 50K notebook settings with the paper's quantile
definition: a five-step lookback, `LSTM(50, activation="relu")`, Adam 0.001,
batch size 128, 5% validation split and 120 epochs. The q50 model is fitted
first; q05 and q95 use their corresponding STDK quantile series and the
median-centred non-crossing output in paper Eq. (7). Forecasting recursively
rolls each quantile history forward.

The public repository does not contain `50k_lstm_data.csv`. Its role is
reconstructed using fitted STDK q05/q50/q95 series at each location, following
the paper's quantile-specific `X^NN` definition but not verifiable byte for
byte. The paper says 100 locations while the released QLSTM notebook loops
over 50 columns; this comparison follows the paper and evaluates all 100.
TensorFlow/Keras is unavailable in the project environment, so the same QLSTM
cell equations and Keras initializer conventions are implemented in PyTorch.

## Reproduction scope

This is a best-effort reproduction based on the public simulation files, the
paper, and the released STDK/ConvLSTM code. It is not an exact rerun of the
authors' original Table 2 pipeline because the following simulation-specific
details are not public:

- the complete 50K STDK-to-QConvLSTM bridge;
- `training_data.csv`, `model_real.h5`, and `50k_lstm_data.csv`;
- the 50K QConvLSTM lookback/output and recursive-versus-direct rule;
- the exact quantile-lambda constant and random seed;
- the selected 50 locations referenced by the public scalar-LSTM notebook;
- the paper's SE aggregation unit.

The final times 496--500 have already been used for evaluation. They should
not be used to tune lambda, interval scale, grid radius, or architecture.
