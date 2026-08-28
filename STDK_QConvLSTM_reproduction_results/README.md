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
│   ├── run_qconvlstm_multiseed.py      # five-seed runner/summary
│   ├── run_qconv_profile_validation.py # archived profile diagnostic
│   └── calibrate_qconvlstm_interval.py # archived calibration utility
├── docs/
│   └── QCONVLSTM_REPRODUCTION.md       # detailed provenance notes
├── data/                               # released 100 x 500 simulation files
├── raw_results/                        # all checkpoints and historical runs
├── summary/
│   ├── table2_multiseed_...json   # machine-readable five-seed summary
│   ├── table2_multiseed_...csv    # one row per seed
│   └── table2_multiseed_...md     # compact comparison table
├── per_seed/
│   ├── table2_aggregate_seed*.json  # configuration and metrics per seed
│   └── table2_forecasts_seed*.csv   # 500 forecasts per seed
└── logs/
    ├── final/                        # logs for the selected final profile
    │   └── STDK_QConvLSTM_*.log
    └── archive/                      # logs from earlier ablations/profiles
        └── qconvlstm_*.log
```

The archived logs document earlier experiments such as activity-regularizer
ablations, the pre-Keras-compatible implementation, and the paper-3x3 pilot.
They are retained for provenance but are not part of the selected five-seed
result reported above.

Large checkpoints and location-level artifacts are stored under:

```text
raw_results/06_stdkval10_qconvval05_current/
```

The selected raw directory contains the five STDK checkpoints and 500
location-level QConvLSTM model files. `raw_results/` also retains all earlier
profiles and ablations for provenance.

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
