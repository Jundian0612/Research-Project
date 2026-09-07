# STDK + QConvLSTM Weather2K 495-to-5 pilot

This directory preserves the completed Weather2K data-adaptation pilot that
transferred the selected STDK + QConvLSTM simulation pipeline without changing
its 100-station, 500-time, direct five-step protocol.

## Status and scope

This is a completed pilot, not the formal Weather2K benchmark. It must not be
placed in the same ranking table as models evaluated with the shared
`100/400/100` station split and `700/150/150` time split.

- variable: `air_temperature`
- stations: 100 sampled without replacement from 1,866 Weather2K stations
- station sampling seed: 41 (fixed across model seeds)
- time range: final 500 three-hourly observations
- split: first 495 observed, final 5 test
- model seeds: 41--45
- lookback/output: 5 local grids directly predict 5 future values
- evaluation per seed: 100 stations x 5 leads = 500 predictions
- total evaluation: 2,500 predictions
- quantile crossings: 0

## Five-seed result

| Metric | Mean | Sample SD (ddof=1) |
|---|---:|---:|
| MSPE | 8.878931 | 0.748675 |
| RMSE | 2.977548 | 0.128170 |
| MPIW | 7.548212 | 0.242243 |
| Coverage (%) | 77.080000 | 4.607819 |

Seed-level results:

| Seed | MSPE | RMSE | MPIW | Coverage (%) |
|---:|---:|---:|---:|---:|
| 41 | 8.753086 | 2.958561 | 7.717813 | 79.6 |
| 42 | 9.463111 | 3.076217 | 7.642479 | 76.2 |
| 43 | 7.650584 | 2.765969 | 7.576726 | 83.6 |
| 44 | 9.064367 | 3.010709 | 7.679078 | 74.0 |
| 45 | 9.463509 | 3.076282 | 7.124966 | 72.0 |

The nominal 90% interval achieved 77.08% coverage. The final five test times
were not used to tune lambda or post-hoc interval scale.

## Directory layout

```text
STDK_QConvLSTM_Weather2K_495to5_pilot_20260830/
├── README.md
├── summary/
│   ├── five_seed_metrics.csv
│   ├── five_seed_summary.json
│   ├── lead_metrics.csv
│   └── forecasts_5seeds.csv
├── per_seed/
│   ├── aggregate_seed41.json
│   ├── ...
│   └── forecasts_seed45.csv
└── raw_results/
    ├── checkpoints/       # 505 .pt files
    └── per_station/       # 1,000 station-level JSON/CSV files
```

## Why raw results are about 4.8 GiB

The experiment trained one separate QConvLSTM quantile bundle for every
station and model seed:

```text
100 stations x 5 seeds = 500 station-level QConvLSTM checkpoints
```

Each checkpoint is approximately 10.2 MB because it stores q05, q50 and q95
model weights for the three 64-filter ConvLSTM blocks and dense output heads.
The 500 QConvLSTM checkpoints alone therefore occupy about 5.1 GB in decimal
units (about 4.75 GiB). Five STDK quantile checkpoints and the small JSON/CSV
outputs account for the remainder.

`raw_results/` is intentionally ignored by Git. The README, summaries and
per-seed aggregate/forecast files are small and suitable for GitHub. For
public archival of the model weights, use a versioned research-artifact store
such as Zenodo or a Hugging Face dataset rather than the main Git repository.

## Next formal experiment

The formal comparison must use the common Weather2K runner protocol:

- 600 sampled stations with `100/400/100` roles
- 500 supervised training stations and 100 held-out stations
- final 1,000 times split into 700 train, 150 validation and 150 test
- time, space and spatio-temporal extrapolation
- seeds 41--45 with mean and sample SD

This pilot remains useful for provenance and pipeline validation, but its
metrics are not directly comparable with that formal benchmark.
