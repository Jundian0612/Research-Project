# Author-data fitted STDK vs STDK + QConvLSTM

Artifacts are separated into `checkpoints/`, `per_seed/`, `summary/`, `logs/`
and `metadata/`.

Both methods are evaluated on all 100 released simulation locations at times 496--500 (500 predictions per seed).

| Seed | STDK MSPE | STDK+Q MSPE | MSPE reduction | STDK RMSE | STDK+Q RMSE |
|---:|---:|---:|---:|---:|---:|
| 41 | 0.454033 | 0.305722 | 32.67% | 0.673819 | 0.552921 |
| 42 | 0.424638 | 0.326305 | 23.16% | 0.651642 | 0.571231 |
| 43 | 0.690607 | 0.348131 | 49.59% | 0.831028 | 0.590027 |
| 44 | 0.371300 | 0.362395 | 2.40% | 0.609344 | 0.601992 |
| 45 | 0.468320 | 0.303671 | 35.16% | 0.684339 | 0.551064 |

| Model | MSPE mean | RMSE mean | MAE mean | MPIW mean | Coverage (%) mean |
|---|---:|---:|---:|---:|---:|
| STDK | 0.481780 | 0.690035 | 0.542854 | 1.172437 | 64.360 |
| STDK + QConvLSTM | 0.329245 | 0.573447 | 0.453977 | 1.734538 | 86.000 |

QConvLSTM reduces mean MSPE by 31.66% and improves MSPE for all 5 seeds.

The wider QConvLSTM intervals increase empirical coverage. The STDK stage is deterministically refitted because the curated original raw checkpoint is unavailable.
