# Author-data STDK, STDK+QLSTM and STDK+QConvLSTM

Artifacts are separated into `checkpoints/`, `per_seed/`, `summary/`, `logs/`
and `metadata/`.

| Model | Reproduced MSPE (mean +/- SD) | Paper MSPE | Reproduced MPIW (mean +/- SD) | Paper MPIW | Reproduced coverage % (mean +/- SD) | Paper coverage (%) |
|---|---:|---:|---:|---:|---:|---:|
| STDK | 0.481780 +/- 0.122506 | -- | 1.172437 +/- 0.070611 | -- | 64.360 +/- 7.586 | -- |
| STDK_plus_QLSTM | 0.348730 +/- 0.022379 | 0.392 | 1.670573 +/- 0.015581 | 1.558 | 84.640 +/- 0.910 | 89.94 |
| STDK_plus_QConvLSTM | 0.329245 +/- 0.025844 | 0.267 | 1.734538 +/- 0.039369 | 1.462 | 86.000 +/- 1.822 | 90.39 |

QConvLSTM versus QLSTM MSPE reduction: `5.59%`.
QConvLSTM has lower MSPE than QLSTM in `4/5` seeds.

The missing `50k_lstm_data.csv` bridge is reconstructed from the quantile-specific fitted STDK q05/q50/q95 series following the paper's X^NN_tau definition; this remains a best-effort reproduction.
The paper values are references, not values copied into the reproduced metrics.
