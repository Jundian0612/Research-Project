# Aligned standalone STDK — seed 41

Status: complete. This run was generated on 2026-09-13 with the standalone
STDK implementation aligned to the pinned Spatial-adapter STDK baseline.

## Protocol

- seed: 41
- one fitted STDK evaluated on all three scenarios
- stations: sampled600 → train500 + strict held-out100
- time: Train700 / Val150 / Test150
- normalization: observed100 × Train700
- validation: batch-average MSE
- checkpoint parameters: EMA enabled
- runtime: 334.54 seconds on the RTX 2000 Ada environment

## Target results

| Scenario | RMSE | MSE | MAE | R² |
|---|---:|---:|---:|---:|
| Time150 | 4.293592 | 18.434929 | 3.377500 | 0.391968 |
| Space100 | 5.369811 | 28.834875 | 4.174917 | 0.307148 |
| ST100×150 | 4.322475 | 18.683786 | 3.409251 | 0.349286 |

## Files

- `2K_stdk_metrics_all_three_train500_test100.json`: configuration, split
  indices, training summary and all evaluation metrics
- `metadata/code.sha256`: hashes of the executed STDK script and pinned
  upstream model/trainer sources

The terminal log remains machine-local because `*.log` is excluded by the
repository `.gitignore`.
