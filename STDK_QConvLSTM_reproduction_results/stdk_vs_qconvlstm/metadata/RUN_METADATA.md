# Run metadata

Formal paired comparison on the authors' released 100-location x 500-time simulation.

- Seeds: 41--45
- STDK training: times 1--495, repository profile, random 10% validation, no normalization
- Test: all 100 locations at times 496--500 (500 predictions per seed)
- QConvLSTM source: curated completed per-seed forecasts in `../per_seed/`
- STDK source: deterministic refit because the original raw STDK checkpoints were not retained in this checkout
- Point comparison: MSPE, RMSE and MAE of q50
- Interval comparison: MPIW and empirical 90% coverage
