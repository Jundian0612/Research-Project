# SVGP hyperparameter tuning

Selection uses only mean best validation RMSE on seeds [41, 42]; test metrics are not used for model selection.

## Best configuration

- Kernel: `matern_periodic`
- Learning rate: `0.001`
- Inducing points: `1024`
- Validation RMSE (scaled): `0.473901 ± 0.003545`
