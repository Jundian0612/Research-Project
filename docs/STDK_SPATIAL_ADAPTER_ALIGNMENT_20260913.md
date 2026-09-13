# STDK alignment with pinned spatial-adapter

Reference revision: `2aea188f3b8d92f948663b6705f0a22850d6e4ee`

Primary references:

- `spatial-adapter/examples/experiments/configs/production/weather2k_80_10_10.yaml`
- `spatial-adapter/examples/experiments/weather2k/experiment.py`
- `spatial-adapter/examples/baselines/stdk/st_interp.py`
- `spatial-adapter/examples/baselines/stdk/trainer.py`
- `spatial-adapter/examples/baselines/stdk/utils/ema.py`

## Aligned STDK behavior

Both `weather2k_exp/2K_STDK_500train_100test.py` and the STDK front end in
`weather2k_exp/2K_STDK_QConvLSTM.py` use the pinned repository's
`STInterpMLP`/`create_model` implementation and its active baseline defaults:

| Item | Current Weather2K comparison | Pinned repository |
|---|---|---|
| Spatial centers | 25, 81, 121 | 25, 81, 121 |
| Temporal centers | 10, 15, 45 | 10, 15, 45 |
| Hidden dimensions | 256, 256, 128 | 256, 256, 128 |
| Spatial basis | fixed uniform Wendland | fixed uniform Wendland |
| Dropout / LayerNorm | 0.1 / enabled | 0.1 / enabled |
| Optimizer | AdamW | AdamW |
| Epochs / patience | 350 / 30 | 350 / 30 |
| LR / weight decay | 1e-3 / 1e-4 | 1e-3 / 1e-4 |
| Batch size | 512 | 512 |
| Mean-model loss | MSE | MSE |
| Validation aggregation | mean of batch losses | mean of batch losses |
| EMA default | enabled | enabled when `use_ema` is omitted |
| EMA decay | `1 - 1/(10*batches_per_epoch)` | same |
| Checkpoint rule | lowest validation loss, patience 30 | same |

The local scripts implement the active training branch directly rather than
instantiating the repository's `Trainer`, because STDK+Q additionally fits
conditional q05/q95 models. The optimizer, EMA, validation and checkpoint
steps above follow the pinned trainer. On 2026-09-13 the standalone validation
loader was changed to the repository's time-major order over the complete
train500 set, and both scripts were changed to honor the `use_ema` switch.

## Intentional Weather2K comparison settings

These are retained because they define this study rather than the upstream
paper experiment:

- fixed last 1000 points with Train700 / Val150 / Test150;
- sampled600 split into train500 and strict held-out100;
- obs100 / unobserved400 roles inside train500;
- target scaling from obs100 x Train700 only;
- coordinates and time mapped to `[0,1]` for the STDK basis domain;
- one fitted model per seed evaluated on Time150, Space100 and ST100x150;
- seeds 41--45 and common station indices across all four compared models;
- held-out100 responses excluded from training and model selection.

The upstream Weather2K experiment uses a different location ratio, time-ratio
protocol and unscaled targets. Copying those choices would replace the study's
benchmark rather than merely align the STDK implementation.

## STDK+Q-specific adaptation

The q50 STDK follows the aligned mean/MSE branch. The q05 and q95 models use
pinball loss plus a median-conditioned non-crossing construction because the
downstream QConvLSTM experiment requires ordered quantile grids. QConvLSTM,
its 5-to-5 windows, tuned grid size/radius and direct quantile outputs are not
part of the spatial-adapter repository's STDK baseline.

## Result validity after alignment change

Results generated before 2026-09-13 retain their original code hashes and are
valid records of those runs. Because validation row ordering can affect an
unweighted mean of batch losses when the final batch is short, the standalone
STDK seed-41 result generated on 2026-09-12 must be rerun before it is treated
as the result of the fully aligned code. Existing SVGP and DLinear+FRK results
are unaffected. STDK+Q already used time-major train500 validation rows; its
formal seed-41 result is unaffected by that ordering correction.
