# Paired Weather2K STDK and STDK+QConvLSTM protocol

The two current Weather2K scripts now share the **same fitted STDK checkpoint**
for each seed and scenario. Earlier Weather2K STDK+Q results fitted a second,
pinball-loss STDK and must not be treated as results of this protocol.

## Provenance and adaptations

| Component | Current implementation | Source or adaptation |
| --- | --- | --- |
| STDK model | `STInterpMLP`, fixed Wendland spatial basis, Gaussian temporal basis, 256/256/128 hidden layers | Pinned `spatial-adapter/examples/baselines/stdk/st_interp.py` |
| STDK fit | Pinned `Trainer`, AdamW, MSE, batch 512, up to 350 epochs, patience 30, EMA best checkpoint | Pinned `spatial-adapter/examples/baselines/stdk/trainer.py` and `examples/experiments/configs/kaust.yaml` |
| Weather2K input | 600 sampled stations; obs100 + unobs400 = train500; held-out100; last 1000 points split 700/150/150; obs100 x Train700 standardization | This project's common comparison protocol |
| Q network | Released three-block ConvLSTM design and quantile pinball/median-constrained heads, independently fitted per target location | Nag et al. paper/notebooks, via `STDK_QConvLSTM_reproduction_results/code/STDK_QConvLSTM_reproduction.py` |
| STDK-to-Q bridge | Local grids and target pseudo-series all come from **one frozen pure-STDK checkpoint**. q05/q50/q95 Q models receive the same STDK grid/series; quantiles are produced by their own heads. | Necessary paired-comparison adaptation. Nag's complete STDK-to-Q data bridge was not released; this is not an exact paper reproduction. |
| Q horizon | Five past STDK grids to five future values, repeated in contiguous blocks over the Weather2K evaluation horizon; no feedback of earlier Q outputs | Five-step author-style head plus Weather2K 150-step adaptation |

The Q script does not fit any STDK. Its checkpoint loader checks SHA256,
architecture/configuration, seed, scenario, station indices, time split and
normalization against the pure-STDK metadata. It freezes all STDK parameters.
The Q report contains the checkpoint path and SHA256; its `fitted_stdk_baseline`
must reproduce the pure-STDK RMSE for that seed and scenario. The code fails
if it does not. STDK is a point model, so its interval fields are null.

The strict held-out100 responses never become Q training labels. For those
target sites, Q is trained against the frozen STDK's pseudo-series; its
validation score measures agreement with STDK, not predictive accuracy against
held-out truth. This limitation must be kept in mind when interpreting results.

## Run a paired comparison

Use a fresh output directory. In the project environment, run:

```bash
cd /home/user/Research-Project
source .venv-wsl/bin/activate
SEED_LIST='[41]' QCONVLSTM_SEEDS=41 \
RUN_STDK=1 RUN_QCONVLSTM=1 RUN_SVGP=0 RUN_DLINEAR=0 \
WEATHER2K_OUTPUT_DIR=weather2k_exp/air_temperature/paired_stdk_nag_seed41 \
python -u weather2k_exp/experiments_runner.py
```

The runner fits pure STDK first for each scenario and saves its selected
checkpoint under `checkpoints/<scenario>_seed41/model_best.pt`. It then passes
that checkpoint directory to the Q script. The companion `metadata.json`
records the data protocol and target metrics. The default Q parameters in
`weather2k_exp/2K_stdk_nag_qconvlstm_params.json` are an **untuned paired
pilot** based on the author-data reconstruction. The 8x8 grid and radius 0.2
were inferred in that reconstruction, not specified completely by the paper.

The old `2K_best_stdk_qconvlstm_params_500to100.json` belongs to the
pinball-STDK workflow and is rejected by the new Q script. To select Q
hyperparameters for this paired workflow, rerun
`weather2k_exp/tune_stdk_qconvlstm_hyperparams.py` using Val150 only; it fits
pure-STDK checkpoints once before its Q trials. Do not combine old seed-41
metrics with new runs or use Test150 to tune parameters.
