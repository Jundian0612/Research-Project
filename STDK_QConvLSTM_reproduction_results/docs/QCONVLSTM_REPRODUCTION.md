# Nag et al. Table 2 QConvLSTM reproduction

Main program: `STDK_QConvLSTM_reproduction_results/code/STDK_QConvLSTM_reproduction.py`

The formal reproduction uses the author's 1--495/496--500 split directly.
Chronological 491--495 validation is an optional diagnostic only and is not
part of the published author workflow.

The public 50K forecasting section is a separate one-step, recursive LSTM
experiment. Its 50-unit LSTM, 120 epochs and batch size 128 are therefore not
used as QConvLSTM settings. The QConvLSTM defaults instead follow the released
`CONV_LSTM.ipynb` architecture (three 64-filter blocks, batch normalization,
25 epochs and batch size 5), adapted to the paper's five-step simulation
horizon. The missing `training_data.csv` and `model_real.h5` prevent an exact
end-to-end author-code reproduction.

## Formal repository-profile run

```bash
cd /home/jundian/Research-Project
/home/jundian/installers/yes/envs/geospatial-neural-adapter/bin/python \
  STDK_QConvLSTM_reproduction_results/code/STDK_QConvLSTM_reproduction.py \
  --all-targets \
  --stdk-profile repository \
  --validation-mode repository_random \
  --normalization none \
  --grid-boundary shift \
  --seed 41
```

Completed target JSON files are reused automatically, so the same command can
resume an interrupted all-location run. Add `--overwrite` only when all target
models should be retrained.

The default output directory for this setting is
`STDK_QConvLSTM_reproduction_results/raw_results/06_stdkval10_qconvval05_current`.
Earlier experiments are archived by method under
`STDK_QConvLSTM_reproduction_results/raw_results/00`--`05`
and are not overwritten by the command above.

## Implemented author settings

- Author simulation files: 100 locations x 500 time points.
- Paper split: times 1--495 train; 496--500 test.
- Paper nonstationary temporal mean is added to the repository stationary field.
- Repository STDK profile: spatial 5/9/11 grids; temporal 70/250/410 Gaussian bases with std 0.2/0.09/0.009.
- STDK DNN: eight 100-unit and four 50-unit ReLU layers; Adam 0.001; batch 512; max 350 epochs; patience 30.
- The fourth and fifth 100-unit Dense layers also use the repository's activity L2 regularization (`1e-5`), scaled by batch size as in Keras.
- Validation follows the two released notebooks separately: STDK uses a 10% random validation split, while QConvLSTM uses a 5% random validation split.
- Local input: STDK-generated 8x8 grid over target +/-0.2 and five chronological frames (`n_steps=5` in the 50K notebook forecasting section).
- STDK first fits the median, then fits lower/upper raw networks with paper
  Eq. (7): `q_tau = q50 +/- lambda*abs(tau-0.5)*sigmoid(raw)`. By default,
  `lambda` is half the range of the first 495 training responses. These
  models produce separate q05/q50/q95 local grids.
- QConvLSTM: 64 filters at 5x5, 3x3 and 1x1; BatchNorm after the first two
  blocks; full-sequence flatten; five-output direct forecast.
- The PyTorch ConvLSTM follows the Keras defaults used by the author notebook:
  separate input/recurrent convolutions, Glorot-uniform input kernels,
  orthogonal recurrent kernels, zero bias with forget-gate bias one, and
  sequence-level BatchNorm with epsilon 0.001 and PyTorch momentum 0.01
  (equivalent to Keras momentum 0.99).
- QConvLSTM also fits the median first and uses paper Eq. (7) for quantiles
  0.05 and 0.95.
- All 100 locations and all five forecast leads are pooled for MSPE, MPIW and coverage.

## Explicit assumptions

The public repositories omit the Table 2 QConvLSTM quantile program,
`synthetic_50000.csv`, `training_data.csv`, `model_real.h5`, the exact lambda
constant used for Table 2, random seed, boundary handling and SE aggregation.
The released 50K code hard-codes a deviation multiplier of 10, whereas paper
Eq. (7) includes `abs(tau-0.5)` and recommends lambda proportional to half the
response range. This
pipeline therefore records these assumptions in every output JSON:

- boundary grids are shifted inside [0,1] rather than clipped;
- quantiles use paper Eq. (7) and are not sorted after prediction;
- the last five observed STDK-grid frames directly output all five future
  values, so no t=496--500 STDK grid is supplied as a forecasting input;
- candidate location-level and prediction-level SD/SE values are all reported;
  none is claimed to be the paper's unknown definition.

The pipeline must not be tuned against 0.267, 1.462 or 90.39%. The results are
a transparent best-effort reproduction, not an exact reproduction claim.

## Alternative paper-only profile

The optional `--qconv-profile paper_3x3` profile implements the paper's
explicit valid 3x3 input convolution with 64 feature maps, reducing an 8x8
map to 6x6. The paper does not disclose recurrent padding, layer count `P`,
grid radius, or the future-grid rule. These remain explicitly marked as
inferred in each output JSON under `parameter_provenance`; the profile does
not claim exact author-code equivalence.

The optional `--stdk-profile paper` setting uses spatial grids
5/9/12, temporal bases 10/15/45, and paper spatial bandwidth equal to 2.5
times adjacent-anchor spacing (`2.5/(m-1)`). The former `repository` STDK
profile remains available. The defaults use the released repository profiles:
`--stdk-profile repository` and `--qconv-profile github_3block`.

## Five-seed run and mean/std

Run seeds 41--45 sequentially and calculate the mean and sample standard
deviation across the five complete seed-level experiments:

```bash
cd /home/jundian/Research-Project
/home/jundian/installers/yes/envs/geospatial-neural-adapter/bin/python \
  STDK_QConvLSTM_reproduction_results/code/run_qconvlstm_multiseed.py
```

The runner reuses a seed only when its Keras-compatible aggregate contains all
100 locations and 500 forecasts. The earlier non-Keras-compatible five-seed
results are deliberately not reused. Incomplete seeds resume from their
existing target JSON files. The final JSON, CSV and Markdown files use the
prefix `table2_multiseed_seeds41-42-43-44-45_repository_kerascompat_`.
