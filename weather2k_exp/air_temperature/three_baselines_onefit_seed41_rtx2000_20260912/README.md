# Weather2K one-fit seed 41 comparison

> Alignment note (2026-09-13): this completed run predates the final standalone
> STDK validation-row ordering correction. Its SVGP and DLinear+FRK outputs
> remain current; rerun standalone STDK before using the four-model table as the
> final aligned seed-41 comparison. `metadata/code_and_params.sha256` preserves
> the exact pre-correction source hashes used here.

This directory contains the completed RTX 2000 Ada seed 41 runs for SVGP,
standalone STDK, and DLinear + differentiable FRK. Each model was fitted once
and the same fitted instance was evaluated on all three scenarios.

The matching STDK+Q run is stored in
`../STDK_QConvLSTM_formal_ema_all3_rtx2000_20260912/`. Its metrics are included
in `tables/comparison_four_models_seed41.csv` so that the four models can be
reviewed together without duplicating the original forecast files.

Directory layout:

- `metrics/`: model JSON outputs.
- `tables/`: scenario tables and comparison tables.
- `logs/`: successful and failed execution logs.
- `metadata/`: Python package snapshot and code/parameter SHA-256 hashes.

## Comparability check

- Seed: 41 only.
- Time split: Train700 / Val150 / Test150.
- Station split: obs100 / unobserved400 / held-out100.
- Normalization source: obs100 x Train700.
- All four runs use identical sampled600, train500, held-out100, and obs100
  station indices. The stored index arrays were compared directly.
- Each model/seed is trained once and that fitted instance is evaluated on
  `Target_Time150`, `Target_Space100`, and `Target_ST100x150`.
- Held-out100 truth is used only for final scoring.

## Seed 41 results

Lower RMSE, MSE, and MAE are better; higher R2 is better.

| Scenario | Best | SVGP RMSE | DLinear+FRK RMSE | STDK RMSE | STDK+Q RMSE |
|---|---|---:|---:|---:|---:|
| Time150 | SVGP | 3.248717 | 3.791848 | 4.293592 | 5.341084 |
| Space100 | DLinear+FRK | 3.169631 | 3.095808 | 5.369811 | 4.652150 |
| ST100x150 | SVGP | 3.203612 | 3.735412 | 4.322475 | 5.525229 |

STDK+Q improves over standalone STDK only in Space100 for this seed. It is
worse in Time150 and ST100x150. In the STDK+Q run, the Q stage also worsens the
same-run fitted STDK in Time150 (4.528860 to 5.341084) and ST100x150 (4.493609
to 5.525229), while improving Space100 (5.287873 to 4.652150).

These are diagnostic seed 41 results. The zero standard deviations printed by
the runner mean that only one seed was supplied; they are not multi-seed
stability estimates. Seeds 42--45 are still required for the final mean and
standard deviation comparison.

## Earlier separate-run comparison

`tables/onefit_vs_separate_seed41.csv` compares these results with the seed 41
files in `../four_models_seed41_20260831/`, where each scenario invoked a
separate training run. DLinear+FRK is numerically identical. SVGP changes by
only 1.58--2.52% in RMSE. STDK changes by less than 0.5% for Time150 and
ST100x150, but its Space100 RMSE is 24.35% worse.

The STDK comparison does not isolate the effect of one-fit evaluation: the new
run also enables the pinned spatial-adapter trainer's EMA, whereas the August
run did not use EMA. Batch-mean validation was already present in standalone
STDK; it was newly aligned in the STDK front end inside STDK+Q, not newly added
to this standalone comparison. A controlled causal comparison would require
running the current STDK code with EMA on and off. The older QConvLSTM file is
also a different model version and was already evaluated as one fitted model,
so it is not included as a separate-training counterpart.

## Relation to pinned spatial-adapter

The pinned `kaust.yaml` supplies `epochs=350`, `lr=1e-3`,
`weight_decay=1e-4`, `batch_size=512`, and `patience=30`. It does not contain a
`use_ema` field. EMA comes from the same pinned repository's generic STDK
trainer, where missing `use_ema` defaults to true and decay is
`1 - 1 / (10 * batches_per_epoch)`. The current seed 41 run used decay
0.9998538011695907 and selected EMA checkpoint epoch 3 by batch-mean Val150
MSE.

Weather2K's Train700/Val150/Test150, obs100/unobserved400/held-out100 split,
obs100 x Train700 normalization, and one-fit/three-evaluation protocol are
project adaptations. They do not come from `kaust.yaml`.

## Runtime

- SVGP training: about 10 minutes 10 seconds.
- standalone STDK training: about 2 minutes 28 seconds.
- DLinear+FRK training: about 2 minutes 2 seconds.
- STDK+Q matching run: about 1 hour 50 minutes.

The first failed attempt is retained as
`logs/run_failed_missing_gpytorch.log`; the successful complete execution is
`logs/run.log`.
