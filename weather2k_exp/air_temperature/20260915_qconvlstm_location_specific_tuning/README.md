# Location-specific STDK+QConvLSTM tuning (2026-09-15/16)

## Status

Completed on the RTX 2000 Ada. All 9 interface trials and 4 capacity trials
finished for seeds 41 and 42. Selection used chronological Val150 only; neither
Test150 nor held-out100 responses were used to choose parameters.

The tuning path trained q50 only. Each trial fitted an independent QConvLSTM for
every held-out target location, using that location's fitted-STDK sequence and
local grids. The front STDK used pinball loss, batch-mean validation and EMA
checkpoint weights.

## Selected configuration

| Parameter | Value |
| --- | ---: |
| Grid size | 11 |
| Neighbourhood radius | 0.3 |
| Convolution filters | 32 |
| Weight decay | 0 |
| Learning rate | 0.0001 |
| Batch size | 64 |
| Epoch limit | 40 |
| Patience | 10 |
| Lookback / horizon | 5 / 5 |

The selected capacity trial obtained scaled q50 Val150 RMSE 0.32721659 for seed
41 and 0.29340894 for seed 42, with mean 0.31031277 and population standard
deviation 0.01690383.

Both selected spatial parameters are at the upper boundary of the searched
range. The result is valid within the tested 5/8/11 grid sizes and 0.1/0.2/0.3
radii, but it does not establish that a larger grid or radius would be worse.

## Contents

- `summary/`: compact stage summaries and selected parameter JSON files.
- `params/formal_params.json`: snapshot copied to
  `weather2k_exp/2K_best_stdk_qconvlstm_params_500to100.json` for formal runs.
- `trials/`: per-trial parameters, summaries and seed validation reports.
- `logs/run.log`: full console output; retained locally and ignored by Git.
- `metadata/`: code commit, source hashes, Python version and package snapshot.

This directory is an archive of a completed run. The raw stage directories were
moved under `trials/`; do not use this archived layout as a resume directory.
