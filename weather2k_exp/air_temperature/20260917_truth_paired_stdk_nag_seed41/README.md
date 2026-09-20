# Paired pure STDK / STDK+QConvLSTM — seed 41

This is the formal seed-41 run using one Spatial Adapter STDK checkpoint per
scenario. The QConvLSTM stage loads and freezes that same checkpoint. Q settings
are fixed in `locked_q_params.json`; Test truth is used for final evaluation.
The full protocol and checks are described in [PREFLIGHT.md](PREFLIGHT.md).

| Location | Contents |
|---|---|
| [`tables/`](tables/) | Three scenario result tables and pure-STDK versus Q comparisons. |
| [`metrics/`](metrics/) | Pure and Q JSON reports, pair verification, and Stage-1 prediction audit. |
| [`forecasts/`](forecasts/) | Raw-scale Q and fitted-STDK forecasts for the three test scenarios. |
| `checkpoints/` | Local fitted STDK checkpoints; `.pt` files are excluded from Git. |
| `run_seed41.sh`, `verify_seed41.py`, `audit_stage1_predictions.py` | Launcher and independent checks. |

The run finished with `exit_code=0`. `code.sha256` records the model and runner
source snapshot; `params.sha256` records the locked Q file. `run.log` is kept
locally and excluded from Git. The [`seed 41–45 summary`](../20260918_truth_paired_stdk_nag_seeds42to45/tables/five_seed_summary.md)
combines this run with the four later seeds.

The runner initially writes scenario outputs in this directory. The launcher
validates each pair, then `weather2k_exp/organize_paired_stdk_results.py` moves
completed reports into the directories above. Checkpoint paths stay unchanged.
