# Seeds 41–45 paired STDK / STDK+Q audit

All 15 seed × scenario pairs passed the independent checks in
`audit_all_stage1_predictions.py` and `audit_all_protocol.py`. The machine-readable
records are `all_five_seed_stage1_prediction_audit.json` and
`all_five_seed_protocol_audit.json`. Source-file checksums still match the
formal-run `code.sha256` snapshot.

| Seed | Pure/Q Stage-1 checkpoint SHA-256 prefix | Largest absolute Stage-1 prediction difference | Saved scaler mean / SD | Best STDK EMA epoch |
|---:|---|---:|---:|---:|
| 41 | `d70b51b4a29b37fe` | 0.000003815 | 23.091389 / 6.660942 | 3 |
| 42 | `4b65f1e290cbbfb3` | 0.000003815 | 23.408493 / 6.664432 | 4 |
| 43 | `e073273d38146d03` | 0.000003815 | 23.625278 / 6.107857 | 4 |
| 44 | `34a52e1c32ec5d8e` | 0.000003815 | 23.019497 / 6.352058 | 3 |
| 45 | `af72860bdcc8fd82` | 0.000003815 | 22.389196 / 6.760205 | 4 |

1. **Stage-1 STDK:** For every pair, the file SHA-256 equals the pure-STDK
   report, checkpoint metadata, and STDK+Q report SHA-256; the pure and Q
   checkpoint paths are identical. The independently recomputed pure-STDK
   outputs match all 875,000 saved Stage-1 outputs to within 3.82×10⁻⁶.
   They are numerically equal, not byte-for-byte equal after float32 inference.
2. **Split:** Pure metadata, pure sampling record, and Q report contain the
   same sampled 600 global station IDs, train500 IDs, held-out100 IDs, and
   obs100 IDs for each seed. Within a seed these IDs are also the same across
   all three scenarios. Full lists and their SHA-256 digests are in the
   protocol JSON. All runs use 1,000 times: Train 0–699, Val 700–849, Test
   850–999.
3. **Normalization:** Both paths save exactly the same mean and SD for each
   seed, with `ddof=0`, using obs100 × Train700. Recomputing from the raw
   Weather2K array agrees to float32 reduction tolerance (<1×10⁻⁵); changes
   in array memory order can change the final few float32 bits.
4. **STDK validation/EMA:** Pure-STDK metadata, the pure result, and Q's
   `stdk_validation.q50` contain the identical best epoch and Val150 MSE.
   Spatial Adapter's trainer evaluates EMA-shadow weights and saves the best
   Val150-loss EMA weights to `model_best.pt`. Q loads and freezes that file;
   its report says `refitted=false`.
5. **Q truth roles:** Train700 labels are from supervised train500 stations.
   Each target-specific q05/q50/q95 model selects its checkpoint with
   Val150 pinball loss on a supervised source station. Held-out targets use
   the nearest train500 source station for Q labels. Global Q settings were
   selected by q50 Val150 RMSE in validation-only tuning runs for seeds 41
   and 42; their reports have no test results. Formal code obtains target
   truth after predictions and uses it for final RMSE, MAE, R², interval
   width, coverage, and forecast files; it does not use it for tuning.
6. **Test target:** The formal forecast CSVs contain exactly the matching
   station/time Cartesian products and raw Weather2K truths: Time = train500
   × Test150 (75,000 points/seed), Space = held-out100 × times 0–849
   (85,000), and Space-Time = the same held-out100 × Test150 (15,000).
   Space's Val-time targets are different stations from the supervised
   Val150 stations used for checkpoint selection.
7. **Locked Q settings:** The seed-41, seed-42–45, and tuning-best parameter
   files are byte-identical, SHA-256
   `6b887ec5cb7973964426d1157b567abc43a5ae4bf6a803efac6d9af491bd9266`.
   Every formal Q report has the same eight configured values: LR 0.001,
   batch 5, max epochs 25, patience 5, filters 32, weight decay 1e-5,
   grid 5, radius 0.3. Per-target best epochs are still selected on that
   seed's Val150 truth, as intended. The quantile constraint scale λ is
   computed by one fixed formula from each seed's **Train700** range; its
   value therefore varies by seed but is never chosen from Test scores.

Seed 42 participated in Val150 hyperparameter selection. The five-seed
summary is therefore a repeatability comparison of the locked protocol, not
five untouched hyperparameter-selection seeds. No Test150 or held-out100
test metric entered global parameter selection.
