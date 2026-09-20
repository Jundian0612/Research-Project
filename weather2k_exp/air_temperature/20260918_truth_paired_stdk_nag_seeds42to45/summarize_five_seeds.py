#!/usr/bin/env python3
"""Summarize validated seed 41–45 paired RMSEs; population SD matches repo tables."""
import csv
import hashlib
import json
import statistics
from pathlib import Path

root = Path(__file__).resolve().parent
first = root.parent / "20260917_truth_paired_stdk_nag_seed41"
assert hashlib.sha256((root / "locked_q_params.json").read_bytes()).digest() == hashlib.sha256(
    (first / "locked_q_params.json").read_bytes()
).digest()
assert (first / "exit_code").read_text().strip() == "0"
scenarios = [
    ("time_extrap_fixed500", "Time150"),
    ("space_extrap_fixed850", "Space100"),
    ("spatiotemp_100x150", "ST100x150"),
]
rows = []
summary = []
for scenario, label in scenarios:
    paired = []
    for seed in range(41, 46):
        directory = first if seed == 41 else root / f"seed{seed}"
        record = json.loads((directory / f"verified_{scenario}.json").read_text())
        assert record["seed"] == seed and record["scenario"] == scenario
        pure, q = record["pure_stdk_rmse"], record["qconv_rmse"]
        rows.append({"scenario": label, "seed": seed, "pure_stdk_rmse": pure,
                     "stdk_qconv_rmse": q, "q_minus_pure_rmse": q - pure,
                     "checkpoint_sha256": record["checkpoint_sha256"]})
        paired.append((pure, q))
    pure_values = [p for p, _ in paired]
    q_values = [q for _, q in paired]
    deltas = [q - p for p, q in paired]
    summary.append({
        "scenario": label, "seed_count": 5,
        "pure_stdk_rmse_mean": statistics.mean(pure_values),
        "pure_stdk_rmse_sd": statistics.pstdev(pure_values),
        "stdk_qconv_rmse_mean": statistics.mean(q_values),
        "stdk_qconv_rmse_sd": statistics.pstdev(q_values),
        "paired_q_minus_pure_rmse_mean": statistics.mean(deltas),
        "paired_q_minus_pure_rmse_sd": statistics.pstdev(deltas),
    })

with (root / "five_seed_per_seed.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0])
    writer.writeheader()
    writer.writerows(rows)
with (root / "five_seed_summary.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=summary[0])
    writer.writeheader()
    writer.writerows(summary)
(root / "five_seed_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
with (root / "five_seed_summary.md").open("w") as handle:
    handle.write("# Paired pure STDK vs STDK+Q, seeds 41–45\n\n")
    handle.write("Q hyperparameters are the same locked file for all seeds. ")
    handle.write("SD uses ddof=0, matching the repository's summary convention.\n\n")
    handle.write("| Scenario | Pure STDK RMSE mean ± SD | STDK+Q RMSE mean ± SD | Paired Q−Pure mean ± SD |\n")
    handle.write("|---|---:|---:|---:|\n")
    for item in summary:
        handle.write(
            f"| {item['scenario']} | "
            f"{item['pure_stdk_rmse_mean']:.6f} ± {item['pure_stdk_rmse_sd']:.6f} | "
            f"{item['stdk_qconv_rmse_mean']:.6f} ± {item['stdk_qconv_rmse_sd']:.6f} | "
            f"{item['paired_q_minus_pure_rmse_mean']:+.6f} ± "
            f"{item['paired_q_minus_pure_rmse_sd']:.6f} |\n"
        )
print(json.dumps(summary, indent=2))
