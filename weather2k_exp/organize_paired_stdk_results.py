#!/usr/bin/env python3
"""Place completed paired STDK/Q run outputs in browsable directories.

The model runners write into each seed's root directory. Move only their
completed, named outputs; keep checkpoints and run metadata at their original
paths so saved checkpoint references remain valid.
"""

import argparse
from pathlib import Path


def organize(directory: Path) -> dict[str, int]:
    counts = {"metrics": 0, "tables": 0, "forecasts": 0}
    for source in directory.iterdir():
        if not source.is_file():
            continue
        name = source.name
        if name.endswith("_forecasts.csv"):
            category = "forecasts"
        elif (
            name.startswith("results_") and source.suffix in {".csv", ".md"}
        ) or name.endswith("_comparison.csv") or name.startswith("five_seed_"):
            category = "tables"
        elif (
            (name.startswith("2K_stdk_metrics_") or name.startswith("paired_stdk_truth_nag_qconvlstm_"))
            and source.suffix == ".json"
        ) or (name.startswith("verified_") and source.suffix == ".json") or name in {
            "stage1_prediction_audit.json",
            "all_five_seed_stage1_prediction_audit.json",
            "all_five_seed_protocol_audit.json",
        }:
            category = "metrics"
        else:
            continue
        destination = directory / category / name
        destination.parent.mkdir(exist_ok=True)
        source.replace(destination)
        counts[category] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    root = args.run_root.resolve(strict=True)
    if not root.is_dir() or not (root / "locked_q_params.json").is_file():
        parser.error("run_root must be a paired STDK/Q output directory")
    directories = [root, *sorted(root.glob("seed4[2-5]"))]
    for directory in directories:
        counts = organize(directory)
        print(f"{directory}: " + ", ".join(f"{key}={value}" for key, value in counts.items()))


if __name__ == "__main__":
    main()
