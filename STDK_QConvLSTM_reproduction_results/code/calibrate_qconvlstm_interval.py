#!/usr/bin/env python3
"""Calibrate QConvLSTM interval width using chronological-validation CSVs only."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


BUNDLE_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    root = BUNDLE_ROOT / "raw_results"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dirs", type=Path, nargs="+",
        default=[
            root / "19_chronological_validation_490_5",
            root / "20_chronological_validation_485_5",
        ],
    )
    parser.add_argument("--profile", choices=("repository", "paper"), default="repository")
    parser.add_argument("--seeds", type=int, nargs="+", default=[41, 42])
    parser.add_argument("--target-coverage", type=float, default=0.90)
    parser.add_argument(
        "--output-dir", type=Path,
        default=root / "21_interval_calibration_repository_paper3x3",
    )
    return parser.parse_args()


def required_scale(row: dict[str, str]) -> float:
    truth, lower = float(row["truth"]), float(row["q05"])
    median, upper = float(row["q50"]), float(row["q95"])
    if truth < median:
        width = median - lower
        return math.inf if width <= 0 else (median - truth) / width
    width = upper - median
    return math.inf if width <= 0 else (truth - median) / width


def summarize(rows: list[dict[str, str]], scale: float) -> dict[str, float | int]:
    covered, widths, squared_errors = 0, [], []
    for row in rows:
        truth, lower = float(row["truth"]), float(row["q05"])
        median, upper = float(row["q50"]), float(row["q95"])
        calibrated_lower = median - scale * (median - lower)
        calibrated_upper = median + scale * (upper - median)
        covered += calibrated_lower <= truth <= calibrated_upper
        widths.append(calibrated_upper - calibrated_lower)
        squared_errors.append((truth - median) ** 2)
    return {
        "prediction_count": len(rows),
        "MSPE": sum(squared_errors) / len(squared_errors),
        "MPIW": sum(widths) / len(widths),
        "Coverage_percent": 100.0 * covered / len(rows),
    }


def main() -> None:
    args = parse_args()
    if not 0 < args.target_coverage <= 1:
        raise ValueError("--target-coverage must be in (0, 1]")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must be unique")

    rows, source_files = [], []
    for directory in args.input_dirs:
        for seed in args.seeds:
            pattern = f"table2_forecasts_seed{seed}_{args.profile}_paper3x3_*validate5.csv"
            matches = sorted(directory.glob(pattern))
            if len(matches) != 1:
                raise FileNotFoundError(
                    f"Expected one validation CSV for seed {seed} in {directory}; found {matches}"
                )
            source_files.append(matches[0])
            with matches[0].open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    row["source_file"] = str(matches[0])
                    row["seed"] = str(seed)
                    rows.append(row)

    needed = sorted(required_scale(row) for row in rows)
    required_covered = math.ceil(args.target_coverage * len(rows))
    selected_scale = needed[required_covered - 1]
    if not math.isfinite(selected_scale):
        raise RuntimeError("The requested coverage cannot be reached from the saved intervals")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["source_file"]].append(row)
    baseline = summarize(rows, 1.0)
    calibrated = summarize(rows, selected_scale)
    per_run = [
        {
            "source_file": source,
            "baseline": summarize(group, 1.0),
            "calibrated": summarize(group, selected_scale),
        }
        for source, group in sorted(grouped.items())
    ]
    result = {
        "selection_data": "chronological validation only; final times 496--500 excluded",
        "profile": args.profile,
        "qconv_profile": "paper_3x3",
        "seeds": args.seeds,
        "target_coverage_percent": 100.0 * args.target_coverage,
        "selected_interval_scale": selected_scale,
        "baseline": baseline,
        "calibrated": calibrated,
        "source_files": [str(path) for path in source_files],
        "per_run": per_run,
        "application": "Pass --interval-scale with the selected value in the final 1--495 training run.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "interval_calibration.json"
    csv_path = args.output_dir / "interval_calibration_per_run.csv"
    md_path = args.output_dir / "README.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = (
            "source_file", "baseline_MSPE", "baseline_MPIW", "baseline_Coverage_percent",
            "calibrated_MSPE", "calibrated_MPIW", "calibrated_Coverage_percent",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in per_run:
            writer.writerow({
                "source_file": item["source_file"],
                **{f"baseline_{key}": item["baseline"][key] for key in ("MSPE", "MPIW", "Coverage_percent")},
                **{f"calibrated_{key}": item["calibrated"][key] for key in ("MSPE", "MPIW", "Coverage_percent")},
            })
    md_path.write_text(
        "\n".join((
            "# Repository STDK + paper 3x3 interval calibration",
            "",
            "Calibration uses only rolling chronological-validation forecasts for times 486--495; final test times 496--500 are excluded.",
            "",
            f"- Selected interval scale: `{selected_scale:.9f}`",
            f"- Target pooled coverage: `{100.0 * args.target_coverage:.2f}%`",
            f"- Baseline: MSPE `{baseline['MSPE']:.6f}`, MPIW `{baseline['MPIW']:.6f}`, coverage `{baseline['Coverage_percent']:.2f}%`.",
            f"- Calibrated: MSPE `{calibrated['MSPE']:.6f}`, MPIW `{calibrated['MPIW']:.6f}`, coverage `{calibrated['Coverage_percent']:.2f}%`.",
            "- q50 is unchanged, so MSPE is unchanged.",
            "",
            "Use this value with `--interval-scale` only after the core model configuration has been fixed.",
            "",
        )),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    print(f"Saved {json_path}")
    print(f"Saved {csv_path}")
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
