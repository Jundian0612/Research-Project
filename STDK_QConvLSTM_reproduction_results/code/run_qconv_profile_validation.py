#!/usr/bin/env python3
"""Compare paper 3x3 and released GitHub 3-block QConvLSTM on rolling validation."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
import subprocess
import sys
from pathlib import Path


BLOCKS = {
    490: "19_chronological_validation_490_5",
    485: "20_chronological_validation_485_5",
}
METRICS = ("MSPE", "MPIW", "Coverage_percent")
BUNDLE_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    root = BUNDLE_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[41, 42])
    parser.add_argument(
        "--target-indices", type=int, nargs="+",
        default=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=root / "raw_results" / "23_qconv_profile_chronological_comparison",
    )
    return parser.parse_args()


def aggregate_path(directory: Path, cutoff: int, seed: int, qconv_tag: str) -> Path:
    return directory / (
        f"table2_aggregate_seed{seed}_repository_{qconv_tag}_papereq7_"
        f"nsteps5_train{cutoff}_validate5.json"
    )


def complete(path: Path, target_count: int, cutoff: int) -> bool:
    if not path.exists():
        return False
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        report.get("completed_locations") == target_count
        and report.get("prediction_count") == 5 * target_count
        and report.get("config", {}).get("train_times") == cutoff
        and report.get("evaluation", {}).get("role") == "chronological_validation"
    )


def run_missing(args: argparse.Namespace, cutoff: int, seed: int) -> Path:
    path = aggregate_path(args.output_dir, cutoff, seed, "kerascompat")
    if complete(path, len(args.target_indices), cutoff):
        print(f"[cutoff {cutoff} seed {seed}] complete; reusing {path.name}", flush=True)
        return path
    results_root = BUNDLE_ROOT / "raw_results"
    checkpoint_name = f"stdk_quantiles_seed{seed}_repository_kerascompat_papereq7.pt"
    source_checkpoint = results_root / BLOCKS[cutoff] / checkpoint_name
    target_checkpoint = args.output_dir / checkpoint_name
    if source_checkpoint.exists():
        shutil.copy2(source_checkpoint, target_checkpoint)
        print(
            f"[cutoff {cutoff} seed {seed}] reusing compatible Repository STDK checkpoint",
            flush=True,
        )
    command = [
        sys.executable, "-u", str(Path(__file__).resolve().parent / "STDK_QConvLSTM_reproduction.py"),
        "--train-times", str(cutoff),
        "--stdk-profile", "repository",
        "--qconv-profile", "github_3block",
        "--grid-size", "8",
        "--neighbourhood-radius", "0.20",
        "--grid-boundary", "shift",
        "--seed", str(seed),
        "--target-indices", *map(str, args.target_indices),
        "--output-dir", str(args.output_dir),
    ]
    print(f"[cutoff {cutoff} seed {seed}] starting GitHub 3-block", flush=True)
    subprocess.run(command, check=True)
    if not complete(path, len(args.target_indices), cutoff):
        raise RuntimeError(f"Run finished without a complete aggregate: {path}")
    return path


def main() -> None:
    args = parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must be unique")
    if len(set(args.target_indices)) != len(args.target_indices):
        raise ValueError("--target-indices must be unique")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_root = BUNDLE_ROOT / "raw_results"

    github_paths = {
        (cutoff, seed): run_missing(args, cutoff, seed)
        for cutoff in BLOCKS
        for seed in args.seeds
    }
    rows, station_differences = [], []
    for cutoff, paper_folder in BLOCKS.items():
        for seed in args.seeds:
            paths = {
                "paper_3x3": aggregate_path(results_root / paper_folder, cutoff, seed, "paper3x3"),
                "github_3block": github_paths[(cutoff, seed)],
            }
            reports = {
                profile: json.loads(path.read_text(encoding="utf-8"))
                for profile, path in paths.items()
            }
            for profile, report in reports.items():
                rows.append({
                    "validation_times": f"{cutoff + 1}-{cutoff + 5}",
                    "cutoff": cutoff,
                    "seed": seed,
                    "qconv_profile": profile,
                    **{metric: float(report["metrics"][metric]) for metric in METRICS},
                })

            target_metrics = {}
            for profile, directory in (
                ("paper_3x3", results_root / paper_folder),
                ("github_3block", args.output_dir),
            ):
                tag = "paper3x3" if profile == "paper_3x3" else "kerascompat"
                target_metrics[profile] = {}
                pattern = (
                    f"qconvlstm_target*_seed{seed}_repository_{tag}_papereq7_"
                    f"nsteps5_train{cutoff}_validate5.json"
                )
                for path in directory.glob(pattern):
                    report = json.loads(path.read_text(encoding="utf-8"))
                    target_metrics[profile][int(report["target_index"])] = float(report["metrics"]["MSPE"])
            for target in args.target_indices:
                station_differences.append(
                    target_metrics["github_3block"][target]
                    - target_metrics["paper_3x3"][target]
                )

    summaries = {}
    for profile in ("paper_3x3", "github_3block"):
        subset = [row for row in rows if row["qconv_profile"] == profile]
        summaries[profile] = {
            metric: {
                "mean": statistics.fmean(row[metric] for row in subset),
                "sample_std": statistics.stdev(row[metric] for row in subset),
                "values": [row[metric] for row in subset],
            }
            for metric in METRICS
        }
    result = {
        "selection_data": "rolling chronological validation only; final times 496--500 excluded",
        "fixed_stdk_profile": "repository",
        "blocks": ["486-490", "491-495"],
        "seeds": args.seeds,
        "target_indices": args.target_indices,
        "rows": rows,
        "summaries": summaries,
        "paired_station_mspe": {
            "pair_count": len(station_differences),
            "github_3block_wins": sum(value < 0 for value in station_differences),
            "paper_3x3_wins": sum(value > 0 for value in station_differences),
            "mean_github_minus_paper": statistics.fmean(station_differences),
            "median_github_minus_paper": statistics.median(station_differences),
        },
    }
    json_path = args.output_dir / "qconv_profile_validation_comparison.json"
    csv_path = args.output_dir / "qconv_profile_validation_comparison.csv"
    md_path = args.output_dir / "RESULTS.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# QConvLSTM profile rolling-validation comparison", "",
        "Final times 496--500 are excluded. STDK is fixed to the repository profile.", "",
        "| QConvLSTM | MSPE mean | MSPE std | MPIW mean | Coverage mean |",
        "|---|---:|---:|---:|---:|",
    ]
    for profile in ("paper_3x3", "github_3block"):
        summary = summaries[profile]
        lines.append(
            f"| {profile} | {summary['MSPE']['mean']:.6f} | "
            f"{summary['MSPE']['sample_std']:.6f} | {summary['MPIW']['mean']:.6f} | "
            f"{summary['Coverage_percent']['mean']:.2f}% |"
        )
    paired = result["paired_station_mspe"]
    lines.extend((
        "", f"Paired station results: GitHub 3-block wins {paired['github_3block_wins']}/"
        f"{paired['pair_count']}; paper 3x3 wins {paired['paper_3x3_wins']}/"
        f"{paired['pair_count']}.", "",
    ))
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved {json_path}")
    print(f"Saved {csv_path}")
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
