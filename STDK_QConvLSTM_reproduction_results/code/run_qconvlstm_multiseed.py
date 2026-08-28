#!/usr/bin/env python3
"""Run QConvLSTM for multiple seeds and summarize seed-level mean and std."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path


METRICS = ("MSPE", "MPIW", "Coverage_percent")
PAPER = {"MSPE": 0.267, "MPIW": 1.462, "Coverage_percent": 90.39}
BUNDLE_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43, 44, 45])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--stdk-profile", choices=("repository", "paper"), default="repository")
    parser.add_argument("--validation-mode", choices=("repository_random", "chronological"), default="repository_random")
    parser.add_argument("--normalization", choices=("none", "train_global"), default="none")
    parser.add_argument("--grid-boundary", choices=("shift", "clip", "allow_outside"), default="shift")
    parser.add_argument("--quantile-lambda", type=float, default=None)
    parser.add_argument("--interval-scale", type=float, default=1.0)
    parser.add_argument(
        "--qconv-profile", choices=("github_3block", "paper_3x3"),
        default="github_3block",
    )
    parser.add_argument("--aggregate-only", action="store_true", help="Do not train; only summarize completed seeds")
    parser.add_argument("--overwrite", action="store_true", help="Retrain existing target files for every seed")
    return parser.parse_args()


def aggregate_path(
    output_dir: Path, seed: int, profile: str, qconv_profile: str,
    interval_scale: float,
) -> Path:
    qconv_tag = "kerascompat" if qconv_profile == "github_3block" else "paper3x3"
    tag = f"{profile}_{qconv_tag}_papereq7_nsteps5"
    if interval_scale != 1.0:
        tag += f"_intervalscale{interval_scale:g}".replace(".", "p")
    return output_dir / f"table2_aggregate_seed{seed}_{tag}.json"


def is_complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    config = report.get("config", {})
    return (
        report.get("completed_locations") == 100
        and report.get("prediction_count") == 500
        and config.get("stdk_validation_fraction") == 0.10
        and config.get("qconv_validation_fraction") == 0.05
        and config.get("stdk_activity_l2") == 1e-5
        and config.get("qconv_profile", "github_3block") in ("github_3block", "paper_3x3")
    )


def run_seed(args: argparse.Namespace, seed: int) -> None:
    path = aggregate_path(
        args.output_dir, seed, args.stdk_profile, args.qconv_profile,
        args.interval_scale,
    )
    if is_complete(path) and not args.overwrite:
        print(f"[seed {seed}] complete; reusing {path.name}", flush=True)
        return
    if args.aggregate_only:
        raise FileNotFoundError(f"seed {seed} is incomplete: {path}")

    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve().parent / "STDK_QConvLSTM_reproduction.py"),
        "--all-targets",
        "--seed", str(seed),
        "--output-dir", str(args.output_dir),
        "--stdk-profile", args.stdk_profile,
        "--validation-mode", args.validation_mode,
        "--normalization", args.normalization,
        "--grid-boundary", args.grid_boundary,
        "--qconv-profile", args.qconv_profile,
        "--interval-scale", str(args.interval_scale),
    ]
    if args.quantile_lambda is not None:
        command.extend(("--quantile-lambda", str(args.quantile_lambda)))
    if args.overwrite:
        command.append("--overwrite")
    print(f"[seed {seed}] starting", flush=True)
    subprocess.run(command, check=True)
    if not is_complete(path):
        raise RuntimeError(f"seed {seed} finished without a complete aggregate: {path}")


def summarize(args: argparse.Namespace) -> dict:
    rows = []
    for seed in args.seeds:
        path = aggregate_path(
            args.output_dir, seed, args.stdk_profile, args.qconv_profile,
            args.interval_scale,
        )
        report = json.loads(path.read_text(encoding="utf-8"))
        row = {"seed": seed, **{metric: float(report["metrics"][metric]) for metric in METRICS}}
        rows.append(row)

    summary_metrics = {}
    for metric in METRICS:
        values = [row[metric] for row in rows]
        summary_metrics[metric] = {
            "mean": statistics.fmean(values),
            "std_sample_ddof1": statistics.stdev(values) if len(values) > 1 else None,
            "values": values,
        }
    summary = {
        "seeds": args.seeds,
        "seed_count": len(args.seeds),
        "aggregation_unit": "one complete 100-location, 500-prediction experiment per seed",
        "std_definition": "sample standard deviation across seed-level metrics (ddof=1)",
        "metrics": summary_metrics,
        "paper_table2": PAPER,
        "interval_scale": args.interval_scale,
    }

    qconv_tag = "kerascompat" if args.qconv_profile == "github_3block" else "paper3x3"
    tag = f"seeds{'-'.join(map(str, args.seeds))}_{args.stdk_profile}_{qconv_tag}_papereq7_nsteps5"
    if args.interval_scale != 1.0:
        tag += f"_intervalscale{args.interval_scale:g}".replace(".", "p")
    json_path = args.output_dir / f"table2_multiseed_{tag}.json"
    csv_path = args.output_dir / f"table2_multiseed_{tag}.csv"
    md_path = args.output_dir / f"table2_multiseed_{tag}.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("seed", *METRICS))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# QConvLSTM multi-seed summary",
        "",
        f"Seeds: {', '.join(map(str, args.seeds))}",
        "",
        "| Metric | Mean | Sample std | Paper Table 2 |",
        "|---|---:|---:|---:|",
    ]
    for metric in METRICS:
        item = summary_metrics[metric]
        std_text = "N/A" if item["std_sample_ddof1"] is None else f'{item["std_sample_ddof1"]:.6f}'
        lines.append(
            f"| {metric} | {item['mean']:.6f} | {std_text} | {PAPER[metric]:.6f} |"
        )
    lines.extend(("", "The standard deviation is calculated across the five seed-level aggregate metrics (ddof=1).", ""))
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved {json_path}")
    print(f"Saved {csv_path}")
    print(f"Saved {md_path}")
    return summary


def main() -> None:
    args = parse_args()
    if args.output_dir is None:
        folder = (
            "16_full_paper_profile_pilot"
            if args.qconv_profile == "paper_3x3" and args.stdk_profile == "paper"
            else "22_selected_repository_paper3x3_final"
            if args.qconv_profile == "paper_3x3" and args.stdk_profile == "repository"
            else "06_stdkval10_qconvval05_current"
            if args.qconv_profile == "github_3block"
            else "09_paper_3x3_stdkval10_qconvval05"
        )
        args.output_dir = BUNDLE_ROOT / "raw_results" / folder
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("seeds must be unique")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        run_seed(args, seed)
    summarize(args)


if __name__ == "__main__":
    main()
