#!/usr/bin/env python3
"""Compare a formal STDK refit with the curated author-data QConvLSTM run.

The curated repository contains QConvLSTM forecasts but not the original raw
STDK checkpoints.  This program deterministically refits the exact STDK stage
for each seed, evaluates it at times 496--500, and pairs those forecasts with
the saved QConvLSTM forecasts for the same seed, location, lead, and truth.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

CODE_DIR = Path(__file__).resolve().parent
BUNDLE_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import STDK_QConvLSTM_reproduction as reproduction


TAG = "repository_kerascompat_papereq7_nsteps5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43, 44, 45])
    parser.add_argument("--data-dir", type=Path, default=BUNDLE_ROOT / "data")
    parser.add_argument("--qconv-dir", type=Path, default=BUNDLE_ROOT / "per_seed")
    parser.add_argument(
        "--output-dir", type=Path,
        default=BUNDLE_ROOT / "stdk_vs_qconvlstm",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--quick", action="store_true",
        help="Two-epoch pipeline check; its STDK scores are not comparable with the formal QConvLSTM scores.",
    )
    return parser.parse_args()


def load_formal_config(qconv_dir: Path, seed: int, quick: bool):
    aggregate_path = qconv_dir / f"table2_aggregate_seed{seed}_{TAG}.json"
    payload = json.loads(aggregate_path.read_text(encoding="utf-8"))
    allowed = {field.name for field in fields(reproduction.Config)}
    config_values = {key: value for key, value in payload["config"].items() if key in allowed}
    cfg = reproduction.Config(**config_values)
    if quick:
        cfg.stdk_epochs = 2
        cfg.stdk_patience = 2
    return cfg, aggregate_path


def load_qconv_forecasts(qconv_dir: Path, seed: int) -> list[dict]:
    path = qconv_dir / f"table2_forecasts_seed{seed}_{TAG}.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 500:
        raise ValueError(f"Expected 500 QConvLSTM forecasts in {path}, found {len(rows)}")
    return rows


def checkpoint_compatible(checkpoint: dict, cfg) -> bool:
    saved = checkpoint.get("config", {})
    keys = (
        "seed", "train_times", "stdk_epochs", "stdk_batch_size", "stdk_lr",
        "stdk_patience", "stdk_activity_l2", "stdk_regularization_profile",
        "stdk_validation_fraction", "add_paper_nonstationary_mean",
        "stdk_profile", "validation_mode", "normalization", "quantile_lambda",
    )
    return all(saved.get(key) == getattr(cfg, key) for key in keys)


def fit_or_load_stdk(cfg, values_model_scale, output_dir: Path, overwrite: bool, device):
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"refit_stdk_quantiles_seed{cfg.seed}_{TAG}.pt"
    train_time_idx = np.arange(cfg.train_times)
    features = reproduction.make_stdk_features(
        np.asarray(_COORDS), train_time_idx, len(values_model_scale), cfg.stdk_profile
    )
    targets = values_model_scale[: cfg.train_times].reshape(-1)
    checkpoint = None
    if checkpoint_path.exists() and not overwrite:
        candidate = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint_compatible(candidate, cfg):
            checkpoint = candidate
            print(f"[seed {cfg.seed}] reusing {checkpoint_path.name}", flush=True)
    if checkpoint is None:
        reproduction.seed_everything(cfg.seed)
        print(f"[seed {cfg.seed}] fitting q50 STDK", flush=True)
        median = reproduction.fit_stdk(features, targets, cfg, device)
        print(f"[seed {cfg.seed}] fitting q05 STDK", flush=True)
        lower = reproduction.fit_stdk_tail(features, targets, median, 0.05, cfg, device)
        print(f"[seed {cfg.seed}] fitting q95 STDK", flush=True)
        upper = reproduction.fit_stdk_tail(features, targets, median, 0.95, cfg, device)
        checkpoint = {
            "state_dict_q05_raw": lower.state_dict(),
            "state_dict_q50": median.state_dict(),
            "state_dict_q95_raw": upper.state_dict(),
            "config": vars(cfg),
            "provenance": "deterministic refit because the curated original STDK checkpoint is unavailable",
        }
        torch.save(checkpoint, checkpoint_path)
    else:
        input_dim = features.shape[1]
        median = reproduction.STDK(input_dim).to(device)
        lower = reproduction.STDK(input_dim).to(device)
        upper = reproduction.STDK(input_dim).to(device)
        median.load_state_dict(checkpoint["state_dict_q50"])
        lower.load_state_dict(checkpoint["state_dict_q05_raw"])
        upper.load_state_dict(checkpoint["state_dict_q95_raw"])
    return {0.05: lower, 0.5: median, 0.95: upper}, checkpoint_path


def predict_stdk(models, cfg, coords, total_times: int, device) -> np.ndarray:
    time_idx = np.arange(cfg.train_times, cfg.train_times + cfg.horizon)
    features = reproduction.make_stdk_features(coords, time_idx, total_times, cfg.stdk_profile)
    x = torch.from_numpy(features).to(device)
    for model in models.values():
        model.eval()
    with torch.no_grad():
        median = models[0.5](x)
        lower = reproduction.constrained_quantile(
            models[0.05](x), median, 0.05, cfg.quantile_lambda
        )
        upper = reproduction.constrained_quantile(
            models[0.95](x), median, 0.95, cfg.quantile_lambda
        )
    # make_stdk_features is time-major; return location-major to match the CSV.
    prediction = torch.cat((lower, median, upper), dim=1).cpu().numpy()
    return prediction.reshape(cfg.horizon, len(coords), 3).transpose(1, 0, 2)


def metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    flat_prediction = prediction.reshape(-1, 3)
    flat_truth = truth.reshape(-1)
    result = reproduction.evaluate(flat_prediction, flat_truth)
    return {
        "MSPE": result["MSPE"], "RMSE": result["RMSE"], "MAE": result["MAE"],
        "MPIW": result["MPIW_90"],
        "Coverage_percent": 100.0 * result["coverage_90"],
    }


def run_seed(args: argparse.Namespace, seed: int, device) -> dict:
    cfg, aggregate_path = load_formal_config(args.qconv_dir, seed, args.quick)
    coords, _, values_raw = reproduction.load_simulation(
        args.data_dir, cfg.add_paper_nonstationary_mean
    )
    global _COORDS
    _COORDS = coords
    if cfg.normalization == "train_global":
        mean = float(values_raw[: cfg.train_times].mean())
        std = float(values_raw[: cfg.train_times].std()) or 1.0
        values_model_scale = ((values_raw - mean) / std).astype(np.float32)
    else:
        mean, std = 0.0, 1.0
        values_model_scale = values_raw
    models, checkpoint_path = fit_or_load_stdk(
        cfg, values_model_scale, args.output_dir, args.overwrite, device
    )
    stdk_prediction = predict_stdk(models, cfg, coords, len(values_raw), device)
    stdk_prediction = stdk_prediction * std + mean
    truth = values_raw[cfg.train_times : cfg.train_times + cfg.horizon].T

    q_rows = load_qconv_forecasts(args.qconv_dir, seed)
    q_prediction = np.empty_like(stdk_prediction)
    q_truth = np.empty_like(truth)
    for row in q_rows:
        location = int(row["target_index"])
        lead = int(row["lead"]) - 1
        q_truth[location, lead] = float(row["truth"])
        q_prediction[location, lead] = [float(row["q05"]), float(row["q50"]), float(row["q95"])]
    if not np.allclose(q_truth, truth, rtol=0, atol=1e-6):
        raise AssertionError("Saved QConvLSTM truth does not match the refit STDK test truth")

    per_seed_dir = args.output_dir / "per_seed"
    per_seed_dir.mkdir(parents=True, exist_ok=True)
    paired_path = per_seed_dir / f"stdk_vs_qconvlstm_seed{seed}_{TAG}.csv"
    with paired_path.open("w", newline="", encoding="utf-8") as handle:
        fields_out = (
            "seed", "target_index", "lead", "time_index", "truth",
            "stdk_q05", "stdk_q50", "stdk_q95",
            "qconv_q05", "qconv_q50", "qconv_q95",
        )
        writer = csv.DictWriter(handle, fieldnames=fields_out)
        writer.writeheader()
        for location in range(len(coords)):
            for lead in range(cfg.horizon):
                writer.writerow({
                    "seed": seed, "target_index": location, "lead": lead + 1,
                    "time_index": cfg.train_times + lead + 1,
                    "truth": float(truth[location, lead]),
                    "stdk_q05": float(stdk_prediction[location, lead, 0]),
                    "stdk_q50": float(stdk_prediction[location, lead, 1]),
                    "stdk_q95": float(stdk_prediction[location, lead, 2]),
                    "qconv_q05": float(q_prediction[location, lead, 0]),
                    "qconv_q50": float(q_prediction[location, lead, 1]),
                    "qconv_q95": float(q_prediction[location, lead, 2]),
                })

    stdk_metrics = metrics(stdk_prediction, truth)
    qconv_metrics = metrics(q_prediction, truth)
    report = {
        "seed": seed,
        "comparison_scope": "same seed/config/data; STDK is a deterministic refit because the original raw checkpoint was not curated",
        "quick_diagnostic": bool(args.quick),
        "source_qconv_aggregate": str(aggregate_path),
        "refit_stdk_checkpoint": str(checkpoint_path),
        "evaluation": {"locations": 100, "times": [496, 500], "predictions": 500},
        "STDK": stdk_metrics,
        "STDK_plus_QConvLSTM": qconv_metrics,
        "Q_minus_STDK": {key: qconv_metrics[key] - stdk_metrics[key] for key in stdk_metrics},
        "q_improves_point_forecast": qconv_metrics["MSPE"] < stdk_metrics["MSPE"],
    }
    report_path = per_seed_dir / f"stdk_vs_qconvlstm_seed{seed}_{TAG}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def write_summary(reports: list[dict], output_dir: Path, quick: bool) -> None:
    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    metric_names = ("MSPE", "RMSE", "MAE", "MPIW", "Coverage_percent")
    summary = {"seeds": [report["seed"] for report in reports], "quick_diagnostic": quick, "models": {}}
    for model in ("STDK", "STDK_plus_QConvLSTM"):
        summary["models"][model] = {
            metric: {
                "mean": statistics.fmean(report[model][metric] for report in reports),
                "std_sample_ddof1": (
                    statistics.stdev(report[model][metric] for report in reports)
                    if len(reports) > 1 else None
                ),
            }
            for metric in metric_names
        }
    stdk_mspe = summary["models"]["STDK"]["MSPE"]["mean"]
    q_mspe = summary["models"]["STDK_plus_QConvLSTM"]["MSPE"]["mean"]
    summary["mean_MSPE_Q_minus_STDK"] = q_mspe - stdk_mspe
    summary["q_improves_mean_point_forecast"] = q_mspe < stdk_mspe
    (summary_dir / "stdk_vs_qconvlstm_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    lines = [
        "# Author-data fitted STDK vs STDK + QConvLSTM", "",
        "Artifacts are separated into `checkpoints/`, `per_seed/`, `summary/`, "
        "`logs/` and `metadata/`.", "",
        "| Model | MSPE mean | RMSE mean | MAE mean | MPIW mean | Coverage (%) mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in ("STDK", "STDK_plus_QConvLSTM"):
        item = summary["models"][model]
        lines.append(
            f"| {model} | {item['MSPE']['mean']:.6f} | {item['RMSE']['mean']:.6f} | "
            f"{item['MAE']['mean']:.6f} | {item['MPIW']['mean']:.6f} | "
            f"{item['Coverage_percent']['mean']:.3f} |"
        )
    lines.extend((
        "", f"QConvLSTM minus STDK mean MSPE: `{summary['mean_MSPE_Q_minus_STDK']:.6f}`.",
        "", "The STDK stage is deterministically refitted because the curated original raw checkpoint is unavailable.", "",
    ))
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


_COORDS: np.ndarray


def main() -> None:
    args = parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("seeds must be unique")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    reports = [run_seed(args, seed, device) for seed in args.seeds]
    write_summary(reports, args.output_dir, args.quick)


if __name__ == "__main__":
    main()
