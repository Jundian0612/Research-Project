#!/usr/bin/env python3
"""Tune Nag-style QConvLSTM on frozen pure Spatial Adapter STDK checkpoints.

Stage 1 selects the STDK-to-QConvLSTM interface (grid radius and size).
Stage 2 selects QConvLSTM learning rate, capacity, and regularization
using the best stage-1 interface. Every trial independently fits q50 for each
held-out target using observed Train700/Val150 responses at its nearest
supervised train500 station and grids from the same frozen STDK checkpoint.
Selection uses mean chronological-Val150 truth RMSE over seeds 41 and 42.
Test150 and held-out100 responses are not used.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "2K_STDK_QConvLSTM.py"
STDK_MODEL = ROOT / "2K_STDK_500train_100test.py"
DEFAULT_OUTPUT = ROOT / "air_temperature/20260917_stdk_nag_truth_qconvlstm_tuning"
FORMAL_PARAMS = ROOT / "2K_stdk_nag_qconvlstm_params.json"

# Reuse the best optimization settings from the completed first tuning round
# while testing the inferred Weather2K spatial interface.
BASE_OPTIMIZATION = {
    "QCONV_LR": 1e-3,
    "QCONV_BATCH_SIZE": 5,
    "QCONV_EPOCHS": 25,
    "QCONV_PATIENCE": 5,
    "CONV_FILTERS": 64,
    "QCONV_WEIGHT_DECAY": 0.0,
}

INTERFACE_CANDIDATES = [
    {"GRID_SIZE": grid_size, "NEIGHBOURHOOD_RADIUS": radius}
    for radius, grid_size in itertools.product((0.1, 0.2, 0.3), (5, 8, 11))
]

CAPACITY_CANDIDATES = [
    {"QCONV_LR": lr, "CONV_FILTERS": filters, "QCONV_WEIGHT_DECAY": weight_decay}
    for lr, filters, weight_decay in itertools.product(
        (1e-4, 1e-3), (32, 64), (0.0, 1e-5)
    )
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[41, 42])
    parser.add_argument(
        "--stage", choices=("all", "interface", "capacity"), default="all",
        help="Run both stages, only stage 1, or resume with only stage 2.",
    )
    parser.add_argument(
        "--max-interface-trials", type=int, default=len(INTERFACE_CANDIDATES)
    )
    parser.add_argument(
        "--max-capacity-trials", type=int, default=len(CAPACITY_CANDIDATES)
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stdk-checkpoint-dir", type=Path,
                        help="Existing pure-STDK checkpoints; generated once under output-dir/checkpoints if omitted.")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def verify_code_snapshot(output):
    """Abort a long sweep if its recorded source files change mid-run."""
    manifest = output / "code.sha256"
    if not manifest.exists():
        return
    result = subprocess.run(
        ["sha256sum", "-c", str(manifest)], cwd=ROOT.parent,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if result.returncode:
        raise RuntimeError(
            "Source snapshot changed during Q tuning: "
            f"{(result.stdout + result.stderr).strip()}"
        )


def prefix_for(stage, trial_no, smoke):
    smoke_label = "smoke_" if smoke else ""
    return f"qconvlstm_{smoke_label}{stage}_trial{trial_no:04d}"


def stage_output_dir(output, stage):
    """Keep raw trials separate from the final selected parameter file."""
    number = "stage1" if stage == "interface" else "stage2"
    return output / f"{number}_{stage}"


def stage_best_path(output, stage, smoke):
    smoke_label = "smoke_" if smoke else ""
    return stage_output_dir(output, stage) / (
        f"qconvlstm_{smoke_label}{stage}_best_params.json"
    )


def compatible(report, params, seed, smoke):
    cfg = report.get("config", {})
    expected = {
        "seed": seed,
        "forecast_mode": "block5to5",
        "prediction_mode": "direct",
        "scenario": "spatiotemp_100x150",
        "lookback": 5,
        "horizon": 5,
        "grid_size": params["GRID_SIZE"],
        "radius": params["NEIGHBOURHOOD_RADIUS"],
        "qconv_lr": params["QCONV_LR"],
        "qconv_weight_decay": params["QCONV_WEIGHT_DECAY"],
        "qconv_batch": 8 if smoke else params["QCONV_BATCH_SIZE"],
        "qconv_epochs": 1 if smoke else params["QCONV_EPOCHS"],
        "qconv_patience": params["QCONV_PATIENCE"],
        "conv_filters": params["CONV_FILTERS"],
        "validation_only": True,
        "q50_only": True,
        "smoke": smoke,
        "stdk_backend": "spatial_adapter",
        "normalization_source": "obs100_train700",
        "stdk_q50_loss": "mse",
        "stdk_checkpoint_source": "pure_stdk",
        "qconv_training_target": "weather2k_observed_train500",
        "qconv_validation_target": "weather2k_observed_train500_val150",
        "heldout_q_source": "nearest_train500_station",
        "qconv_model_scope": "location_specific",
        "q50_checkpoint_selection": "pinball",
        "stdk_validation_aggregation": "batch_mean",
        "stdk_use_ema": True,
    }
    return (
        all(cfg.get(key) == value for key, value in expected.items())
        and report.get("tuning_quantiles") == [0.5]
    )


def summarize_seed(report):
    q50 = report["validation"]["q50"]
    return {
        "seed": report["config"]["seed"],
        "q50_val_truth_rmse_scaled": q50["validation_rmse_scaled"],
        "q50_validation_pinball": q50["validation_pinball"],
        "elapsed_seconds": report["elapsed_seconds"],
    }


def run_trial(stage, trial_no, params, seeds, output, checkpoint_dir, smoke):
    verify_code_snapshot(output)
    stage_output = stage_output_dir(output, stage)
    stage_output.mkdir(parents=True, exist_ok=True)
    prefix = prefix_for(stage, trial_no, smoke)
    write_json(stage_output / f"{prefix}_params.json", params)
    seed_metrics = []
    for seed in seeds:
        verify_code_snapshot(output)
        report_path = stage_output / f"{prefix}_seed{seed}_validation.json"
        if report_path.exists():
            report = read_json(report_path)
            if compatible(report, params, seed, smoke):
                print(f"[resume] stage={stage} trial={trial_no} seed={seed}", flush=True)
                seed_metrics.append(summarize_seed(report))
                continue

        command = [
            sys.executable, str(MODEL),
            "--forecast-mode", "block5to5",
            "--prediction-mode", "direct",
            "--scenario", "spatiotemp_100x150",
            "--validation-only", "--q50-only",
            "--seed", str(seed),
            "--output-dir", str(stage_output),
            "--stdk-checkpoint-dir", str(checkpoint_dir),
            "--grid-size", str(params["GRID_SIZE"]),
            "--neighbourhood-radius", str(params["NEIGHBOURHOOD_RADIUS"]),
            "--qconv-lr", str(params["QCONV_LR"]),
            "--qconv-weight-decay", str(params["QCONV_WEIGHT_DECAY"]),
            "--qconv-batch-size", str(params["QCONV_BATCH_SIZE"]),
            "--qconv-epochs", str(params["QCONV_EPOCHS"]),
            "--qconv-patience", str(params["QCONV_PATIENCE"]),
            "--conv-filters", str(params["CONV_FILTERS"]),
        ]
        if smoke:
            command.append("--smoke-test")
        print(
            f"[run] stage={stage} trial={trial_no} seed={seed} params={params}",
            flush=True,
        )
        completed = subprocess.run(command, cwd=ROOT.parent)
        verify_code_snapshot(output)
        if completed.returncode:
            raise RuntimeError(
                f"stage {stage} trial {trial_no} seed {seed} failed with "
                f"exit code {completed.returncode}"
            )
        generated = stage_output / (
            f"paired_stdk_truth_nag_qconvlstm_block5to5_spatiotemp_100x150_seed{seed}_validation.json"
        )
        if not generated.exists():
            raise FileNotFoundError(generated)
        generated.replace(report_path)
        seed_metrics.append(summarize_seed(read_json(report_path)))

    values = [item["q50_val_truth_rmse_scaled"] for item in seed_metrics]
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    result = {
        "stage": stage,
        "normalization_source": "obs100_train700",
        "stdk_q50_loss": "mse",
        "stdk_checkpoint_source": "pure_stdk",
        "qconv_training_target": "weather2k_observed_train500",
        "qconv_validation_target": "weather2k_observed_train500_val150",
        "heldout_q_source": "nearest_train500_station",
        "qconv_model_scope": "location_specific",
        "q50_checkpoint_selection": "pinball",
        "stdk_validation_aggregation": "batch_mean",
        "stdk_use_ema": True,
        "trial": trial_no,
        "params": params,
        "seeds": seeds,
        "selection_uses_test_metrics": False,
        "selection_uses_heldout100_truth": False,
        "tuning_quantiles": [0.5],
        "q50_val_truth_rmse_scaled_by_seed": values,
        "mean_q50_val_truth_rmse_scaled": mean,
        "std_q50_val_truth_rmse_scaled": variance**0.5,
        "seed_metrics": seed_metrics,
    }
    write_json(stage_output / f"{prefix}_summary.json", result)
    return result


def write_stage_outputs(output, stage, results, smoke):
    if not results:
        raise ValueError(f"No {stage} tuning results")
    smoke_label = "smoke_" if smoke else ""
    stage_output = stage_output_dir(output, stage)
    stage_output.mkdir(parents=True, exist_ok=True)
    csv_path = stage_output / f"qconvlstm_{smoke_label}{stage}_summary.csv"
    rows = []
    for result in results:
        row = {
            "stage": stage,
            "trial": result["trial"],
            "mean_q50_val_truth_rmse_scaled": result["mean_q50_val_truth_rmse_scaled"],
            "std_q50_val_truth_rmse_scaled": result["std_q50_val_truth_rmse_scaled"],
        }
        row.update(result["params"])
        rows.append(row)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    best = min(results, key=lambda item: item["mean_q50_val_truth_rmse_scaled"])
    write_json(stage_best_path(output, stage, smoke), best)
    return best


def write_formal_outputs(output, best, seeds, smoke):
    payload = {
        "stdk_backend": "spatial_adapter",
        "normalization_source": "obs100_train700",
        "stdk_q50_loss": "mse",
        "stdk_checkpoint_source": "pure_stdk",
        "qconv_training_target": "weather2k_observed_train500",
        "qconv_validation_target": "weather2k_observed_train500_val150",
        "heldout_q_source": "nearest_train500_station",
        "qconv_model_scope": "location_specific",
        "q50_checkpoint_selection": "pinball",
        "stdk_validation_aggregation": "batch_mean",
        "stdk_use_ema": True,
        "selection_status": (
            "smoke_truth_validation_plumbing" if smoke
            else "validation_truth_tuned_on_paired_stdk_checkpoint"
        ),
        "selection_metric": "mean_q50_val_truth_rmse_scaled",
        "selection_uses_test_metrics": False,
        "selection_uses_heldout100_truth": False,
        "tuning_seeds": seeds,
        "tuning_quantiles": [0.5],
        "tuning_protocol": [
            "stage1_interface_grid_size_and_neighbourhood_radius",
            "stage2_learning_rate_filters_and_weight_decay",
        ],
        "best_stage": best["stage"],
        "best_trial": best["trial"],
        "best_value": best["mean_q50_val_truth_rmse_scaled"],
        "best_std": best["std_q50_val_truth_rmse_scaled"],
        "formal_run_params": best["params"],
        "fixed_paper_flow": {
            "LOOKBACK": 5,
            "HORIZON": 5,
            "QCONV_PROFILE": "github_3block",
        },
    }
    best_name = (
        "qconvlstm_smoke_tuning_best_params.json"
        if smoke else "qconvlstm_tuning_best_params.json"
    )
    write_json(output / best_name, payload)
    if not smoke:
        write_json(FORMAL_PARAMS, payload)

    readme_name = "SMOKE_README.md" if smoke else "README.md"
    status = (
        "smoke-test plumbing check; not formal tuning"
        if smoke else "formal two-stage validation-only tuning"
    )
    (output / readme_name).write_text(
        "# STDK+QConvLSTM tuning\n\n"
        f"Status: {status}\n\n"
        f"Seeds: {seeds}\n\n"
        "Every trial trains q50 only. Selection uses the mean chronological "
        "Val150 q50 RMSE against observed train500-station truth for independently "
        "fitted held-out target-location models using nearest train500 proxies. "
        "Q validation-only trials do not use Test150 or held-out100 truth. "
        "Pure-STDK checkpoint preparation may write its own final metrics, "
        "but Q tuning does not use them.\n\n"
        "The front STDK is the exact pure-STDK MSE/EMA checkpoint; "
        "QConvLSTM tuning does not refit it.\n\n"
        "Stage 1 searches grid size and neighbourhood radius. Stage 2 uses "
        "the best interface and searches learning rate, filters, and weight "
        "decay. The best epoch is selected separately by truth-based Val150 "
        "pinball loss in every Q fit.\n\n"
        f"Best stage: {best['stage']}\n\n"
        f"Best trial: {best['trial']}\n\n"
        f"Best mean RMSE: {best['mean_q50_val_truth_rmse_scaled']:.8f}\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2), flush=True)


def limited(candidates, count):
    return candidates[:max(1, min(count, len(candidates)))]


def ensure_stdk_checkpoints(seeds, checkpoint_dir, output, smoke):
    """Fit each pure STDK once, before any QConvLSTM hyperparameter trial."""
    for seed in seeds:
        directory = checkpoint_dir / f"spatiotemp_100x150_seed{seed}"
        if (directory / "model_best.pt").is_file() and (directory / "metadata.json").is_file():
            continue
        if checkpoint_dir != output / "checkpoints":
            raise FileNotFoundError(f"Missing pure-STDK checkpoint: {directory}")
        env = os.environ.copy()
        env.update({
            "EXPERIMENT_SCENARIO": "spatiotemp_100x150",
            "WEATHER2K_OUTPUT_DIR": str(output),
            "SEED_LIST": json.dumps([seed]),
            "STDK_EPOCHS": "1" if smoke else "350",
        })
        if smoke:
            env.update({
                "N_SAMPLE_TARGET": "16", "N_TRAIN_TARGET": "4",
                "N_UNKNOWN_PRIMARY_TARGET": "8", "N_UNKNOWN_EVAL_TARGET": "4",
                "N_LAST": "50", "TIME_TRAIN_LEN": "30",
                "TIME_VAL_LEN": "10", "TIME_TEST_LEN": "10",
            })
        subprocess.run([sys.executable, str(STDK_MODEL)], cwd=ROOT.parent,
                       env=env, check=True)


def main():
    args = parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not args.seeds:
        raise SystemExit("At least one tuning seed is required")
    checkpoint_dir = (args.stdk_checkpoint_dir or output / "checkpoints").expanduser().resolve()
    verify_code_snapshot(output)
    ensure_stdk_checkpoints(args.seeds, checkpoint_dir, output, args.smoke_test)
    verify_code_snapshot(output)

    best_interface = None
    if args.stage in ("all", "interface"):
        interface_results = []
        for trial_no, interface in enumerate(
            limited(INTERFACE_CANDIDATES, args.max_interface_trials)
        ):
            params = dict(BASE_OPTIMIZATION)
            params.update(interface)
            interface_results.append(
                run_trial(
                    "interface", trial_no, params, args.seeds, output, checkpoint_dir,
                    args.smoke_test,
                )
            )
        best_interface = write_stage_outputs(
            output, "interface", interface_results, args.smoke_test
        )

    if args.stage == "interface":
        print(json.dumps(best_interface, indent=2), flush=True)
        return

    if best_interface is None:
        path = stage_best_path(output, "interface", args.smoke_test)
        if not path.exists():
            raise SystemExit(
                f"Cannot run capacity stage without {path}; "
                "run --stage interface first"
            )
        best_interface = read_json(path)
        if (best_interface.get("normalization_source") != "obs100_train700"
                or best_interface.get("q50_checkpoint_selection") != "pinball"
                or best_interface.get("stdk_q50_loss") != "mse"
                or best_interface.get("stdk_checkpoint_source") != "pure_stdk"
                or best_interface.get("qconv_training_target") != "weather2k_observed_train500"
                or best_interface.get("qconv_validation_target") != "weather2k_observed_train500_val150"
                or best_interface.get("heldout_q_source") != "nearest_train500_station"
                or best_interface.get("qconv_model_scope") != "location_specific"
                or best_interface.get("stdk_validation_aggregation") != "batch_mean"
                or best_interface.get("stdk_use_ema") is not True
                or best_interface.get("seeds") != args.seeds):
            raise SystemExit(
                "Stage 1 protocol/seeds mismatch: rerun interface with current "
                "normalization, paired MSE/EMA STDK checkpoint, batch-mean validation "
                "and EMA."
            )

    capacity_results = []
    for trial_no, capacity in enumerate(
        limited(CAPACITY_CANDIDATES, args.max_capacity_trials)
    ):
        params = dict(best_interface["params"])
        params.update(capacity)
        capacity_results.append(
            run_trial(
                "capacity", trial_no, params, args.seeds, output, checkpoint_dir,
                args.smoke_test,
            )
        )
    best_capacity = write_stage_outputs(
        output, "capacity", capacity_results, args.smoke_test
    )
    write_formal_outputs(output, best_capacity, args.seeds, args.smoke_test)


if __name__ == "__main__":
    main()
