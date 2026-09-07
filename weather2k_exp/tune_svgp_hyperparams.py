#!/usr/bin/env python3
"""Two-stage SVGP tuning using validation RMSE only.

Stage 1: kernel x learning rate with 1024 inducing points.
Stage 2: inducing-point count using the best stage-1 kernel and learning rate.
The final test metrics are never read when selecting hyperparameters.
"""

import csv
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULT_ROOT = Path(
    os.environ.get("WEATHER2K_OUTPUT_DIR", str(ROOT))
).expanduser().resolve()
SVGP_SCRIPT = ROOT / "2K_SVGP_500train_100test.py"
OUTPUT_DIR = ROOT / os.environ.get("SVGP_TUNING_DIR", "")
JSON_DIR = OUTPUT_DIR
SEEDS = os.environ.get("SVGP_TUNE_SEEDS", "[41, 42]")
EPOCHS = os.environ.get("SVGP_TUNE_EPOCHS", "500")
PATIENCE = os.environ.get("SVGP_TUNE_PATIENCE", "30")

STAGE1_KERNELS = [
    value.strip()
    for value in os.environ.get(
        "SVGP_TUNE_KERNELS",
        "rbf,matern,matern_periodic",
    ).split(",")
    if value.strip()
]
STAGE1_LRS = [
    value.strip()
    for value in os.environ.get(
        "SVGP_TUNE_LRS",
        "0.001,0.003,0.01",
    ).split(",")
    if value.strip()
]
STAGE2_INDUCING = [
    int(value.strip())
    for value in os.environ.get(
        "SVGP_TUNE_INDUCING",
        "512,1024,2048",
    ).split(",")
    if value.strip()
]


def safe_label(value: str) -> str:
    return value.replace(".", "p").replace("/", "_")


def root_result_path(suffix: str) -> Path:
    return RESULT_ROOT / f"2K_svgp_metrics_{suffix}.json"


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def run_config(stage: str, kernel: str, lr: str, inducing: int) -> dict:
    label = (
        f"{stage}_{kernel}_m{inducing}_lr{safe_label(lr)}_"
        f"e{EPOCHS}_seeds{safe_label(SEEDS)}"
    )
    suffix = f"tune_{label}"
    path = JSON_DIR / f"2K_svgp_metrics_{suffix}.json"

    expected = {
        "kernel": kernel,
        "lr": float(lr),
        "num_inducing": int(inducing),
        "epochs": int(EPOCHS),
    }
    if path.exists():
        payload = load_json(path)
        params = payload.get("params_used", {})
        if all(params.get(key) == value for key, value in expected.items()):
            print(f"[resume] {label}", flush=True)
            return summarize(stage, label, payload)

    # The short-period product kernel is substantially more ill-conditioned
    # than RBF/Matérn in float32 and requires stronger numerical jitter.
    default_jitter = "1e-2" if kernel == "matern_periodic" else "1e-4"
    env = os.environ.copy()
    env.update(
        {
            "EXPERIMENT_SCENARIO": "time_extrap_fixed500",
            "N_SAMPLE_TARGET": "600",
            "N_TRAIN_TARGET": "100",
            "N_UNKNOWN_PRIMARY_TARGET": "400",
            "N_UNKNOWN_EVAL_TARGET": "100",
            "N_LAST": "1000",
            "TIME_TRAIN_LEN": "700",
            "TIME_VAL_LEN": "150",
            "TIME_TEST_LEN": "150",
            "SEED_LIST": SEEDS,
            "SVGP_KERNEL": kernel,
            "SVGP_LR": lr,
            "SVGP_NUM_INDUCING": str(inducing),
            "SVGP_EPOCHS": EPOCHS,
            "SVGP_PATIENCE": PATIENCE,
            "SVGP_VARIATIONAL_JITTER": os.environ.get(
                "SVGP_TUNE_VARIATIONAL_JITTER", default_jitter
            ),
            "RESULT_SUFFIX": suffix,
            "WEATHER2K_OUTPUT_DIR": str(RESULT_ROOT),
        }
    )
    print(
        f"[run] {stage}: kernel={kernel}, lr={lr}, inducing={inducing}",
        flush=True,
    )
    completed = subprocess.run([sys.executable, str(SVGP_SCRIPT)], env=env)
    if completed.returncode != 0:
        print(
            f"[failed] {label}: exit code {completed.returncode}; continuing",
            flush=True,
        )
        return failed_summary(stage, label, kernel, lr, inducing, completed.returncode)
    generated_path = root_result_path(suffix)
    if not generated_path.exists():
        raise FileNotFoundError(f"Expected tuning result was not created: {generated_path}")
    generated_path.replace(path)
    return summarize(stage, label, load_json(path))


def failed_summary(
    stage: str,
    label: str,
    kernel: str,
    lr: str,
    inducing: int,
    returncode: int,
) -> dict:
    return {
        "stage": stage,
        "label": label,
        "kernel": kernel,
        "lr": float(lr),
        "num_inducing": int(inducing),
        "epochs": int(EPOCHS),
        "seed_count": 0,
        "mean_best_val_rmse_scaled": "",
        "std_best_val_rmse_scaled": "",
        "mean_best_epoch": "",
        "mean_epochs_ran": "",
        "weight_decay": "",
        "batch_size": "",
        "patience": "",
        "init_noise": "",
        "variational_jitter": "",
        "matern_nu": "",
        "period_length": "",
        "status": "failed",
        "error": f"subprocess exit code {returncode}",
    }


def summarize(stage: str, label: str, payload: dict) -> dict:
    validation = []
    best_epochs = []
    epochs_ran = []
    for seed_run in payload.get("seed_runs", []):
        training = seed_run.get("payload", {}).get("training_summary", {})
        validation.append(float(training["best_val_rmse_scaled"]))
        best_epochs.append(int(training["best_epoch"]))
        epochs_ran.append(int(training["epochs_ran"]))
    if not validation:
        raise RuntimeError(f"No validation summaries found for {label}")
    params = payload["params_used"]
    mean = sum(validation) / len(validation)
    variance = sum((value - mean) ** 2 for value in validation) / len(validation)
    return {
        "stage": stage,
        "label": label,
        "kernel": params["kernel"],
        "lr": float(params["lr"]),
        "num_inducing": int(params["num_inducing"]),
        "epochs": int(params["epochs"]),
        "seed_count": len(validation),
        "mean_best_val_rmse_scaled": mean,
        "std_best_val_rmse_scaled": variance**0.5,
        "mean_best_epoch": sum(best_epochs) / len(best_epochs),
        "mean_epochs_ran": sum(epochs_ran) / len(epochs_ran),
        "weight_decay": float(params["weight_decay"]),
        "batch_size": int(params["batch_size"]),
        "patience": int(params["patience"]),
        "init_noise": float(params["init_noise"]),
        "variational_jitter": float(params["variational_jitter"]),
        "matern_nu": float(params["matern_nu"]),
        "period_length": float(params["period_length"]),
        "status": "success",
        "error": "",
    }


def write_summary(rows: list[dict], best: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "svgp_tuning_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    best_payload = {
        "selection_metric": "mean_best_val_rmse_scaled",
        "selection_uses_test_metrics": False,
        "tuning_seeds": SEEDS,
        "formal_run_params": {
            key: best[key]
            for key in (
                "kernel",
                "lr",
                "num_inducing",
                "epochs",
                "weight_decay",
                "batch_size",
                "patience",
                "init_noise",
                "variational_jitter",
                "matern_nu",
                "period_length",
            )
        },
        "best": best,
        "all_results": rows,
    }
    with open(OUTPUT_DIR / "svgp_tuning_best_params.json", "w", encoding="utf-8") as handle:
        json.dump(best_payload, handle, ensure_ascii=False, indent=2)

    with open(OUTPUT_DIR / "SVGP_TUNING_README.md", "w", encoding="utf-8") as handle:
        handle.write("# SVGP hyperparameter tuning\n\n")
        handle.write(
            "Selection uses only mean best validation RMSE on seeds "
            f"{SEEDS}; test metrics are not used for model selection.\n\n"
        )
        handle.write("## Best configuration\n\n")
        handle.write(
            f"- Kernel: `{best['kernel']}`\n"
            f"- Learning rate: `{best['lr']}`\n"
            f"- Inducing points: `{best['num_inducing']}`\n"
            f"- Validation RMSE (scaled): "
            f"`{best['mean_best_val_rmse_scaled']:.6f} ± "
            f"{best['std_best_val_rmse_scaled']:.6f}`\n"
        )


def main() -> None:
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for kernel in STAGE1_KERNELS:
        for lr in STAGE1_LRS:
            rows.append(run_config("stage1", kernel, lr, 1024))

    stage1_success = [
        row for row in rows
        if row["stage"] == "stage1" and row["status"] == "success"
    ]
    if not stage1_success:
        raise RuntimeError("All stage-1 SVGP tuning configurations failed")
    stage1_best = min(
        stage1_success,
        key=lambda row: row["mean_best_val_rmse_scaled"],
    )
    print(f"[stage1 best] {stage1_best}", flush=True)

    for inducing in STAGE2_INDUCING:
        if inducing == int(stage1_best["num_inducing"]):
            reused = dict(stage1_best)
            reused["stage"] = "stage2"
            reused["label"] = f"stage2_reuse_{stage1_best['label']}"
            rows.append(reused)
        else:
            rows.append(
                run_config(
                    "stage2",
                    str(stage1_best["kernel"]),
                    str(stage1_best["lr"]),
                    inducing,
                )
            )

    stage2_success = [
        row for row in rows
        if row["stage"] == "stage2" and row["status"] == "success"
    ]
    if not stage2_success:
        raise RuntimeError("All stage-2 SVGP tuning configurations failed")
    stage2_best = min(
        stage2_success,
        key=lambda row: row["mean_best_val_rmse_scaled"],
    )
    write_summary(rows, stage2_best)
    print(f"[final best] {stage2_best}", flush=True)
    print(f"Saved tuning summary to: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
