#!/usr/bin/env python3
"""Tune DLinear and FRK jointly using validation data only.

Each Optuna trial launches ``2K_DLinear_FRK_hybridloss.py`` with a temporary
combined parameter JSON. Model selection uses the mean best Val150 RMSE on
the 500 supervised stations (100 observed + 400 supervised-unobserved).
Metrics for the held-out target100 stations are never read for selection.
"""

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import optuna


ROOT = Path(__file__).resolve().parent
RESULT_ROOT = Path(
    os.environ.get("WEATHER2K_OUTPUT_DIR", str(ROOT))
).expanduser().resolve()
MODEL_SCRIPT = ROOT / "2K_DLinear_FRK_hybridloss.py"
OUTPUT_DIR = ROOT / os.environ.get("DLINEAR_FRK_TUNING_DIR", "")
TRIAL_DIR = OUTPUT_DIR
BEST_PARAMS_PATH = ROOT / "2K_best_dlinear_and_frk_params_500to100.json"
STUDY_PATH = OUTPUT_DIR / "dlinear_frk_optuna_study.sqlite3"

N_TRIALS = int(os.environ.get("DLINEAR_FRK_TUNE_TRIALS", "30"))
TUNE_SEEDS = os.environ.get("DLINEAR_FRK_TUNE_SEEDS", "[41, 42]")
TIMEOUT_SECONDS = os.environ.get("DLINEAR_FRK_TUNE_TIMEOUT_SECONDS", "").strip()
TIMEOUT_SECONDS = int(TIMEOUT_SECONDS) if TIMEOUT_SECONDS else None
OPTUNA_SEED = int(os.environ.get("DLINEAR_FRK_OPTUNA_SEED", "42"))


SEARCH_SPACE = {
    "INPUT_CHUNK_LENGTH": [24, 36, 48],
    "OUTPUT_CHUNK_LENGTH": [12, 24],
    "KERNEL_SIZE": [15, 25],
    "N_EPOCHS": [350],
    "BATCH_SIZE": [16, 32, 64],
    "LR": [1e-4, 2e-4, 3e-4],
    "WEIGHT_DECAY": [0.0, 1e-5],
    "CONST_INIT": [True, False],
    # FRK-related parameters that affect the current formal pipeline.
    "FRK_TRAIN_N_NEIGHBOR": [3, 5, 8, 10],
    "SPATIAL_SURROGATE_BANDWIDTH": ["auto", "0.1", "0.2", "0.3", "0.5"],
    "DIFF_FRK_OBS_LOSS_WEIGHT": [0.5, 1.0, 1.5, 2.0],
    "DIFF_FRK_LOSS_WEIGHT": [0.5, 1.0, 1.5, 2.0],
}

FIXED_PARAMS = {
    "FRK_LOSS_WEIGHT": 0.01,
    "FRK_OBS_RECON_WEIGHT": 1.0,
    "FRK_UNOBS_MEAN_WEIGHT": 0.0,
    "FRK_UNOBS_STD_WEIGHT": 0.0,
    "FRK_LOSS_APPLY_EVERY": 1,
    # Used only by the legacy AutoFRK helper, not by the current validation objective.
    "FRK_TEST_N_NEIGHBOR": 5,
}


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def suggest_params(trial: optuna.Trial) -> dict:
    params = {
        name: trial.suggest_categorical(name, choices)
        for name, choices in SEARCH_SPACE.items()
    }
    params.update(FIXED_PARAMS)
    return params


def result_paths(suffix: str) -> tuple[Path, Path]:
    return (
        RESULT_ROOT / f"dlinear_autofrk_frkloss_test_100to500_metrics_{suffix}.json",
        RESULT_ROOT / f"2K_best_dlinear_frkloss_rerun_metrics_500to100_{suffix}.json",
    )


def validation_values(payload: dict) -> list[float]:
    values = []
    for seed_run in payload.get("seed_runs", []):
        summary = seed_run.get("loss_summary", {})
        value = summary.get("best_val_4plus5_rmse_raw")
        if value is not None:
            values.append(float(value))
    return values


def objective(trial: optuna.Trial) -> float:
    params = suggest_params(trial)
    label = f"dlinear_frk_tune_trial{trial.number:04d}"
    trial_params_path = TRIAL_DIR / f"{label}_params.json"
    trial_result_path = TRIAL_DIR / f"{label}_metrics.json"
    dump_json(params, trial_params_path)

    env = os.environ.copy()
    env.update(
        {
            "EXPERIMENT_SCENARIO": "time_extrap_fixed500",
            "N_SAMPLE_TARGET": "600",
            "N_STDK": "100",
            "N_TRAIN_TARGET": "100",
            "N_UNKNOWN_PRIMARY_TARGET": "400",
            "N_UNKNOWN_EVAL_TARGET": "100",
            "N_LAST": "1000",
            "TIME_TRAIN_LEN": "700",
            "TIME_VAL_LEN": "150",
            "TIME_TEST_LEN": "150",
            "SEED_LIST": TUNE_SEEDS,
            "RESULT_SUFFIX": label,
            "DLINEAR_FRK_PARAMS_PATH": str(trial_params_path),
            "WEATHER2K_OUTPUT_DIR": str(RESULT_ROOT),
            "PRINT_EPOCH_LOSS": os.environ.get("PRINT_EPOCH_LOSS", "0"),
        }
    )

    print(f"\n[trial {trial.number + 1}/{N_TRIALS}] {params}", flush=True)
    completed = subprocess.run([sys.executable, str(MODEL_SCRIPT)], env=env)
    generated_frk, generated_dlinear = result_paths(label)
    if completed.returncode != 0:
        raise RuntimeError(f"Trial subprocess failed with exit code {completed.returncode}")
    if not generated_frk.exists():
        raise FileNotFoundError(f"Expected result was not created: {generated_frk}")

    payload = load_json(generated_frk)
    values = validation_values(payload)
    if not values:
        raise RuntimeError("No validation RMSE values found in trial result")
    mean_rmse = sum(values) / len(values)
    variance = sum((value - mean_rmse) ** 2 for value in values) / len(values)
    trial.set_user_attr("validation_rmse_by_seed", values)
    trial.set_user_attr("validation_rmse_std", variance**0.5)
    trial.set_user_attr("selection_uses_test_metrics", False)

    generated_frk.replace(trial_result_path)
    if generated_dlinear.exists():
        generated_dlinear.unlink()
    return float(mean_rmse)


def write_outputs(study: optuna.Study) -> None:
    completed = [
        trial for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed:
        raise RuntimeError("No successful DLinear+FRK tuning trials")

    rows = []
    for trial in study.trials:
        row = {
            "trial": trial.number,
            "state": trial.state.name,
            "mean_val_4plus5_rmse_raw": trial.value if trial.value is not None else "",
            "std_val_4plus5_rmse_raw": trial.user_attrs.get("validation_rmse_std", ""),
        }
        row.update(trial.params)
        rows.append(row)

    summary_path = OUTPUT_DIR / "dlinear_frk_tuning_summary.csv"
    fieldnames = list(rows[0])
    with open(summary_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    best = study.best_trial
    formal_run_params = dict(best.params)
    formal_run_params.update(FIXED_PARAMS)
    best_payload = {
        "selection_metric": "mean_best_val_4plus5_rmse_raw",
        "selection_uses_test_metrics": False,
        "tuning_seeds": json.loads(TUNE_SEEDS),
        "space_split": {
            "observed": 100,
            "supervised_unobserved": 400,
            "held_out_target": 100,
        },
        "time_split": {"train": 700, "validation": 150, "test": 150},
        "best_trial_number": best.number,
        "best_validation_rmse": float(best.value),
        "best_validation_rmse_by_seed": best.user_attrs.get("validation_rmse_by_seed", []),
        "formal_run_params": formal_run_params,
        "search_space": SEARCH_SPACE,
        "fixed_params": FIXED_PARAMS,
        "n_requested_trials": N_TRIALS,
    }
    dump_json(best_payload, OUTPUT_DIR / "dlinear_frk_tuning_best_params.json")
    dump_json(best_payload, BEST_PARAMS_PATH)

    readme = f"""# DLinear+FRK hyperparameter tuning

Optuna jointly searches DLinear model parameters and the FRK-related parameters
used by the current differentiable spatial pipeline. Selection uses only the
mean best Val150 RMSE over supervised stations 4+5 and seeds {TUNE_SEEDS}.
Held-out target100 test metrics are not read for model selection.

## Best result

- Trial: `{best.number}`
- Mean validation RMSE: `{best.value:.6f}`
- Formal parameter file: `{BEST_PARAMS_PATH.name}`

## Search protocol

- Space split: 100 observed / 400 supervised-unobserved / 100 held-out target
- Time split: Train700 / Val150 / Test150
- Objective: mean best validation 4+5 RMSE on the original temperature scale
- Optuna trials: {N_TRIALS}
"""
    (OUTPUT_DIR / "DLINEAR_FRK_TUNING_README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    TRIAL_DIR.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{STUDY_PATH}"
    study = optuna.create_study(
        study_name="weather2k_dlinear_frk_train500_test100",
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=OPTUNA_SEED),
    )
    remaining = max(0, N_TRIALS - len(study.trials))
    if remaining:
        study.optimize(
            objective,
            n_trials=remaining,
            timeout=TIMEOUT_SECONDS,
            gc_after_trial=True,
            catch=(RuntimeError, FileNotFoundError),
        )
    write_outputs(study)
    print(f"\nSaved tuning outputs to: {OUTPUT_DIR.resolve()}")
    print(f"Saved formal parameters to: {BEST_PARAMS_PATH.resolve()}")


if __name__ == "__main__":
    main()
