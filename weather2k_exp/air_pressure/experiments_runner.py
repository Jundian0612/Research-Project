"""
簡易實驗 runner：執行三個指定的外推情境，呼叫 DLinear+FRK 與 STDK 腳本。
時間固定使用最後 1000 個時間點並切成 700/150/150。
空間固定抽 600 個空間點；obs100 + unobs400 共 500 站提供訓練監督，
其餘 100 站保留作空間/時空測試。
會產生對應的結果檔案，檔名會包含場景後綴。
"""
import os
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DLIN_SCRIPT = ROOT / "2K_DLinear_FRK_hybridloss.py"
STDK_SCRIPT = ROOT / "2K_STDK_500train_100test.py"
PY = sys.executable
ALPHA_LIST = [value.strip() for value in os.environ.get("ALPHA_LIST", "1.0").split(",") if value.strip()]
LAMBDA_LIST = [value.strip() for value in os.environ.get("LAMBDA_LIST", "1.0").split(",") if value.strip()]
SPACE_SPLITS = [
    tuple(int(part) for part in value.strip().replace(":", "/").split("/"))
    for value in os.environ.get("SPACE_SPLITS", "100/400/100").split(",")
    if value.strip()
]
RUN_STDK = os.environ.get("RUN_STDK", "1") == "1"
RUN_DLINEAR = os.environ.get("RUN_DLINEAR", "1") == "1"

TARGET_PREFIX_BY_SCENARIO = {
    "time_extrap_fixed500": "Target_Time150",
    "space_extrap_fixed850": "Target_Space100",
    "spatiotemp_100x150": "Target_ST100x150",
}

SCENARIOS = [
    {
        "name": "time_extrap_fixed500",
        # 固定 500 個受監督空間點（obs100 + unobs400）做時間外推。
        "env": {
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
            "RESULT_SUFFIX": "time500",
        },
    },
    {
        "name": "space_extrap_fixed850",
        # 固定前 850 個訓練/驗證時間點，評估保留的 100 個空間點。
        "env": {
            "EXPERIMENT_SCENARIO": "space_extrap_fixed850",
            "N_SAMPLE_TARGET": "600",
            "N_STDK": "100",
            "N_TRAIN_TARGET": "100",
            "N_UNKNOWN_PRIMARY_TARGET": "400",
            "N_UNKNOWN_EVAL_TARGET": "100",
            "N_LAST": "1000",
            "TIME_TRAIN_LEN": "700",
            "TIME_VAL_LEN": "150",
            "TIME_TEST_LEN": "150",
            "RESULT_SUFFIX": "space850",
        },
    },
    {
        "name": "spatiotemp_100x150",
        # 同時外推 100 個空間評估點與最後 150 個時間點。
        "env": {
            "EXPERIMENT_SCENARIO": "spatiotemp_100x150",
            "N_SAMPLE_TARGET": "600",
            "N_STDK": "100",
            "N_TRAIN_TARGET": "100",
            "N_UNKNOWN_PRIMARY_TARGET": "400",
            "N_UNKNOWN_EVAL_TARGET": "100",
            "N_LAST": "1000",
            "TIME_TRAIN_LEN": "700",
            "TIME_VAL_LEN": "150",
            "TIME_TEST_LEN": "150",
            "RESULT_SUFFIX": "st_100x150",
        },
    },
]

def run_script(script_path: Path, env: dict) -> int:
    call_env = os.environ.copy()
    call_env.update(env)
    print(f"Running {script_path.name} with env: {env}")
    proc = subprocess.run([PY, str(script_path)], env=call_env)
    return proc.returncode


def lambda_suffix(lambda_value: str) -> str:
    return f"lam{lambda_value.replace('.', 'p')}"


def alpha_suffix(alpha_value: str) -> str:
    return f"alpha{alpha_value.replace('.', 'p')}"


def split_suffix(space_split: tuple[int, int, int]) -> str:
    train_n, unobs_n, target_n = space_split
    return f"s{train_n}u{unobs_n}e{target_n}"


def apply_space_split(env: dict, space_split: tuple[int, int, int]) -> dict:
    train_n, unobs_n, target_n = space_split
    updated = dict(env)
    updated["N_SAMPLE_TARGET"] = str(train_n + unobs_n + target_n)
    updated["N_STDK"] = str(train_n)
    updated["N_TRAIN_TARGET"] = str(train_n)
    updated["N_UNKNOWN_PRIMARY_TARGET"] = str(unobs_n)
    updated["N_UNKNOWN_EVAL_TARGET"] = str(target_n)
    updated["SPACE_SPLIT_LABEL"] = f"{train_n}/{unobs_n}/{target_n}"
    return updated


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _metric_value(metrics: dict, prefix: str, metric_name: str):
    return metrics.get(f"{prefix}_{metric_name}", metrics.get(metric_name))


def _fmt_metric(mean_metrics: dict, std_metrics: dict, prefix: str, metric_name: str) -> str:
    mean_value = _metric_value(mean_metrics, prefix, metric_name)
    std_value = _metric_value(std_metrics, prefix, metric_name)
    if mean_value is None:
        return ""
    if std_value is None:
        return f"{float(mean_value):.6f}"
    return f"{float(mean_value):.6f} +/- {float(std_value):.6f}"


def _split_columns(space_split: tuple[int, int, int]) -> dict:
    train_n, unobs_n, target_n = space_split
    return {
        "space_split": f"{train_n}/{unobs_n}/{target_n}",
        "n_train_space": train_n,
        "n_unobs_space": unobs_n,
        "n_target_space": target_n,
    }


def _summary_row(
    payload: dict,
    scenario_name: str,
    space_split: tuple[int, int, int],
    alpha_value: str = "",
    lambda_value: str = "",
) -> dict:
    target_prefix = TARGET_PREFIX_BY_SCENARIO[scenario_name]
    mean_metrics = payload.get("mean_metrics", {})
    std_metrics = payload.get("std_metrics", {})
    return {
        "scenario": scenario_name,
        "target": target_prefix,
        **_split_columns(space_split),
        "model": payload.get("Model", ""),
        "alpha_obs": alpha_value,
        "lambda_unobs": lambda_value,
        "row_type": "mean_std",
        "seed": "mean +/- std",
        "RMSE": _fmt_metric(mean_metrics, std_metrics, target_prefix, "RMSE"),
        "MSE": _fmt_metric(mean_metrics, std_metrics, target_prefix, "MSE"),
        "MAE": _fmt_metric(mean_metrics, std_metrics, target_prefix, "MAE"),
        "R2": _fmt_metric(mean_metrics, std_metrics, target_prefix, "R2"),
    }


def _fmt_seed_metric(metrics: dict, prefix: str, metric_name: str) -> str:
    value = _metric_value(metrics, prefix, metric_name)
    if value is None:
        return ""
    return f"{float(value):.6f}"


def _seed_rows(
    payload: dict,
    scenario_name: str,
    space_split: tuple[int, int, int],
    alpha_value: str = "",
    lambda_value: str = "",
) -> list[dict]:
    target_prefix = TARGET_PREFIX_BY_SCENARIO[scenario_name]
    rows = []
    for run in payload.get("seed_runs", []):
        metrics = run.get("frk_metrics", run.get("metrics", {}))
        rows.append(
            {
                "scenario": scenario_name,
                "target": target_prefix,
                **_split_columns(space_split),
                "model": payload.get("Model", ""),
                "alpha_obs": alpha_value,
                "lambda_unobs": lambda_value,
                "row_type": "seed",
                "seed": run.get("seed", ""),
                "RMSE": _fmt_seed_metric(metrics, target_prefix, "RMSE"),
                "MSE": _fmt_seed_metric(metrics, target_prefix, "MSE"),
                "MAE": _fmt_seed_metric(metrics, target_prefix, "MAE"),
                "R2": _fmt_seed_metric(metrics, target_prefix, "R2"),
            }
        )
    return rows


def _print_summary_table(rows: list[dict], scenario_name: str) -> None:
    if not rows:
        return
    headers = ["space_split", "model", "alpha_obs", "lambda_unobs", "target", "RMSE", "MSE", "MAE", "R2"]
    widths = {header: len(header) for header in headers}
    for row in rows:
        for header in headers:
            widths[header] = max(widths[header], len(str(row.get(header, ""))))
    print(f"\n===== Scenario summary: {scenario_name} =====")
    print(" | ".join(header.ljust(widths[header]) for header in headers))
    print("-+-".join("-" * widths[header] for header in headers))
    for row in rows:
        print(" | ".join(str(row.get(header, "")).ljust(widths[header]) for header in headers))


def _write_result_files(rows: list[dict], scenario_name: str) -> None:
    if not rows:
        return
    csv_path = ROOT / f"results_{scenario_name}.csv"
    md_path = ROOT / f"results_{scenario_name}.md"
    headers = [
        "scenario",
        "target",
        "space_split",
        "n_train_space",
        "n_unobs_space",
        "n_target_space",
        "model",
        "alpha_obs",
        "lambda_unobs",
        "row_type",
        "seed",
        "RMSE",
        "MSE",
        "MAE",
        "R2",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Scenario summary: {scenario_name}\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |\n")
    print(f"Saved result table: {csv_path.resolve()}")
    print(f"Saved result table: {md_path.resolve()}")


if __name__ == "__main__":
    for sc in SCENARIOS:
        print("\n=== Scenario:", sc["name"], "===")
        result_rows = []
        summary_rows = []
        rc = 0

        for space_split in SPACE_SPLITS:
            sc_env = apply_space_split(sc["env"], space_split)
            train_n, unobs_n, target_n = space_split
            base_suffix = (
                f"{sc['env']['RESULT_SUFFIX']}_{split_suffix(space_split)}_"
                f"train{train_n + unobs_n}_test{target_n}"
            )
            print(f"\n=== Space split {sc_env['SPACE_SPLIT_LABEL']} ===")

            if RUN_DLINEAR:
                for alpha_value in ALPHA_LIST:
                    for lambda_value in LAMBDA_LIST:
                        dlinear_env = dict(sc_env)
                        dlinear_env["DIFF_FRK_OBS_LOSS_WEIGHT"] = str(alpha_value)
                        dlinear_env["DIFF_FRK_LOSS_WEIGHT"] = str(lambda_value)
                        dlinear_env["RESULT_SUFFIX"] = (
                            f"{base_suffix}_"
                            f"{alpha_suffix(str(alpha_value))}_{lambda_suffix(str(lambda_value))}"
                        )
                        print(
                            f"\n--- DLinear+FRK split={sc_env['SPACE_SPLIT_LABEL']}, "
                            f"alpha={alpha_value}, lambda={lambda_value} ---"
                        )
                        rc = run_script(DLIN_SCRIPT, dlinear_env)
                        if rc != 0:
                            print(
                                f"Script {DLIN_SCRIPT.name} exited with code {rc}. "
                                f"Aborting scenario {sc['name']} at split={sc_env['SPACE_SPLIT_LABEL']}, "
                                f"alpha={alpha_value}, lambda={lambda_value}"
                            )
                            break
                        result_path = ROOT / f"dlinear_autofrk_frkloss_test_100to500_metrics_{dlinear_env['RESULT_SUFFIX']}.json"
                        if result_path.exists():
                            payload = _load_json(result_path)
                            result_rows.extend(_seed_rows(payload, sc["name"], space_split, alpha_value, lambda_value))
                            summary_row = _summary_row(payload, sc["name"], space_split, alpha_value, lambda_value)
                            result_rows.append(summary_row)
                            summary_rows.append(summary_row)
                            _write_result_files(result_rows, sc["name"])
                    if rc != 0:
                        break
                else:
                    rc = 0
                if rc != 0:
                    break

            if RUN_STDK:
                stdk_env = dict(sc_env)
                stdk_env["RESULT_SUFFIX"] = base_suffix
                print(f"\n--- STDK baseline split={sc_env['SPACE_SPLIT_LABEL']} (lambda-independent) ---")
                rc = run_script(STDK_SCRIPT, stdk_env)
                if rc != 0:
                    print(
                        f"Script {STDK_SCRIPT.name} exited with code {rc}. "
                        f"Aborting scenario {sc['name']} at split={sc_env['SPACE_SPLIT_LABEL']}"
                    )
                    break
                result_path = ROOT / f"2K_stdk_metrics_{stdk_env['RESULT_SUFFIX']}.json"
                if result_path.exists():
                    payload = _load_json(result_path)
                    result_rows.extend(_seed_rows(payload, sc["name"], space_split))
                    summary_row = _summary_row(payload, sc["name"], space_split)
                    result_rows.append(summary_row)
                    summary_rows.append(summary_row)
                    _write_result_files(result_rows, sc["name"])
        else:
            rc = 0

        if rc != 0:
            continue

        _print_summary_table(summary_rows, sc["name"])
        print(f"Scenario {sc['name']} completed.\n")

    print("All scenarios invoked. Check the generated JSON files in the weather2k_exp folder.")
