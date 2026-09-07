"""
簡易實驗 runner：執行三個指定的外推情境，呼叫 SVGP、DLinear+FRK 與 STDK 腳本。
可選擇執行一次 STDK+SharedQConvLSTM，並將同一個 fitted model 的時間、
空間與時空外推結果分別加入三張比較表。
時間固定使用最後 1000 個時間點並切成 700/150/150。
空間固定抽 600 個空間點；obs100 + unobs400 共 500 站提供訓練監督，
其餘 100 站保留作空間/時空測試。
會產生對應的結果檔案，檔名會包含場景後綴。
"""
import os
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULT_ROOT = Path(
    os.environ.get("WEATHER2K_OUTPUT_DIR", str(ROOT))
).expanduser().resolve()
SVGP_SCRIPT = ROOT / "2K_SVGP_500train_100test.py"
DLIN_SCRIPT = ROOT / "2K_DLinear_FRK_hybridloss.py"
STDK_SCRIPT = ROOT / "2K_STDK_500train_100test.py"
QCONV_SCRIPT = ROOT / "2K_STDK_QConvLSTM.py"
PY = sys.executable
PARAMS_PATH = ROOT / "2K_best_dlinear_and_frk_params_500to100.json"
if not PARAMS_PATH.exists():
    raise FileNotFoundError(
        f"Missing tuned DLinear+FRK parameters: {PARAMS_PATH}. "
        "Run tune_dlinear_and_frk_hyperparams.py first."
    )
with open(PARAMS_PATH, "r", encoding="utf-8") as _params_handle:
    _params_payload = json.load(_params_handle)
FORMAL_PARAMS = _params_payload.get("formal_run_params", _params_payload)
FORMAL_ALPHA = str(FORMAL_PARAMS["DIFF_FRK_OBS_LOSS_WEIGHT"])
FORMAL_LAMBDA = str(FORMAL_PARAMS["DIFF_FRK_LOSS_WEIGHT"])

_alpha_override = os.environ.get("ALPHA_LIST")
_lambda_override = os.environ.get("LAMBDA_LIST")
if (_alpha_override is None) != (_lambda_override is None):
    raise ValueError("Set both ALPHA_LIST and LAMBDA_LIST, or leave both unset to use tuned values.")
ALPHA_LIST = (
    [value.strip() for value in _alpha_override.split(",") if value.strip()]
    if _alpha_override is not None else [None]
)
LAMBDA_LIST = (
    [value.strip() for value in _lambda_override.split(",") if value.strip()]
    if _lambda_override is not None else [None]
)
SPACE_SPLITS = [
    tuple(int(part) for part in value.strip().replace(":", "/").split("/"))
    for value in os.environ.get(
        "SPACE_SPLITS",
        "100/400/100",
    ).split(",")
    if value.strip()
]
RUN_STDK = os.environ.get("RUN_STDK", "1") == "1"
RUN_DLINEAR = os.environ.get("RUN_DLINEAR", "1") == "1"
RUN_SVGP = os.environ.get("RUN_SVGP", "1") == "1"
# One formal seed currently takes about 6.25 hours on the local GTX 1050 Ti.
# Keep this expensive model opt-in even though it is integrated into the runner.
RUN_QCONVLSTM = os.environ.get("RUN_QCONVLSTM", "0") == "1"
QCONVLSTM_SEEDS = [
    int(value.strip())
    for value in os.environ.get("QCONVLSTM_SEEDS", "41,42,43,44,45").split(",")
    if value.strip()
]
QCONVLSTM_FORECAST_MODE = os.environ.get(
    "QCONVLSTM_FORECAST_MODE", "block5to5"
)
if QCONVLSTM_FORECAST_MODE not in {"direct5to150", "block5to5"}:
    raise ValueError(
        "QCONVLSTM_FORECAST_MODE must be direct5to150 or block5to5"
    )
QCONVLSTM_OUTPUT_DIR = Path(
    os.environ.get(
        "QCONVLSTM_OUTPUT_DIR",
        str(RESULT_ROOT),
    )
).expanduser().resolve()
ARCHIVE_DIR = (
    Path(os.environ["RESULT_ARCHIVE_DIR"]).expanduser().resolve()
    if os.environ.get("RESULT_ARCHIVE_DIR") else None
)

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

_scenario_filter = {
    value.strip() for value in os.environ.get("SCENARIO_FILTER", "").split(",")
    if value.strip()
}
if _scenario_filter:
    known = {scenario["name"] for scenario in SCENARIOS}
    unknown = _scenario_filter - known
    if unknown:
        raise ValueError(f"Unknown SCENARIO_FILTER values: {sorted(unknown)}")
    SCENARIOS = [scenario for scenario in SCENARIOS if scenario["name"] in _scenario_filter]

SVGP_RESULT_TARGET_PREFIX = {
    "time_extrap_fixed500": "Target_Time150",
    "space_extrap_fixed850": "Target_Space100",
    "spatiotemp_100x150": "Target_ST100x150",
}

def run_script(script_path: Path, env: dict, args: list[str] | None = None) -> int:
    call_env = os.environ.copy()
    call_env.update(env)
    call_env.setdefault("WEATHER2K_OUTPUT_DIR", str(RESULT_ROOT))
    print(f"Running {script_path.name} with env: {env}")
    proc = subprocess.run([PY, str(script_path), *(args or [])], env=call_env)
    return proc.returncode


def lambda_suffix(lambda_value: str) -> str:
    return f"lam{lambda_value.replace('.', 'p')}"


def alpha_suffix(alpha_value: str) -> str:
    return f"alpha{alpha_value.replace('.', 'p')}"


def svgp_suffix() -> str:
    kernel = os.environ.get("SVGP_KERNEL", "matern_periodic").strip().lower()
    inducing = os.environ.get("SVGP_NUM_INDUCING", "1024").strip()
    lr = os.environ.get("SVGP_LR", "0.001").strip().replace(".", "p")
    epochs = os.environ.get("SVGP_EPOCHS", "500").strip()
    return f"{kernel}_m{inducing}_lr{lr}_e{epochs}"


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
    destinations = [RESULT_ROOT]
    if ARCHIVE_DIR is not None:
        destinations.append(ARCHIVE_DIR / "tables")
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
    for destination in destinations:
        destination.mkdir(parents=True, exist_ok=True)
        csv_path = destination / f"results_{scenario_name}.csv"
        md_path = destination / f"results_{scenario_name}.md"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# Scenario summary: {scenario_name}\n\n")
            models = {str(row.get("model", "")) for row in rows}
            if "STDK+SharedLSTM" in models:
                f.write(
                    "> **重要：** `STDK+SharedLSTM` 是自訂 Weather2K deterministic "
                    "shared-LSTM baseline，不符合論文 QLSTM／QConvLSTM 的 quantile "
                    "probabilistic forecasting 流程，不能標示為 paper reproduction。"
                    "SVGP 與 DLinear+FRK 列不受此註記影響。\n\n"
                )
            f.write("| " + " | ".join(headers) + " |\n")
            f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
            for row in rows:
                f.write("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |\n")
        print(f"Saved result table: {csv_path.resolve()}")
        print(f"Saved result table: {md_path.resolve()}")


def _archive_json(path: Path) -> None:
    if ARCHIVE_DIR is None or not path.exists():
        return
    destination = ARCHIVE_DIR / "json" / path.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)
    print(f"Archived JSON: {destination}")


def _archive_qconvlstm_forecasts(output_dir: Path) -> None:
    if ARCHIVE_DIR is None:
        return
    destination_dir = ARCHIVE_DIR / "forecasts"
    destination_dir.mkdir(parents=True, exist_ok=True)
    suffixes = ("_time_forecasts.csv", "_space_forecasts.csv", "_forecasts.csv")
    for seed in QCONVLSTM_SEEDS:
        stem = f"shared_qconvlstm_{QCONVLSTM_FORECAST_MODE}_seed{seed}"
        for suffix in suffixes:
            source = output_dir / f"{stem}{suffix}"
            if source.exists():
                destination = destination_dir / source.name
                shutil.copy2(source, destination)
                print(f"Archived QConvLSTM forecast: {destination}")


def _update_archive_readme(summary_rows: list[dict], scenario_name: str) -> None:
    if ARCHIVE_DIR is None or scenario_name != "space_extrap_fixed850":
        return
    by_model = {row.get("model"): row for row in summary_rows}
    required = ("SVGP", "STDK", "DLINEAR(best)")
    if not all(model in by_model for model in required):
        print("Archive README not updated: the three space summaries are incomplete.")
        return
    readme = ARCHIVE_DIR / "README.md"
    if not readme.exists():
        return
    text = readme.read_text(encoding="utf-8")
    old_status = "| 空間外推 | Target_Space100 | 5 seeds | 無 | 無 | 未完成，不可做三模型比較 |"
    new_status = "| 空間外推 | Target_Space100 | 5 seeds | 5 seeds | 5 seeds | 完成 |"
    text = text.replace(old_status, new_status)
    start = text.find("### 空間外推：Target_Space100")
    end = text.find("\n## STDK + Shared LSTM 流程", start)
    if start != -1 and end != -1:
        lines = [
            "### 空間外推：Target_Space100",
            "",
            "| 模型 | RMSE | MAE | R2 |",
            "|---|---:|---:|---:|",
        ]
        for model in required:
            row = by_model[model]
            lines.append(f"| {model} | {row['RMSE']} | {row['MAE']} | {row['R2']} |")
        lines.extend(["", "三個模型的空間外推均已完成；詳細逐 seed 結果見 `tables/results_space_extrap_fixed850.md`。", ""])
        text = text[:start] + "\n".join(lines) + text[end:]
    readme.write_text(text, encoding="utf-8")
    print(f"Updated archive README: {readme}")


def _svgp_result_path(result_suffix: str) -> Path:
    return RESULT_ROOT / f"2K_svgp_metrics_{result_suffix}.json"


def _svgp_summary_row(payload: dict, scenario_name: str, space_split: tuple[int, int, int]) -> dict:
    target_prefix = SVGP_RESULT_TARGET_PREFIX[scenario_name]
    mean_metrics = payload.get("mean_metrics", {})
    std_metrics = payload.get("std_metrics", {})
    return {
        "scenario": scenario_name,
        "target": target_prefix,
        **_split_columns(space_split),
        "model": payload.get("Model", ""),
        "alpha_obs": "",
        "lambda_unobs": "",
        "row_type": "mean_std",
        "seed": "mean +/- std",
        "RMSE": _fmt_metric(mean_metrics, std_metrics, target_prefix, "RMSE"),
        "MSE": _fmt_metric(mean_metrics, std_metrics, target_prefix, "MSE"),
        "MAE": _fmt_metric(mean_metrics, std_metrics, target_prefix, "MAE"),
        "R2": _fmt_metric(mean_metrics, std_metrics, target_prefix, "R2"),
    }


def _svgp_seed_rows(payload: dict, scenario_name: str, space_split: tuple[int, int, int]) -> list[dict]:
    target_prefix = SVGP_RESULT_TARGET_PREFIX[scenario_name]
    rows = []
    for run in payload.get("seed_runs", []):
        metrics = run.get("metrics", {})
        rows.append(
            {
                "scenario": scenario_name,
                "target": target_prefix,
                **_split_columns(space_split),
                "model": payload.get("Model", ""),
                "alpha_obs": "",
                "lambda_unobs": "",
                "row_type": "seed",
                "seed": run.get("seed", ""),
                "RMSE": _fmt_seed_metric(metrics, target_prefix, "RMSE"),
                "MSE": _fmt_seed_metric(metrics, target_prefix, "MSE"),
                "MAE": _fmt_seed_metric(metrics, target_prefix, "MAE"),
                "R2": _fmt_seed_metric(metrics, target_prefix, "R2"),
            }
        )
    return rows


def _qconvlstm_rows(
    output_dir: Path,
    scenario_name: str,
    space_split: tuple[int, int, int],
) -> tuple[list[dict], dict]:
    """Load per-seed SharedQConvLSTM reports and build common runner rows."""
    reports = []
    for seed in QCONVLSTM_SEEDS:
        report_path = output_dir / (
            f"shared_qconvlstm_{QCONVLSTM_FORECAST_MODE}_seed{seed}.json"
        )
        if not report_path.exists():
            raise FileNotFoundError(f"Missing SharedQConvLSTM report: {report_path}")
        _archive_json(report_path)
        reports.append(_load_json(report_path))

    target_prefix = TARGET_PREFIX_BY_SCENARIO[scenario_name]
    model_name = f"STDK+SharedQConvLSTM({QCONVLSTM_FORECAST_MODE})"
    rows = []
    for report in reports:
        metrics = report.get("results", {}).get(target_prefix)
        if metrics is None:
            # Compatibility with the original one-scenario seed41 report.
            metrics = report.get("metrics", {}) if scenario_name == "spatiotemp_100x150" else {}
        rows.append(
            {
                "scenario": scenario_name,
                "target": target_prefix,
                **_split_columns(space_split),
                "model": model_name,
                "alpha_obs": "",
                "lambda_unobs": "",
                "row_type": "seed",
                "seed": report.get("config", {}).get("seed", ""),
                "RMSE": _fmt_seed_metric(metrics, "", "RMSE"),
                "MSE": _fmt_seed_metric(metrics, "", "MSE"),
                "MAE": _fmt_seed_metric(metrics, "", "MAE"),
                "R2": _fmt_seed_metric(metrics, "", "R2"),
            }
        )

    def mean_std(metric_name: str) -> str:
        values = [
            float(metrics[metric_name])
            for report in reports
            for metrics in [report.get("results", {}).get(target_prefix, {})]
            if metrics.get(metric_name) is not None
        ]
        if not values:
            return ""
        mean_value = sum(values) / len(values)
        if len(values) == 1:
            return f"{mean_value:.6f}"
        # Match the existing Weather2K SVGP/STDK/DLinear tables (ddof=0).
        variance = sum((value - mean_value) ** 2 for value in values) / len(values)
        return f"{mean_value:.6f} +/- {variance ** 0.5:.6f}"

    summary = {
        "scenario": scenario_name,
        "target": target_prefix,
        **_split_columns(space_split),
        "model": model_name,
        "alpha_obs": "",
        "lambda_unobs": "",
        "row_type": "mean_std",
        "seed": "mean +/- std",
        "RMSE": mean_std("RMSE"),
        "MSE": mean_std("MSE"),
        "MAE": mean_std("MAE"),
        "R2": mean_std("R2"),
    }
    rows.append(summary)
    return rows, summary


if __name__ == "__main__":
    qconv_attempted = False
    qconv_ready = False
    for sc in SCENARIOS:
        print("\n=== Scenario:", sc["name"], "===")
        result_rows = []
        summary_rows = []
        rc = 0

        # STDK and SVGP both consume the same combined train500, so run each
        # baseline once per scenario. 100/400 is retained only as the canonical
        # interface split; it does not change their combined 500-station input.
        baseline_split = (100, 400, 100)
        baseline_env = apply_space_split(sc["env"], baseline_split)
        baseline_suffix = f"{sc['env']['RESULT_SUFFIX']}_train500_test100"
        svgp_env = dict(baseline_env)
        svgp_env["RESULT_SUFFIX"] = f"{baseline_suffix}_{svgp_suffix()}"

        if RUN_SVGP:
            print("\n--- SVGP baseline: train500/test100 (run once) ---")
            rc = run_script(SVGP_SCRIPT, svgp_env)
            if rc == 0:
                result_path = _svgp_result_path(svgp_env["RESULT_SUFFIX"])
                if result_path.exists():
                    _archive_json(result_path)
                    payload = _load_json(result_path)
                    result_rows.extend(_svgp_seed_rows(payload, sc["name"], baseline_split))
                    summary_row = _svgp_summary_row(payload, sc["name"], baseline_split)
                    result_rows.append(summary_row)
                    summary_rows.append(summary_row)
                    _write_result_files(result_rows, sc["name"])
        elif ARCHIVE_DIR is not None:
            archived_svgp = ARCHIVE_DIR / "json" / _svgp_result_path(
                svgp_env["RESULT_SUFFIX"]
            ).name
            if archived_svgp.exists():
                print(f"Reusing archived SVGP result: {archived_svgp}")
                payload = _load_json(archived_svgp)
                result_rows.extend(_svgp_seed_rows(payload, sc["name"], baseline_split))
                summary_row = _svgp_summary_row(payload, sc["name"], baseline_split)
                result_rows.append(summary_row)
                summary_rows.append(summary_row)
                _write_result_files(result_rows, sc["name"])
            else:
                raise FileNotFoundError(
                    f"RUN_SVGP=0 but archived SVGP result is missing: {archived_svgp}"
                )

        if rc != 0:
            print(f"SVGP failed for scenario {sc['name']}; skipping remaining models.")
            continue

        if RUN_STDK:
            stdk_env = dict(baseline_env)
            stdk_env["RESULT_SUFFIX"] = baseline_suffix
            print("\n--- STDK baseline: train500/test100 (run once) ---")
            rc = run_script(STDK_SCRIPT, stdk_env)
            if rc == 0:
                result_path = RESULT_ROOT / f"2K_stdk_metrics_{stdk_env['RESULT_SUFFIX']}.json"
                if result_path.exists():
                    _archive_json(result_path)
                    payload = _load_json(result_path)
                    result_rows.extend(_seed_rows(payload, sc["name"], baseline_split))
                    summary_row = _summary_row(payload, sc["name"], baseline_split)
                    result_rows.append(summary_row)
                    summary_rows.append(summary_row)
                    _write_result_files(result_rows, sc["name"])

        if rc != 0:
            print(f"STDK failed for scenario {sc['name']}; skipping DLinear+FRK sweep.")
            continue

        if RUN_QCONVLSTM:
            if not qconv_attempted:
                qconv_attempted = True
                qconv_args = [
                    "--seeds",
                    *(str(seed) for seed in QCONVLSTM_SEEDS),
                    "--output-dir",
                    str(QCONVLSTM_OUTPUT_DIR),
                    "--forecast-mode",
                    QCONVLSTM_FORECAST_MODE,
                ]
                if os.environ.get("QCONV_STDK_EPOCHS"):
                    qconv_args.extend(["--stdk-epochs", os.environ["QCONV_STDK_EPOCHS"]])
                if os.environ.get("QCONV_EPOCHS"):
                    qconv_args.extend(["--qconv-epochs", os.environ["QCONV_EPOCHS"]])
                if os.environ.get("QCONV_BATCH_SIZE"):
                    qconv_args.extend(["--qconv-batch-size", os.environ["QCONV_BATCH_SIZE"]])
                print(
                    "\n--- STDK+SharedQConvLSTM: one fit for time, space, and "
                    "spatiotemporal targets; "
                    f"mode={QCONVLSTM_FORECAST_MODE} (opt-in) ---"
                )
                qconv_rc = run_script(QCONV_SCRIPT, baseline_env, qconv_args)
                qconv_ready = qconv_rc == 0
                if not qconv_ready:
                    print(
                        f"SharedQConvLSTM failed with code {qconv_rc}; "
                        "continuing with DLinear+FRK."
                    )
            if qconv_ready:
                qconv_rows, qconv_summary = _qconvlstm_rows(
                    QCONVLSTM_OUTPUT_DIR, sc["name"], baseline_split
                )
                summary_path = QCONVLSTM_OUTPUT_DIR / (
                    f"shared_qconvlstm_{QCONVLSTM_FORECAST_MODE}_five_seed_summary.json"
                )
                _archive_json(summary_path)
                _archive_qconvlstm_forecasts(QCONVLSTM_OUTPUT_DIR)
                result_rows.extend(qconv_rows)
                summary_rows.append(qconv_summary)
                _write_result_files(result_rows, sc["name"])

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
                        effective_alpha = FORMAL_ALPHA if alpha_value is None else str(alpha_value)
                        effective_lambda = FORMAL_LAMBDA if lambda_value is None else str(lambda_value)
                        if alpha_value is not None:
                            dlinear_env["DIFF_FRK_OBS_LOSS_WEIGHT"] = effective_alpha
                            dlinear_env["DIFF_FRK_LOSS_WEIGHT"] = effective_lambda
                        dlinear_env["RESULT_SUFFIX"] = (
                            f"{base_suffix}_"
                            f"{alpha_suffix(effective_alpha)}_{lambda_suffix(effective_lambda)}"
                        )
                        print(
                            f"\n--- DLinear+FRK split={sc_env['SPACE_SPLIT_LABEL']}, "
                            f"alpha={effective_alpha}, lambda={effective_lambda} "
                            f"({'override' if alpha_value is not None else 'tuned JSON'}) ---"
                        )
                        rc = run_script(DLIN_SCRIPT, dlinear_env)
                        if rc != 0:
                            print(
                                f"Script {DLIN_SCRIPT.name} exited with code {rc}. "
                                f"Aborting scenario {sc['name']} at split={sc_env['SPACE_SPLIT_LABEL']}, "
                                f"alpha={effective_alpha}, lambda={effective_lambda}"
                            )
                            break
                        result_path = RESULT_ROOT / f"dlinear_autofrk_frkloss_test_100to500_metrics_{dlinear_env['RESULT_SUFFIX']}.json"
                        if result_path.exists():
                            _archive_json(result_path)
                            rerun_path = RESULT_ROOT / f"2K_best_dlinear_frkloss_rerun_metrics_500to100_{dlinear_env['RESULT_SUFFIX']}.json"
                            _archive_json(rerun_path)
                            payload = _load_json(result_path)
                            result_rows.extend(_seed_rows(payload, sc["name"], space_split, effective_alpha, effective_lambda))
                            summary_row = _summary_row(payload, sc["name"], space_split, effective_alpha, effective_lambda)
                            result_rows.append(summary_row)
                            summary_rows.append(summary_row)
                            _write_result_files(result_rows, sc["name"])
                    if rc != 0:
                        break
                else:
                    rc = 0
                if rc != 0:
                    break
        else:
            rc = 0

        if rc != 0:
            continue

        _print_summary_table(summary_rows, sc["name"])
        _update_archive_readme(summary_rows, sc["name"])
        print(f"Scenario {sc['name']} completed.\n")

    print(f"All scenarios invoked. Check generated outputs in {RESULT_ROOT}.")
