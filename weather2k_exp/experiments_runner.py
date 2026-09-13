"""
簡易實驗 runner：每個模型、每個 seed、每組超參數只訓練一次，再用同一個
fitted model 評估時間、空間與時空三個外推情境。
時間固定使用最後 1000 個時間點並切成 700/150/150。
空間固定抽 600 個空間點；obs100 + unobs400 共 500 站提供訓練監督，
其餘 100 站保留作空間/時空測試。
會產生一份含三種 target 的模型 JSON，再輸出三張情境比較表。
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
QCONVLSTM_PREDICTION_MODE = os.environ.get(
    "QCONVLSTM_PREDICTION_MODE", "direct"
)
if QCONVLSTM_PREDICTION_MODE not in {"direct", "residual"}:
    raise ValueError("QCONVLSTM_PREDICTION_MODE must be direct or residual")
QCONVLSTM_OUTPUT_DIR = Path(
    os.environ.get(
        "QCONVLSTM_OUTPUT_DIR",
        str(RESULT_ROOT),
    )
).expanduser().resolve()
QCONVLSTM_PARAMS_FILE = Path(
    os.environ.get(
        "QCONVLSTM_PARAMS_FILE",
        str(ROOT / "2K_best_stdk_qconvlstm_params_500to100.json"),
    )
).expanduser().resolve()
if RUN_QCONVLSTM and not QCONVLSTM_PARAMS_FILE.exists():
    raise FileNotFoundError(
        f"Missing QConvLSTM parameter file: {QCONVLSTM_PARAMS_FILE}"
    )
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
    # All scenarios share the same Train700, Val150, and station split. Fit each
    # model/configuration once, then read its three target metrics into the
    # scenario-specific tables below.
    baseline_split = (100, 400, 100)
    canonical_env = apply_space_split(SCENARIOS[0]["env"], baseline_split)
    canonical_env["EXPERIMENT_SCENARIO"] = "all_three"

    svgp_payload = None
    stdk_payload = None
    dlinear_payloads = []

    baseline_suffix = "all_three_train500_test100"
    svgp_env = dict(canonical_env)
    svgp_env["RESULT_SUFFIX"] = f"{baseline_suffix}_{svgp_suffix()}"
    if RUN_SVGP:
        print("\n--- SVGP: one fit per seed, three evaluations ---")
        rc = run_script(SVGP_SCRIPT, svgp_env)
        if rc != 0:
            raise SystemExit(f"SVGP failed with exit code {rc}")
        svgp_path = _svgp_result_path(svgp_env["RESULT_SUFFIX"])
        if not svgp_path.exists():
            raise FileNotFoundError(svgp_path)
        _archive_json(svgp_path)
        svgp_payload = _load_json(svgp_path)
    elif ARCHIVE_DIR is not None:
        svgp_path = ARCHIVE_DIR / "json" / _svgp_result_path(
            svgp_env["RESULT_SUFFIX"]
        ).name
        if not svgp_path.exists():
            raise FileNotFoundError(
                f"RUN_SVGP=0 but unified all-three result is missing: {svgp_path}"
            )
        svgp_payload = _load_json(svgp_path)

    stdk_env = dict(canonical_env)
    stdk_env["RESULT_SUFFIX"] = baseline_suffix
    if RUN_STDK:
        print("\n--- STDK: one fit per seed, three evaluations ---")
        rc = run_script(STDK_SCRIPT, stdk_env)
        if rc != 0:
            raise SystemExit(f"STDK failed with exit code {rc}")
        stdk_path = RESULT_ROOT / f"2K_stdk_metrics_{stdk_env['RESULT_SUFFIX']}.json"
        if not stdk_path.exists():
            raise FileNotFoundError(stdk_path)
        _archive_json(stdk_path)
        stdk_payload = _load_json(stdk_path)
    elif ARCHIVE_DIR is not None:
        stdk_path = ARCHIVE_DIR / "json" / (
            f"2K_stdk_metrics_{stdk_env['RESULT_SUFFIX']}.json"
        )
        if stdk_path.exists():
            stdk_payload = _load_json(stdk_path)

    qconv_ready = False
    if RUN_QCONVLSTM:
        qconv_args = [
            "--seeds", *(str(seed) for seed in QCONVLSTM_SEEDS),
            "--params-file", str(QCONVLSTM_PARAMS_FILE),
            "--output-dir", str(QCONVLSTM_OUTPUT_DIR),
            "--forecast-mode", QCONVLSTM_FORECAST_MODE,
            "--prediction-mode", QCONVLSTM_PREDICTION_MODE,
        ]
        if os.environ.get("QCONV_STDK_EPOCHS"):
            qconv_args.extend(["--stdk-epochs", os.environ["QCONV_STDK_EPOCHS"]])
        if os.environ.get("QCONV_EPOCHS"):
            qconv_args.extend(["--qconv-epochs", os.environ["QCONV_EPOCHS"]])
        if os.environ.get("QCONV_BATCH_SIZE"):
            qconv_args.extend(["--qconv-batch-size", os.environ["QCONV_BATCH_SIZE"]])
        print("\n--- STDK+QConvLSTM: one fit per seed, three evaluations ---")
        qconv_ready = run_script(QCONV_SCRIPT, canonical_env, qconv_args) == 0
        if qconv_ready:
            summary_path = QCONVLSTM_OUTPUT_DIR / (
                f"shared_qconvlstm_{QCONVLSTM_FORECAST_MODE}_"
                + (
                    "five_seed"
                    if QCONVLSTM_SEEDS == [41, 42, 43, 44, 45]
                    else "seeds" + "_".join(str(seed) for seed in QCONVLSTM_SEEDS)
                )
                + "_summary.json"
            )
            _archive_json(summary_path)
            _archive_qconvlstm_forecasts(QCONVLSTM_OUTPUT_DIR)
        else:
            print("STDK+QConvLSTM failed; its rows will be omitted.")

    if RUN_DLINEAR:
        for space_split in SPACE_SPLITS:
            split_env = apply_space_split(canonical_env, space_split)
            split_env["EXPERIMENT_SCENARIO"] = "all_three"
            train_n, unobs_n, target_n = space_split
            base_suffix = (
                f"all_three_{split_suffix(space_split)}_"
                f"train{train_n + unobs_n}_test{target_n}"
            )
            for alpha_value in ALPHA_LIST:
                for lambda_value in LAMBDA_LIST:
                    dlinear_env = dict(split_env)
                    effective_alpha = FORMAL_ALPHA if alpha_value is None else str(alpha_value)
                    effective_lambda = FORMAL_LAMBDA if lambda_value is None else str(lambda_value)
                    if alpha_value is not None:
                        dlinear_env["DIFF_FRK_OBS_LOSS_WEIGHT"] = effective_alpha
                        dlinear_env["DIFF_FRK_LOSS_WEIGHT"] = effective_lambda
                    dlinear_env["RESULT_SUFFIX"] = (
                        f"{base_suffix}_{alpha_suffix(effective_alpha)}_"
                        f"{lambda_suffix(effective_lambda)}"
                    )
                    print(
                        "\n--- DLinear+FRK: one fit per seed, three evaluations; "
                        f"split={dlinear_env['SPACE_SPLIT_LABEL']}, "
                        f"alpha={effective_alpha}, lambda={effective_lambda} ---"
                    )
                    rc = run_script(DLIN_SCRIPT, dlinear_env)
                    if rc != 0:
                        raise SystemExit(
                            f"DLinear+FRK failed with exit code {rc} for "
                            f"split={dlinear_env['SPACE_SPLIT_LABEL']}"
                        )
                    result_path = RESULT_ROOT / (
                        "dlinear_autofrk_frkloss_test_100to500_metrics_"
                        f"{dlinear_env['RESULT_SUFFIX']}.json"
                    )
                    if not result_path.exists():
                        raise FileNotFoundError(result_path)
                    _archive_json(result_path)
                    rerun_path = RESULT_ROOT / (
                        "2K_best_dlinear_frkloss_rerun_metrics_500to100_"
                        f"{dlinear_env['RESULT_SUFFIX']}.json"
                    )
                    _archive_json(rerun_path)
                    dlinear_payloads.append(
                        (
                            space_split,
                            effective_alpha,
                            effective_lambda,
                            _load_json(result_path),
                        )
                    )

    for sc in SCENARIOS:
        scenario_name = sc["name"]
        print("\n=== Build table for scenario:", scenario_name, "===")
        result_rows = []
        summary_rows = []

        if svgp_payload is not None:
            result_rows.extend(
                _svgp_seed_rows(svgp_payload, scenario_name, baseline_split)
            )
            summary_row = _svgp_summary_row(
                svgp_payload, scenario_name, baseline_split
            )
            result_rows.append(summary_row)
            summary_rows.append(summary_row)

        if stdk_payload is not None:
            result_rows.extend(
                _seed_rows(stdk_payload, scenario_name, baseline_split)
            )
            summary_row = _summary_row(
                stdk_payload, scenario_name, baseline_split
            )
            result_rows.append(summary_row)
            summary_rows.append(summary_row)

        if qconv_ready:
            qconv_rows, qconv_summary = _qconvlstm_rows(
                QCONVLSTM_OUTPUT_DIR, scenario_name, baseline_split
            )
            result_rows.extend(qconv_rows)
            summary_rows.append(qconv_summary)

        for space_split, alpha_value, lambda_value, payload in dlinear_payloads:
            result_rows.extend(
                _seed_rows(
                    payload, scenario_name, space_split,
                    alpha_value, lambda_value,
                )
            )
            summary_row = _summary_row(
                payload, scenario_name, space_split,
                alpha_value, lambda_value,
            )
            result_rows.append(summary_row)
            summary_rows.append(summary_row)

        _write_result_files(result_rows, scenario_name)
        _print_summary_table(summary_rows, scenario_name)
        _update_archive_readme(summary_rows, scenario_name)

    print(
        f"All selected scenarios evaluated from one fitted instance per "
        f"model/seed/configuration. Check outputs in {RESULT_ROOT}."
    )
