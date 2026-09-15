#!/usr/bin/env python3
"""Build the author-data STDK/QLSTM/QConvLSTM comparison table.

Public author settings are followed directly. The unavailable
``50k_lstm_data.csv`` bridge is reconstructed as the fitted STDK
quantile-specific series at each location. QLSTM is then fitted separately at every location following
the released 50K notebook: lookback 5, LSTM(50, relu), Dense(1), batch 128,
5% validation, 120 epochs, and recursive five-step forecasting with each
quantile prediction fed back into its corresponding history. For the interval heads, the paper's
non-crossing Eq. (7) takes precedence over the notebook's three independent
linear outputs: q50 is fitted first, followed by median-centred q05 and q95.

TensorFlow/Keras is absent from the project environment, so the QLSTM cell and
Keras initializers are implemented compatibly in PyTorch. The author did not
publish a seed, so bit-for-bit equality with the notebook is not claimed.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

CODE_DIR = Path(__file__).resolve().parent
BUNDLE_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import STDK_QConvLSTM_reproduction as reproduction
import compare_refit_stdk_vs_qconvlstm as baseline


TAG = baseline.TAG
MODEL_NAMES = ("STDK", "STDK_plus_QLSTM", "STDK_plus_QConvLSTM")
METRICS = ("MSPE", "RMSE", "MAE", "MPIW", "Coverage_percent")
PAPER_TABLE2 = {
    "STDK": None,
    "STDK_plus_QLSTM": {"MSPE": 0.392, "MPIW": 1.558, "Coverage_percent": 89.94},
    "STDK_plus_QConvLSTM": {"MSPE": 0.267, "MPIW": 1.462, "Coverage_percent": 90.39},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43, 44, 45])
    parser.add_argument("--data-dir", type=Path, default=BUNDLE_ROOT / "data")
    parser.add_argument("--qconv-dir", type=Path, default=BUNDLE_ROOT / "per_seed")
    parser.add_argument(
        "--stdk-dir", type=Path, default=BUNDLE_ROOT / "stdk_vs_qconvlstm",
        help="Directory containing the compatible refitted STDK checkpoints",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=BUNDLE_ROOT / "table2_stdk_qlstm_qconvlstm",
    )
    parser.add_argument("--qlstm-epochs", type=int, default=120)
    parser.add_argument("--target-indices", type=int, nargs="+", choices=range(100))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--quick", action="store_true",
        help="Use two QLSTM epochs; only validates the pipeline.",
    )
    return parser.parse_args()


class KerasReluLSTM(nn.Module):
    """One-layer Keras-compatible LSTM with activation=relu."""

    def __init__(self, units: int = 50):
        super().__init__()
        self.units = units
        self.kernel = nn.Parameter(torch.empty(1, 4 * units))
        self.recurrent_kernel = nn.Parameter(torch.empty(units, 4 * units))
        self.bias = nn.Parameter(torch.zeros(4 * units))
        self.output_kernel = nn.Parameter(torch.empty(units, 1))
        self.output_bias = nn.Parameter(torch.zeros(1))
        nn.init.xavier_uniform_(self.kernel)
        nn.init.orthogonal_(self.recurrent_kernel)
        nn.init.xavier_uniform_(self.output_kernel)
        with torch.no_grad():
            self.bias[units : 2 * units].fill_(1.0)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        batch = sequence.shape[0]
        h = sequence.new_zeros((batch, self.units))
        c = sequence.new_zeros((batch, self.units))
        for step in range(sequence.shape[1]):
            gates = sequence[:, step] @ self.kernel + h @ self.recurrent_kernel + self.bias
            i, f, candidate, o = gates.chunk(4, dim=1)
            i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
            candidate = torch.relu(candidate)
            c = f * c + i * candidate
            h = o * torch.relu(c)
        return (h @ self.output_kernel + self.output_bias).squeeze(1)


def qlstm_windows(series: np.ndarray, lookback: int = 5):
    # Reproduce notebook: training_size = 495 - (n_steps + n_output).
    count = len(series) - (lookback + 1)
    x = np.stack([series[start : start + lookback] for start in range(count)])
    y = np.asarray([series[start + lookback] for start in range(count)])
    return x[:, :, None].astype(np.float32), y.astype(np.float32)


def fit_quantile_qlstm(
    x: np.ndarray, y: np.ndarray, quantile: float, epochs: int,
    seed: int, device: torch.device, median_model: KerasReluLSTM | None = None,
    quantile_lambda: float | None = None, median_x: np.ndarray | None = None,
) -> KerasReluLSTM:
    # Keras validation_split takes the final fraction before training shuffles.
    split_at = int(len(x) * 0.95)
    tensors = [torch.from_numpy(x[:split_at]), torch.from_numpy(y[:split_at])]
    if median_model is not None:
        if median_x is None or len(median_x) != len(x):
            raise ValueError("tail QLSTM requires aligned q50 input windows")
        tensors.append(torch.from_numpy(median_x[:split_at]))
    train_ds = TensorDataset(*tensors)
    loader = DataLoader(
        train_ds, batch_size=128, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    model = KerasReluLSTM(50).to(device)
    # Keras Adam defaults to epsilon=1e-7.
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-7)
    if median_model is not None:
        median_model.eval()
        for parameter in median_model.parameters():
            parameter.requires_grad_(False)
    for _ in range(epochs):
        model.train()
        for batch in loader:
            xb, yb = batch[:2]
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            raw = model(xb)
            if median_model is None:
                prediction = raw
            else:
                median_xb = batch[2].to(device)
                with torch.no_grad():
                    median = median_model(median_xb)
                prediction = reproduction.constrained_quantile(
                    raw, median, quantile, float(quantile_lambda)
                )
            loss = reproduction.pinball(prediction, yb, quantile)
            loss.backward()
            optimizer.step()
    if median_model is not None:
        for parameter in median_model.parameters():
            parameter.requires_grad_(True)
    return model


def forecast_qlstm_location(
    quantile_series: np.ndarray, seed: int, epochs: int, device: torch.device,
    quantile_lambda: float,
) -> tuple[np.ndarray, dict[str, dict[str, torch.Tensor]]]:
    if quantile_series.shape[1] != 3:
        raise ValueError("expected q05/q50/q95 STDK series")
    lower_x, lower_y = qlstm_windows(quantile_series[:, 0])
    median_x, median_y = qlstm_windows(quantile_series[:, 1])
    upper_x, upper_y = qlstm_windows(quantile_series[:, 2])
    reproduction.seed_everything(seed)
    # Paper Eq. (7) requires the fitted median before either interval head.
    median_model = fit_quantile_qlstm(median_x, median_y, 0.5, epochs, seed, device)
    lower_raw = fit_quantile_qlstm(
        lower_x, lower_y, 0.05, epochs, seed, device, median_model,
        quantile_lambda, median_x,
    )
    upper_raw = fit_quantile_qlstm(
        upper_x, upper_y, 0.95, epochs, seed, device, median_model,
        quantile_lambda, median_x,
    )
    histories = [
        torch.from_numpy(quantile_series[-5:, index, None][None].astype(np.float32)).to(device)
        for index in range(3)
    ]
    predictions = []
    for _ in range(5):
        with torch.no_grad():
            median = median_model(histories[1])
            lower = reproduction.constrained_quantile(
                lower_raw(histories[0]), median, 0.05, quantile_lambda
            )
            upper = reproduction.constrained_quantile(
                upper_raw(histories[2]), median, 0.95, quantile_lambda
            )
            step = np.array([lower.item(), median.item(), upper.item()], dtype=np.float32)
        predictions.append(step)
        for index in range(3):
            new_value = torch.tensor([[[step[index]]]], dtype=torch.float32, device=device)
            histories[index] = torch.cat((histories[index][:, 1:], new_value), dim=1)
    states = {
        "q05_raw": lower_raw.state_dict(),
        "q50": median_model.state_dict(),
        "q95_raw": upper_raw.state_dict(),
    }
    return np.stack(predictions), states


def predict_stdk_quantile_series(models, cfg, coords, total_times: int, device) -> np.ndarray:
    features = reproduction.make_stdk_features(
        coords, np.arange(cfg.train_times), total_times, cfg.stdk_profile
    )
    parts = []
    for model in models.values():
        model.eval()
    with torch.no_grad():
        for start in range(0, len(features), cfg.stdk_batch_size):
            xb = torch.from_numpy(features[start : start + cfg.stdk_batch_size]).to(device)
            median = models[0.5](xb)
            lower = reproduction.constrained_quantile(
                models[0.05](xb), median, 0.05, cfg.quantile_lambda
            )
            upper = reproduction.constrained_quantile(
                models[0.95](xb), median, 0.95, cfg.quantile_lambda
            )
            parts.append(torch.cat((lower, median, upper), dim=1).cpu().numpy())
    # Feature ordering is time-major; return location-major.
    return np.concatenate(parts).reshape(cfg.train_times, len(coords), 3).transpose(1, 0, 2)


def load_stdk_models(cfg, coords, values_model_scale, args, device):
    baseline._COORDS = coords
    models, checkpoint_path = baseline.fit_or_load_stdk(
        cfg, values_model_scale, args.stdk_dir, False, device
    )
    return models, checkpoint_path


def load_qconv_prediction(qconv_dir: Path, seed: int, shape) -> tuple[np.ndarray, np.ndarray]:
    rows = baseline.load_qconv_forecasts(qconv_dir, seed)
    prediction = np.empty((*shape, 3), dtype=np.float32)
    truth = np.empty(shape, dtype=np.float32)
    for row in rows:
        location, lead = int(row["target_index"]), int(row["lead"]) - 1
        truth[location, lead] = float(row["truth"])
        prediction[location, lead] = [float(row["q05"]), float(row["q50"]), float(row["q95"])]
    return prediction, truth


def run_seed(args: argparse.Namespace, seed: int, device) -> dict:
    cfg, source_aggregate = baseline.load_formal_config(args.qconv_dir, seed, False)
    coords, _, values_raw = reproduction.load_simulation(args.data_dir, cfg.add_paper_nonstationary_mean)
    if cfg.normalization == "train_global":
        mean = float(values_raw[: cfg.train_times].mean())
        std = float(values_raw[: cfg.train_times].std()) or 1.0
        values_model_scale = ((values_raw - mean) / std).astype(np.float32)
    else:
        mean, std, values_model_scale = 0.0, 1.0, values_raw
    stdk_models, stdk_checkpoint = load_stdk_models(cfg, coords, values_model_scale, args, device)
    stdk_train_quantiles = predict_stdk_quantile_series(
        stdk_models, cfg, coords, len(values_raw), device
    )
    stdk_test = baseline.predict_stdk(stdk_models, cfg, coords, len(values_raw), device) * std + mean
    truth = values_raw[cfg.train_times : cfg.train_times + cfg.horizon].T
    qconv, qconv_truth = load_qconv_prediction(args.qconv_dir, seed, truth.shape)
    if not np.allclose(qconv_truth, truth, rtol=0, atol=1e-6):
        raise AssertionError("QConvLSTM truth mismatch")

    targets = args.target_indices if args.target_indices is not None else list(range(len(coords)))
    qlstm = np.full((len(coords), cfg.horizon, 3), np.nan, dtype=np.float32)
    epochs = 2 if args.quick else args.qlstm_epochs
    location_dir = args.output_dir / "checkpoints" / "qlstm_locations" / "current"
    location_dir.mkdir(parents=True, exist_ok=True)
    for position, target in enumerate(targets, 1):
        stem = f"qlstm_papereq7_qspecific_target{target:03d}_seed{seed}_epochs{epochs}"
        json_path, checkpoint_path = location_dir / f"{stem}.json", location_dir / f"{stem}.pt"
        prediction = None
        if json_path.exists() and checkpoint_path.exists() and not args.overwrite:
            saved = json.loads(json_path.read_text(encoding="utf-8"))
            if saved.get("seed") == seed and saved.get("target_index") == target and saved.get("epochs") == epochs:
                prediction = np.asarray(saved["prediction"], dtype=np.float32)
        if prediction is None:
            prediction_model_scale, states = forecast_qlstm_location(
                stdk_train_quantiles[target], seed + target, epochs, device,
                cfg.quantile_lambda,
            )
            prediction = prediction_model_scale * std + mean
            torch.save(
                {"seed": seed, "target_index": target, "epochs": epochs, "state_dicts": states},
                checkpoint_path,
            )
            json_path.write_text(json.dumps({
                "seed": seed, "target_index": target, "epochs": epochs,
                "input": "quantile-specific fitted STDK q05/q50/q95 series, following paper X^NN_tau definition",
                "interval_heads": "paper Eq. (7), median-centred non-crossing q05/q95",
                "quantile_lambda": cfg.quantile_lambda,
                "prediction": prediction.tolist(),
            }, indent=2), encoding="utf-8")
        qlstm[target] = prediction
        if position == 1 or position % 10 == 0 or position == len(targets):
            print(f"[seed {seed}] QLSTM locations {position}/{len(targets)}", flush=True)

    selected = np.asarray(targets, dtype=int)
    model_predictions = {
        "STDK": stdk_test[selected],
        "STDK_plus_QLSTM": qlstm[selected],
        "STDK_plus_QConvLSTM": qconv[selected],
    }
    selected_truth = truth[selected]
    model_metrics = {
        model: baseline.metrics(prediction, selected_truth)
        for model, prediction in model_predictions.items()
    }
    report = {
        "seed": seed,
        "quick_diagnostic": bool(args.quick),
        "completed_locations": len(targets),
        "evaluation": {"times": [496, 500], "predictions": len(targets) * 5},
        "sources": {
            "STDK": str(stdk_checkpoint),
            "QConvLSTM": str(source_aggregate),
            "QLSTM": "paper quantile-specific X^NN series and Eq. (7), with architecture/training from the author 50K notebook",
        },
        "author_setting_mapping": {
            "data_and_split": "paper Section 4.2: released 100x500 simulation; train times 1--495, forecast 496--500",
            "STDK": "released 50kSimulation-space-time_DeepKriging notebook; deterministic compatible refit",
            "QLSTM": "released 50K notebook architecture/training plus paper Eq. (7) median-centred q05/q95; rolling recursive five-step forecast",
            "QConvLSTM": "released CONV_LSTM notebook architecture combined with paper quantile objective; previously completed paired forecasts",
            "implementation_runtime": "PyTorch Keras-compatible reconstruction because TensorFlow/Keras is not installed",
        },
        "models": model_metrics,
        "paper_table2": PAPER_TABLE2,
        "comparisons": {
            "QLSTM_minus_STDK_MSPE": model_metrics["STDK_plus_QLSTM"]["MSPE"] - model_metrics["STDK"]["MSPE"],
            "QConvLSTM_minus_QLSTM_MSPE": model_metrics["STDK_plus_QConvLSTM"]["MSPE"] - model_metrics["STDK_plus_QLSTM"]["MSPE"],
            "QConvLSTM_minus_STDK_MSPE": model_metrics["STDK_plus_QConvLSTM"]["MSPE"] - model_metrics["STDK"]["MSPE"],
        },
        "assumptions": {
            "missing_bridge": "50k_lstm_data.csv is unavailable; q05/q50/q95 inputs are reconstructed from their corresponding fitted STDK quantile series following paper X^NN_tau",
            "qlstm_locations": "paper says 100; released notebook loop says 50; this comparison uses all 100",
            "random_seed": "not reported by authors; study seeds 41--45 are used",
            "recursion_conflict": "repo code retains the oldest four lags; paper Algorithm 1 implies a rolling window, which is used here",
        },
    }
    per_seed_dir = args.output_dir / "per_seed"
    per_seed_dir.mkdir(parents=True, exist_ok=True)
    out = per_seed_dir / f"table2_three_models_seed{seed}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    forecast_path = per_seed_dir / f"table2_three_models_forecasts_seed{seed}.csv"
    with forecast_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("seed", "target_index", "lead", "truth", "model", "q05", "q50", "q95"))
        for local_index, target in enumerate(targets):
            for lead in range(cfg.horizon):
                for model in MODEL_NAMES:
                    row = model_predictions[model][local_index, lead]
                    writer.writerow((
                        seed, target, lead + 1, float(selected_truth[local_index, lead]), model,
                        float(row[0]), float(row[1]), float(row[2]),
                    ))
    print(json.dumps(report, indent=2), flush=True)
    return report


def summarize(reports: list[dict], output_dir: Path, quick: bool) -> dict:
    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    def mean_sd(item: dict, digits: int) -> str:
        mean = item["mean"]
        std = item["std_sample_ddof1"]
        return f"{mean:.{digits}f}" if std is None else f"{mean:.{digits}f} +/- {std:.{digits}f}"

    summary = {
        "seeds": [report["seed"] for report in reports],
        "quick_diagnostic": quick,
        "models": {},
        "paper_table2": PAPER_TABLE2,
    }
    for model in MODEL_NAMES:
        summary["models"][model] = {}
        for metric in METRICS:
            values = [report["models"][model][metric] for report in reports]
            summary["models"][model][metric] = {
                "mean": statistics.fmean(values),
                "std_sample_ddof1": statistics.stdev(values) if len(values) > 1 else None,
                "values": values,
            }
    qlstm_mspe = summary["models"]["STDK_plus_QLSTM"]["MSPE"]["mean"]
    qconv_mspe = summary["models"]["STDK_plus_QConvLSTM"]["MSPE"]["mean"]
    summary["qconv_vs_qlstm_MSPE_reduction_percent"] = 100.0 * (qlstm_mspe - qconv_mspe) / qlstm_mspe
    summary["qconv_better_than_qlstm_seed_count"] = sum(
        report["models"]["STDK_plus_QConvLSTM"]["MSPE"]
        < report["models"]["STDK_plus_QLSTM"]["MSPE"]
        for report in reports
    )
    json_path = summary_dir / "table2_three_models_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (summary_dir / "table2_three_models_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("model", *METRICS))
        for model in MODEL_NAMES:
            writer.writerow((model, *(summary["models"][model][metric]["mean"] for metric in METRICS)))
    lines = [
        "# Author-data STDK, STDK+QLSTM and STDK+QConvLSTM", "",
        "Artifacts are separated into `checkpoints/`, `per_seed/`, `summary/`, "
        "`logs/` and `metadata/`.", "",
        "| Model | Reproduced MSPE (mean +/- SD) | Paper MSPE | Reproduced MPIW (mean +/- SD) | Paper MPIW | Reproduced coverage % (mean +/- SD) | Paper coverage (%) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODEL_NAMES:
        item, paper = summary["models"][model], PAPER_TABLE2[model]
        paper_mspe = "--" if paper is None else f"{paper['MSPE']:.3f}"
        paper_mpiw = "--" if paper is None else f"{paper['MPIW']:.3f}"
        paper_coverage = "--" if paper is None else f"{paper['Coverage_percent']:.2f}"
        lines.append(
            f"| {model} | {mean_sd(item['MSPE'], 6)} | {paper_mspe} | "
            f"{mean_sd(item['MPIW'], 6)} | {paper_mpiw} | "
            f"{mean_sd(item['Coverage_percent'], 3)} | {paper_coverage} |"
        )
    lines.extend((
        "", f"QConvLSTM versus QLSTM MSPE reduction: `{summary['qconv_vs_qlstm_MSPE_reduction_percent']:.2f}%`.",
        f"QConvLSTM has lower MSPE than QLSTM in `{summary['qconv_better_than_qlstm_seed_count']}/{len(reports)}` seeds.",
        "", "The missing `50k_lstm_data.csv` bridge is reconstructed from the quantile-specific fitted STDK q05/q50/q95 series following the paper's X^NN_tau definition; this remains a best-effort reproduction.",
        "The paper values are references, not values copied into the reproduced metrics.", "",
    ))
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {json_path}", flush=True)
    return summary


def main() -> None:
    args = parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("seeds must be unique")
    if args.target_indices is not None and len(set(args.target_indices)) != len(args.target_indices):
        raise ValueError("target indices must be unique")
    if args.qlstm_epochs < 1:
        raise ValueError("QLSTM epochs must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    reports = [run_seed(args, seed, device) for seed in args.seeds]
    if args.target_indices is None:
        summarize(reports, args.output_dir, args.quick)
    else:
        print("Subset pilot completed; full Table 2 summary was not written.", flush=True)


if __name__ == "__main__":
    main()
