#!/usr/bin/env python3
"""Recompute pure-STDK forecasts and compare every point with Q's Stage 1."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = ROOT / "Josh's Weather2K/Weather2K/weather2k.npy"
MODEL_SOURCE = ROOT / "spatial-adapter/examples/baselines/stdk/st_interp.py"
SCENARIOS = {
    "time_extrap_fixed500": ("Target_Time150", "train_local", 850, 1000, "stdk_time_forecasts.csv"),
    "space_extrap_fixed850": ("Target_Space100", "heldout_local", 0, 850, "stdk_space_forecasts.csv"),
    "spatiotemp_100x150": ("Target_ST100x150", "heldout_local", 850, 1000, "stdk_forecasts.csv"),
}


def main():
    spec = importlib.util.spec_from_file_location("stdk_stage1_audit", MODEL_SOURCE)
    spatial = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(spatial)
    raw = np.load(DATA, mmap_mode="r")
    original_coords = np.column_stack((raw[:, 1, 0], raw[:, 0, 0])).astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}

    for scenario, (target, stations_key, start_time, end_time, suffix) in SCENARIOS.items():
        checkpoint_dir = HERE / "checkpoints" / f"{scenario}_seed41"
        checkpoint = checkpoint_dir / "model_best.pt"
        metadata = json.loads((checkpoint_dir / "metadata.json").read_text())
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        assert digest == metadata["checkpoint_sha256"]
        chosen = np.asarray(metadata["sampled_global"], dtype=int)
        coords = original_coords[chosen]
        lower, upper = coords.min(axis=0), coords.max(axis=0)
        span = np.where(upper - lower < 1e-12, 1.0, upper - lower)
        coords = ((coords - lower) / span).astype(np.float32)
        train = np.asarray(metadata["train_local"], dtype=int)
        stations = np.asarray(metadata[stations_key], dtype=int)
        model = spatial.create_model(metadata["model_config"], train_coords=coords[train]).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        model.eval()
        times = np.arange(start_time, end_time, dtype=np.int64)
        batch_size = int(metadata["model_config"]["batch_size"])
        predictions = np.empty((len(times), len(stations)), dtype=np.float32)
        with torch.no_grad():
            for index, time_index in enumerate(times):
                time_value = np.float32(time_index / (metadata["time"]["n_last"] - 1))
                for offset in range(0, len(stations), batch_size):
                    selected = stations[offset : offset + batch_size]
                    c = torch.from_numpy(coords[selected]).to(device)
                    t = torch.full((len(selected), 1), float(time_value), device=device)
                    x = torch.empty((len(selected), 0), device=device)
                    predictions[index, offset : offset + len(selected)] = (
                        model(x, c, t).cpu().numpy().reshape(-1)
                    )
        norm = metadata["normalization"]
        predictions = predictions * norm["std"] + norm["mean"]

        csv_path = HERE / "forecasts" / f"paired_stdk_truth_nag_qconvlstm_block5to5_{scenario}_seed41_{suffix}"
        with csv_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == predictions.size
        observed = {(int(row["time_index"]), int(row["station_local"])): row for row in rows}
        differences = []
        squared_errors = []
        for i, time_index in enumerate(times):
            for j, station in enumerate(stations):
                row = observed[(int(time_index), int(station))]
                q50 = float(row["q50"])
                assert float(row["q05"]) == q50 == float(row["q95"])
                differences.append(abs(float(predictions[i, j]) - q50))
                squared_errors.append((float(predictions[i, j]) - float(row["truth"])) ** 2)
        recomputed_rmse = float(np.sqrt(np.mean(squared_errors)))
        pure_rmse = float(metadata["target_metrics"][target]["RMSE"])
        max_abs = max(differences)
        assert abs(recomputed_rmse - pure_rmse) < 1e-4, (scenario, recomputed_rmse, pure_rmse)
        assert max_abs < 1e-4, (scenario, max_abs)
        results[scenario] = {
            "target": target,
            "checkpoint_sha256": digest,
            "compared_predictions": len(rows),
            "max_absolute_prediction_difference": max_abs,
            "pure_stdk_rmse": pure_rmse,
            "recomputed_pure_stdk_rmse": recomputed_rmse,
            "point_predictions_same_within_1e-4": True,
        }
        print(json.dumps({scenario: results[scenario]}, ensure_ascii=False), flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    (HERE / "metrics" / "stage1_prediction_audit.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
