#!/usr/bin/env python3
"""Audit paired data, normalization, validation, test targets, and Q settings."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SEED41 = HERE.parent / "20260917_truth_paired_stdk_nag_seed41"
TUNING = HERE.parent / "20260917_stdk_nag_truth_qconvlstm_tuning"
SCENARIOS = {
    "time_extrap_fixed500": ("Target_Time150", "time500", "train_local", 850, 1000, "time_forecasts.csv"),
    "space_extrap_fixed850": ("Target_Space100", "space850", "heldout_local", 0, 850, "space_forecasts.csv"),
    "spatiotemp_100x150": ("Target_ST100x150", "st_100x150", "heldout_local", 850, 1000, "forecasts.csv"),
}
PARAMETERS = {
    "QCONV_LR": "qconv_lr", "QCONV_BATCH_SIZE": "qconv_batch",
    "QCONV_EPOCHS": "qconv_epochs", "QCONV_PATIENCE": "qconv_patience",
    "CONV_FILTERS": "conv_filters", "QCONV_WEIGHT_DECAY": "qconv_weight_decay",
    "GRID_SIZE": "grid_size", "NEIGHBOURHOOD_RADIUS": "radius",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    first_params = SEED41 / "locked_q_params.json"
    params = HERE / "locked_q_params.json"
    tuned = TUNING / "qconvlstm_tuning_best_params.json"
    assert sha(first_params) == sha(params) == sha(tuned)
    locked = json.loads(params.read_text())
    assert locked["tuning_seeds"] == [41, 42]
    assert locked["selection_metric"] == "mean_q50_val_truth_rmse_scaled"
    assert locked["selection_uses_test_metrics"] is False
    assert locked["selection_uses_heldout100_truth"] is False
    assert locked["best_stage"] == "capacity"
    for seed in (41, 42):
        report_path = TUNING / "stage2_capacity" / (
            f"qconvlstm_capacity_trial{locked['best_trial']:04d}_seed{seed}_validation.json"
        )
        tuning_report = json.loads(report_path.read_text())
        assert tuning_report["config"]["validation_only"] is True
        assert tuning_report["config"]["q50_only"] is True
        assert "results" not in tuning_report
        assert tuning_report["split"]["heldout_truth_used_for_selection"] is False

    raw = np.load(ROOT / "Josh's Weather2K/Weather2K/weather2k.npy", mmap_mode="r")
    all_values = np.asarray(raw[:, 4, :], dtype=np.float32)
    prediction_audit = json.loads((HERE / "metrics" / "all_five_seed_stage1_prediction_audit.json").read_text())
    prediction_by_pair = {(r["seed"], r["scenario"]): r for r in prediction_audit}
    assert len(prediction_by_pair) == 15
    cases = []
    seed_splits = {}
    for seed in range(41, 46):
        output = SEED41 if seed == 41 else HERE / f"seed{seed}"
        for scenario, (target, suffix, station_key, start, end, forecast_suffix) in SCENARIOS.items():
            checkpoint = output / "checkpoints" / f"{scenario}_seed{seed}" / "model_best.pt"
            metadata = json.loads(checkpoint.with_name("metadata.json").read_text())
            pure = json.loads(
                (output / "metrics" / f"2K_stdk_metrics_{suffix}_train500_test100.json").read_text()
            )["seed_runs"][0]["payload"]
            q = json.loads(
                (output / "metrics" / f"paired_stdk_truth_nag_qconvlstm_block5to5_{scenario}_seed{seed}.json").read_text()
            )
            config, sampling, split = q["config"], pure["sampling_info"], q["split"]
            digest = sha(checkpoint)
            assert digest == metadata["checkpoint_sha256"]
            assert digest == pure["stdk_checkpoint"]["sha256"] == q["stdk_checkpoint"]["sha256"]
            assert str(checkpoint.resolve()) == pure["stdk_checkpoint"]["path"] == q["stdk_checkpoint"]["path"]
            assert q["fitted_stdk_baseline"]["refitted"] is False
            assert prediction_by_pair[(seed, scenario)]["pure_checkpoint_sha256"] == digest
            assert prediction_by_pair[(seed, scenario)]["point_predictions_same_within_1e-4"] is True

            sampled = metadata["sampled_global"]
            train = metadata["train_local"]
            held = metadata["heldout_local"]
            obs = metadata["obs_local"]
            assert sampled == sampling["sample_idx_global"] == split["sampled_global"]
            assert train == sampling["sample_idx_train500"] == split["train_local"]
            assert held == sampling["sample_idx_unknown_eval"] == split["heldout_local"]
            assert obs == sampling["sample_idx_train"] == q["normalization"]["station_local"]
            assert len(sampled) == 600 and len(train) == 500 and len(held) == len(obs) == 100
            assert sorted(train + held) == list(range(600))
            assert set(obs).issubset(set(train))
            station_signature = {"sampled_global": sampled, "train_local": train,
                                 "heldout_local": held, "obs_local": obs}
            if seed in seed_splits:
                assert seed_splits[seed] == station_signature
            else:
                seed_splits[seed] = station_signature

            expected_time = {"n_last": 1000, "train": 700, "val": 150, "test": 150, "stride": 1}
            assert metadata["time"] == expected_time
            assert (config["n_last"], config["train_times"], config["val_times"], config["test_times"]) == (1000, 700, 150, 150)
            assert (sampling["time_train_len"], sampling["time_val_len"], sampling["time_test_len"]) == (700, 150, 150)

            source = all_values[np.asarray(sampled)[np.asarray(obs)], -1000:][:, :700]
            expected_mean = float(source.mean())
            expected_std = float(source.std(ddof=0))
            norm = metadata["normalization"]
            qnorm = q["normalization"]
            assert norm["source"] == qnorm["source"] == "obs100_train700"
            assert norm["ddof"] == qnorm["ddof"] == 0
            assert norm["mean"] == qnorm["mean"]
            assert norm["std"] == qnorm["std"]
            # NumPy float32 reduction may accumulate the same values in a
            # different order after slicing; the saved scalers must be exact.
            assert abs(norm["mean"] - expected_mean) < 1e-5
            assert abs(norm["std"] - expected_std) < 1e-5
            assert q["fitted_stdk_baseline"]["normalization"] == qnorm

            stdk_training = metadata["training_summary"]
            assert pure["training_summary"] == stdk_training == q["stdk_validation"]["q50"]
            assert pure["params_used"] == metadata["model_config"]
            assert metadata["model_config"]["regression_type"] == "mean"
            assert metadata["model_config"]["use_ema"] is True
            assert stdk_training["validation_loss_name"] == "mse"
            assert stdk_training["validation_aggregation"] == "batch_mean"
            assert stdk_training["use_ema"] is True
            assert stdk_training["best_epoch"] >= 1

            assert config["stdk_backend"] == "spatial_adapter"
            assert config["stdk_checkpoint_source"] == "pure_stdk"
            assert config["stdk_q50_loss"] == "mse"
            assert config["stdk_use_ema"] is True
            assert config["prediction_mode"] == "direct"
            assert config["forecast_mode"] == "block5to5"
            assert config["lookback"] == config["horizon"] == 5
            assert config["smoke"] is False and config["validation_only"] is False
            assert config["q50_only"] is False
            assert config["qconv_training_target"] == "weather2k_observed_train500"
            assert config["qconv_validation_target"] == "weather2k_observed_train500_val150"
            assert config["heldout_q_source"] == "nearest_train500_station"
            assert config["q50_checkpoint_selection"] == "pinball"
            assert split["heldout_truth_used_for_training"] is False
            assert split["heldout_truth_role"] == "final_metrics_only"
            assert set(q["validation"]) == {"q05", "q50", "q95"}
            for source_name, config_name in PARAMETERS.items():
                assert config[config_name] == locked["formal_run_params"][source_name]

            expected_stations = train if station_key == "train_local" else held
            assert len(q["location_validation"]) == len(expected_stations)
            for record, station in zip(q["location_validation"], expected_stations):
                assert record["station_local"] == station
                assert record["training_source_station_local"] in train
                assert record["training_source_is_target"] == (scenario == "time_extrap_fixed500")
                assert set(record["validation"]) == {"q05", "q50", "q95"}
                for val in record["validation"].values():
                    assert val["checkpoint_selection_loss"] == "pinball"
                    assert val["training_windows"] > 0 and val["validation_windows"] > 0

            forecast = output / "forecasts" / (
                f"paired_stdk_truth_nag_qconvlstm_block5to5_{scenario}_seed{seed}_{forecast_suffix}"
            )
            expected_stations_set = set(expected_stations)
            expected_times_set = set(range(start, end))
            observed_keys = set()
            with forecast.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    assert int(row["seed"]) == seed
                    station = int(row["station_local"])
                    time_index = int(row["time_index"])
                    assert station in expected_stations_set and time_index in expected_times_set
                    key = (station, time_index)
                    assert key not in observed_keys
                    observed_keys.add(key)
                    truth = float(all_values[sampled[station], -1000 + time_index])
                    assert float(row["truth"]) == truth
            assert len(observed_keys) == len(expected_stations) * (end - start)

            cases.append({
                "seed": seed, "scenario": scenario, "target": target,
                "checkpoint_sha256": digest,
                "sampled_global_sha256": hashlib.sha256(json.dumps(sampled).encode()).hexdigest(),
                "train_local_sha256": hashlib.sha256(json.dumps(train).encode()).hexdigest(),
                "heldout_local_sha256": hashlib.sha256(json.dumps(held).encode()).hexdigest(),
                "obs_local_sha256": hashlib.sha256(json.dumps(obs).encode()).hexdigest(),
                "normalization_mean": expected_mean, "normalization_std": expected_std,
                "best_ema_epoch": stdk_training["best_epoch"],
                "best_ema_val_mse": stdk_training["best_validation_loss"],
                "test_station_count": len(expected_stations),
                "test_time_start_inclusive": start,
                "test_time_end_exclusive": end,
                "test_point_count": len(observed_keys),
                "max_abs_stage1_prediction_difference": prediction_by_pair[(seed, scenario)]["max_absolute_prediction_difference"],
                "all_checks_passed": True,
            })
            print(f"validated seed={seed} scenario={scenario} points={len(observed_keys)}", flush=True)

    report = {
        "case_count": len(cases),
        "test_point_count": sum(case["test_point_count"] for case in cases),
        "locked_q_params_sha256": sha(params),
        "tuning_seeds": locked["tuning_seeds"],
        "tuning_used_test_metrics": False,
        "note": "Seed 42 participated in Val150 hyperparameter selection; Test targets were not used.",
        "seed_station_ids": seed_splits,
        "cases": cases,
    }
    assert len(cases) == 15
    (HERE / "metrics" / "all_five_seed_protocol_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("case_count", "test_point_count", "locked_q_params_sha256")}, indent=2))


if __name__ == "__main__":
    main()
