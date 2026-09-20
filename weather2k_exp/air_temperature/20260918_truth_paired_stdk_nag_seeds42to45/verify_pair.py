#!/usr/bin/env python3
"""Verify one formal pure-STDK / STDK+Q pair before accepting its test score."""
import hashlib
import json
import math
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
seed = int(sys.argv[1])
scenario = sys.argv[2]
assert seed in (42, 43, 44, 45)
target, suffix, target_count = {
    "time_extrap_fixed500": ("Target_Time150", "time500", 500),
    "space_extrap_fixed850": ("Target_Space100", "space850", 100),
    "spatiotemp_100x150": ("Target_ST100x150", "st_100x150", 100),
}[scenario]
output = root / f"seed{seed}"
checkpoint = output / "checkpoints" / f"{scenario}_seed{seed}" / "model_best.pt"
digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
metadata = json.loads(checkpoint.with_name("metadata.json").read_text())
pure = json.loads(
    (output / f"2K_stdk_metrics_{suffix}_train500_test100.json").read_text()
)["seed_runs"][0]["payload"]
paired = json.loads(
    (output / f"paired_stdk_truth_nag_qconvlstm_block5to5_{scenario}_seed{seed}.json").read_text()
)
locked = json.loads((root / "locked_q_params.json").read_text())
assert locked["selection_uses_test_metrics"] is False
assert locked["selection_uses_heldout100_truth"] is False
assert pure["seed"] == paired["config"]["seed"] == metadata["seed"] == seed
assert metadata["scenario"] == paired["config"]["scenario"] == scenario
assert checkpoint.resolve() == Path(pure["stdk_checkpoint"]["path"])
assert checkpoint.resolve() == Path(paired["stdk_checkpoint"]["path"])
assert digest == metadata["checkpoint_sha256"]
assert digest == pure["stdk_checkpoint"]["sha256"]
assert digest == paired["stdk_checkpoint"]["sha256"]
assert paired["fitted_stdk_baseline"]["refitted"] is False
assert math.isclose(
    pure["results"][target]["RMSE"],
    paired["fitted_stdk_baseline"]["results"][target]["RMSE"], abs_tol=1e-4,
)
assert math.isclose(
    metadata["target_metrics"][target]["RMSE"],
    pure["results"][target]["RMSE"], abs_tol=1e-4,
)
config = paired["config"]
assert config["validation_only"] is False and config["q50_only"] is False
assert config["smoke"] is False
assert config["forecast_mode"] == "block5to5"
assert config["prediction_mode"] == "direct"
assert config["train_times"] == 700 and config["val_times"] == config["test_times"] == 150
assert config["n_train"] == 500 and config["n_heldout"] == 100
parameter_names = {
    "QCONV_LR": "qconv_lr", "QCONV_BATCH_SIZE": "qconv_batch",
    "QCONV_EPOCHS": "qconv_epochs", "QCONV_PATIENCE": "qconv_patience",
    "CONV_FILTERS": "conv_filters", "QCONV_WEIGHT_DECAY": "qconv_weight_decay",
    "GRID_SIZE": "grid_size", "NEIGHBOURHOOD_RADIUS": "radius",
}
for source_name, config_name in parameter_names.items():
    assert config[config_name] == locked["formal_run_params"][source_name], source_name
assert set(paired["validation"]) == {"q05", "q50", "q95"}
assert len(paired["location_validation"]) == target_count
train = set(paired["split"]["train_local"])
held = set(paired["split"]["heldout_local"])
assert not (train & held)
assert paired["split"]["heldout_truth_used_for_training"] is False
assert paired["split"]["heldout_truth_role"] == "final_metrics_only"
for location in paired["location_validation"]:
    assert location["training_source_station_local"] in train
    if scenario == "time_extrap_fixed500":
        assert location["training_source_is_target"]
    else:
        assert location["station_local"] in held
        assert not location["training_source_is_target"]
    assert set(location["validation"]) == {"q05", "q50", "q95"}

result = {
    "scenario": scenario,
    "seed": seed,
    "target": target,
    "checkpoint_sha256": digest,
    "pure_stdk_rmse": pure["results"][target]["RMSE"],
    "paired_fitted_stdk_rmse": paired["fitted_stdk_baseline"]["results"][target]["RMSE"],
    "qconv_rmse": paired["results"][target]["RMSE"],
    "q_quantiles": sorted(paired["validation"]),
    "target_location_count": target_count,
    "heldout_truth_used_for_training": False,
}
(output / f"verified_{scenario}.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
