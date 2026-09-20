#!/usr/bin/env bash
set -Eeuo pipefail

project_root=/home/user/Research-Project
run_root="$project_root/weather2k_exp/air_temperature/20260918_truth_paired_stdk_nag_seeds42to45"
python_bin="$project_root/.venv-wsl/bin/python"
cd "$project_root"

on_exit() {
  status=$?
  printf '%s\n' "$status" > "$run_root/exit_code"
  printf 'Finished: %s; exit code: %s\n' "$(date -Is)" "$status" | tee -a "$run_root/run.log"
}
trap on_exit EXIT

printf 'Started: %s\nSeeds: 42, 43, 44, 45\n' "$(date -Is)" | tee -a "$run_root/run.log"
sha256sum -c "$run_root/code.sha256" | tee -a "$run_root/run.log"
sha256sum -c "$run_root/params.sha256" | tee -a "$run_root/run.log"
"$python_bin" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"'

for seed in 42 43 44 45; do
  seed_root="$run_root/seed$seed"
  mkdir -p "$seed_root"
  for scenario in spatiotemp_100x150 space_extrap_fixed850 time_extrap_fixed500; do
    sha256sum -c "$run_root/code.sha256" > /dev/null
    sha256sum -c "$run_root/params.sha256" > /dev/null
    if [[ -f "$seed_root/verified_${scenario}.json" ]]; then
      "$python_bin" "$run_root/verify_pair.py" "$seed" "$scenario" | tee -a "$run_root/run.log"
      printf 'Reused verified seed=%s scenario=%s\n' "$seed" "$scenario" | tee -a "$run_root/run.log"
      continue
    fi
    printf 'Starting seed=%s scenario=%s at %s\n' "$seed" "$scenario" "$(date -Is)" | tee -a "$run_root/run.log"
    env RUN_SVGP=0 RUN_DLINEAR=0 RUN_STDK=1 RUN_QCONVLSTM=1 \
      "SEED_LIST=[$seed]" QCONVLSTM_SEEDS="$seed" SCENARIO_FILTER="$scenario" \
      WEATHER2K_NPY="$project_root/Josh's Weather2K/Weather2K/weather2k.npy" \
      WEATHER2K_OUTPUT_DIR="$seed_root" QCONVLSTM_OUTPUT_DIR="$seed_root" \
      QCONVLSTM_PARAMS_FILE="$run_root/locked_q_params.json" \
      QCONVLSTM_FORECAST_MODE=block5to5 QCONVLSTM_PREDICTION_MODE=direct \
      QCONVLSTM_SMOKE_TEST=0 QCONV_STDK_EPOCHS=350 STDK_EPOCHS=350 \
      QCONV_EPOCHS=25 QCONV_BATCH_SIZE=5 QCONV_FILTERS=32 \
      TIME_STRIDE=1 SPACE_SPLITS=100/400/100 PYTHONUNBUFFERED=1 \
      "$python_bin" -u weather2k_exp/experiments_runner.py 2>&1 | tee -a "$run_root/run.log"
    sha256sum -c "$run_root/code.sha256" > /dev/null
    sha256sum -c "$run_root/params.sha256" > /dev/null
    "$python_bin" "$run_root/verify_pair.py" "$seed" "$scenario" | tee -a "$run_root/run.log"
  done
done

"$python_bin" "$run_root/summarize_five_seeds.py" | tee -a "$run_root/run.log"
