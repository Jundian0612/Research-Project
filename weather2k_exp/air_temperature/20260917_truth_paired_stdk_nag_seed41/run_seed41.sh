#!/usr/bin/env bash
set -Eeuo pipefail

project_root=/home/user/Research-Project
run_root="$project_root/weather2k_exp/air_temperature/20260917_truth_paired_stdk_nag_seed41"
python_bin="$project_root/.venv-wsl/bin/python"
cd "$project_root"

on_exit() {
  status=$?
  printf '%s\n' "$status" > "$run_root/exit_code"
  printf 'Finished: %s; exit code: %s\n' "$(date -Is)" "$status" | tee -a "$run_root/run.log"
}
trap on_exit EXIT

printf 'Started: %s\nSeed: 41\nScenarios: ST100x150, Space100, Time150\n' "$(date -Is)" | tee -a "$run_root/run.log"
sha256sum -c "$run_root/code.sha256" | tee -a "$run_root/run.log"
sha256sum -c "$run_root/params.sha256" | tee -a "$run_root/run.log"
"$python_bin" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"'

for scenario in spatiotemp_100x150 space_extrap_fixed850 time_extrap_fixed500; do
  sha256sum -c "$run_root/code.sha256" > /dev/null
  sha256sum -c "$run_root/params.sha256" > /dev/null
  if [[ -f "$run_root/verified_${scenario}.json" ]]; then
    "$python_bin" "$run_root/verify_seed41.py" "$scenario" | tee -a "$run_root/run.log"
    printf 'Reused verified scenario: %s\n' "$scenario" | tee -a "$run_root/run.log"
    continue
  fi
  printf 'Starting scenario: %s at %s\n' "$scenario" "$(date -Is)" | tee -a "$run_root/run.log"
  env RUN_SVGP=0 RUN_DLINEAR=0 RUN_STDK=1 RUN_QCONVLSTM=1 \
    'SEED_LIST=[41]' SCENARIO_FILTER="$scenario" \
    WEATHER2K_OUTPUT_DIR="$run_root" QCONVLSTM_OUTPUT_DIR="$run_root" \
    QCONVLSTM_PARAMS_FILE="$run_root/locked_q_params.json" PYTHONUNBUFFERED=1 \
    "$python_bin" -u weather2k_exp/experiments_runner.py 2>&1 | tee -a "$run_root/run.log"
  sha256sum -c "$run_root/code.sha256" > /dev/null
  sha256sum -c "$run_root/params.sha256" > /dev/null
  "$python_bin" "$run_root/verify_seed41.py" "$scenario" | tee -a "$run_root/run.log"
done
