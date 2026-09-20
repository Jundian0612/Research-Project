#!/usr/bin/env bash
set -uo pipefail

project_root=/home/user/Research-Project
run_root="$project_root/weather2k_exp/air_temperature/20260917_stdk_nag_truth_qconvlstm_tuning"
cd "$project_root" || exit 2

source_files=(
  weather2k_exp/2K_STDK_QConvLSTM.py
  weather2k_exp/2K_STDK_500train_100test.py
  weather2k_exp/tune_stdk_qconvlstm_hyperparams.py
  STDK_QConvLSTM_reproduction_results/code/STDK_QConvLSTM_reproduction.py
  spatial-adapter/examples/baselines/stdk/st_interp.py
  spatial-adapter/examples/baselines/stdk/trainer.py
)
if [[ -f "$run_root/code.sha256" ]]; then
  sha256sum -c "$run_root/code.sha256" || exit 3
else
  sha256sum "${source_files[@]}" > "$run_root/code.sha256" || exit 3
  git rev-parse HEAD > "$run_root/git_head.txt"
fi

"$project_root/.venv-wsl/bin/python" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"' || exit 4
export PYTHONUNBUFFERED=1
printf 'Started: %s\nSeeds: 41 42\nStage: all (9 interface + 8 capacity trials)\n' "$(date -Is)" | tee -a "$run_root/run.log"
"$project_root/.venv-wsl/bin/python" -u weather2k_exp/tune_stdk_qconvlstm_hyperparams.py \
  --stage all --seeds 41 42 --output-dir "$run_root" 2>&1 | tee -a "$run_root/run.log"
run_status=${PIPESTATUS[0]}
printf '%s\n' "$run_status" > "$run_root/exit_code"
printf 'Finished: %s; exit code: %s\n' "$(date -Is)" "$run_status" | tee -a "$run_root/run.log"
exit "$run_status"
