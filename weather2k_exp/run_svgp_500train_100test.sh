#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

SPATIAL_ENV_NAME="${SPATIAL_ENV_NAME:-geospatial-neural-adapter}"
PYTHON_BIN="${PYTHON_BIN:-/home/jundian/installers/yes/envs/${SPATIAL_ENV_NAME}/bin/python}"

export EXPERIMENT_SCENARIO="${EXPERIMENT_SCENARIO:-spatiotemp_100x150}"
export N_SAMPLE_TARGET="${N_SAMPLE_TARGET:-600}"
export N_TRAIN_TARGET="${N_TRAIN_TARGET:-100}"
export N_UNKNOWN_PRIMARY_TARGET="${N_UNKNOWN_PRIMARY_TARGET:-400}"
export N_UNKNOWN_EVAL_TARGET="${N_UNKNOWN_EVAL_TARGET:-100}"
export N_LAST="${N_LAST:-1000}"
export TIME_TRAIN_LEN="${TIME_TRAIN_LEN:-700}"
export TIME_VAL_LEN="${TIME_VAL_LEN:-150}"
export TIME_TEST_LEN="${TIME_TEST_LEN:-150}"
export SEED_LIST="${SEED_LIST:-[41, 42, 43, 44, 45]}"
export RESULT_SUFFIX="${RESULT_SUFFIX:-500train_100test}"
# Validation-selected formal-run defaults from svgp_tuning_current.
export SVGP_EPOCHS="${SVGP_EPOCHS:-500}"
export SVGP_PATIENCE="${SVGP_PATIENCE:-30}"
export SVGP_NUM_INDUCING="${SVGP_NUM_INDUCING:-1024}"
export SVGP_KERNEL="${SVGP_KERNEL:-matern_periodic}"
export SVGP_LR="${SVGP_LR:-0.001}"
export SVGP_VARIATIONAL_JITTER="${SVGP_VARIATIONAL_JITTER:-0.01}"

if [[ ! -f "$SCRIPT_DIR/2K_SVGP_500train_100test.py" ]]; then
  echo "[Error] Missing script: $SCRIPT_DIR/2K_SVGP_500train_100test.py"
  exit 1
fi

echo "[SVGP] running Weather2K experiment from: $SCRIPT_DIR/2K_SVGP_500train_100test.py"
echo "[SVGP] conda env: $SPATIAL_ENV_NAME"
echo "[SVGP] python: $PYTHON_BIN"
echo "[SVGP] dataset override: ${WEATHER2K_NPY:-<default search paths>}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[Error] Python interpreter not found or not executable: $PYTHON_BIN"
  exit 1
fi

"$PYTHON_BIN" "$SCRIPT_DIR/2K_SVGP_500train_100test.py" "$@"
