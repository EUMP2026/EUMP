#!/usr/bin/env bash
set -euo pipefail

# Run fixed-test Traffic cross validation for the current EUMP Swin suite.
#
# Defaults:
#   - suite: mae_v3
#   - model: current Swin pooled-embedding Traffic head
#   - folds: 5-fold train+val CV with the original test split fixed
#   - data: eump_full/server_v2
#
# Examples:
#   bash script/experiments/run_current_traffic_cv.sh
#   bash script/experiments/run_current_traffic_cv.sh --dry-run --folds 0
#   bash script/experiments/run_current_traffic_cv.sh --suite mae_v2 --models all
#   bash script/experiments/run_current_traffic_cv.sh --models baselines
#   bash script/experiments/run_current_traffic_cv.sh --prepare-embeddings-only

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

usage() {
  cat <<'EOF'
Usage:
  bash script/experiments/run_current_traffic_cv.sh [options] [hydra_overrides...]

Options:
  --suite NAME                 EUMP suite name under configs/pipeline/eump_full. Default: mae_v3
  --models LIST                current|swin|baselines|all|comma-separated model names.
                               Default: current
  --folds LIST                 Comma-separated fold ids, e.g. 0 or 0,1. Default: all folds
  --data NAME                  Hydra data config. Default: eump_full/server_v2
  --seed INT                   Fold seed. Default: 42
  --run-name NAME              Output run name under log/eump_traffic_cv. Default: traffic_cv_<suite>_seed<seed>
  --experiment-group NAME      Experiment group for logs/models. Default: eump_traffic_cv
  --prepare-embeddings-only    Only export/reuse Swin pooled embeddings.
  --dry-run                    Print commands without running heavy jobs.
  -h, --help                   Show this help.

Useful env vars:
  GPU=0,1,2,3,4,5,6,7
  NPROC=8
  LOG_PROGRESS_BAR=0
  PYTHON_BIN=python
  CHECKPOINT_FILE=eump_reconstruction_best_checkpoint.pt

Outputs:
  log/<experiment-group>/<run-name>/commands.log
  log/<experiment-group>/<run-name>/folds.json
  log/<experiment-group>/<run-name>/traffic_cv_results.jsonl
  log/<experiment-group>/<run-name>/traffic_cv_fold_metrics.csv
  log/<experiment-group>/<run-name>/traffic_cv_summary.csv
EOF
}

SUITE="${SUITE:-mae_v3}"
MODELS="${MODELS:-current}"
FOLDS="${FOLDS:-}"
DATA_CONFIG="${DATA_CONFIG:-eump_full/server_v2}"
FOLD_SEED="${FOLD_SEED:-42}"
EXPERIMENT_GROUP="${EXPERIMENT_GROUP:-eump_traffic_cv}"
RUN_NAME="${RUN_NAME:-}"
PREPARE_EMBEDDINGS_ONLY="${PREPARE_EMBEDDINGS_ONLY:-0}"
DRY_RUN="${DRY_RUN:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

EXTRA_OVERRIDES=()
while (($# > 0)); do
  case "$1" in
    --suite)
      SUITE="$2"
      shift 2
      ;;
    --models)
      MODELS="$2"
      shift 2
      ;;
    --folds)
      FOLDS="$2"
      shift 2
      ;;
    --data)
      DATA_CONFIG="$2"
      shift 2
      ;;
    --seed)
      FOLD_SEED="$2"
      shift 2
      ;;
    --run-name)
      RUN_NAME="$2"
      shift 2
      ;;
    --experiment-group)
      EXPERIMENT_GROUP="$2"
      shift 2
      ;;
    --prepare-embeddings-only)
      PREPARE_EMBEDDINGS_ONLY=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      EXTRA_OVERRIDES+=("$1")
      shift
      ;;
  esac
done

case "${MODELS}" in
  current|swin)
    SELECTED_MODELS="swin_embed_head_128_d05"
    ;;
  baselines)
    SELECTED_MODELS="scalar_mlp_128_d05,aef_mlp_128_d05"
    ;;
  all)
    SELECTED_MODELS=""
    ;;
  *)
    SELECTED_MODELS="${MODELS}"
    ;;
esac

RUN_NAME="${RUN_NAME:-traffic_cv_${SUITE}_seed${FOLD_SEED}}"
LOG_ROOT="${REPO_ROOT}/log/${EXPERIMENT_GROUP}/${RUN_NAME}"
RUNTIME_CONFIG="${LOG_ROOT}/traffic_cv_runtime.yaml"
mkdir -p "${LOG_ROOT}"

GPU="${GPU:-0,1,2,3,4,5,6,7}"
NPROC="${NPROC:-8}"
LOG_PROGRESS_BAR="${LOG_PROGRESS_BAR:-0}"
EUMP_TORCH_SHARING_STRATEGY="${EUMP_TORCH_SHARING_STRATEGY:-file_system}"
CHECKPOINT_FILE="${CHECKPOINT_FILE:-eump_reconstruction_best_checkpoint.pt}"
ENCODER_MODEL_NAME="${ENCODER_MODEL_NAME:-eump_full_${SUITE}/pretrain}"
TARGET_PIPELINE="${TARGET_PIPELINE:-eump_full/${SUITE}/traffic}"
EMBEDDING_PIPELINE="${EMBEDDING_PIPELINE:-eump_full/${SUITE}/pretrain}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-128}"
HEAD_BATCH_SIZE="${HEAD_BATCH_SIZE:-128}"
HEAD_EPOCHS="${HEAD_EPOCHS:-100}"
HEAD_LR="${HEAD_LR:-5.0e-4}"
HEAD_DROPOUT="${HEAD_DROPOUT:-0.5}"
HEAD_HIDDEN_DIM="${HEAD_HIDDEN_DIM:-128}"

cat > "${RUNTIME_CONFIG}" <<EOF
traffic_cv:
  folds: 5
  fold_seed: ${FOLD_SEED}
  experiment_group: ${EXPERIMENT_GROUP}
  run_name: ${RUN_NAME}
  fold_manifest_dir: null
  embedding_cache_dir: null
  target_pipeline: ${TARGET_PIPELINE}
  embedding:
    model_name: ${ENCODER_MODEL_NAME}
    checkpoint_file: ${CHECKPOINT_FILE}
    checkpoint_source: auto
    pipeline: ${EMBEDDING_PIPELINE}
    years:
      - 2017
      - 2020
    batch_size: ${EMBEDDING_BATCH_SIZE}
    workers_per_gpu: 4
    torch_threads: 1
    overwrite: false
  models:
    - name: swin_embed_head_128_d05
      kind: swin_embedding_head
      scalar_hidden_dim: ${HEAD_HIDDEN_DIM}
      dropout: ${HEAD_DROPOUT}
      lr: ${HEAD_LR}
      weight_decay: 1.0e-2
      batch_size: ${HEAD_BATCH_SIZE}
      epochs: ${HEAD_EPOCHS}
      loss: mae
      selection_metric: val_loss
    - name: scalar_mlp_128_d05
      kind: launcher
      mode: mlp-traffic
      launch: eump_full/scalar_mlp
      overrides:
        - model.mlp_hidden_dims=[128]
        - model.dropout=0.5
        - downstream.feature_standardization=feature_cache_train_split
    - name: aef_mlp_128_d05
      kind: launcher
      mode: aef-mlp-traffic
      launch: eump_full/aef_mlp
      overrides:
        - model.mlp_hidden_dims=[128]
        - model.dropout=0.5
        - downstream.feature_standardization=none
EOF

CMD=(
  "${PYTHON_BIN}"
  script/experiments/run_city_scalar_cv.py
  --config
  "${RUNTIME_CONFIG}"
)

if [[ -n "${SELECTED_MODELS}" ]]; then
  CMD+=(--models "${SELECTED_MODELS}")
fi
if [[ -n "${FOLDS}" ]]; then
  CMD+=(--folds "${FOLDS}")
fi
if [[ "${PREPARE_EMBEDDINGS_ONLY}" == "1" ]]; then
  CMD+=(--prepare-embeddings-only)
fi

HAS_DATA_OVERRIDE=0
for override in "${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"}"; do
  case "${override}" in
    data=*|+data=*|++data=*)
      HAS_DATA_OVERRIDE=1
      ;;
  esac
done
if [[ "${HAS_DATA_OVERRIDE}" == "0" ]]; then
  CMD+=("data=${DATA_CONFIG}")
fi
if ((${#EXTRA_OVERRIDES[@]})); then
  CMD+=("${EXTRA_OVERRIDES[@]}")
fi

export GPU NPROC LOG_PROGRESS_BAR DRY_RUN EUMP_TORCH_SHARING_STRATEGY
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

printf '+'
printf ' %q' \
  "GPU=${GPU}" \
  "NPROC=${NPROC}" \
  "LOG_PROGRESS_BAR=${LOG_PROGRESS_BAR}" \
  "DRY_RUN=${DRY_RUN}" \
  "${CMD[@]}"
printf '\n'

"${CMD[@]}"
