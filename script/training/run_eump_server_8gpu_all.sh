#!/usr/bin/env bash
set -euo pipefail

# Server-oriented hardware wrapper around script/training/run_eump_full_train.sh.
# Training, model, data, and checkpoint defaults live in Hydra configs.
#
# Examples:
#   bash script/training/run_eump_server_8gpu_all.sh
#   bash script/training/run_eump_server_8gpu_all.sh pretrain
#   bash script/training/run_eump_server_8gpu_all.sh lst downstream.epochs=500 downstream.resume=true
#   DRY_RUN=1 bash script/training/run_eump_server_8gpu_all.sh launch=eump_full/mae_v3

if (($# > 0)) && [[ "$1" != *"="* ]]; then
  MODE="$1"
  shift
else
  MODE=""
fi

export GPU="${GPU:-0,1,2,3,4,5,6,7}"
export NPROC="${NPROC:-8}"
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export EUMP_TORCH_SHARING_STRATEGY="${EUMP_TORCH_SHARING_STRATEGY:-file_system}"
export LOG_PROGRESS_BAR="${LOG_PROGRESS_BAR:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -n "${MODE}" ]]; then
  exec bash script/training/run_eump_full_train.sh "${MODE}" "$@"
fi

exec bash script/training/run_eump_full_train.sh "$@"
