#!/usr/bin/env bash
set -euo pipefail

# Channel-drop robustness eval suite (EVAL-ONLY, no training).
#
# Scores two arms under the SAME inference-time channel-drop pipeline:
#   1. drop-trained arm -- models from train.trained_group (already trained)
#   2. clean arm        -- mainline eump_full_mae_v3 models, scored under drop
#
# lst/precip eval in-process; pm25/traffic run CV (launcher eval-only for the
# drop arm, pretrained embedding-head eval for the clean arm). Results land in
# outputs/eump_mae_v3_channel_drop/<eval_group>/ (separate subdir per arm).
#
# Prerequisites (all verified present on the server):
#   - drop-trained: model/eump_full_mae_v3_channel_drop_trained/{lst,precip,
#                   pm25_cv/*,traffic_cv/*}
#   - clean:        model/eump_full_mae_v3/{lst,precip,pm25_cv/*,traffic_cv/*}
#
# Usage:
#   GPU=1,2,3 bash experiments/eump_channel_drop/run_channel_drop_suite.sh    # NPROC auto = 3
#   GPU=1 bash experiments/eump_channel_drop/run_channel_drop_suite.sh        # single GPU
#   DRY_RUN=1 bash experiments/eump_channel_drop/run_channel_drop_suite.sh   # print only
#   TASKS=lst,pm25 bash .../run_channel_drop_suite.sh                        # subset
# NOTE: NPROC must equal the GPU count; the script derives it from GPU and errors
#       on mismatch (avoids torchrun "invalid device ordinal").

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_PY="experiments/eump_channel_drop/run.py"

# GPU drives both the CV subprocess (GPU/NPROC) and the in-process lst/precip
# eval worker (EVAL_GPU/EVAL_NPROC). Keep them aligned to one device by default.
export GPU="${GPU:-0}"

# NPROC must equal the number of GPUs in GPU, otherwise torchrun spawns ranks
# that set_device() a non-visible ordinal -> "CUDA error: invalid device
# ordinal". Derive it from GPU by default; if the caller sets NPROC explicitly,
# validate it matches.
_GPU_COUNT="$(awk -F, '{n=0; for(i=1;i<=NF;i++) if($i!="") n++; print n}' <<< "${GPU}")"
export NPROC="${NPROC:-${_GPU_COUNT}}"
if [[ "${NPROC}" != "${_GPU_COUNT}" ]]; then
  echo "ERROR: NPROC=${NPROC} but GPU='${GPU}' has ${_GPU_COUNT} device(s)." >&2
  echo "       Set NPROC to ${_GPU_COUNT}, or widen GPU (e.g. GPU=1,2,3 NPROC=3)." >&2
  exit 1
fi

# lst/precip eval runs single-process; pin it to the first GPU in the list.
_FIRST_GPU="${GPU%%,*}"
export EVAL_GPU="${EVAL_GPU:-${_FIRST_GPU}}"
export EVAL_NPROC="${EVAL_NPROC:-1}"
export LOG_PROGRESS_BAR="${LOG_PROGRESS_BAR:-0}"
export EUMP_TORCH_SHARING_STRATEGY="${EUMP_TORCH_SHARING_STRATEGY:-file_system}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export DRY_RUN="${DRY_RUN:-0}"

TASKS_ARG=()
if [[ -n "${TASKS:-}" ]]; then
  TASKS_ARG=(--tasks "${TASKS}")
fi

echo "=========================================================="
echo " Channel-drop eval suite (eval-only)"
echo " GPU=${GPU} NPROC=${NPROC} EVAL_GPU=${EVAL_GPU} DRY_RUN=${DRY_RUN}"
echo " tasks=${TASKS:-all}"
echo "=========================================================="

echo ""
echo "=== Arm 1/2: drop-trained models under channel drop ==="
"${PYTHON_BIN}" "${RUN_PY}" "${TASKS_ARG[@]}"

echo ""
echo "=== Arm 2/2: clean mae_v3 models under channel drop ==="
"${PYTHON_BIN}" "${RUN_PY}" "${TASKS_ARG[@]}" --trained-group clean

echo ""
echo "Done. Results per arm:"
echo "  drop-trained: experiments/eump_channel_drop/outputs/eump_mae_v3_channel_drop/eump_full_mae_v3_channel_drop_trained/"
echo "  clean:        experiments/eump_channel_drop/outputs/eump_mae_v3_channel_drop/eump_full_mae_v3/"
