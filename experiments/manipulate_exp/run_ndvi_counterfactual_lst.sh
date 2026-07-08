#!/usr/bin/env bash
set -euo pipefail

# Predict mae_v3 LST maps for six counterfactual NDVI scenarios:
# ndvi0, urban +0.1, buffer +0.1, and urban+buffer +0.1 for 5 km / 10 km rings.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-0}"
NPROC="${NPROC:-1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-0}"
YEAR="${YEAR:-2020}"
INPUT_NPZ="${INPUT_NPZ:-experiments/manipulate_exp/ndvi_counterfactual_scenarios.npz}"
META_CSV="${META_CSV:-}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/manipulate_exp/outputs/ndvi_counterfactual_lst_${YEAR}}"
SCENARIOS="${SCENARIOS:-ndvi0,ndvi_urban,ndvi_buffer_5km,ndvi_buffer_10km,ndvi_both_5km,ndvi_both_10km}"
DATA_OVERRIDE="${DATA_OVERRIDE:-data=eump_full/server_v2}"
PIPELINE_OVERRIDE="${PIPELINE_OVERRIDE:-pipeline=eump_full/mae_v3/lst}"

COMMON_ARGS=(
  experiments/manipulate_exp/run_custom_ndvi.py
  --input-npz "${INPUT_NPZ}"
  --year "${YEAR}"
  --source NDVI
  --scenarios "${SCENARIOS}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --output-dir "${OUTPUT_DIR}"
)

if [[ -n "${META_CSV}" ]]; then
  COMMON_ARGS+=(--meta-csv "${META_CSV}")
fi
COMMON_ARGS+=("${DATA_OVERRIDE}" "${PIPELINE_OVERRIDE}")

if [[ "${NPROC}" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" "${COMMON_ARGS[@]}" "$@"
else
  CUDA_VISIBLE_DEVICES="${GPU}" torchrun --nproc_per_node="${NPROC}" "${COMMON_ARGS[@]}" "$@"
fi
