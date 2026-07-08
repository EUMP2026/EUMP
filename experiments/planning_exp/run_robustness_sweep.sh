#!/usr/bin/env bash
set -euo pipefail

# Cooling-only robustness sweep over urbanfrac large cities.
#
# Validates optimization robustness along three independent axes, each varying
# one parameter from the baseline (lst_delta_base=1.00, lr=0.001, epochs=1000):
#   - LST target : 1.00, 1.01, 0.99
#   - learning rate : 0.001 (base), 0.0005 (half), 0.002 (double)
#   - steps : 1000 (base), 2000
# The baseline is shared by all three axes, so the 6 unique runs are:
#   1 lst=1.00 lr=0.001  steps=1000   (baseline)
#   2 lst=1.01 lr=0.001  steps=1000
#   3 lst=0.99 lr=0.001  steps=1000
#   4 lst=1.00 lr=0.0005 steps=1000
#   5 lst=1.00 lr=0.002  steps=1000
#   6 lst=1.00 lr=0.001  steps=2000
#
# Each run is a full multi-GPU urbanfrac statistics pass that writes its own
# planning_data.pkl under a per-config output directory.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
MODE="cooling_only"
CITY_CSV="${CITY_CSV:-experiments/planning_exp/urbanfrac_cities_ge0.1.csv}"
CONFIG_PATH="${CONFIG_PATH:-experiments/planning_exp/config.yaml}"
DATA_OVERRIDE="${DATA_OVERRIDE:-data=eump_full/server_v2}"
YEAR="${YEAR:-2020}"
NUM_WORKERS="${NUM_WORKERS:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/planning_exp/outputs/robustness_sweep_cooling_only}"
LIMIT="${LIMIT:-}"  # optional: cap city count for a quick smoke test

# Each entry: label|lst_delta_base|lr|epochs
CONFIGS=(
  "base_lst1p00_lr1e-3_s1000|1.00|0.001|1000"
  "lst1p01_lr1e-3_s1000|1.01|0.001|1000"
  "lst0p99_lr1e-3_s1000|0.99|0.001|1000"
  "lst1p00_lr5e-4_s1000|1.00|0.0005|1000"
  "lst1p00_lr2e-3_s1000|1.00|0.002|1000"
  "lst1p00_lr1e-3_s2000|1.00|0.001|2000"
)

LIMIT_ARGS=()
if [[ -n "${LIMIT}" ]]; then
  LIMIT_ARGS+=(--limit "${LIMIT}")
fi

echo "Robustness sweep: ${#CONFIGS[@]} configs, mode=${MODE}, gpus=${GPUS}"
echo "Output root: ${OUTPUT_ROOT}"

for entry in "${CONFIGS[@]}"; do
  IFS='|' read -r label lst lr epochs <<< "${entry}"
  output_dir="${OUTPUT_ROOT}/${label}"

  echo
  echo "=== ${label}: lst_delta_base=${lst}, lr=${lr}, epochs=${epochs} ==="
  echo "=== output: ${output_dir} ==="

  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 \
  "${PYTHON_BIN}" experiments/planning_exp/run_urbanfrac_city_stats.py \
    --city-csv "${CITY_CSV}" \
    --config-path "${CONFIG_PATH}" \
    --mode "${MODE}" \
    --gpus "${GPUS}" \
    --year "${YEAR}" \
    --num-workers "${NUM_WORKERS}" \
    --epochs "${epochs}" \
    --lr "${lr}" \
    --output-dir "${output_dir}" \
    "${LIMIT_ARGS[@]}" \
    --data-override "${DATA_OVERRIDE}" \
    --lst-delta-base "${lst}"
done

echo
echo "All ${#CONFIGS[@]} configs done. Per-config planning_data.pkl under: ${OUTPUT_ROOT}"
