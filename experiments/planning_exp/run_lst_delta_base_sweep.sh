#!/usr/bin/env bash
set -euo pipefail

# Run the current five planning test cities with several LST cooling targets.
# Each run collects both joint and cooling-only outputs so Fig4 can compare them.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${CONFIG_PATH:-experiments/planning_exp/config.yaml}"
DATA_OVERRIDE="${DATA_OVERRIDE:-data=eump_full/server_v2}"

CITIES="${CITIES:-782,339,216,475,261}"
YEAR="${YEAR:-2020}"
OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/planning_exp/outputs/lst_delta_base_joint_sweep_5cities}"

TARGETS=("1.01" "1.00" "0.99")

RUN_ARGS=()
if [[ -n "${EPOCHS:-}" ]]; then
  RUN_ARGS+=(--epochs "${EPOCHS}")
fi
if [[ -n "${LR:-}" ]]; then
  RUN_ARGS+=(--lr "${LR}")
fi
if [[ -n "${DEVICE:-}" ]]; then
  RUN_ARGS+=(--device "${DEVICE}")
fi
if [[ -n "${NUM_WORKERS:-}" ]]; then
  RUN_ARGS+=(--num-workers "${NUM_WORKERS}")
fi

for target in "${TARGETS[@]}"; do
  label="${target/./p}"
  output_dir="${OUTPUT_ROOT}/lst_delta_${label}"

  echo
  echo "=== planning_exp both: lst_delta_base=${target}, cities=${CITIES}, output=${output_dir} ==="
  "${PYTHON_BIN}" experiments/planning_exp/run.py \
    --config-path "${CONFIG_PATH}" \
    --mode both \
    --cities "${CITIES}" \
    --year "${YEAR}" \
    --lst-delta-base "${target}" \
    --output-dir "${output_dir}" \
    "${RUN_ARGS[@]}" \
    "${DATA_OVERRIDE}" \
    "$@"

  echo
  echo "=== fig4: ${output_dir} ==="
  "${PYTHON_BIN}" experiments/planning_exp/visualize_fig4.py \
    --input-dir "${output_dir}" \
    --cities "${CITIES}" \
    --output "${output_dir}/figures/fig4.png"
done

echo "=== summarize city x target table ==="
"${PYTHON_BIN}" experiments/planning_exp/summarize_lst_delta_base_sweep.py \
  --input-root "${OUTPUT_ROOT}" \
  --targets "$(IFS=,; echo "${TARGETS[*]}")" \
  --cities "${CITIES}"

echo
echo "All outputs written under: ${OUTPUT_ROOT}"
