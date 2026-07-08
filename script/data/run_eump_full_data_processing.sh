#!/usr/bin/env bash
set -euo pipefail

# Complete full-server EUMP data processing launcher.
#
# Run from any directory after activating the target environment:
#   conda activate EUMP
#
# Default stage is `full`:
#   bash script/data/run_eump_full_data_processing.sh
#
# Run one stage:
#   bash script/data/run_eump_full_data_processing.sh preprocess
#   bash script/data/run_eump_full_data_processing.sh report
#
# Run multiple stages in order:
#   bash script/data/run_eump_full_data_processing.sh split,preprocess,report
#
# Common server overrides:
#   OVERWRITE=1 WORKERS=32 bash script/data/run_eump_full_data_processing.sh full
#   EUMP_ROOT=data/processed_v1 bash script/data/run_eump_full_data_processing.sh split,preprocess,report

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

STAGE="${1:-full}"
PYTHON_BIN="${PYTHON:-python}"

RAW_ROOT="${RAW_ROOT:-data/raw_data}"
CSV_ROOT="${CSV_ROOT:-data/csv_data}"
EUMP_ROOT="${EUMP_ROOT:-data/processed_v1}"
BH8_ROOT="${BH8_ROOT:-${RAW_ROOT}/BH8_v1}"

WORKERS="${WORKERS:-0}"
PREPROCESS_WORKERS="${PREPROCESS_WORKERS:-${WORKERS}}"
OVERWRITE="${OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"
EXPORT_SKIP_STATS="${EXPORT_SKIP_STATS:-0}"
VALIDATE_BH8_PADDING="${VALIDATE_BH8_PADDING:-0}"

PREPROCESSING_PATH="${PREPROCESSING_PATH:-${EUMP_ROOT}/manifest/stats/preprocessing_v1.json}"
CITY_CROP_SIZE="${CITY_CROP_SIZE:-256}"
CITY_CROP_SAMPLES_PER_SCENE_YEAR="${CITY_CROP_SAMPLES_PER_SCENE_YEAR:-0}"
CITY_CROP_MIN_CITY_MASK_COVERAGE="${CITY_CROP_MIN_CITY_MASK_COVERAGE:-0.03}"
CITY_CROP_RANDOM_SEED="${CITY_CROP_RANDOM_SEED:-42}"
PREPROCESSING_STATS_SPLIT="${PREPROCESSING_STATS_SPLIT:-train}"

SPLIT_CSV_PATH="${SPLIT_CSV_PATH:-${CSV_ROOT}/df_norm_v3.csv}"
SPLIT_MANIFEST_PATH="${SPLIT_MANIFEST_PATH:-${EUMP_ROOT}/manifest/splits/legacy_compatible_seed42.json}"
SPLIT_SEED="${SPLIT_SEED:-42}"
SPLIT_TEST_SIZE="${SPLIT_TEST_SIZE:-0.3}"
SPLIT_YEAR_START="${SPLIT_YEAR_START:-2003}"
SPLIT_YEAR_END="${SPLIT_YEAR_END:-2020}"
SPLIT_FULL_CSV_POOL="${SPLIT_FULL_CSV_POOL:-0}"
SPLIT_ANNOTATE_CSV="${SPLIT_ANNOTATE_CSV:-1}"
SPLIT_ANNOTATED_CSV_PATH="${SPLIT_ANNOTATED_CSV_PATH:-}"
SPLIT_COLUMN="${SPLIT_COLUMN:-split}"

REPORT_DIR="${REPORT_DIR:-data/channel_distribution_report/processed_v1_preprocessing_v1}"
REPORT_MAX_SAMPLES_PER_SOURCE="${REPORT_MAX_SAMPLES_PER_SOURCE:-100000}"
REPORT_BINS="${REPORT_BINS:-80}"
REPORT_DPI="${REPORT_DPI:-150}"
REPORT_SEED="${REPORT_SEED:-42}"
ZERO_EXCLUDED_SOURCES="${ZERO_EXCLUDED_SOURCES:-BF BH DEM Electricity GAIA GDP NL POP}"

VALIDATE_TASK_CONFIG="${VALIDATE_TASK_CONFIG:-}"

usage() {
  cat <<'EOF'
Usage:
  bash script/data/run_eump_full_data_processing.sh [stage]

Stages:
  full              export -> validate -> split -> preprocess -> report
  export            export raw raster/scalar data into the EUMP scene store
  validate          validate the exported scene store; set VALIDATE_BH8_PADDING=1 for BH8 padding facts
  preprocess        build preprocessing_v1.json with stats and zero_excluded_stats
  split             build the fixed legacy-compatible split manifest
  report            build preprocessing distribution plots and CSV summaries
  post_export       validate -> split -> preprocess -> report
  refresh_stats     split -> preprocess -> report
  help              show this message

Multiple stages can be comma-separated, for example:
  bash script/data/run_eump_full_data_processing.sh split,preprocess,report

Common environment overrides:
  RAW_ROOT=data/raw_data
  CSV_ROOT=data/csv_data
  EUMP_ROOT=data/processed_v1
  WORKERS=32
  PREPROCESS_WORKERS=32
  OVERWRITE=1
  VALIDATE_BH8_PADDING=0
  PREPROCESSING_PATH=<eump-root>/manifest/stats/preprocessing_v1.json
  PREPROCESSING_STATS_SPLIT=train
  SPLIT_ANNOTATE_CSV=1
  SPLIT_COLUMN=split
  REPORT_DIR=data/channel_distribution_report/processed_v1_preprocessing_v1
  ZERO_EXCLUDED_SOURCES="BF BH DEM Electricity GAIA GDP NL POP"

Safety:
  OVERWRITE defaults to 0. Set OVERWRITE=1 only when you want the export stage
  to delete and rebuild EUMP_ROOT.
EOF
}

is_true() {
  case "${1}" in
    1|true|TRUE|yes|YES|y|Y)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

run_cmd() {
  printf '\n[run] '
  printf '%q ' "$@"
  printf '\n'
  if ! is_true "${DRY_RUN}"; then
    "$@"
  fi
}

print_config() {
  cat <<EOF
[config]
  STAGE=${STAGE}
  RAW_ROOT=${RAW_ROOT}
  CSV_ROOT=${CSV_ROOT}
  EUMP_ROOT=${EUMP_ROOT}
  BH8_ROOT=${BH8_ROOT}
  WORKERS=${WORKERS}
  PREPROCESS_WORKERS=${PREPROCESS_WORKERS}
  OVERWRITE=${OVERWRITE}
  EXPORT_SKIP_STATS=${EXPORT_SKIP_STATS}
  VALIDATE_BH8_PADDING=${VALIDATE_BH8_PADDING}
  PREPROCESSING_PATH=${PREPROCESSING_PATH}
  PREPROCESSING_STATS_SPLIT=${PREPROCESSING_STATS_SPLIT}
  SPLIT_MANIFEST_PATH=${SPLIT_MANIFEST_PATH}
  SPLIT_ANNOTATE_CSV=${SPLIT_ANNOTATE_CSV}
  SPLIT_COLUMN=${SPLIT_COLUMN}
  REPORT_DIR=${REPORT_DIR}
  ZERO_EXCLUDED_SOURCES=${ZERO_EXCLUDED_SOURCES}
EOF
}

run_export() {
  local cmd=(
    "${PYTHON_BIN}" script/data/export_eump_scene_store.py
    --raw-root "${RAW_ROOT}"
    --csv-root "${CSV_ROOT}"
    --output-root "${EUMP_ROOT}"
    --workers "${WORKERS}"
  )
  if is_true "${OVERWRITE}"; then
    cmd+=(--overwrite)
  elif [[ -d "${EUMP_ROOT}" ]]; then
    printf '\n[warn] %s already exists and OVERWRITE=0; export may fail if the store is not empty.\n' "${EUMP_ROOT}" >&2
    printf '[warn] Set OVERWRITE=1 to rebuild it from scratch.\n' >&2
  fi
  if is_true "${EXPORT_SKIP_STATS}"; then
    cmd+=(--skip-stats)
  fi
  if is_true "${VALIDATE_BH8_PADDING}"; then
    cmd+=(--validate-bh8-padding)
  fi
  run_cmd "${cmd[@]}"
}

run_validate() {
  local cmd=(
    "${PYTHON_BIN}" script/data/validate_eump_scene_store.py
    --bh8-root "${BH8_ROOT}"
    --eump-root "${EUMP_ROOT}"
  )
  if [[ -n "${VALIDATE_TASK_CONFIG}" ]]; then
    cmd+=(--task-config "${VALIDATE_TASK_CONFIG}")
  fi
  if is_true "${VALIDATE_BH8_PADDING}"; then
    cmd+=(--validate-bh8-padding)
  fi
  run_cmd "${cmd[@]}"
}

run_preprocess() {
  run_cmd \
    "${PYTHON_BIN}" script/data/build_eump_preprocessing.py \
    --eump-root "${EUMP_ROOT}" \
    --output-path "${PREPROCESSING_PATH}" \
    --split-manifest-path "${SPLIT_MANIFEST_PATH}" \
    --stats-split "${PREPROCESSING_STATS_SPLIT}" \
    --year-range "${SPLIT_YEAR_START}" "${SPLIT_YEAR_END}" \
    --city-crop-size "${CITY_CROP_SIZE}" \
    --city-crop-samples-per-scene-year "${CITY_CROP_SAMPLES_PER_SCENE_YEAR}" \
    --city-crop-min-city-mask-coverage "${CITY_CROP_MIN_CITY_MASK_COVERAGE}" \
    --city-crop-random-seed "${CITY_CROP_RANDOM_SEED}" \
    --workers "${PREPROCESS_WORKERS}"
}

run_split() {
  local cmd=(
    "${PYTHON_BIN}" script/data/build_eump_split_manifest.py
    --eump-root "${EUMP_ROOT}"
    --csv-path "${SPLIT_CSV_PATH}"
    --output-path "${SPLIT_MANIFEST_PATH}"
    --seed "${SPLIT_SEED}"
    --test-size "${SPLIT_TEST_SIZE}"
    --year-range "${SPLIT_YEAR_START}" "${SPLIT_YEAR_END}"
  )
  if is_true "${SPLIT_FULL_CSV_POOL}"; then
    cmd+=(--full-csv-pool)
  fi
  if is_true "${SPLIT_ANNOTATE_CSV}"; then
    cmd+=(--annotate-csv --split-column "${SPLIT_COLUMN}")
    if [[ -n "${SPLIT_ANNOTATED_CSV_PATH}" ]]; then
      cmd+=(--annotated-csv-path "${SPLIT_ANNOTATED_CSV_PATH}")
    fi
  fi
  run_cmd "${cmd[@]}"
}

run_report() {
  local zero_excluded_args=()
  read -r -a zero_excluded_args <<< "${ZERO_EXCLUDED_SOURCES}"
  run_cmd \
    "${PYTHON_BIN}" test/plot_eump_preprocessing_distributions.py \
    --eump-root "${EUMP_ROOT}" \
    --preprocessing-path "${PREPROCESSING_PATH}" \
    --out "${REPORT_DIR}" \
    --max-samples-per-source "${REPORT_MAX_SAMPLES_PER_SOURCE}" \
    --bins "${REPORT_BINS}" \
    --dpi "${REPORT_DPI}" \
    --seed "${REPORT_SEED}" \
    --zero-excluded-sources "${zero_excluded_args[@]}"
}

run_stage() {
  case "${1}" in
    full)
      run_export
      run_validate
      run_split
      run_preprocess
      run_report
      ;;
    export|store)
      run_export
      ;;
    validate)
      run_validate
      ;;
    preprocess|preprocessing|stats)
      run_preprocess
      ;;
    split)
      run_split
      ;;
    report|plots|distribution)
      run_report
      ;;
    post_export)
      run_validate
      run_split
      run_preprocess
      run_report
      ;;
    refresh_stats)
      run_split
      run_preprocess
      run_report
      ;;
    help|--help|-h)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown stage: %s\n\n' "${1}" >&2
      usage >&2
      exit 2
      ;;
  esac
}

IFS=',' read -r -a requested_stages <<< "${STAGE}"
if [[ "${#requested_stages[@]}" -eq 1 ]]; then
  case "${requested_stages[0]}" in
    help|--help|-h)
      usage
      exit 0
      ;;
  esac
fi

print_config
for requested_stage in "${requested_stages[@]}"; do
  run_stage "${requested_stage}"
done
