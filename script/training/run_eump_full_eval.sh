#!/usr/bin/env bash
set -euo pipefail

# Run downstream EUMP evaluation from any directory after activating the target
# environment:
#   conda activate EUMP
#
# Examples:
#   EVAL_GPU=0 EVAL_NPROC=1 bash script/training/run_eump_full_eval.sh launch=eump_full/mae_v2_aux_raster0 evaluation.split=val
#   EVAL_GPU=0 EVAL_NPROC=1 bash script/training/run_eump_full_eval.sh all launch=eump_full/mae_v3
#   EVAL_GPU=0 EVAL_NPROC=1 bash script/training/run_eump_full_eval.sh pm25 launch=eump_full/mae_v3
#   EVAL_GPU=0 EVAL_NPROC=1 bash script/training/run_eump_full_eval.sh pm25-city launch=eump_full/mae_v3
#   EVAL_GPU=0 EVAL_NPROC=1 bash script/training/run_eump_full_eval.sh lst launch=eump_full/mae_v2_aux_raster0
#   EVAL_GPU=0 EVAL_NPROC=1 bash script/training/run_eump_full_eval.sh ghg-sum launch=eump_full/mae_v2_aux_raster0
#   EVAL_GPU=0 EVAL_NPROC=1 bash script/training/run_eump_full_eval.sh mlp-ghg launch=eump_full/scalar_mlp
#   EVAL_GPU=0 EVAL_NPROC=1 bash script/training/run_eump_full_eval.sh mlp-pm25 launch=eump_full/scalar_mlp
#   EVAL_GPU=0 EVAL_NPROC=1 bash script/training/run_eump_full_eval.sh aef-mlp-pm25 launch=eump_full/aef_mlp

if (($# > 0)) && [[ "$1" != *"="* ]]; then
  MODE="$1"
  shift
else
  MODE=""
fi
EXTRA_OVERRIDES=()
if (($# > 0)); then
  EXTRA_OVERRIDES=("$@")
fi

# Evaluation is intentionally single-GPU by default and uses EVAL_* resource
# knobs so train-time GPU/NPROC exports do not leak into evaluation.
GPU="${EVAL_GPU:-0}"
NPROC="${EVAL_NPROC:-1}"
NNODES="${EVAL_NNODES:-1}"
NODE_RANK="${EVAL_NODE_RANK:-0}"
MASTER_ADDR="${EVAL_MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${EVAL_MASTER_PORT:-29501}"

EUMP_TORCH_SHARING_STRATEGY="${EUMP_TORCH_SHARING_STRATEGY:-file_system}"
LOG_PROGRESS_BAR="${LOG_PROGRESS_BAR:-1}"
DRY_RUN="${DRY_RUN:-0}"
DEFAULT_LAUNCH_CONFIG="${EUMP_LAUNCH_CONFIG:-eump_full/mae_v2}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

normalize_bool_override() {
  local raw_value="$1"
  local lower_value
  lower_value="$(printf '%s' "${raw_value}" | tr '[:upper:]' '[:lower:]')"
  case "${lower_value}" in
    1|true|yes|on)
      printf 'true\n'
      ;;
    0|false|no|off)
      printf 'false\n'
      ;;
    *)
      echo "Expected boolean-like value, got: ${raw_value}" >&2
      exit 2
      ;;
  esac
}

LOG_PROGRESS_BAR_OVERRIDE="$(normalize_bool_override "${LOG_PROGRESS_BAR}")"

print_command() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
}

launch_config_from_overrides() {
  local launch_config="${DEFAULT_LAUNCH_CONFIG}"
  if ((${#EXTRA_OVERRIDES[@]} == 0)); then
    printf '%s\n' "${launch_config}"
    return 0
  fi
  local override
  for override in "${EXTRA_OVERRIDES[@]}"; do
    case "${override}" in
      launch=*)
        launch_config="${override#launch=}"
        ;;
      +launch=*|++launch=*)
        launch_config="${override#*=}"
        ;;
      pipeline=*|+pipeline=*|++pipeline=*)
        echo "Use launch=<name> to select an experiment launch config; pipeline= is reserved for runnable pipeline jobs." >&2
        exit 2
        ;;
    esac
  done
  printf '%s\n' "${launch_config}"
}

LAUNCH_CONFIG="$(launch_config_from_overrides)"

shared_overrides() {
  if ((${#EXTRA_OVERRIDES[@]} == 0)); then
    return 0
  fi
  local override
  for override in "${EXTRA_OVERRIDES[@]}"; do
    case "${override}" in
      launch=*|+launch=*|++launch=*)
        ;;
      *)
        printf '%s\n' "${override}"
        ;;
    esac
  done
}

SHARED_OVERRIDES=()
while IFS= read -r override; do
  SHARED_OVERRIDES+=("${override}")
done < <(shared_overrides)

cache_builder_extra_overrides() {
  if ((${#SHARED_OVERRIDES[@]} == 0)); then
    return 0
  fi
  local override
  for override in "${SHARED_OVERRIDES[@]}"; do
    case "${override}" in
      eval=*|+eval=*|evaluation.*|+evaluation.*)
        ;;
      *)
        printf '%s\n' "${override}"
        ;;
    esac
  done
}

resolve_launch_jobs() {
  local phase="$1"
  local mode="$2"
  local -a cmd=(
    python
    -m
    urban_embedding.cli.resolve_eump_launch
    --phase
    "${phase}"
    --launch
    "${LAUNCH_CONFIG}"
    --config-dir
    "${REPO_ROOT}/configs"
  )
  if [[ -n "${mode}" ]]; then
    cmd+=(--mode "${mode}")
  fi
  "${cmd[@]}"
}

run_torchrun() {
  local -a cmd=(
    torchrun
    --nproc_per_node="${NPROC}"
    --nnodes="${NNODES}"
    --node_rank="${NODE_RANK}"
    --master_addr="${MASTER_ADDR}"
    --master_port="${MASTER_PORT}"
    main/evaluate_eump.py
  )
  cmd+=("$@")

  print_command env "CUDA_VISIBLE_DEVICES=${GPU}" "EUMP_TORCH_SHARING_STRATEGY=${EUMP_TORCH_SHARING_STRATEGY}" "${cmd[@]}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  CUDA_VISIBLE_DEVICES="${GPU}" EUMP_TORCH_SHARING_STRATEGY="${EUMP_TORCH_SHARING_STRATEGY}" "${cmd[@]}"
}

run_python_cache_builder() {
  local -a cmd=(
    python
    -m
    urban_embedding.cli.build_eump_city_scalar_mlp_cache
  )
  cmd+=("$@")

  print_command "${cmd[@]}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  "${cmd[@]}"
}

render_task_progress() {
  local completed="$1"
  local total="$2"
  local task_name="$3"
  if [[ "${LOG_PROGRESS_BAR_OVERRIDE}" != "true" ]]; then
    return 0
  fi

  local width=20
  local filled=0
  if (( total > 0 )); then
    filled=$((completed * width / total))
  fi
  local empty=$((width - filled))
  local bar
  printf -v bar '%*s' "${filled}" ''
  bar="${bar// /#}"
  local padding
  printf -v padding '%*s' "${empty}" ''
  padding="${padding// /-}"
  local percent=0
  if (( total > 0 )); then
    percent=$((completed * 100 / total))
  fi

  printf '[%s%s] %d/%d %3d%% task=%s\n' "${bar}" "${padding}" "${completed}" "${total}" "${percent}" "${task_name}" >&2
}

run_eval_job() {
  local kind="$1"
  local name="$2"
  local pipeline_name="$3"
  shift 3
  local -a job_overrides=("$@")
  local -a overrides=(
    "logging.progress_bar=${LOG_PROGRESS_BAR_OVERRIDE}"
    "pipeline=${pipeline_name}"
  )
  if ((${#SHARED_OVERRIDES[@]})); then
    overrides+=("${SHARED_OVERRIDES[@]}")
  fi
  if ((${#job_overrides[@]})); then
    overrides+=("${job_overrides[@]}")
  fi

  case "${kind}" in
    swin_downstream)
      run_torchrun "${overrides[@]}"
      ;;
    city_scalar_mlp)
      local -a cache_overrides=(
        "logging.progress_bar=${LOG_PROGRESS_BAR_OVERRIDE}"
        "pipeline=${pipeline_name}"
      )
      local override
      while IFS= read -r override; do
        cache_overrides+=("${override}")
      done < <(cache_builder_extra_overrides)
      run_python_cache_builder "${cache_overrides[@]}"
      run_torchrun "${overrides[@]}"
      ;;
    *)
      echo "Unsupported eval job kind=${kind} for job=${name}" >&2
      exit 2
      ;;
  esac
}

run_resolved_jobs() {
  local job_output
  job_output="$(resolve_launch_jobs eval "${MODE}")"
  local -a job_lines=()
  if [[ -n "${job_output}" ]]; then
    while IFS= read -r line; do
      job_lines+=("${line}")
    done <<< "${job_output}"
  fi
  local total="${#job_lines[@]}"
  local completed=0
  local line
  for line in "${job_lines[@]}"; do
    local -a fields=()
    IFS=$'\t' read -r -a fields <<< "${line}"
    render_task_progress "${completed}" "${total}" "${fields[1]}"
    run_eval_job "${fields[0]}" "${fields[1]}" "${fields[2]}" "${fields[@]:3}"
    completed=$((completed + 1))
    render_task_progress "${completed}" "${total}" "${fields[1]}"
  done
}

run_resolved_jobs
