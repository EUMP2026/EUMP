#!/usr/bin/env bash
set -euo pipefail

# Run from any directory after activating the target environment:
#   conda activate EUMP
#
# Examples:
#   GPU=0,1 bash script/training/run_eump_full_train.sh all
#   GPU=0,1 bash script/training/run_eump_full_train.sh pm25
#   GPU=0,1 bash script/training/run_eump_full_train.sh all launch=eump_full/mae_v3
#   GPU=0,1 bash script/training/run_eump_full_train.sh lst launch=eump_full/mae_v3
#   GPU=0,1 bash script/training/run_eump_full_train.sh mlp-lst launch=eump_full/scalar_mlp
#   GPU=0,1 bash script/training/run_eump_full_train.sh mlp-pm25 launch=eump_full/scalar_mlp
#   GPU=0,1 bash script/training/run_eump_full_train.sh aef-mlp-pm25 launch=eump_full/aef_mlp
#   bash script/training/run_eump_full_train.sh verify-data launch=eump_full/mae_v3

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

GPU="${GPU:-0}"
if [[ -z "${NPROC:-}" ]]; then
  IFS=',' read -r -a _EUMP_GPU_LIST <<< "${GPU}"
  NPROC="${#_EUMP_GPU_LIST[@]}"
fi
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

EUMP_TORCH_SHARING_STRATEGY="${EUMP_TORCH_SHARING_STRATEGY:-file_system}"
LOG_PROGRESS_BAR="${LOG_PROGRESS_BAR:-1}"
DRY_RUN="${DRY_RUN:-0}"
DEFAULT_LAUNCH_CONFIG="${EUMP_LAUNCH_CONFIG:-eump_full/mae_v3}"

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
    main/train_eump.py
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

shared_override_value() {
  local target_key="$1"
  local override
  for override in "${SHARED_OVERRIDES[@]}"; do
    local raw_key="${override%%=*}"
    local raw_value="${override#*=}"
    raw_key="${raw_key#+}"
    raw_key="${raw_key#+}"
    if [[ "${raw_key}" == "${target_key}" && "${override}" == *"="* ]]; then
      printf '%s\n' "${raw_value}"
      return 0
    fi
  done
  return 1
}

default_experiment_group_from_launch() {
  printf '%s\n' "${LAUNCH_CONFIG//\//_}"
}

city_scalar_cv_experiment_group() {
  local override_group
  override_group="$(shared_override_value experiment.group || true)"
  if [[ -n "${override_group}" ]]; then
    printf '%s\n' "${override_group}"
    return 0
  fi
  default_experiment_group_from_launch
}

run_city_scalar_cv_job() {
  local name="$1"
  local target_pipeline="$2"
  shift 2
  local -a job_overrides=("$@")

  local group
  group="$(city_scalar_cv_experiment_group)"
  local fold_seed="${CV_FOLD_SEED:-42}"
  local py_bin="${PYTHON_BIN:-python}"

  local output_prefix run_name config_path model_name
  case "${name}" in
    traffic_cv|traffic)
      output_prefix="traffic_cv"
      run_name="traffic_cv_seed${fold_seed}"
      config_path="${REPO_ROOT}/configs/run/eump_full/traffic_cv.yaml"
      model_name="swin_embed_head_128_d05"
      ;;
    pm25_cv|pm25)
      output_prefix="pm25_cv"
      run_name="pm25_cv_seed${fold_seed}"
      config_path="${REPO_ROOT}/configs/run/eump_full/pm25_cv.yaml"
      model_name="swin_embed_head_128_d05"
      ;;
    *)
      echo "Unsupported city-scalar CV job=${name}" >&2
      exit 2
      ;;
  esac

  local -a cmd=(
    "${py_bin}"
    "script/experiments/run_city_scalar_cv.py"
    --config
    "${config_path}"
    --models
    "${model_name}"
    --experiment-group
    "${group}"
    --run-name
    "${run_name}"
    --fold-seed
    "${fold_seed}"
    --output-prefix
    "${output_prefix}"
  )
  if ((${#SHARED_OVERRIDES[@]})); then
    cmd+=("${SHARED_OVERRIDES[@]}")
  fi
  if ((${#job_overrides[@]})); then
    cmd+=("${job_overrides[@]}")
  fi

  print_command env \
    "GPU=${GPU}" \
    "NPROC=${NPROC}" \
    "CUDA_VISIBLE_DEVICES=${GPU}" \
    "LOG_PROGRESS_BAR=${LOG_PROGRESS_BAR}" \
    "EUMP_TORCH_SHARING_STRATEGY=${EUMP_TORCH_SHARING_STRATEGY}" \
    "PYTHONPATH=${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${cmd[@]}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  GPU="${GPU}" \
  NPROC="${NPROC}" \
  CUDA_VISIBLE_DEVICES="${GPU}" \
  LOG_PROGRESS_BAR="${LOG_PROGRESS_BAR}" \
  EUMP_TORCH_SHARING_STRATEGY="${EUMP_TORCH_SHARING_STRATEGY}" \
  PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
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

run_training_job() {
  local kind="$1"
  local name="$2"
  local pipeline_name="$3"
  shift 3
  local -a job_overrides=("$@")
  local -a overrides=(
    "logging.progress_bar=${LOG_PROGRESS_BAR_OVERRIDE}"
    "pipeline=${pipeline_name}"
  )

  case "${kind}" in
    swin_pretrain|swin_downstream|city_scalar_mlp)
      if ((${#SHARED_OVERRIDES[@]})); then
        overrides+=("${SHARED_OVERRIDES[@]}")
      fi
      if ((${#job_overrides[@]})); then
        overrides+=("${job_overrides[@]}")
      fi
      run_torchrun "${overrides[@]}"
      ;;
    city_scalar_cache)
      local -a cache_overrides=("${overrides[@]}")
      local override
      while IFS= read -r override; do
        cache_overrides+=("${override}")
      done < <(cache_builder_extra_overrides)
      if ((${#job_overrides[@]})); then
        cache_overrides+=("${job_overrides[@]}")
      fi
      run_python_cache_builder "${cache_overrides[@]}"
      ;;
    city_scalar_cv)
      run_city_scalar_cv_job "${name}" "${pipeline_name}" "${job_overrides[@]}"
      ;;
    *)
      echo "Unsupported train job kind=${kind} for job=${name}" >&2
      exit 2
      ;;
  esac
}

run_resolved_jobs() {
  local phase="$1"
  local mode="$2"
  local job_output
  job_output="$(resolve_launch_jobs "${phase}" "${mode}")"
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
    run_training_job "${fields[0]}" "${fields[1]}" "${fields[2]}" "${fields[@]:3}"
    completed=$((completed + 1))
    render_task_progress "${completed}" "${total}" "${fields[1]}"
  done
}

run_verify_data() {
  local job_output
  job_output="$(resolve_launch_jobs verify "${MODE}")"
  local -a job_lines=()
  if [[ -n "${job_output}" ]]; then
    while IFS= read -r line; do
      job_lines+=("${line}")
    done <<< "${job_output}"
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    local -a dry_run_cmd=(verify-eump-data "launch=${LAUNCH_CONFIG}")
    if ((${#SHARED_OVERRIDES[@]})); then
      dry_run_cmd+=("${SHARED_OVERRIDES[@]}")
    fi
    print_command "${dry_run_cmd[@]}"
    return 0
  fi

  local -a verify_pipelines=()
  local line
  for line in "${job_lines[@]}"; do
    local -a fields=()
    IFS=$'\t' read -r -a fields <<< "${line}"
    verify_pipelines+=("${fields[2]}")
  done

  local verify_pipelines_csv
  local IFS=','
  verify_pipelines_csv="${verify_pipelines[*]}"
  unset IFS

  local -a verify_cmd=(python -)
  if ((${#SHARED_OVERRIDES[@]})); then
    verify_cmd+=("${SHARED_OVERRIDES[@]}")
  fi
  EUMP_VERIFY_PIPELINES="${verify_pipelines_csv}" "${verify_cmd[@]}" <<'PY'
from __future__ import annotations

import os
from pathlib import Path
import sys

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from urban_embedding.data.processed import (
    EUMP_CITY_SCALAR_DOWNSTREAM_KIND,
    EUMP_DOWNSTREAM_KIND,
    PipelineSpec,
    ProcessedSceneStoreReader,
    build_eump_batch_schema,
    get_single_downstream_target_schema,
    load_split_manifest,
    load_preprocessing_spec,
    normalize_eump_pipeline_kind,
    validate_eump_downstream_spec,
    validate_preprocessing_spec,
    validate_preprocessing_stats_scope,
    validate_eump_scalar_mlp_downstream_spec,
)
from urban_embedding.training.eump.city_scalar_mlp.cache import (
    build_aef_feature_layout,
    normalize_city_scalar_cache_options,
)
from urban_embedding.training.eump.city_scalar_mlp.features import build_city_level_feature_layout

repo_root = Path.cwd()
config_dir = repo_root / "configs"
extra_overrides = list(sys.argv[1:])
pipelines = tuple(value for value in os.environ["EUMP_VERIFY_PIPELINES"].split(",") if value)
reader_cache = {}


def reader_for(data_root: Path):
    key = str(data_root)
    if key not in reader_cache:
        reader = ProcessedSceneStoreReader(data_root)
        preprocessing = load_preprocessing_spec(data_root)
        validate_preprocessing_spec(preprocessing)
        reader_cache[key] = (reader, preprocessing)
    return reader_cache[key]


with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
    for pipeline_name in pipelines:
        cfg = compose(
            config_name="train",
            overrides=[
                f"pipeline={pipeline_name}",
                *extra_overrides,
            ],
        )
        data_root = Path(str(cfg.data.eump_root))
        reader, preprocessing = reader_for(data_root)
        pipeline_config = OmegaConf.to_container(cfg.pipeline, resolve=True)
        pipeline = PipelineSpec.from_mapping(pipeline_config)
        pipeline_kind = normalize_eump_pipeline_kind(pipeline.kind)
        if pipeline_kind == EUMP_DOWNSTREAM_KIND:
            validate_eump_downstream_spec(pipeline)
        elif pipeline_kind == EUMP_CITY_SCALAR_DOWNSTREAM_KIND:
            validate_eump_scalar_mlp_downstream_spec(pipeline)
        task_config = pipeline.to_task_config()
        reader.validate_task_config(task_config)
        split_manifest_path = cfg.data.get("split_manifest_path")
        if split_manifest_path:
            load_split_manifest(
                split_manifest_path,
                available_scene_ids=reader.scene_index.keys(),
                expected_year_range=pipeline.year_range,
            )
            validate_preprocessing_stats_scope(
                preprocessing,
                split_manifest_path=split_manifest_path,
                expected_year_range=pipeline.year_range,
            )
        rows = [
            row
            for row in reader.iter_scene_years(year_range=pipeline.year_range)
            if not reader.missing_sources_for_task(row["scene_id"], int(row["year"]), task_config)
        ]
        if not rows:
            raise SystemExit(f"{pipeline_name}: no compatible rows under data.eump_root={data_root}")
        schema = build_eump_batch_schema(pipeline, reader.source_specs, preprocessing)
        if pipeline_kind == EUMP_CITY_SCALAR_DOWNSTREAM_KIND:
            cache_options = normalize_city_scalar_cache_options(pipeline_config)
            if cache_options.feature_source.kind == "aef_csv":
                layout = build_aef_feature_layout(cache_options.feature_source)
            else:
                layout = build_city_level_feature_layout(
                    pipeline_spec=pipeline,
                    source_specs=reader.source_specs,
                    preprocessing_spec=preprocessing,
                    batch_schema=schema,
                )
            target = get_single_downstream_target_schema(schema)
            print(
                f"{pipeline_name}: root={data_root} rows={len(rows)} "
                f"scalar_inputs={layout.feature_dim} "
                f"scalar_targets={{'{target.name}': {target.channels}}}"
            )
        else:
            print(
                f"{pipeline_name}: root={data_root} rows={len(rows)} "
                f"raster_inputs={schema.raster_input_channels} "
                f"raster_targets={schema.raster_output_channels} "
                f"scalar_targets={schema.scalar_output_dims}"
            )
PY
}

if [[ "${MODE}" == "verify-data" ]]; then
  run_verify_data
else
  run_resolved_jobs train "${MODE}"
fi
