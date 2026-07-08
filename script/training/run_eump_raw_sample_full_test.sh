#!/usr/bin/env bash
set -euo pipefail

# Run from any directory after activating the target environment:
#   conda activate EUMP
#
# Usage:
#   bash script/training/run_eump_raw_sample_full_test.sh prepare
#   bash script/training/run_eump_raw_sample_full_test.sh smoke
#   bash script/training/run_eump_raw_sample_full_test.sh full
#   bash script/training/run_eump_raw_sample_full_test.sh verify
#   bash script/training/run_eump_raw_sample_full_test.sh all
#
# Common overrides:
#   SAMPLE_COUNT=10 WORKERS=8 GPU=0 bash script/training/run_eump_raw_sample_full_test.sh all
#   RECON_FULL_EPOCHS=200 DOWN_FULL_EPOCHS=100 bash script/training/run_eump_raw_sample_full_test.sh full

MODE="${1:-all}"

SAMPLE_COUNT="${SAMPLE_COUNT:-10}"
SEED="${SEED:-42}"
WORKERS="${WORKERS:-8}"
GPU="${GPU:-0}"

RECON_SMOKE_BATCH="${RECON_SMOKE_BATCH:-2}"
RECON_SMOKE_EPOCHS="${RECON_SMOKE_EPOCHS:-1}"
DOWN_SMOKE_BATCH="${DOWN_SMOKE_BATCH:-2}"
DOWN_SMOKE_EPOCHS="${DOWN_SMOKE_EPOCHS:-1}"

RECON_FULL_BATCH="${RECON_FULL_BATCH:-128}"
RECON_FULL_EPOCHS="${RECON_FULL_EPOCHS:-200}"
DOWN_FULL_BATCH="${DOWN_FULL_BATCH:-128}"
DOWN_FULL_EPOCHS="${DOWN_FULL_EPOCHS:-100}"

EXPERIMENT_GROUP="${EXPERIMENT_GROUP:-eump_raw_sample}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

run_prepare() {
  python script/data/build_raw_data_sample.py \
    --source-root data/raw_data \
    --output-root data/raw_data_sample \
    --sample-count "${SAMPLE_COUNT}" \
    --seed "${SEED}" \
    --require-datasets BH8_v1 VI3_v1 NL_v1 land_cover LCZ PM25 road_density electricity GDP lst_v1 \
    --overwrite

  python script/data/export_eump_scene_store.py \
    --raw-root data/raw_data_sample \
    --csv-root data/csv_data \
    --output-root data/eump_raw_sample \
    --overwrite \
    --workers "${WORKERS}"

  python script/data/build_eump_preprocessing.py \
    --eump-root data/eump_raw_sample

  python script/data/validate_eump_scene_store.py \
    --bh8-root data/raw_data_sample/BH8_v1 \
    --eump-root data/eump_raw_sample
}

run_reconstruct() {
  local batch_size="$1"
  local epochs="$2"
  shift 2

  CUDA_VISIBLE_DEVICES="${GPU}" torchrun --nproc_per_node=1 --nnodes=1 main/train_eump.py \
    data=eump_full/raw_sample pipeline=eump_full/mae_v2/pretrain \
    experiment.group="${EXPERIMENT_GROUP}" experiment.pretrain_model=swin_T \
    reconstruct.batch_size="${batch_size}" reconstruct.epochs="${epochs}" \
    "$@"
}

run_lst_downstream() {
  local batch_size="$1"
  local epochs="$2"
  shift 2

  CUDA_VISIBLE_DEVICES="${GPU}" torchrun --nproc_per_node=1 --nnodes=1 main/train_eump.py \
    data=eump_full/raw_sample pipeline=eump_full/mae_v2/lst \
    experiment.group="${EXPERIMENT_GROUP}" experiment.pretrain_model=swin_T \
    downstream.encoder_model_name="${EXPERIMENT_GROUP}/pretrain" downstream.encoder_checkpoint_file=eump_reconstruction_model.pt \
    downstream.batch_size="${batch_size}" downstream.epochs="${epochs}" \
    "$@"
}

run_ghg_downstream() {
  local batch_size="$1"
  local epochs="$2"
  shift 2

  CUDA_VISIBLE_DEVICES="${GPU}" torchrun --nproc_per_node=1 --nnodes=1 main/train_eump.py \
    data=eump_full/raw_sample pipeline=eump_full/mae_v2/ghg \
    experiment.group="${EXPERIMENT_GROUP}" experiment.pretrain_model=swin_T \
    downstream.encoder_model_name="${EXPERIMENT_GROUP}/pretrain" downstream.encoder_checkpoint_file=eump_reconstruction_model.pt \
    downstream.batch_size="${batch_size}" downstream.epochs="${epochs}" \
    "$@"
}

run_smoke() {
  run_reconstruct "${RECON_SMOKE_BATCH}" "${RECON_SMOKE_EPOCHS}" pipeline.num_workers=0
  run_lst_downstream "${DOWN_SMOKE_BATCH}" "${DOWN_SMOKE_EPOCHS}" pipeline.num_workers=0
  run_ghg_downstream "${DOWN_SMOKE_BATCH}" "${DOWN_SMOKE_EPOCHS}" pipeline.num_workers=0
}

run_full() {
  run_reconstruct "${RECON_FULL_BATCH}" "${RECON_FULL_EPOCHS}"
  run_lst_downstream "${DOWN_FULL_BATCH}" "${DOWN_FULL_EPOCHS}"
  run_ghg_downstream "${DOWN_FULL_BATCH}" "${DOWN_FULL_EPOCHS}"
}

run_verify() {
  python -m unittest discover -s test
  python -m compileall urban_embedding main script
}

case "${MODE}" in
  prepare)
    run_prepare
    ;;
  smoke)
    run_smoke
    ;;
  full)
    run_full
    ;;
  verify)
    run_verify
    ;;
  all)
    run_prepare
    run_smoke
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    echo "Expected one of: prepare, smoke, full, verify, all" >&2
    exit 2
    ;;
esac
