"""Shared LST model and data helpers for fixed-intervention evaluation."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

from hydra import compose, initialize_config_dir
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from urban_embedding.core.logging import setup_logger
from urban_embedding.core.paths import MODEL_DIR, resolve_model_artifact_path
from urban_embedding.data.processed.dataset import build_eump_dataloaders
from urban_embedding.data.processed.pipeline_spec import (
    PipelineSpec,
    get_single_downstream_target_schema,
    validate_eump_downstream_spec,
)
from urban_embedding.models.eump.swin.downstream import build_eump_downstream_model
from urban_embedding.training.checkpointing import load_model_state_for_evaluation
from urban_embedding.training.evaluation import normalize_evaluation_config

CONFIG_DIR = str(PROJECT_ROOT / "configs")
logger = setup_logger(log_dir=None, rank=0)


def _compose_hydra_config(
    pipeline: str,
    overrides: list[str],
    num_workers: int,
    pipeline_overrides: list[str] | tuple[str, ...] = (),
) -> dict[str, Any]:
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        raw = compose(
            config_name="evaluate",
            overrides=[f"pipeline={pipeline}", *[str(v) for v in pipeline_overrides], *overrides],
        )
    cfg = normalize_evaluation_config(raw)
    cfg["pipeline"]["num_workers"] = num_workers
    cfg["pipeline"]["center_ratio"] = 1.0
    return cfg


def _resolve_device(requested: str) -> torch.device:
    if str(requested).strip().lower() == "auto":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return torch.device(requested)


def _resolve_source_channel(
    *,
    source_name: str,
    preprocessing_spec: dict[str, Any],
    raster_input_channel_names: tuple[str, ...],
) -> tuple[int, str, dict[str, Any]]:
    source_spec = preprocessing_spec["sources"].get(source_name)
    if source_spec is None:
        raise KeyError(f"Source {source_name!r} not found in preprocessing spec")
    channel_names = tuple(str(n) for n in source_spec.get("output_channel_names", ()))
    if len(channel_names) != 1:
        raise ValueError(f"Source {source_name!r} must have exactly one channel, got {channel_names!r}")
    channel_name = channel_names[0]
    if channel_name not in raster_input_channel_names:
        raise ValueError(f"Channel {channel_name!r} not in raster inputs {raster_input_channel_names!r}")
    return raster_input_channel_names.index(channel_name), channel_name, source_spec


def _build_model(
    *,
    hydra_cfg: dict[str, Any],
    batch_schema: Any,
    target_schema: Any,
    device: torch.device,
    freeze_encoder: bool,
) -> nn.Module:
    encoder_source = str(hydra_cfg["downstream"]["encoder_source"])
    encoder_checkpoint_path = None
    if encoder_source in {
        "reconstruction_checkpoint",
        "convnext_reconstruction_checkpoint",
        "convnext_pretrained",
    }:
        encoder_checkpoint_path = resolve_model_artifact_path(
            str(hydra_cfg["downstream"]["encoder_model_name"]),
            str(hydra_cfg["downstream"]["encoder_checkpoint_file"]),
        )
    model = build_eump_downstream_model(
        model_config=dict(hydra_cfg["model"]),
        batch_schema=batch_schema,
        target_schema=target_schema,
        base_swin_path=str(Path(MODEL_DIR) / str(hydra_cfg["experiment"]["pretrain_model"])),
        encoder_source=encoder_source,
        encoder_checkpoint_path=encoder_checkpoint_path,
        load_pretrained=not bool(hydra_cfg["experiment"]["scratch_backbone"]),
        freeze_encoder=freeze_encoder,
        map_location=device,
    ).to(device)

    checkpoint_source = str(hydra_cfg["evaluation"]["checkpoint_source"])
    checkpoint_file = (
        str(hydra_cfg["evaluation"]["checkpoint_file"])
        if hydra_cfg["evaluation"]["checkpoint_file"]
        else (
            str(hydra_cfg["downstream"]["save_file"])
            if checkpoint_source == "best_model"
            else str(hydra_cfg["downstream"]["checkpoint_file"])
        )
    )
    load_model_state_for_evaluation(
        model=model,
        model_name=str(hydra_cfg["experiment"]["model_name"]),
        checkpoint_source=checkpoint_source,
        checkpoint_file=checkpoint_file,
        map_location=device,
    )
    model.eval()
    return model


def _find_city_entry(
    loaders: dict[str, DataLoader],
    *,
    scene_id: int,
    year: int,
) -> tuple[str, int] | None:
    for split_name in ("test", "val", "train"):
        dataset = loaders[split_name].dataset
        rows = getattr(dataset, "rows", None)
        if rows is None:
            continue
        for index, row in enumerate(rows):
            if int(row.get("scene_id", -1)) == scene_id and int(row.get("year", -1)) == year:
                return split_name, index
    return None
