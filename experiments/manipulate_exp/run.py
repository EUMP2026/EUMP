#!/usr/bin/env python3
"""
experiments/manipulate_exp/run.py

Input-manipulation experiment: perturb one raster input channel in physical space
for a specific city and year, then measure the downstream predictor's spatial response.

Default usage (Beijing, NDVI +0.1, LST predictor):

    python experiments/manipulate_exp/run.py \\
        --scene-id 782 --year 2020 --delta 0.1 \\
        data=eump_full/server_v2 pipeline=eump_full/mae_v3/lst

Outputs are written to:
    experiments/manipulate_exp/outputs/<scene_id>_<year>/
        delta_lst_<scene_id>_<year>.npz
        delta_lst_<scene_id>_<year>.png
        summary_<scene_id>_<year>.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import yaml
from hydra import compose, initialize_config_dir
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as torch_mp
from torch.utils.data import DataLoader, Subset


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
from urban_embedding.training.distributed import (
    destroy_distributed_process_group,
    distributed_barrier,
    get_rank,
    get_world_size,
    init_distributed_process_group,
    resolve_device as resolve_distributed_device,
)
from urban_embedding.training.evaluation import normalize_evaluation_config
from urban_embedding.training.regression_metrics import inverse_processed_tensor


CONFIG_DIR = str(PROJECT_ROOT / "configs")
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = EXPERIMENT_DIR / "config.yaml"
SUPPORTED_SOURCE_MODES = {"pass_through", "divide", "zscore", "log_zscore"}
DEFAULT_BULK_DELTAS = (0.1, 0.3, 0.5)
BULK_CHANNEL_NAMES = ("ndvi0", "ndvi1", "lst0", "lst1", "city_mask")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Perturb one raster input channel in physical space for a target city/year "
            "and save before/after downstream predictions as a spatial delta map."
        )
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to experiment config.yaml. Default: {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument(
        "--scene-id",
        default=None,
        help="Target scene ID (city). Overrides config.yaml scene_id. Example: 782",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Target year. Overrides config.yaml year. Example: 2020",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Input channel name to perturb. Overrides config.yaml source. Default: NDVI",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=None,
        help="Additive delta in physical units. Overrides config.yaml delta. Default: 0.1",
    )
    parser.add_argument(
        "--deltas",
        default=None,
        help=(
            "Comma-separated additive deltas for --all-cities mode. "
            "Default: 0.1,0.3,0.5"
        ),
    )
    parser.add_argument(
        "--all-cities",
        action="store_true",
        help=(
            "Export all compatible cities for the selected year. Writes one merged NPZ "
            "and does not render PNGs."
        ),
    )
    parser.add_argument(
        "--limit-cities",
        type=int,
        default=None,
        help="Optional cap for --all-cities smoke tests. Applied after deterministic sorting.",
    )
    parser.add_argument(
        "--mask-scope",
        choices=("city", "all"),
        default=None,
        help="Perturb within city_mask ('city') or all valid pixels ('all'). Default: city",
    )
    parser.add_argument(
        "--min-value",
        type=float,
        default=None,
        help="Lower clamp for perturbed physical values (NDVI default: -1.0).",
    )
    parser.add_argument(
        "--max-value",
        type=float,
        default=None,
        help="Upper clamp for perturbed physical values (NDVI default: 1.0).",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default=None,
        help="Evaluation split to search. Overrides config.yaml split. Default: test",
    )
    parser.add_argument(
        "--checkpoint-source",
        choices=("best_model", "training_checkpoint"),
        default=None,
        help=(
            "Evaluation checkpoint source. Default: config.yaml/evaluation default, "
            "normally best_model."
        ),
    )
    parser.add_argument(
        "--checkpoint-file",
        default=None,
        help=(
            "Optional checkpoint file override. Absolute paths are accepted; relative "
            "paths resolve under model/<experiment.model_name>/."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Overrides config.yaml output_dir.",
    )
    parser.add_argument(
        "--gpu",
        default=None,
        help=(
            "GPU id(s) to use, e.g. 0 or 0,1. Sets CUDA_VISIBLE_DEVICES before "
            "any CUDA initialization. Use with --device auto (default) or "
            "--device cuda to let PyTorch pick the first visible GPU."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device: auto, cpu, cuda, cuda:0, etc. Default: auto",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers. Default: 0",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for the data loader. Default: 4",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help=(
            "Hydra config overrides, e.g. "
            "data=eump_full/server_v2 pipeline=eump_full/mae_v3/lst"
        ),
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_exp_config(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _merge_args(exp_cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cfg = dict(exp_cfg)
    if args.scene_id is not None:
        cfg["scene_id"] = str(args.scene_id)
    if args.year is not None:
        cfg["year"] = int(args.year)
    if args.source is not None:
        cfg["source"] = str(args.source)
    if args.delta is not None:
        cfg["delta"] = float(args.delta)
    if args.deltas is not None:
        cfg["deltas"] = str(args.deltas)
    if args.mask_scope is not None:
        cfg["mask_scope"] = str(args.mask_scope)
    elif args.all_cities:
        cfg["mask_scope"] = "all"
    if args.min_value is not None:
        cfg["min_value"] = float(args.min_value)
    if args.max_value is not None:
        cfg["max_value"] = float(args.max_value)
    if args.split is not None:
        cfg["split"] = str(args.split)
    if args.checkpoint_source is not None:
        cfg["checkpoint_source"] = str(args.checkpoint_source)
    if args.checkpoint_file is not None:
        cfg["checkpoint_file"] = str(args.checkpoint_file)
    if args.output_dir is not None:
        cfg["output_dir"] = str(args.output_dir)
    cfg.setdefault("scene_id", "782")
    cfg.setdefault("year", 2020)
    cfg.setdefault("source", "NDVI")
    cfg.setdefault("delta", 0.1)
    cfg.setdefault("deltas", list(DEFAULT_BULK_DELTAS))
    cfg.setdefault("mask_scope", "city")
    cfg.setdefault("split", "test")
    cfg.setdefault("checkpoint_source", "best_model")
    cfg.setdefault("checkpoint_file", None)
    cfg.setdefault("output_dir", "experiments/manipulate_exp/outputs")
    return cfg


def _compose_hydra_config(args: argparse.Namespace, num_workers: int) -> dict[str, Any]:
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        raw = compose(config_name="evaluate", overrides=args.overrides)
    cfg = normalize_evaluation_config(raw)
    cfg["pipeline"]["num_workers"] = num_workers
    cfg["pipeline"]["center_ratio"] = 1.0
    return cfg


def _resolve_device(requested: str) -> torch.device:
    if str(requested).strip().lower() == "auto":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested CUDA device {requested!r} but CUDA is unavailable")
    return device


def _parse_deltas(value: Any) -> tuple[float, ...]:
    if value is None:
        return DEFAULT_BULK_DELTAS
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if not parts:
            raise ValueError("--deltas must contain at least one numeric value")
        return tuple(float(part) for part in parts)
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("deltas must contain at least one numeric value")
        return tuple(float(part) for part in value)
    return (float(value),)


def _delta_key_suffix(delta: float) -> str:
    text = f"{float(delta):g}".lower()
    return text.replace("-", "m").replace("+", "").replace(".", "")


def _resolve_source_bounds(
    source_name: str,
    min_value: Any,
    max_value: Any,
) -> tuple[float | None, float | None]:
    if min_value is not None or max_value is not None:
        return (
            None if min_value is None else float(min_value),
            None if max_value is None else float(max_value),
        )
    normalized = str(source_name).upper()
    if normalized == "NDVI":
        return -1.0, 1.0
    if normalized == "POP":
        return 0.0, None
    return None, None


def _apply_checkpoint_overrides(hydra_cfg: dict[str, Any], cfg: dict[str, Any]) -> None:
    checkpoint_source = str(cfg.get("checkpoint_source") or "best_model")
    if checkpoint_source not in {"best_model", "training_checkpoint"}:
        raise ValueError(
            "checkpoint_source must be 'best_model' or 'training_checkpoint', "
            f"got {checkpoint_source!r}"
        )
    hydra_cfg["evaluation"]["checkpoint_source"] = checkpoint_source
    checkpoint_file = cfg.get("checkpoint_file")
    if checkpoint_file not in (None, ""):
        hydra_cfg["evaluation"]["checkpoint_file"] = str(checkpoint_file)


def _configure_torch_multiprocessing(logger: Any | None = None, *, rank: int = 0) -> None:
    requested_strategy = os.environ.get("EUMP_TORCH_SHARING_STRATEGY")
    if not requested_strategy:
        return

    available_strategies = set(torch_mp.get_all_sharing_strategies())
    if requested_strategy not in available_strategies:
        available = ", ".join(sorted(available_strategies))
        raise ValueError(
            f"Unsupported EUMP_TORCH_SHARING_STRATEGY={requested_strategy!r}. "
            f"Available strategies: {available}"
        )

    current_strategy = torch_mp.get_sharing_strategy()
    if current_strategy != requested_strategy:
        torch_mp.set_sharing_strategy(requested_strategy)

    if int(rank) == 0 and logger is not None:
        logger.info(f"Use torch multiprocessing sharing strategy: {requested_strategy}")


def _resolve_output_base(path_value: Any) -> Path:
    output_base = Path(str(path_value))
    if not output_base.is_absolute():
        output_base = (PROJECT_ROOT / output_base).resolve()
    return output_base


# ---------------------------------------------------------------------------
# Perturbation helpers  (adapted from script/experiments/run_eump_input_counterfactual.py)
# ---------------------------------------------------------------------------

def _resolve_source_channel(
    *,
    source_name: str,
    preprocessing_spec: dict[str, Any],
    raster_input_channel_names: tuple[str, ...],
) -> tuple[int, str, dict[str, Any]]:
    source_spec = preprocessing_spec["sources"].get(source_name)
    if source_spec is None:
        raise KeyError(f"Source {source_name!r} is missing from the active preprocessing spec")
    mode = str(source_spec.get("mode", "")).lower()
    if mode not in SUPPORTED_SOURCE_MODES:
        raise ValueError(
            f"Source {source_name!r} uses unsupported mode={mode!r}. "
            f"Supported modes: {sorted(SUPPORTED_SOURCE_MODES)}"
        )
    channel_names = tuple(str(n) for n in source_spec.get("output_channel_names", ()))
    if len(channel_names) != 1:
        raise ValueError(
            f"Source {source_name!r} must resolve to exactly one input channel, got {channel_names!r}"
        )
    channel_name = channel_names[0]
    if channel_name not in raster_input_channel_names:
        raise ValueError(
            f"Source channel {channel_name!r} is not present in raster inputs "
            f"{raster_input_channel_names!r}"
        )
    return raster_input_channel_names.index(channel_name), channel_name, source_spec


def _physical_to_processed_tensor(
    values: torch.Tensor,
    valid_mask: torch.Tensor,
    source_spec: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    mode = str(source_spec["mode"]).lower()
    valid = valid_mask.bool() & torch.isfinite(values)

    if mode == "pass_through":
        prepared = values.clone()
    elif mode == "divide":
        prepared = torch.zeros_like(values)
        prepared[valid] = values[valid] / float(source_spec["divisor"])
    elif mode == "zscore":
        prepared = values.clone()
    elif mode == "log_zscore":
        domain_mask = values > -1.0
        valid = valid & domain_mask
        prepared = torch.zeros_like(values)
        prepared[valid] = torch.log1p(values[valid]) / math.log(10.0)
    else:
        raise ValueError(f"Unsupported physical-to-processed mode: {mode!r}")

    if mode in {"zscore", "log_zscore"}:
        stats = source_spec.get("stats")
        if not stats:
            raise ValueError(f"Missing preprocessing stats for source {source_spec.get('source_name')!r}")
        mean = float(stats["mean"])
        std = float(stats["std"])
        output = torch.zeros_like(values)
        output[valid] = (prepared[valid] - mean) / std
        prepared = output

    prepared = prepared.to(dtype=torch.float32)
    prepared[~valid] = 0.0
    return prepared, valid


def _city_or_global_mask(
    *,
    valid_mask: torch.Tensor,
    city_mask: torch.Tensor,
    mask_scope: str,
) -> torch.Tensor:
    mask = valid_mask.bool()
    if mask_scope == "all":
        return mask
    expanded_city = city_mask > 0
    while expanded_city.dim() < mask.dim():
        expanded_city = expanded_city.unsqueeze(1)
    return mask & expanded_city.expand_as(mask)


def _perturb_raster_input_channel(
    *,
    raster_inputs: torch.Tensor,
    raster_valid_mask: torch.Tensor,
    city_mask: torch.Tensor,
    channel_index: int,
    source_spec: dict[str, Any],
    delta: float,
    mask_scope: str,
    min_value: float | None,
    max_value: float | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    selected = raster_inputs[:, channel_index : channel_index + 1, :, :]
    selected_valid = raster_valid_mask[:, channel_index : channel_index + 1, :, :]
    source_before = inverse_processed_tensor(selected, source_spec)
    perturbation_mask = _city_or_global_mask(
        valid_mask=selected_valid,
        city_mask=city_mask,
        mask_scope=mask_scope,
    )

    source_after = source_before.clone()
    source_after[perturbation_mask] += float(delta)
    if min_value is not None or max_value is not None:
        source_after = torch.clamp(source_after, min=min_value, max=max_value)

    processed_after, valid_after = _physical_to_processed_tensor(
        source_after, selected_valid, source_spec
    )

    perturbed_inputs = raster_inputs.clone()
    perturbed_inputs[:, channel_index : channel_index + 1, :, :] = processed_after
    return perturbed_inputs, valid_after, source_before, source_after, perturbation_mask


# ---------------------------------------------------------------------------
# Batch filtering
# ---------------------------------------------------------------------------

def _find_scene_year_indices(
    meta: list[dict[str, Any]],
    scene_id: str,
    year: int,
) -> list[int]:
    return [
        i for i, m in enumerate(meta)
        if str(m.get("scene_id", "")) == scene_id and int(m.get("year", -1)) == year
    ]


def _target_kind(batch_schema: Any, target_name: str) -> str:
    if any(target.name == target_name for target in batch_schema.raster_targets):
        return "raster"
    if any(target.name == target_name for target in batch_schema.scalar_targets):
        return "scalar"
    raise ValueError(f"Target {target_name!r} is missing from the active batch schema")


def _squeeze_channel(arr: np.ndarray) -> np.ndarray:
    while arr.ndim > 2:
        arr = arr.squeeze(0)
    return arr


def _scalar_stat(arr: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float | None]:
    flat = arr.ravel()
    if mask is not None and mask.size == arr.size:
        flat = flat[mask.ravel().astype(bool)]
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None, "std": None}
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "std": float(np.std(finite)),
    }


def _stat_values(values: np.ndarray) -> dict[str, int | float | None]:
    finite = np.asarray(values).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None, "std": None}
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "std": float(np.std(finite)),
    }


def _write_bulk_raster_stats_csv(
    *,
    output_payload: dict[str, np.ndarray],
    stats_csv_path: Path,
    target_name: str,
    source_name: str,
    deltas: np.ndarray,
) -> dict[str, Any]:
    """Write per-city raster response stats; positive cooling means target decreased."""
    fieldnames = [
        "scene_id",
        "year",
        "split",
        "crop_top",
        "crop_left",
        "crop_bottom",
        "crop_right",
        "delta",
        "array_key",
        "city_pixel_count",
        "mean_source_delta_city",
        "mean_target_before_city",
        "mean_target_after_city",
        "mean_delta_target_city",
        "mean_cooling_city",
        "median_cooling_city",
        "min_cooling_city",
        "max_cooling_city",
        "std_cooling_city",
    ]
    stats_csv_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate: dict[str, Any] = {
        "definition": {
            "delta_target": "target_after - target_before",
            "cooling": "target_before - target_after; positive means lower prediction after perturbation",
            "scope": "city_mask pixels with finite before/after target predictions",
        },
        "per_delta": {},
    }

    with stats_csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for delta in deltas:
            delta_float = float(delta)
            array_key = _bulk_key(target_name, source_name, delta_float)
            if array_key not in output_payload:
                continue

            stack = output_payload[array_key]
            city_mean_cooling: list[float] = []
            city_mean_delta: list[float] = []
            weighted_cooling_sum = 0.0
            weighted_delta_sum = 0.0
            total_city_pixels = 0

            for idx in range(int(stack.shape[0])):
                crop_box = output_payload["crop_boxes"][idx]
                source_before = stack[idx, 0]
                source_after = stack[idx, 1]
                target_before = stack[idx, 2]
                target_after = stack[idx, 3]
                city_mask = stack[idx, 4] > 0
                valid = city_mask & np.isfinite(target_before) & np.isfinite(target_after)
                pixel_count = int(np.count_nonzero(valid))

                row: dict[str, Any] = {
                    "scene_id": str(output_payload["scene_ids"][idx]),
                    "year": int(output_payload["years"][idx]),
                    "split": str(output_payload["splits"][idx]),
                    "crop_top": int(crop_box[0]),
                    "crop_left": int(crop_box[1]),
                    "crop_bottom": int(crop_box[2]),
                    "crop_right": int(crop_box[3]),
                    "delta": delta_float,
                    "array_key": array_key,
                    "city_pixel_count": pixel_count,
                    "mean_source_delta_city": "",
                    "mean_target_before_city": "",
                    "mean_target_after_city": "",
                    "mean_delta_target_city": "",
                    "mean_cooling_city": "",
                    "median_cooling_city": "",
                    "min_cooling_city": "",
                    "max_cooling_city": "",
                    "std_cooling_city": "",
                }
                if pixel_count > 0:
                    source_delta_values = (source_after - source_before)[valid]
                    before_values = target_before[valid]
                    after_values = target_after[valid]
                    delta_values = after_values - before_values
                    cooling_values = before_values - after_values
                    cooling_stats = _stat_values(cooling_values)
                    mean_delta = float(np.mean(delta_values))
                    mean_cooling = float(cooling_stats["mean"])

                    row.update(
                        {
                            "mean_source_delta_city": float(np.mean(source_delta_values)),
                            "mean_target_before_city": float(np.mean(before_values)),
                            "mean_target_after_city": float(np.mean(after_values)),
                            "mean_delta_target_city": mean_delta,
                            "mean_cooling_city": mean_cooling,
                            "median_cooling_city": cooling_stats["median"],
                            "min_cooling_city": cooling_stats["min"],
                            "max_cooling_city": cooling_stats["max"],
                            "std_cooling_city": cooling_stats["std"],
                        }
                    )
                    city_mean_cooling.append(mean_cooling)
                    city_mean_delta.append(mean_delta)
                    weighted_cooling_sum += float(np.sum(cooling_values))
                    weighted_delta_sum += float(np.sum(delta_values))
                    total_city_pixels += pixel_count

                writer.writerow(row)

            aggregate["per_delta"][f"{delta_float:g}"] = {
                "array_key": array_key,
                "city_count": int(stack.shape[0]),
                "valid_city_count": int(len(city_mean_cooling)),
                "city_pixel_count": int(total_city_pixels),
                "city_mean_cooling": _stat_values(np.asarray(city_mean_cooling, dtype=np.float32)),
                "city_mean_delta_target": _stat_values(np.asarray(city_mean_delta, dtype=np.float32)),
                "pixel_weighted_mean_cooling": (
                    None if total_city_pixels <= 0 else float(weighted_cooling_sum / total_city_pixels)
                ),
                "pixel_weighted_mean_delta_target": (
                    None if total_city_pixels <= 0 else float(weighted_delta_sum / total_city_pixels)
                ),
            }

    return aggregate


def _build_model_and_load_checkpoint(
    *,
    hydra_cfg: dict[str, Any],
    batch_schema: Any,
    target_schema: Any,
    device: torch.device,
) -> tuple[torch.nn.Module, str, str, str]:
    model = build_eump_downstream_model(
        model_config=dict(hydra_cfg["model"]),
        batch_schema=batch_schema,
        target_schema=target_schema,
        base_swin_path=str(Path(MODEL_DIR) / str(hydra_cfg["experiment"]["pretrain_model"])),
        encoder_source=str(hydra_cfg["downstream"]["encoder_source"]),
        encoder_checkpoint_path=resolve_model_artifact_path(
            str(hydra_cfg["downstream"]["encoder_model_name"]),
            str(hydra_cfg["downstream"]["encoder_checkpoint_file"]),
        ),
        load_pretrained=not bool(hydra_cfg["experiment"]["scratch_backbone"]),
        freeze_encoder=bool(hydra_cfg["downstream"]["freeze_encoder"]),
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
    checkpoint_path = load_model_state_for_evaluation(
        model=model,
        model_name=str(hydra_cfg["experiment"]["model_name"]),
        checkpoint_source=checkpoint_source,
        checkpoint_file=checkpoint_file,
        map_location=device,
    )
    model.eval()
    return model, str(checkpoint_path), checkpoint_source, checkpoint_file


def _selected_subset_loader(
    base_loader: DataLoader,
    indices: list[int],
    *,
    batch_size: int,
) -> DataLoader | None:
    if not indices:
        return None
    dataset = Subset(base_loader.dataset, indices)
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "shuffle": False,
        "pin_memory": bool(getattr(base_loader, "pin_memory", False)),
        "num_workers": int(getattr(base_loader, "num_workers", 0)),
        "collate_fn": base_loader.collate_fn,
        "worker_init_fn": getattr(base_loader, "worker_init_fn", None),
    }
    if kwargs["num_workers"] > 0:
        prefetch_factor = getattr(base_loader, "prefetch_factor", None)
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = prefetch_factor
        kwargs["persistent_workers"] = bool(getattr(base_loader, "persistent_workers", False))
    return DataLoader(**kwargs)


def _scene_sort_key(entry: dict[str, Any]) -> tuple[int, int | str, str]:
    scene_id = str(entry["scene_id"])
    try:
        scene_key: int | str = int(scene_id)
        is_text = 0
    except ValueError:
        scene_key = scene_id
        is_text = 1
    return is_text, scene_key, str(entry["split"])


def _collect_bulk_entries(
    loaders: dict[str, DataLoader],
    *,
    year: int,
    limit_cities: int | None,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for split_name in ("train", "val", "test"):
        dataset = loaders[split_name].dataset
        rows = getattr(dataset, "rows", None)
        if rows is None:
            raise TypeError(f"Expected split dataset {split_name!r} to expose rows")
        for index, row in enumerate(rows):
            row_year = int(row["year"])
            scene_id = str(row["scene_id"])
            if row_year != int(year):
                continue
            key = (scene_id, row_year)
            if key in seen:
                continue
            seen.add(key)
            entries.append(
                {
                    "split": split_name,
                    "index": int(index),
                    "scene_id": scene_id,
                    "year": row_year,
                }
            )
    entries.sort(key=_scene_sort_key)
    if limit_cities is not None:
        limit = int(limit_cities)
        if limit <= 0:
            raise ValueError("--limit-cities must be > 0 when provided")
        entries = entries[:limit]
    for order_index, entry in enumerate(entries):
        entry["order_index"] = int(order_index)
    return entries


def _find_scene_year_entry(
    loaders: dict[str, DataLoader],
    *,
    scene_id: str,
    year: int,
    search_order: list[str],
) -> tuple[str, int] | None:
    target_scene_id = str(scene_id)
    target_year = int(year)
    for split_name in search_order:
        dataset = loaders[split_name].dataset
        rows = getattr(dataset, "rows", None)
        if rows is None:
            raise TypeError(f"Expected split dataset {split_name!r} to expose rows")
        for index, row in enumerate(rows):
            if str(row["scene_id"]) == target_scene_id and int(row["year"]) == target_year:
                return split_name, int(index)
    return None


def _bulk_key(target_name: str, source_name: str, delta: float) -> str:
    return f"{target_name.lower()}_{source_name.lower()}{_delta_key_suffix(delta)}"


def _write_rank_shard(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(path), **payload)


def _concat_chunks(chunks: list[np.ndarray], *, shape_tail: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    if chunks:
        return np.concatenate(chunks, axis=0)
    return np.empty((0, *shape_tail), dtype=dtype)


def _merge_bulk_shards(
    *,
    shard_paths: list[Path],
    output_path: Path,
    summary_path: Path,
    summary: dict[str, Any],
    keep_shards: bool = False,
) -> None:
    loaded = [np.load(str(path), allow_pickle=False) for path in shard_paths]
    try:
        scene_ids = np.concatenate([item["scene_ids"] for item in loaded], axis=0)
        years = np.concatenate([item["years"] for item in loaded], axis=0)
        splits = np.concatenate([item["splits"] for item in loaded], axis=0)
        crop_boxes = np.concatenate([item["crop_boxes"] for item in loaded], axis=0)
        order_indices = np.concatenate([item["order_indices"] for item in loaded], axis=0)
        deltas = loaded[0]["deltas"] if loaded else np.asarray([], dtype=np.float32)
        channel_names = loaded[0]["channel_names"] if loaded else np.asarray(BULK_CHANNEL_NAMES)
        keys = [
            key
            for key in loaded[0].files
            if key
            not in {
                "scene_ids",
                "years",
                "splits",
                "crop_boxes",
                "order_indices",
                "deltas",
                "channel_names",
            }
        ] if loaded else []

        order = np.argsort(order_indices, kind="stable")
        output_payload: dict[str, np.ndarray] = {
            "scene_ids": scene_ids[order],
            "years": years[order],
            "splits": splits[order],
            "crop_boxes": crop_boxes[order],
            "deltas": deltas,
            "channel_names": channel_names,
        }
        for key in keys:
            output_payload[key] = np.concatenate([item[key] for item in loaded], axis=0)[order]

        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(output_path), **output_payload)
        stats_csv_path: Path | None = None
        if str(summary.get("target", {}).get("kind")) == "raster":
            target_name = str(summary.get("target", {}).get("name"))
            source_name = str(summary.get("perturbation", {}).get("source_name"))
            stats_csv_path = output_path.with_name(f"{output_path.stem}_stats.csv")
            summary["stats"] = _write_bulk_raster_stats_csv(
                output_payload=output_payload,
                stats_csv_path=stats_csv_path,
                target_name=target_name,
                source_name=source_name,
                deltas=deltas,
            )
        summary["outputs"] = {
            "npz": str(output_path),
            "stats_csv": None if stats_csv_path is None else str(stats_csv_path),
            "summary": str(summary_path),
        }
        summary["data"]["total_cities"] = int(scene_ids.shape[0])
        summary_path.write_text(json.dumps(_jsonable(summary), indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    finally:
        for item in loaded:
            item.close()

    if not keep_shards:
        shard_dir = shard_paths[0].parent if shard_paths else None
        if shard_dir is not None and shard_dir.exists():
            shutil.rmtree(shard_dir)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _save_delta_map(
    *,
    source_name: str,
    target_name: str,
    delta: float,
    scene_id: str,
    year: int,
    input_before: np.ndarray,
    input_after: np.ndarray,
    delta_pred: np.ndarray,
    city_mask: np.ndarray,
    output_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Squeeze batch and channel dims to [H, W]
    def _squeeze(arr: np.ndarray) -> np.ndarray:
        while arr.ndim > 2:
            arr = arr.squeeze(0)
        return arr

    before_2d = _squeeze(input_before)
    after_2d = _squeeze(input_after)
    delta_2d = _squeeze(delta_pred)
    city_2d = _squeeze(city_mask).astype(bool)

    # Mask non-city pixels in the delta map for display
    delta_display = np.where(city_2d, delta_2d, np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"scene_id={scene_id}  year={year}  source={source_name}  "
        f"delta={delta:+g}  target={target_name}",
        fontsize=11,
    )

    # Panel 1: input before
    im0 = axes[0].imshow(before_2d, cmap="YlGn", aspect="auto")
    axes[0].contour(city_2d, levels=[0.5], colors="k", linewidths=0.8)
    axes[0].set_title(f"{source_name} before (physical)")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Panel 2: input after
    im1 = axes[1].imshow(after_2d, cmap="YlGn", aspect="auto")
    axes[1].contour(city_2d, levels=[0.5], colors="k", linewidths=0.8)
    axes[1].set_title(f"{source_name} after (+{delta:g})")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: delta prediction
    vmax = float(np.nanpercentile(np.abs(delta_display[city_2d]) if city_2d.any() else [0.0], 95) or 1.0)
    im2 = axes[2].imshow(delta_display, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    axes[2].contour(city_2d, levels=[0.5], colors="k", linewidths=0.8)
    axes[2].set_title(f"Δ{target_name} (after − before, physical)")
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# JSON helper
# ---------------------------------------------------------------------------

def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

def _run_single_scene(args: argparse.Namespace, cfg_merged: dict[str, Any]) -> int:
    target_scene_id = str(cfg_merged["scene_id"])
    target_year = int(cfg_merged["year"])
    source_name = str(cfg_merged["source"])
    delta = float(cfg_merged["delta"])
    mask_scope = str(cfg_merged["mask_scope"])
    min_value, max_value = _resolve_source_bounds(
        source_name,
        cfg_merged.get("min_value"),
        cfg_merged.get("max_value"),
    )
    split = str(cfg_merged["split"])

    output_dir = _resolve_output_base(cfg_merged["output_dir"]) / f"{target_scene_id}_{target_year}"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(str(args.device))
    hydra_cfg = _compose_hydra_config(args, num_workers=int(args.num_workers))
    _apply_checkpoint_overrides(hydra_cfg, cfg_merged)
    hydra_cfg["pipeline"]["year_range"] = [target_year, target_year]
    hydra_cfg["evaluation"]["split"] = split

    logger = setup_logger(log_dir=None, rank=0, log_level=str(hydra_cfg["logging"]["level"]))
    _configure_torch_multiprocessing(logger, rank=0)
    logger.info(
        f"manipulate_exp: scene_id={target_scene_id} year={target_year} "
        f"source={source_name} delta={delta:+g} split={split} "
        f"device={device} center_ratio=1.0 year_range=[{target_year}, {target_year}]"
    )

    pipeline_spec = PipelineSpec.from_mapping(hydra_cfg["pipeline"])
    validate_eump_downstream_spec(pipeline_spec)

    train_dl, val_dl, test_dl, batch_schema = build_eump_dataloaders(
        data_config=hydra_cfg["data"],
        pipeline_spec=pipeline_spec,
        logger=logger,
        random_seed=int(hydra_cfg["experiment"]["random_seed"]),
        batch_size=int(args.batch_size),
        local_rank=0,
        world_size=1,
        is_distributed=False,
        shuffle_train=False,
    )
    loader_map = {"train": train_dl, "val": val_dl, "test": test_dl}
    preprocessing_spec = loader_map[split].dataset.preprocessing_spec

    channel_index, channel_name, source_spec = _resolve_source_channel(
        source_name=source_name,
        preprocessing_spec=preprocessing_spec,
        raster_input_channel_names=batch_schema.raster_input_channel_names,
    )
    logger.info(
        f"Resolved source {source_name!r} -> channel {channel_name!r} "
        f"(index {channel_index}), physical clamp=[{min_value}, {max_value}]"
    )

    target_schema = get_single_downstream_target_schema(batch_schema)
    target_name = target_schema.name
    target_kind = _target_kind(batch_schema, target_name)
    target_source_name = target_schema.spec.source
    if target_source_name is None:
        raise ValueError(f"Downstream target {target_name!r} has no source name in the pipeline spec")
    target_source_spec = preprocessing_spec["sources"][target_source_name]

    model, checkpoint_path, checkpoint_source, _ = _build_model_and_load_checkpoint(
        hydra_cfg=hydra_cfg,
        batch_schema=batch_schema,
        target_schema=target_schema,
        device=device,
    )
    logger.info(f"Loaded checkpoint: {checkpoint_path}")

    search_order = [split] + [s for s in ("test", "train", "val") if s != split]
    located = _find_scene_year_entry(
        loader_map,
        scene_id=target_scene_id,
        year=target_year,
        search_order=search_order,
    )
    if located is None:
        raise ValueError(
            f"scene_id={target_scene_id!r} year={target_year} not found in any split "
            f"(searched: {search_order}). Check --scene-id, --year, and the pipeline year_range."
        )
    located_split, located_index = located
    if located_split != split:
        logger.info(
            f"scene_id={target_scene_id} year={target_year} not found "
            f"in split={split!r}; found in split={located_split!r} instead"
        )
    logger.info(f"Found scene_id={target_scene_id} year={target_year} in split={located_split}")

    result: dict[str, np.ndarray] | None = None
    sample_meta: dict[str, Any] | None = None

    with torch.no_grad():
        sample_loader = _selected_subset_loader(
            loader_map[located_split],
            [located_index],
            batch_size=1,
        )
        if sample_loader is None:
            raise RuntimeError(f"Unable to create a one-sample loader for {located_split}:{located_index}")
        batch = next(iter(sample_loader))
        raster_inputs = batch.raster_inputs.to(device)
        scalar_inputs = batch.scalar_inputs.to(device)
        raster_valid = batch.raster_valid_mask.to(device)
        city_mask = batch.city_mask.to(device)

        perturbed_inputs, _, src_before, src_after, pert_mask = _perturb_raster_input_channel(
            raster_inputs=raster_inputs,
            raster_valid_mask=raster_valid,
            city_mask=city_mask,
            channel_index=channel_index,
            source_spec=source_spec,
            delta=delta,
            mask_scope=mask_scope,
            min_value=min_value,
            max_value=max_value,
        )

        raster_pred_before, scalar_pred_before = model(raster_inputs, scalar_inputs)
        raster_pred_after, scalar_pred_after = model(perturbed_inputs, scalar_inputs)

        if target_kind == "raster":
            pred_before = inverse_processed_tensor(
                raster_pred_before[target_name], target_source_spec
            )
            pred_after = inverse_processed_tensor(
                raster_pred_after[target_name], target_source_spec
            )
            target_block = batch.raster_targets[target_name]
            target_data = target_block.data.to(device)
            target_valid = target_block.valid_mask.to(device)
            target_physical = inverse_processed_tensor(target_data, target_source_spec)
        else:
            pred_before = inverse_processed_tensor(
                scalar_pred_before[target_name], target_source_spec
            )
            pred_after = inverse_processed_tensor(
                scalar_pred_after[target_name], target_source_spec
            )
            target_block = batch.scalar_targets[target_name]
            target_data = target_block.data.to(device)
            target_valid = target_block.valid_mask.to(device)
            target_physical = inverse_processed_tensor(target_data, target_source_spec)

        delta_pred = pred_after - pred_before
        result = {
            "delta_prediction": delta_pred.detach().cpu().numpy(),
            "pred_before": pred_before.detach().cpu().numpy(),
            "pred_after": pred_after.detach().cpu().numpy(),
            "input_before": src_before.detach().cpu().numpy(),
            "input_after": src_after.detach().cpu().numpy(),
            "perturbation_mask": pert_mask.detach().cpu().numpy().astype(np.uint8),
            "city_mask": city_mask.detach().cpu().numpy(),
            "target": target_physical.detach().cpu().numpy(),
            "target_valid_mask": target_valid.detach().cpu().numpy(),
        }
        sample_meta = _jsonable(batch.meta[0])
    split = str(located_split)

    stem = f"delta_{target_name}_{target_scene_id}_{target_year}"
    npz_path = output_dir / f"{stem}.npz"
    np.savez_compressed(str(npz_path), **result)
    logger.info(f"Saved arrays: {npz_path}")

    png_path: Path | None = None
    if target_kind == "raster":
        png_path = output_dir / f"{stem}.png"
        _save_delta_map(
            source_name=source_name,
            target_name=target_name,
            delta=delta,
            scene_id=target_scene_id,
            year=target_year,
            input_before=result["input_before"],
            input_after=result["input_after"],
            delta_pred=result["delta_prediction"],
            city_mask=result["city_mask"],
            output_path=png_path,
        )
        logger.info(f"Saved map:    {png_path}")

    stats = {
        "delta_prediction_global": _scalar_stat(result["delta_prediction"]),
        "input_before_city": _scalar_stat(result["input_before"], result["city_mask"]),
        "input_after_city": _scalar_stat(result["input_after"], result["city_mask"]),
    }
    if target_kind == "raster":
        stats["delta_prediction_city"] = _scalar_stat(result["delta_prediction"], result["city_mask"])
        stats["cooling_city"] = _scalar_stat(-result["delta_prediction"], result["city_mask"])
        stats["cooling_definition"] = (
            "cooling = pred_before - pred_after = -delta_prediction; "
            "positive means lower predicted target after perturbation"
        )

    summary = {
        "experiment": {
            "group": str(hydra_cfg["experiment"]["group"]),
            "name": str(hydra_cfg["experiment"]["name"]),
            "model_name": str(hydra_cfg["experiment"]["model_name"]),
            "pretrain_model": str(hydra_cfg["experiment"]["pretrain_model"]),
            "checkpoint_path": _jsonable(checkpoint_path),
            "checkpoint_source": checkpoint_source,
        },
        "data": {
            "split": split,
            "eump_root": str(hydra_cfg["data"]["eump_root"]),
            "preprocessing_path": str(hydra_cfg["data"]["preprocessing_path"]),
            "split_manifest_path": hydra_cfg["data"].get("split_manifest_path"),
            "pipeline_year_range": (
                list(pipeline_spec.year_range) if pipeline_spec.year_range is not None else None
            ),
            "center_ratio": float(pipeline_spec.center_ratio),
        },
        "perturbation": {
            "source_name": source_name,
            "source_channel_name": channel_name,
            "source_channel_index": channel_index,
            "source_mode": str(source_spec["mode"]),
            "delta": delta,
            "mask_scope": mask_scope,
            "min_value": min_value,
            "max_value": max_value,
            "scale": "physical",
        },
        "target": {
            "name": target_name,
            "kind": target_kind,
            "source": target_source_name,
        },
        "sample": sample_meta,
        "stats": stats,
        "outputs": {
            "npz": str(npz_path),
            "png": None if png_path is None else str(png_path),
        },
    }
    summary_path = output_dir / f"summary_{target_scene_id}_{target_year}.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    logger.info(f"Saved summary: {summary_path}")

    if target_kind == "raster":
        city_bool = result["city_mask"].ravel().astype(bool)
        if city_bool.size == result["delta_prediction"].size:
            city_cooling = (-result["delta_prediction"]).ravel()[city_bool]
            city_finite = city_cooling[np.isfinite(city_cooling)]
            if city_finite.size > 0:
                logger.info(
                    f"Cooling {target_name} inside city_mask (before-after): "
                    f"mean={float(np.mean(city_finite)):+.4f}  "
                    f"min={float(np.min(city_finite)):+.4f}  "
                    f"max={float(np.max(city_finite)):+.4f}"
                )
    else:
        logger.info(f"Delta {target_name} scalar: {result['delta_prediction'].reshape(-1).tolist()}")

    return 0


def _run_all_cities(args: argparse.Namespace, cfg_merged: dict[str, Any]) -> int:
    target_year = int(cfg_merged["year"])
    source_name = str(cfg_merged["source"])
    deltas = _parse_deltas(cfg_merged.get("deltas"))
    mask_scope = str(cfg_merged["mask_scope"])
    min_value, max_value = _resolve_source_bounds(
        source_name,
        cfg_merged.get("min_value"),
        cfg_merged.get("max_value"),
    )

    hydra_cfg = _compose_hydra_config(args, num_workers=int(args.num_workers))
    _apply_checkpoint_overrides(hydra_cfg, cfg_merged)
    hydra_cfg["pipeline"]["year_range"] = [target_year, target_year]
    hydra_cfg["evaluation"]["split"] = "test"

    distributed_requested = int(os.environ.get("WORLD_SIZE", "1")) > 1
    initialized_distributed = False
    if distributed_requested and not (dist.is_available() and dist.is_initialized()):
        init_distributed_process_group()
        initialized_distributed = True

    try:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        rank = get_rank()
        world_size = get_world_size()
        if distributed_requested and str(args.device).strip().lower() == "auto":
            device = resolve_distributed_device(local_rank)
        else:
            device = _resolve_device(str(args.device))

        logger = setup_logger(log_dir=None, rank=rank, log_level=str(hydra_cfg["logging"]["level"]))
        _configure_torch_multiprocessing(logger, rank=rank)
        if rank == 0:
            logger.info(
                f"manipulate_exp bulk: year={target_year} source={source_name} "
                f"deltas={','.join(f'{delta:g}' for delta in deltas)} "
                f"mask_scope={mask_scope} device={device} world_size={world_size}"
            )

        pipeline_spec = PipelineSpec.from_mapping(hydra_cfg["pipeline"])
        validate_eump_downstream_spec(pipeline_spec)

        train_dl, val_dl, test_dl, batch_schema = build_eump_dataloaders(
            data_config=hydra_cfg["data"],
            pipeline_spec=pipeline_spec,
            logger=logger,
            random_seed=int(hydra_cfg["experiment"]["random_seed"]),
            batch_size=int(args.batch_size),
            local_rank=local_rank,
            world_size=world_size,
            is_distributed=False,
            shuffle_train=False,
        )
        loaders = {"train": train_dl, "val": val_dl, "test": test_dl}
        preprocessing_spec = train_dl.dataset.preprocessing_spec

        channel_index, channel_name, source_spec = _resolve_source_channel(
            source_name=source_name,
            preprocessing_spec=preprocessing_spec,
            raster_input_channel_names=batch_schema.raster_input_channel_names,
        )

        target_schema = get_single_downstream_target_schema(batch_schema)
        target_name = target_schema.name
        target_kind = _target_kind(batch_schema, target_name)
        target_source_name = target_schema.spec.source
        if target_source_name is None:
            raise ValueError(f"Downstream target {target_name!r} has no source name in the pipeline spec")
        target_source_spec = preprocessing_spec["sources"][target_source_name]

        model, checkpoint_path, checkpoint_source, _ = _build_model_and_load_checkpoint(
            hydra_cfg=hydra_cfg,
            batch_schema=batch_schema,
            target_schema=target_schema,
            device=device,
        )
        if rank == 0:
            logger.info(f"Loaded checkpoint: {checkpoint_path}")
            logger.info(
                f"Resolved source {source_name!r} -> channel {channel_name!r} "
                f"(index {channel_index}), physical clamp=[{min_value}, {max_value}]"
            )

        all_entries = _collect_bulk_entries(
            loaders,
            year=target_year,
            limit_cities=args.limit_cities,
        )
        if not all_entries:
            raise ValueError(f"No compatible scene-year rows found for year={target_year}")
        selected_entries = all_entries[rank::world_size]
        indices_by_split: dict[str, list[int]] = {"train": [], "val": [], "test": []}
        order_by_sample: dict[tuple[str, str, int], int] = {}
        for entry in selected_entries:
            split_name = str(entry["split"])
            indices_by_split[split_name].append(int(entry["index"]))
            order_by_sample[(split_name, str(entry["scene_id"]), int(entry["year"]))] = int(entry["order_index"])

        output_dir = _resolve_output_base(cfg_merged["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        shard_dir = output_dir / "_shards"
        if rank == 0 and shard_dir.exists():
            shutil.rmtree(shard_dir)
        distributed_barrier(local_rank)
        shard_dir.mkdir(parents=True, exist_ok=True)

        crop_size = pipeline_spec.crop_size
        if isinstance(crop_size, tuple):
            crop_h, crop_w = int(crop_size[0]), int(crop_size[1])
        else:
            crop_h = crop_w = int(crop_size)

        meta_chunks: dict[str, list[np.ndarray]] = {
            "scene_ids": [],
            "years": [],
            "splits": [],
            "crop_boxes": [],
            "order_indices": [],
        }
        raster_chunks = {
            _bulk_key(target_name, source_name, delta): []
            for delta in deltas
        }
        scalar_chunks: dict[str, list[np.ndarray]] = {}
        if target_kind == "scalar":
            scalar_chunks[f"{target_name.lower()}0"] = []
            for delta in deltas:
                suffix_key = _bulk_key(target_name, source_name, delta)
                scalar_chunks[f"{suffix_key}_1"] = []
                scalar_chunks[f"{suffix_key}_delta"] = []

        with torch.no_grad():
            for split_name in ("train", "val", "test"):
                split_loader = _selected_subset_loader(
                    loaders[split_name],
                    indices_by_split[split_name],
                    batch_size=int(args.batch_size),
                )
                if split_loader is None:
                    continue
                for batch in split_loader:
                    batch_size = int(batch.raster_inputs.shape[0])
                    raster_inputs = batch.raster_inputs.to(device)
                    scalar_inputs = batch.scalar_inputs.to(device)
                    raster_valid = batch.raster_valid_mask.to(device)
                    city_mask = batch.city_mask.to(device)

                    raster_pred_before, scalar_pred_before = model(raster_inputs, scalar_inputs)
                    if target_kind == "raster":
                        pred_before = inverse_processed_tensor(
                            raster_pred_before[target_name], target_source_spec
                        )
                    else:
                        pred_before = inverse_processed_tensor(
                            scalar_pred_before[target_name], target_source_spec
                        )
                        scalar_chunks[f"{target_name.lower()}0"].append(
                            pred_before.detach().cpu().reshape(batch_size, -1).numpy().astype(np.float32)
                        )

                    scene_ids: list[str] = []
                    years: list[int] = []
                    splits: list[str] = []
                    crop_boxes: list[tuple[int, int, int, int]] = []
                    order_indices: list[int] = []
                    for meta in batch.meta:
                        scene_id = str(meta["scene_id"])
                        year = int(meta["year"])
                        scene_ids.append(scene_id)
                        years.append(year)
                        splits.append(split_name)
                        crop_boxes.append(tuple(int(value) for value in meta.get("crop_box", (0, 0, crop_h, crop_w))))
                        order_indices.append(order_by_sample[(split_name, scene_id, year)])
                    meta_chunks["scene_ids"].append(np.asarray(scene_ids))
                    meta_chunks["years"].append(np.asarray(years, dtype=np.int32))
                    meta_chunks["splits"].append(np.asarray(splits))
                    meta_chunks["crop_boxes"].append(np.asarray(crop_boxes, dtype=np.int32))
                    meta_chunks["order_indices"].append(np.asarray(order_indices, dtype=np.int64))

                    for delta in deltas:
                        perturbed_inputs, _, src_before, src_after, _ = _perturb_raster_input_channel(
                            raster_inputs=raster_inputs,
                            raster_valid_mask=raster_valid,
                            city_mask=city_mask,
                            channel_index=channel_index,
                            source_spec=source_spec,
                            delta=delta,
                            mask_scope=mask_scope,
                            min_value=min_value,
                            max_value=max_value,
                        )
                        raster_pred_after, scalar_pred_after = model(perturbed_inputs, scalar_inputs)
                        if target_kind == "raster":
                            pred_after = inverse_processed_tensor(
                                raster_pred_after[target_name], target_source_spec
                            )
                            stacked = torch.cat(
                                [
                                    src_before,
                                    src_after,
                                    pred_before,
                                    pred_after,
                                    (city_mask > 0).to(dtype=torch.float32),
                                ],
                                dim=1,
                            )
                            raster_chunks[_bulk_key(target_name, source_name, delta)].append(
                                stacked.detach().cpu().numpy().astype(np.float32, copy=False)
                            )
                        else:
                            pred_after = inverse_processed_tensor(
                                scalar_pred_after[target_name], target_source_spec
                            )
                            suffix_key = _bulk_key(target_name, source_name, delta)
                            pred_after_np = pred_after.detach().cpu().reshape(batch_size, -1).numpy().astype(np.float32)
                            pred_delta_np = (
                                pred_after - pred_before
                            ).detach().cpu().reshape(batch_size, -1).numpy().astype(np.float32)
                            scalar_chunks[f"{suffix_key}_1"].append(pred_after_np)
                            scalar_chunks[f"{suffix_key}_delta"].append(pred_delta_np)

        shard_payload: dict[str, np.ndarray] = {
            "scene_ids": _concat_chunks(meta_chunks["scene_ids"], shape_tail=(), dtype=np.dtype("<U32")),
            "years": _concat_chunks(meta_chunks["years"], shape_tail=(), dtype=np.dtype(np.int32)),
            "splits": _concat_chunks(meta_chunks["splits"], shape_tail=(), dtype=np.dtype("<U8")),
            "crop_boxes": _concat_chunks(meta_chunks["crop_boxes"], shape_tail=(4,), dtype=np.dtype(np.int32)),
            "order_indices": _concat_chunks(
                meta_chunks["order_indices"],
                shape_tail=(),
                dtype=np.dtype(np.int64),
            ),
            "deltas": np.asarray(deltas, dtype=np.float32),
            "channel_names": np.asarray(BULK_CHANNEL_NAMES if target_kind == "raster" else (), dtype="<U16"),
        }
        if target_kind == "raster":
            for key, chunks in raster_chunks.items():
                shard_payload[key] = _concat_chunks(
                    chunks,
                    shape_tail=(5, crop_h, crop_w),
                    dtype=np.dtype(np.float32),
                )
        else:
            for key, chunks in scalar_chunks.items():
                shard_payload[key] = _concat_chunks(chunks, shape_tail=(1,), dtype=np.dtype(np.float32))

        shard_path = shard_dir / f"rank{rank:05d}-of-{world_size:05d}.npz"
        _write_rank_shard(shard_path, shard_payload)
        logger.info(f"Rank {rank} wrote shard: {shard_path} ({int(shard_payload['scene_ids'].shape[0])} rows)")
        distributed_barrier(local_rank)

        if rank == 0:
            shard_paths = [shard_dir / f"rank{idx:05d}-of-{world_size:05d}.npz" for idx in range(world_size)]
            missing = [str(path) for path in shard_paths if not path.exists()]
            if missing:
                raise FileNotFoundError(f"Missing bulk export shard(s): {missing}")
            output_path = output_dir / f"all_cities_{source_name.lower()}_{target_name.lower()}_{target_year}.npz"
            summary_path = output_dir / "summary.json"
            summary = {
                "experiment": {
                    "group": str(hydra_cfg["experiment"]["group"]),
                    "name": str(hydra_cfg["experiment"]["name"]),
                    "model_name": str(hydra_cfg["experiment"]["model_name"]),
                    "pretrain_model": str(hydra_cfg["experiment"]["pretrain_model"]),
                    "checkpoint_path": checkpoint_path,
                    "checkpoint_source": checkpoint_source,
                },
                "data": {
                    "year": target_year,
                    "eump_root": str(hydra_cfg["data"]["eump_root"]),
                    "preprocessing_path": str(hydra_cfg["data"]["preprocessing_path"]),
                    "split_manifest_path": hydra_cfg["data"].get("split_manifest_path"),
                    "pipeline_year_range": list(pipeline_spec.year_range),
                    "center_ratio": float(pipeline_spec.center_ratio),
                    "total_cities": 0,
                },
                "perturbation": {
                    "source_name": source_name,
                    "source_channel_name": channel_name,
                    "source_channel_index": channel_index,
                    "source_mode": str(source_spec["mode"]),
                    "deltas": list(deltas),
                    "mask_scope": mask_scope,
                    "min_value": min_value,
                    "max_value": max_value,
                    "scale": "physical",
                },
                "target": {
                    "name": target_name,
                    "kind": target_kind,
                    "source": target_source_name,
                },
                "runtime": {
                    "distributed": world_size > 1,
                    "world_size": int(world_size),
                    "batch_size_per_rank": int(args.batch_size),
                    "num_workers_per_rank": int(getattr(train_dl, "num_workers", 0)),
                    "limit_cities": args.limit_cities,
                    "selected_total_before_limit": int(len(all_entries)),
                },
            }
            _merge_bulk_shards(
                shard_paths=shard_paths,
                output_path=output_path,
                summary_path=summary_path,
                summary=summary,
            )
            logger.info(f"Merged bulk output: {output_path}")
        distributed_barrier(local_rank)
    finally:
        if initialized_distributed:
            destroy_distributed_process_group()

    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    exp_cfg = _load_exp_config(args.config_path)
    cfg_merged = _merge_args(exp_cfg, args)

    if bool(args.all_cities):
        return _run_all_cities(args, cfg_merged)
    return _run_single_scene(args, cfg_merged)


if __name__ == "__main__":
    raise SystemExit(main())
