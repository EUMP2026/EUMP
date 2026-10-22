#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

from hydra import compose, initialize_config_dir
import numpy as np
import torch


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
from urban_embedding.training.regression_metrics import inverse_processed_tensor


CONFIG_DIR = str(PROJECT_ROOT / "configs")
SUPPORTED_SOURCE_MODES = {"pass_through", "divide", "zscore", "log_zscore"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a simple downstream EUMP input-counterfactual experiment by "
            "perturbing one continuous raster input source in physical space and "
            "saving before/after predictions."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for summary.json and results.npz.",
    )
    parser.add_argument(
        "--source",
        default="NDVI",
        help="Continuous source name to perturb. Default: NDVI.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.05,
        help="Additive perturbation applied in physical source units. Default: 0.05.",
    )
    parser.add_argument(
        "--mask-scope",
        choices=("city", "all"),
        default="city",
        help="Apply perturbation only in city_mask or across the full valid crop. Default: city.",
    )
    parser.add_argument(
        "--min-value",
        type=float,
        default=None,
        help="Optional lower clamp in physical units. Default: source-specific fallback for NDVI.",
    )
    parser.add_argument(
        "--max-value",
        type=float,
        default=None,
        help="Optional upper clamp in physical units. Default: source-specific fallback for NDVI.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=8,
        help="Maximum number of samples to save. Default: 8.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Counterfactual batch size. Default: 2.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers for this script. Default: 0 for portability.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default=None,
        help="Override evaluation.split for this script.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device: auto, cpu, cuda, cuda:0, etc. Default: auto.",
    )
    parser.add_argument(
        "--config-name",
        default="evaluate",
        help="Hydra config name to compose. Default: evaluate.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help=(
            "Optional Hydra overrides, for example "
            "pipeline=eump_full/mae_v3/lst experiment.group=eump_full_mae_v3."
        ),
    )
    return parser.parse_args(argv)


def _compose_config(args: argparse.Namespace) -> dict[str, Any]:
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name=args.config_name, overrides=args.overrides)
    normalized = normalize_evaluation_config(cfg)
    if args.split is not None:
        normalized["evaluation"]["split"] = args.split
    normalized["pipeline"]["num_workers"] = int(args.num_workers)
    return normalized


def _resolve_device(requested: str) -> torch.device:
    value = str(requested).strip().lower()
    if value == "auto":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested CUDA device {requested!r}, but CUDA is unavailable")
    return device


def _resolve_source_bounds(source_name: str, args: argparse.Namespace) -> tuple[float | None, float | None]:
    if args.min_value is not None or args.max_value is not None:
        return args.min_value, args.max_value
    if str(source_name).upper() == "NDVI":
        return -1.0, 1.0
    return None, None


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
            f"Counterfactual source {source_name!r} uses unsupported mode={mode!r}; "
            f"expected one of {sorted(SUPPORTED_SOURCE_MODES)}"
        )
    channel_names = tuple(str(name) for name in source_spec.get("output_channel_names", ()))
    if len(channel_names) != 1:
        raise ValueError(
            f"Counterfactual source {source_name!r} must resolve to exactly one input channel, got {channel_names!r}"
        )
    channel_name = channel_names[0]
    if channel_name not in raster_input_channel_names:
        raise ValueError(
            f"Counterfactual source channel {channel_name!r} is not present in raster inputs "
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
    selected_inputs = raster_inputs[:, channel_index : channel_index + 1, :, :]
    selected_valid = raster_valid_mask[:, channel_index : channel_index + 1, :, :]
    source_before_physical = inverse_processed_tensor(selected_inputs, source_spec)
    perturbation_mask = _city_or_global_mask(
        valid_mask=selected_valid,
        city_mask=city_mask,
        mask_scope=mask_scope,
    )

    source_after_physical = source_before_physical.clone()
    source_after_physical[perturbation_mask] += float(delta)
    if min_value is not None or max_value is not None:
        lower = min_value if min_value is not None else None
        upper = max_value if max_value is not None else None
        source_after_physical = torch.clamp(source_after_physical, min=lower, max=upper)

    processed_after, valid_after = _physical_to_processed_tensor(
        source_after_physical,
        selected_valid,
        source_spec,
    )

    perturbed_inputs = raster_inputs.clone()
    perturbed_inputs[:, channel_index : channel_index + 1, :, :] = processed_after
    return (
        perturbed_inputs,
        valid_after,
        source_before_physical,
        source_after_physical,
        perturbation_mask,
    )


def _sample_city_mean(
    *,
    values: torch.Tensor,
    valid_mask: torch.Tensor,
    city_mask: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    expanded_city = city_mask > 0
    while expanded_city.dim() < values.dim():
        expanded_city = expanded_city.unsqueeze(1)
    mask = valid_mask.bool() & expanded_city.expand_as(values) & torch.isfinite(values)
    weights = mask.to(dtype=values.dtype)
    support = weights.sum(dim=(-2, -1))
    sums = (values * weights).sum(dim=(-2, -1))
    mean = torch.zeros_like(sums)
    positive = support > 0
    mean[positive] = sums[positive] / support[positive]
    return (
        mean.detach().cpu().numpy(),
        positive.detach().cpu().numpy(),
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _target_kind(batch_schema, target_name: str) -> str:
    if any(target.name == target_name for target in batch_schema.raster_targets):
        return "raster"
    if any(target.name == target_name for target in batch_schema.scalar_targets):
        return "scalar"
    raise ValueError(f"Target {target_name!r} is missing from the batch schema")


def _select_loader(split: str, train_dl, val_dl, test_dl):
    if split == "train":
        return train_dl
    if split == "val":
        return val_dl
    return test_dl


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if int(args.limit) <= 0:
        raise ValueError("--limit must be > 0")
    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be > 0")

    cfg = _compose_config(args)
    device = _resolve_device(args.device)
    logger = setup_logger(log_dir=None, rank=0, log_level=str(cfg["logging"]["level"]))

    pipeline_spec = PipelineSpec.from_mapping(cfg["pipeline"])
    validate_eump_downstream_spec(pipeline_spec)

    train_dl, val_dl, test_dl, batch_schema = build_eump_dataloaders(
        data_config=cfg["data"],
        pipeline_spec=pipeline_spec,
        logger=logger,
        random_seed=int(cfg["experiment"]["random_seed"]),
        batch_size=int(args.batch_size),
        local_rank=0,
        world_size=1,
        is_distributed=False,
        shuffle_train=False,
    )
    split = str(cfg["evaluation"]["split"]).lower()
    loader = _select_loader(split, train_dl, val_dl, test_dl)
    preprocessing_spec = loader.dataset.preprocessing_spec

    channel_index, channel_name, source_spec = _resolve_source_channel(
        source_name=str(args.source),
        preprocessing_spec=preprocessing_spec,
        raster_input_channel_names=batch_schema.raster_input_channel_names,
    )
    min_value, max_value = _resolve_source_bounds(str(args.source), args)

    target_schema = get_single_downstream_target_schema(batch_schema)
    target_kind = _target_kind(batch_schema, target_schema.name)
    target_source_name = target_schema.spec.source
    if target_source_name is None:
        raise ValueError(f"Downstream target {target_schema.name!r} requires a source")
    target_source_spec = preprocessing_spec["sources"][target_source_name]

    model = build_eump_downstream_model(
        model_config=dict(cfg["model"]),
        batch_schema=batch_schema,
        target_schema=target_schema,
        base_swin_path=str(Path(MODEL_DIR) / str(cfg["experiment"]["pretrain_model"])),
        encoder_source=str(cfg["downstream"]["encoder_source"]),
        encoder_checkpoint_path=resolve_model_artifact_path(
            str(cfg["downstream"]["encoder_model_name"]),
            str(cfg["downstream"]["encoder_checkpoint_file"]),
        ),
        load_pretrained=not bool(cfg["experiment"]["scratch_backbone"]),
        freeze_encoder=bool(cfg["downstream"]["freeze_encoder"]),
        map_location=device,
    ).to(device)
    checkpoint_file = (
        str(cfg["evaluation"]["checkpoint_file"])
        if cfg["evaluation"]["checkpoint_file"]
        else (
            str(cfg["downstream"]["save_file"])
            if str(cfg["evaluation"]["checkpoint_source"]) == "best_model"
            else str(cfg["downstream"]["checkpoint_file"])
        )
    )
    checkpoint_path = load_model_state_for_evaluation(
        model=model,
        model_name=str(cfg["experiment"]["model_name"]),
        checkpoint_source=str(cfg["evaluation"]["checkpoint_source"]),
        checkpoint_file=checkpoint_file,
        map_location=device,
    )
    model.eval()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    collected: dict[str, list[np.ndarray]] = {
        "input_before_physical": [],
        "input_after_physical": [],
        "input_delta_physical": [],
        "input_valid_mask": [],
        "perturbation_mask": [],
        "prediction_before_physical": [],
        "prediction_after_physical": [],
        "prediction_delta_physical": [],
        "target_physical": [],
        "target_valid_mask": [],
        "city_mask": [],
    }
    if target_kind == "raster":
        collected.update(
            {
                "prediction_before_city_mean": [],
                "prediction_after_city_mean": [],
                "prediction_delta_city_mean": [],
                "target_city_mean": [],
                "target_city_mean_valid": [],
            }
        )

    metadata: list[dict[str, Any]] = []
    collected_count = 0

    with torch.no_grad():
        for batch in loader:
            remaining = int(args.limit) - collected_count
            if remaining <= 0:
                break
            take = min(remaining, int(batch.raster_inputs.shape[0]))

            raster_inputs = batch.raster_inputs[:take].to(device)
            scalar_inputs = batch.scalar_inputs[:take].to(device)
            raster_valid_mask = batch.raster_valid_mask[:take].to(device)
            city_mask = batch.city_mask[:take].to(device)

            perturbed_inputs, perturbed_valid, source_before, source_after, perturbation_mask = (
                _perturb_raster_input_channel(
                    raster_inputs=raster_inputs,
                    raster_valid_mask=raster_valid_mask,
                    city_mask=city_mask,
                    channel_index=channel_index,
                    source_spec=source_spec,
                    delta=float(args.delta),
                    mask_scope=str(args.mask_scope),
                    min_value=min_value,
                    max_value=max_value,
                )
            )

            raster_predictions_before, scalar_predictions_before = model(raster_inputs, scalar_inputs)
            raster_predictions_after, scalar_predictions_after = model(perturbed_inputs, scalar_inputs)

            if target_kind == "raster":
                target_block = batch.raster_targets[target_schema.name]
                target_data = target_block.data[:take].to(device)
                target_valid = target_block.valid_mask[:take].to(device)
                prediction_before = inverse_processed_tensor(
                    raster_predictions_before[target_schema.name],
                    target_source_spec,
                )
                prediction_after = inverse_processed_tensor(
                    raster_predictions_after[target_schema.name],
                    target_source_spec,
                )
                target_physical = inverse_processed_tensor(target_data, target_source_spec)
                delta_prediction = prediction_after - prediction_before

                city_before, city_before_valid = _sample_city_mean(
                    values=prediction_before,
                    valid_mask=target_valid,
                    city_mask=city_mask,
                )
                city_after, _ = _sample_city_mean(
                    values=prediction_after,
                    valid_mask=target_valid,
                    city_mask=city_mask,
                )
                city_target, _ = _sample_city_mean(
                    values=target_physical,
                    valid_mask=target_valid,
                    city_mask=city_mask,
                )

                collected["prediction_before_city_mean"].append(city_before)
                collected["prediction_after_city_mean"].append(city_after)
                collected["prediction_delta_city_mean"].append(city_after - city_before)
                collected["target_city_mean"].append(city_target)
                collected["target_city_mean_valid"].append(city_before_valid.astype(np.uint8))
            else:
                target_block = batch.scalar_targets[target_schema.name]
                target_data = target_block.data[:take].to(device)
                prediction_before = inverse_processed_tensor(
                    scalar_predictions_before[target_schema.name],
                    target_source_spec,
                )
                prediction_after = inverse_processed_tensor(
                    scalar_predictions_after[target_schema.name],
                    target_source_spec,
                )
                target_physical = inverse_processed_tensor(target_data, target_source_spec)
                delta_prediction = prediction_after - prediction_before

            collected["input_before_physical"].append(source_before.detach().cpu().numpy())
            collected["input_after_physical"].append(source_after.detach().cpu().numpy())
            collected["input_delta_physical"].append((source_after - source_before).detach().cpu().numpy())
            collected["input_valid_mask"].append(
                raster_valid_mask[:, channel_index : channel_index + 1, :, :].detach().cpu().numpy()
            )
            collected["perturbation_mask"].append(perturbation_mask.detach().cpu().numpy().astype(np.uint8))
            collected["prediction_before_physical"].append(prediction_before.detach().cpu().numpy())
            collected["prediction_after_physical"].append(prediction_after.detach().cpu().numpy())
            collected["prediction_delta_physical"].append(delta_prediction.detach().cpu().numpy())
            collected["target_physical"].append(target_physical.detach().cpu().numpy())
            if target_kind == "raster":
                collected["target_valid_mask"].append(target_valid.detach().cpu().numpy())
            else:
                collected["target_valid_mask"].append(target_block.valid_mask[:take].detach().cpu().numpy())
            collected["city_mask"].append(city_mask.detach().cpu().numpy())

            for meta in batch.meta[:take]:
                metadata.append(_jsonable(meta))
            collected_count += take

    if collected_count <= 0:
        raise ValueError("No samples were collected. Check split and year_range overrides.")

    results = {name: np.concatenate(chunks, axis=0) for name, chunks in collected.items()}
    summary = {
        "experiment": {
            "group": str(cfg["experiment"]["group"]),
            "name": str(cfg["experiment"]["name"]),
            "model_name": str(cfg["experiment"]["model_name"]),
            "pretrain_model": str(cfg["experiment"]["pretrain_model"]),
            "checkpoint_path": checkpoint_path,
            "checkpoint_source": str(cfg["evaluation"]["checkpoint_source"]),
            "pipeline_override": args.overrides,
        },
        "data": {
            "split": split,
            "limit": collected_count,
            "pipeline_year_range": list(pipeline_spec.year_range) if pipeline_spec.year_range is not None else None,
            "eump_root": str(cfg["data"]["eump_root"]),
            "preprocessing_path": str(cfg["data"]["preprocessing_path"]),
            "split_manifest_path": cfg["data"].get("split_manifest_path"),
        },
        "perturbation": {
            "source_name": str(args.source),
            "source_channel_name": channel_name,
            "source_channel_index": channel_index,
            "source_mode": str(source_spec["mode"]),
            "delta": float(args.delta),
            "mask_scope": str(args.mask_scope),
            "min_value": min_value,
            "max_value": max_value,
        },
        "target": {
            "name": target_schema.name,
            "kind": target_kind,
            "source": target_source_name,
        },
        "runtime": {
            "device": str(device),
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "results_npz": "results.npz",
        },
        "samples": metadata,
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    np.savez_compressed(output_dir / "results.npz", **results)

    logger.info(
        "Saved counterfactual outputs to {} (samples={}, source={}, target={})".format(
            output_dir,
            collected_count,
            args.source,
            target_schema.name,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
