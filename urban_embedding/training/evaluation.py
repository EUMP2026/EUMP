from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch.multiprocessing as torch_mp
from hydra.core.hydra_config import HydraConfig

from urban_embedding.core.logging import save_runtime_config, setup_logger
from urban_embedding.core.paths import resolve_model_artifact_path
from urban_embedding.core.reproducibility import set_reproducible

from .distributed import (
    destroy_distributed_process_group,
    get_rank,
    get_world_size,
    init_distributed_process_group,
    resolve_device,
)
from .eump.city_scalar_mlp.evaluator import EumpScalarMlpEvaluator
from .eump.registry import RuntimeContext, resolve_evaluation_runtime
from .eump.swin_downstream.evaluator import EumpDownstreamEvaluator
from .regression_metrics import (
    accumulate_regression_stats,
    empty_regression_stats,
    inverse_processed_tensor,
    raster_city_mean_metric_tensors,
    raster_global_metric_tensors,
    scalar_metric_tensors,
    summarize_regression_stats,
)
from .runner import normalize_runtime_config


SUPPORTED_EVALUATION_SPLITS = {"train", "val", "test"}
SUPPORTED_EVALUATION_METRIC_SCALES = {"physical", "processed"}
SUPPORTED_EVALUATION_CHECKPOINT_SOURCES = {"best_model", "training_checkpoint"}


def _configure_torch_multiprocessing(logger) -> None:
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

    if get_rank() == 0:
        logger.info(f"Use torch multiprocessing sharing strategy: {requested_strategy}")


def normalize_evaluation_config(config: Any) -> dict[str, Any]:
    cfg = normalize_runtime_config(config)
    evaluation_cfg = cfg.setdefault("evaluation", {})
    evaluation_cfg.setdefault("split", "test")
    metric_scale = str(evaluation_cfg.get("metric_scale", "physical")).lower()
    if metric_scale == "training":
        metric_scale = "processed"
    evaluation_cfg["metric_scale"] = metric_scale
    evaluation_cfg.setdefault("center_ratio", 1.0)
    evaluation_cfg.setdefault("checkpoint_source", "best_model")
    evaluation_cfg.setdefault("checkpoint_file", None)
    evaluation_cfg.setdefault("save_json", True)
    evaluation_cfg.setdefault("extra_center_city_mean", False)
    return cfg


def resolve_evaluation_log_dir(config: dict[str, Any]) -> str:
    group = config["experiment"]["group"]
    name = config["experiment"]["name"]
    explicit_log_dir = config["logging"].get("log_dir")
    if explicit_log_dir:
        return os.path.join(explicit_log_dir, group, name, "evaluation")

    if HydraConfig.initialized():
        return HydraConfig.get().run.dir

    return os.path.join("log", group, name, "evaluation")


def _validate_eump_evaluation_runtime(cfg: dict[str, Any]):
    data_backend = cfg["data"].get("backend")
    if data_backend != "eump_v1":
        raise ValueError(f"EUMP evaluation requires data.backend='eump_v1', got {data_backend!r}")

    if cfg["experiment"].get("reconstruct"):
        raise ValueError("EUMP downstream evaluation requires experiment.reconstruct=false")
    if not cfg["experiment"].get("downstream"):
        raise ValueError("EUMP downstream evaluation requires experiment.downstream=true")

    split = str(cfg["evaluation"]["split"]).lower()
    if split not in SUPPORTED_EVALUATION_SPLITS:
        raise ValueError(
            f"evaluation.split must be one of {sorted(SUPPORTED_EVALUATION_SPLITS)}, got {split!r}"
        )
    metric_scale = str(cfg["evaluation"]["metric_scale"]).lower()
    if metric_scale not in SUPPORTED_EVALUATION_METRIC_SCALES:
        raise ValueError(
            "evaluation.metric_scale must be one of "
            f"{sorted(SUPPORTED_EVALUATION_METRIC_SCALES)}, got {metric_scale!r}"
        )
    checkpoint_source = str(cfg["evaluation"]["checkpoint_source"]).lower()
    if checkpoint_source not in SUPPORTED_EVALUATION_CHECKPOINT_SOURCES:
        raise ValueError(
            "evaluation.checkpoint_source must be one of "
            f"{sorted(SUPPORTED_EVALUATION_CHECKPOINT_SOURCES)}, got {checkpoint_source!r}"
        )
    return resolve_evaluation_runtime(cfg)


def _save_evaluation_summary(model_name: str, split: str, summary: Mapping[str, Any]) -> str:
    output_path = resolve_model_artifact_path(model_name, os.path.join("evaluation", f"{split}_metrics.json"))
    if output_path is None:
        raise ValueError("Unable to resolve evaluation output path")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, sort_keys=False, allow_nan=False)
    return str(output)


def run_evaluation(config: Any, source: str = "runtime") -> dict[str, Any] | None:
    cfg = normalize_evaluation_config(config)
    init_distributed_process_group()

    try:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        world_size = get_world_size()
        device = resolve_device(local_rank)
        log_dir = resolve_evaluation_log_dir(cfg)
        logger = setup_logger(log_dir=log_dir, rank=get_rank(), log_level=cfg["logging"]["level"])

        if get_rank() == 0:
            logger.info(f"Starting EUMP evaluation via {source}")
            logger.info(f"Resolved evaluation log directory: {log_dir}")
            saved_config_path = save_runtime_config(log_dir, cfg)
            if saved_config_path:
                logger.info(f"Saved runtime config to {saved_config_path}")

        _configure_torch_multiprocessing(logger)
        set_reproducible(cfg["experiment"]["random_seed"])
        runtime_spec = _validate_eump_evaluation_runtime(cfg)
        context = RuntimeContext(
            logger=logger,
            device=device,
            local_rank=local_rank,
            world_size=world_size,
            log_dir=log_dir,
            random_seed=cfg["experiment"]["random_seed"],
        )
        if runtime_spec.evaluator_factory is None:
            raise ValueError(
                "Resolved EUMP runtime has no evaluator: "
                f"pipeline.kind={runtime_spec.pipeline_kind!r}, model.kind={runtime_spec.model_kind!r}"
            )
        evaluator = runtime_spec.evaluator_factory(cfg, context)
        summary = evaluator.evaluate()
        if get_rank() == 0 and cfg["evaluation"]["save_json"]:
            output_path = _save_evaluation_summary(
                cfg["experiment"]["model_name"],
                str(cfg["evaluation"]["split"]).lower(),
                summary,
            )
            logger.info(f"Saved evaluation metrics to {output_path}")
        return summary if get_rank() == 0 else None
    finally:
        destroy_distributed_process_group()


__all__ = [
    "EumpDownstreamEvaluator",
    "EumpScalarMlpEvaluator",
    "SUPPORTED_EVALUATION_CHECKPOINT_SOURCES",
    "SUPPORTED_EVALUATION_METRIC_SCALES",
    "SUPPORTED_EVALUATION_SPLITS",
    "accumulate_regression_stats",
    "empty_regression_stats",
    "inverse_processed_tensor",
    "normalize_evaluation_config",
    "raster_city_mean_metric_tensors",
    "raster_global_metric_tensors",
    "resolve_evaluation_log_dir",
    "run_evaluation",
    "scalar_metric_tensors",
    "summarize_regression_stats",
]
