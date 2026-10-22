from __future__ import annotations

import os
from copy import deepcopy
from typing import Any

import torch.multiprocessing as torch_mp
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from urban_embedding.core.logging import save_runtime_config, setup_logger
from urban_embedding.core.reproducibility import set_reproducible
from urban_embedding.data.processed.pipeline_spec import (
    EUMP_CITY_SCALAR_DOWNSTREAM_KIND,
    EUMP_CONTRASTIVE_KIND,
    EUMP_DOWNSTREAM_KIND,
    EUMP_RECONSTRUCTION_KIND,
)

from .distributed import (
    destroy_distributed_process_group,
    get_rank,
    get_world_size,
    init_distributed_process_group,
    resolve_device,
)
from .eump.registry import (
    CITY_SCALAR_MLP_MODEL_KIND,
    RuntimeContext,
    infer_eump_model_kind,
    normalize_runtime_identity,
    resolve_training_runtime,
)
from .eump.swin_reconstruction.trainer import (
    normalize_grad_norm_debug_config,
    normalize_scalar_head_mode,
)
from .eump.city_scalar_mlp.cache import normalize_city_scalar_feature_standardization
from .loops import fit
from .schedulers import normalize_warmup_cosine_scheduler_config


def _normalize_downstream_selection(selection: Any) -> dict[str, Any]:
    if selection is None:
        selection = {}
    if not isinstance(selection, dict):
        raise TypeError(f"downstream.selection must be a mapping, got {type(selection)!r}")
    normalized = deepcopy(selection)
    normalized.setdefault("metric", "val_loss")
    normalized.setdefault("metric_scale", "physical")
    normalized.setdefault("loss_view", "default")
    normalized.setdefault("center_ratio", 1.0)
    normalized.setdefault("raster_view", "global_pixels")
    normalized.setdefault("smoothing_window", 1)
    normalized["loss_view"] = str(normalized["loss_view"]).lower()
    normalized["center_ratio"] = float(normalized["center_ratio"])
    normalized["smoothing_window"] = int(normalized["smoothing_window"])
    return normalized


def _normalize_scheduler_config(scheduler: Any) -> dict[str, float]:
    normalized = normalize_warmup_cosine_scheduler_config(scheduler)
    return {
        "warmup_fraction": normalized.warmup_fraction,
        "lr_start_factor": normalized.lr_start_factor,
        "lr_end_factor": normalized.lr_end_factor,
    }


def _normalize_experiment_identity(experiment_cfg: dict[str, Any]) -> None:
    group = str(experiment_cfg.get("group", "")).strip().strip("/\\")
    raw_name = str(experiment_cfg.get("name", "")).strip().strip("/\\")
    if not group or not raw_name:
        raise ValueError("EUMP training requires experiment.group and experiment.name")
    if any(sep in group for sep in ("/", "\\")):
        raise ValueError(
            "experiment.group must be a single path segment; "
            f"got group={group!r}"
        )
    if "\\" in raw_name:
        raise ValueError(
            "experiment.name must be a safe relative path under experiment.group; "
            f"got name={raw_name!r}"
        )
    name_parts = [part for part in raw_name.split("/") if part]
    name = "/".join(name_parts)
    if not name or any(part in {".", ".."} for part in name_parts):
        raise ValueError(
            "experiment.name must be a safe relative path under experiment.group; "
            f"got name={raw_name!r}"
        )

    experiment_cfg["group"] = group
    experiment_cfg["name"] = name
    experiment_cfg["model_name"] = f"{group}/{name}"


def normalize_runtime_config(config: Any) -> dict[str, Any]:
    if isinstance(config, dict):
        config_dict = deepcopy(config)
    elif isinstance(config, DictConfig):
        config_dict = OmegaConf.to_container(config, resolve=True)
    else:
        raise TypeError(f"Unsupported config type: {type(config)!r}")

    experiment_cfg = config_dict.setdefault("experiment", {})
    reconstruct_cfg = config_dict.get("reconstruct")
    downstream_cfg = config_dict.get("downstream")
    contrastive_cfg = config_dict.get("contrastive")
    logging_cfg = config_dict.setdefault("logging", {})
    data_cfg = config_dict.setdefault("data", {})
    pipeline_candidate = config_dict.get("pipeline")
    if "launch" in config_dict and (
        not isinstance(pipeline_candidate, dict) or "kind" not in pipeline_candidate
    ):
        raise ValueError(
            "The selected config is a launch config, not a runnable pipeline. "
            "Use script/training/run_eump_full_train.sh or run_eump_full_eval.sh "
            "with launch=<name>, or select a runnable pipeline such as "
            "pipeline=eump_full/mae_v3/pretrain."
        )
    pipeline_cfg = config_dict.setdefault("pipeline", {})
    model_cfg = config_dict.setdefault("model", {})

    has_reconstruct_flag = "reconstruct" in experiment_cfg
    has_downstream_flag = "downstream" in experiment_cfg
    experiment_cfg.setdefault("scratch_backbone", False)
    experiment_cfg.setdefault("test", False)
    _normalize_experiment_identity(experiment_cfg)

    pipeline_cfg.setdefault("kind", "eump_reconstruction")
    model_cfg.setdefault("downstream_arch", "light_dual_head")
    model_cfg.setdefault("kind", infer_eump_model_kind(str(pipeline_cfg["kind"]), model_cfg))
    pipeline_kind, _ = normalize_runtime_identity(config_dict)
    if pipeline_kind in {EUMP_RECONSTRUCTION_KIND, EUMP_CONTRASTIVE_KIND}:
        expected_reconstruct = True
        expected_downstream = False
    elif pipeline_kind in {EUMP_DOWNSTREAM_KIND, EUMP_CITY_SCALAR_DOWNSTREAM_KIND}:
        expected_reconstruct = False
        expected_downstream = True
    else:
        raise ValueError(f"Unsupported EUMP pipeline.kind={pipeline_kind!r}")
    if has_reconstruct_flag and bool(experiment_cfg["reconstruct"]) != expected_reconstruct:
        raise ValueError(
            f"pipeline.kind={pipeline_kind!r} requires "
            f"experiment.reconstruct={str(expected_reconstruct).lower()}"
        )
    if has_downstream_flag and bool(experiment_cfg["downstream"]) != expected_downstream:
        raise ValueError(
            f"pipeline.kind={pipeline_kind!r} requires "
            f"experiment.downstream={str(expected_downstream).lower()}"
        )
    experiment_cfg["reconstruct"] = expected_reconstruct
    experiment_cfg["downstream"] = expected_downstream

    if reconstruct_cfg is not None:
        reconstruct_cfg.setdefault("save_file", "eump_reconstruction_model.pt")
        reconstruct_cfg.setdefault("checkpoint_file", "eump_reconstruction_checkpoint.pt")
        reconstruct_cfg.setdefault("resume", False)
        reconstruct_cfg.setdefault("resume_checkpoint", None)
        reconstruct_cfg.setdefault("accumulation_steps", 1)
        reconstruct_cfg.setdefault("save_best_model", False)
        reconstruct_cfg.setdefault("save_best_checkpoint", False)
        reconstruct_cfg.setdefault("save_final_model", True)
        reconstruct_cfg.setdefault("best_checkpoint_file", "eump_reconstruction_best_checkpoint.pt")
        reconstruct_cfg["checkpoint_every_n_epochs"] = _normalize_checkpoint_every_n_epochs(
            reconstruct_cfg.get("checkpoint_every_n_epochs"),
            name="reconstruct.checkpoint_every_n_epochs",
        )
        reconstruct_cfg["scalar_head_mode"] = normalize_scalar_head_mode(reconstruct_cfg.get("scalar_head_mode"))
        reconstruct_cfg["scheduler"] = _normalize_scheduler_config(reconstruct_cfg.get("scheduler"))
        debug_grad_norm = normalize_grad_norm_debug_config(reconstruct_cfg.get("debug_grad_norm"))
        reconstruct_cfg["debug_grad_norm"] = {
            "enabled": debug_grad_norm.enabled,
            "every_n_steps": debug_grad_norm.every_n_steps,
            "parameter_scope": debug_grad_norm.parameter_scope,
        }

    if contrastive_cfg is not None:
        contrastive_cfg.setdefault("save_file", "eump_contrastive_model.pt")
        contrastive_cfg.setdefault("checkpoint_file", "eump_contrastive_checkpoint.pt")
        contrastive_cfg.setdefault("best_checkpoint_file", "eump_contrastive_best_checkpoint.pt")
        contrastive_cfg.setdefault("resume", False)
        contrastive_cfg.setdefault("resume_checkpoint", None)
        contrastive_cfg.setdefault("accumulation_steps", 1)
        contrastive_cfg.setdefault("save_best_model", False)
        contrastive_cfg.setdefault("save_best_checkpoint", True)
        contrastive_cfg.setdefault("save_final_model", True)
        contrastive_cfg.setdefault("mask_ratio", 0.6)
        contrastive_cfg.setdefault("temperature", 0.1)
        contrastive_cfg.setdefault("projection_hidden_dim", 2048)
        contrastive_cfg.setdefault("projection_dim", 128)
        contrastive_cfg.setdefault("channel_dropout_p", 0.1)
        contrastive_cfg.setdefault("noise_std", 0.01)
        contrastive_cfg["checkpoint_every_n_epochs"] = _normalize_checkpoint_every_n_epochs(
            contrastive_cfg.get("checkpoint_every_n_epochs"),
            name="contrastive.checkpoint_every_n_epochs",
        )
        contrastive_cfg["scheduler"] = _normalize_scheduler_config(contrastive_cfg.get("scheduler"))

    if downstream_cfg is not None:
        if model_cfg["kind"] == CITY_SCALAR_MLP_MODEL_KIND:
            downstream_save_file = "eump_scalar_mlp_model.pt"
            downstream_checkpoint_file = "eump_scalar_mlp_checkpoint.pt"
        else:
            downstream_save_file = "eump_downstream_model.pt"
            downstream_checkpoint_file = "eump_downstream_checkpoint.pt"
        downstream_cfg.setdefault("save_file", downstream_save_file)
        downstream_cfg.setdefault("checkpoint_file", downstream_checkpoint_file)
        downstream_cfg.setdefault("resume", False)
        downstream_cfg.setdefault("resume_checkpoint", None)
        downstream_cfg.setdefault("accumulation_steps", 1)
        downstream_cfg["checkpoint_every_n_epochs"] = _normalize_checkpoint_every_n_epochs(
            downstream_cfg.get("checkpoint_every_n_epochs"),
            name="downstream.checkpoint_every_n_epochs",
        )
        downstream_cfg.setdefault("encoder_source", "reconstruction_checkpoint")
        downstream_cfg.setdefault("encoder_checkpoint_file", "eump_reconstruction_best_checkpoint.pt")
        downstream_cfg.setdefault("encoder_model_name", f"{experiment_cfg['group']}/pretrain")
        downstream_cfg.setdefault("freeze_encoder", True)
        downstream_cfg["feature_standardization"] = normalize_city_scalar_feature_standardization(
            downstream_cfg.get("feature_standardization")
        )
        downstream_cfg["scheduler"] = _normalize_scheduler_config(downstream_cfg.get("scheduler"))
        downstream_cfg["selection"] = _normalize_downstream_selection(downstream_cfg.get("selection"))

    logging_cfg.setdefault("level", "INFO")
    logging_cfg.setdefault("log_dir", None)
    logging_cfg.setdefault("progress_bar", True)
    data_cfg.setdefault("backend", "eump_v1")

    return config_dict


def resolve_log_dir(config: dict[str, Any]) -> str:
    group = config["experiment"]["group"]
    name = config["experiment"]["name"]
    explicit_log_dir = config["logging"].get("log_dir")
    if explicit_log_dir:
        return os.path.join(explicit_log_dir, group, name)

    if HydraConfig.initialized():
        return HydraConfig.get().run.dir

    return os.path.join("log", group, name)


def _validate_eump_runtime(cfg: dict[str, Any]):
    data_backend = cfg["data"].get("backend")
    if data_backend != "eump_v1":
        raise ValueError(f"EUMP training requires data.backend='eump_v1', got {data_backend!r}")
    return resolve_training_runtime(cfg)


def _normalize_checkpoint_every_n_epochs(value: Any, *, name: str) -> int:
    interval = 10 if value is None else int(value)
    if interval <= 0:
        raise ValueError(f"{name} must be > 0, got {interval}")
    return interval


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


def run_training(config: Any, source: str = "runtime") -> None:
    cfg = normalize_runtime_config(config)
    init_distributed_process_group()

    try:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        world_size = get_world_size()
        device = resolve_device(local_rank)
        log_dir = resolve_log_dir(cfg)
        logger = setup_logger(log_dir=log_dir, rank=get_rank(), log_level=cfg["logging"]["level"])

        if get_rank() == 0:
            logger.info(f"Starting EUMP training via {source}")
            logger.info(f"Resolved log directory: {log_dir}")
            if device.type == "cpu":
                logger.warning(
                    "CUDA is unavailable. CPU DDP is enabled for validation runs, "
                    "but large training jobs are still expected to use GPUs."
                )
            saved_config_path = save_runtime_config(log_dir, cfg)
            if saved_config_path:
                logger.info(f"Saved runtime config to {saved_config_path}")

        _configure_torch_multiprocessing(logger)
        set_reproducible(cfg["experiment"]["random_seed"])
        runtime_spec = _validate_eump_runtime(cfg)
        context = RuntimeContext(
            logger=logger,
            device=device,
            local_rank=local_rank,
            world_size=world_size,
            log_dir=log_dir,
            random_seed=cfg["experiment"]["random_seed"],
        )
        trainer = runtime_spec.trainer_factory(cfg, context)
        fit(trainer, test=cfg["experiment"]["test"])
    finally:
        destroy_distributed_process_group()


__all__ = [
    "normalize_runtime_config",
    "resolve_log_dir",
    "run_training",
]
