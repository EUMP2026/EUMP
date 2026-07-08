from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from urban_embedding.data.processed.pipeline_spec import (
    EUMP_CITY_SCALAR_DOWNSTREAM_KIND,
    EUMP_CONTRASTIVE_KIND,
    EUMP_DOWNSTREAM_KIND,
    EUMP_RECONSTRUCTION_KIND,
    normalize_eump_pipeline_kind,
)
from urban_embedding.training.eump.city_scalar_mlp.evaluator import EumpScalarMlpEvaluator
from urban_embedding.training.eump.city_scalar_mlp.trainer import EumpScalarMlpTrainer
from urban_embedding.training.eump.swin_contrastive.trainer import EumpSwinContrastiveTrainer
from urban_embedding.training.eump.swin_downstream.evaluator import EumpDownstreamEvaluator
from urban_embedding.training.eump.swin_downstream.trainer import EumpDownstreamTrainer
from urban_embedding.training.eump.swin_reconstruction.trainer import EumpSwinReconstructionTrainer


SWIN_RECONSTRUCTION_MODEL_KIND = "swin_reconstruction"
SWIN_UNET_RECONSTRUCTION_MODEL_KIND = "swin_unet_reconstruction"
CONVNEXT_UNET_RECONSTRUCTION_MODEL_KIND = "convnext_unet_reconstruction"
SWIN_CONTRASTIVE_MODEL_KIND = "swin_contrastive"
SWIN_MOCO_MODEL_KIND = "swin_moco"
SWIN_DOWNSTREAM_LIGHT_MODEL_KIND = "swin_downstream_light"
CITY_SCALAR_MLP_MODEL_KIND = "city_scalar_mlp"


@dataclass(frozen=True, slots=True)
class RuntimeContext:
    logger: Any
    device: torch.device
    local_rank: int
    world_size: int
    log_dir: str
    random_seed: int

    def local_batch_size(self, *, name: str, batch_size: Any) -> int:
        global_batch_size = _validate_global_batch_size(name, batch_size, self.world_size)
        return global_batch_size // max(1, int(self.world_size))


TrainerFactory = Callable[[dict[str, Any], RuntimeContext], Any]
EvaluatorFactory = Callable[[dict[str, Any], RuntimeContext], Any]


@dataclass(frozen=True, slots=True)
class TrainingRuntimeSpec:
    pipeline_kind: str
    model_kind: str
    trainer_cls: type
    evaluator_cls: type | None
    train_config_section: str
    default_save_file: str
    default_checkpoint_file: str
    trainer_factory: TrainerFactory
    evaluator_factory: EvaluatorFactory | None = None

    @property
    def key(self) -> tuple[str, str]:
        return self.pipeline_kind, self.model_kind


def _validate_global_batch_size(name: str, batch_size: Any, world_size: int) -> int:
    batch_size = int(batch_size)
    world_size = max(1, int(world_size))
    if batch_size <= 0:
        raise ValueError(f"{name} must be > 0, got {batch_size}")
    if batch_size % world_size != 0:
        raise ValueError(f"{name}={batch_size} must be divisible by world_size={world_size}")
    return batch_size


def infer_eump_model_kind(pipeline_kind: str, model_config: dict[str, Any] | None) -> str:
    model_config = {} if model_config is None else model_config
    configured = model_config.get("kind")
    if configured:
        return str(configured)

    pipeline_kind = normalize_eump_pipeline_kind(pipeline_kind)
    if pipeline_kind == EUMP_RECONSTRUCTION_KIND:
        return SWIN_RECONSTRUCTION_MODEL_KIND
    if pipeline_kind == EUMP_CONTRASTIVE_KIND:
        return SWIN_CONTRASTIVE_MODEL_KIND
    if pipeline_kind == EUMP_CITY_SCALAR_DOWNSTREAM_KIND:
        return CITY_SCALAR_MLP_MODEL_KIND
    if pipeline_kind == EUMP_DOWNSTREAM_KIND:
        arch = str(model_config.get("downstream_arch", "light_dual_head")).lower()
        if arch == "light_dual_head":
            return SWIN_DOWNSTREAM_LIGHT_MODEL_KIND
    raise ValueError(
        "model.kind is required for this EUMP runtime. "
        f"Unable to infer it from pipeline.kind={pipeline_kind!r} and model={model_config!r}."
    )


def normalize_runtime_identity(cfg: dict[str, Any]) -> tuple[str, str]:
    pipeline_cfg = cfg.setdefault("pipeline", {})
    model_cfg = cfg.setdefault("model", {})
    pipeline_kind = normalize_eump_pipeline_kind(pipeline_cfg.get("kind", EUMP_RECONSTRUCTION_KIND))
    model_kind = infer_eump_model_kind(pipeline_kind, model_cfg)
    pipeline_cfg["kind"] = pipeline_kind
    model_cfg["kind"] = model_kind
    return pipeline_kind, model_kind


def _require_training_flags(
    cfg: dict[str, Any],
    *,
    reconstruct: bool,
    downstream: bool,
    name: str,
) -> None:
    if bool(cfg["experiment"].get("reconstruct")) != reconstruct:
        raise ValueError(f"{name} requires experiment.reconstruct={str(reconstruct).lower()}")
    if bool(cfg["experiment"].get("downstream")) != downstream:
        raise ValueError(f"{name} requires experiment.downstream={str(downstream).lower()}")


def _build_swin_reconstruction_trainer(cfg: dict[str, Any], context: RuntimeContext) -> EumpSwinReconstructionTrainer:
    _require_training_flags(cfg, reconstruct=True, downstream=False, name=EUMP_RECONSTRUCTION_KIND)
    reconstruct_batch_size = context.local_batch_size(
        name="reconstruct.batch_size",
        batch_size=cfg["reconstruct"]["batch_size"],
    )
    if context.local_rank == 0:
        context.logger.info("Start EUMP Swin reconstruction training")

    trainer = EumpSwinReconstructionTrainer(
        logger=context.logger,
        device=context.device,
        local_rank=context.local_rank,
        world_size=context.world_size,
        model_name=cfg["experiment"]["model_name"],
        pretrain_model=cfg["experiment"]["pretrain_model"],
        data_config=cfg["data"],
        pipeline_config=cfg["pipeline"],
        learning_rate=cfg["reconstruct"]["lr"],
        weight_decay=cfg["reconstruct"]["weight_decay"],
        batch_size=reconstruct_batch_size,
        n_epochs=cfg["reconstruct"]["epochs"],
        resume=cfg["reconstruct"]["resume"],
        resume_checkpoint=cfg["reconstruct"]["resume_checkpoint"],
        random_seed=context.random_seed,
        mask_ratio=cfg["reconstruct"]["mask_ratio"],
        accumulation_steps=cfg["reconstruct"]["accumulation_steps"],
        scratch_backbone=cfg["experiment"]["scratch_backbone"],
        save_file_name=cfg["reconstruct"]["save_file"],
        checkpoint_file_name=cfg["reconstruct"]["checkpoint_file"],
        log_dir=context.log_dir,
        save_best_model=cfg["reconstruct"]["save_best_model"],
        save_best_checkpoint=cfg["reconstruct"]["save_best_checkpoint"],
        save_final_model=cfg["reconstruct"]["save_final_model"],
        best_checkpoint_file_name=cfg["reconstruct"]["best_checkpoint_file"],
        scheduler_config=cfg["reconstruct"]["scheduler"],
        debug_grad_norm=cfg["reconstruct"]["debug_grad_norm"],
        scalar_head_mode=cfg["reconstruct"]["scalar_head_mode"],
        model_config=cfg.get("model"),
    )
    trainer.progress_bar_enabled = bool(cfg["logging"]["progress_bar"])
    trainer.checkpoint_every_n_epochs = int(cfg["reconstruct"]["checkpoint_every_n_epochs"])
    return trainer


def _build_swin_downstream_trainer(cfg: dict[str, Any], context: RuntimeContext) -> EumpDownstreamTrainer:
    _require_training_flags(cfg, reconstruct=False, downstream=True, name=EUMP_DOWNSTREAM_KIND)
    downstream_batch_size = context.local_batch_size(
        name="downstream.batch_size",
        batch_size=cfg["downstream"]["batch_size"],
    )
    if context.local_rank == 0:
        context.logger.info("Start EUMP Swin downstream training")

    trainer = EumpDownstreamTrainer(
        logger=context.logger,
        device=context.device,
        local_rank=context.local_rank,
        world_size=context.world_size,
        model_name=cfg["experiment"]["model_name"],
        pretrain_model=cfg["experiment"]["pretrain_model"],
        data_config=cfg["data"],
        pipeline_config=cfg["pipeline"],
        model_config=cfg["model"],
        learning_rate=cfg["downstream"]["lr"],
        weight_decay=cfg["downstream"]["weight_decay"],
        batch_size=downstream_batch_size,
        n_epochs=cfg["downstream"]["epochs"],
        resume=cfg["downstream"]["resume"],
        resume_checkpoint=cfg["downstream"]["resume_checkpoint"],
        random_seed=context.random_seed,
        accumulation_steps=cfg["downstream"]["accumulation_steps"],
        scratch_backbone=cfg["experiment"]["scratch_backbone"],
        save_file_name=cfg["downstream"]["save_file"],
        checkpoint_file_name=cfg["downstream"]["checkpoint_file"],
        log_dir=context.log_dir,
        encoder_source=cfg["downstream"]["encoder_source"],
        encoder_checkpoint_file=cfg["downstream"]["encoder_checkpoint_file"],
        encoder_model_name=cfg["downstream"]["encoder_model_name"],
        freeze_encoder=cfg["downstream"]["freeze_encoder"],
        selection_config=cfg["downstream"]["selection"],
        scheduler_config=cfg["downstream"]["scheduler"],
    )
    trainer.progress_bar_enabled = bool(cfg["logging"]["progress_bar"])
    trainer.checkpoint_every_n_epochs = int(cfg["downstream"]["checkpoint_every_n_epochs"])
    return trainer


def _build_city_scalar_mlp_trainer(cfg: dict[str, Any], context: RuntimeContext) -> EumpScalarMlpTrainer:
    _require_training_flags(cfg, reconstruct=False, downstream=True, name=EUMP_CITY_SCALAR_DOWNSTREAM_KIND)
    downstream_batch_size = context.local_batch_size(
        name="downstream.batch_size",
        batch_size=cfg["downstream"]["batch_size"],
    )
    if context.local_rank == 0:
        context.logger.info("Start EUMP city-scalar MLP downstream baseline training")

    trainer = EumpScalarMlpTrainer(
        logger=context.logger,
        device=context.device,
        local_rank=context.local_rank,
        world_size=context.world_size,
        model_name=cfg["experiment"]["model_name"],
        data_config=cfg["data"],
        pipeline_config=cfg["pipeline"],
        model_config=cfg["model"],
        learning_rate=cfg["downstream"]["lr"],
        weight_decay=cfg["downstream"]["weight_decay"],
        batch_size=downstream_batch_size,
        n_epochs=cfg["downstream"]["epochs"],
        resume=cfg["downstream"]["resume"],
        resume_checkpoint=cfg["downstream"]["resume_checkpoint"],
        random_seed=context.random_seed,
        accumulation_steps=cfg["downstream"]["accumulation_steps"],
        save_file_name=cfg["downstream"]["save_file"],
        checkpoint_file_name=cfg["downstream"]["checkpoint_file"],
        log_dir=context.log_dir,
        scheduler_config=cfg["downstream"]["scheduler"],
        feature_standardization=cfg["downstream"]["feature_standardization"],
    )
    trainer.progress_bar_enabled = bool(cfg["logging"]["progress_bar"])
    trainer.checkpoint_every_n_epochs = int(cfg["downstream"]["checkpoint_every_n_epochs"])
    return trainer


def _build_swin_downstream_evaluator(cfg: dict[str, Any], context: RuntimeContext) -> EumpDownstreamEvaluator:
    return EumpDownstreamEvaluator(
        logger=context.logger,
        device=context.device,
        model_name=cfg["experiment"]["model_name"],
        pretrain_model=cfg["experiment"]["pretrain_model"],
        data_config=cfg["data"],
        pipeline_config=cfg["pipeline"],
        model_config=cfg["model"],
        downstream_config=cfg["downstream"],
        evaluation_config=cfg["evaluation"],
        local_rank=context.local_rank,
        world_size=context.world_size,
        random_seed=context.random_seed,
        scratch_backbone=cfg["experiment"]["scratch_backbone"],
    )


def _build_city_scalar_mlp_evaluator(cfg: dict[str, Any], context: RuntimeContext) -> EumpScalarMlpEvaluator:
    return EumpScalarMlpEvaluator(
        logger=context.logger,
        device=context.device,
        model_name=cfg["experiment"]["model_name"],
        data_config=cfg["data"],
        pipeline_config=cfg["pipeline"],
        model_config=cfg["model"],
        downstream_config=cfg["downstream"],
        evaluation_config=cfg["evaluation"],
        local_rank=context.local_rank,
        world_size=context.world_size,
        random_seed=context.random_seed,
    )


def _build_swin_contrastive_trainer(cfg: dict[str, Any], context: RuntimeContext) -> EumpSwinContrastiveTrainer:
    _require_training_flags(cfg, reconstruct=True, downstream=False, name=EUMP_CONTRASTIVE_KIND)
    contrastive_cfg = cfg.get("contrastive")
    if not isinstance(contrastive_cfg, dict):
        raise ValueError(
            "pipeline.kind=eump_contrastive requires a top-level 'contrastive' config section"
        )
    batch_size = context.local_batch_size(
        name="contrastive.batch_size",
        batch_size=contrastive_cfg["batch_size"],
    )
    if context.local_rank == 0:
        context.logger.info("Start EUMP Swin contrastive pretraining")

    trainer = EumpSwinContrastiveTrainer(
        logger=context.logger,
        device=context.device,
        local_rank=context.local_rank,
        world_size=context.world_size,
        model_name=cfg["experiment"]["model_name"],
        pretrain_model=cfg["experiment"]["pretrain_model"],
        data_config=cfg["data"],
        pipeline_config=cfg["pipeline"],
        learning_rate=contrastive_cfg["lr"],
        weight_decay=contrastive_cfg["weight_decay"],
        batch_size=batch_size,
        n_epochs=contrastive_cfg["epochs"],
        resume=contrastive_cfg["resume"],
        resume_checkpoint=contrastive_cfg["resume_checkpoint"],
        random_seed=context.random_seed,
        accumulation_steps=contrastive_cfg["accumulation_steps"],
        scratch_backbone=cfg["experiment"]["scratch_backbone"],
        save_file_name=contrastive_cfg["save_file"],
        checkpoint_file_name=contrastive_cfg["checkpoint_file"],
        log_dir=context.log_dir,
        save_best_model=contrastive_cfg["save_best_model"],
        save_best_checkpoint=contrastive_cfg["save_best_checkpoint"],
        save_final_model=contrastive_cfg["save_final_model"],
        best_checkpoint_file_name=contrastive_cfg["best_checkpoint_file"],
        scheduler_config=contrastive_cfg["scheduler"],
        mask_ratio=contrastive_cfg["mask_ratio"],
        temperature=contrastive_cfg["temperature"],
        projection_hidden_dim=contrastive_cfg["projection_hidden_dim"],
        projection_dim=contrastive_cfg["projection_dim"],
        channel_dropout_p=contrastive_cfg["channel_dropout_p"],
        noise_std=contrastive_cfg["noise_std"],
    )
    trainer.progress_bar_enabled = bool(cfg["logging"]["progress_bar"])
    trainer.checkpoint_every_n_epochs = int(contrastive_cfg["checkpoint_every_n_epochs"])
    return trainer


def _build_swin_moco_trainer(cfg: dict[str, Any], context: RuntimeContext) -> EumpSwinContrastiveTrainer:
    _require_training_flags(cfg, reconstruct=True, downstream=False, name=EUMP_CONTRASTIVE_KIND)
    contrastive_cfg = cfg.get("contrastive")
    if not isinstance(contrastive_cfg, dict):
        raise ValueError(
            "pipeline.kind=eump_contrastive requires a top-level 'contrastive' config section"
        )
    batch_size = context.local_batch_size(
        name="contrastive.batch_size",
        batch_size=contrastive_cfg["batch_size"],
    )
    if context.local_rank == 0:
        context.logger.info("Start EUMP Swin MoCo contrastive pretraining")

    trainer = EumpSwinContrastiveTrainer(
        logger=context.logger,
        device=context.device,
        local_rank=context.local_rank,
        world_size=context.world_size,
        model_name=cfg["experiment"]["model_name"],
        pretrain_model=cfg["experiment"]["pretrain_model"],
        data_config=cfg["data"],
        pipeline_config=cfg["pipeline"],
        learning_rate=contrastive_cfg["lr"],
        weight_decay=contrastive_cfg["weight_decay"],
        batch_size=batch_size,
        n_epochs=contrastive_cfg["epochs"],
        resume=contrastive_cfg["resume"],
        resume_checkpoint=contrastive_cfg["resume_checkpoint"],
        random_seed=context.random_seed,
        accumulation_steps=contrastive_cfg["accumulation_steps"],
        scratch_backbone=cfg["experiment"]["scratch_backbone"],
        save_file_name=contrastive_cfg["save_file"],
        checkpoint_file_name=contrastive_cfg["checkpoint_file"],
        log_dir=context.log_dir,
        save_best_model=contrastive_cfg["save_best_model"],
        save_best_checkpoint=contrastive_cfg["save_best_checkpoint"],
        save_final_model=contrastive_cfg["save_final_model"],
        best_checkpoint_file_name=contrastive_cfg["best_checkpoint_file"],
        scheduler_config=contrastive_cfg["scheduler"],
        mask_ratio=contrastive_cfg["mask_ratio"],
        temperature=contrastive_cfg["temperature"],
        projection_hidden_dim=contrastive_cfg["projection_hidden_dim"],
        projection_dim=contrastive_cfg["projection_dim"],
        channel_dropout_p=contrastive_cfg["channel_dropout_p"],
        noise_std=contrastive_cfg["noise_std"],
        momentum=contrastive_cfg.get("momentum", 0.996),
    )
    trainer.progress_bar_enabled = bool(cfg["logging"]["progress_bar"])
    trainer.checkpoint_every_n_epochs = int(contrastive_cfg["checkpoint_every_n_epochs"])
    return trainer


TRAINING_RUNTIME_SPECS: tuple[TrainingRuntimeSpec, ...] = (
    TrainingRuntimeSpec(
        pipeline_kind=EUMP_RECONSTRUCTION_KIND,
        model_kind=SWIN_RECONSTRUCTION_MODEL_KIND,
        trainer_cls=EumpSwinReconstructionTrainer,
        evaluator_cls=None,
        train_config_section="reconstruct",
        default_save_file="eump_reconstruction_model.pt",
        default_checkpoint_file="eump_reconstruction_checkpoint.pt",
        trainer_factory=_build_swin_reconstruction_trainer,
    ),
    TrainingRuntimeSpec(
        pipeline_kind=EUMP_RECONSTRUCTION_KIND,
        model_kind=SWIN_UNET_RECONSTRUCTION_MODEL_KIND,
        trainer_cls=EumpSwinReconstructionTrainer,
        evaluator_cls=None,
        train_config_section="reconstruct",
        default_save_file="eump_reconstruction_model.pt",
        default_checkpoint_file="eump_reconstruction_checkpoint.pt",
        trainer_factory=_build_swin_reconstruction_trainer,
    ),
    TrainingRuntimeSpec(
        pipeline_kind=EUMP_RECONSTRUCTION_KIND,
        model_kind=CONVNEXT_UNET_RECONSTRUCTION_MODEL_KIND,
        trainer_cls=EumpSwinReconstructionTrainer,
        evaluator_cls=None,
        train_config_section="reconstruct",
        default_save_file="eump_reconstruction_model.pt",
        default_checkpoint_file="eump_reconstruction_checkpoint.pt",
        trainer_factory=_build_swin_reconstruction_trainer,
    ),
    TrainingRuntimeSpec(
        pipeline_kind=EUMP_CONTRASTIVE_KIND,
        model_kind=SWIN_CONTRASTIVE_MODEL_KIND,
        trainer_cls=EumpSwinContrastiveTrainer,
        evaluator_cls=None,
        train_config_section="contrastive",
        default_save_file="eump_contrastive_model.pt",
        default_checkpoint_file="eump_contrastive_checkpoint.pt",
        trainer_factory=_build_swin_contrastive_trainer,
    ),
    TrainingRuntimeSpec(
        pipeline_kind=EUMP_CONTRASTIVE_KIND,
        model_kind=SWIN_MOCO_MODEL_KIND,
        trainer_cls=EumpSwinContrastiveTrainer,
        evaluator_cls=None,
        train_config_section="contrastive",
        default_save_file="eump_contrastive_model.pt",
        default_checkpoint_file="eump_contrastive_checkpoint.pt",
        trainer_factory=_build_swin_moco_trainer,
    ),
    TrainingRuntimeSpec(
        pipeline_kind=EUMP_DOWNSTREAM_KIND,
        model_kind=SWIN_DOWNSTREAM_LIGHT_MODEL_KIND,
        trainer_cls=EumpDownstreamTrainer,
        evaluator_cls=EumpDownstreamEvaluator,
        train_config_section="downstream",
        default_save_file="eump_downstream_model.pt",
        default_checkpoint_file="eump_downstream_checkpoint.pt",
        trainer_factory=_build_swin_downstream_trainer,
        evaluator_factory=_build_swin_downstream_evaluator,
    ),
    TrainingRuntimeSpec(
        pipeline_kind=EUMP_CITY_SCALAR_DOWNSTREAM_KIND,
        model_kind=CITY_SCALAR_MLP_MODEL_KIND,
        trainer_cls=EumpScalarMlpTrainer,
        evaluator_cls=EumpScalarMlpEvaluator,
        train_config_section="downstream",
        default_save_file="eump_scalar_mlp_model.pt",
        default_checkpoint_file="eump_scalar_mlp_checkpoint.pt",
        trainer_factory=_build_city_scalar_mlp_trainer,
        evaluator_factory=_build_city_scalar_mlp_evaluator,
    ),
)

_TRAINING_RUNTIME_BY_KEY = {spec.key: spec for spec in TRAINING_RUNTIME_SPECS}


def supported_runtime_keys(*, evaluation: bool = False) -> tuple[tuple[str, str], ...]:
    return tuple(
        spec.key
        for spec in TRAINING_RUNTIME_SPECS
        if not evaluation or spec.evaluator_factory is not None
    )


def _format_supported(keys: tuple[tuple[str, str], ...]) -> str:
    return ", ".join(f"({pipeline_kind}, {model_kind})" for pipeline_kind, model_kind in keys)


def resolve_training_runtime(cfg: dict[str, Any]) -> TrainingRuntimeSpec:
    key = normalize_runtime_identity(cfg)
    spec = _TRAINING_RUNTIME_BY_KEY.get(key)
    if spec is None:
        raise ValueError(
            "Unsupported EUMP training runtime "
            f"(pipeline.kind={key[0]!r}, model.kind={key[1]!r}). "
            f"Supported runtimes: {_format_supported(supported_runtime_keys())}"
        )
    return spec


def resolve_evaluation_runtime(cfg: dict[str, Any]) -> TrainingRuntimeSpec:
    spec = resolve_training_runtime(cfg)
    if spec.evaluator_factory is None:
        raise ValueError(
            "Unsupported EUMP evaluation runtime "
            f"(pipeline.kind={spec.pipeline_kind!r}, model.kind={spec.model_kind!r}). "
            f"Supported evaluation runtimes: {_format_supported(supported_runtime_keys(evaluation=True))}"
        )
    return spec


__all__ = [
    "CITY_SCALAR_MLP_MODEL_KIND",
    "RuntimeContext",
    "SWIN_DOWNSTREAM_LIGHT_MODEL_KIND",
    "SWIN_RECONSTRUCTION_MODEL_KIND",
    "TRAINING_RUNTIME_SPECS",
    "TrainingRuntimeSpec",
    "infer_eump_model_kind",
    "normalize_runtime_identity",
    "resolve_evaluation_runtime",
    "resolve_training_runtime",
    "supported_runtime_keys",
]
