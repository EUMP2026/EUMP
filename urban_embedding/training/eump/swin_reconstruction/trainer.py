from __future__ import annotations

from collections.abc import Iterable, Mapping
import math
import os
from dataclasses import dataclass

import torch

from urban_embedding.core.paths import MODEL_DIR
from urban_embedding.data.processed.batch import ProcessedModelBatch
from urban_embedding.data.processed.dataset import build_eump_dataloaders
from urban_embedding.data.processed.pipeline_spec import (
    PipelineLossConfig,
    PipelineSpec,
    ProcessedBatchSchema,
    TargetSchema,
)
from urban_embedding.models.eump.swin.reconstruction import EumpSwinReconstructionModel, SCALAR_HEAD_MODES
from urban_embedding.models.eump.swin_unet.reconstruction import EumpSwinUnetReconstructionModel
from urban_embedding.models.eump.convnext_unet.reconstruction import EumpConvNextUnetReconstructionModel
from urban_embedding.training.distributed import get_model, wrap_ddp
from urban_embedding.training.masking import generate_mask
from urban_embedding.training.eump_loss import (
    LossReport,
    accumulate_metrics,
    aggregate_loss_report,
    average_metrics,
    compute_target_loss,
    coverage,
    tensor_to_float,
)
from urban_embedding.training.schedulers import (
    build_warmup_cosine_scheduler,
    normalize_warmup_cosine_scheduler_config,
    warmup_cosine_metadata,
)
from urban_embedding.training.tensorboard import TensorBoardScalarWriter


GRAD_NORM_DEBUG_PARAMETER_SCOPES = {"backbone", "all"}


@dataclass(frozen=True, slots=True)
class GradNormDebugConfig:
    enabled: bool = False
    every_n_steps: int = 10
    parameter_scope: str = "backbone"


def normalize_grad_norm_debug_config(payload: Mapping[str, object] | None) -> GradNormDebugConfig:
    if payload is None:
        payload = {}
    elif not isinstance(payload, Mapping):
        raise TypeError(f"reconstruct.debug_grad_norm must be a mapping, got {type(payload)!r}")
    else:
        payload = dict(payload)
    parameter_scope = str(payload.get("parameter_scope", "backbone"))
    if parameter_scope not in GRAD_NORM_DEBUG_PARAMETER_SCOPES:
        raise ValueError(
            f"Unsupported reconstruct.debug_grad_norm.parameter_scope={parameter_scope!r}; "
            f"expected one of {sorted(GRAD_NORM_DEBUG_PARAMETER_SCOPES)}"
        )
    every_n_steps = int(payload.get("every_n_steps", 10))
    if every_n_steps <= 0:
        raise ValueError("reconstruct.debug_grad_norm.every_n_steps must be positive")
    return GradNormDebugConfig(
        enabled=bool(payload.get("enabled", False)),
        every_n_steps=every_n_steps,
        parameter_scope=parameter_scope,
    )


def normalize_scalar_head_mode(value: object | None) -> str:
    mode = "per_target" if value is None else str(value).lower()
    if mode not in SCALAR_HEAD_MODES:
        raise ValueError(
            f"Unsupported reconstruct.scalar_head_mode={mode!r}; "
            f"expected one of {sorted(SCALAR_HEAD_MODES)}"
        )
    return mode


class EumpSwinReconstructionTrainer:
    def __init__(
        self,
        *,
        logger,
        device: torch.device,
        model_name: str,
        pretrain_model: str,
        data_config: dict,
        pipeline_config: dict,
        learning_rate: float = 0.0001,
        weight_decay: float = 0.01,
        batch_size: int = 64,
        n_epochs: int = 300,
        random_seed: int = 42,
        mask_ratio: float = 0.6,
        local_rank: int = 0,
        world_size: int = 1,
        accumulation_steps: int = 1,
        scratch_backbone: bool = False,
        save_file_name: str = "eump_reconstruction_model.pt",
        checkpoint_file_name: str = "eump_reconstruction_checkpoint.pt",
        resume: bool = False,
        resume_checkpoint: str | None = None,
        log_dir: str | None = None,
        save_best_model: bool = False,
        save_best_checkpoint: bool = False,
        save_final_model: bool = True,
        best_checkpoint_file_name: str = "eump_reconstruction_best_checkpoint.pt",
        scheduler_config: Mapping[str, object] | None = None,
        debug_grad_norm: Mapping[str, object] | None = None,
        scalar_head_mode: str = "per_target",
        model_config: Mapping[str, object] | None = None,
    ) -> None:
        self.logger = logger
        self.device = device
        self.model_name = model_name
        self.pretrain_model = pretrain_model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.random_seed = random_seed
        self.mask_ratio = mask_ratio
        self.local_rank = local_rank
        self.world_size = world_size
        self.accumulation_steps = accumulation_steps
        self.scratch_backbone = scratch_backbone
        self.save_file_name = save_file_name
        self.checkpoint_file_name = checkpoint_file_name
        self.resume = resume
        self.resume_checkpoint = resume_checkpoint
        self.save_best_model_enabled = bool(save_best_model)
        self.save_best_checkpoint_enabled = bool(save_best_checkpoint)
        self.save_final_model_enabled = bool(save_final_model)
        self.best_checkpoint_file_name = best_checkpoint_file_name
        self.scheduler_config = normalize_warmup_cosine_scheduler_config(scheduler_config)
        self.debug_grad_norm_config = normalize_grad_norm_debug_config(debug_grad_norm)
        self.scalar_head_mode = normalize_scalar_head_mode(scalar_head_mode)
        self.model_config = dict(model_config) if model_config else {}
        self.ddp_find_unused_parameters = bool(self.mask_ratio <= 0.0)
        self.start_epoch = 1
        self.best_val_loss = float("inf")
        self.global_step = 0
        if self.accumulation_steps < 1:
            raise ValueError(f"accumulation_steps must be >= 1, got {self.accumulation_steps}")
        if self.n_epochs < 1:
            raise ValueError(f"n_epochs must be >= 1, got {self.n_epochs}")

        self.pipeline_spec = PipelineSpec.from_mapping(pipeline_config)
        self.loss_config = self.pipeline_spec.loss_config
        self.tensorboard = TensorBoardScalarWriter(
            log_dir=log_dir,
            enabled=self.loss_config.tensorboard,
            rank=self.local_rank,
        )
        self.train_dl, self.val_dl, self.test_dl, self.batch_schema = build_eump_dataloaders(
            data_config=data_config,
            pipeline_spec=self.pipeline_spec,
            logger=self.logger,
            random_seed=self.random_seed,
            batch_size=self.batch_size,
            local_rank=self.local_rank,
            world_size=self.world_size,
        )
        self.model = self.initialize_model(self.batch_schema)
        if self.ddp_find_unused_parameters and self.local_rank == 0:
            self.logger.info(
                "Enable DDP find_unused_parameters for reconstruction because mask_ratio<=0 "
                "can leave Swin mask_token unused."
            )
        self.model = wrap_ddp(
            self.model,
            self.device,
            self.local_rank,
            find_unused_parameters=self.ddp_find_unused_parameters,
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self.scheduler = self.build_scheduler(self.n_epochs)

    def initialize_model(self, batch_schema: ProcessedBatchSchema):
        model_path = os.path.join(MODEL_DIR, self.pretrain_model)
        if batch_schema.raster_input_channels <= 0:
            raise ValueError("EUMP Swin reconstruction requires at least one raster input channel")
        if batch_schema.raster_output_channels <= 0 and not batch_schema.scalar_output_dims:
            raise ValueError("EUMP Swin reconstruction requires at least one configured target")

        model_kind = str(self.model_config.get("kind", "swin_reconstruction"))
        if model_kind == "swin_reconstruction":
            model = EumpSwinReconstructionModel(
                model_path=model_path,
                input_channels=batch_schema.raster_input_channels,
                batch_schema=batch_schema,
                load_pretrained=not self.scratch_backbone,
                scalar_head_mode=self.scalar_head_mode,
            )
            origin = "scratch" if self.scratch_backbone else "pretrained"
            if self.local_rank == 0:
                self.logger.info(f"Init EumpSwinReconstructionModel ({origin}) from {model_path}")
        elif model_kind == "swin_unet_reconstruction":
            model = EumpSwinUnetReconstructionModel(
                model_path=model_path,
                input_channels=batch_schema.raster_input_channels,
                batch_schema=batch_schema,
                load_pretrained=not self.scratch_backbone,
                decoder_stages=int(self.model_config.get("decoder_stages", 5)),
                decoder_min_channels=int(self.model_config.get("decoder_min_channels", 64)),
            )
            origin = "scratch" if self.scratch_backbone else "pretrained"
            if self.local_rank == 0:
                self.logger.info(f"Init EumpSwinUnetReconstructionModel ({origin}) from {model_path}")
        elif model_kind == "convnext_unet_reconstruction":
            model = EumpConvNextUnetReconstructionModel(
                input_channels=batch_schema.raster_input_channels,
                batch_schema=batch_schema,
                depths=tuple(self.model_config.get("depths", (3, 3, 27, 3))),
                dims=tuple(self.model_config.get("dims", (192, 384, 768, 1536))),
                drop_path_rate=float(self.model_config.get("drop_path_rate", 0.0)),
                decoder_stages=int(self.model_config.get("decoder_stages", 5)),
                decoder_min_channels=int(self.model_config.get("decoder_min_channels", 64)),
            )
            if self.local_rank == 0:
                self.logger.info(
                    "Init EumpConvNextUnetReconstructionModel (scratch) "
                    f"depths={self.model_config.get('depths')} dims={self.model_config.get('dims')}"
                )
        else:
            raise ValueError(
                f"Unsupported model.kind={model_kind!r} for EumpSwinReconstructionTrainer; "
                "expected one of swin_reconstruction, swin_unet_reconstruction, "
                "convnext_unet_reconstruction"
            )
        return model.to(self.device)

    def build_scheduler(self, n_epochs=None):
        schedule_epochs = self.n_epochs if n_epochs is None else n_epochs
        scheduler_config = getattr(self, "scheduler_config", normalize_warmup_cosine_scheduler_config(None))
        scheduler, metadata = build_warmup_cosine_scheduler(
            optimizer=self.optimizer,
            base_lr=self.learning_rate,
            num_batches=len(self.train_dl),
            accumulation_steps=self.accumulation_steps,
            epochs=schedule_epochs,
            warmup_fraction=scheduler_config.warmup_fraction,
            lr_start_factor=scheduler_config.lr_start_factor,
            lr_end_factor=scheduler_config.lr_end_factor,
        )
        self.scheduler_metadata = metadata
        return scheduler

    def current_scheduler_metadata(self):
        return getattr(self, "scheduler_metadata", None)

    def scheduler_metadata_for_epochs(self, n_epochs):
        scheduler_config = getattr(self, "scheduler_config", normalize_warmup_cosine_scheduler_config(None))
        return warmup_cosine_metadata(
            base_lr=self.learning_rate,
            num_batches=len(self.train_dl),
            accumulation_steps=self.accumulation_steps,
            epochs=n_epochs,
            warmup_fraction=scheduler_config.warmup_fraction,
            lr_start_factor=scheduler_config.lr_start_factor,
            lr_end_factor=scheduler_config.lr_end_factor,
        )

    def reset_scheduler(self, n_epochs):
        self.scheduler = self.build_scheduler(n_epochs)

    def _build_bool_mask(self, batch: ProcessedModelBatch) -> torch.Tensor | None:
        base_model = get_model(self.model)
        image_size = int(batch.raster_inputs.shape[-1])
        patch_size = int(base_model.config.patch_size)
        num_patches = (image_size // patch_size) ** 2
        num_masked_patches = math.ceil(num_patches * self.mask_ratio)
        return generate_mask(
            batch.raster_inputs.shape[0],
            num_patches,
            num_masked_patches,
            device=self.device,
        )

    def _masked_patch_tensor(
        self,
        bool_masked_pos: torch.Tensor | None,
        *,
        batch_size: int,
        height: int,
        width: int,
    ) -> torch.Tensor | None:
        if bool_masked_pos is None:
            return None
        base_model = get_model(self.model)
        patch_size = int(base_model.config.patch_size)
        patch_h = height // patch_size
        patch_w = width // patch_size
        mask = bool_masked_pos.reshape(batch_size, patch_h, patch_w)
        return (
            mask.repeat_interleave(patch_size, 1)
            .repeat_interleave(patch_size, 2)
            .unsqueeze(1)
            .contiguous()
        )

    def _split_raster_predictions(self, raster_prediction: torch.Tensor) -> dict[str, torch.Tensor]:
        predictions: dict[str, torch.Tensor] = {}
        start = 0
        for target in self.batch_schema.raster_targets:
            end = start + target.prediction_channels
            predictions[target.name] = raster_prediction[:, start:end, ...]
            start = end
        if start != raster_prediction.shape[1]:
            raise ValueError(
                f"Raster prediction channels consumed {start}, but model returned {raster_prediction.shape[1]}"
            )
        return predictions

    def _target_loss(
        self,
        *,
        target: TargetSchema,
        prediction: torch.Tensor,
        target_data: torch.Tensor,
        valid_mask: torch.Tensor,
        city_mask: torch.Tensor | None,
        masked_patch: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
        return compute_target_loss(
            target=target,
            prediction=prediction,
            target_data=target_data,
            valid_mask=valid_mask,
            city_mask=city_mask,
            masked_patch=masked_patch,
            loss_config=self._loss_config(),
        )

    def _loss_config(self) -> PipelineLossConfig:
        return getattr(self, "loss_config", PipelineLossConfig())

    def _aggregate_loss_report(
        self,
        *,
        target_raw: dict[str, torch.Tensor],
        target_weighted: dict[str, torch.Tensor],
        target_groups: dict[str, str],
        target_weights: dict[str, torch.Tensor | float],
        target_sources: dict[str, str],
        target_debug_terms: dict[str, dict[str, torch.Tensor]],
        mask_stats: dict[str, torch.Tensor],
        zero: torch.Tensor,
    ) -> LossReport:
        return aggregate_loss_report(
            loss_config=self._loss_config(),
            target_raw=target_raw,
            target_weighted=target_weighted,
            target_groups=target_groups,
            target_weights=target_weights,
            target_sources=target_sources,
            target_debug_terms=target_debug_terms,
            mask_stats=mask_stats,
            zero=zero,
        )

    def _compute_loss(
        self,
        batch: ProcessedModelBatch,
        raster_prediction: torch.Tensor,
        scalar_predictions: dict[str, torch.Tensor],
        bool_masked_pos: torch.Tensor | None,
    ) -> LossReport:
        batch_size, _, height, width = batch.raster_inputs.shape
        masked_patch = self._masked_patch_tensor(
            bool_masked_pos,
            batch_size=batch_size,
            height=height,
            width=width,
        )
        raster_predictions = self._split_raster_predictions(raster_prediction)
        zero = raster_prediction.sum() * 0.0
        target_raw: dict[str, torch.Tensor] = {}
        target_weighted: dict[str, torch.Tensor] = {}
        target_groups: dict[str, str] = {}
        target_weights: dict[str, torch.Tensor | float] = {}
        target_sources: dict[str, str] = {}
        target_debug_terms: dict[str, dict[str, torch.Tensor]] = {}
        mask_stats: dict[str, torch.Tensor] = {
            "mask/city_mask_coverage": coverage(batch.city_mask > 0, like=zero),
            "mask/masked_mask_coverage": coverage(masked_patch, like=zero),
        }

        for target in self.batch_schema.raster_targets:
            target_block = batch.raster_targets[target.name]
            raw_loss, stats, support_weight, debug_terms = self._target_loss(
                target=target,
                prediction=raster_predictions[target.name],
                target_data=target_block.data,
                valid_mask=target_block.valid_mask,
                city_mask=batch.city_mask,
                masked_patch=masked_patch,
            )
            weight = float(target.spec.weight)
            effective_weight = raw_loss.new_tensor(weight) * support_weight
            target_raw[target.name] = raw_loss
            target_weighted[target.name] = raw_loss * effective_weight
            target_groups[target.name] = target.spec.group
            target_weights[target.name] = effective_weight
            target_sources[target.name] = target.spec.source or target.name
            target_debug_terms[target.name] = debug_terms
            mask_stats.update(stats)

        for target in self.batch_schema.scalar_targets:
            target_block = batch.scalar_targets[target.name]
            if target.name not in scalar_predictions:
                raise KeyError(f"Missing scalar prediction for target {target.name!r}")
            raw_loss, stats, support_weight, debug_terms = self._target_loss(
                target=target,
                prediction=scalar_predictions[target.name],
                target_data=target_block.data,
                valid_mask=target_block.valid_mask,
                city_mask=None,
                masked_patch=None,
            )
            weight = float(target.spec.weight)
            effective_weight = raw_loss.new_tensor(weight) * support_weight
            target_raw[target.name] = raw_loss
            target_weighted[target.name] = raw_loss * effective_weight
            target_groups[target.name] = target.spec.group
            target_weights[target.name] = effective_weight
            target_sources[target.name] = target.spec.source or target.name
            target_debug_terms[target.name] = debug_terms
            mask_stats.update(stats)

        return self._aggregate_loss_report(
            target_raw=target_raw,
            target_weighted=target_weighted,
            target_groups=target_groups,
            target_weights=target_weights,
            target_sources=target_sources,
            target_debug_terms=target_debug_terms,
            mask_stats=mask_stats,
            zero=zero,
        )

    def _debug_grad_norm_parameters(self) -> list[torch.nn.Parameter]:
        base_model = get_model(self.model)
        config = getattr(self, "debug_grad_norm_config", GradNormDebugConfig())
        if config.parameter_scope == "backbone" and hasattr(base_model, "backbone"):
            parameters: Iterable[torch.nn.Parameter] = base_model.backbone.parameters()
        else:
            parameters = base_model.parameters()
        return [parameter for parameter in parameters if parameter.requires_grad]

    def _should_compute_grad_norm_debug(self, *, train: bool, batch_idx: int, num_batches: int) -> bool:
        config = getattr(self, "debug_grad_norm_config", GradNormDebugConfig())
        if not train or not config.enabled:
            return False
        is_optimizer_step = (batch_idx + 1) % self.accumulation_steps == 0 or (batch_idx + 1) == num_batches
        if not is_optimizer_step:
            return False
        next_step = int(getattr(self, "global_step", 0)) + 1
        return next_step == 1 or next_step % config.every_n_steps == 0

    def _grad_l2_norm(self, term: torch.Tensor, parameters: list[torch.nn.Parameter]) -> torch.Tensor:
        if not parameters:
            return term.detach().new_tensor(0.0)
        if not torch.is_tensor(term) or not term.requires_grad:
            return parameters[0].detach().new_tensor(0.0)
        gradients = torch.autograd.grad(
            term,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        norm_sq = parameters[0].detach().new_tensor(0.0)
        for gradient in gradients:
            if gradient is None:
                continue
            norm_sq = norm_sq + gradient.detach().float().pow(2).sum()
        return norm_sq.sqrt()

    def _compute_grad_norm_debug_metrics(self, report: LossReport) -> dict[str, torch.Tensor]:
        parameters = self._debug_grad_norm_parameters()
        if not parameters or not report.grad_debug_terms:
            return {}

        metrics: dict[str, torch.Tensor] = {}
        for name, term in report.grad_debug_terms.items():
            metrics[f"grad_norm_debug/{name}"] = self._grad_l2_norm(term, parameters)

        total_norm = metrics.get("grad_norm_debug/total")
        if total_norm is not None:
            denom = total_norm.clamp_min(1.0e-12)
            for metric_name, value in list(metrics.items()):
                if metric_name == "grad_norm_debug/total":
                    continue
                suffix = metric_name.removeprefix("grad_norm_debug/")
                metrics[f"grad_norm_debug_ratio_to_total/{suffix}"] = value / denom

        target_norms = [
            value
            for metric_name, value in metrics.items()
            if metric_name.startswith("grad_norm_debug/target/")
            or metric_name.startswith("grad_norm_debug/slice/")
        ]
        if target_norms:
            stacked_norms = torch.stack([value.detach().float() for value in target_norms])
            positive = stacked_norms > 0
            positive_count = positive.float().sum()
            max_norm = torch.where(positive, stacked_norms, stacked_norms.new_tensor(0.0)).max()
            min_norm = torch.where(positive, stacked_norms, stacked_norms.new_tensor(float("inf"))).min()
            ratio = torch.where(
                positive_count >= 2,
                max_norm / min_norm.clamp_min(1.0e-12),
                stacked_norms.new_tensor(0.0),
            )
        else:
            ratio = parameters[0].detach().new_tensor(0.0)
        metrics["grad_norm_debug/max_target_to_min_target_ratio"] = ratio
        return metrics

    def _accumulate_metrics(
        self,
        sums: dict[str, torch.Tensor],
        counts: dict[str, int],
        metrics: Mapping[str, torch.Tensor],
    ) -> None:
        accumulate_metrics(sums, counts, metrics)

    def _average_metrics(self, sums: dict[str, torch.Tensor], counts: dict[str, int]) -> dict[str, torch.Tensor]:
        return average_metrics(sums, counts, device=self.device)

    def _current_learning_rate(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _write_tensorboard(
        self,
        *,
        phase: str,
        level: str,
        metrics: Mapping[str, torch.Tensor | float | int],
        step: int,
    ) -> None:
        writer = getattr(self, "tensorboard", None)
        if writer is None:
            return
        scalars = {name: tensor_to_float(value) for name, value in metrics.items()}
        writer.add_scalars(phase=phase, level=level, scalars=scalars, step=step)

    def _run_epoch(self, dataloader, *, train: bool, epoch: int, phase: str) -> dict[str, torch.Tensor]:
        dataset = getattr(dataloader, "dataset", None)
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)
        sampler = getattr(dataloader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        self.model.train(mode=train)
        metric_sums: dict[str, torch.Tensor] = {}
        metric_counts: dict[str, int] = {}

        if train:
            self.optimizer.zero_grad()

        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for batch_idx, batch in enumerate(dataloader):
                batch = batch.to(self.device)
                bool_masked_pos = self._build_bool_mask(batch)
                raster_prediction, scalar_predictions = self.model(batch.raster_inputs, bool_masked_pos)
                report = self._compute_loss(
                    batch,
                    raster_prediction,
                    scalar_predictions,
                    bool_masked_pos,
                )
                loss = report.total
                grad_norm: torch.Tensor | None = None
                grad_debug_metrics: dict[str, torch.Tensor] = {}

                if train:
                    if self._should_compute_grad_norm_debug(
                        train=train,
                        batch_idx=batch_idx,
                        num_batches=len(dataloader),
                    ):
                        grad_debug_metrics = self._compute_grad_norm_debug_metrics(report)
                        if grad_debug_metrics:
                            self._write_tensorboard(
                                phase=phase,
                                level="step",
                                metrics=grad_debug_metrics,
                                step=int(getattr(self, "global_step", 0)) + 1,
                            )
                    scaled_loss = loss / self.accumulation_steps
                    scaled_loss.backward()
                    if (batch_idx + 1) % self.accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                        clipped_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        grad_norm = clipped_norm.detach()
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                        self.global_step = getattr(self, "global_step", 0) + 1

                batch_metrics = report.metric_tensors()
                if grad_debug_metrics:
                    batch_metrics.update(grad_debug_metrics)
                if train and grad_norm is not None:
                    batch_metrics["state/grad_norm"] = grad_norm
                self._accumulate_metrics(metric_sums, metric_counts, batch_metrics)

        epoch_metrics = self._average_metrics(metric_sums, metric_counts)
        tensorboard_metrics: dict[str, torch.Tensor | float | int] = dict(epoch_metrics)
        tensorboard_metrics.update(
            {
                "state/learning_rate": self._current_learning_rate(),
                "state/epoch": epoch,
                "state/global_step": getattr(self, "global_step", 0),
            }
        )
        self._write_tensorboard(
            phase=phase,
            level="epoch",
            metrics=tensorboard_metrics,
            step=epoch,
        )
        return epoch_metrics

    def format_epoch_log(self, phase: str, metrics: Mapping[str, float]) -> str:
        reconstruct = float(metrics.get("loss/group_weighted/reconstruct", 0.0))
        aux = float(metrics.get("loss/group_weighted/aux", 0.0))
        total = float(metrics.get("loss/total", 0.0))
        return f"{phase} reconstruct={reconstruct:.6f} aux={aux:.6f} total={total:.6f}"

    def format_training_summary(self, duration_minutes: float) -> str:
        return f"Final model saved, Total Training Time: {duration_minutes:.2f} mins"

    def close(self) -> None:
        writer = getattr(self, "tensorboard", None)
        if writer is not None:
            writer.close()

    def train_epoch(self, epoch):
        return self._run_epoch(self.train_dl, train=True, epoch=epoch, phase="train")

    def validate_epoch(self, epoch):
        return self._run_epoch(self.val_dl, train=False, epoch=epoch, phase="val")

    def test_epoch(self, epoch):
        return self._run_epoch(self.test_dl, train=False, epoch=epoch, phase="test")


__all__ = [
    "GradNormDebugConfig",
    "LossReport",
    "EumpSwinReconstructionTrainer",
    "normalize_grad_norm_debug_config",
    "normalize_scalar_head_mode",
]
