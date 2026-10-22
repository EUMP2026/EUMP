from __future__ import annotations

import os
from collections.abc import Mapping
from copy import deepcopy
import math
from types import SimpleNamespace

import torch

from urban_embedding.core.paths import MODEL_DIR, resolve_model_artifact_path
from urban_embedding.data.processed.batch import ProcessedModelBatch
from urban_embedding.data.processed.dataset import build_eump_dataloaders
from urban_embedding.data.processed.pipeline_spec import (
    PipelineLossConfig,
    PipelineSpec,
    ProcessedBatchSchema,
    TargetSchema,
    build_active_preprocessing_spec,
    get_single_downstream_target_schema,
    validate_eump_downstream_spec,
)
from urban_embedding.data.processed.processing import (
    load_preprocessing_spec,
    validate_preprocessing_spec,
    validate_preprocessing_stats_scope,
)
from urban_embedding.models.eump.swin.downstream import build_eump_downstream_model
from urban_embedding.training.distributed import wrap_ddp
from urban_embedding.training.eump_loss import (
    LossReport,
    accumulate_metrics,
    aggregate_loss_report,
    average_metrics,
    compute_target_loss,
    coverage,
    tensor_to_float,
)
from urban_embedding.training.regression_metrics import (
    accumulate_regression_stats,
    empty_regression_stats,
    inverse_processed_tensor,
    raster_city_mean_metric_tensors,
    raster_global_metric_tensors,
    scalar_metric_tensors,
    summarize_regression_stats,
)
from urban_embedding.training.losses import compute_weighted_loss
from urban_embedding.training.schedulers import (
    build_warmup_cosine_scheduler,
    normalize_warmup_cosine_scheduler_config,
    warmup_cosine_metadata,
)
from urban_embedding.training.tensorboard import TensorBoardScalarWriter


class EumpDownstreamTrainer:
    def __init__(
        self,
        *,
        logger,
        device: torch.device,
        model_name: str,
        pretrain_model: str,
        data_config: dict,
        pipeline_config: dict,
        model_config: dict | None = None,
        learning_rate: float = 0.00025,
        weight_decay: float = 0.0001,
        batch_size: int = 64,
        n_epochs: int = 100,
        random_seed: int = 42,
        local_rank: int = 0,
        world_size: int = 1,
        accumulation_steps: int = 1,
        scratch_backbone: bool = False,
        save_file_name: str = "eump_downstream_model.pt",
        checkpoint_file_name: str = "eump_downstream_checkpoint.pt",
        resume: bool = False,
        resume_checkpoint: str | None = None,
        log_dir: str | None = None,
        encoder_source: str = "reconstruction_checkpoint",
        encoder_checkpoint_file: str | None = None,
        encoder_model_name: str | None = None,
        freeze_encoder: bool = True,
        selection_config: dict | None = None,
        scheduler_config: dict | None = None,
    ) -> None:
        self.logger = logger
        self.device = device
        self.model_name = model_name
        self.pretrain_model = pretrain_model
        self.pipeline_config = deepcopy(pipeline_config)
        self.model_config = {} if model_config is None else dict(model_config)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.random_seed = random_seed
        self.local_rank = local_rank
        self.world_size = world_size
        self.accumulation_steps = accumulation_steps
        self.scratch_backbone = scratch_backbone
        self.save_file_name = save_file_name
        self.checkpoint_file_name = checkpoint_file_name
        self.resume = resume
        self.resume_checkpoint = resume_checkpoint
        self.encoder_source = encoder_source
        self.encoder_checkpoint_file = encoder_checkpoint_file
        self.encoder_model_name = encoder_model_name or model_name
        self.freeze_encoder = freeze_encoder
        self.selection_config = self._normalize_selection_config(selection_config)
        self.scheduler_config = normalize_warmup_cosine_scheduler_config(scheduler_config)
        self.save_best_model_enabled = True
        self.save_final_model_enabled = False
        self.start_epoch = 1
        self.best_val_loss = float("inf")
        self.best_selection_metric = float("inf")
        self.best_selection_epoch: int | None = None
        self.selection_history: list[float] = []
        self.selection_metric_name = self._raw_selection_metric_name()
        self.best_selection_metric_name = self.selection_metric_name
        self.global_step = 0
        if self.accumulation_steps < 1:
            raise ValueError(f"accumulation_steps must be >= 1, got {self.accumulation_steps}")
        if self.n_epochs < 1:
            raise ValueError(f"n_epochs must be >= 1, got {self.n_epochs}")

        self.pipeline_spec = PipelineSpec.from_mapping(pipeline_config)
        validate_eump_downstream_spec(self.pipeline_spec)
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
        self.target_schema = get_single_downstream_target_schema(self.batch_schema)
        self.target_kind = "raster" if self._is_raster_target() else "scalar"
        self._validate_selection_config_for_target()
        self.preprocessing_spec = self._active_preprocessing_spec(data_config)
        self.source_spec = self._target_source_spec()
        self.model = self.initialize_model(self.batch_schema, self.target_schema)
        self.model = wrap_ddp(self.model, self.device, self.local_rank)
        self.selection_val_dl = (
            self._build_selection_val_loader(data_config)
            if self._uses_selection_val_loader()
            else None
        )

        trainable_params = [param for param in self.model.parameters() if param.requires_grad]
        if not trainable_params:
            raise ValueError("EUMP downstream model has no trainable parameters")
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self.scheduler = self.build_scheduler(self.n_epochs)

    def initialize_model(self, batch_schema: ProcessedBatchSchema, target_schema: TargetSchema):
        if batch_schema.raster_input_channels <= 0:
            raise ValueError("EUMP downstream requires at least one raster input channel")
        self._validate_scalar_input_consumption(batch_schema)

        model_path = os.path.join(MODEL_DIR, self.pretrain_model)
        checkpoint_path = None
        if self.encoder_source in {
            "reconstruction_checkpoint",
            "convnext_reconstruction_checkpoint",
            "convnext_pretrained",
        }:
            if self.encoder_checkpoint_file is None:
                raise ValueError(
                    "downstream.encoder_checkpoint_file is required when loading a pretrained encoder"
                )
            checkpoint_path = resolve_model_artifact_path(self.encoder_model_name, self.encoder_checkpoint_file)

        model = build_eump_downstream_model(
            model_config=self.model_config,
            batch_schema=batch_schema,
            target_schema=target_schema,
            base_swin_path=model_path,
            encoder_source=self.encoder_source,
            encoder_checkpoint_path=checkpoint_path,
            load_pretrained=not self.scratch_backbone,
            freeze_encoder=self.freeze_encoder,
            map_location=self.device,
        )
        if self.local_rank == 0:
            if checkpoint_path is not None:
                self.logger.info(f"Load EUMP downstream encoder from: {checkpoint_path}")
            elif self.encoder_source == "convnext_scratch":
                self.logger.info("Initialize ConvNeXt downstream encoder from scratch")
            else:
                self.logger.info(f"Load EUMP downstream encoder base model from: {model_path}")
            self.logger.info(
                "Initialize EUMP downstream "
                f"arch={self.model_config.get('downstream_arch', 'light_dual_head')} "
                f"freeze_encoder={self.freeze_encoder}"
            )
        return model.to(self.device)

    def _normalize_selection_config(self, selection_config: dict | None) -> dict[str, object]:
        selection = {} if selection_config is None else dict(selection_config)
        metric = str(selection.get("metric", "val_loss")).lower()
        if metric not in {"val_loss", "rmse"}:
            raise ValueError(
                "eump_downstream selection.metric must be 'val_loss' or 'rmse', "
                f"got {metric!r}"
            )
        metric_scale = str(selection.get("metric_scale", "physical")).lower()
        if metric_scale not in {"physical", "processed"}:
            raise ValueError(
                "eump_downstream selection.metric_scale must be 'physical' or 'processed', "
                f"got {metric_scale!r}"
            )
        loss_view = str(selection.get("loss_view", "default")).lower()
        if loss_view not in {"default", "global_pixels", "city_mean"}:
            raise ValueError(
                "eump_downstream selection.loss_view must be 'default', 'global_pixels', or 'city_mean', "
                f"got {loss_view!r}"
            )
        if metric != "val_loss" and loss_view != "default":
            raise ValueError("eump_downstream selection.loss_view is supported only when selection.metric='val_loss'")
        raster_view = str(selection.get("raster_view", "global_pixels")).lower()
        if raster_view != "global_pixels":
            raise ValueError(
                "eump_downstream selection.raster_view currently supports only 'global_pixels', "
                f"got {raster_view!r}"
            )
        smoothing_window = int(selection.get("smoothing_window", 1))
        if smoothing_window < 1:
            raise ValueError(
                f"eump_downstream selection.smoothing_window must be >= 1, got {smoothing_window}"
            )
        return {
            "metric": metric,
            "metric_scale": metric_scale,
            "loss_view": loss_view,
            "center_ratio": float(selection.get("center_ratio", 1.0)),
            "raster_view": raster_view,
            "smoothing_window": smoothing_window,
        }

    def _uses_selection_val_loader(self) -> bool:
        if self.selection_config["metric"] == "rmse":
            return True
        return self.selection_config["metric"] == "val_loss" and self.selection_config["loss_view"] != "default"

    def _validate_selection_config_for_target(self) -> None:
        loss_view = str(self.selection_config["loss_view"])
        if loss_view == "default":
            return
        if self.target_kind != "raster":
            raise ValueError(
                "eump_downstream selection.loss_view='global_pixels'/'city_mean' is supported only "
                "for raster downstream targets"
            )
        loss = str(self.target_schema.spec.loss).lower()
        if loss_view == "city_mean" and loss not in {"mse", "mae", "huber"}:
            raise ValueError(
                "eump_downstream selection.loss_view='city_mean' supports only regression losses "
                f"'mse', 'mae', or 'huber', got {loss!r}"
            )

    def _raw_selection_metric_name(self) -> str:
        metric = str(self.selection_config["metric"])
        if metric == "val_loss":
            return "selection/val_loss"
        if metric == "rmse":
            return "selection/rmse"
        raise ValueError(f"Unsupported downstream selection metric {metric!r}")

    def _smoothed_selection_metric_name(self) -> str:
        return f"{self._raw_selection_metric_name()}_smoothed"

    def _target_source_spec(self) -> dict:
        source_name = self.target_schema.spec.source
        if source_name is None:
            raise ValueError(f"Downstream selection target {self.target_schema.name!r} requires a source")
        source_spec = self.preprocessing_spec["sources"].get(source_name)
        if source_spec is None:
            raise KeyError(f"Source {source_name!r} is missing from preprocessing spec")
        return source_spec

    def _active_preprocessing_spec(self, data_config: dict) -> dict:
        dataset = getattr(self.train_dl, "dataset", None)
        preprocessing_spec = getattr(dataset, "preprocessing_spec", None)
        if preprocessing_spec is not None:
            return preprocessing_spec

        preprocessing_spec = load_preprocessing_spec(data_config.get("preprocessing_path", data_config["eump_root"]))
        validate_preprocessing_spec(preprocessing_spec)
        validate_preprocessing_stats_scope(
            preprocessing_spec,
            split_manifest_path=data_config.get("split_manifest_path"),
            expected_year_range=self.pipeline_spec.year_range,
        )
        reader = getattr(dataset, "reader", None)
        if reader is None:
            return preprocessing_spec
        return build_active_preprocessing_spec(self.pipeline_spec, reader.source_specs, preprocessing_spec)

    def _selection_logger(self):
        return SimpleNamespace(info=lambda *args, **kwargs: None)

    def _build_selection_val_loader(self, data_config: dict):
        target_center_ratio = float(self.selection_config["center_ratio"])
        if math.isclose(float(self.pipeline_spec.center_ratio), target_center_ratio, rel_tol=0.0, abs_tol=1.0e-12):
            return self.val_dl

        selection_pipeline_config = deepcopy(self.pipeline_config)
        selection_pipeline_config["center_ratio"] = target_center_ratio
        selection_pipeline_spec = PipelineSpec.from_mapping(selection_pipeline_config)
        validate_eump_downstream_spec(selection_pipeline_spec)
        _, selection_val_dl, _, selection_schema = build_eump_dataloaders(
            data_config=data_config,
            pipeline_spec=selection_pipeline_spec,
            logger=self._selection_logger(),
            random_seed=self.random_seed,
            batch_size=self.batch_size,
            local_rank=self.local_rank,
            world_size=self.world_size,
        )
        if selection_schema.raster_input_channel_names != self.batch_schema.raster_input_channel_names:
            raise ValueError("Selection validation loader changed raster input schema unexpectedly")
        if selection_schema.scalar_output_dims != self.batch_schema.scalar_output_dims:
            raise ValueError("Selection validation loader changed scalar target schema unexpectedly")
        if tuple(target.name for target in selection_schema.raster_targets) != tuple(
            target.name for target in self.batch_schema.raster_targets
        ):
            raise ValueError("Selection validation loader changed raster target schema unexpectedly")
        return selection_val_dl

    def _validate_scalar_input_consumption(self, batch_schema: ProcessedBatchSchema) -> None:
        if batch_schema.scalar_input_dim <= 0 or self.pipeline_spec.broadcast_scalar:
            return

        arch = str(self.model_config.get("downstream_arch", "light_dual_head"))
        scalar_input_consuming_arches: set[str] = set()
        if arch in scalar_input_consuming_arches:
            return

        raise ValueError(
            "eump_downstream has scalar input sources "
            f"{batch_schema.scalar_input_names}, but model.downstream_arch={arch!r} "
            "does not consume scalar_inputs directly. Set pipeline.broadcast_scalar=true "
            "to route scalar covariates through the encoder, or remove scalar sources "
            "from pipeline.inputs.sources."
        )

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

    def _metric_tensor(self, values: torch.Tensor) -> torch.Tensor:
        if self.selection_config["metric_scale"] == "physical":
            return inverse_processed_tensor(values, self.source_spec)
        return values

    def _reduce_selection_stats(self, stats: torch.Tensor) -> torch.Tensor:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
        return stats

    def _selection_loader_epoch(self, epoch: int) -> None:
        dataloader = self.selection_val_dl
        if dataloader is None:
            return
        dataset = getattr(dataloader, "dataset", None)
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)
        sampler = getattr(dataloader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

    def _reduce_selection_loss(self, loss_sum: torch.Tensor, count: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        stats = torch.stack([loss_sum.detach(), count.detach()]).to(self.device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
        reduced_loss_sum, reduced_count = stats[0], stats[1]
        if reduced_count <= 0:
            raise ValueError(
                f"EUMP downstream selection found no valid validation samples for target {self.target_schema.name!r}"
            )
        return reduced_loss_sum / reduced_count.clamp_min(1.0), reduced_count

    def _compute_global_pixels_selection_loss(
        self,
        batch: ProcessedModelBatch,
        raster_predictions: dict[str, torch.Tensor],
        scalar_predictions: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        report = self._compute_loss(batch, raster_predictions, scalar_predictions)
        return report.total.detach(), report.total.detach().new_tensor(1.0)

    def _compute_city_mean_selection_loss(
        self,
        batch: ProcessedModelBatch,
        raster_predictions: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prediction = raster_predictions[self.target_schema.name]
        target_block = batch.raster_targets[self.target_schema.name]
        pred_values, target_values = raster_city_mean_metric_tensors(
            prediction=prediction,
            target=target_block.data,
            valid_mask=target_block.valid_mask,
            city_mask=batch.city_mask,
            mask_strategy=self.target_schema.spec.mask,
        )
        count = pred_values.detach().new_tensor(float(pred_values.numel()))
        if pred_values.numel() == 0:
            return prediction.sum().detach() * 0.0, count
        raw_loss = compute_weighted_loss(
            str(self.target_schema.spec.loss),
            pred_values,
            target_values,
            mask=None,
        )
        if float(self.target_schema.spec.weight) <= 0.0:
            return raw_loss.detach() * 0.0, count
        group_weight = float(self.loss_config.group_weight(self.target_schema.spec.group))
        return (raw_loss * group_weight).detach() * count, count

    def _compute_eval_like_selection_loss(self, epoch: int) -> dict[str, float]:
        if self.selection_val_dl is None:
            raise ValueError("Eval-like downstream loss selection requires a selection validation dataloader")
        self._selection_loader_epoch(epoch)
        loss_sum = torch.zeros((), device=self.device)
        count = torch.zeros((), device=self.device)
        loss_view = str(self.selection_config["loss_view"])
        self.model.eval()
        with torch.no_grad():
            for batch in self.selection_val_dl:
                batch = batch.to(self.device)
                raster_predictions, scalar_predictions = self.model(batch.raster_inputs, batch.scalar_inputs)
                if loss_view == "global_pixels":
                    batch_loss_sum, batch_count = self._compute_global_pixels_selection_loss(
                        batch,
                        raster_predictions,
                        scalar_predictions,
                    )
                elif loss_view == "city_mean":
                    batch_loss_sum, batch_count = self._compute_city_mean_selection_loss(batch, raster_predictions)
                else:
                    raise ValueError(f"Unsupported eval-like selection loss_view {loss_view!r}")
                loss_sum = loss_sum + batch_loss_sum.to(self.device)
                count = count + batch_count.to(self.device)

        loss, reduced_count = self._reduce_selection_loss(loss_sum, count)
        return {
            "selection/val_loss": float(loss.detach().cpu().item()),
            "selection/count": float(reduced_count.detach().cpu().item()),
        }

    def compute_selection_metrics(self, epoch: int) -> dict[str, float] | None:
        if self.selection_config["metric"] == "val_loss":
            if self.selection_config["loss_view"] != "default":
                return self._compute_eval_like_selection_loss(epoch)
            return None
        if self.selection_val_dl is None:
            raise ValueError("RMSE downstream selection requires a selection validation dataloader")
        self._selection_loader_epoch(epoch)
        stats = empty_regression_stats(device=self.device)
        self.model.eval()
        with torch.no_grad():
            for batch in self.selection_val_dl:
                batch = batch.to(self.device)
                raster_predictions, scalar_predictions = self.model(batch.raster_inputs, batch.scalar_inputs)
                if self.target_kind == "scalar":
                    prediction = self._metric_tensor(scalar_predictions[self.target_schema.name])
                    target_block = batch.scalar_targets[self.target_schema.name]
                    target_data = self._metric_tensor(target_block.data)
                    pred_values, target_values = scalar_metric_tensors(
                        prediction=prediction,
                        target=target_data,
                        valid_mask=target_block.valid_mask,
                        mask_strategy=self.target_schema.spec.mask,
                    )
                else:
                    prediction = self._metric_tensor(raster_predictions[self.target_schema.name])
                    target_block = batch.raster_targets[self.target_schema.name]
                    target_data = self._metric_tensor(target_block.data)
                    pred_values, target_values = raster_global_metric_tensors(
                        prediction=prediction,
                        target=target_data,
                        valid_mask=target_block.valid_mask,
                        city_mask=batch.city_mask,
                        mask_strategy=self.target_schema.spec.mask,
                    )
                accumulate_regression_stats(stats, pred_values, target_values)

        summary, _ = summarize_regression_stats(self._reduce_selection_stats(stats))
        if summary["count"] <= 0 or summary["rmse"] is None:
            raise ValueError(
                f"EUMP downstream selection found no valid validation samples for target {self.target_schema.name!r}"
            )
        r2 = summary["r2"]
        return {
            "selection/rmse": float(summary["rmse"]),
            "selection/r2": float("nan") if r2 is None else float(r2),
            "selection/count": float(summary["count"]),
        }

    def process_epoch_selection(
        self,
        *,
        epoch: int,
        val_metrics: Mapping[str, float],
    ) -> dict[str, object]:
        val_loss = float(val_metrics["loss/total"])
        self.best_val_loss = min(float(getattr(self, "best_val_loss", float("inf"))), val_loss)

        raw_metric_name = self._raw_selection_metric_name()
        if self.selection_config["metric"] == "val_loss":
            raw_metric_value = float(val_metrics.get("selection/val_loss", val_loss))
        else:
            raw_metric_value = float(val_metrics["selection/rmse"])
        self.selection_history.append(raw_metric_value)

        smoothing_window = int(self.selection_config["smoothing_window"])
        result: dict[str, object] = {
            "save_best": False,
            "message": None,
            "extra_metrics": {raw_metric_name: raw_metric_value},
        }
        metric_value = raw_metric_value
        metric_name = raw_metric_name
        if smoothing_window > 1:
            if len(self.selection_history) < smoothing_window:
                result["extra_metrics"][self._smoothed_selection_metric_name()] = float("nan")
                return result
            metric_value = sum(self.selection_history[-smoothing_window:]) / smoothing_window
            metric_name = self._smoothed_selection_metric_name()
            result["extra_metrics"][metric_name] = metric_value

        if metric_value < self.best_selection_metric:
            self.best_selection_metric = metric_value
            self.best_selection_epoch = epoch
            self.best_selection_metric_name = metric_name
            result["save_best"] = True
            result["message"] = (
                f"Best model saved with {metric_name} {metric_value:.6f} "
                f"(raw {raw_metric_name}={raw_metric_value:.6f}) at epoch {epoch}"
            )
        return result

    def training_state_dict(self) -> dict[str, object]:
        return {
            "best_selection_metric": float(self.best_selection_metric),
            "best_selection_epoch": self.best_selection_epoch,
            "selection_metric_name": self.best_selection_metric_name,
            "selection_history": list(self.selection_history),
            "selection_config": dict(self.selection_config),
        }

    def load_training_state_dict(self, state: Mapping[str, object]) -> None:
        saved_config = state.get("selection_config")
        if not isinstance(saved_config, Mapping):
            self._reset_selection_state_on_resume(
                "Resume checkpoint has no downstream selection_config snapshot. "
                "Resetting best-selection state to avoid comparing metrics across unknown criteria."
            )
            return

        try:
            normalized_saved_config = self._normalize_selection_config(dict(saved_config))
        except Exception as exc:
            self._reset_selection_state_on_resume(
                "Resume checkpoint downstream selection_config is invalid "
                f"({exc}). Resetting best-selection state."
            )
            return
        if normalized_saved_config != self.selection_config:
            self._reset_selection_state_on_resume(
                "Resume checkpoint downstream selection_config differs from the current config. "
                "Resetting best-selection state to avoid comparing metrics across criteria."
            )
            return

        if "best_selection_metric" in state:
            self.best_selection_metric = float(state["best_selection_metric"])
        if "best_selection_epoch" in state and state["best_selection_epoch"] is not None:
            self.best_selection_epoch = int(state["best_selection_epoch"])
        if "selection_metric_name" in state:
            self.best_selection_metric_name = str(state["selection_metric_name"])
        if "selection_history" in state:
            self.selection_history = [float(value) for value in state["selection_history"]]

    def _reset_selection_state_on_resume(self, message: str) -> None:
        self.best_selection_metric = float("inf")
        self.best_selection_epoch = None
        self.selection_history = []
        self.selection_metric_name = self._raw_selection_metric_name()
        self.best_selection_metric_name = self.selection_metric_name
        if self.local_rank == 0 and hasattr(self.logger, "warning"):
            self.logger.warning(message)

    def _loss_config(self) -> PipelineLossConfig:
        return getattr(self, "loss_config", PipelineLossConfig())

    def _is_raster_target(self) -> bool:
        return any(target.name == self.target_schema.name for target in self.batch_schema.raster_targets)

    def _compute_loss(
        self,
        batch: ProcessedModelBatch,
        raster_predictions: dict[str, torch.Tensor],
        scalar_predictions: dict[str, torch.Tensor],
    ) -> LossReport:
        target = self.target_schema
        if self._is_raster_target():
            if target.name not in raster_predictions:
                raise KeyError(f"Missing raster prediction for target {target.name!r}")
            prediction = raster_predictions[target.name]
            target_block = batch.raster_targets[target.name]
            city_mask = batch.city_mask
        else:
            if target.name not in scalar_predictions:
                raise KeyError(f"Missing scalar prediction for target {target.name!r}")
            prediction = scalar_predictions[target.name]
            target_block = batch.scalar_targets[target.name]
            city_mask = None

        zero = prediction.sum() * 0.0
        raw_loss, target_stats, support_weight, target_debug_terms = compute_target_loss(
            target=target,
            prediction=prediction,
            target_data=target_block.data,
            valid_mask=target_block.valid_mask,
            city_mask=city_mask,
            masked_patch=None,
            loss_config=self._loss_config(),
        )
        weight = float(target.spec.weight)
        effective_weight = raw_loss.new_tensor(weight) * support_weight
        mask_stats: dict[str, torch.Tensor] = {
            "mask/city_mask_coverage": coverage(batch.city_mask > 0, like=zero),
        }
        mask_stats.update(target_stats)

        return aggregate_loss_report(
            loss_config=self._loss_config(),
            target_raw={target.name: raw_loss},
            target_weighted={target.name: raw_loss * effective_weight},
            target_groups={target.name: target.spec.group},
            target_weights={target.name: effective_weight},
            target_sources={target.name: target.spec.source or target.name},
            target_debug_terms={target.name: target_debug_terms},
            mask_stats=mask_stats,
            zero=zero,
        )

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
                raster_predictions, scalar_predictions = self.model(batch.raster_inputs, batch.scalar_inputs)
                report = self._compute_loss(batch, raster_predictions, scalar_predictions)
                loss = report.total
                grad_norm: torch.Tensor | None = None

                if train:
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
                if train and grad_norm is not None:
                    batch_metrics["state/grad_norm"] = grad_norm
                accumulate_metrics(metric_sums, metric_counts, batch_metrics)

        epoch_metrics = average_metrics(metric_sums, metric_counts, device=self.device)
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
        total = float(metrics.get("loss/total", 0.0))
        parts = [f"{phase} target={self.target_schema.name} total={total:.6f}"]
        if "selection/val_loss" in metrics:
            parts.append(f"selection_val_loss={float(metrics['selection/val_loss']):.6f}")
        if "selection/rmse" in metrics:
            parts.append(f"selection_rmse={float(metrics['selection/rmse']):.6f}")
        if "selection/r2" in metrics:
            selection_r2 = float(metrics["selection/r2"])
            parts.append(
                "selection_r2=null" if math.isnan(selection_r2) else f"selection_r2={selection_r2:.6f}"
            )
        if "selection/count" in metrics:
            parts.append(f"selection_count={int(round(float(metrics['selection/count'])))}")
        if "selection/val_loss_smoothed" in metrics:
            smoothed = float(metrics["selection/val_loss_smoothed"])
            parts.append(
                "selection_val_loss_smoothed=null"
                if math.isnan(smoothed)
                else f"selection_val_loss_smoothed={smoothed:.6f}"
            )
        if "selection/rmse_smoothed" in metrics:
            smoothed = float(metrics["selection/rmse_smoothed"])
            parts.append(
                "selection_rmse_smoothed=null"
                if math.isnan(smoothed)
                else f"selection_rmse_smoothed={smoothed:.6f}"
            )
        return " ".join(parts)

    def format_training_summary(self, duration_minutes: float) -> str:
        if self.best_selection_epoch is None or not math.isfinite(self.best_selection_metric):
            return f"No downstream best model was selected, Total Training Time: {duration_minutes:.2f} mins"
        summary = (
            f"Best {self.best_selection_metric_name}: {self.best_selection_metric:.6f} "
            f"at epoch {self.best_selection_epoch}"
        )
        return f"{summary}, Total Training Time: {duration_minutes:.2f} mins"

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


__all__ = ["EumpDownstreamTrainer"]
