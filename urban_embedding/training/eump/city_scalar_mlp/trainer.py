from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy

import torch
import torch.distributed as dist

from urban_embedding.data.processed.pipeline_spec import (
    PipelineSpec,
    ProcessedBatchSchema,
    TargetSchema,
    validate_eump_scalar_mlp_downstream_spec,
)
from urban_embedding.models.eump.city_scalar_mlp.model import build_eump_scalar_mlp_model
from urban_embedding.training.distributed import wrap_ddp
from urban_embedding.training.eump_loss import accumulate_metrics, average_metrics, tensor_to_float
from urban_embedding.training.losses import compute_weighted_loss
from urban_embedding.training.regression_metrics import (
    accumulate_regression_stats,
    empty_regression_stats,
    inverse_processed_tensor,
    summarize_regression_stats,
)
from .cache import (
    CachedCityScalarBatch,
    normalize_city_scalar_feature_standardization,
    prepare_city_scalar_mlp_data,
)
from .features import (
    CityLevelFeatureLayout,
)
from urban_embedding.training.schedulers import (
    build_warmup_cosine_scheduler,
    normalize_warmup_cosine_scheduler_config,
    warmup_cosine_metadata,
)
from urban_embedding.training.tensorboard import TensorBoardScalarWriter


class EumpScalarMlpTrainer:
    def __init__(
        self,
        *,
        logger,
        device: torch.device,
        model_name: str,
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
        save_file_name: str = "eump_scalar_mlp_model.pt",
        checkpoint_file_name: str = "eump_scalar_mlp_checkpoint.pt",
        resume: bool = False,
        resume_checkpoint: str | None = None,
        log_dir: str | None = None,
        scheduler_config: dict | None = None,
        feature_standardization: str = "train_split_zscore",
    ) -> None:
        self.logger = logger
        self.device = device
        self.model_name = model_name
        self.pipeline_config = deepcopy(pipeline_config)
        self.model_config = {} if model_config is None else dict(model_config)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.n_epochs = int(n_epochs)
        self.random_seed = int(random_seed)
        self.local_rank = int(local_rank)
        self.world_size = int(world_size)
        self.accumulation_steps = int(accumulation_steps)
        self.save_file_name = save_file_name
        self.checkpoint_file_name = checkpoint_file_name
        self.resume = bool(resume)
        self.resume_checkpoint = resume_checkpoint
        self.feature_standardization = normalize_city_scalar_feature_standardization(feature_standardization)
        self.scheduler_config = normalize_warmup_cosine_scheduler_config(scheduler_config)
        self.save_best_model_enabled = True
        self.save_final_model_enabled = False
        self.start_epoch = 1
        self.best_val_loss = float("inf")
        self.global_step = 0
        if self.accumulation_steps < 1:
            raise ValueError(f"accumulation_steps must be >= 1, got {self.accumulation_steps}")
        if self.n_epochs < 1:
            raise ValueError(f"n_epochs must be >= 1, got {self.n_epochs}")

        self.pipeline_spec = PipelineSpec.from_mapping(pipeline_config)
        validate_eump_scalar_mlp_downstream_spec(self.pipeline_spec)
        self.loss_config = self.pipeline_spec.loss_config
        self.tensorboard = TensorBoardScalarWriter(
            log_dir=log_dir,
            enabled=self.loss_config.tensorboard,
            rank=self.local_rank,
        )
        prepared = prepare_city_scalar_mlp_data(
            data_config=data_config,
            pipeline_config=self.pipeline_config,
            logger=self.logger,
            random_seed=self.random_seed,
            batch_size=self.batch_size,
            local_rank=self.local_rank,
            world_size=self.world_size,
            device=self.device,
            log_dir=log_dir,
            feature_standardization=self.feature_standardization,
        )
        self.train_dl = prepared.train_dl
        self.val_dl = prepared.val_dl
        self.test_dl = prepared.test_dl
        self.batch_schema = prepared.batch_schema
        self.target_schema = prepared.target_schema
        self.feature_layout = prepared.feature_layout
        self.preprocessing_spec = prepared.preprocessing_spec
        self.cache_metadata = prepared.cache_metadata
        self.source_spec = self._target_source_spec()
        standardization = prepared.standardization
        self.standardization_sample_count = standardization.count
        self.model = self.initialize_model(self.feature_layout, self.target_schema, self.batch_schema)
        self.model.set_standardization(
            feature_mean=standardization.feature_mean.to(self.device),
            feature_std=standardization.feature_std.to(self.device),
            target_mean=standardization.target_mean.to(self.device),
            target_std=standardization.target_std.to(self.device),
        )
        self.model = wrap_ddp(self.model, self.device, self.local_rank)
        self.optimizer = torch.optim.AdamW(
            [param for param in self.model.parameters() if param.requires_grad],
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self.scheduler = self.build_scheduler(self.n_epochs)
        if self.local_rank == 0:
            self.logger.info(
                "Initialize EUMP scalar MLP baseline "
                f"features={self.feature_layout.feature_dim} "
                f"target={self.target_schema.name} "
                "target_scale=processed "
                f"feature_standardization={self.feature_standardization} "
                f"train_stats_count={self.standardization_sample_count}"
            )

    def _target_source_spec(self) -> dict:
        source_name = self.target_schema.spec.source
        if source_name is None:
            raise ValueError(f"Scalar MLP target {self.target_schema.name!r} requires a source")
        source_spec = self.preprocessing_spec["sources"].get(source_name)
        if source_spec is None:
            raise KeyError(f"Source {source_name!r} is missing from preprocessing spec")
        return source_spec

    def initialize_model(
        self,
        feature_layout: CityLevelFeatureLayout,
        target_schema: TargetSchema,
        batch_schema: ProcessedBatchSchema,
    ):
        del batch_schema
        return build_eump_scalar_mlp_model(
            model_config=self.model_config,
            input_dim=feature_layout.feature_dim,
            output_dim=target_schema.channels,
        ).to(self.device)

    def build_scheduler(self, n_epochs=None):
        schedule_epochs = self.n_epochs if n_epochs is None else int(n_epochs)
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
            epochs=int(n_epochs),
            warmup_fraction=scheduler_config.warmup_fraction,
            lr_start_factor=scheduler_config.lr_start_factor,
            lr_end_factor=scheduler_config.lr_end_factor,
        )

    def reset_scheduler(self, n_epochs):
        self.scheduler = self.build_scheduler(n_epochs)

    def _current_learning_rate(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _physical_metric_tensor(self, values: torch.Tensor) -> torch.Tensor:
        return inverse_processed_tensor(values, self.source_spec)

    def _batch_tensors(self, batch: CachedCityScalarBatch):
        prediction_processed = self.model(batch.features)
        target_processed = batch.target
        prediction_physical = self._physical_metric_tensor(prediction_processed)
        target_physical = self._physical_metric_tensor(target_processed)
        sample_mask = torch.ones(batch.features.shape[0], dtype=torch.bool, device=batch.features.device)
        return prediction_processed, prediction_physical, target_processed, target_physical, sample_mask

    def _batch_loss(
        self,
        prediction_processed: torch.Tensor,
        target_processed: torch.Tensor,
        sample_mask: torch.Tensor,
    ) -> torch.Tensor:
        if not bool(sample_mask.any().detach().cpu()):
            return prediction_processed.sum() * 0.0
        loss = compute_weighted_loss(
            self.target_schema.spec.loss,
            prediction_processed[sample_mask],
            target_processed[sample_mask],
            mask=None,
        )
        return loss * float(self.target_schema.spec.weight)

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

    def _reduce_stats(self, stats: torch.Tensor) -> torch.Tensor:
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        return stats

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
        stats = empty_regression_stats(device=self.device)

        if train:
            self.optimizer.zero_grad()

        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for batch_idx, batch in enumerate(dataloader):
                batch = batch.to(self.device)
                (
                    prediction_processed,
                    prediction_physical,
                    target_processed,
                    target_physical,
                    sample_mask,
                ) = self._batch_tensors(batch)
                loss = self._batch_loss(prediction_processed, target_processed, sample_mask)
                if bool(sample_mask.any().detach().cpu()):
                    accumulate_metrics(metric_sums, metric_counts, {"loss/total": loss})
                    accumulate_regression_stats(
                        stats,
                        prediction_physical[sample_mask],
                        target_physical[sample_mask],
                    )

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
                if train and grad_norm is not None:
                    accumulate_metrics(metric_sums, metric_counts, {"state/grad_norm": grad_norm})

        if "loss/total" not in metric_sums:
            metric_sums["loss/total"] = torch.tensor(0.0, device=self.device)
            metric_counts["loss/total"] = 1
        epoch_metrics = average_metrics(metric_sums, metric_counts, device=self.device)
        summary, _ = summarize_regression_stats(self._reduce_stats(stats))
        epoch_metrics["metric/rmse"] = torch.tensor(
            float("nan") if summary["rmse"] is None else float(summary["rmse"]),
            device=self.device,
        )
        epoch_metrics["metric/r2"] = torch.tensor(
            float("nan") if summary["r2"] is None else float(summary["r2"]),
            device=self.device,
        )
        epoch_metrics["metric/count"] = torch.tensor(float(summary["count"]), device=self.device)

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
        rmse = float(metrics.get("metric/rmse", float("nan")))
        r2 = float(metrics.get("metric/r2", float("nan")))
        count = int(round(float(metrics.get("metric/count", 0.0))))
        return (
            f"{phase} target={self.target_schema.name} scope=city_mean total={total:.6f} "
            f"rmse={'null' if math.isnan(rmse) else f'{rmse:.6f}'} "
            f"r2={'null' if math.isnan(r2) else f'{r2:.6f}'} "
            f"count={count}"
        )

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


__all__ = ["EumpScalarMlpTrainer"]
