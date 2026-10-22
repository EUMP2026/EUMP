from __future__ import annotations

import os
import math
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import torch
import torch.distributed as dist

from urban_embedding.core.paths import MODEL_DIR, resolve_model_artifact_path
from urban_embedding.data.processed.dataset import build_eump_dataloaders
from urban_embedding.data.processed.pipeline_spec import (
    PipelineSpec,
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
from urban_embedding.training.checkpointing import load_model_state_for_evaluation
from urban_embedding.training.distributed import get_rank
from urban_embedding.training.regression_metrics import (
    accumulate_regression_stats,
    empty_regression_stats,
    inverse_processed_tensor,
    raster_city_mean_metric_tensors,
    raster_global_metric_tensors,
    scalar_metric_tensors,
    summarize_regression_stats,
)


class EumpDownstreamEvaluator:
    def __init__(
        self,
        *,
        logger,
        device: torch.device,
        model_name: str,
        pretrain_model: str,
        data_config: dict[str, Any],
        pipeline_config: dict[str, Any],
        model_config: dict[str, Any] | None,
        downstream_config: dict[str, Any],
        evaluation_config: dict[str, Any],
        local_rank: int,
        world_size: int,
        random_seed: int,
        scratch_backbone: bool,
    ) -> None:
        self.logger = logger
        self.device = device
        self.model_name = model_name
        self.pretrain_model = pretrain_model
        self.data_config = data_config
        self.model_config = {} if model_config is None else dict(model_config)
        self.local_rank = int(local_rank)
        self.world_size = int(world_size)
        self.random_seed = int(random_seed)
        self.scratch_backbone = bool(scratch_backbone)
        self.evaluation_config = dict(evaluation_config)
        self.downstream_config = dict(downstream_config)
        self.pipeline_config = deepcopy(pipeline_config)
        self.split = str(self.evaluation_config["split"]).lower()
        self.metric_scale = str(self.evaluation_config["metric_scale"]).lower()
        self.checkpoint_source = str(self.evaluation_config["checkpoint_source"]).lower()
        self.extra_center_city_mean = bool(self.evaluation_config.get("extra_center_city_mean", False))
        self.extended_metrics = bool(self.evaluation_config.get("extended_metrics", False))
        self.extended_spatial_metrics = bool(
            self.evaluation_config.get("extended_spatial_metrics", False)
        )

        pipeline_payload = deepcopy(self.pipeline_config)
        pipeline_payload["center_ratio"] = float(self.evaluation_config["center_ratio"])
        self.pipeline_spec = PipelineSpec.from_mapping(pipeline_payload)
        validate_eump_downstream_spec(self.pipeline_spec)

        if self.downstream_config["batch_size"] % max(1, self.world_size) != 0:
            raise ValueError(
                f"downstream.batch_size={self.downstream_config['batch_size']} must be divisible "
                f"by world_size={self.world_size}"
            )

        self.train_dl, self.val_dl, self.test_dl, self.batch_schema = build_eump_dataloaders(
            data_config=data_config,
            pipeline_spec=self.pipeline_spec,
            logger=self.logger,
            random_seed=self.random_seed,
            batch_size=self.downstream_config["batch_size"] // max(1, self.world_size),
            local_rank=self.local_rank,
            world_size=self.world_size,
        )
        self.target_schema = get_single_downstream_target_schema(self.batch_schema)
        self.target_kind = (
            "raster"
            if any(target.name == self.target_schema.name for target in self.batch_schema.raster_targets)
            else "scalar"
        )
        self._validate_target_for_evaluation()
        self.preprocessing_spec = self._active_preprocessing_spec(data_config)
        self.source_spec = self._target_source_spec()
        self.model = self._build_model()
        self.checkpoint_file = self._resolve_checkpoint_file()
        self.checkpoint_path = load_model_state_for_evaluation(
            model=self.model,
            model_name=self.model_name,
            checkpoint_source=self.checkpoint_source,
            checkpoint_file=self.checkpoint_file,
            map_location=self.device,
        )
        self.model.eval()

    def _build_model(self):
        arch = str(self.model_config.get("downstream_arch", "light_dual_head"))
        if self.batch_schema.scalar_input_dim > 0 and not self.pipeline_spec.broadcast_scalar and arch == "light_dual_head":
            raise ValueError(
                "eump_downstream evaluation has scalar input sources "
                f"{self.batch_schema.scalar_input_names}, but model.downstream_arch={arch!r} "
                "does not consume scalar_inputs directly. Set pipeline.broadcast_scalar=true "
                "to route scalar covariates through the encoder."
            )

        encoder_checkpoint_path = None
        encoder_source = str(self.downstream_config["encoder_source"])
        if encoder_source in {
            "reconstruction_checkpoint",
            "convnext_reconstruction_checkpoint",
            "convnext_pretrained",
        }:
            encoder_checkpoint_path = resolve_model_artifact_path(
                str(self.downstream_config["encoder_model_name"]),
                str(self.downstream_config["encoder_checkpoint_file"]),
            )

        model = build_eump_downstream_model(
            model_config=self.model_config,
            batch_schema=self.batch_schema,
            target_schema=self.target_schema,
            base_swin_path=os.path.join(MODEL_DIR, self.pretrain_model),
            encoder_source=str(self.downstream_config["encoder_source"]),
            encoder_checkpoint_path=encoder_checkpoint_path,
            load_pretrained=not self.scratch_backbone,
            freeze_encoder=bool(self.downstream_config["freeze_encoder"]),
            map_location=self.device,
        )
        return model.to(self.device)

    def _target_source_spec(self) -> dict[str, Any]:
        source_name = self.target_schema.spec.source
        if source_name is None:
            raise ValueError(f"Downstream evaluation target {self.target_schema.name!r} requires a source")
        source_spec = self.preprocessing_spec["sources"].get(source_name)
        if source_spec is None:
            raise KeyError(f"Source {source_name!r} is missing from preprocessing spec")
        return source_spec

    def _active_preprocessing_spec(self, data_config: dict[str, Any]) -> dict[str, Any]:
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

    def _resolve_checkpoint_file(self) -> str:
        configured = self.evaluation_config.get("checkpoint_file")
        if configured:
            return str(configured)
        if self.checkpoint_source == "best_model":
            return str(self.downstream_config["save_file"])
        return str(self.downstream_config["checkpoint_file"])

    def _validate_target_for_evaluation(self) -> None:
        loss = str(self.target_schema.spec.loss).lower()
        if loss not in {"mse", "mae", "huber"}:
            raise ValueError(
                "EUMP downstream evaluation currently supports only regression targets with "
                f"loss 'mse', 'mae', or 'huber', got {loss!r}"
            )

    def _metric_tensor(self, values: torch.Tensor) -> torch.Tensor:
        if self.metric_scale == "physical":
            return inverse_processed_tensor(values, self.source_spec)
        return values

    def _selected_loader(self):
        if self.split == "train":
            return self.train_dl
        if self.split == "val":
            return self.val_dl
        return self.test_dl

    def _selected_from(self, train_dl, val_dl, test_dl):
        if self.split == "train":
            return train_dl
        if self.split == "val":
            return val_dl
        return test_dl

    def _build_center_crop_loader(self):
        pipeline_payload = deepcopy(self.pipeline_config)
        pipeline_payload["center_ratio"] = 1.0
        pipeline_spec = PipelineSpec.from_mapping(pipeline_payload)
        validate_eump_downstream_spec(pipeline_spec)
        train_dl, val_dl, test_dl, batch_schema = build_eump_dataloaders(
            data_config=self.data_config,
            pipeline_spec=pipeline_spec,
            logger=self.logger,
            random_seed=self.random_seed,
            batch_size=self.downstream_config["batch_size"] // max(1, self.world_size),
            local_rank=self.local_rank,
            world_size=self.world_size,
        )
        if batch_schema.raster_input_channel_names != self.batch_schema.raster_input_channel_names:
            raise ValueError("Center-crop evaluation loader changed raster input schema unexpectedly")
        if tuple(target.name for target in batch_schema.raster_targets) != tuple(
            target.name for target in self.batch_schema.raster_targets
        ):
            raise ValueError("Center-crop evaluation loader changed raster target schema unexpectedly")
        return self._selected_from(train_dl, val_dl, test_dl)

    def _reduce_stats(self, stats: torch.Tensor) -> torch.Tensor:
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        return stats

    @staticmethod
    def _accumulate_correlation_stats(
        stats: torch.Tensor,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        prediction = prediction.reshape(-1).to(dtype=torch.float64)
        target = target.reshape(-1).to(dtype=torch.float64)
        if prediction.numel() == 0:
            return
        stats[0] += prediction.numel()
        stats[1] += prediction.sum()
        stats[2] += target.sum()
        stats[3] += (prediction * prediction).sum()
        stats[4] += (target * target).sum()
        stats[5] += (prediction * target).sum()

    @staticmethod
    def _accumulate_mae_stats(
        stats: torch.Tensor,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        prediction = prediction.reshape(-1).to(dtype=torch.float64)
        target = target.reshape(-1).to(dtype=torch.float64)
        stats[0] += target.numel()
        stats[1] += (prediction - target).abs().sum()

    @staticmethod
    def _summarize_correlation_stats(stats: torch.Tensor) -> float | None:
        count = float(stats[0].detach().cpu().item())
        if count < 2:
            return None
        sum_prediction, sum_target = (
            float(stats[index].detach().cpu().item()) for index in (1, 2)
        )
        prediction_ss = float(stats[3].detach().cpu().item()) - sum_prediction ** 2 / count
        target_ss = float(stats[4].detach().cpu().item()) - sum_target ** 2 / count
        cross = float(stats[5].detach().cpu().item()) - sum_prediction * sum_target / count
        denominator = math.sqrt(max(prediction_ss, 0.0) * max(target_ss, 0.0))
        return None if denominator <= 1.0e-12 else float(cross / denominator)

    def _scalar_metrics(
        self,
        batch,
        scalar_predictions: Mapping[str, torch.Tensor],
        stats: torch.Tensor,
        mae_stats: torch.Tensor | None,
    ) -> None:
        prediction = scalar_predictions[self.target_schema.name]
        target_block = batch.scalar_targets[self.target_schema.name]
        metric_prediction = self._metric_tensor(prediction)
        metric_target = self._metric_tensor(target_block.data)
        pred_values, target_values = scalar_metric_tensors(
            prediction=metric_prediction,
            target=metric_target,
            valid_mask=target_block.valid_mask,
            mask_strategy=self.target_schema.spec.mask,
        )
        accumulate_regression_stats(stats, pred_values, target_values)
        if mae_stats is not None:
            self._accumulate_mae_stats(mae_stats, pred_values, target_values)

    def _raster_metrics(
        self,
        batch,
        raster_predictions: Mapping[str, torch.Tensor],
        global_stats: torch.Tensor,
        city_mean_stats: torch.Tensor,
        mae_stats: torch.Tensor | None,
        global_correlation_stats: torch.Tensor | None,
        sample_correlation_stats: torch.Tensor | None,
    ) -> None:
        prediction = raster_predictions[self.target_schema.name]
        target_block = batch.raster_targets[self.target_schema.name]
        metric_prediction = self._metric_tensor(prediction)
        metric_target = self._metric_tensor(target_block.data)

        pred_values, target_values = raster_global_metric_tensors(
            prediction=metric_prediction,
            target=metric_target,
            valid_mask=target_block.valid_mask,
            city_mask=batch.city_mask,
            mask_strategy=self.target_schema.spec.mask,
        )
        accumulate_regression_stats(global_stats, pred_values, target_values)
        if mae_stats is not None:
            self._accumulate_mae_stats(mae_stats, pred_values, target_values)
        if global_correlation_stats is not None:
            self._accumulate_correlation_stats(global_correlation_stats, pred_values, target_values)

        for sample_index in range(int(prediction.shape[0])) if sample_correlation_stats is not None else ():
            sample_prediction, sample_target = raster_global_metric_tensors(
                prediction=metric_prediction[sample_index:sample_index + 1],
                target=metric_target[sample_index:sample_index + 1],
                valid_mask=target_block.valid_mask[sample_index:sample_index + 1],
                city_mask=batch.city_mask[sample_index:sample_index + 1],
                mask_strategy=self.target_schema.spec.mask,
            )
            sample_stats = torch.zeros(6, dtype=torch.float64, device=self.device)
            self._accumulate_correlation_stats(sample_stats, sample_prediction, sample_target)
            sample_correlation = self._summarize_correlation_stats(sample_stats)
            if sample_correlation is not None:
                sample_correlation_stats[0] += 1.0
                sample_correlation_stats[1] += sample_correlation

        city_pred_values, city_target_values = raster_city_mean_metric_tensors(
            prediction=metric_prediction,
            target=metric_target,
            valid_mask=target_block.valid_mask,
            city_mask=batch.city_mask,
            mask_strategy=self.target_schema.spec.mask,
        )
        accumulate_regression_stats(city_mean_stats, city_pred_values, city_target_values)

    def _raster_center_city_mean_metrics(self, dataloader) -> dict[str, Any]:
        stats = empty_regression_stats(device=self.device)
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                raster_predictions, _ = self.model(batch.raster_inputs, batch.scalar_inputs)
                prediction = raster_predictions[self.target_schema.name]
                target_block = batch.raster_targets[self.target_schema.name]
                metric_prediction = self._metric_tensor(prediction)
                metric_target = self._metric_tensor(target_block.data)
                pred_values, target_values = raster_city_mean_metric_tensors(
                    prediction=metric_prediction,
                    target=metric_target,
                    valid_mask=target_block.valid_mask,
                    city_mask=batch.city_mask,
                    mask_strategy=self.target_schema.spec.mask,
                )
                accumulate_regression_stats(stats, pred_values, target_values)
        metrics, zero_variance = summarize_regression_stats(self._reduce_stats(stats))
        if metrics["count"] <= 0:
            raise ValueError(
                f"EUMP center-crop city_mean evaluation found no valid samples for target "
                f"{self.target_schema.name!r} on split {self.split!r}"
            )
        if get_rank() == 0 and zero_variance:
            self.logger.warning(
                f"Evaluation target={self.target_schema.name} split={self.split}: "
                "city_mean_center_crop R2 is undefined because target variance is zero."
            )
        return metrics

    def evaluate(self) -> dict[str, Any]:
        dataloader = self._selected_loader()
        stats = {
            "global": empty_regression_stats(device=self.device),
        }
        if self.extended_metrics:
            stats["mae"] = torch.zeros(2, dtype=torch.float64, device=self.device)
        if self.target_kind == "raster":
            stats["city_mean"] = empty_regression_stats(device=self.device)
            if self.extended_spatial_metrics:
                stats["global_correlation"] = torch.zeros(6, dtype=torch.float64, device=self.device)
                stats["sample_correlation"] = torch.zeros(2, dtype=torch.float64, device=self.device)

        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                raster_predictions, scalar_predictions = self.model(batch.raster_inputs, batch.scalar_inputs)
                if self.target_kind == "scalar":
                    self._scalar_metrics(
                        batch,
                        scalar_predictions,
                        stats["global"],
                        stats.get("mae"),
                    )
                else:
                    self._raster_metrics(
                        batch,
                        raster_predictions,
                        stats["global"],
                        stats["city_mean"],
                        stats.get("mae"),
                        stats.get("global_correlation"),
                        stats.get("sample_correlation"),
                    )

        summary: dict[str, Any] = {
            "experiment": {
                "group": self.model_name.split("/", maxsplit=1)[0],
                "name": self.model_name.split("/", maxsplit=1)[1],
                "model_name": self.model_name,
            },
            "checkpoint": {
                "source": self.checkpoint_source,
                "file": self.checkpoint_file,
                "path": self.checkpoint_path,
            },
            "split": self.split,
            "target": self.target_schema.name,
            "target_kind": self.target_kind,
            "metric_scale": self.metric_scale,
            "center_ratio": float(self.pipeline_spec.center_ratio),
            "metrics": {},
        }

        reduced_global = self._reduce_stats(stats["global"])
        global_metrics, global_zero_variance = summarize_regression_stats(reduced_global)
        if self.extended_metrics:
            reduced_mae = self._reduce_stats(stats["mae"])
            mae_count = int(round(float(reduced_mae[0].detach().cpu().item())))
            global_metrics["mae"] = (
                None
                if mae_count <= 0
                else float(reduced_mae[1].detach().cpu().item()) / mae_count
            )
        if global_metrics["count"] <= 0:
            raise ValueError(
                f"EUMP evaluation found no valid samples for target {self.target_schema.name!r} on split {self.split!r}"
            )
        if self.target_kind == "scalar":
            summary["metrics"]["global"] = global_metrics
            if get_rank() == 0 and global_zero_variance:
                self.logger.warning(
                    f"Evaluation target={self.target_schema.name} split={self.split}: "
                    "global R2 is undefined because target variance is zero."
                )
        else:
            summary["metrics"]["global_pixels"] = global_metrics
            if self.extended_spatial_metrics:
                reduced_correlation = self._reduce_stats(stats["global_correlation"])
                reduced_sample_correlation = self._reduce_stats(stats["sample_correlation"])
                global_metrics["spatial_correlation"] = self._summarize_correlation_stats(
                    reduced_correlation
                )
                sample_count = int(round(float(reduced_sample_correlation[0].detach().cpu().item())))
                global_metrics["spatial_correlation_mean_per_sample"] = (
                    None
                    if sample_count <= 0
                    else float(reduced_sample_correlation[1].detach().cpu().item()) / sample_count
                )
                global_metrics["spatial_correlation_sample_count"] = sample_count
            if get_rank() == 0 and global_zero_variance:
                self.logger.warning(
                    f"Evaluation target={self.target_schema.name} split={self.split}: "
                    "global_pixels R2 is undefined because target variance is zero."
                )
            reduced_city = self._reduce_stats(stats["city_mean"])
            city_metrics, city_zero_variance = summarize_regression_stats(reduced_city)
            summary["metrics"]["city_mean"] = city_metrics
            if get_rank() == 0 and city_metrics["count"] > 0 and city_zero_variance:
                self.logger.warning(
                    f"Evaluation target={self.target_schema.name} split={self.split}: "
                    "city_mean R2 is undefined because target variance is zero."
                )
            if self.extra_center_city_mean:
                center_loader = self._build_center_crop_loader()
                summary["metrics"]["city_mean_center_crop"] = self._raster_center_city_mean_metrics(center_loader)

        if get_rank() == 0:
            if self.target_kind == "scalar":
                metrics = summary["metrics"]["global"]
                self.logger.info(
                    "EVAL split={split} target={target} rmse={rmse} r2={r2} count={count}".format(
                        split=self.split,
                        target=self.target_schema.name,
                        rmse=("null" if metrics["rmse"] is None else f"{metrics['rmse']:.6f}"),
                        r2=("null" if metrics["r2"] is None else f"{metrics['r2']:.6f}"),
                        count=metrics["count"],
                    )
                )
            else:
                metric_names = ["global_pixels", "city_mean"]
                if self.extra_center_city_mean:
                    metric_names.append("city_mean_center_crop")
                for metric_name in metric_names:
                    metrics = summary["metrics"][metric_name]
                    self.logger.info(
                        "EVAL split={split} target={target} scope={scope} rmse={rmse} r2={r2} count={count}".format(
                            split=self.split,
                            target=self.target_schema.name,
                            scope=metric_name,
                            rmse=("null" if metrics["rmse"] is None else f"{metrics['rmse']:.6f}"),
                            r2=("null" if metrics["r2"] is None else f"{metrics['r2']:.6f}"),
                            count=metrics["count"],
                        )
                    )

        return summary


__all__ = ["EumpDownstreamEvaluator"]
