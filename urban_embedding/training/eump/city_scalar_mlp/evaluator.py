from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist

from urban_embedding.data.processed.pipeline_spec import (
    PipelineSpec,
    validate_eump_scalar_mlp_downstream_spec,
)
from urban_embedding.models.eump.city_scalar_mlp.model import build_eump_scalar_mlp_model
from urban_embedding.training.checkpointing import load_model_state_for_evaluation
from urban_embedding.training.distributed import get_rank
from urban_embedding.training.regression_metrics import (
    accumulate_regression_stats,
    empty_regression_stats,
    inverse_processed_tensor,
    summarize_regression_stats,
)
from .cache import normalize_city_scalar_feature_standardization, prepare_city_scalar_mlp_data


def _tensor_max_abs_delta(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() <= 0 and right.numel() <= 0:
        return 0.0
    if tuple(left.shape) != tuple(right.shape):
        return float("inf")
    delta = torch.abs(left.detach().cpu().to(torch.float64) - right.detach().cpu().to(torch.float64))
    return float(delta.max().item()) if delta.numel() else 0.0


def validate_scalar_mlp_checkpoint_standardization(
    *,
    model,
    standardization,
    target_name: str,
    checkpoint_path: str,
    rtol: float = 1.0e-4,
    atol: float = 1.0e-4,
) -> None:
    """Reject checkpoints whose saved standardization buffers do not match the active cache."""

    mismatches: list[str] = []
    for name in ("feature_mean", "feature_std", "target_mean", "target_std"):
        actual = getattr(model, name)
        expected = getattr(standardization, name)
        actual_cpu = actual.detach().cpu().to(torch.float64)
        expected_cpu = expected.detach().cpu().to(torch.float64)
        if tuple(actual_cpu.shape) != tuple(expected_cpu.shape):
            mismatches.append(
                f"{name}: checkpoint_shape={tuple(actual_cpu.shape)} cache_shape={tuple(expected_cpu.shape)}"
            )
            continue
        if not torch.allclose(actual_cpu, expected_cpu, rtol=float(rtol), atol=float(atol)):
            mismatches.append(f"{name}: max_abs_diff={_tensor_max_abs_delta(actual_cpu, expected_cpu):.6g}")

    if mismatches:
        details = "; ".join(mismatches)
        raise RuntimeError(
            "Scalar MLP checkpoint standardization does not match the current city-scalar cache "
            f"for target {target_name!r} at {checkpoint_path}: {details}. "
            "The checkpoint was likely trained with stale cache/preprocessing. "
            "Retrain the scalar MLP baseline with the current processed-scale cache and evaluate the new checkpoint."
        )


class EumpScalarMlpEvaluator:
    def __init__(
        self,
        *,
        logger,
        device: torch.device,
        model_name: str,
        data_config: dict[str, Any],
        pipeline_config: dict[str, Any],
        model_config: dict[str, Any] | None,
        downstream_config: dict[str, Any],
        evaluation_config: dict[str, Any],
        local_rank: int,
        world_size: int,
        random_seed: int,
    ) -> None:
        self.logger = logger
        self.device = device
        self.model_name = model_name
        self.model_config = {} if model_config is None else dict(model_config)
        self.downstream_config = dict(downstream_config)
        self.evaluation_config = dict(evaluation_config)
        self.feature_standardization = normalize_city_scalar_feature_standardization(
            self.downstream_config.get("feature_standardization")
        )
        self.local_rank = int(local_rank)
        self.world_size = int(world_size)
        self.random_seed = int(random_seed)
        self.split = str(self.evaluation_config["split"]).lower()
        self.metric_scale = str(self.evaluation_config["metric_scale"]).lower()
        self.checkpoint_source = str(self.evaluation_config["checkpoint_source"]).lower()
        if self.metric_scale != "physical":
            raise ValueError("Scalar MLP baseline evaluation supports only evaluation.metric_scale='physical'")
        batch_size = int(self.downstream_config["batch_size"])
        if batch_size % max(1, self.world_size) != 0:
            raise ValueError(
                f"downstream.batch_size={batch_size} must be divisible by world_size={self.world_size}"
            )

        self.pipeline_spec = PipelineSpec.from_mapping(pipeline_config)
        validate_eump_scalar_mlp_downstream_spec(self.pipeline_spec)
        prepared = prepare_city_scalar_mlp_data(
            data_config=data_config,
            pipeline_config=dict(pipeline_config),
            logger=self.logger,
            random_seed=self.random_seed,
            batch_size=batch_size // max(1, self.world_size),
            local_rank=self.local_rank,
            world_size=self.world_size,
            device=self.device,
            log_dir=None,
            feature_standardization=self.feature_standardization,
        )
        self.train_dl = prepared.train_dl
        self.val_dl = prepared.val_dl
        self.test_dl = prepared.test_dl
        self.batch_schema = prepared.batch_schema
        self.target_schema = prepared.target_schema
        self.feature_layout = prepared.feature_layout
        self.cache_metadata = prepared.cache_metadata
        self.preprocessing_spec = prepared.preprocessing_spec
        self.source_spec = self._target_source_spec()
        self.standardization = prepared.standardization
        self.model = build_eump_scalar_mlp_model(
            model_config=self.model_config,
            input_dim=self.feature_layout.feature_dim,
            output_dim=self.target_schema.channels,
        ).to(self.device)
        self.checkpoint_file = self._resolve_checkpoint_file()
        self.checkpoint_path = load_model_state_for_evaluation(
            model=self.model,
            model_name=self.model_name,
            checkpoint_source=self.checkpoint_source,
            checkpoint_file=self.checkpoint_file,
            map_location=self.device,
        )
        validate_scalar_mlp_checkpoint_standardization(
            model=self.model,
            standardization=self.standardization,
            target_name=self.target_schema.name,
            checkpoint_path=self.checkpoint_path,
        )

    def _target_source_spec(self) -> dict[str, Any]:
        source_name = self.target_schema.spec.source
        if source_name is None:
            raise ValueError(f"Scalar MLP target {self.target_schema.name!r} requires a source")
        source_spec = self.preprocessing_spec["sources"].get(source_name)
        if source_spec is None:
            raise KeyError(f"Source {source_name!r} is missing from preprocessing spec")
        return source_spec

    def _resolve_checkpoint_file(self) -> str:
        configured = self.evaluation_config.get("checkpoint_file")
        if configured:
            return str(configured)
        if self.checkpoint_source == "best_model":
            return str(self.downstream_config["save_file"])
        return str(self.downstream_config["checkpoint_file"])

    def _selected_loader(self):
        if self.split == "train":
            return self.train_dl
        if self.split == "val":
            return self.val_dl
        return self.test_dl

    def _reduce_stats(self, stats: torch.Tensor) -> torch.Tensor:
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        return stats

    def _physical_metric_tensor(self, values: torch.Tensor) -> torch.Tensor:
        return inverse_processed_tensor(values, self.source_spec)

    def evaluate(self) -> dict[str, Any]:
        dataloader = self._selected_loader()
        stats = empty_regression_stats(device=self.device)
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                prediction = self._physical_metric_tensor(self.model(batch.features))
                target = self._physical_metric_tensor(batch.target)
                accumulate_regression_stats(
                    stats,
                    prediction,
                    target,
                )

        metrics, zero_variance = summarize_regression_stats(self._reduce_stats(stats))
        if metrics["count"] <= 0:
            raise ValueError(
                f"Scalar MLP evaluation found no valid samples for target {self.target_schema.name!r} "
                f"on split {self.split!r}"
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
            "target_kind": "city_scalar",
            "training_target_scale": "processed",
            "metric_scale": "physical",
            "center_ratio": float(self.pipeline_spec.center_ratio),
            "feature_names": list(self.feature_layout.feature_names),
            "cache": self.cache_metadata,
            "metrics": {
                "global": metrics,
                "city_mean": metrics,
            },
        }
        if get_rank() == 0:
            if zero_variance:
                self.logger.warning(
                    f"Scalar MLP evaluation target={self.target_schema.name} split={self.split}: "
                    "global R2 is undefined because target variance is zero."
                )
            self.logger.info(
                "EVAL scalar_mlp split={split} target={target} scope=city_mean rmse={rmse} r2={r2} count={count}".format(
                    split=self.split,
                    target=self.target_schema.name,
                    rmse=("null" if metrics["rmse"] is None else f"{metrics['rmse']:.6f}"),
                    r2=("null" if metrics["r2"] is None else f"{metrics['r2']:.6f}"),
                    count=metrics["count"],
                )
            )
        return summary


__all__ = ["EumpScalarMlpEvaluator", "validate_scalar_mlp_checkpoint_standardization"]
