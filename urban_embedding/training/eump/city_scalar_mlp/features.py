from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from urban_embedding.data.processed.batch import ProcessedModelBatch
from urban_embedding.data.processed.pipeline_spec import (
    PipelineSpec,
    ProcessedBatchSchema,
    TargetSchema,
    source_output_names,
)

from urban_embedding.training.regression_metrics import inverse_processed_tensor


@dataclass(frozen=True, slots=True)
class SourceFeatureSlice:
    source_name: str
    channel_names: tuple[str, ...]
    indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class CityLevelFeatureLayout:
    raster_slices: tuple[SourceFeatureSlice, ...]
    scalar_slices: tuple[SourceFeatureSlice, ...]
    feature_names: tuple[str, ...]
    include_city_mask_mean: bool = False

    @property
    def feature_dim(self) -> int:
        return len(self.feature_names)


@dataclass(frozen=True, slots=True)
class CityLevelTensors:
    data: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True, slots=True)
class StandardizationStats:
    feature_mean: torch.Tensor
    feature_std: torch.Tensor
    target_mean: torch.Tensor
    target_std: torch.Tensor
    count: int


def _channel_means(
    *,
    data: torch.Tensor,
    valid_mask: torch.Tensor,
    city_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    city = city_mask > 0
    while city.dim() < data.dim():
        city = city.unsqueeze(1)
    city = city.expand_as(data)

    effective = valid_mask.bool() & torch.isfinite(data) & city
    weights = effective.to(dtype=data.dtype)
    support = weights.sum(dim=(-2, -1))
    safe_data = torch.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    values = (safe_data * weights).sum(dim=(-2, -1)) / support.clamp_min(1.0e-8)
    return values, support > 0


def _prepared_log10p1(values: torch.Tensor) -> torch.Tensor:
    return torch.log1p(values) / values.new_tensor(10.0).log()


def _normalize_prepared_values(values: torch.Tensor, source_spec: dict[str, Any]) -> torch.Tensor:
    stats = source_spec.get("stats")
    if not stats:
        raise ValueError(f"Missing preprocessing stats for source {source_spec.get('source_name')!r}")
    return (values - float(stats["mean"])) / float(stats["std"])


def _log_zscore_city_scalar(
    *,
    data: torch.Tensor,
    valid_mask: torch.Tensor,
    city_mask: torch.Tensor,
    source_spec: dict[str, Any],
) -> CityLevelTensors:
    physical = inverse_processed_tensor(data, source_spec)
    city = city_mask > 0
    while city.dim() < physical.dim():
        city = city.unsqueeze(1)
    city = city.expand_as(physical)

    valid = valid_mask.bool() & torch.isfinite(physical) & city & (physical > -1.0)
    if source_spec.get("active_stats_key") == "zero_excluded_stats":
        valid &= physical > 0

    prepared_pixels = _prepared_log10p1(physical)
    weights = valid.to(dtype=prepared_pixels.dtype)
    support = weights.sum(dim=(-2, -1))
    safe_prepared = torch.nan_to_num(prepared_pixels, nan=0.0, posinf=0.0, neginf=0.0)
    prepared_mean = (safe_prepared * weights).sum(dim=(-2, -1)) / support.clamp_min(1.0e-8)
    processed = _normalize_prepared_values(prepared_mean, source_spec)
    return CityLevelTensors(
        data=torch.where(support > 0, processed, torch.zeros_like(processed)),
        valid_mask=support > 0,
    )


def _raster_city_features(
    *,
    data: torch.Tensor,
    valid_mask: torch.Tensor,
    city_mask: torch.Tensor,
    source_spec: dict[str, Any],
) -> CityLevelTensors:
    if str(source_spec.get("mode", "")).lower() == "log_zscore":
        return _log_zscore_city_scalar(
            data=data,
            valid_mask=valid_mask,
            city_mask=city_mask,
            source_spec=source_spec,
        )
    values, valid = _channel_means(
        data=data,
        valid_mask=valid_mask,
        city_mask=city_mask,
    )
    return CityLevelTensors(data=values, valid_mask=valid)


def _city_mask_crop_mean(city_mask: torch.Tensor) -> CityLevelTensors:
    valid = torch.isfinite(city_mask)
    weights = valid.to(dtype=city_mask.dtype)
    support = weights.sum(dim=(-2, -1))
    safe_mask = torch.nan_to_num(city_mask, nan=0.0, posinf=0.0, neginf=0.0)
    values = (safe_mask * weights).sum(dim=(-2, -1)) / support.clamp_min(1.0e-8)
    return CityLevelTensors(data=values, valid_mask=support > 0)


def _append_source_slice(
    slices: list[SourceFeatureSlice],
    *,
    source_name: str,
    channel_names: tuple[str, ...],
    offset: int,
) -> int:
    width = len(channel_names)
    slices.append(
        SourceFeatureSlice(
            source_name=source_name,
            channel_names=channel_names,
            indices=tuple(range(offset, offset + width)),
        )
    )
    return offset + width


def build_city_level_feature_layout(
    *,
    pipeline_spec: PipelineSpec,
    source_specs: dict[str, dict[str, Any]],
    preprocessing_spec: dict[str, Any],
    batch_schema: ProcessedBatchSchema,
) -> CityLevelFeatureLayout:
    static_raster_sources: list[tuple[str, tuple[str, ...]]] = []
    dynamic_raster_sources: list[tuple[str, tuple[str, ...]]] = []
    scalar_sources: list[tuple[str, tuple[str, ...]]] = []

    for source_name in pipeline_spec.input_sources:
        if source_name == pipeline_spec.city_mask_source:
            continue
        source_schema = source_specs[source_name]
        channel_names = source_output_names(preprocessing_spec, source_name)
        if source_schema["modality"] == "scalar":
            scalar_sources.append((source_name, channel_names))
        elif source_schema["storage_level"] == "static":
            static_raster_sources.append((source_name, channel_names))
        else:
            dynamic_raster_sources.append((source_name, channel_names))

    raster_slices: list[SourceFeatureSlice] = []
    raster_offset = 0
    for source_name, channel_names in static_raster_sources + dynamic_raster_sources:
        raster_offset = _append_source_slice(
            raster_slices,
            source_name=source_name,
            channel_names=channel_names,
            offset=raster_offset,
        )

    scalar_slices: list[SourceFeatureSlice] = []
    scalar_offset = 0
    for source_name, channel_names in scalar_sources:
        scalar_offset = _append_source_slice(
            scalar_slices,
            source_name=source_name,
            channel_names=channel_names,
            offset=scalar_offset,
        )

    expected_raster_names = tuple(
        channel_name
        for source_slice in raster_slices
        for channel_name in source_slice.channel_names
    )
    expected_scalar_names = tuple(
        channel_name
        for source_slice in scalar_slices
        for channel_name in source_slice.channel_names
    )
    actual_raster_prefix = tuple(batch_schema.raster_input_channel_names[: len(expected_raster_names)])
    if actual_raster_prefix != expected_raster_names:
        raise ValueError(
            "Scalar MLP feature layout does not match raster input schema: "
            f"expected prefix={expected_raster_names!r}, actual={actual_raster_prefix!r}"
        )
    if tuple(batch_schema.scalar_input_names) != expected_scalar_names:
        raise ValueError(
            "Scalar MLP feature layout does not match scalar input schema: "
            f"expected={expected_scalar_names!r}, actual={tuple(batch_schema.scalar_input_names)!r}"
        )

    feature_names = tuple(
        f"{source_slice.source_name}:{channel_name}:mean"
        for source_slice in tuple(raster_slices) + tuple(scalar_slices)
        for channel_name in source_slice.channel_names
    )
    include_city_mask_mean = bool(
        pipeline_spec.include_city_mask_as_input
        and pipeline_spec.city_mask_source not in (None, "")
    )
    if include_city_mask_mean:
        feature_names = feature_names + ("city_mask:city_mask:crop_mean",)
    if not feature_names:
        raise ValueError("Scalar MLP baseline requires at least one input feature")
    return CityLevelFeatureLayout(
        raster_slices=tuple(raster_slices),
        scalar_slices=tuple(scalar_slices),
        feature_names=feature_names,
        include_city_mask_mean=include_city_mask_mean,
    )


def build_city_level_features(
    *,
    batch: ProcessedModelBatch,
    layout: CityLevelFeatureLayout,
    preprocessing_spec: dict[str, Any],
) -> CityLevelTensors:
    feature_values: list[torch.Tensor] = []
    feature_valid: list[torch.Tensor] = []

    for source_slice in layout.raster_slices:
        indices = list(source_slice.indices)
        source_spec = preprocessing_spec["sources"][source_slice.source_name]
        source_features = _raster_city_features(
            data=batch.raster_inputs[:, indices, ...],
            valid_mask=batch.raster_valid_mask[:, indices, ...],
            city_mask=batch.city_mask,
            source_spec=source_spec,
        )
        feature_values.append(source_features.data)
        feature_valid.append(source_features.valid_mask)

    for source_slice in layout.scalar_slices:
        indices = list(source_slice.indices)
        processed = batch.scalar_inputs[:, indices]
        valid = batch.scalar_valid_mask[:, indices].bool() & torch.isfinite(processed)
        feature_values.append(torch.nan_to_num(processed, nan=0.0, posinf=0.0, neginf=0.0))
        feature_valid.append(valid)

    if layout.include_city_mask_mean:
        city_mask_feature = _city_mask_crop_mean(batch.city_mask)
        feature_values.append(city_mask_feature.data)
        feature_valid.append(city_mask_feature.valid_mask)

    if not feature_values:
        raise ValueError("Scalar MLP baseline requires at least one input feature")
    return CityLevelTensors(
        data=torch.cat(feature_values, dim=1),
        valid_mask=torch.cat(feature_valid, dim=1),
    )


def build_city_level_target(
    *,
    batch: ProcessedModelBatch,
    target_schema: TargetSchema,
    preprocessing_spec: dict[str, Any],
) -> CityLevelTensors:
    source_name = target_schema.spec.source
    if source_name is None:
        raise ValueError(f"Scalar MLP target {target_schema.name!r} requires a source")

    if target_schema.name in batch.raster_targets:
        block = batch.raster_targets[target_schema.name]
        values, valid = _channel_means(
            data=block.data,
            valid_mask=block.valid_mask,
            city_mask=batch.city_mask,
        )
        return CityLevelTensors(data=values, valid_mask=valid)

    if target_schema.name in batch.scalar_targets:
        block = batch.scalar_targets[target_schema.name]
        valid = block.valid_mask.bool() & torch.isfinite(block.data)
        return CityLevelTensors(
            data=torch.nan_to_num(block.data, nan=0.0, posinf=0.0, neginf=0.0),
            valid_mask=valid,
        )

    raise KeyError(f"Target {target_schema.name!r} is missing from batch targets")


def valid_city_level_sample_mask(features: CityLevelTensors, target: CityLevelTensors) -> torch.Tensor:
    return features.valid_mask.all(dim=1) & target.valid_mask.all(dim=1)


def compute_city_level_standardization_stats(
    *,
    dataloader,
    layout: CityLevelFeatureLayout,
    target_schema: TargetSchema,
    preprocessing_spec: dict[str, Any],
    device: torch.device,
    variance_epsilon: float = 1.0e-12,
) -> StandardizationStats:
    feature_sum = torch.zeros(layout.feature_dim, dtype=torch.float64, device=device)
    feature_sum_sq = torch.zeros_like(feature_sum)
    target_dim = target_schema.channels
    count = torch.zeros(1, dtype=torch.float64, device=device)

    dataset = getattr(dataloader, "dataset", None)
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(0)
    sampler = getattr(dataloader, "sampler", None)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(0)

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            features = build_city_level_features(
                batch=batch,
                layout=layout,
                preprocessing_spec=preprocessing_spec,
            )
            target = build_city_level_target(
                batch=batch,
                target_schema=target_schema,
                preprocessing_spec=preprocessing_spec,
            )
            sample_mask = valid_city_level_sample_mask(features, target)
            if not bool(sample_mask.any().detach().cpu()):
                continue
            selected_features = features.data[sample_mask].to(dtype=torch.float64)
            feature_sum += selected_features.sum(dim=0)
            feature_sum_sq += (selected_features * selected_features).sum(dim=0)
            count += selected_features.shape[0]

    if dist.is_available() and dist.is_initialized():
        for tensor in (feature_sum, feature_sum_sq, count):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    sample_count = int(round(float(count.item())))
    if sample_count <= 0:
        raise ValueError("Scalar MLP baseline found no valid training samples for standardization")

    feature_mean = feature_sum / count.clamp_min(1.0)
    feature_var = (feature_sum_sq / count.clamp_min(1.0)) - (feature_mean * feature_mean)
    feature_std = torch.sqrt(torch.clamp(feature_var, min=0.0))
    feature_std = torch.where(
        feature_std > variance_epsilon,
        feature_std,
        torch.ones_like(feature_std),
    )
    return StandardizationStats(
        feature_mean=feature_mean.float(),
        feature_std=feature_std.float(),
        target_mean=torch.zeros(target_dim, dtype=torch.float32, device=device),
        target_std=torch.ones(target_dim, dtype=torch.float32, device=device),
        count=sample_count,
    )


__all__ = [
    "CityLevelFeatureLayout",
    "CityLevelTensors",
    "SourceFeatureSlice",
    "StandardizationStats",
    "build_city_level_feature_layout",
    "build_city_level_features",
    "build_city_level_target",
    "compute_city_level_standardization_stats",
    "valid_city_level_sample_mask",
]
