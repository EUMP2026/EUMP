from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from urban_embedding.data.contracts import CropScalarTargetSpec
from urban_embedding.data.processed.schema import CITY_MASK_SOURCE
from urban_embedding.data.processed.task_config import ProcessedTaskConfig


DEFAULT_CONTINUOUS_LOSS = "mse"
DEFAULT_INPUT_RECONSTRUCTION_NAME = "input_reconstruction"
EUMP_DOWNSTREAM_KIND = "eump_downstream"
EUMP_RECONSTRUCTION_KIND = "eump_reconstruction"
EUMP_CITY_SCALAR_DOWNSTREAM_KIND = "eump_city_scalar_downstream"
EUMP_SCALAR_MLP_DOWNSTREAM_KIND = "eump_scalar_mlp_downstream"
EUMP_CONTRASTIVE_KIND = "eump_contrastive"
EUMP_PIPELINE_KIND_ALIASES = {
    EUMP_SCALAR_MLP_DOWNSTREAM_KIND: EUMP_CITY_SCALAR_DOWNSTREAM_KIND,
}
EUMP_DOWNSTREAM_LOSSES = {"mse", "mae", "huber"}
HURDLE_LOSSES = {"hurdle_mse", "hurdle_mae", "hurdle_huber"}
RECONSTRUCTION_BINARY_LOSS_SOURCES = {"water", CITY_MASK_SOURCE, "road_density"}
RECONSTRUCTION_SUPPORTED_SLICE_LOSSES = {"mse", "mae", "huber", "bce", "ce", "kl"} | HURDLE_LOSSES
LOSS_GROUPS = {"reconstruct", "aux"}
LOSS_GROUP_REDUCTIONS = {"weighted_mean"}
RECONSTRUCT_REDUCTIONS = {"slice_weighted_mean", "source_weighted_mean"}
AUX_REDUCTIONS = {"target_weighted_mean", "source_weighted_mean"}
HURDLE_REDUCTIONS = {"sum", "weighted_mean"}
NORMALIZATION_STATS_KEYS = {"stats", "zero_excluded_stats"}


def _as_tuple(value: Any) -> tuple:
    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _as_mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a mapping, got {type(value)!r}")
    return value


def _dedupe(values: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return tuple(out)


def normalize_eump_pipeline_kind(kind: Any) -> str:
    normalized = str(kind)
    return EUMP_PIPELINE_KIND_ALIASES.get(normalized, normalized)


@dataclass(frozen=True, slots=True)
class PipelineLossConfig:
    reconstruct_weight: float = 1.0
    aux_weight: float = 1.0
    group_reduction: str = "weighted_mean"
    reconstruct_reduction: str = "slice_weighted_mean"
    aux_reduction: str = "target_weighted_mean"
    hurdle_reduction: str = "sum"
    hurdle_cls_weight: float = 1.0
    hurdle_reg_weight: float = 1.0
    aux_source_weights: dict[str, float] | None = None
    log_every_n_steps: int = 10
    tensorboard: bool = True

    def __post_init__(self) -> None:
        if self.reconstruct_weight < 0:
            raise ValueError(f"pipeline.loss.reconstruct_weight must be non-negative, got {self.reconstruct_weight}")
        if self.aux_weight < 0:
            raise ValueError(f"pipeline.loss.aux_weight must be non-negative, got {self.aux_weight}")
        if self.hurdle_cls_weight < 0:
            raise ValueError(f"pipeline.loss.hurdle_cls_weight must be non-negative, got {self.hurdle_cls_weight}")
        if self.hurdle_reg_weight < 0:
            raise ValueError(f"pipeline.loss.hurdle_reg_weight must be non-negative, got {self.hurdle_reg_weight}")
        if self.group_reduction not in LOSS_GROUP_REDUCTIONS:
            raise ValueError(
                f"Unsupported pipeline.loss.group_reduction={self.group_reduction!r}; "
                f"expected one of {sorted(LOSS_GROUP_REDUCTIONS)}"
            )
        if self.reconstruct_reduction not in RECONSTRUCT_REDUCTIONS:
            raise ValueError(
                f"Unsupported pipeline.loss.reconstruct_reduction={self.reconstruct_reduction!r}; "
                f"expected one of {sorted(RECONSTRUCT_REDUCTIONS)}"
            )
        if self.aux_reduction not in AUX_REDUCTIONS:
            raise ValueError(
                f"Unsupported pipeline.loss.aux_reduction={self.aux_reduction!r}; "
                f"expected one of {sorted(AUX_REDUCTIONS)}"
            )
        if self.hurdle_reduction not in HURDLE_REDUCTIONS:
            raise ValueError(
                f"Unsupported pipeline.loss.hurdle_reduction={self.hurdle_reduction!r}; "
                f"expected one of {sorted(HURDLE_REDUCTIONS)}"
            )
        raw_aux_source_weights = {} if self.aux_source_weights is None else _as_mapping(
            self.aux_source_weights,
            "pipeline.loss.aux_source_weights",
        )
        aux_source_weights = {
            str(source_name): float(weight)
            for source_name, weight in raw_aux_source_weights.items()
        }
        invalid_aux_weights = {
            source_name: weight
            for source_name, weight in aux_source_weights.items()
            if weight < 0
        }
        if invalid_aux_weights:
            raise ValueError(
                "pipeline.loss.aux_source_weights must be non-negative: "
                f"{invalid_aux_weights!r}"
            )
        object.__setattr__(self, "aux_source_weights", aux_source_weights)
        if self.log_every_n_steps <= 0:
            raise ValueError("pipeline.loss.log_every_n_steps must be positive")

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> "PipelineLossConfig":
        payload = {} if payload is None else _as_mapping(payload, "pipeline.loss")
        group_reduction = str(payload.get("group_reduction", "weighted_mean"))
        log_every_n_steps = int(payload.get("log_every_n_steps", 10))
        return cls(
            reconstruct_weight=float(payload.get("reconstruct_weight", 1.0)),
            aux_weight=float(payload.get("aux_weight", 1.0)),
            group_reduction=group_reduction,
            reconstruct_reduction=str(payload.get("reconstruct_reduction", "slice_weighted_mean")),
            aux_reduction=str(payload.get("aux_reduction", "target_weighted_mean")),
            hurdle_reduction=str(payload.get("hurdle_reduction", "sum")),
            hurdle_cls_weight=float(payload.get("hurdle_cls_weight", 1.0)),
            hurdle_reg_weight=float(payload.get("hurdle_reg_weight", 1.0)),
            aux_source_weights=dict(
                _as_mapping(payload.get("aux_source_weights"), "pipeline.loss.aux_source_weights")
            ),
            log_every_n_steps=log_every_n_steps,
            tensorboard=bool(payload.get("tensorboard", True)),
        )

    def group_weight(self, group: str) -> float:
        if group == "reconstruct":
            return self.reconstruct_weight
        if group == "aux":
            return self.aux_weight
        raise ValueError(f"Unsupported loss group {group!r}")


@dataclass(frozen=True, slots=True)
class PipelineNormalizationConfig:
    hurdle_stats_key: str = "stats"
    source_stats_overrides: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if self.hurdle_stats_key not in NORMALIZATION_STATS_KEYS:
            raise ValueError(
                f"Unsupported pipeline.normalization.hurdle_stats_key={self.hurdle_stats_key!r}; "
                f"expected one of {sorted(NORMALIZATION_STATS_KEYS)}"
            )
        raw_overrides = {} if self.source_stats_overrides is None else _as_mapping(
            self.source_stats_overrides,
            "pipeline.normalization.source_stats_overrides",
        )
        overrides = {str(source_name): str(stats_key) for source_name, stats_key in raw_overrides.items()}
        invalid = {
            source_name: stats_key
            for source_name, stats_key in overrides.items()
            if stats_key not in NORMALIZATION_STATS_KEYS
        }
        if invalid:
            raise ValueError(
                "Unsupported pipeline.normalization.source_stats_overrides entries "
                f"{invalid!r}; expected values from {sorted(NORMALIZATION_STATS_KEYS)}"
            )
        object.__setattr__(self, "source_stats_overrides", overrides)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> "PipelineNormalizationConfig":
        payload = {} if payload is None else _as_mapping(payload, "pipeline.normalization")
        return cls(
            hurdle_stats_key=str(payload.get("hurdle_stats_key", "stats")),
            source_stats_overrides=dict(
                _as_mapping(
                    payload.get("source_stats_overrides"),
                    "pipeline.normalization.source_stats_overrides",
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class InputChannelDropConfig:
    """Inference-time random input channel dropout.

    When enabled, each crop sample independently zero-fills ``k`` of its encoder
    input channels (``k`` drawn uniformly from ``[min_drop, max_drop]``) and sets
    the corresponding raster ``valid_mask`` rows to ``False``. The draw is
    deterministic per ``(seed, scene_id, year, crop_box)`` so runs reproduce and
    DDP ranks agree. This never removes channels from the encoder input conv; it
    only blanks them, matching ``optional_sources`` zero-fill semantics.
    """

    enabled: bool = False
    min_drop: int = 1
    max_drop: int = 2
    seed: int = 0
    droppable_channels: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not self.enabled:
            return
        if self.min_drop < 1:
            raise ValueError(f"pipeline.input_channel_drop.min_drop must be >= 1, got {self.min_drop}")
        if self.max_drop < self.min_drop:
            raise ValueError(
                "pipeline.input_channel_drop.max_drop must be >= min_drop, got "
                f"max_drop={self.max_drop}, min_drop={self.min_drop}"
            )
        if self.droppable_channels is not None and len(self.droppable_channels) != len(set(self.droppable_channels)):
            raise ValueError(
                "pipeline.input_channel_drop.droppable_channels must not contain duplicates: "
                f"{self.droppable_channels!r}"
            )

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> "InputChannelDropConfig":
        payload = {} if payload is None else _as_mapping(payload, "pipeline.input_channel_drop")
        raw_channels = payload.get("droppable_channels")
        droppable_channels = None if raw_channels is None else tuple(str(name) for name in _as_tuple(raw_channels))
        return cls(
            enabled=bool(payload.get("enabled", False)),
            min_drop=int(payload.get("min_drop", 1)),
            max_drop=int(payload.get("max_drop", 2)),
            seed=int(payload.get("seed", 0)),
            droppable_channels=droppable_channels,
        )

    def resolve_droppable_indices(self, channel_names: tuple[str, ...]) -> tuple[int, ...]:
        """Resolve configured droppable channel names to indices in ``channel_names``.

        ``droppable_channels=None`` means every encoder input channel is eligible.
        """
        if self.droppable_channels is None:
            indices = tuple(range(len(channel_names)))
        else:
            name_to_index = {str(name): idx for idx, name in enumerate(channel_names)}
            unknown = [name for name in self.droppable_channels if name not in name_to_index]
            if unknown:
                raise ValueError(
                    "pipeline.input_channel_drop.droppable_channels references channels not in the "
                    f"encoder input set {channel_names!r}: {unknown!r}"
                )
            indices = tuple(name_to_index[name] for name in self.droppable_channels)
        if self.enabled and self.max_drop > len(indices):
            raise ValueError(
                "pipeline.input_channel_drop.max_drop="
                f"{self.max_drop} exceeds droppable channel count {len(indices)}"
            )
        return indices


@dataclass(frozen=True, slots=True)
class PipelineTargetSpec:
    name: str
    target_type: str
    source: str | None = None
    loss: str = DEFAULT_CONTINUOUS_LOSS
    weight: float = 1.0
    group: str = "aux"
    mask: str = "valid"
    source_kind: str | None = None
    channel_indices: tuple[int, ...] | None = None
    reduction: str = "weighted_mean"
    value_transform: str = "identity"
    use_city_mask: bool = True
    min_valid_ratio: float = 0.0
    min_city_mask_coverage: float = 0.0
    output_names: tuple[str, ...] | None = None
    optional: bool = False
    aggregation_scale: str = "processed"
    normalization_scope: str = "global"
    positive_only: bool = False
    city_weight: float = 1.0
    source_losses: dict[str, str] | None = None
    source_weights: dict[str, float] | None = None
    scalar_sources: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.group not in LOSS_GROUPS:
            raise ValueError(f"Unsupported loss group {self.group!r}; expected one of {sorted(LOSS_GROUPS)}")
        if self.weight < 0:
            raise ValueError(f"Target {self.name!r} weight must be non-negative, got {self.weight}")
        if self.min_valid_ratio < 0 or self.min_valid_ratio > 1:
            raise ValueError(
                f"Target {self.name!r} min_valid_ratio must be in [0, 1], got {self.min_valid_ratio}"
            )
        if self.min_city_mask_coverage < 0 or self.min_city_mask_coverage > 1:
            raise ValueError(
                f"Target {self.name!r} min_city_mask_coverage must be in [0, 1], "
                f"got {self.min_city_mask_coverage}"
            )
        if self.target_type == "crop_scalar" and self.aggregation_scale not in {"processed", "physical"}:
            raise ValueError(
                f"Crop scalar target {self.name!r} has unsupported aggregation_scale={self.aggregation_scale!r}; "
                "expected 'processed' or 'physical'"
            )
        if self.target_type == "crop_scalar" and self.normalization_scope not in {"global", "city_crop", "zero_excluded"}:
            raise ValueError(
                f"Crop scalar target {self.name!r} has unsupported normalization_scope={self.normalization_scope!r}; "
                "expected 'global', 'city_crop', or 'zero_excluded'"
            )
        if self.target_type == "crop_scalar" and self.value_transform not in {"identity", "positive_indicator"}:
            raise ValueError(
                f"Crop scalar target {self.name!r} has unsupported value_transform={self.value_transform!r}; "
                "expected 'identity' or 'positive_indicator'"
            )
        if (
            self.target_type == "crop_scalar"
            and self.normalization_scope in {"city_crop", "zero_excluded"}
            and self.aggregation_scale != "physical"
        ):
            raise ValueError(
                f"Crop scalar target {self.name!r} uses normalization_scope={self.normalization_scope!r} "
                "but aggregation_scale is not 'physical'"
            )
        if self.target_type == "crop_scalar" and self.positive_only and self.aggregation_scale != "physical":
            raise ValueError(f"Crop scalar target {self.name!r} uses positive_only=true but aggregation_scale is not 'physical'")
        if self.target_type == "crop_scalar" and self.positive_only and self.value_transform != "identity":
            raise ValueError(
                f"Crop scalar target {self.name!r} cannot combine positive_only=true with "
                f"value_transform={self.value_transform!r}"
            )
        if self.source_losses is not None:
            invalid = {
                key: value
                for key, value in self.source_losses.items()
                if str(value).lower() not in RECONSTRUCTION_SUPPORTED_SLICE_LOSSES
            }
            if invalid:
                raise ValueError(
                    f"Target {self.name!r} has unsupported source_losses entries {invalid!r}; "
                    f"expected one of {sorted(RECONSTRUCTION_SUPPORTED_SLICE_LOSSES)}"
                )
        if self.source_weights is not None:
            invalid_weights = {
                key: value
                for key, value in self.source_weights.items()
                if float(value) < 0
            }
            if invalid_weights:
                raise ValueError(f"Target {self.name!r} source_weights must be non-negative: {invalid_weights!r}")

    @classmethod
    def input_reconstruction(cls, payload: dict[str, Any] | None = None) -> "PipelineTargetSpec":
        payload = {} if payload is None else dict(payload)
        source_losses = {
            str(key): str(value).lower()
            for key, value in _as_mapping(payload.get("source_losses"), "input_reconstruction.source_losses").items()
        }
        source_weights = {
            str(key): float(value)
            for key, value in _as_mapping(payload.get("source_weights"), "input_reconstruction.source_weights").items()
        }
        scalar_sources = []
        for item in _as_tuple(payload.get("scalar_sources", ())):
            if isinstance(item, dict):
                item_payload = _as_mapping(item, "input_reconstruction.scalar_sources entry")
                source = item_payload.get("source", item_payload.get("name"))
                if source is None:
                    raise ValueError("input_reconstruction.scalar_sources mapping entries require source or name")
                scalar_sources.append(str(source))
            else:
                scalar_sources.append(str(item))
        return cls(
            name=str(payload.get("name", DEFAULT_INPUT_RECONSTRUCTION_NAME)),
            target_type="input_reconstruction",
            loss=str(payload.get("loss", DEFAULT_CONTINUOUS_LOSS)),
            weight=float(payload.get("weight", 1.0)),
            group=str(payload.get("group", "reconstruct")),
            mask=str(payload.get("mask", "valid_and_masked")),
            source_losses=source_losses,
            source_weights=source_weights,
            scalar_sources=_dedupe(scalar_sources),
        )

    @classmethod
    def from_mapping(cls, payload: dict[str, Any], target_type: str) -> "PipelineTargetSpec":
        source = payload.get("source")
        name = str(payload.get("name", source if source is not None else target_type))
        channel_indices = payload.get("channel_indices")
        if channel_indices is not None:
            channel_indices = tuple(int(idx) for idx in channel_indices)
        output_names = payload.get("output_names")
        if output_names is not None:
            output_names = tuple(str(name) for name in output_names)
        return cls(
            name=name,
            target_type=target_type,
            source=None if source is None else str(source),
            loss=str(payload.get("loss", DEFAULT_CONTINUOUS_LOSS)),
            weight=float(payload.get("weight", 1.0)),
            group=str(payload.get("group", "aux")),
            mask=str(payload.get("mask", "valid")),
            source_kind=payload.get("source_kind"),
            channel_indices=channel_indices,
            reduction=str(payload.get("reduction", "weighted_mean")),
            value_transform=str(payload.get("value_transform", "identity")),
            use_city_mask=bool(payload.get("use_city_mask", True)),
            min_valid_ratio=float(payload.get("min_valid_ratio", 0.0)),
            min_city_mask_coverage=float(payload.get("min_city_mask_coverage", 0.0)),
            output_names=output_names,
            optional=bool(payload.get("optional", False)),
            aggregation_scale=str(payload.get("aggregation_scale", "processed")),
            normalization_scope=str(payload.get("normalization_scope", "global")),
            positive_only=bool(payload.get("positive_only", False)),
            city_weight=float(payload.get("city_weight", 1.0)),
        )

    def to_crop_scalar_spec(self, *, apply_min_city_mask_coverage: bool) -> CropScalarTargetSpec:
        if self.target_type != "crop_scalar":
            raise ValueError(f"Target {self.name!r} is not a crop_scalar target")
        if self.source_kind is None:
            raise ValueError(f"Crop scalar target {self.name!r} requires source_kind")
        return CropScalarTargetSpec(
            source_kind=self.source_kind,
            source_name=self.source,
            channel_indices=self.channel_indices,
            reduction=self.reduction,
            value_transform=self.value_transform,
            use_city_mask=self.use_city_mask,
            min_valid_ratio=self.min_valid_ratio,
            min_city_mask_coverage=self.min_city_mask_coverage if apply_min_city_mask_coverage else 0.0,
            output_names=self.output_names,
            aggregation_scale=self.aggregation_scale,
            normalization_scope=self.normalization_scope,
            positive_only=self.positive_only,
            optional=self.optional,
        )


def _input_reconstruction_scalar_targets(
    payload: dict[str, Any],
    *,
    input_sources: tuple[str, ...],
    input_reconstruction_target: PipelineTargetSpec,
) -> tuple[PipelineTargetSpec, ...]:
    scalar_sources = payload.get("scalar_sources", ())
    if not scalar_sources:
        return ()

    scalar_loss = str(payload.get("scalar_loss", input_reconstruction_target.loss)).lower()
    scalar_weight = float(payload.get("scalar_weight", input_reconstruction_target.weight))
    scalar_group = str(payload.get("scalar_group", input_reconstruction_target.group))
    scalar_mask = str(payload.get("scalar_mask", "valid"))
    source_losses = {
        str(key): str(value).lower()
        for key, value in _as_mapping(
            payload.get("scalar_source_losses"),
            "input_reconstruction.scalar_source_losses",
        ).items()
    }
    source_weights = {
        str(key): float(value)
        for key, value in _as_mapping(
            payload.get("scalar_source_weights"),
            "input_reconstruction.scalar_source_weights",
        ).items()
    }

    targets: list[PipelineTargetSpec] = []
    for item in _as_tuple(scalar_sources):
        item_payload: dict[str, Any]
        if isinstance(item, dict):
            item_payload = dict(item)
            source = item_payload.get("source", item_payload.get("name"))
            if source is None:
                raise ValueError("input_reconstruction.scalar_sources mapping entries require source or name")
            source = str(source)
            name = str(item_payload.get("name", source))
        else:
            source = str(item)
            name = source
            item_payload = {}

        if source not in input_sources:
            raise ValueError(
                "input_reconstruction.scalar_sources entries must also be present in "
                f"pipeline.inputs.sources; missing {source!r}"
            )

        targets.append(
            PipelineTargetSpec.from_mapping(
                {
                    "name": name,
                    "source": source,
                    "loss": item_payload.get("loss", source_losses.get(source, scalar_loss)),
                    "weight": item_payload.get("weight", source_weights.get(source, scalar_weight)),
                    "group": item_payload.get("group", scalar_group),
                    "mask": item_payload.get("mask", scalar_mask),
                },
                "scalar",
            )
        )

    unknown_loss_keys = sorted(set(source_losses) - {target.source for target in targets})
    unknown_weight_keys = sorted(set(source_weights) - {target.source for target in targets})
    if unknown_loss_keys:
        raise ValueError(
            "input_reconstruction.scalar_source_losses contains keys not present in "
            f"input_reconstruction.scalar_sources: {unknown_loss_keys}"
        )
    if unknown_weight_keys:
        raise ValueError(
            "input_reconstruction.scalar_source_weights contains keys not present in "
            f"input_reconstruction.scalar_sources: {unknown_weight_keys}"
        )

    return tuple(targets)


@dataclass(frozen=True, slots=True)
class ReconstructionLossSlice:
    name: str
    source: str
    channel_indices: tuple[int, ...]
    channel_names: tuple[str, ...]
    loss: str
    weight: float = 1.0
    source_weight: float = 1.0
    inner_weight: float = 1.0
    prediction_channel_indices: tuple[int, ...] = ()
    hurdle_logit_channel_indices: tuple[int, ...] = ()
    positive_thresholds: tuple[float, ...] = ()

    @property
    def channels(self) -> int:
        return len(self.channel_indices)

    @property
    def prediction_channels(self) -> int:
        if not self.prediction_channel_indices and not self.hurdle_logit_channel_indices:
            return self.channels
        return len(self.prediction_channel_indices) + len(self.hurdle_logit_channel_indices)


@dataclass(frozen=True, slots=True)
class TargetSchema:
    spec: PipelineTargetSpec
    channel_names: tuple[str, ...]
    loss_slices: tuple[ReconstructionLossSlice, ...] = ()
    prediction_channels_override: int | None = None
    positive_thresholds: tuple[float, ...] = ()

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def channels(self) -> int:
        return len(self.channel_names)

    @property
    def prediction_channels(self) -> int:
        if self.prediction_channels_override is not None:
            return int(self.prediction_channels_override)
        if self.loss_slices:
            return sum(loss_slice.prediction_channels for loss_slice in self.loss_slices)
        if self.spec.loss.lower() in HURDLE_LOSSES:
            return self.channels * 2
        return self.channels


@dataclass(frozen=True, slots=True)
class ProcessedBatchSchema:
    raster_input_channel_names: tuple[str, ...]
    scalar_input_names: tuple[str, ...]
    raster_targets: tuple[TargetSchema, ...]
    scalar_targets: tuple[TargetSchema, ...]

    @property
    def raster_input_channels(self) -> int:
        return len(self.raster_input_channel_names)

    @property
    def scalar_input_dim(self) -> int:
        return len(self.scalar_input_names)

    @property
    def raster_output_channels(self) -> int:
        return sum(target.prediction_channels for target in self.raster_targets)

    @property
    def scalar_output_dims(self) -> dict[str, int]:
        return {target.name: target.channels for target in self.scalar_targets}


@dataclass(frozen=True, slots=True)
class PipelineSpec:
    kind: str
    input_sources: tuple[str, ...]
    targets: tuple[PipelineTargetSpec, ...]
    optional_input_sources: tuple[str, ...] = ()
    loss_config: PipelineLossConfig = PipelineLossConfig()
    normalization_config: PipelineNormalizationConfig = PipelineNormalizationConfig()
    input_channel_drop: InputChannelDropConfig = InputChannelDropConfig()
    crop_size: int | tuple[int, int] = 256
    center_ratio: float = 0.0
    broadcast_scalar: bool = False
    city_mask_source: str | None = CITY_MASK_SOURCE
    include_city_mask_as_input: bool = False
    year_range: tuple[int, int] | None = None
    test_size: float = 0.3
    num_workers: int = 16
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = True
    cache_scenes: bool = False
    preload_scenes: bool = False

    def __post_init__(self) -> None:
        unknown_optional_inputs = sorted(set(self.optional_input_sources) - set(self.input_sources))
        if unknown_optional_inputs:
            raise ValueError(
                "pipeline.inputs.optional_sources must be a subset of pipeline.inputs.sources; "
                f"unknown optional input(s): {unknown_optional_inputs}"
            )
        if self.num_workers < 0:
            raise ValueError(f"pipeline.num_workers must be >= 0, got {self.num_workers}")
        if self.prefetch_factor < 1:
            raise ValueError(f"pipeline.prefetch_factor must be >= 1, got {self.prefetch_factor}")
        if self.include_city_mask_as_input and self.city_mask_source is None:
            raise ValueError("pipeline.include_city_mask_as_input=true requires pipeline.city_mask_source")
        for target in self.targets:
            if not target.optional:
                continue
            if self.kind in {EUMP_DOWNSTREAM_KIND, EUMP_CITY_SCALAR_DOWNSTREAM_KIND}:
                raise ValueError(f"{self.kind} targets cannot be optional")
            if target.target_type not in {"raster", "scalar", "crop_scalar"} or target.group != "aux":
                raise ValueError(
                    "pipeline target optional=true is supported only for reconstruction aux "
                    "raster/scalar/crop_scalar targets"
                )

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> "PipelineSpec":
        payload = {} if payload is None else dict(payload)
        inputs_payload = _as_mapping(payload.get("inputs"), "pipeline.inputs")
        targets_payload = _as_mapping(payload.get("targets"), "pipeline.targets")

        input_sources = inputs_payload.get("sources", payload.get("input_sources", ()))
        input_sources = tuple(str(source) for source in _as_tuple(input_sources))
        optional_input_sources = tuple(
            str(source)
            for source in _as_tuple(
                inputs_payload.get("optional_sources", payload.get("optional_input_sources", ()))
            )
        )

        targets: list[PipelineTargetSpec] = []
        input_reconstruction = targets_payload.get("input_reconstruction", {})
        if input_reconstruction is True:
            input_reconstruction_target = PipelineTargetSpec.input_reconstruction()
            targets.append(input_reconstruction_target)
        elif input_reconstruction:
            input_reconstruction = _as_mapping(input_reconstruction, "input_reconstruction")
            if input_reconstruction.get("enabled", True):
                input_reconstruction_target = PipelineTargetSpec.input_reconstruction(input_reconstruction)
                targets.append(input_reconstruction_target)
                targets.extend(
                    _input_reconstruction_scalar_targets(
                        input_reconstruction,
                        input_sources=input_sources,
                        input_reconstruction_target=input_reconstruction_target,
                    )
                )

        for item in targets_payload.get("raster", ()):
            targets.append(PipelineTargetSpec.from_mapping(_as_mapping(item, "raster target"), "raster"))
        for item in targets_payload.get("scalar", ()):
            targets.append(PipelineTargetSpec.from_mapping(_as_mapping(item, "scalar target"), "scalar"))
        for item in targets_payload.get("crop_scalar", ()):
            targets.append(PipelineTargetSpec.from_mapping(_as_mapping(item, "crop scalar target"), "crop_scalar"))

        crop_size = payload.get("crop_size", 256)
        if isinstance(crop_size, list):
            crop_size = tuple(int(value) for value in crop_size)
        elif isinstance(crop_size, tuple):
            crop_size = tuple(int(value) for value in crop_size)
        else:
            crop_size = int(crop_size)

        year_range = payload.get("year_range")
        if year_range is not None:
            year_range = tuple(int(year) for year in year_range)
            if len(year_range) != 2:
                raise ValueError("pipeline.year_range must contain exactly two years")

        return cls(
            kind=normalize_eump_pipeline_kind(payload.get("kind", EUMP_RECONSTRUCTION_KIND)),
            input_sources=input_sources,
            optional_input_sources=optional_input_sources,
            targets=tuple(targets),
            loss_config=PipelineLossConfig.from_mapping(payload.get("loss")),
            normalization_config=PipelineNormalizationConfig.from_mapping(payload.get("normalization")),
            input_channel_drop=InputChannelDropConfig.from_mapping(payload.get("input_channel_drop")),
            crop_size=crop_size,
            center_ratio=float(payload.get("center_ratio", 0.0)),
            broadcast_scalar=bool(payload.get("broadcast_scalar", False)),
            city_mask_source=payload.get("city_mask_source", CITY_MASK_SOURCE),
            include_city_mask_as_input=bool(payload.get("include_city_mask_as_input", False)),
            year_range=year_range,
            test_size=float(payload.get("test_size", 0.3)),
            num_workers=int(payload.get("num_workers", 16)),
            pin_memory=bool(payload.get("pin_memory", True)),
            prefetch_factor=int(payload.get("prefetch_factor", 2)),
            persistent_workers=bool(payload.get("persistent_workers", True)),
            cache_scenes=bool(payload.get("cache_scenes", False)),
            preload_scenes=bool(payload.get("preload_scenes", False)),
        )

    def to_task_config(self) -> ProcessedTaskConfig:
        aux_sources: list[str] = []
        optional_aux_sources: list[str] = []
        prediction_sources: list[str] = []
        reconstruct_target_source: str | None = None

        for target in self.targets:
            if target.target_type in {"raster", "scalar"}:
                if target.source is None:
                    raise ValueError(f"Target {target.name!r} requires a source")
                if target.optional:
                    optional_aux_sources.append(target.source)
                else:
                    aux_sources.append(target.source)
            elif target.target_type == "crop_scalar" and target.source is not None:
                if target.source_kind == "prediction_raster":
                    prediction_sources.append(target.source)
                elif target.source_kind == "reconstruct_raster":
                    reconstruct_target_source = reconstruct_target_source or target.source
                elif target.source_kind in {"aux_raster"}:
                    if target.optional:
                        optional_aux_sources.append(target.source)
                    else:
                        aux_sources.append(target.source)

        return ProcessedTaskConfig(
            input_sources=self.input_sources,
            optional_input_sources=self.optional_input_sources,
            reconstruct_target_source=reconstruct_target_source,
            aux_target_sources=_dedupe(aux_sources),
            optional_aux_target_sources=_dedupe(optional_aux_sources),
            prediction_sources=_dedupe(prediction_sources),
            city_mask_source=self.city_mask_source,
        )

    @property
    def crop_scalar_specs(self) -> dict[str, CropScalarTargetSpec]:
        apply_min_city_mask_coverage = self.kind == EUMP_RECONSTRUCTION_KIND
        return {
            target.name: target.to_crop_scalar_spec(
                apply_min_city_mask_coverage=apply_min_city_mask_coverage,
            )
            for target in self.targets
            if target.target_type == "crop_scalar"
        }


def source_output_names(preprocessing_spec: dict[str, Any], source_name: str) -> tuple[str, ...]:
    source_spec = preprocessing_spec["sources"][source_name]
    return tuple(str(name) for name in source_spec["output_channel_names"])


def _select_channel_names(names: tuple[str, ...], target: PipelineTargetSpec) -> tuple[str, ...]:
    if target.output_names is not None:
        return target.output_names
    if target.channel_indices is None:
        return names
    return tuple(names[idx] for idx in target.channel_indices)


def _resolve_reconstruction_override(
    mapping: dict[str, Any] | None,
    *,
    source_name: str,
    channel_names: tuple[str, ...],
    default: Any,
    used_keys: set[str],
) -> Any:
    if not mapping:
        return default
    if source_name in mapping:
        used_keys.add(source_name)
        return mapping[source_name]

    channel_values: list[Any] = []
    for channel_name in channel_names:
        if channel_name in mapping:
            used_keys.add(channel_name)
            channel_values.append(mapping[channel_name])
    if not channel_values:
        return default
    first = channel_values[0]
    if any(value != first for value in channel_values[1:]):
        raise ValueError(
            f"Conflicting input_reconstruction overrides for source {source_name!r} "
            f"channels {channel_names!r}"
        )
    return first


def _resolve_reconstruction_source_weight(
    mapping: dict[str, float] | None,
    *,
    source_name: str,
    used_keys: set[str],
) -> float:
    if mapping and source_name in mapping:
        used_keys.add(source_name)
        return float(mapping[source_name])
    return 1.0


def _resolve_reconstruction_inner_weight(
    mapping: dict[str, float] | None,
    *,
    channel_names: tuple[str, ...],
    used_keys: set[str],
) -> float:
    if not mapping:
        return 1.0
    channel_values: list[float] = []
    for channel_name in channel_names:
        if channel_name in mapping:
            used_keys.add(channel_name)
            channel_values.append(float(mapping[channel_name]))
    if not channel_values:
        return 1.0
    first = channel_values[0]
    if any(value != first for value in channel_values[1:]):
        raise ValueError(
            "Conflicting input_reconstruction source_weights for channels "
            f"{channel_names!r}"
        )
    return first


def _default_reconstruction_loss(
    *,
    source_name: str,
    preprocessing_source_spec: dict[str, Any],
    fallback_loss: str,
) -> str:
    if str(preprocessing_source_spec.get("mode", "")).lower() == "one_hot":
        return "ce"
    if source_name in RECONSTRUCTION_BINARY_LOSS_SOURCES:
        return "bce"
    return fallback_loss.lower()


def is_hurdle_loss(loss: str) -> bool:
    return str(loss).lower() in HURDLE_LOSSES


def _stats_for_key(source_name: str, source_spec: dict[str, Any], stats_key: str) -> dict[str, Any]:
    stats = source_spec.get(stats_key)
    if not stats:
        raise ValueError(
            f"Source {source_name!r} requires preprocessing stats {stats_key!r}. "
            "Regenerate preprocessing_v1.json with the current build_eump_preprocessing.py."
        )
    return stats


def _processed_zero_threshold(source_name: str, source_spec: dict[str, Any]) -> float:
    mode = str(source_spec.get("mode", "")).lower()
    if mode in {"log_zscore", "zscore"}:
        stats = _stats_for_key(source_name, source_spec, "stats")
        prepared_zero = 0.0
        return (prepared_zero - float(stats["mean"])) / float(stats["std"])
    if mode in {"pass_through", "divide"}:
        return 0.0
    raise ValueError(f"Hurdle loss is unsupported for source {source_name!r} with preprocessing mode={mode!r}")


def _source_for_output_name(preprocessing_spec: dict[str, Any], output_name: str) -> str | None:
    for source_name, source_spec in preprocessing_spec.get("sources", {}).items():
        if output_name in tuple(str(name) for name in source_spec.get("output_channel_names", ())):
            return str(source_name)
    return None


def _input_reconstruction_hurdle_sources(
    target: PipelineTargetSpec,
    pipeline_spec: "PipelineSpec",
    source_specs: dict[str, dict[str, Any]],
    preprocessing_spec: dict[str, Any],
) -> set[str]:
    sources: set[str] = set()
    if is_hurdle_loss(target.loss):
        sources.update(str(source_name) for source_name in pipeline_spec.input_sources)
    for key, loss in (target.source_losses or {}).items():
        if not is_hurdle_loss(loss):
            continue
        if key in preprocessing_spec.get("sources", {}):
            sources.add(str(key))
            continue
        source_name = _source_for_output_name(preprocessing_spec, str(key))
        if source_name is not None:
            sources.add(source_name)

    selected: set[str] = set()
    for source_name in sources:
        source_spec = preprocessing_spec.get("sources", {}).get(source_name)
        source_schema = source_specs.get(source_name, {})
        if not source_spec or source_schema.get("modality") != "raster":
            continue
        if str(source_spec.get("mode", "")).lower() == "one_hot":
            continue
        selected.add(source_name)
    return selected


def hurdle_normalized_source_names(
    pipeline_spec: "PipelineSpec",
    source_specs: dict[str, dict[str, Any]],
    preprocessing_spec: dict[str, Any],
) -> set[str]:
    sources: set[str] = set()
    for target in pipeline_spec.targets:
        if target.target_type == "input_reconstruction":
            sources.update(
                _input_reconstruction_hurdle_sources(
                    target,
                    pipeline_spec,
                    source_specs,
                    preprocessing_spec,
                )
            )
        elif target.target_type == "raster" and target.source is not None and is_hurdle_loss(target.loss):
            sources.add(target.source)
        elif (
            target.target_type == "crop_scalar"
            and target.source is not None
            and target.normalization_scope == "zero_excluded"
        ):
            sources.add(target.source)
    return sources


def build_active_preprocessing_spec(
    pipeline_spec: "PipelineSpec",
    source_specs: dict[str, dict[str, Any]],
    preprocessing_spec: dict[str, Any],
) -> dict[str, Any]:
    stats_key = pipeline_spec.normalization_config.hurdle_stats_key
    active_stats_by_source: dict[str, str] = {}

    if stats_key != "stats":
        for source_name in hurdle_normalized_source_names(pipeline_spec, source_specs, preprocessing_spec):
            active_stats_by_source[source_name] = stats_key
    active_stats_by_source.update(pipeline_spec.normalization_config.source_stats_overrides or {})
    if not active_stats_by_source:
        return preprocessing_spec

    active_spec = deepcopy(preprocessing_spec)
    changed = False
    for source_name, source_stats_key in sorted(active_stats_by_source.items()):
        source_spec = active_spec["sources"].get(source_name)
        if source_spec is None:
            raise KeyError(f"Source {source_name!r} is missing from preprocessing spec")
        if source_stats_key == "stats":
            continue
        mode = str(source_spec.get("mode", "")).lower()
        replacement_stats = deepcopy(_stats_for_key(source_name, source_spec, source_stats_key))
        source_spec["global_stats"] = deepcopy(source_spec.get("stats"))
        source_spec["stats"] = replacement_stats
        source_spec["active_stats_key"] = source_stats_key
        if mode in {"log_zscore", "zscore"}:
            changed = True
    return active_spec if changed else preprocessing_spec


def _hurdle_thresholds_for_loss(
    *,
    source_name: str,
    source_spec: dict[str, Any],
    loss: str,
    channels: int,
) -> tuple[float, ...]:
    if not is_hurdle_loss(loss):
        return ()
    threshold = _processed_zero_threshold(source_name, source_spec)
    return tuple(float(threshold) for _ in range(int(channels)))


def _build_reconstruction_loss_slices(
    target: PipelineTargetSpec,
    raster_input_items: list[tuple[str, tuple[str, ...]]],
    preprocessing_spec: dict[str, Any],
) -> tuple[ReconstructionLossSlice, ...]:
    slices: list[ReconstructionLossSlice] = []
    used_loss_keys: set[str] = set()
    used_weight_keys: set[str] = set()
    offset = 0
    prediction_offset = 0

    for source_name, channel_names in raster_input_items:
        preprocessing_source_spec = preprocessing_spec["sources"].get(source_name, {})
        mode = str(preprocessing_source_spec.get("mode", "")).lower()
        if mode == "one_hot":
            indices = tuple(range(offset, offset + len(channel_names)))
            default_loss = _default_reconstruction_loss(
                source_name=source_name,
                preprocessing_source_spec=preprocessing_source_spec,
                fallback_loss=target.loss,
            )
            loss = str(
                _resolve_reconstruction_override(
                    target.source_losses,
                    source_name=source_name,
                    channel_names=channel_names,
                    default=default_loss,
                    used_keys=used_loss_keys,
                )
            ).lower()
            weight = float(
                _resolve_reconstruction_override(
                    target.source_weights,
                    source_name=source_name,
                    channel_names=channel_names,
                    default=1.0,
                    used_keys=used_weight_keys,
                )
            )
            source_weight = _resolve_reconstruction_source_weight(
                target.source_weights,
                source_name=source_name,
                used_keys=used_weight_keys,
            )
            inner_weight = _resolve_reconstruction_inner_weight(
                target.source_weights,
                channel_names=channel_names,
                used_keys=used_weight_keys,
            )
            if is_hurdle_loss(loss):
                raise ValueError(f"Hurdle loss is unsupported for one-hot reconstruction source {source_name!r}")
            prediction_indices = tuple(range(prediction_offset, prediction_offset + len(indices)))
            prediction_offset += len(prediction_indices)
            slices.append(
                ReconstructionLossSlice(
                    name=source_name,
                    source=source_name,
                    channel_indices=indices,
                    channel_names=channel_names,
                    loss=loss,
                    weight=weight,
                    source_weight=source_weight,
                    inner_weight=inner_weight,
                    prediction_channel_indices=prediction_indices,
                )
            )
        else:
            for local_index, channel_name in enumerate(channel_names):
                slice_channel_names = (channel_name,)
                default_loss = _default_reconstruction_loss(
                    source_name=source_name,
                    preprocessing_source_spec=preprocessing_source_spec,
                    fallback_loss=target.loss,
                )
                loss = str(
                    _resolve_reconstruction_override(
                        target.source_losses,
                        source_name=source_name,
                        channel_names=slice_channel_names,
                        default=default_loss,
                        used_keys=used_loss_keys,
                    )
                ).lower()
                weight = float(
                    _resolve_reconstruction_override(
                        target.source_weights,
                        source_name=source_name,
                        channel_names=slice_channel_names,
                        default=1.0,
                        used_keys=used_weight_keys,
                    )
                )
                source_weight = _resolve_reconstruction_source_weight(
                    target.source_weights,
                    source_name=source_name,
                    used_keys=used_weight_keys,
                )
                inner_weight = _resolve_reconstruction_inner_weight(
                    target.source_weights,
                    channel_names=slice_channel_names,
                    used_keys=used_weight_keys,
                )
                prediction_indices = (prediction_offset,)
                prediction_offset += 1
                hurdle_logit_indices: tuple[int, ...] = ()
                if is_hurdle_loss(loss):
                    hurdle_logit_indices = (prediction_offset,)
                    prediction_offset += 1
                slices.append(
                    ReconstructionLossSlice(
                        name=channel_name,
                        source=source_name,
                        channel_indices=(offset + local_index,),
                        channel_names=slice_channel_names,
                        loss=loss,
                        weight=weight,
                        source_weight=source_weight,
                        inner_weight=inner_weight,
                        prediction_channel_indices=prediction_indices,
                        hurdle_logit_channel_indices=hurdle_logit_indices,
                        positive_thresholds=_hurdle_thresholds_for_loss(
                            source_name=source_name,
                            source_spec=preprocessing_source_spec,
                            loss=loss,
                            channels=1,
                        ),
                    )
                )
        offset += len(channel_names)

    unknown_loss_keys = sorted(set(target.source_losses or {}) - used_loss_keys)
    unknown_weight_keys = sorted(set(target.source_weights or {}) - used_weight_keys)
    if unknown_loss_keys:
        raise ValueError(
            f"input_reconstruction.source_losses contains keys not present in raster inputs: {unknown_loss_keys}"
        )
    if unknown_weight_keys:
        raise ValueError(
            f"input_reconstruction.source_weights contains keys not present in raster inputs: {unknown_weight_keys}"
        )

    return tuple(slices)


def _target_positive_thresholds(
    target: PipelineTargetSpec,
    preprocessing_spec: dict[str, Any],
    channels: int,
) -> tuple[float, ...]:
    if target.source is None:
        return ()
    source_spec = preprocessing_spec["sources"].get(target.source, {})
    return _hurdle_thresholds_for_loss(
        source_name=target.source,
        source_spec=source_spec,
        loss=target.loss,
        channels=channels,
    )


def build_eump_batch_schema(
    pipeline_spec: PipelineSpec,
    source_specs: dict[str, dict[str, Any]],
    preprocessing_spec: dict[str, Any],
) -> ProcessedBatchSchema:
    preprocessing_spec = build_active_preprocessing_spec(pipeline_spec, source_specs, preprocessing_spec)
    static_input_names: list[str] = []
    dynamic_input_names: list[str] = []
    scalar_input_names: list[str] = []
    static_input_items: list[tuple[str, tuple[str, ...]]] = []
    dynamic_input_items: list[tuple[str, tuple[str, ...]]] = []
    scalar_input_items: list[tuple[str, tuple[str, ...]]] = []

    for source_name in pipeline_spec.input_sources:
        if source_name == pipeline_spec.city_mask_source:
            continue
        source_schema = source_specs[source_name]
        names = source_output_names(preprocessing_spec, source_name)
        if source_schema["modality"] == "scalar":
            scalar_input_names.extend(names)
            scalar_input_items.append((source_name, names))
        elif source_schema["storage_level"] == "static":
            static_input_names.extend(names)
            static_input_items.append((source_name, names))
        else:
            dynamic_input_names.extend(names)
            dynamic_input_items.append((source_name, names))

    raster_input_items = static_input_items + dynamic_input_items
    raster_input_names = tuple(static_input_names + dynamic_input_names)
    if pipeline_spec.broadcast_scalar:
        raster_input_items = raster_input_items + scalar_input_items
        raster_input_names = raster_input_names + tuple(scalar_input_names)
    if pipeline_spec.include_city_mask_as_input:
        raster_input_items = raster_input_items + [(CITY_MASK_SOURCE, (CITY_MASK_SOURCE,))]
        raster_input_names = raster_input_names + (CITY_MASK_SOURCE,)

    raster_targets: list[TargetSchema] = []
    scalar_targets: list[TargetSchema] = []
    for target in pipeline_spec.targets:
        if target.target_type == "input_reconstruction":
            loss_slices = _build_reconstruction_loss_slices(
                target,
                raster_input_items,
                preprocessing_spec,
            )
            prediction_channels = sum(loss_slice.prediction_channels for loss_slice in loss_slices)
            raster_targets.append(
                TargetSchema(
                    target,
                    raster_input_names,
                    loss_slices,
                    prediction_channels_override=prediction_channels,
                )
            )
            continue
        if target.target_type == "raster":
            if target.source is None:
                raise ValueError(f"Raster target {target.name!r} requires a source")
            names = source_output_names(preprocessing_spec, target.source)
            raster_targets.append(
                TargetSchema(
                    target,
                    names,
                    positive_thresholds=_target_positive_thresholds(target, preprocessing_spec, len(names)),
                )
            )
            continue
        if target.target_type == "scalar":
            if target.source is None:
                raise ValueError(f"Scalar target {target.name!r} requires a source")
            scalar_targets.append(TargetSchema(target, source_output_names(preprocessing_spec, target.source)))
            continue
        if target.target_type == "crop_scalar":
            if target.source is None:
                raise ValueError(f"Crop scalar target {target.name!r} requires a source")
            names = source_output_names(preprocessing_spec, target.source)
            scalar_targets.append(TargetSchema(target, _select_channel_names(names, target)))

    return ProcessedBatchSchema(
        raster_input_channel_names=raster_input_names,
        scalar_input_names=tuple(scalar_input_names),
        raster_targets=tuple(raster_targets),
        scalar_targets=tuple(scalar_targets),
    )


def validate_eump_downstream_spec(pipeline_spec: PipelineSpec) -> PipelineTargetSpec:
    if pipeline_spec.kind != EUMP_DOWNSTREAM_KIND:
        raise ValueError(
            f"EUMP downstream requires pipeline.kind={EUMP_DOWNSTREAM_KIND!r}, "
            f"got {pipeline_spec.kind!r}"
        )
    if len(pipeline_spec.targets) != 1:
        raise ValueError(
            "eump_downstream v1 requires exactly one configured target; "
            f"got {len(pipeline_spec.targets)}"
        )

    target = pipeline_spec.targets[0]
    if target.target_type == "input_reconstruction":
        raise ValueError("eump_downstream does not support input_reconstruction targets")
    if target.optional:
        raise ValueError("eump_downstream targets cannot be optional")
    if target.target_type not in {"raster", "scalar", "crop_scalar"}:
        raise ValueError(
            "eump_downstream target must be raster, scalar, or crop_scalar; "
            f"got {target.target_type!r}"
        )
    if target.loss.lower() not in EUMP_DOWNSTREAM_LOSSES:
        raise ValueError(
            f"eump_downstream supports only regression losses "
            f"{sorted(EUMP_DOWNSTREAM_LOSSES)}; got {target.loss!r}"
        )
    return target


def validate_eump_scalar_mlp_downstream_spec(pipeline_spec: PipelineSpec) -> PipelineTargetSpec:
    if normalize_eump_pipeline_kind(pipeline_spec.kind) != EUMP_CITY_SCALAR_DOWNSTREAM_KIND:
        raise ValueError(
            "EUMP city-scalar downstream requires "
            f"pipeline.kind={EUMP_CITY_SCALAR_DOWNSTREAM_KIND!r} "
            f"(or legacy alias {EUMP_SCALAR_MLP_DOWNSTREAM_KIND!r}), "
            f"got {pipeline_spec.kind!r}"
        )
    if len(pipeline_spec.targets) != 1:
        raise ValueError(
            "eump_city_scalar_downstream requires exactly one configured target; "
            f"got {len(pipeline_spec.targets)}"
        )

    target = pipeline_spec.targets[0]
    if target.target_type not in {"scalar", "crop_scalar"}:
        raise ValueError(
            "eump_city_scalar_downstream target must be scalar or crop_scalar; "
            f"got {target.target_type!r}"
        )
    if target.optional:
        raise ValueError("eump_city_scalar_downstream targets cannot be optional")
    if target.loss.lower() not in EUMP_DOWNSTREAM_LOSSES:
        raise ValueError(
            "eump_city_scalar_downstream supports only regression losses "
            f"{sorted(EUMP_DOWNSTREAM_LOSSES)}; got {target.loss!r}"
        )
    if target.source is not None and target.source in pipeline_spec.input_sources:
        raise ValueError(
            "eump_city_scalar_downstream target source must not also be an input source "
            f"to avoid target leakage: {target.source!r}"
        )
    return target


def get_single_downstream_target_schema(batch_schema: ProcessedBatchSchema) -> TargetSchema:
    targets = tuple(batch_schema.raster_targets) + tuple(batch_schema.scalar_targets)
    if len(targets) != 1:
        raise ValueError(
            "eump_downstream batch schema must contain exactly one target; "
            f"got {len(targets)}"
        )
    return targets[0]


__all__ = [
    "DEFAULT_CONTINUOUS_LOSS",
    "DEFAULT_INPUT_RECONSTRUCTION_NAME",
    "EUMP_CITY_SCALAR_DOWNSTREAM_KIND",
    "EUMP_CONTRASTIVE_KIND",
    "EUMP_DOWNSTREAM_KIND",
    "EUMP_DOWNSTREAM_LOSSES",
    "EUMP_PIPELINE_KIND_ALIASES",
    "EUMP_RECONSTRUCTION_KIND",
    "EUMP_SCALAR_MLP_DOWNSTREAM_KIND",
    "HURDLE_LOSSES",
    "PipelineNormalizationConfig",
    "PipelineSpec",
    "PipelineLossConfig",
    "PipelineTargetSpec",
    "ProcessedBatchSchema",
    "ReconstructionLossSlice",
    "TargetSchema",
    "build_active_preprocessing_spec",
    "build_eump_batch_schema",
    "get_single_downstream_target_schema",
    "hurdle_normalized_source_names",
    "is_hurdle_loss",
    "normalize_eump_pipeline_kind",
    "source_output_names",
    "validate_eump_downstream_spec",
    "validate_eump_scalar_mlp_downstream_spec",
]
