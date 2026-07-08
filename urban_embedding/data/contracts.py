from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _ensure_array(name: str, value: np.ndarray) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray, got {type(value)!r}")
    return value


def _normalize_city_mask(city_mask: np.ndarray | None, spatial_shape: tuple[int, int]) -> np.ndarray | None:
    if city_mask is None:
        return None
    city_mask = _ensure_array("city_mask", city_mask)
    if city_mask.ndim == 2:
        city_mask = city_mask[None, ...]
    if city_mask.ndim != 3 or city_mask.shape[0] != 1:
        raise ValueError("city_mask must have shape [H, W] or [1, H, W]")
    if tuple(city_mask.shape[-2:]) != tuple(spatial_shape):
        raise ValueError(
            f"city_mask spatial shape {city_mask.shape[-2:]} does not match expected {spatial_shape}"
        )
    return city_mask.astype(np.float32, copy=False)


def _crop_hw(array: np.ndarray, crop_box: "CropBox") -> np.ndarray:
    if array.ndim == 2:
        return array[crop_box.top : crop_box.bottom, crop_box.left : crop_box.right]
    if array.ndim == 3:
        return array[:, crop_box.top : crop_box.bottom, crop_box.left : crop_box.right]
    raise ValueError(f"Expected a 2D or 3D array for cropping, got shape {array.shape}")


@dataclass(slots=True)
class RasterBlock:
    data: np.ndarray
    valid_mask: np.ndarray | None = None
    channel_names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        self.data = _ensure_array("data", self.data)
        if self.data.ndim != 3:
            raise ValueError(f"RasterBlock.data must have shape [C, H, W], got {self.data.shape}")

        if self.valid_mask is None:
            self.valid_mask = np.ones_like(self.data, dtype=bool)
        else:
            self.valid_mask = _ensure_array("valid_mask", self.valid_mask)
            if self.valid_mask.ndim == 2:
                self.valid_mask = np.broadcast_to(self.valid_mask[None, ...], self.data.shape)
            if self.valid_mask.shape != self.data.shape:
                raise ValueError(
                    f"RasterBlock.valid_mask must match data shape {self.data.shape}, got {self.valid_mask.shape}"
                )
            self.valid_mask = self.valid_mask.astype(bool, copy=False)

        if self.channel_names is not None and len(self.channel_names) != self.data.shape[0]:
            raise ValueError(
                f"channel_names length {len(self.channel_names)} does not match channel count {self.data.shape[0]}"
            )

    @property
    def spatial_shape(self) -> tuple[int, int]:
        return int(self.data.shape[-2]), int(self.data.shape[-1])

    @property
    def channels(self) -> int:
        return int(self.data.shape[0])

    def crop(self, crop_box: "CropBox") -> "RasterBlock":
        return RasterBlock(
            data=_crop_hw(self.data, crop_box),
            valid_mask=_crop_hw(self.valid_mask, crop_box),
            channel_names=self.channel_names,
        )


@dataclass(slots=True)
class ScalarBlock:
    data: np.ndarray
    valid_mask: np.ndarray | None = None
    names: tuple[str, ...] | None = None
    support: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.data = _ensure_array("data", self.data)
        if self.data.ndim != 1:
            raise ValueError(f"ScalarBlock.data must have shape [D], got {self.data.shape}")

        if self.valid_mask is None:
            self.valid_mask = np.ones_like(self.data, dtype=bool)
        else:
            self.valid_mask = _ensure_array("valid_mask", self.valid_mask)
            if self.valid_mask.shape != self.data.shape:
                raise ValueError(
                    f"ScalarBlock.valid_mask must match data shape {self.data.shape}, got {self.valid_mask.shape}"
                )
            self.valid_mask = self.valid_mask.astype(bool, copy=False)

        if self.support is not None:
            self.support = _ensure_array("support", self.support)
            if self.support.shape != self.data.shape:
                raise ValueError(
                    f"ScalarBlock.support must match data shape {self.data.shape}, got {self.support.shape}"
                )
            self.support = self.support.astype(np.float32, copy=False)

        if self.names is not None and len(self.names) != self.data.shape[0]:
            raise ValueError(f"names length {len(self.names)} does not match scalar count {self.data.shape[0]}")

    def broadcast_to_raster(self, spatial_shape: tuple[int, int]) -> RasterBlock:
        data = np.broadcast_to(self.data[:, None, None], (self.data.shape[0], *spatial_shape))
        valid_mask = np.broadcast_to(self.valid_mask[:, None, None], (self.data.shape[0], *spatial_shape))
        return RasterBlock(data=data, valid_mask=valid_mask, channel_names=self.names)


@dataclass(frozen=True, slots=True)
class CropScalarTargetSpec:
    source_kind: str
    source_name: str | None = None
    channel_indices: tuple[int, ...] | None = None
    reduction: str = "weighted_mean"
    value_transform: str = "identity"
    use_city_mask: bool = True
    min_valid_ratio: float = 0.0
    min_city_mask_coverage: float = 0.0
    output_names: tuple[str, ...] | None = None
    aggregation_scale: str = "processed"
    normalization_scope: str = "global"
    positive_only: bool = False
    optional: bool = False

    def __post_init__(self) -> None:
        valid_source_kinds = {
            "static_raster",
            "dynamic_raster",
            "reconstruct_raster",
            "aux_raster",
            "prediction_raster",
            "city_mask",
        }
        if self.source_kind not in valid_source_kinds:
            raise ValueError(f"Unsupported source_kind={self.source_kind!r}, expected one of {sorted(valid_source_kinds)}")

        valid_reductions = {"weighted_mean", "weighted_sum"}
        if self.reduction not in valid_reductions:
            raise ValueError(f"Unsupported reduction={self.reduction!r}, expected one of {sorted(valid_reductions)}")

        valid_value_transforms = {"identity", "positive_indicator"}
        if self.value_transform not in valid_value_transforms:
            raise ValueError(
                f"Unsupported value_transform={self.value_transform!r}, "
                f"expected one of {sorted(valid_value_transforms)}"
            )

        valid_aggregation_scales = {"processed", "physical"}
        if self.aggregation_scale not in valid_aggregation_scales:
            raise ValueError(
                f"Unsupported aggregation_scale={self.aggregation_scale!r}, "
                f"expected one of {sorted(valid_aggregation_scales)}"
            )

        valid_normalization_scopes = {"global", "city_crop", "zero_excluded"}
        if self.normalization_scope not in valid_normalization_scopes:
            raise ValueError(
                f"Unsupported normalization_scope={self.normalization_scope!r}, "
                f"expected one of {sorted(valid_normalization_scopes)}"
            )
        if self.normalization_scope in {"city_crop", "zero_excluded"} and self.aggregation_scale != "physical":
            raise ValueError(f"normalization_scope={self.normalization_scope!r} requires aggregation_scale='physical'")
        if self.positive_only and self.aggregation_scale != "physical":
            raise ValueError("positive_only=true requires aggregation_scale='physical'")
        if self.positive_only and self.value_transform != "identity":
            raise ValueError("positive_only=true cannot be combined with value_transform other than 'identity'")

        if self.source_kind in {"aux_raster", "prediction_raster"} and not self.source_name:
            raise ValueError(f"source_name is required when source_kind={self.source_kind!r}")
        if self.source_kind == "city_mask" and self.source_name not in {None, "city_mask"}:
            raise ValueError("source_name must be omitted or 'city_mask' when source_kind='city_mask'")

        if self.min_valid_ratio < 0 or self.min_valid_ratio > 1:
            raise ValueError(f"min_valid_ratio must be in [0, 1], got {self.min_valid_ratio}")
        if self.min_city_mask_coverage < 0 or self.min_city_mask_coverage > 1:
            raise ValueError(
                f"min_city_mask_coverage must be in [0, 1], got {self.min_city_mask_coverage}"
            )


@dataclass(slots=True)
class SceneInputs:
    static_raster: RasterBlock | None = None
    dynamic_raster: RasterBlock | None = None
    scalar: ScalarBlock | None = None
    city_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        raster_shapes = []
        if self.static_raster is not None:
            raster_shapes.append(self.static_raster.spatial_shape)
        if self.dynamic_raster is not None:
            raster_shapes.append(self.dynamic_raster.spatial_shape)
        if raster_shapes and len(set(raster_shapes)) != 1:
            raise ValueError(f"SceneInputs raster blocks must share one spatial shape, got {raster_shapes}")

        spatial_shape = self.spatial_shape
        if spatial_shape is None:
            raise ValueError("SceneInputs requires at least one raster block to define spatial_shape")
        self.city_mask = _normalize_city_mask(self.city_mask, spatial_shape)

    @property
    def spatial_shape(self) -> tuple[int, int] | None:
        if self.static_raster is not None:
            return self.static_raster.spatial_shape
        if self.dynamic_raster is not None:
            return self.dynamic_raster.spatial_shape
        if self.city_mask is not None:
            city_mask = _ensure_array("city_mask", self.city_mask)
            if city_mask.ndim in {2, 3}:
                return int(city_mask.shape[-2]), int(city_mask.shape[-1])
        return None

    def crop(self, crop_box: "CropBox") -> "SceneInputs":
        city_mask = None if self.city_mask is None else _crop_hw(self.city_mask, crop_box)
        return SceneInputs(
            static_raster=None if self.static_raster is None else self.static_raster.crop(crop_box),
            dynamic_raster=None if self.dynamic_raster is None else self.dynamic_raster.crop(crop_box),
            scalar=self.scalar,
            city_mask=city_mask,
        )


@dataclass(slots=True)
class SceneTargets:
    reconstruct_raster: RasterBlock | None = None
    aux_raster: dict[str, RasterBlock] = field(default_factory=dict)
    aux_scalar: dict[str, ScalarBlock] = field(default_factory=dict)
    prediction_raster: dict[str, RasterBlock] = field(default_factory=dict)
    prediction_scalar: dict[str, ScalarBlock] = field(default_factory=dict)
    crop_scalar_specs: dict[str, CropScalarTargetSpec] = field(default_factory=dict)

    def crop(self, crop_box: "CropBox") -> "SceneTargets":
        return SceneTargets(
            reconstruct_raster=None
            if self.reconstruct_raster is None
            else self.reconstruct_raster.crop(crop_box),
            aux_raster={name: block.crop(crop_box) for name, block in self.aux_raster.items()},
            aux_scalar=self.aux_scalar.copy(),
            prediction_raster={name: block.crop(crop_box) for name, block in self.prediction_raster.items()},
            prediction_scalar=self.prediction_scalar.copy(),
            crop_scalar_specs=self.crop_scalar_specs.copy(),
        )


@dataclass(frozen=True, slots=True)
class CropBox:
    top: int
    left: int
    height: int
    width: int

    @property
    def bottom(self) -> int:
        return self.top + self.height

    @property
    def right(self) -> int:
        return self.left + self.width


@dataclass(slots=True)
class SceneRecord:
    scene_id: str
    time_id: str | int | None
    inputs: SceneInputs
    targets: SceneTargets
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        spatial_shape = self.spatial_shape
        target_shapes = []
        if self.targets.reconstruct_raster is not None:
            target_shapes.append(("reconstruct_raster", self.targets.reconstruct_raster.spatial_shape))
        for name, block in self.targets.aux_raster.items():
            target_shapes.append((f"aux_raster.{name}", block.spatial_shape))
        for name, block in self.targets.prediction_raster.items():
            target_shapes.append((f"prediction_raster.{name}", block.spatial_shape))

        for target_name, target_shape in target_shapes:
            if tuple(target_shape) != tuple(spatial_shape):
                raise ValueError(
                    f"{target_name} spatial shape {target_shape} does not match input spatial shape {spatial_shape}"
                )

    @property
    def spatial_shape(self) -> tuple[int, int]:
        spatial_shape = self.inputs.spatial_shape
        if spatial_shape is None:
            raise ValueError("SceneRecord.inputs has no spatial_shape")
        return spatial_shape


@dataclass(slots=True)
class CropSample:
    scene_id: str
    time_id: str | int | None
    crop_box: CropBox
    inputs: SceneInputs
    targets: SceneTargets
    meta: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "CropBox",
    "CropSample",
    "CropScalarTargetSpec",
    "RasterBlock",
    "ScalarBlock",
    "SceneInputs",
    "SceneRecord",
    "SceneTargets",
]
