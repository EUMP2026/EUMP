from __future__ import annotations

import numpy as np

from .contracts import (
    CropBox,
    CropSample,
    CropScalarTargetSpec,
    RasterBlock,
    ScalarBlock,
    SceneInputs,
    SceneRecord,
    SceneTargets,
    _crop_hw,
)


def sample_random_crop_box(
    spatial_shape: tuple[int, int],
    crop_size: int | tuple[int, int],
    center_ratio: float = 0.0,
    rng: np.random.Generator | None = None,
) -> CropBox:
    if rng is None:
        rng = np.random.default_rng()

    if isinstance(crop_size, int):
        crop_h = crop_w = crop_size
    else:
        crop_h, crop_w = crop_size

    height, width = spatial_shape
    if crop_h > height or crop_w > width:
        raise ValueError(f"Crop size {(crop_h, crop_w)} exceeds spatial_shape {spatial_shape}")

    use_center = rng.random() < center_ratio

    if height == crop_h:
        top = 0
    elif use_center:
        top = (height - crop_h) // 2
    else:
        top = int(rng.integers(0, height - crop_h + 1))

    if width == crop_w:
        left = 0
    elif use_center:
        left = (width - crop_w) // 2
    else:
        left = int(rng.integers(0, width - crop_w + 1))

    return CropBox(top=top, left=left, height=crop_h, width=crop_w)


def materialize_crop_sample(
    scene: SceneRecord,
    crop_box: CropBox,
    *,
    preprocessing_spec: dict | None = None,
) -> CropSample:
    height, width = scene.spatial_shape
    if crop_box.bottom > height or crop_box.right > width:
        raise ValueError(
            f"crop_box {crop_box} exceeds scene spatial_shape {scene.spatial_shape}"
        )

    crop_meta = dict(scene.meta)
    crop_meta["crop_box"] = (crop_box.top, crop_box.left, crop_box.bottom, crop_box.right)
    crop_inputs = scene.inputs.crop(crop_box)
    crop_targets = scene.targets.crop(crop_box)
    derived_scalar_targets = materialize_crop_scalar_targets(
        crop_inputs=crop_inputs,
        crop_targets=crop_targets,
        specs=crop_targets.crop_scalar_specs,
        preprocessing_spec=preprocessing_spec,
    )
    if derived_scalar_targets:
        crop_targets.aux_scalar.update(derived_scalar_targets)
        crop_targets.crop_scalar_specs = {}

    return CropSample(
        scene_id=scene.scene_id,
        time_id=scene.time_id,
        crop_box=crop_box,
        inputs=crop_inputs,
        targets=crop_targets,
        meta=crop_meta,
    )


def _select_raster_source(
    crop_inputs: SceneInputs,
    crop_targets: SceneTargets,
    spec: CropScalarTargetSpec,
) -> RasterBlock:
    if spec.source_kind == "city_mask":
        if crop_inputs.city_mask is None:
            raise ValueError("Crop scalar source 'city_mask' is unavailable")
        return RasterBlock(
            data=crop_inputs.city_mask.astype(np.float32, copy=False),
            valid_mask=np.ones_like(crop_inputs.city_mask, dtype=bool),
            channel_names=("city_mask",),
        )

    if spec.source_kind == "static_raster":
        if crop_inputs.static_raster is None:
            raise ValueError("Crop scalar source 'static_raster' is unavailable")
        return crop_inputs.static_raster

    if spec.source_kind == "dynamic_raster":
        if crop_inputs.dynamic_raster is None:
            raise ValueError("Crop scalar source 'dynamic_raster' is unavailable")
        return crop_inputs.dynamic_raster

    if spec.source_kind == "reconstruct_raster":
        if crop_targets.reconstruct_raster is None:
            raise ValueError("Crop scalar source 'reconstruct_raster' is unavailable")
        return crop_targets.reconstruct_raster

    if spec.source_kind == "aux_raster":
        if spec.source_name not in crop_targets.aux_raster:
            raise ValueError(f"Crop scalar aux_raster source {spec.source_name!r} is unavailable")
        return crop_targets.aux_raster[spec.source_name]

    if spec.source_kind == "prediction_raster":
        if spec.source_name not in crop_targets.prediction_raster:
            raise ValueError(f"Crop scalar prediction_raster source {spec.source_name!r} is unavailable")
        return crop_targets.prediction_raster[spec.source_name]

    raise ValueError(f"Unsupported source_kind={spec.source_kind!r}")


def _source_spec(preprocessing_spec: dict | None, source_name: str | None) -> dict | None:
    if preprocessing_spec is None or source_name is None:
        return None
    return preprocessing_spec.get("sources", {}).get(source_name)


def _select_channel_indices_by_names(raster: RasterBlock, channel_names: tuple[str, ...]) -> tuple[int, ...] | None:
    if raster.channel_names is None:
        return None
    index = {str(name): idx for idx, name in enumerate(raster.channel_names)}
    if all(channel_name in index for channel_name in channel_names):
        return tuple(index[channel_name] for channel_name in channel_names)
    return None


def _select_channels(raster: RasterBlock, channel_indices: tuple[int, ...]) -> RasterBlock:
    data = raster.data[list(channel_indices), ...]
    valid_mask = raster.valid_mask[list(channel_indices), ...]
    channel_names = None
    if raster.channel_names is not None:
        channel_names = tuple(raster.channel_names[idx] for idx in channel_indices)
    return RasterBlock(data=data, valid_mask=valid_mask, channel_names=channel_names)


def _maybe_select_channels(
    raster: RasterBlock,
    spec: CropScalarTargetSpec,
    *,
    preprocessing_spec: dict | None = None,
) -> RasterBlock:
    selected = raster
    source_spec = _source_spec(preprocessing_spec, spec.source_name)
    if spec.source_name is not None:
        source_channel_names: tuple[str, ...] | None = None
        if source_spec is not None:
            source_channel_names = tuple(str(name) for name in source_spec["output_channel_names"])
        elif raster.channel_names is not None and spec.source_name in raster.channel_names:
            source_channel_names = (spec.source_name,)

        if source_channel_names is not None:
            source_indices = _select_channel_indices_by_names(selected, source_channel_names)
            if source_indices is None:
                raise ValueError(
                    f"Crop scalar source {spec.source_name!r} cannot be selected from raster channels "
                    f"{selected.channel_names!r}"
                )
            selected = _select_channels(selected, source_indices)

    if spec.channel_indices is not None:
        channel_indices = tuple(int(idx) for idx in spec.channel_indices)
        selected = _select_channels(selected, channel_indices)

    return selected


def _aggregate_selected_block(
    selected: RasterBlock,
    city_mask: np.ndarray | None,
    spec: CropScalarTargetSpec,
) -> ScalarBlock:
    channels, height, width = selected.data.shape
    use_city_mask = bool(spec.use_city_mask) and spec.source_kind != "city_mask"
    city_mask_coverage = np.float32(1.0)

    base_weight = np.ones((channels, height, width), dtype=np.float32)
    if use_city_mask:
        if city_mask is None:
            raise ValueError("city_mask is required for crop scalar aggregation when use_city_mask=True")
        city_mask_coverage = np.float32(np.mean(city_mask > 0))
        base_weight = np.broadcast_to(city_mask.astype(np.float32, copy=False), (channels, height, width))

    valid_weight = base_weight * selected.valid_mask.astype(np.float32, copy=False)
    weight_sum = valid_weight.reshape(channels, -1).sum(axis=1)
    base_sum = base_weight.reshape(channels, -1).sum(axis=1)
    support = np.divide(
        weight_sum,
        np.maximum(base_sum, 1e-8),
        out=np.zeros_like(weight_sum, dtype=np.float32),
        where=base_sum > 0,
    ).astype(np.float32, copy=False)

    weighted_sum = (selected.data * valid_weight).reshape(channels, -1).sum(axis=1)
    value = weighted_sum.astype(np.float32, copy=True)
    positive_weight = weight_sum > 0
    if spec.reduction == "weighted_mean":
        value[positive_weight] = value[positive_weight] / weight_sum[positive_weight]

    min_valid_mask = support > 0 if spec.min_valid_ratio <= 0 else support >= spec.min_valid_ratio
    valid_mask = positive_weight & min_valid_mask
    if use_city_mask and spec.min_city_mask_coverage > 0:
        valid_mask &= city_mask_coverage >= spec.min_city_mask_coverage

    names = spec.output_names
    if names is None:
        names = selected.channel_names

    return ScalarBlock(
        data=value,
        valid_mask=valid_mask,
        names=names,
        support=support,
    )


def _restore_processed_raster_block(block: RasterBlock, source_spec: dict) -> RasterBlock:
    from urban_embedding.data.processed.processing import restore_processed_raster_block

    return restore_processed_raster_block(block, source_spec)


def _process_physical_scalar_block(block: ScalarBlock, source_spec: dict, *, stats_key: str = "stats") -> ScalarBlock:
    from urban_embedding.data.processed.processing import process_physical_scalar_block

    return process_physical_scalar_block(block, source_spec, stats_key=stats_key)


def _rename_scalar_block(block: ScalarBlock, names: tuple[str, ...] | None) -> ScalarBlock:
    return ScalarBlock(
        data=block.data.astype(np.float32, copy=False),
        valid_mask=block.valid_mask.astype(bool, copy=False),
        names=names,
        support=None if block.support is None else block.support.astype(np.float32, copy=False),
    )


def aggregate_crop_scalar_target(
    raster: RasterBlock,
    city_mask: np.ndarray | None,
    spec: CropScalarTargetSpec,
    *,
    preprocessing_spec: dict | None = None,
) -> ScalarBlock:
    selected = _maybe_select_channels(raster, spec, preprocessing_spec=preprocessing_spec)
    output_names = spec.output_names if spec.output_names is not None else selected.channel_names
    source_spec = _source_spec(preprocessing_spec, spec.source_name)

    if spec.aggregation_scale == "physical":
        if source_spec is None:
            raise ValueError(
                f"Crop scalar target from source {spec.source_name!r} requests aggregation_scale='physical', "
                "but no preprocessing spec for that source is available"
            )
        if source_spec["mode"] == "one_hot":
            return _rename_scalar_block(
                _aggregate_selected_block(selected, city_mask, spec),
                output_names,
            )
        physical_block = _restore_processed_raster_block(selected, source_spec)
        if spec.value_transform == "positive_indicator":
            indicator_data = (physical_block.data > 0).astype(np.float32, copy=False)
            physical_block = RasterBlock(
                data=indicator_data,
                valid_mask=physical_block.valid_mask,
                channel_names=physical_block.channel_names,
            )
            return _rename_scalar_block(
                _aggregate_selected_block(physical_block, city_mask, spec),
                output_names,
            )
        if spec.positive_only:
            physical_block = RasterBlock(
                data=physical_block.data,
                valid_mask=physical_block.valid_mask & (physical_block.data > 0),
                channel_names=physical_block.channel_names,
            )
        physical_scalar = _aggregate_selected_block(physical_block, city_mask, spec)
        stats_key_by_scope = {
            "global": "stats",
            "city_crop": "city_crop_stats",
            "zero_excluded": "zero_excluded_stats",
        }
        stats_key = stats_key_by_scope[spec.normalization_scope]
        processed_scalar = _process_physical_scalar_block(physical_scalar, source_spec, stats_key=stats_key)
        return _rename_scalar_block(processed_scalar, output_names)

    if spec.value_transform != "identity":
        raise ValueError("value_transform other than 'identity' requires aggregation_scale='physical'")
    return _rename_scalar_block(
        _aggregate_selected_block(selected, city_mask, spec),
        output_names,
    )


def materialize_crop_scalar_targets(
    crop_inputs: SceneInputs,
    crop_targets: SceneTargets,
    specs: dict[str, CropScalarTargetSpec],
    *,
    preprocessing_spec: dict | None = None,
) -> dict[str, ScalarBlock]:
    derived_targets: dict[str, ScalarBlock] = {}
    for target_name, spec in specs.items():
        try:
            raster_source = _select_raster_source(crop_inputs, crop_targets, spec)
        except ValueError:
            if spec.optional:
                continue
            raise
        scalar_block = aggregate_crop_scalar_target(
            raster=raster_source,
            city_mask=crop_inputs.city_mask,
            spec=spec,
            preprocessing_spec=preprocessing_spec,
        )
        if scalar_block.names is None:
            if scalar_block.data.shape[0] == 1:
                scalar_block.names = (target_name,)
            else:
                scalar_block.names = tuple(f"{target_name}_{idx}" for idx in range(scalar_block.data.shape[0]))
        derived_targets[target_name] = scalar_block
    return derived_targets


def stack_model_raster_inputs(
    scene_inputs: SceneInputs,
    include_city_mask: bool = False,
    broadcast_scalar: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    raster_blocks: list[RasterBlock] = []

    if scene_inputs.static_raster is not None:
        raster_blocks.append(scene_inputs.static_raster)
    if scene_inputs.dynamic_raster is not None:
        raster_blocks.append(scene_inputs.dynamic_raster)

    if broadcast_scalar and scene_inputs.scalar is not None:
        spatial_shape = scene_inputs.spatial_shape
        if spatial_shape is None:
            raise ValueError("Cannot broadcast scalar inputs without a raster spatial_shape")
        raster_blocks.append(scene_inputs.scalar.broadcast_to_raster(spatial_shape))

    if include_city_mask and scene_inputs.city_mask is not None:
        raster_blocks.append(
            RasterBlock(
                data=scene_inputs.city_mask.astype(np.float32, copy=False),
                valid_mask=np.ones_like(scene_inputs.city_mask, dtype=bool),
                channel_names=("city_mask",),
            )
        )

    if not raster_blocks:
        raise ValueError("No raster inputs are available to stack")

    raster = np.concatenate([block.data for block in raster_blocks], axis=0)
    valid_mask = np.concatenate([block.valid_mask for block in raster_blocks], axis=0)
    return raster, valid_mask


__all__ = [
    "aggregate_crop_scalar_target",
    "materialize_crop_sample",
    "materialize_crop_scalar_targets",
    "sample_random_crop_box",
    "stack_model_raster_inputs",
]
