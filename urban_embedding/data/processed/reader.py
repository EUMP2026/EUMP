from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np

from urban_embedding.core.paths import EUMP_DATA_DIR
from urban_embedding.data.contracts import RasterBlock, ScalarBlock, SceneInputs, SceneRecord, SceneTargets

from .alignment import import_zarr
from .manifest import load_eump_store_schema, read_jsonl, scene_years_manifest_path, scenes_manifest_path
from .schema import CITY_MASK_SOURCE, build_runtime_derived_source_specs
from .task_config import ProcessedTaskConfig


def normalize_scene_id(scene_id: str | int) -> str:
    return str(scene_id)


def coerce_channel_names(source_name: str, attrs: dict[str, Any], width: int) -> tuple[str, ...]:
    channel_names = attrs.get("channel_names") or [source_name]
    if len(channel_names) != width:
        return tuple(f"{source_name}_{idx}" for idx in range(width))
    return tuple(str(name) for name in channel_names)


def concat_raster_blocks(blocks: list[tuple[np.ndarray, np.ndarray, tuple[str, ...]]]) -> RasterBlock | None:
    if not blocks:
        return None
    data = np.concatenate([block[0] for block in blocks], axis=0)
    valid_mask = np.concatenate([block[1] for block in blocks], axis=0)
    channel_names: list[str] = []
    for _, _, names in blocks:
        channel_names.extend(names)
    return RasterBlock(data=data, valid_mask=valid_mask, channel_names=tuple(channel_names))


def concat_scalar_blocks(blocks: list[tuple[np.ndarray, np.ndarray, tuple[str, ...]]]) -> ScalarBlock | None:
    if not blocks:
        return None
    data = np.concatenate([block[0] for block in blocks], axis=0)
    valid_mask = np.concatenate([block[1] for block in blocks], axis=0)
    names: list[str] = []
    for _, _, block_names in blocks:
        names.extend(block_names)
    return ScalarBlock(data=data, valid_mask=valid_mask, names=tuple(names))


def _spatial_shape_from_scene(
    scene_row: dict[str, Any],
    source_cache: dict[str, tuple[np.ndarray, np.ndarray, tuple[str, ...], dict[str, Any]]],
) -> tuple[int, int]:
    if "spatial_shape" in scene_row:
        spatial_shape = scene_row["spatial_shape"]
        if len(spatial_shape) == 2:
            return int(spatial_shape[0]), int(spatial_shape[1])
    for data, _, _, _ in source_cache.values():
        if np.asarray(data).ndim == 3:
            return int(data.shape[-2]), int(data.shape[-1])
    raise ValueError("Unable to infer spatial shape for missing optional raster input")


class ProcessedSceneStoreReader:
    def __init__(self, eump_root: str | Path = EUMP_DATA_DIR) -> None:
        self.eump_root = Path(eump_root)
        self.schema = load_eump_store_schema(self.eump_root)
        self.source_specs = self.schema["sources"]
        self.runtime_source_specs = {
            name: spec.to_dict()
            for name, spec in build_runtime_derived_source_specs().items()
        }
        self.available_source_specs = {
            **self.source_specs,
            **self.runtime_source_specs,
        }
        self.scene_rows = read_jsonl(scenes_manifest_path(self.eump_root))
        self.scene_year_rows = read_jsonl(scene_years_manifest_path(self.eump_root))
        self.scene_index = {normalize_scene_id(row["scene_id"]): row for row in self.scene_rows}
        self.scene_year_index = {
            (normalize_scene_id(row["scene_id"]), int(row["year"])): row
            for row in self.scene_year_rows
        }

    def validate_task_config(self, task_config: ProcessedTaskConfig) -> None:
        unknown_sources = [source_name for source_name in task_config.referenced_sources(include_city_mask=True) if source_name not in self.available_source_specs]
        if unknown_sources:
            unknown_list = ", ".join(sorted(unknown_sources))
            raise ValueError(f"Unknown source(s) in processed task config: {unknown_list}")

        derived_sources = [
            source_name
            for source_name in task_config.referenced_sources(include_city_mask=True)
            if source_name in self.runtime_source_specs
        ]
        if derived_sources:
            derived_list = ", ".join(sorted(derived_sources))
            raise ValueError(
                "Runtime-derived and crop-aggregated processed sources are not supported by the current role-based "
                f"task config. Unsupported source(s): {derived_list}"
            )

        reconstruct_target_source = task_config.reconstruct_target_source
        if reconstruct_target_source is None:
            return

        reconstruct_spec = self.source_specs[reconstruct_target_source]
        if reconstruct_spec["modality"] != "raster":
            raise ValueError(
                f"reconstruct_target_source={reconstruct_target_source!r} must be a raster source, "
                f"got modality={reconstruct_spec['modality']!r}"
            )

    def missing_sources_for_task(
        self,
        scene_id: str | int,
        year: int,
        task_config: ProcessedTaskConfig,
    ) -> list[str]:
        self.validate_task_config(task_config)
        scene_key = normalize_scene_id(scene_id)
        scene_row = self.scene_index[scene_key]
        scene_year_row = self.scene_year_index[(scene_key, int(year))]
        static_sources = set(scene_row["present_sources"])
        yearly_sources = set(scene_year_row["present_sources"])

        missing: list[str] = []
        for source_name in task_config.referenced_sources(include_city_mask=True, include_optional=False):
            spec = self.source_specs[source_name]
            available_sources = static_sources if spec["storage_level"] == "static" else yearly_sources
            if source_name not in available_sources:
                missing.append(source_name)
        return missing

    def _source_is_present(
        self,
        source_name: str,
        scene_row: dict[str, Any],
        scene_year_row: dict[str, Any],
    ) -> bool:
        spec = self.source_specs[source_name]
        available_sources = (
            set(scene_row["present_sources"])
            if spec["storage_level"] == "static"
            else set(scene_year_row["present_sources"])
        )
        return source_name in available_sources

    def find_compatible_scene_year(
        self,
        task_config: ProcessedTaskConfig,
        scene_ids: Iterable[str | int] | None = None,
        year_range: tuple[int, int] | None = None,
    ) -> dict[str, Any] | None:
        for row in self.iter_scene_years(scene_ids=scene_ids, year_range=year_range):
            if not self.missing_sources_for_task(row["scene_id"], int(row["year"]), task_config):
                return row
        return None

    def iter_scene_years(
        self,
        scene_ids: Iterable[str | int] | None = None,
        year_range: tuple[int, int] | None = None,
    ) -> list[dict[str, Any]]:
        allowed_scene_ids = None if scene_ids is None else {normalize_scene_id(scene_id) for scene_id in scene_ids}
        rows: list[dict[str, Any]] = []
        for row in self.scene_year_rows:
            scene_id = normalize_scene_id(row["scene_id"])
            year = int(row["year"])
            if allowed_scene_ids is not None and scene_id not in allowed_scene_ids:
                continue
            if year_range is not None and not (year_range[0] <= year <= year_range[1]):
                continue
            rows.append(row)
        return rows

    def _open_group(self, relative_path: str):
        zarr = import_zarr()
        return zarr.open_group(str(self.eump_root / relative_path), mode="r")

    def _load_source_arrays(
        self,
        static_group,
        year_group,
        source_name: str,
    ) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], dict[str, Any]]:
        spec = self.source_specs[source_name]
        source_group = static_group[source_name] if spec["storage_level"] == "static" else year_group[source_name]
        data = np.asarray(source_group["data"])
        valid_mask = np.asarray(source_group["valid_mask"]).astype(bool, copy=False)
        attrs = dict(source_group.attrs)
        channel_names = coerce_channel_names(source_name, attrs, int(data.shape[0]))
        return data, valid_mask, channel_names, attrs

    def _load_required_sources(
        self,
        static_group,
        year_group,
        task_config: ProcessedTaskConfig,
        scene_row: dict[str, Any],
        scene_year_row: dict[str, Any],
    ) -> dict[str, tuple[np.ndarray, np.ndarray, tuple[str, ...], dict[str, Any]]]:
        source_cache: dict[str, tuple[np.ndarray, np.ndarray, tuple[str, ...], dict[str, Any]]] = {}
        optional_sources = set(task_config.optional_input_sources) | set(task_config.optional_aux_target_sources)
        for source_name in task_config.referenced_sources(include_city_mask=True):
            if source_name in optional_sources and not self._source_is_present(source_name, scene_row, scene_year_row):
                continue
            source_cache[source_name] = self._load_source_arrays(static_group, year_group, source_name)
        return source_cache

    def _missing_optional_input_arrays(
        self,
        source_name: str,
        scene_row: dict[str, Any],
        source_cache: dict[str, tuple[np.ndarray, np.ndarray, tuple[str, ...], dict[str, Any]]],
    ) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
        spec = self.source_specs[source_name]
        if spec["modality"] == "raster":
            height, width = _spatial_shape_from_scene(scene_row, source_cache)
            data = np.zeros((1, height, width), dtype=np.float32)
            valid_mask = np.zeros_like(data, dtype=bool)
        else:
            data = np.zeros((1,), dtype=np.float32)
            valid_mask = np.zeros_like(data, dtype=bool)
        return data, valid_mask, (source_name,)

    def _materialize_source_block(
        self,
        source_name: str,
        source_cache: dict[str, tuple[np.ndarray, np.ndarray, tuple[str, ...], dict[str, Any]]],
    ) -> RasterBlock | ScalarBlock:
        data, valid_mask, channel_names, _ = source_cache[source_name]
        spec = self.source_specs[source_name]
        if spec["modality"] == "raster":
            return RasterBlock(
                data=data.astype(np.float32, copy=False),
                valid_mask=valid_mask,
                channel_names=channel_names,
            )
        return ScalarBlock(
            data=data.astype(np.float32, copy=False),
            valid_mask=valid_mask,
            names=channel_names,
        )

    def _build_input_blocks(
        self,
        task_config: ProcessedTaskConfig,
        source_cache: dict[str, tuple[np.ndarray, np.ndarray, tuple[str, ...], dict[str, Any]]],
        scene_row: dict[str, Any],
    ) -> tuple[RasterBlock | None, RasterBlock | None, ScalarBlock | None]:
        static_rasters: list[tuple[np.ndarray, np.ndarray, tuple[str, ...]]] = []
        dynamic_rasters: list[tuple[np.ndarray, np.ndarray, tuple[str, ...]]] = []
        scalars: list[tuple[np.ndarray, np.ndarray, tuple[str, ...]]] = []
        optional_inputs = set(task_config.optional_input_sources)

        for source_name in task_config.input_sources:
            if source_name == task_config.city_mask_source:
                continue
            spec = self.source_specs[source_name]
            if source_name in source_cache:
                data, valid_mask, channel_names, _ = source_cache[source_name]
            elif source_name in optional_inputs:
                data, valid_mask, channel_names = self._missing_optional_input_arrays(
                    source_name,
                    scene_row,
                    source_cache,
                )
            else:
                raise KeyError(f"Required input source {source_name!r} is missing from source cache")
            if spec["modality"] == "raster":
                bucket = static_rasters if spec["storage_level"] == "static" else dynamic_rasters
                bucket.append((data.astype(np.float32, copy=False), valid_mask, channel_names))
            else:
                scalars.append((data.astype(np.float32, copy=False), valid_mask, channel_names))

        return concat_raster_blocks(static_rasters), concat_raster_blocks(dynamic_rasters), concat_scalar_blocks(scalars)

    def _load_city_mask(
        self,
        source_name: str | None,
        source_cache: dict[str, tuple[np.ndarray, np.ndarray, tuple[str, ...], dict[str, Any]]],
    ) -> np.ndarray | None:
        if source_name is None:
            return None
        data = source_cache[source_name][0].astype(np.float32, copy=False)
        if data.ndim != 3 or data.shape[0] != 1:
            raise ValueError(f"{source_name!r} must have shape [1, H, W], got {data.shape}")
        return data

    def _build_targets(
        self,
        task_config: ProcessedTaskConfig,
        source_cache: dict[str, tuple[np.ndarray, np.ndarray, tuple[str, ...], dict[str, Any]]],
    ) -> SceneTargets:
        reconstruct_raster = None
        aux_raster: dict[str, RasterBlock] = {}
        aux_scalar: dict[str, ScalarBlock] = {}
        prediction_raster: dict[str, RasterBlock] = {}
        prediction_scalar: dict[str, ScalarBlock] = {}

        if task_config.reconstruct_target_source is not None:
            block = self._materialize_source_block(task_config.reconstruct_target_source, source_cache)
            if not isinstance(block, RasterBlock):
                raise ValueError(
                    f"reconstruct_target_source={task_config.reconstruct_target_source!r} must resolve to a RasterBlock"
                )
            reconstruct_raster = block

        for source_name in tuple(task_config.aux_target_sources) + tuple(task_config.optional_aux_target_sources):
            if source_name not in source_cache:
                continue
            block = self._materialize_source_block(source_name, source_cache)
            if isinstance(block, RasterBlock):
                aux_raster[source_name] = block
            else:
                aux_scalar[source_name] = block

        for source_name in task_config.prediction_sources:
            block = self._materialize_source_block(source_name, source_cache)
            if isinstance(block, RasterBlock):
                prediction_raster[source_name] = block
            else:
                prediction_scalar[source_name] = block

        return SceneTargets(
            reconstruct_raster=reconstruct_raster,
            aux_raster=aux_raster,
            aux_scalar=aux_scalar,
            prediction_raster=prediction_raster,
            prediction_scalar=prediction_scalar,
        )

    def load_scene_record(
        self,
        scene_id: str | int,
        year: int,
        task_config: ProcessedTaskConfig,
    ) -> SceneRecord:
        self.validate_task_config(task_config)
        scene_key = normalize_scene_id(scene_id)
        scene_row = self.scene_index[scene_key]
        scene_year_row = self.scene_year_index[(scene_key, int(year))]
        missing_sources = self.missing_sources_for_task(scene_key, int(year), task_config)
        if missing_sources:
            missing_str = ", ".join(sorted(missing_sources))
            raise ValueError(
                f"Scene {scene_row['scene_id']} year {int(year)} is unavailable for the requested task config. "
                f"Missing sources: {missing_str}"
            )
        static_group = self._open_group(scene_row["static_path"])
        year_group = self._open_group(scene_year_row["year_path"])
        source_cache = self._load_required_sources(
            static_group,
            year_group,
            task_config,
            scene_row,
            scene_year_row,
        )

        static_raster, dynamic_raster, scalar_block = self._build_input_blocks(task_config, source_cache, scene_row)
        city_mask = self._load_city_mask(task_config.city_mask_source, source_cache)
        targets = self._build_targets(task_config, source_cache)

        return SceneRecord(
            scene_id=scene_key,
            time_id=int(year),
            inputs=SceneInputs(
                static_raster=static_raster,
                dynamic_raster=dynamic_raster,
                scalar=scalar_block,
                city_mask=city_mask,
            ),
            targets=targets,
            meta={
                "scene_id": scene_row["scene_id"],
                "sample_id": scene_row.get("sample_id"),
                "year": int(year),
                "present_sources": list(scene_year_row["present_sources"]),
                "missing_sources": list(scene_year_row["missing_sources"]),
                "role_sources": {
                    "input_sources": list(task_config.input_sources),
                    "optional_input_sources": list(task_config.optional_input_sources),
                    "reconstruct_target_source": task_config.reconstruct_target_source,
                    "aux_target_sources": list(task_config.aux_target_sources),
                    "optional_aux_target_sources": list(task_config.optional_aux_target_sources),
                    "prediction_sources": list(task_config.prediction_sources),
                    "city_mask_source": task_config.city_mask_source,
                },
                "optional_missing_sources": [
                    source_name
                    for source_name in task_config.optional_aux_target_sources
                    if source_name not in source_cache
                ],
                "optional_missing_input_sources": [
                    source_name
                    for source_name in task_config.optional_input_sources
                    if source_name not in source_cache
                ],
            },
        )


__all__ = [
    "ProcessedSceneStoreReader",
    "coerce_channel_names",
    "concat_raster_blocks",
    "concat_scalar_blocks",
    "normalize_scene_id",
]
