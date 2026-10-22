from __future__ import annotations

from dataclasses import asdict, dataclass, field
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from urban_embedding.core.reproducibility import seeded_torch_generator
from urban_embedding.data.processed.dataset import build_eump_dataloaders
from urban_embedding.data.processed.pipeline_spec import (
    PipelineSpec,
    ProcessedBatchSchema,
    TargetSchema,
    build_active_preprocessing_spec,
    build_eump_batch_schema,
    get_single_downstream_target_schema,
    validate_eump_scalar_mlp_downstream_spec,
)
from urban_embedding.data.processed.processing import (
    load_preprocessing_spec,
    validate_preprocessing_spec,
    validate_preprocessing_stats_scope,
)
from urban_embedding.data.processed.reader import ProcessedSceneStoreReader
from urban_embedding.data.processed.split_manifest import (
    load_split_manifest,
    split_label_lookup_from_manifest,
)
from urban_embedding.training.distributed import distributed_barrier as torch_distributed_barrier

from .features import (
    CityLevelFeatureLayout,
    StandardizationStats,
    build_city_level_feature_layout,
    build_city_level_features,
    build_city_level_target,
)


SPLIT_NAME_TO_ID = {"train": 0, "val": 1, "test": 2}
SPLIT_ID_TO_NAME = {value: key for key, value in SPLIT_NAME_TO_ID.items()}
CACHE_VERSION = 9
PROCESSED_CITY_SCALAR_FEATURE_KIND = "processed_city_scalar"
AEF_CSV_FEATURE_KIND = "aef_csv"
DEFAULT_AEF_FEATURE_COLUMNS = tuple(f"A{index:02d}" for index in range(64))
DEFAULT_AEF_YEAR_RANGE = (2017, 2020)
FEATURE_STANDARDIZATION_TRAIN_SPLIT_ZSCORE = "train_split_zscore"
FEATURE_STANDARDIZATION_FEATURE_CACHE_TRAIN_SPLIT = "feature_cache_train_split"
FEATURE_STANDARDIZATION_NONE = "none"
FEATURE_STANDARDIZATION_MODES = {
    FEATURE_STANDARDIZATION_FEATURE_CACHE_TRAIN_SPLIT,
    FEATURE_STANDARDIZATION_TRAIN_SPLIT_ZSCORE,
    FEATURE_STANDARDIZATION_NONE,
}


def normalize_city_scalar_feature_standardization(value: Any = None) -> str:
    mode = FEATURE_STANDARDIZATION_TRAIN_SPLIT_ZSCORE if value is None else str(value).lower()
    if mode not in FEATURE_STANDARDIZATION_MODES:
        supported = ", ".join(sorted(FEATURE_STANDARDIZATION_MODES))
        raise ValueError(f"downstream.feature_standardization must be one of {supported}, got {mode!r}")
    return mode


@dataclass(frozen=True, slots=True)
class CityScalarFeatureSourceOptions:
    kind: str = PROCESSED_CITY_SCALAR_FEATURE_KIND
    root: Path | None = None
    file_pattern: str = "AEF_{year}.csv"
    year_range: tuple[int, int] | None = None
    id_column: str = "ID"
    feature_columns: tuple[str, ...] = DEFAULT_AEF_FEATURE_COLUMNS


@dataclass(frozen=True, slots=True)
class CityScalarCacheOptions:
    enabled: bool = True
    rebuild: bool = False
    rebuild_features: bool | None = None
    rebuild_targets: bool | None = None
    cache_dir: Path | None = None
    num_workers: int = 4
    batch_size: int = 1024
    feature_source: CityScalarFeatureSourceOptions = field(default_factory=CityScalarFeatureSourceOptions)


@dataclass(frozen=True, slots=True)
class CityScalarFeatureCache:
    path: Path
    fingerprint: str
    features: np.ndarray
    feature_valid_mask: np.ndarray
    row_keys: np.ndarray
    scene_ids: np.ndarray
    sample_ids: np.ndarray
    years: np.ndarray
    split_ids: np.ndarray
    feature_names: tuple[str, ...]
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CityScalarTargetCache:
    path: Path
    fingerprint: str
    targets: np.ndarray
    target_valid_mask: np.ndarray
    row_keys: np.ndarray
    scene_ids: np.ndarray
    sample_ids: np.ndarray
    years: np.ndarray
    split_ids: np.ndarray
    target_name: str
    target_channel_names: tuple[str, ...]
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CityScalarJoinedArrays:
    features: np.ndarray
    targets: np.ndarray
    split_ids: np.ndarray
    row_keys: np.ndarray
    scene_ids: np.ndarray
    sample_ids: np.ndarray
    years: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CachedCityScalarBatch:
    features: torch.Tensor
    target: torch.Tensor

    def to(self, device: torch.device | str, non_blocking: bool = True) -> "CachedCityScalarBatch":
        return CachedCityScalarBatch(
            features=self.features.to(device, non_blocking=non_blocking),
            target=self.target.to(device, non_blocking=non_blocking),
        )


@dataclass(frozen=True, slots=True)
class CityScalarPreparedData:
    train_dl: DataLoader
    val_dl: DataLoader
    test_dl: DataLoader
    batch_schema: ProcessedBatchSchema
    target_schema: TargetSchema
    feature_layout: CityLevelFeatureLayout
    standardization: StandardizationStats
    cache_metadata: dict[str, Any]
    preprocessing_spec: dict[str, Any]


class CityScalarTensorDataset(Dataset):
    def __init__(self, *, features: np.ndarray, targets: np.ndarray) -> None:
        if features.ndim != 2:
            raise ValueError(f"features must have shape [N, F], got {features.shape}")
        if targets.ndim != 2:
            raise ValueError(f"targets must have shape [N, T], got {targets.shape}")
        if features.shape[0] != targets.shape[0]:
            raise ValueError(
                f"features and targets row counts differ: {features.shape[0]} != {targets.shape[0]}"
            )
        self.features = torch.as_tensor(features.astype(np.float32, copy=False))
        self.targets = torch.as_tensor(targets.astype(np.float32, copy=False))

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.targets[index]


def collate_cached_city_scalar_samples(samples: list[tuple[torch.Tensor, torch.Tensor]]) -> CachedCityScalarBatch:
    features, targets = zip(*samples)
    return CachedCityScalarBatch(features=torch.stack(features, dim=0), target=torch.stack(targets, dim=0))


def _normalize_year_range(value: Any, *, default: tuple[int, int] | None = None) -> tuple[int, int] | None:
    if value is None:
        return default
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"year_range must be a two-item sequence, got {value!r}")
    start, end = (int(value[0]), int(value[1]))
    if start > end:
        raise ValueError(f"year_range start must be <= end, got {(start, end)!r}")
    return start, end


def _normalize_feature_columns(value: Any) -> tuple[str, ...]:
    if value in (None, ""):
        return DEFAULT_AEF_FEATURE_COLUMNS
    columns = tuple(str(column) for column in value)
    if not columns:
        raise ValueError("pipeline.city_scalar_cache.feature_source.feature_columns must not be empty")
    return columns


def normalize_city_scalar_feature_source_options(payload: Any) -> CityScalarFeatureSourceOptions:
    if payload is None:
        payload = {}
    if isinstance(payload, str):
        payload = {"kind": payload}
    if not isinstance(payload, dict):
        raise TypeError(
            "pipeline.city_scalar_cache.feature_source must be a mapping or string, "
            f"got {type(payload)!r}"
        )

    kind = str(payload.get("kind", PROCESSED_CITY_SCALAR_FEATURE_KIND)).lower()
    if kind == PROCESSED_CITY_SCALAR_FEATURE_KIND:
        return CityScalarFeatureSourceOptions(kind=PROCESSED_CITY_SCALAR_FEATURE_KIND)
    if kind != AEF_CSV_FEATURE_KIND:
        raise ValueError(
            "pipeline.city_scalar_cache.feature_source.kind must be "
            f"{PROCESSED_CITY_SCALAR_FEATURE_KIND!r} or {AEF_CSV_FEATURE_KIND!r}, got {kind!r}"
        )

    root = payload.get("root", payload.get("dir", "data/AEF"))
    file_pattern = str(payload.get("file_pattern", "AEF_{year}.csv"))
    if "{year}" not in file_pattern:
        raise ValueError("AEF feature_source.file_pattern must contain '{year}'")
    id_column = str(payload.get("id_column", "ID"))
    if not id_column:
        raise ValueError("AEF feature_source.id_column must be non-empty")
    return CityScalarFeatureSourceOptions(
        kind=AEF_CSV_FEATURE_KIND,
        root=Path(str(root)),
        file_pattern=file_pattern,
        year_range=_normalize_year_range(payload.get("year_range"), default=DEFAULT_AEF_YEAR_RANGE),
        id_column=id_column,
        feature_columns=_normalize_feature_columns(payload.get("feature_columns")),
    )


def normalize_city_scalar_cache_options(
    pipeline_config: dict[str, Any],
    *,
    log_dir: str | None = None,
) -> CityScalarCacheOptions:
    del log_dir
    payload = pipeline_config.get("city_scalar_cache", {})
    if isinstance(payload, bool):
        payload = {"enabled": payload}
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise TypeError(f"pipeline.city_scalar_cache must be a mapping or bool, got {type(payload)!r}")

    cache_dir = payload.get("cache_dir", payload.get("dir", payload.get("root")))
    return CityScalarCacheOptions(
        enabled=bool(payload.get("enabled", True)),
        rebuild=bool(payload.get("rebuild", False)),
        rebuild_features=(
            None
            if payload.get("rebuild_features") is None
            else bool(payload.get("rebuild_features"))
        ),
        rebuild_targets=(
            None
            if payload.get("rebuild_targets") is None
            else bool(payload.get("rebuild_targets"))
        ),
        cache_dir=None if cache_dir in (None, "") else Path(str(cache_dir)),
        num_workers=max(0, int(payload.get("num_workers", 4))),
        batch_size=max(1, int(payload.get("batch_size", 1024))),
        feature_source=normalize_city_scalar_feature_source_options(payload.get("feature_source")),
    )


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _stable_hash(payload: Any) -> str:
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _path_payload(path: str | os.PathLike | None, *, hash_file: bool) -> dict[str, Any] | None:
    if path in (None, ""):
        return None
    resolved = Path(path).expanduser().resolve(strict=False)
    payload: dict[str, Any] = {"path": str(resolved), "exists": resolved.exists()}
    if hash_file and resolved.is_file():
        digest = hashlib.sha256()
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        payload["sha256"] = digest.hexdigest()
    return payload


def _crop_size_payload(crop_size: int | tuple[int, int]) -> int | list[int]:
    if isinstance(crop_size, tuple):
        return [int(value) for value in crop_size]
    return int(crop_size)


def _inclusive_years(year_range: tuple[int, int] | None) -> tuple[int, ...]:
    if year_range is None:
        return ()
    start, end = year_range
    return tuple(range(int(start), int(end) + 1))


def _aef_csv_path(feature_source: CityScalarFeatureSourceOptions, year: int) -> Path:
    if feature_source.root is None:
        raise ValueError("AEF CSV feature source requires a root directory")
    return feature_source.root / feature_source.file_pattern.format(year=int(year))


def _feature_source_payload(feature_source: CityScalarFeatureSourceOptions) -> dict[str, Any]:
    if feature_source.kind == PROCESSED_CITY_SCALAR_FEATURE_KIND:
        return {"kind": PROCESSED_CITY_SCALAR_FEATURE_KIND}
    if feature_source.kind != AEF_CSV_FEATURE_KIND:
        raise ValueError(f"Unsupported city-scalar feature source kind: {feature_source.kind!r}")

    years = _inclusive_years(feature_source.year_range)
    return {
        "kind": AEF_CSV_FEATURE_KIND,
        "root": _path_payload(feature_source.root, hash_file=False),
        "file_pattern": feature_source.file_pattern,
        "year_range": list(feature_source.year_range or ()),
        "id_column": feature_source.id_column,
        "feature_columns": list(feature_source.feature_columns),
        "files": [
            {
                "year": int(year),
                "file": _path_payload(_aef_csv_path(feature_source, year), hash_file=True),
            }
            for year in years
        ],
    }


def city_scalar_feature_fingerprint_payload(
    *,
    data_config: dict[str, Any],
    pipeline_spec: PipelineSpec,
    feature_source_options: CityScalarFeatureSourceOptions | None = None,
) -> dict[str, Any]:
    feature_source = feature_source_options or CityScalarFeatureSourceOptions()
    return {
        "version": CACHE_VERSION,
        "kind": "city_scalar_mlp_features",
        "feature_source": _feature_source_payload(feature_source),
        "data": {
            "eump_root": _path_payload(data_config.get("eump_root"), hash_file=False),
            "preprocessing_path": _path_payload(
                data_config.get("preprocessing_path", data_config.get("eump_root")),
                hash_file=True,
            ),
            "split_manifest_path": _path_payload(data_config.get("split_manifest_path"), hash_file=True),
        },
        "pipeline": {
            "kind": pipeline_spec.kind,
            "input_sources": list(pipeline_spec.input_sources),
            "year_range": None if pipeline_spec.year_range is None else list(pipeline_spec.year_range),
            "crop_size": _crop_size_payload(pipeline_spec.crop_size),
            "center_ratio": float(pipeline_spec.center_ratio),
            "city_mask_source": pipeline_spec.city_mask_source,
            "broadcast_scalar": bool(pipeline_spec.broadcast_scalar),
            "include_city_mask_as_input": bool(pipeline_spec.include_city_mask_as_input),
            "normalization": asdict(pipeline_spec.normalization_config),
        },
    }


def city_scalar_feature_fingerprint(
    *,
    data_config: dict[str, Any],
    pipeline_spec: PipelineSpec,
    feature_source_options: CityScalarFeatureSourceOptions | None = None,
) -> str:
    return _stable_hash(
        city_scalar_feature_fingerprint_payload(
            data_config=data_config,
            pipeline_spec=pipeline_spec,
            feature_source_options=feature_source_options,
        )
    )


def _target_spec_payload(target_schema: TargetSchema) -> dict[str, Any]:
    return {
        "spec": asdict(target_schema.spec),
        "channel_names": list(target_schema.channel_names),
        "channels": int(target_schema.channels),
    }


def city_scalar_target_fingerprint_payload(
    *,
    feature_fingerprint: str,
    target_schema: TargetSchema,
) -> dict[str, Any]:
    return {
        "version": CACHE_VERSION,
        "kind": "city_scalar_mlp_target",
        "feature_fingerprint": feature_fingerprint,
        "target": _target_spec_payload(target_schema),
    }


def city_scalar_target_fingerprint(*, feature_fingerprint: str, target_schema: TargetSchema) -> str:
    return _stable_hash(
        city_scalar_target_fingerprint_payload(
            feature_fingerprint=feature_fingerprint,
            target_schema=target_schema,
        )
    )


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return safe or "target"


def _resolve_cache_root(
    *,
    options: CityScalarCacheOptions,
    data_config: dict[str, Any],
    log_dir: str | None,
) -> Path:
    if options.cache_dir is not None:
        candidates = [options.cache_dir]
    elif options.enabled:
        candidates = [Path(str(data_config["eump_root"])) / "manifest" / "cache" / "city_scalar_mlp"]
        candidates.append(Path("log") / "city_scalar_mlp_cache")
        if log_dir:
            candidates.append(Path(log_dir) / "city_scalar_mlp_cache")
    else:
        candidates = [Path(log_dir or ".") / "city_scalar_mlp_cache" / "runtime"]

    last_error: OSError | None = None
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        except OSError as exc:
            last_error = exc
            continue
    raise OSError(f"Unable to create city scalar cache directory from candidates={candidates!r}") from last_error


def _feature_cache_path(cache_root: Path, fingerprint: str) -> Path:
    return cache_root / f"features_{fingerprint[:20]}.npz"


def _target_cache_path(cache_root: Path, target_schema: TargetSchema, fingerprint: str) -> Path:
    return cache_root / f"targets_{_safe_name(target_schema.name)}_{fingerprint[:20]}.npz"


def _cache_pipeline_config(
    pipeline_config: dict[str, Any],
    *,
    options: CityScalarCacheOptions,
    feature_only: bool,
) -> dict[str, Any]:
    cache_pipeline_config = deepcopy(pipeline_config)
    cache_pipeline_config["num_workers"] = int(options.num_workers)
    cache_pipeline_config["pin_memory"] = False
    if int(options.num_workers) <= 0:
        cache_pipeline_config["persistent_workers"] = False
    if feature_only:
        cache_pipeline_config["targets"] = {"raster": [], "scalar": [], "crop_scalar": []}
    return cache_pipeline_config


def _target_cache_pipeline_config(
    pipeline_config: dict[str, Any],
    *,
    options: CityScalarCacheOptions,
) -> dict[str, Any]:
    cache_pipeline_config = _cache_pipeline_config(
        pipeline_config,
        options=options,
        feature_only=False,
    )
    inputs_payload = cache_pipeline_config.get("inputs", {})
    if inputs_payload is None:
        inputs_payload = {}
    if not isinstance(inputs_payload, dict):
        raise TypeError(f"pipeline.inputs must be a mapping, got {type(inputs_payload)!r}")

    inputs_payload = dict(inputs_payload)
    city_mask_source = cache_pipeline_config.get("city_mask_source")
    if city_mask_source not in (None, ""):
        inputs_payload["sources"] = []
    # Without a city mask, the dataset still needs a raster input to infer
    # crop spatial_shape while building scalar target caches.
    cache_pipeline_config["inputs"] = inputs_payload
    cache_pipeline_config["broadcast_scalar"] = False
    if city_mask_source not in (None, ""):
        cache_pipeline_config["include_city_mask_as_input"] = True
    else:
        cache_pipeline_config["include_city_mask_as_input"] = False
    return cache_pipeline_config


def _atomic_savez(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("wb") as handle:
        np.savez(handle, **arrays)
    tmp_path.replace(path)


def _string_array(values: list[str] | tuple[str, ...]) -> np.ndarray:
    return np.asarray([str(value) for value in values], dtype=np.str_)


def _scalar_string(value: str) -> np.ndarray:
    return np.asarray(str(value), dtype=np.str_)


def _metadata_array(metadata: dict[str, Any]) -> np.ndarray:
    return _scalar_string(_stable_json(metadata))


def _split_batch_arrays(
    chunks: list[np.ndarray],
    *,
    width: int,
    dtype,
) -> np.ndarray:
    if not chunks:
        return np.empty((0, int(width)), dtype=dtype)
    return np.concatenate(chunks, axis=0).astype(dtype, copy=False)


def _append_meta(
    *,
    metas: list[dict[str, Any]],
    split_id: int,
    row_keys: list[str],
    scene_ids: list[str],
    sample_ids: list[str],
    years: list[int],
    split_ids: list[int],
) -> None:
    for meta in metas:
        scene_id = str(meta.get("scene_id", ""))
        year = int(meta.get("year"))
        sample_id = "" if meta.get("sample_id") is None else str(meta.get("sample_id"))
        row_keys.append(f"{scene_id}:{year}")
        scene_ids.append(scene_id)
        sample_ids.append(sample_id)
        years.append(year)
        split_ids.append(int(split_id))


def _load_active_preprocessing(
    *,
    data_config: dict[str, Any],
    pipeline_spec: PipelineSpec,
    reader: ProcessedSceneStoreReader,
) -> dict[str, Any]:
    preprocessing = load_preprocessing_spec(data_config.get("preprocessing_path", data_config["eump_root"]))
    validate_preprocessing_spec(preprocessing)
    validate_preprocessing_stats_scope(
        preprocessing,
        split_manifest_path=data_config.get("split_manifest_path"),
        expected_year_range=pipeline_spec.year_range,
    )
    return build_active_preprocessing_spec(pipeline_spec, reader.source_specs, preprocessing)


def build_aef_feature_layout(feature_source: CityScalarFeatureSourceOptions) -> CityLevelFeatureLayout:
    if feature_source.kind != AEF_CSV_FEATURE_KIND:
        raise ValueError(f"AEF feature layout requires feature_source.kind={AEF_CSV_FEATURE_KIND!r}")
    return CityLevelFeatureLayout(
        raster_slices=(),
        scalar_slices=(),
        feature_names=tuple(f"AEF:{column}:embedding" for column in feature_source.feature_columns),
    )


def _normalize_aef_ids(series: pd.Series, *, id_column: str, path: Path) -> list[str]:
    values = pd.to_numeric(series, errors="raise")
    if values.isna().any():
        raise ValueError(f"AEF CSV {path} contains missing {id_column!r} values")
    non_integer = values[np.floor(values.astype(float)) != values.astype(float)]
    if len(non_integer) > 0:
        raise ValueError(f"AEF CSV {path} contains non-integer {id_column!r} values")
    return [str(int(value)) for value in values.tolist()]


def _validate_aef_feature_columns(
    table: pd.DataFrame,
    *,
    expected_columns: tuple[str, ...],
    path: Path,
) -> None:
    detected = tuple(column for column in table.columns if re.fullmatch(r"A\d{2}", str(column)))
    missing = [column for column in expected_columns if column not in table.columns]
    extra = [column for column in detected if column not in expected_columns]
    if missing or extra:
        parts: list[str] = []
        if missing:
            parts.append(f"missing={missing!r}")
        if extra:
            parts.append(f"unexpected={extra!r}")
        raise ValueError(f"AEF CSV {path} feature columns do not match the configured AEF columns: {'; '.join(parts)}")


def _validate_aef_year_range(
    *,
    feature_source: CityScalarFeatureSourceOptions,
    pipeline_spec: PipelineSpec,
) -> tuple[int, ...]:
    years = _inclusive_years(feature_source.year_range)
    if not years:
        raise ValueError("AEF feature_source.year_range must select at least one year")
    if pipeline_spec.year_range is not None:
        start, end = pipeline_spec.year_range
        outside = [year for year in years if int(year) < int(start) or int(year) > int(end)]
        if outside:
            raise ValueError(
                "AEF feature_source.year_range must be inside pipeline.year_range: "
                f"aef={feature_source.year_range!r} pipeline={pipeline_spec.year_range!r}"
            )
    return years


def build_aef_csv_feature_cache(
    *,
    path: Path,
    fingerprint: str,
    data_config: dict[str, Any],
    pipeline_spec: PipelineSpec,
    feature_source: CityScalarFeatureSourceOptions,
    logger,
) -> None:
    if feature_source.kind != AEF_CSV_FEATURE_KIND:
        raise ValueError(f"AEF CSV feature cache requires feature_source.kind={AEF_CSV_FEATURE_KIND!r}")
    split_manifest_path = data_config.get("split_manifest_path")
    if not split_manifest_path:
        raise ValueError("AEF CSV feature cache requires data.split_manifest_path for split labels")
    split_manifest = load_split_manifest(
        split_manifest_path,
        expected_year_range=pipeline_spec.year_range,
    )
    split_lookup = split_label_lookup_from_manifest(split_manifest)
    years = _validate_aef_year_range(feature_source=feature_source, pipeline_spec=pipeline_spec)

    feature_chunks: list[np.ndarray] = []
    valid_chunks: list[np.ndarray] = []
    row_keys: list[str] = []
    scene_ids: list[str] = []
    sample_ids: list[str] = []
    year_values: list[int] = []
    split_ids: list[int] = []
    total_csv_rows = 0
    skipped_unassigned_rows = 0

    for year in years:
        csv_path = _aef_csv_path(feature_source, year)
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing AEF CSV for year {year}: {csv_path}")
        table = pd.read_csv(csv_path)
        if feature_source.id_column not in table.columns:
            raise ValueError(f"AEF CSV {csv_path} is missing id column {feature_source.id_column!r}")
        _validate_aef_feature_columns(
            table,
            expected_columns=feature_source.feature_columns,
            path=csv_path,
        )
        ids = _normalize_aef_ids(table[feature_source.id_column], id_column=feature_source.id_column, path=csv_path)
        duplicate_ids = sorted({scene_id for scene_id in ids if ids.count(scene_id) > 1})
        if duplicate_ids:
            raise ValueError(f"AEF CSV {csv_path} contains duplicate ID values: {duplicate_ids[:20]!r}")

        values = table.loc[:, list(feature_source.feature_columns)].to_numpy(dtype=np.float32, copy=True)
        valid = np.isfinite(values)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

        selected_indices: list[int] = []
        for row_index, scene_id in enumerate(ids):
            split_name = split_lookup.get(scene_id)
            if split_name is None:
                skipped_unassigned_rows += 1
                continue
            selected_indices.append(row_index)
            row_keys.append(f"{scene_id}:{int(year)}")
            scene_ids.append(scene_id)
            sample_ids.append(scene_id)
            year_values.append(int(year))
            split_ids.append(SPLIT_NAME_TO_ID[split_name])

        total_csv_rows += len(ids)
        if selected_indices:
            selected = np.asarray(selected_indices, dtype=np.int64)
            feature_chunks.append(values[selected])
            valid_chunks.append(valid[selected])

    features = _split_batch_arrays(feature_chunks, width=len(feature_source.feature_columns), dtype=np.float32)
    feature_valid_mask = _split_batch_arrays(valid_chunks, width=len(feature_source.feature_columns), dtype=bool)
    valid_feature_rows = int(np.sum(feature_valid_mask.all(axis=1) & np.isfinite(features).all(axis=1)))
    metadata = {
        "cache_version": CACHE_VERSION,
        "cache_kind": "features",
        "feature_source": AEF_CSV_FEATURE_KIND,
        "feature_scale": "aef_embedding",
        "feature_aggregation": "none",
        "rows": int(features.shape[0]),
        "valid_feature_rows": valid_feature_rows,
        "feature_dim": int(features.shape[1]),
        "aef_root": None if feature_source.root is None else str(feature_source.root),
        "aef_file_pattern": feature_source.file_pattern,
        "aef_year_range": list(feature_source.year_range or ()),
        "aef_id_column": feature_source.id_column,
        "total_csv_rows": int(total_csv_rows),
        "skipped_unassigned_rows": int(skipped_unassigned_rows),
    }
    _atomic_savez(
        path,
        features=features,
        feature_valid_mask=feature_valid_mask,
        row_keys=_string_array(row_keys),
        scene_ids=_string_array(scene_ids),
        sample_ids=_string_array(sample_ids),
        years=np.asarray(year_values, dtype=np.int32),
        split_ids=np.asarray(split_ids, dtype=np.int8),
        feature_names=_string_array(tuple(f"AEF:{column}:embedding" for column in feature_source.feature_columns)),
        fingerprint=_scalar_string(fingerprint),
        metadata_json=_metadata_array(metadata),
    )
    if logger is not None:
        logger.info(
            "Built AEF CSV feature cache "
            f"path={path} years={list(years)} rows={features.shape[0]} valid_rows={valid_feature_rows}"
        )


def build_city_scalar_feature_cache(
    *,
    path: Path,
    fingerprint: str,
    data_config: dict[str, Any],
    pipeline_config: dict[str, Any],
    options: CityScalarCacheOptions,
    logger,
    random_seed: int,
    feature_layout: CityLevelFeatureLayout,
) -> None:
    cache_pipeline = PipelineSpec.from_mapping(
        _cache_pipeline_config(pipeline_config, options=options, feature_only=True)
    )
    train_dl, val_dl, test_dl, batch_schema = build_eump_dataloaders(
        data_config=data_config,
        pipeline_spec=cache_pipeline,
        logger=logger,
        random_seed=random_seed,
        batch_size=options.batch_size,
        local_rank=0,
        world_size=1,
        is_distributed=False,
        shuffle_train=False,
    )
    reader = getattr(train_dl.dataset, "reader", None)
    if reader is None:
        raise ValueError("Unable to resolve EUMP reader while building city scalar feature cache")
    preprocessing_spec = _load_active_preprocessing(
        data_config=data_config,
        pipeline_spec=cache_pipeline,
        reader=reader,
    )
    build_city_level_feature_layout(
        pipeline_spec=cache_pipeline,
        source_specs=reader.source_specs,
        preprocessing_spec=preprocessing_spec,
        batch_schema=batch_schema,
    )

    feature_chunks: list[np.ndarray] = []
    valid_chunks: list[np.ndarray] = []
    row_keys: list[str] = []
    scene_ids: list[str] = []
    sample_ids: list[str] = []
    years: list[int] = []
    split_ids: list[int] = []

    for split_name, dataloader in (("train", train_dl), ("val", val_dl), ("test", test_dl)):
        split_id = SPLIT_NAME_TO_ID[split_name]
        for batch in dataloader:
            features = build_city_level_features(
                batch=batch,
                layout=feature_layout,
                preprocessing_spec=preprocessing_spec,
            )
            feature_chunks.append(features.data.detach().cpu().numpy().astype(np.float32, copy=False))
            valid_chunks.append(features.valid_mask.detach().cpu().numpy().astype(bool, copy=False))
            _append_meta(
                metas=batch.meta,
                split_id=split_id,
                row_keys=row_keys,
                scene_ids=scene_ids,
                sample_ids=sample_ids,
                years=years,
                split_ids=split_ids,
            )

    features = _split_batch_arrays(feature_chunks, width=feature_layout.feature_dim, dtype=np.float32)
    feature_valid_mask = _split_batch_arrays(valid_chunks, width=feature_layout.feature_dim, dtype=bool)
    valid_feature_rows = int(np.sum(feature_valid_mask.all(axis=1) & np.isfinite(features).all(axis=1)))
    metadata = {
        "cache_version": CACHE_VERSION,
        "cache_kind": "features",
        "feature_scale": "processed_city_scalar",
        "feature_aggregation": "log_mean_then_processed_for_log_zscore_plus_city_mask_crop_mean",
        "rows": int(features.shape[0]),
        "valid_feature_rows": valid_feature_rows,
        "feature_dim": int(feature_layout.feature_dim),
    }
    _atomic_savez(
        path,
        features=features,
        feature_valid_mask=feature_valid_mask,
        row_keys=_string_array(row_keys),
        scene_ids=_string_array(scene_ids),
        sample_ids=_string_array(sample_ids),
        years=np.asarray(years, dtype=np.int32),
        split_ids=np.asarray(split_ids, dtype=np.int8),
        feature_names=_string_array(feature_layout.feature_names),
        fingerprint=_scalar_string(fingerprint),
        metadata_json=_metadata_array(metadata),
    )


def build_city_scalar_target_cache(
    *,
    path: Path,
    fingerprint: str,
    data_config: dict[str, Any],
    pipeline_config: dict[str, Any],
    options: CityScalarCacheOptions,
    logger,
    random_seed: int,
    target_schema: TargetSchema,
) -> None:
    cache_pipeline = PipelineSpec.from_mapping(
        _target_cache_pipeline_config(pipeline_config, options=options)
    )
    train_dl, val_dl, test_dl, batch_schema = build_eump_dataloaders(
        data_config=data_config,
        pipeline_spec=cache_pipeline,
        logger=logger,
        random_seed=random_seed,
        batch_size=options.batch_size,
        local_rank=0,
        world_size=1,
        is_distributed=False,
        shuffle_train=False,
    )
    reader = getattr(train_dl.dataset, "reader", None)
    if reader is None:
        raise ValueError("Unable to resolve EUMP reader while building city scalar target cache")
    preprocessing_spec = _load_active_preprocessing(
        data_config=data_config,
        pipeline_spec=cache_pipeline,
        reader=reader,
    )
    cache_target_schema = get_single_downstream_target_schema(batch_schema)
    if cache_target_schema.name != target_schema.name or cache_target_schema.channels != target_schema.channels:
        raise ValueError(
            "City scalar target cache schema changed during build: "
            f"expected {target_schema.name}/{target_schema.channels}, "
            f"got {cache_target_schema.name}/{cache_target_schema.channels}"
        )

    target_chunks: list[np.ndarray] = []
    valid_chunks: list[np.ndarray] = []
    row_keys: list[str] = []
    scene_ids: list[str] = []
    sample_ids: list[str] = []
    years: list[int] = []
    split_ids: list[int] = []

    for split_name, dataloader in (("train", train_dl), ("val", val_dl), ("test", test_dl)):
        split_id = SPLIT_NAME_TO_ID[split_name]
        for batch in dataloader:
            target = build_city_level_target(
                batch=batch,
                target_schema=target_schema,
                preprocessing_spec=preprocessing_spec,
            )
            target_chunks.append(target.data.detach().cpu().numpy().astype(np.float32, copy=False))
            valid_chunks.append(target.valid_mask.detach().cpu().numpy().astype(bool, copy=False))
            _append_meta(
                metas=batch.meta,
                split_id=split_id,
                row_keys=row_keys,
                scene_ids=scene_ids,
                sample_ids=sample_ids,
                years=years,
                split_ids=split_ids,
            )

    targets = _split_batch_arrays(target_chunks, width=target_schema.channels, dtype=np.float32)
    target_valid_mask = _split_batch_arrays(valid_chunks, width=target_schema.channels, dtype=bool)
    valid_target_rows = int(np.sum(target_valid_mask.all(axis=1) & np.isfinite(targets).all(axis=1)))
    metadata = {
        "cache_version": CACHE_VERSION,
        "cache_kind": "target",
        "target_scale": "processed",
        "rows": int(targets.shape[0]),
        "valid_target_rows": valid_target_rows,
        "target_name": target_schema.name,
        "target_dim": int(target_schema.channels),
    }
    _atomic_savez(
        path,
        targets=targets,
        target_valid_mask=target_valid_mask,
        row_keys=_string_array(row_keys),
        scene_ids=_string_array(scene_ids),
        sample_ids=_string_array(sample_ids),
        years=np.asarray(years, dtype=np.int32),
        split_ids=np.asarray(split_ids, dtype=np.int8),
        target_name=_scalar_string(target_schema.name),
        target_channel_names=_string_array(target_schema.channel_names),
        fingerprint=_scalar_string(fingerprint),
        metadata_json=_metadata_array(metadata),
    )


def _metadata_from_payload(payload) -> dict[str, Any]:
    if "metadata_json" not in payload:
        return {}
    raw = payload["metadata_json"]
    value = raw.item() if getattr(raw, "shape", ()) == () else raw.tolist()
    return json.loads(str(value))


def _scalar_from_payload(payload, name: str) -> str:
    raw = payload[name]
    return str(raw.item() if getattr(raw, "shape", ()) == () else raw.tolist())


def load_city_scalar_feature_cache(path: Path, *, expected_fingerprint: str | None = None) -> CityScalarFeatureCache:
    with np.load(path, allow_pickle=False) as payload:
        fingerprint = _scalar_from_payload(payload, "fingerprint")
        if expected_fingerprint is not None and fingerprint != expected_fingerprint:
            raise ValueError(
                f"City scalar feature cache fingerprint mismatch for {path}: "
                f"expected {expected_fingerprint}, got {fingerprint}"
            )
        return CityScalarFeatureCache(
            path=Path(path),
            fingerprint=fingerprint,
            features=payload["features"].astype(np.float32, copy=False),
            feature_valid_mask=payload["feature_valid_mask"].astype(bool, copy=False),
            row_keys=payload["row_keys"].astype(str, copy=False),
            scene_ids=payload["scene_ids"].astype(str, copy=False),
            sample_ids=payload["sample_ids"].astype(str, copy=False),
            years=payload["years"].astype(np.int32, copy=False),
            split_ids=payload["split_ids"].astype(np.int8, copy=False),
            feature_names=tuple(str(value) for value in payload["feature_names"].tolist()),
            metadata=_metadata_from_payload(payload),
        )


def load_city_scalar_target_cache(path: Path, *, expected_fingerprint: str | None = None) -> CityScalarTargetCache:
    with np.load(path, allow_pickle=False) as payload:
        fingerprint = _scalar_from_payload(payload, "fingerprint")
        if expected_fingerprint is not None and fingerprint != expected_fingerprint:
            raise ValueError(
                f"City scalar target cache fingerprint mismatch for {path}: "
                f"expected {expected_fingerprint}, got {fingerprint}"
            )
        return CityScalarTargetCache(
            path=Path(path),
            fingerprint=fingerprint,
            targets=payload["targets"].astype(np.float32, copy=False),
            target_valid_mask=payload["target_valid_mask"].astype(bool, copy=False),
            row_keys=payload["row_keys"].astype(str, copy=False),
            scene_ids=payload["scene_ids"].astype(str, copy=False),
            sample_ids=payload["sample_ids"].astype(str, copy=False),
            years=payload["years"].astype(np.int32, copy=False),
            split_ids=payload["split_ids"].astype(np.int8, copy=False),
            target_name=_scalar_from_payload(payload, "target_name"),
            target_channel_names=tuple(str(value) for value in payload["target_channel_names"].tolist()),
            metadata=_metadata_from_payload(payload),
        )


def join_city_scalar_caches(
    feature_cache: CityScalarFeatureCache,
    target_cache: CityScalarTargetCache,
) -> CityScalarJoinedArrays:
    feature_index: dict[str, int] = {}
    for idx, row_key in enumerate(feature_cache.row_keys.tolist()):
        row_key = str(row_key)
        if row_key in feature_index:
            raise ValueError(f"Duplicate row_key {row_key!r} in feature cache {feature_cache.path}")
        feature_index[row_key] = idx

    feature_indices: list[int] = []
    target_indices: list[int] = []
    for target_idx, row_key in enumerate(target_cache.row_keys.tolist()):
        feature_idx = feature_index.get(str(row_key))
        if feature_idx is None:
            continue
        feature_indices.append(feature_idx)
        target_indices.append(target_idx)

    if feature_indices:
        f_idx = np.asarray(feature_indices, dtype=np.int64)
        t_idx = np.asarray(target_indices, dtype=np.int64)
        features = feature_cache.features[f_idx]
        feature_valid = feature_cache.feature_valid_mask[f_idx]
        targets = target_cache.targets[t_idx]
        target_valid = target_cache.target_valid_mask[t_idx]
        split_ids = target_cache.split_ids[t_idx]
        feature_split_ids = feature_cache.split_ids[f_idx]
        if not np.array_equal(split_ids, feature_split_ids):
            raise ValueError("Feature and target cache split assignments differ for joined rows")
        row_keys = target_cache.row_keys[t_idx]
        scene_ids = target_cache.scene_ids[t_idx]
        sample_ids = target_cache.sample_ids[t_idx]
        years = target_cache.years[t_idx]
    else:
        features = np.empty((0, feature_cache.features.shape[1]), dtype=np.float32)
        feature_valid = np.empty((0, feature_cache.feature_valid_mask.shape[1]), dtype=bool)
        targets = np.empty((0, target_cache.targets.shape[1]), dtype=np.float32)
        target_valid = np.empty((0, target_cache.target_valid_mask.shape[1]), dtype=bool)
        split_ids = np.empty((0,), dtype=np.int8)
        row_keys = np.empty((0,), dtype=str)
        scene_ids = np.empty((0,), dtype=str)
        sample_ids = np.empty((0,), dtype=str)
        years = np.empty((0,), dtype=np.int32)

    valid_rows = (
        feature_valid.all(axis=1)
        & target_valid.all(axis=1)
        & np.isfinite(features).all(axis=1)
        & np.isfinite(targets).all(axis=1)
    )
    metadata = {
        "feature_cache": str(feature_cache.path),
        "target_cache": str(target_cache.path),
        "feature_rows": int(feature_cache.features.shape[0]),
        "valid_feature_rows": int(
            np.sum(feature_cache.feature_valid_mask.all(axis=1) & np.isfinite(feature_cache.features).all(axis=1))
        ),
        "target_rows": int(target_cache.targets.shape[0]),
        "valid_target_rows": int(
            np.sum(target_cache.target_valid_mask.all(axis=1) & np.isfinite(target_cache.targets).all(axis=1))
        ),
        "joined_rows": int(features.shape[0]),
        "valid_joined_rows": int(np.sum(valid_rows)),
        "split_counts": {
            split_name: int(np.sum(split_ids[valid_rows] == split_id))
            for split_name, split_id in SPLIT_NAME_TO_ID.items()
        },
    }
    return CityScalarJoinedArrays(
        features=features[valid_rows].astype(np.float32, copy=False),
        targets=targets[valid_rows].astype(np.float32, copy=False),
        split_ids=split_ids[valid_rows].astype(np.int8, copy=False),
        row_keys=row_keys[valid_rows],
        scene_ids=scene_ids[valid_rows],
        sample_ids=sample_ids[valid_rows],
        years=years[valid_rows].astype(np.int32, copy=False),
        metadata=metadata,
    )


def compute_cached_city_level_standardization_stats(
    *,
    features: np.ndarray,
    targets: np.ndarray,
    device: torch.device,
    feature_standardization: str = FEATURE_STANDARDIZATION_TRAIN_SPLIT_ZSCORE,
    variance_epsilon: float = 1.0e-12,
) -> StandardizationStats:
    if features.shape[0] <= 0:
        raise ValueError("Scalar MLP baseline found no valid training samples for standardization")
    feature_standardization = normalize_city_scalar_feature_standardization(feature_standardization)
    if feature_standardization == FEATURE_STANDARDIZATION_NONE:
        feature_mean = torch.zeros(features.shape[1], dtype=torch.float32, device=device)
        feature_std = torch.ones(features.shape[1], dtype=torch.float32, device=device)
    else:
        feature_tensor = torch.as_tensor(features, dtype=torch.float64, device=device)
        feature_mean = feature_tensor.mean(dim=0)
        feature_std = torch.sqrt(torch.clamp(feature_tensor.var(dim=0, unbiased=False), min=0.0))
        feature_std = torch.where(feature_std > variance_epsilon, feature_std, torch.ones_like(feature_std))
    return StandardizationStats(
        feature_mean=feature_mean.float(),
        feature_std=feature_std.float(),
        target_mean=torch.zeros(targets.shape[1], dtype=torch.float32, device=device),
        target_std=torch.ones(targets.shape[1], dtype=torch.float32, device=device),
        count=int(features.shape[0]),
    )


def _feature_cache_split_features(cache: CityScalarFeatureCache, split_name: str) -> np.ndarray:
    split_id = SPLIT_NAME_TO_ID[split_name]
    valid_rows = (
        (cache.split_ids == split_id)
        & cache.feature_valid_mask.all(axis=1)
        & np.isfinite(cache.features).all(axis=1)
    )
    features = cache.features[valid_rows].astype(np.float32, copy=False)
    if features.shape[0] <= 0:
        raise ValueError(
            f"Scalar MLP baseline found no valid feature-cache {split_name!r} samples for standardization"
        )
    return features


def _split_arrays(joined: CityScalarJoinedArrays, split_name: str) -> tuple[np.ndarray, np.ndarray]:
    split_id = SPLIT_NAME_TO_ID[split_name]
    mask = joined.split_ids == split_id
    return joined.features[mask], joined.targets[mask]


def _cache_build_data_config(data_config: dict[str, Any]) -> dict[str, Any]:
    cache_data_config = dict(data_config)
    cache_data_config["runtime_split_manifest_path"] = None
    return cache_data_config


def _runtime_split_manifest_path(data_config: dict[str, Any]) -> str | None:
    path = data_config.get("runtime_split_manifest_path")
    if path in (None, ""):
        return None
    return str(path)


def _split_ids_from_manifest_for_scene_ids(scene_ids: np.ndarray, split_manifest_path: str) -> np.ndarray:
    split_manifest = load_split_manifest(split_manifest_path)
    split_lookup = split_label_lookup_from_manifest(split_manifest)
    split_ids: list[int] = []
    missing: list[str] = []
    for scene_id in scene_ids.astype(str, copy=False):
        split_name = split_lookup.get(str(scene_id))
        if split_name is None:
            missing.append(str(scene_id))
            split_ids.append(-1)
        else:
            split_ids.append(SPLIT_NAME_TO_ID[split_name])
    if missing:
        preview = ", ".join(sorted(set(missing))[:10])
        suffix = "" if len(set(missing)) <= 10 else f", ... ({len(set(missing))} total)"
        raise ValueError(
            f"runtime split manifest {split_manifest_path} does not assign scene id(s): {preview}{suffix}"
        )
    return np.asarray(split_ids, dtype=np.int8)


def _relabel_feature_cache_splits(
    cache: CityScalarFeatureCache,
    *,
    split_manifest_path: str | None,
) -> CityScalarFeatureCache:
    if split_manifest_path is None:
        return cache
    metadata = dict(cache.metadata)
    metadata["stored_split_ids_source"] = metadata.get("runtime_split_manifest_path") or metadata.get(
        "split_manifest_path"
    )
    metadata["runtime_split_manifest_path"] = str(split_manifest_path)
    return CityScalarFeatureCache(
        path=cache.path,
        fingerprint=cache.fingerprint,
        features=cache.features,
        feature_valid_mask=cache.feature_valid_mask,
        row_keys=cache.row_keys,
        scene_ids=cache.scene_ids,
        sample_ids=cache.sample_ids,
        years=cache.years,
        split_ids=_split_ids_from_manifest_for_scene_ids(cache.scene_ids, split_manifest_path),
        feature_names=cache.feature_names,
        metadata=metadata,
    )


def _relabel_target_cache_splits(
    cache: CityScalarTargetCache,
    *,
    split_manifest_path: str | None,
) -> CityScalarTargetCache:
    if split_manifest_path is None:
        return cache
    metadata = dict(cache.metadata)
    metadata["stored_split_ids_source"] = metadata.get("runtime_split_manifest_path") or metadata.get(
        "split_manifest_path"
    )
    metadata["runtime_split_manifest_path"] = str(split_manifest_path)
    return CityScalarTargetCache(
        path=cache.path,
        fingerprint=cache.fingerprint,
        targets=cache.targets,
        target_valid_mask=cache.target_valid_mask,
        row_keys=cache.row_keys,
        scene_ids=cache.scene_ids,
        sample_ids=cache.sample_ids,
        years=cache.years,
        split_ids=_split_ids_from_manifest_for_scene_ids(cache.scene_ids, split_manifest_path),
        target_name=cache.target_name,
        target_channel_names=cache.target_channel_names,
        metadata=metadata,
    )


def _build_cached_loader(
    *,
    dataset: CityScalarTensorDataset,
    batch_size: int,
    random_seed: int,
    local_rank: int,
    world_size: int,
    train: bool,
    pin_memory: bool,
) -> DataLoader:
    sampler = None
    if train or int(world_size) > 1:
        sampler = DistributedSampler(
            dataset,
            shuffle=train,
            seed=int(random_seed),
            num_replicas=int(world_size),
            rank=int(local_rank),
        )
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=(sampler is None and train),
        sampler=sampler,
        num_workers=0,
        pin_memory=bool(pin_memory),
        collate_fn=collate_cached_city_scalar_samples,
        generator=seeded_torch_generator(
            random_seed,
            offset=(1 if train else 2) + int(local_rank) * 1_000_000,
        ),
    )


def _distributed_rank(local_rank: int) -> int:
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank())
    return int(local_rank)


def _distributed_barrier(local_rank: int) -> None:
    torch_distributed_barrier(local_rank)


def ensure_city_scalar_cache_files(
    *,
    data_config: dict[str, Any],
    pipeline_config: dict[str, Any],
    logger,
    random_seed: int,
    local_rank: int,
    log_dir: str | None,
) -> tuple[
    CityScalarFeatureCache,
    CityScalarTargetCache,
    ProcessedBatchSchema,
    TargetSchema,
    CityLevelFeatureLayout,
    dict[str, Any],
]:
    pipeline_spec = PipelineSpec.from_mapping(pipeline_config)
    validate_eump_scalar_mlp_downstream_spec(pipeline_spec)
    options = normalize_city_scalar_cache_options(pipeline_config, log_dir=log_dir)
    cache_data_config = _cache_build_data_config(data_config)
    runtime_split_manifest_path = _runtime_split_manifest_path(data_config)
    reader = ProcessedSceneStoreReader(data_config["eump_root"])
    preprocessing = load_preprocessing_spec(data_config.get("preprocessing_path", data_config["eump_root"]))
    validate_preprocessing_spec(preprocessing)
    validate_preprocessing_stats_scope(
        preprocessing,
        split_manifest_path=data_config.get("split_manifest_path"),
        expected_year_range=pipeline_spec.year_range,
    )
    active_preprocessing = build_active_preprocessing_spec(pipeline_spec, reader.source_specs, preprocessing)
    batch_schema = build_eump_batch_schema(pipeline_spec, reader.source_specs, preprocessing)
    target_schema = get_single_downstream_target_schema(batch_schema)
    if options.feature_source.kind == AEF_CSV_FEATURE_KIND:
        feature_layout = build_aef_feature_layout(options.feature_source)
    else:
        feature_layout = build_city_level_feature_layout(
            pipeline_spec=pipeline_spec,
            source_specs=reader.source_specs,
            preprocessing_spec=active_preprocessing,
            batch_schema=batch_schema,
        )

    cache_root = _resolve_cache_root(options=options, data_config=data_config, log_dir=log_dir)
    feature_fingerprint = city_scalar_feature_fingerprint(
        data_config=cache_data_config,
        pipeline_spec=pipeline_spec,
        feature_source_options=options.feature_source,
    )
    target_fingerprint = city_scalar_target_fingerprint(
        feature_fingerprint=feature_fingerprint,
        target_schema=target_schema,
    )
    feature_path = _feature_cache_path(cache_root, feature_fingerprint)
    target_path = _target_cache_path(cache_root, target_schema, target_fingerprint)
    rebuild_features = options.rebuild if options.rebuild_features is None else options.rebuild_features
    rebuild_targets = options.rebuild if options.rebuild_targets is None else options.rebuild_targets

    rank = _distributed_rank(local_rank)
    if rank == 0:
        if rebuild_features or not feature_path.exists():
            if logger is not None:
                logger.info(f"Build city-scalar shared feature cache: {feature_path}")
            if options.feature_source.kind == AEF_CSV_FEATURE_KIND:
                build_aef_csv_feature_cache(
                    path=feature_path,
                    fingerprint=feature_fingerprint,
                    data_config=cache_data_config,
                    pipeline_spec=pipeline_spec,
                    feature_source=options.feature_source,
                    logger=logger,
                )
            else:
                build_city_scalar_feature_cache(
                    path=feature_path,
                    fingerprint=feature_fingerprint,
                    data_config=cache_data_config,
                    pipeline_config=pipeline_config,
                    options=options,
                    logger=logger,
                    random_seed=random_seed,
                    feature_layout=feature_layout,
                )
        elif logger is not None:
            logger.info(f"Reuse city-scalar shared feature cache: {feature_path}")

        if rebuild_targets or not target_path.exists():
            if logger is not None:
                logger.info(f"Build city-scalar target cache for {target_schema.name}: {target_path}")
            build_city_scalar_target_cache(
                path=target_path,
                fingerprint=target_fingerprint,
                data_config=cache_data_config,
                pipeline_config=pipeline_config,
                options=options,
                logger=logger,
                random_seed=random_seed,
                target_schema=target_schema,
            )
        elif logger is not None:
            logger.info(f"Reuse city-scalar target cache for {target_schema.name}: {target_path}")

    _distributed_barrier(local_rank)
    feature_cache = load_city_scalar_feature_cache(feature_path, expected_fingerprint=feature_fingerprint)
    target_cache = load_city_scalar_target_cache(target_path, expected_fingerprint=target_fingerprint)
    feature_cache = _relabel_feature_cache_splits(
        feature_cache,
        split_manifest_path=runtime_split_manifest_path,
    )
    target_cache = _relabel_target_cache_splits(
        target_cache,
        split_manifest_path=runtime_split_manifest_path,
    )
    if tuple(feature_cache.feature_names) != tuple(feature_layout.feature_names):
        raise ValueError(
            "City scalar feature cache names do not match current feature layout: "
            f"cache={feature_cache.feature_names!r}, current={feature_layout.feature_names!r}"
        )
    if tuple(target_cache.target_channel_names) != tuple(target_schema.channel_names):
        raise ValueError(
            "City scalar target cache names do not match current target schema: "
            f"cache={target_cache.target_channel_names!r}, current={target_schema.channel_names!r}"
        )
    return feature_cache, target_cache, batch_schema, target_schema, feature_layout, active_preprocessing


def prepare_city_scalar_mlp_data(
    *,
    data_config: dict[str, Any],
    pipeline_config: dict[str, Any],
    logger,
    random_seed: int,
    batch_size: int,
    local_rank: int,
    world_size: int,
    device: torch.device,
    log_dir: str | None,
    feature_standardization: str = FEATURE_STANDARDIZATION_TRAIN_SPLIT_ZSCORE,
) -> CityScalarPreparedData:
    feature_standardization = normalize_city_scalar_feature_standardization(feature_standardization)
    (
        feature_cache,
        target_cache,
        batch_schema,
        target_schema,
        feature_layout,
        preprocessing_spec,
    ) = ensure_city_scalar_cache_files(
        data_config=data_config,
        pipeline_config=pipeline_config,
        logger=logger,
        random_seed=random_seed,
        local_rank=local_rank,
        log_dir=log_dir,
    )
    pipeline_spec = PipelineSpec.from_mapping(pipeline_config)
    joined = join_city_scalar_caches(feature_cache, target_cache)
    train_features, train_targets = _split_arrays(joined, "train")
    val_features, val_targets = _split_arrays(joined, "val")
    test_features, test_targets = _split_arrays(joined, "test")
    if train_features.shape[0] <= 0:
        raise ValueError(f"City-scalar MLP target {target_schema.name!r} has no valid train samples after cache join")

    if feature_standardization == FEATURE_STANDARDIZATION_FEATURE_CACHE_TRAIN_SPLIT:
        standardization_features = _feature_cache_split_features(feature_cache, "train")
        standardization_source = "feature_cache_train_split"
    else:
        standardization_features = train_features
        standardization_source = "train"

    standardization = compute_cached_city_level_standardization_stats(
        features=standardization_features,
        targets=train_targets,
        device=device,
        feature_standardization=feature_standardization,
    )
    train_dataset = CityScalarTensorDataset(features=train_features, targets=train_targets)
    val_dataset = CityScalarTensorDataset(features=val_features, targets=val_targets)
    test_dataset = CityScalarTensorDataset(features=test_features, targets=test_targets)
    train_dl = _build_cached_loader(
        dataset=train_dataset,
        batch_size=batch_size,
        random_seed=random_seed,
        local_rank=local_rank,
        world_size=world_size,
        train=True,
        pin_memory=pipeline_spec.pin_memory,
    )
    val_dl = _build_cached_loader(
        dataset=val_dataset,
        batch_size=batch_size,
        random_seed=random_seed,
        local_rank=local_rank,
        world_size=world_size,
        train=False,
        pin_memory=pipeline_spec.pin_memory,
    )
    test_dl = _build_cached_loader(
        dataset=test_dataset,
        batch_size=batch_size,
        random_seed=random_seed,
        local_rank=local_rank,
        world_size=world_size,
        train=False,
        pin_memory=pipeline_spec.pin_memory,
    )
    cache_metadata = {
        **joined.metadata,
        "feature_fingerprint": feature_cache.fingerprint,
        "target_fingerprint": target_cache.fingerprint,
        "feature_source": feature_cache.metadata.get("feature_source", PROCESSED_CITY_SCALAR_FEATURE_KIND),
        "feature_scale": feature_cache.metadata.get("feature_scale", "processed_city_scalar"),
        "feature_aggregation": feature_cache.metadata.get(
            "feature_aggregation",
            "log_mean_then_processed_for_log_zscore_plus_city_mask_crop_mean",
        ),
        "target_scale": "processed",
        "feature_standardization": feature_standardization,
        "target_training_standardization": "identity",
        "standardization_source": standardization_source,
        "preprocessing_stats_split_manifest_path": data_config.get("split_manifest_path"),
        "runtime_split_manifest_path": data_config.get("runtime_split_manifest_path")
        or data_config.get("split_manifest_path"),
        "feature_names": list(feature_layout.feature_names),
        "target_name": target_schema.name,
        "target_channel_names": list(target_schema.channel_names),
    }
    if _distributed_rank(local_rank) == 0 and logger is not None:
        counts = cache_metadata["split_counts"]
        logger.info(
            "Prepared city-scalar cached dataset "
            f"target={target_schema.name} "
            f"features={feature_layout.feature_dim} "
            f"joined_valid={cache_metadata['valid_joined_rows']} "
            f"train={counts['train']} val={counts['val']} test={counts['test']}"
        )
    return CityScalarPreparedData(
        train_dl=train_dl,
        val_dl=val_dl,
        test_dl=test_dl,
        batch_schema=batch_schema,
        target_schema=target_schema,
        feature_layout=feature_layout,
        standardization=standardization,
        cache_metadata=cache_metadata,
        preprocessing_spec=preprocessing_spec,
    )


__all__ = [
    "AEF_CSV_FEATURE_KIND",
    "CachedCityScalarBatch",
    "CityScalarCacheOptions",
    "CityScalarFeatureSourceOptions",
    "CityScalarFeatureCache",
    "CityScalarJoinedArrays",
    "CityScalarPreparedData",
    "CityScalarTargetCache",
    "CityScalarTensorDataset",
    "FEATURE_STANDARDIZATION_FEATURE_CACHE_TRAIN_SPLIT",
    "FEATURE_STANDARDIZATION_NONE",
    "FEATURE_STANDARDIZATION_TRAIN_SPLIT_ZSCORE",
    "PROCESSED_CITY_SCALAR_FEATURE_KIND",
    "build_aef_csv_feature_cache",
    "build_aef_feature_layout",
    "city_scalar_feature_fingerprint",
    "city_scalar_feature_fingerprint_payload",
    "city_scalar_target_fingerprint",
    "city_scalar_target_fingerprint_payload",
    "collate_cached_city_scalar_samples",
    "compute_cached_city_level_standardization_stats",
    "ensure_city_scalar_cache_files",
    "join_city_scalar_caches",
    "load_city_scalar_feature_cache",
    "load_city_scalar_target_cache",
    "normalize_city_scalar_cache_options",
    "normalize_city_scalar_feature_standardization",
    "normalize_city_scalar_feature_source_options",
    "prepare_city_scalar_mlp_data",
]
