from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from urban_embedding.core.paths import EUMP_DATA_DIR
from urban_embedding.data.contracts import RasterBlock, ScalarBlock, SceneInputs, SceneRecord, SceneTargets

from .alignment import import_zarr
from .manifest import (
    load_eump_store_schema,
    preprocessing_stats_path,
    read_json,
    read_jsonl,
    scene_years_manifest_path,
    scenes_manifest_path,
    write_json,
)
from .reader import concat_raster_blocks, concat_scalar_blocks, normalize_scene_id
from .split_manifest import load_split_manifest, split_scene_ids_from_manifest
from .task_config import ProcessedTaskConfig


PREPROCESSING_VERSION = "preprocessing_v1"
_LC_CATEGORIES = tuple(range(1, 18))
_LN_10 = np.float32(np.log(np.float32(10.0)))
PREPROCESSING_STATS_SCOPE_SOURCE_ALL = "all_scene_years"
PREPROCESSING_STATS_SCOPE_SOURCE_SPLIT = "split_manifest"


@dataclass(frozen=True, slots=True)
class SourceProcessingAudit:
    total_count: int
    input_valid_count: int
    excluded_by_valid_mask_count: int
    excluded_nonfinite_count: int
    excluded_domain_count: int
    output_valid_count: int

    def to_dict(self) -> dict[str, int]:
        return {
            "total_count": int(self.total_count),
            "input_valid_count": int(self.input_valid_count),
            "excluded_by_valid_mask_count": int(self.excluded_by_valid_mask_count),
            "excluded_nonfinite_count": int(self.excluded_nonfinite_count),
            "excluded_domain_count": int(self.excluded_domain_count),
            "output_valid_count": int(self.output_valid_count),
        }


@dataclass(slots=True)
class ProcessedSourceArrays:
    data: np.ndarray
    valid_mask: np.ndarray
    audit: SourceProcessingAudit
    category_counts: dict[str, int] | None = None


def build_preprocessing_rule_registry() -> dict[str, dict[str, Any]]:
    log_zscore_sources = {
        "BH",
        "BF",
        "DEM",
        "NL",
        "POP",
        "GAIA",
        "PM25",
        "PM25_mean",
        "Electricity",
        "GDP",
        "precip",
        "GHG_emission",
        "GHG_mean",
    }
    zscore_sources = {
        "NDVI",
        "LST",
        "Ta",
        "precip_bg",
        "wind",
    }

    registry: dict[str, dict[str, Any]] = {}

    def identity_restore() -> dict[str, Any]:
        return {"kind": "identity"}

    def affine_restore(scale: float, offset: float = 0.0) -> dict[str, Any]:
        return {"kind": "affine", "scale": float(scale), "offset": float(offset)}

    for source_name in log_zscore_sources:
        registry[source_name] = {
            "mode": "log_zscore",
            "restore": identity_restore(),
        }

    for source_name in zscore_sources:
        registry[source_name] = {
            "mode": "zscore",
            "restore": identity_restore(),
        }

    registry["NDVI"]["restore"] = affine_restore(scale=1.0 / 10000.0, offset=-1.0)
    registry["LST"]["restore"] = affine_restore(scale=1.0 / 50.0)

    registry["water"] = {
        "mode": "divide",
        "restore": identity_restore(),
        "divisor": 100.0,
    }

    for source_name in ("city_mask", "road_density", "Traffic"):
        registry[source_name] = {
            "mode": "pass_through",
            "restore": identity_restore(),
        }

    for source_name in ("LC", "LCZ"):
        registry[source_name] = {
            "mode": "one_hot",
            "restore": identity_restore(),
            "categories": list(_LC_CATEGORIES),
        }

    return registry


def _inverse_spec_for_rule(rule: dict[str, Any]) -> dict[str, Any]:
    mode = rule["mode"]
    if mode == "log_zscore":
        return {"kind": "zscore_then_pow10m1", "returns": "physical"}
    if mode == "zscore":
        return {"kind": "zscore_inverse", "returns": "physical"}
    if mode == "divide":
        return {"kind": "multiply", "factor": float(rule["divisor"]), "returns": "physical"}
    if mode == "pass_through":
        return {"kind": "identity", "returns": "physical"}
    if mode == "one_hot":
        return {
            "kind": "one_hot_argmax",
            "categories": list(rule["categories"]),
            "returns": "physical_category_id",
        }
    raise ValueError(f"Unsupported preprocessing mode {mode!r}")


def _output_channel_names(source_name: str, rule: dict[str, Any]) -> list[str]:
    if rule["mode"] == "one_hot":
        return [f"{source_name}_{category}" for category in rule["categories"]]
    return [source_name]


def _validate_rule_registry(source_specs: dict[str, dict[str, Any]], registry: dict[str, dict[str, Any]]) -> None:
    persisted_sources = set(source_specs)
    registry_sources = set(registry)
    missing = persisted_sources - registry_sources
    extra = registry_sources - persisted_sources
    if missing or extra:
        parts: list[str] = []
        if missing:
            parts.append(f"missing rule(s): {', '.join(sorted(missing))}")
        if extra:
            parts.append(f"unexpected rule(s): {', '.join(sorted(extra))}")
        raise ValueError(f"Preprocessing registry mismatch with processed schema: {'; '.join(parts)}")


def validate_preprocessing_spec(preprocessing_spec: dict[str, Any]) -> None:
    registry = build_preprocessing_rule_registry()
    source_specs = preprocessing_spec.get("sources")
    if not isinstance(source_specs, dict):
        raise ValueError("preprocessing spec must contain a 'sources' mapping")

    errors: list[str] = []
    for source_name, rule in registry.items():
        source_spec = source_specs.get(source_name)
        if source_spec is None:
            errors.append(f"{source_name}: missing from preprocessing spec")
            continue
        if source_spec.get("mode") != rule.get("mode"):
            errors.append(
                f"{source_name}: mode={source_spec.get('mode')!r}, expected {rule.get('mode')!r}"
            )
        if source_spec.get("restore") != rule.get("restore"):
            errors.append(
                f"{source_name}: restore={source_spec.get('restore')!r}, expected {rule.get('restore')!r}"
            )
        if source_spec.get("divisor") != rule.get("divisor"):
            errors.append(
                f"{source_name}: divisor={source_spec.get('divisor')!r}, expected {rule.get('divisor')!r}"
            )
        expected_categories = rule.get("categories")
        if expected_categories is not None and source_spec.get("categories") != expected_categories:
            errors.append(f"{source_name}: categories do not match current preprocessing registry")
        expected_channels = _output_channel_names(source_name, rule)
        if source_spec.get("output_channel_names") != expected_channels:
            errors.append(
                f"{source_name}: output_channel_names={source_spec.get('output_channel_names')!r}, "
                f"expected {expected_channels!r}"
            )

    extra_sources = sorted(set(source_specs) - set(registry))
    if extra_sources:
        errors.append(f"unexpected source(s): {', '.join(extra_sources)}")

    if errors:
        joined = "; ".join(errors)
        raise ValueError(
            "preprocessing_v1.json is out of sync with the current preprocessing registry. "
            "Regenerate it with `conda run -n EUMP python script/data/build_eump_preprocessing.py`. "
            f"Mismatch: {joined}"
        )


def _normalize_year_range(year_range: Any) -> tuple[int, int] | None:
    if year_range is None:
        return None
    if not isinstance(year_range, (list, tuple)) or len(year_range) != 2:
        raise ValueError(f"year_range must be null or a two-item sequence, got {year_range!r}")
    return (int(year_range[0]), int(year_range[1]))


def _scene_ids_digest(scene_ids: set[str] | list[str] | tuple[str, ...]) -> str:
    normalized = sorted(normalize_scene_id(scene_id) for scene_id in scene_ids)
    payload = "\n".join(normalized).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalize_stats_split(value: str) -> str:
    split = str(value).lower()
    if split not in {"all", "train", "val", "test"}:
        raise ValueError(f"preprocessing stats split must be one of 'all', 'train', 'val', or 'test', got {value!r}")
    return split


def validate_preprocessing_stats_scope(
    preprocessing_spec: dict[str, Any],
    *,
    split_manifest_path: str | Path | None = None,
    expected_year_range: tuple[int, int] | list[int] | None = None,
) -> None:
    """Ensure preprocessing stats were fitted from the configured train split."""
    if split_manifest_path is None:
        return

    scope = preprocessing_spec.get("stats_scope")
    if not isinstance(scope, dict):
        raise ValueError(
            "preprocessing_v1.json does not record a train-split stats_scope. "
            "Regenerate it after building the split manifest, for example with "
            "`bash script/data/run_eump_full_data_processing.sh split,preprocess,report`."
        )
    if scope.get("source") != PREPROCESSING_STATS_SCOPE_SOURCE_SPLIT or scope.get("split") != "train":
        raise ValueError(
            "preprocessing_v1.json was not standardized from the train split. "
            f"stats_scope={scope!r}. Regenerate preprocessing_v1.json with "
            "`script/data/build_eump_preprocessing.py --split-manifest-path <split.json> --stats-split train`."
        )

    split_manifest = load_split_manifest(
        split_manifest_path,
        expected_year_range=_normalize_year_range(expected_year_range),
    )
    train_scene_ids, _, _ = split_scene_ids_from_manifest(split_manifest)
    expected_digest = _scene_ids_digest(train_scene_ids)
    actual_digest = str(scope.get("scene_ids_digest", ""))
    if actual_digest != expected_digest:
        raise ValueError(
            "preprocessing_v1.json train-scene fingerprint does not match the configured split manifest. "
            "Regenerate preprocessing_v1.json from the current split manifest."
        )

    expected_range = _normalize_year_range(expected_year_range)
    scope_range = _normalize_year_range(scope.get("year_range"))
    if (
        expected_range is not None
        and (
            scope_range is None
            or not (int(scope_range[0]) <= int(expected_range[0]) and int(expected_range[1]) <= int(scope_range[1]))
        )
    ):
        raise ValueError(
            "preprocessing_v1.json year_range does not cover the configured pipeline. "
            f"preprocessing={scope_range!r} pipeline={expected_range!r}. "
            "Regenerate preprocessing_v1.json with the current split/year_range."
        )


def _apply_restore(data: np.ndarray, restore_spec: dict[str, Any]) -> np.ndarray:
    restore_kind = restore_spec["kind"]
    restored = data.astype(np.float32, copy=True)
    if restore_kind == "identity":
        return restored
    if restore_kind == "affine":
        restored *= np.float32(restore_spec["scale"])
        restored += np.float32(restore_spec.get("offset", 0.0))
        return restored
    raise ValueError(f"Unsupported restore kind {restore_kind!r}")


def _log10p1(values: np.ndarray) -> np.ndarray:
    values32 = values.astype(np.float32, copy=False)
    return np.log1p(values32) / _LN_10


def _pow10m1(values: np.ndarray) -> np.ndarray:
    values32 = values.astype(np.float32, copy=False)
    return np.expm1(values32 * _LN_10).astype(np.float32, copy=False)


def _build_processing_audit(
    total_count: int,
    input_valid_count: int,
    excluded_nonfinite_count: int,
    excluded_domain_count: int,
    output_valid_count: int,
) -> SourceProcessingAudit:
    excluded_by_valid_mask_count = int(total_count) - int(input_valid_count)
    return SourceProcessingAudit(
        total_count=int(total_count),
        input_valid_count=int(input_valid_count),
        excluded_by_valid_mask_count=excluded_by_valid_mask_count,
        excluded_nonfinite_count=int(excluded_nonfinite_count),
        excluded_domain_count=int(excluded_domain_count),
        output_valid_count=int(output_valid_count),
    )


def _prepare_numeric_values_with_audit(
    data: np.ndarray,
    valid_mask: np.ndarray,
    source_spec: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, SourceProcessingAudit]:
    restored = _apply_restore(data, source_spec["restore"])
    input_valid = np.asarray(valid_mask, dtype=bool)
    finite_mask = np.isfinite(restored)
    valid = input_valid & finite_mask
    mode = source_spec["mode"]

    excluded_nonfinite_count = int(np.count_nonzero(input_valid & ~finite_mask))
    excluded_domain_count = 0

    if mode == "log_zscore":
        domain_mask = restored > -1.0
        excluded_domain_count = int(np.count_nonzero(valid & ~domain_mask))
        valid &= domain_mask
        prepared = np.zeros_like(restored, dtype=np.float32)
        prepared[valid] = _log10p1(restored[valid])
    elif mode == "zscore":
        prepared = restored
    elif mode == "divide":
        prepared = np.zeros_like(restored, dtype=np.float32)
        prepared[valid] = restored[valid] / np.float32(source_spec["divisor"])
    elif mode == "pass_through":
        prepared = restored
    else:
        raise ValueError(f"Numeric value preparation is unsupported for mode={mode!r}")

    audit = _build_processing_audit(
        total_count=data.size,
        input_valid_count=int(np.count_nonzero(input_valid)),
        excluded_nonfinite_count=excluded_nonfinite_count,
        excluded_domain_count=excluded_domain_count,
        output_valid_count=int(np.count_nonzero(valid)),
    )
    return prepared, valid, audit


def _prepare_numeric_values(data: np.ndarray, valid_mask: np.ndarray, source_spec: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    prepared, valid, _ = _prepare_numeric_values_with_audit(data, valid_mask, source_spec)
    return prepared, valid


def _prepare_physical_numeric_values_with_audit(
    data: np.ndarray,
    valid_mask: np.ndarray,
    source_spec: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, SourceProcessingAudit]:
    physical = np.asarray(data).astype(np.float32, copy=True)
    input_valid = np.asarray(valid_mask, dtype=bool)
    finite_mask = np.isfinite(physical)
    valid = input_valid & finite_mask
    mode = source_spec["mode"]

    excluded_nonfinite_count = int(np.count_nonzero(input_valid & ~finite_mask))
    excluded_domain_count = 0

    if mode == "log_zscore":
        domain_mask = physical > -1.0
        excluded_domain_count = int(np.count_nonzero(valid & ~domain_mask))
        valid &= domain_mask
        prepared = np.zeros_like(physical, dtype=np.float32)
        prepared[valid] = _log10p1(physical[valid])
    elif mode == "zscore":
        prepared = physical
    elif mode == "divide":
        prepared = np.zeros_like(physical, dtype=np.float32)
        prepared[valid] = physical[valid] / np.float32(source_spec["divisor"])
    elif mode == "pass_through":
        prepared = physical
    else:
        raise ValueError(f"Physical scalar value preparation is unsupported for mode={mode!r}")

    audit = _build_processing_audit(
        total_count=physical.size,
        input_valid_count=int(np.count_nonzero(input_valid)),
        excluded_nonfinite_count=excluded_nonfinite_count,
        excluded_domain_count=excluded_domain_count,
        output_valid_count=int(np.count_nonzero(valid)),
    )
    return prepared, valid, audit


def physical_zero_mask_for_source_arrays(
    data: np.ndarray,
    valid_mask: np.ndarray,
    source_spec: dict[str, Any],
) -> np.ndarray:
    """Return valid, finite positions whose restored physical value is zero."""
    restored = _apply_restore(np.asarray(data), source_spec["restore"])
    raw_valid_mask = np.asarray(valid_mask, dtype=bool)
    return raw_valid_mask & np.isfinite(restored) & (restored == 0)


def process_source_arrays(
    data: np.ndarray,
    valid_mask: np.ndarray,
    source_spec: dict[str, Any],
    *,
    stats_key: str = "stats",
) -> ProcessedSourceArrays:
    source_name = str(source_spec.get("source_name", ""))
    raw_data = np.asarray(data)
    raw_valid_mask = np.asarray(valid_mask, dtype=bool)
    if raw_data.shape != raw_valid_mask.shape:
        raise ValueError(
            f"Source {source_name!r} data shape {raw_data.shape} does not match valid_mask shape {raw_valid_mask.shape}"
        )

    mode = source_spec["mode"]
    if mode == "one_hot":
        categories = [int(category) for category in source_spec["categories"]]
        if raw_data.ndim != 3 or raw_data.shape[0] != 1:
            raise ValueError(
                f"One-hot processing expects source {source_name!r} to have shape [1, H, W], got {raw_data.shape}"
            )
        restored = _apply_restore(raw_data, source_spec["restore"])
        valid = raw_valid_mask.astype(bool, copy=False) & np.isfinite(restored)
        output = np.zeros((len(categories), *raw_data.shape[-2:]), dtype=np.float32)
        expanded_valid = np.broadcast_to(valid[:1], output.shape).copy()
        category_counts = {str(int(category)): 0 for category in categories}
        if np.any(valid):
            valid_values = restored[0][valid[0]]
            for idx, category in enumerate(categories):
                matches = valid_values == category
                category_counts[str(int(category))] = int(np.count_nonzero(matches))
                output[idx][valid[0]] = matches.astype(np.float32, copy=False)
        audit = _build_processing_audit(
            total_count=raw_data.size,
            input_valid_count=int(np.count_nonzero(raw_valid_mask)),
            excluded_nonfinite_count=int(np.count_nonzero(raw_valid_mask & ~np.isfinite(restored))),
            excluded_domain_count=0,
            output_valid_count=int(np.count_nonzero(valid)),
        )
        return ProcessedSourceArrays(
            data=output,
            valid_mask=expanded_valid,
            audit=audit,
            category_counts=category_counts,
        )

    prepared, valid, audit = _prepare_numeric_values_with_audit(raw_data, raw_valid_mask, source_spec)
    if mode in {"log_zscore", "zscore"}:
        stats = source_spec.get(stats_key)
        if not stats:
            raise ValueError(f"Missing preprocessing stats {stats_key!r} for source {source_name!r}")
        output = np.zeros_like(prepared, dtype=np.float32)
        output[valid] = (prepared[valid] - np.float32(stats["mean"])) / np.float32(stats["std"])
        prepared = output
    prepared = prepared.astype(np.float32, copy=False)
    prepared[~valid] = 0.0
    return ProcessedSourceArrays(
        data=prepared,
        valid_mask=valid,
        audit=audit,
        category_counts=None,
    )


def _make_numeric_accumulator() -> dict[str, Any]:
    return {
        "kind": "numeric",
        "count": 0,
        "sum": 0.0,
        "sum_sq": 0.0,
        "min": None,
        "max": None,
    }


def _make_one_hot_accumulator(categories: list[int]) -> dict[str, Any]:
    return {
        "kind": "one_hot",
        "valid_count": 0,
        "category_counts": {str(int(category)): 0 for category in categories},
    }


def _make_accumulator_for_rule(rule: dict[str, Any]) -> dict[str, Any]:
    if rule["mode"] == "one_hot":
        return _make_one_hot_accumulator(rule["categories"])
    return _make_numeric_accumulator()


def _merge_numeric_accumulator(target: dict[str, Any], source: dict[str, Any]) -> None:
    count = int(source["count"])
    if count <= 0:
        return
    target["count"] += count
    target["sum"] += float(source["sum"])
    target["sum_sq"] += float(source["sum_sq"])
    source_min = source.get("min")
    source_max = source.get("max")
    if source_min is not None:
        target["min"] = float(source_min) if target["min"] is None else min(float(target["min"]), float(source_min))
    if source_max is not None:
        target["max"] = float(source_max) if target["max"] is None else max(float(target["max"]), float(source_max))


def _merge_one_hot_accumulator(target: dict[str, Any], source: dict[str, Any]) -> None:
    target["valid_count"] += int(source["valid_count"])
    for category, count in source["category_counts"].items():
        target["category_counts"][str(category)] += int(count)


def _merge_accumulator(target: dict[str, Any], source: dict[str, Any]) -> None:
    if target["kind"] != source["kind"]:
        raise ValueError(f"Cannot merge accumulator kind {source['kind']!r} into {target['kind']!r}")
    if target["kind"] == "numeric":
        _merge_numeric_accumulator(target, source)
        return
    _merge_one_hot_accumulator(target, source)


def _update_numeric_accumulator(accumulator: dict[str, Any], values: np.ndarray) -> None:
    if values.size == 0:
        return
    values64 = values.astype(np.float64, copy=False)
    values64 = values64[np.isfinite(values64)]
    if values64.size == 0:
        return
    accumulator["count"] += int(values64.size)
    accumulator["sum"] += float(values64.sum())
    accumulator["sum_sq"] += float(np.square(values64).sum())
    current_min = float(values64.min())
    current_max = float(values64.max())
    accumulator["min"] = current_min if accumulator["min"] is None else min(accumulator["min"], current_min)
    accumulator["max"] = current_max if accumulator["max"] is None else max(accumulator["max"], current_max)


def _update_one_hot_accumulator(accumulator: dict[str, Any], data: np.ndarray, valid_mask: np.ndarray, categories: list[int]) -> None:
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(data)
    if not np.any(valid):
        return
    valid_values = data[valid].astype(np.int64, copy=False)
    accumulator["valid_count"] += int(valid_values.size)
    for category in categories:
        accumulator["category_counts"][str(int(category))] += int(np.count_nonzero(valid_values == int(category)))


def _finalize_numeric_stats(accumulator: dict[str, Any]) -> dict[str, Any] | None:
    count = int(accumulator["count"])
    if count == 0:
        return None
    mean = accumulator["sum"] / count
    variance = max(accumulator["sum_sq"] / count - mean ** 2, 0.0)
    std = float(np.sqrt(variance))
    if not np.isfinite(std) or std <= 0:
        std = 1.0
    return {
        "count": count,
        "mean": float(mean),
        "std": std,
        "min": accumulator["min"],
        "max": accumulator["max"],
    }


def _finalize_accumulator(accumulator: dict[str, Any]) -> dict[str, Any]:
    if accumulator["kind"] == "numeric":
        return {
            "stats": _finalize_numeric_stats(accumulator),
            "category_counts": None,
            "valid_count": int(accumulator["count"]),
        }
    return {
        "stats": None,
        "category_counts": dict(accumulator["category_counts"]),
        "valid_count": int(accumulator["valid_count"]),
    }


def _compute_group_preprocessing_accumulators(
    eump_root: str,
    relative_path: str,
    rules: dict[str, dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    zarr = import_zarr()
    group = zarr.open_group(str(Path(eump_root) / relative_path), mode="r")
    accumulators: dict[str, dict[str, Any]] = {}
    zero_excluded_accumulators: dict[str, dict[str, Any]] = {}

    for source_name in group.group_keys():
        rule = rules[source_name]
        data = np.asarray(group[source_name]["data"])
        valid_mask = np.asarray(group[source_name]["valid_mask"], dtype=bool)

        accumulator = _make_accumulator_for_rule(rule)
        if rule["mode"] == "one_hot":
            _update_one_hot_accumulator(accumulator, data, valid_mask, rule["categories"])
        else:
            prepared, valid = _prepare_numeric_values(data, valid_mask, rule)
            _update_numeric_accumulator(accumulator, prepared[valid])
        accumulators[source_name] = accumulator

        if rule["mode"] in {"log_zscore", "zscore"}:
            zero_accumulator = _make_numeric_accumulator()
            prepared, valid = _prepare_numeric_values(data, valid_mask, rule)
            zero_mask = physical_zero_mask_for_source_arrays(data, valid_mask, rule)
            _update_numeric_accumulator(zero_accumulator, prepared[valid & ~zero_mask])
            zero_excluded_accumulators[source_name] = zero_accumulator

    return {
        "accumulators": accumulators,
        "zero_excluded_accumulators": zero_excluded_accumulators,
    }


def _apply_forward_raster(block: RasterBlock, source_spec: dict[str, Any]) -> RasterBlock:
    processed = process_source_arrays(block.data, block.valid_mask, source_spec)
    return RasterBlock(
        data=processed.data,
        valid_mask=processed.valid_mask,
        channel_names=tuple(source_spec["output_channel_names"]),
    )


def _apply_forward_scalar(block: ScalarBlock, source_spec: dict[str, Any], *, stats_key: str = "stats") -> ScalarBlock:
    if source_spec["mode"] == "one_hot":
        raise ValueError(f"One-hot processing is unsupported for scalar source {source_spec['source_name']!r}")
    processed = process_source_arrays(block.data, block.valid_mask, source_spec, stats_key=stats_key)
    return ScalarBlock(
        data=processed.data,
        valid_mask=processed.valid_mask,
        names=tuple(source_spec["output_channel_names"]),
        support=None if block.support is None else block.support.copy(),
    )


def process_physical_scalar_block(
    block: ScalarBlock,
    source_spec: dict[str, Any],
    *,
    stats_key: str = "stats",
) -> ScalarBlock:
    """Apply a processed source's forward preprocessing rule to scalar values."""
    if source_spec["mode"] == "one_hot":
        raise ValueError(f"One-hot processing is unsupported for physical scalar source {source_spec['source_name']!r}")

    prepared, valid, _ = _prepare_physical_numeric_values_with_audit(block.data, block.valid_mask, source_spec)
    mode = source_spec["mode"]
    if mode in {"log_zscore", "zscore"}:
        stats = source_spec.get(stats_key)
        if not stats:
            raise ValueError(f"Missing preprocessing stats {stats_key!r} for source {source_spec['source_name']!r}")
        output = np.zeros_like(prepared, dtype=np.float32)
        output[valid] = (prepared[valid] - np.float32(stats["mean"])) / np.float32(stats["std"])
        prepared = output
    prepared = prepared.astype(np.float32, copy=False)
    prepared[~valid] = 0.0
    return ScalarBlock(
        data=prepared,
        valid_mask=valid,
        names=tuple(source_spec["output_channel_names"]),
        support=None if block.support is None else block.support.copy(),
    )


def _apply_inverse_raster(block: RasterBlock, source_spec: dict[str, Any]) -> RasterBlock:
    mode = source_spec["mode"]
    if mode == "one_hot":
        categories = [int(category) for category in source_spec["categories"]]
        if block.data.shape[0] != len(categories):
            raise ValueError(
                f"One-hot raster restore expects {len(categories)} channels for {source_spec['source_name']!r}, "
                f"got {block.data.shape[0]}"
            )
        restored = np.zeros((1, *block.data.shape[-2:]), dtype=np.float32)
        valid = np.any(block.valid_mask, axis=0, keepdims=True)
        if np.any(valid):
            indices = np.argmax(block.data, axis=0)
            category_values = np.asarray(categories, dtype=np.float32)
            restored[0][valid[0]] = category_values[indices[valid[0]]]
        return RasterBlock(
            data=restored,
            valid_mask=valid,
            channel_names=(source_spec["source_name"],),
        )

    restored = np.zeros_like(block.data, dtype=np.float32)
    valid = block.valid_mask.astype(bool, copy=False)
    if mode in {"log_zscore", "zscore"}:
        stats = source_spec.get("stats")
        if not stats:
            raise ValueError(f"Missing preprocessing stats for source {source_spec['source_name']!r}")
        restored[valid] = block.data[valid] * np.float32(stats["std"]) + np.float32(stats["mean"])
        if mode == "log_zscore":
            restored[valid] = _pow10m1(restored[valid])
    elif mode == "divide":
        restored[valid] = block.data[valid] * np.float32(source_spec["divisor"])
    elif mode == "pass_through":
        restored[valid] = block.data[valid]
    else:
        raise ValueError(f"Unsupported preprocessing mode {mode!r}")

    return RasterBlock(
        data=restored,
        valid_mask=valid,
        channel_names=(source_spec["source_name"],),
    )


def restore_processed_raster_block(block: RasterBlock, source_spec: dict[str, Any]) -> RasterBlock:
    """Restore a processed raster block to its physical source scale."""
    return _apply_inverse_raster(block, source_spec)


def _apply_inverse_scalar(block: ScalarBlock, source_spec: dict[str, Any]) -> ScalarBlock:
    mode = source_spec["mode"]
    restored = np.zeros_like(block.data, dtype=np.float32)
    valid = block.valid_mask.astype(bool, copy=False)
    if mode in {"log_zscore", "zscore"}:
        stats = source_spec.get("stats")
        if not stats:
            raise ValueError(f"Missing preprocessing stats for source {source_spec['source_name']!r}")
        restored[valid] = block.data[valid] * np.float32(stats["std"]) + np.float32(stats["mean"])
        if mode == "log_zscore":
            restored[valid] = _pow10m1(restored[valid])
    elif mode == "divide":
        restored[valid] = block.data[valid] * np.float32(source_spec["divisor"])
    elif mode == "pass_through":
        restored[valid] = block.data[valid]
    else:
        raise ValueError(f"Unsupported preprocessing mode {mode!r} for scalar restore")

    return ScalarBlock(
        data=restored,
        valid_mask=valid,
        names=(source_spec["source_name"],),
        support=None if block.support is None else block.support.copy(),
    )


def _split_raster_block(block: RasterBlock | None) -> dict[str, RasterBlock]:
    if block is None or block.channel_names is None:
        return {}
    split: dict[str, RasterBlock] = {}
    for idx, channel_name in enumerate(block.channel_names):
        split[str(channel_name)] = RasterBlock(
            data=block.data[idx : idx + 1].astype(np.float32, copy=False),
            valid_mask=block.valid_mask[idx : idx + 1],
            channel_names=(str(channel_name),),
        )
    return split


def _split_scalar_block(block: ScalarBlock | None) -> dict[str, ScalarBlock]:
    if block is None or block.names is None:
        return {}
    split: dict[str, ScalarBlock] = {}
    for idx, name in enumerate(block.names):
        support = None if block.support is None else block.support[idx : idx + 1]
        split[str(name)] = ScalarBlock(
            data=block.data[idx : idx + 1].astype(np.float32, copy=False),
            valid_mask=block.valid_mask[idx : idx + 1],
            names=(str(name),),
            support=support,
        )
    return split


def _build_source_lookup(scene_record: SceneRecord) -> dict[str, RasterBlock | ScalarBlock]:
    lookup: dict[str, RasterBlock | ScalarBlock] = {}
    lookup.update(_split_raster_block(scene_record.inputs.static_raster))
    lookup.update(_split_raster_block(scene_record.inputs.dynamic_raster))
    lookup.update(_split_scalar_block(scene_record.inputs.scalar))

    if scene_record.targets.reconstruct_raster is not None:
        lookup.update(_split_raster_block(scene_record.targets.reconstruct_raster))

    lookup.update(scene_record.targets.aux_raster)
    lookup.update(scene_record.targets.aux_scalar)
    lookup.update(scene_record.targets.prediction_raster)
    lookup.update(scene_record.targets.prediction_scalar)
    return lookup


def _subset_raster_block_by_names(block: RasterBlock | None, channel_names: list[str]) -> RasterBlock | None:
    if block is None or block.channel_names is None:
        return None
    index = {str(name): idx for idx, name in enumerate(block.channel_names)}
    if any(channel_name not in index for channel_name in channel_names):
        return None
    indices = [index[channel_name] for channel_name in channel_names]
    return RasterBlock(
        data=block.data[indices].astype(np.float32, copy=False),
        valid_mask=block.valid_mask[indices],
        channel_names=tuple(channel_names),
    )


def _lookup_processed_source_block(
    scene_record: SceneRecord,
    source_name: str,
    source_spec: dict[str, Any],
) -> RasterBlock | ScalarBlock | None:
    if source_spec["mode"] != "one_hot":
        return None
    output_channel_names = list(source_spec["output_channel_names"])

    for block in (
        scene_record.inputs.static_raster,
        scene_record.inputs.dynamic_raster,
        scene_record.targets.reconstruct_raster,
    ):
        subset = _subset_raster_block_by_names(block, output_channel_names)
        if subset is not None:
            return subset

    for mapping in (scene_record.targets.aux_raster, scene_record.targets.prediction_raster):
        if source_name in mapping:
            return mapping[source_name]
    return None


def _transform_scene_record(
    scene_record: SceneRecord,
    task_config: ProcessedTaskConfig,
    preprocessing_spec: dict[str, Any],
    forward: bool,
) -> SceneRecord:
    source_specs = preprocessing_spec["sources"]
    source_lookup = _build_source_lookup(scene_record)
    transformed_cache: dict[str, RasterBlock | ScalarBlock] = {}

    def transform_source(source_name: str) -> RasterBlock | ScalarBlock:
        if source_name in transformed_cache:
            return transformed_cache[source_name]
        if source_name not in source_specs:
            raise KeyError(f"Source {source_name!r} is missing from preprocessing spec")
        source_spec = source_specs[source_name]
        raw_block = source_lookup.get(source_name)
        if raw_block is None and not forward:
            raw_block = _lookup_processed_source_block(scene_record, source_name, source_spec)
        if raw_block is None:
            raise KeyError(f"Source {source_name!r} is unavailable in SceneRecord")
        if isinstance(raw_block, RasterBlock):
            transformed = _apply_forward_raster(raw_block, source_spec) if forward else _apply_inverse_raster(raw_block, source_spec)
        else:
            transformed = _apply_forward_scalar(raw_block, source_spec) if forward else _apply_inverse_scalar(raw_block, source_spec)
        transformed_cache[source_name] = transformed
        return transformed

    static_blocks: list[tuple[np.ndarray, np.ndarray, tuple[str, ...]]] = []
    dynamic_blocks: list[tuple[np.ndarray, np.ndarray, tuple[str, ...]]] = []
    scalar_blocks: list[tuple[np.ndarray, np.ndarray, tuple[str, ...]]] = []

    for source_name in task_config.input_sources:
        if source_name == task_config.city_mask_source:
            continue
        transformed = transform_source(source_name)
        source_spec = source_specs[source_name]
        if isinstance(transformed, RasterBlock):
            bucket = static_blocks if source_spec["storage_level"] == "static" else dynamic_blocks
            bucket.append((transformed.data, transformed.valid_mask, transformed.channel_names or (source_name,)))
        else:
            scalar_blocks.append((transformed.data, transformed.valid_mask, transformed.names or (source_name,)))

    reconstruct_raster = None
    if task_config.reconstruct_target_source is not None:
        transformed = transform_source(task_config.reconstruct_target_source)
        if not isinstance(transformed, RasterBlock):
            raise ValueError(
                f"reconstruct_target_source={task_config.reconstruct_target_source!r} must resolve to a raster block"
            )
        reconstruct_raster = transformed

    aux_raster: dict[str, RasterBlock] = {}
    aux_scalar: dict[str, ScalarBlock] = {}
    for source_name in task_config.aux_target_sources:
        transformed = transform_source(source_name)
        if isinstance(transformed, RasterBlock):
            aux_raster[source_name] = transformed
        else:
            aux_scalar[source_name] = transformed
    for source_name in task_config.optional_aux_target_sources:
        if source_name not in source_lookup:
            source_spec = source_specs.get(source_name)
            if source_spec is None:
                transform_source(source_name)
            if forward or _lookup_processed_source_block(scene_record, source_name, source_spec) is None:
                continue
        transformed = transform_source(source_name)
        if isinstance(transformed, RasterBlock):
            aux_raster[source_name] = transformed
        else:
            aux_scalar[source_name] = transformed

    prediction_raster: dict[str, RasterBlock] = {}
    prediction_scalar: dict[str, ScalarBlock] = {}
    for source_name in task_config.prediction_sources:
        transformed = transform_source(source_name)
        if isinstance(transformed, RasterBlock):
            prediction_raster[source_name] = transformed
        else:
            prediction_scalar[source_name] = transformed

    next_meta = dict(scene_record.meta)
    next_meta["preprocessing_version"] = preprocessing_spec["preprocessing_version"]
    next_meta["processed_state"] = "normalized" if forward else "physical_scale"

    return SceneRecord(
        scene_id=scene_record.scene_id,
        time_id=scene_record.time_id,
        inputs=SceneInputs(
            static_raster=concat_raster_blocks(static_blocks),
            dynamic_raster=concat_raster_blocks(dynamic_blocks),
            scalar=concat_scalar_blocks(scalar_blocks),
            city_mask=None if scene_record.inputs.city_mask is None else scene_record.inputs.city_mask.astype(np.float32, copy=True),
        ),
        targets=SceneTargets(
            reconstruct_raster=reconstruct_raster,
            aux_raster=aux_raster,
            aux_scalar=aux_scalar,
            prediction_raster=prediction_raster,
            prediction_scalar=prediction_scalar,
            crop_scalar_specs=scene_record.targets.crop_scalar_specs.copy(),
        ),
        meta=next_meta,
    )


def process_scene_record(
    scene_record: SceneRecord,
    task_config: ProcessedTaskConfig,
    preprocessing_spec: dict[str, Any],
) -> SceneRecord:
    return _transform_scene_record(scene_record, task_config, preprocessing_spec, forward=True)


def restore_scene_record(
    scene_record: SceneRecord,
    task_config: ProcessedTaskConfig,
    preprocessing_spec: dict[str, Any],
) -> SceneRecord:
    return _transform_scene_record(scene_record, task_config, preprocessing_spec, forward=False)


def load_preprocessing_spec(path_or_root: str | Path = EUMP_DATA_DIR) -> dict[str, Any]:
    candidate = Path(path_or_root)
    if candidate.is_dir():
        candidate = preprocessing_stats_path(candidate)
    return read_json(candidate)


class ProcessedPreprocessingBuilder:
    def __init__(
        self,
        eump_root: str | Path = EUMP_DATA_DIR,
        output_path: str | Path | None = None,
        *,
        split_manifest_path: str | Path | None = None,
        stats_split: str = "all",
        year_range: tuple[int, int] | list[int] | None = None,
        city_crop_size: int = 256,
        city_crop_samples_per_scene_year: int = 0,
        city_crop_min_city_mask_coverage: float = 0.03,
        city_crop_random_seed: int = 42,
        num_workers: int = 0,
    ) -> None:
        self.eump_root = Path(eump_root)
        self.output_path = preprocessing_stats_path(self.eump_root) if output_path is None else Path(output_path)
        self.schema = load_eump_store_schema(self.eump_root)
        self.source_specs = self.schema["sources"]
        self.rules = build_preprocessing_rule_registry()
        _validate_rule_registry(self.source_specs, self.rules)
        self.scene_rows = read_jsonl(scenes_manifest_path(self.eump_root))
        self.scene_year_rows = read_jsonl(scene_years_manifest_path(self.eump_root))
        self.scene_index = {str(row["scene_id"]): row for row in self.scene_rows}
        self.split_manifest_path = None if split_manifest_path is None else Path(split_manifest_path)
        self.stats_split = _normalize_stats_split(stats_split)
        self.stats_year_range = _normalize_year_range(year_range)
        self.stats_scene_ids: set[str] | None = None
        self.split_manifest: dict[str, Any] | None = None
        if self.split_manifest_path is not None:
            self.split_manifest = load_split_manifest(
                self.split_manifest_path,
                available_scene_ids=self.scene_index.keys(),
                expected_year_range=self.stats_year_range,
            )
            if self.stats_year_range is None:
                self.stats_year_range = _normalize_year_range(self.split_manifest.get("year_range"))
            train_ids, val_ids, test_ids = split_scene_ids_from_manifest(self.split_manifest)
            split_ids = {
                "train": train_ids,
                "val": val_ids,
                "test": test_ids,
                "all": train_ids | val_ids | test_ids,
            }
            self.stats_scene_ids = set(split_ids[self.stats_split])
        elif self.stats_split != "all":
            raise ValueError("A split_manifest_path is required when stats_split is not 'all'")
        self.city_crop_size = int(city_crop_size)
        self.city_crop_samples_per_scene_year = int(city_crop_samples_per_scene_year)
        self.city_crop_min_city_mask_coverage = float(city_crop_min_city_mask_coverage)
        self.city_crop_random_seed = int(city_crop_random_seed)
        self.num_workers = int(num_workers)

    def _open_group(self, relative_path: str):
        zarr = import_zarr()
        return zarr.open_group(str(self.eump_root / relative_path), mode="r")

    def _make_accumulators(self) -> dict[str, dict[str, Any]]:
        accumulators: dict[str, dict[str, Any]] = {}
        for source_name, rule in self.rules.items():
            accumulators[source_name] = _make_accumulator_for_rule(rule)
        return accumulators

    def _make_city_crop_accumulators(self) -> dict[str, dict[str, Any]]:
        accumulators: dict[str, dict[str, Any]] = {}
        for source_name, source_schema in self.source_specs.items():
            rule = self.rules[source_name]
            if source_schema["modality"] != "raster" or rule["mode"] not in {"log_zscore", "zscore"}:
                continue
            accumulators[source_name] = _make_numeric_accumulator()
        return accumulators

    def _make_zero_excluded_accumulators(self) -> dict[str, dict[str, Any]]:
        accumulators: dict[str, dict[str, Any]] = {}
        for source_name, rule in self.rules.items():
            if rule["mode"] in {"log_zscore", "zscore"}:
                accumulators[source_name] = _make_numeric_accumulator()
        return accumulators

    def _stats_scene_rows(self) -> list[dict[str, Any]]:
        stats_scene_ids = getattr(self, "stats_scene_ids", None)
        rows = list(self.scene_rows)
        if stats_scene_ids is None:
            return rows
        return [row for row in rows if normalize_scene_id(row["scene_id"]) in stats_scene_ids]

    def _stats_scene_year_rows(self) -> list[dict[str, Any]]:
        stats_scene_ids = getattr(self, "stats_scene_ids", None)
        stats_year_range = getattr(self, "stats_year_range", None)
        rows = list(self.scene_year_rows)
        if stats_scene_ids is not None:
            rows = [row for row in rows if normalize_scene_id(row["scene_id"]) in stats_scene_ids]
        if stats_year_range is not None:
            start_year, end_year = stats_year_range
            rows = [row for row in rows if start_year <= int(row["year"]) <= end_year]
        return rows

    def _stats_scope_payload(
        self,
        *,
        scene_rows: list[dict[str, Any]],
        scene_year_rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        stats_scene_ids = getattr(self, "stats_scene_ids", None)
        stats_year_range = getattr(self, "stats_year_range", None)
        stats_split = getattr(self, "stats_split", "all")
        split_manifest = getattr(self, "split_manifest", None)
        if stats_scene_ids is None:
            scene_id_values = {
                normalize_scene_id(row["scene_id"])
                for row in scene_rows
                if "scene_id" in row
            }
            scope = {
                "source": PREPROCESSING_STATS_SCOPE_SOURCE_ALL,
                "split": "all",
                "year_range": None
                if stats_year_range is None
                else [int(stats_year_range[0]), int(stats_year_range[1])],
                "scene_count": int(len(scene_rows)),
                "scene_year_count": int(len(scene_year_rows)),
                "scene_ids_digest": None if not scene_id_values else _scene_ids_digest(scene_id_values),
            }
        else:
            scope = {
                "source": PREPROCESSING_STATS_SCOPE_SOURCE_SPLIT,
                "split": stats_split,
                "year_range": None
                if stats_year_range is None
                else [int(stats_year_range[0]), int(stats_year_range[1])],
                "scene_count": int(len(scene_rows)),
                "scene_year_count": int(len(scene_year_rows)),
                "scene_ids_digest": _scene_ids_digest(stats_scene_ids),
            }
            if getattr(self, "split_manifest_path", None) is not None:
                scope["split_manifest_path"] = str(Path(self.split_manifest_path))
            if split_manifest is not None:
                scope["split_manifest_version"] = split_manifest.get("version")
                scope["split_manifest_generator"] = split_manifest.get("generator")
                scope["split_manifest_seed"] = split_manifest.get("seed")
                scope["split_manifest_test_size"] = split_manifest.get("test_size")
        return scope

    def _update_source_accumulator(
        self,
        accumulators: dict[str, dict[str, Any]],
        source_name: str,
        data: np.ndarray,
        valid_mask: np.ndarray,
    ) -> None:
        rule = self.rules[source_name]
        accumulator = accumulators[source_name]
        if rule["mode"] == "one_hot":
            _update_one_hot_accumulator(accumulator, data, valid_mask, rule["categories"])
            return
        prepared, valid = _prepare_numeric_values(np.asarray(data), np.asarray(valid_mask, dtype=bool), rule)
        _update_numeric_accumulator(accumulator, prepared[valid])

    def _update_zero_excluded_accumulator(
        self,
        accumulators: dict[str, dict[str, Any]],
        source_name: str,
        data: np.ndarray,
        valid_mask: np.ndarray,
    ) -> None:
        if source_name not in accumulators:
            return
        rule = self.rules[source_name]
        prepared, valid = _prepare_numeric_values(np.asarray(data), np.asarray(valid_mask, dtype=bool), rule)
        zero_mask = physical_zero_mask_for_source_arrays(data, valid_mask, rule)
        _update_numeric_accumulator(accumulators[source_name], prepared[valid & ~zero_mask])

    def _resolve_worker_count(self, row_count: int) -> int:
        if row_count <= 1 or self.num_workers <= 0:
            return 1
        return max(1, min(int(self.num_workers), int(row_count)))

    def _merge_group_stats(
        self,
        accumulators: dict[str, dict[str, Any]],
        zero_excluded_accumulators: dict[str, dict[str, Any]],
        payload: dict[str, dict[str, dict[str, Any]]],
    ) -> None:
        for source_name, accumulator in payload["accumulators"].items():
            _merge_accumulator(accumulators[source_name], accumulator)
        for source_name, accumulator in payload["zero_excluded_accumulators"].items():
            _merge_accumulator(zero_excluded_accumulators[source_name], accumulator)

    def _update_group_stats_serial(
        self,
        rows: list[dict[str, Any]],
        path_key: str,
        *,
        desc: str,
        unit: str,
        accumulators: dict[str, dict[str, Any]],
        zero_excluded_accumulators: dict[str, dict[str, Any]],
    ) -> None:
        for row in tqdm(rows, desc=desc, unit=unit):
            group = self._open_group(row[path_key])
            for source_name in group.group_keys():
                data = np.asarray(group[source_name]["data"])
                valid_mask = np.asarray(group[source_name]["valid_mask"], dtype=bool)
                self._update_source_accumulator(
                    accumulators,
                    source_name,
                    data,
                    valid_mask,
                )
                self._update_zero_excluded_accumulator(
                    zero_excluded_accumulators,
                    source_name,
                    data,
                    valid_mask,
                )

    def _update_group_stats_parallel(
        self,
        rows: list[dict[str, Any]],
        path_key: str,
        *,
        desc: str,
        unit: str,
        accumulators: dict[str, dict[str, Any]],
        zero_excluded_accumulators: dict[str, dict[str, Any]],
    ) -> bool:
        worker_count = self._resolve_worker_count(len(rows))
        if worker_count <= 1:
            return False

        try:
            with ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=mp.get_context("spawn"),
            ) as executor:
                future_map = {
                    executor.submit(
                        _compute_group_preprocessing_accumulators,
                        str(self.eump_root),
                        str(row[path_key]),
                        self.rules,
                    ): row
                    for row in rows
                }
                for future in tqdm(
                    as_completed(future_map),
                    total=len(future_map),
                    desc=f"{desc} ({worker_count} workers)",
                    unit=unit,
                ):
                    self._merge_group_stats(
                        accumulators,
                        zero_excluded_accumulators,
                        future.result(),
                    )
        except (OSError, PermissionError):
            return False
        return True

    def _update_group_stats(
        self,
        rows: list[dict[str, Any]],
        path_key: str,
        *,
        desc: str,
        unit: str,
        accumulators: dict[str, dict[str, Any]],
        zero_excluded_accumulators: dict[str, dict[str, Any]],
    ) -> None:
        if not self._update_group_stats_parallel(
            rows,
            path_key,
            desc=desc,
            unit=unit,
            accumulators=accumulators,
            zero_excluded_accumulators=zero_excluded_accumulators,
        ):
            self._update_group_stats_serial(
                rows,
                path_key,
                desc=desc,
                unit=unit,
                accumulators=accumulators,
                zero_excluded_accumulators=zero_excluded_accumulators,
            )

    def _city_crop_box(self, spatial_shape: tuple[int, int], rng: np.random.Generator):
        from urban_embedding.data.crop import sample_random_crop_box

        height, width = spatial_shape
        configured_size = int(getattr(self, "city_crop_size", 256))
        crop_size = (min(configured_size, int(height)), min(configured_size, int(width)))
        return sample_random_crop_box(spatial_shape, crop_size, center_ratio=0.0, rng=rng)

    def _update_city_crop_source_accumulator(
        self,
        accumulator: dict[str, Any],
        source_name: str,
        data: np.ndarray,
        valid_mask: np.ndarray,
        city_mask: np.ndarray,
        crop_box,
    ) -> None:
        rule = self.rules[source_name]
        restored = _apply_restore(
            data[:, crop_box.top : crop_box.bottom, crop_box.left : crop_box.right],
            rule["restore"],
        )
        source_valid = np.asarray(
            valid_mask[:, crop_box.top : crop_box.bottom, crop_box.left : crop_box.right],
            dtype=bool,
        )
        city = city_mask[:, crop_box.top : crop_box.bottom, crop_box.left : crop_box.right] > 0
        city = np.broadcast_to(city, restored.shape)
        valid = source_valid & city & np.isfinite(restored) & (restored > 0)
        positive_count = valid.reshape(restored.shape[0], -1).sum(axis=1)
        if not np.any(positive_count > 0):
            return
        safe_restored = np.where(valid, restored, 0.0).astype(np.float32, copy=False)
        positive_sum = safe_restored.reshape(restored.shape[0], -1).sum(axis=1)
        physical_mean = np.zeros((restored.shape[0],), dtype=np.float32)
        physical_mean[positive_count > 0] = positive_sum[positive_count > 0] / positive_count[positive_count > 0]
        if rule["mode"] == "log_zscore":
            prepared = _log10p1(physical_mean[positive_count > 0])
        elif rule["mode"] == "zscore":
            prepared = physical_mean[positive_count > 0]
        else:
            return
        _update_numeric_accumulator(accumulator, prepared)

    def _update_city_crop_accumulators(self, accumulators: dict[str, dict[str, Any]]) -> None:
        samples_per_scene_year = int(getattr(self, "city_crop_samples_per_scene_year", 0))
        if samples_per_scene_year <= 0 or not accumulators:
            return

        for row_index, row in enumerate(
            tqdm(self._stats_scene_year_rows(), desc="Build preprocessing stats: city crops", unit="scene-year")
        ):
            scene_index = getattr(self, "scene_index", {})
            scene_row = scene_index.get(str(row.get("scene_id")))
            if scene_row is None:
                continue
            static_group = self._open_group(scene_row["static_path"])
            year_group = self._open_group(row["year_path"])
            if "city_mask" not in static_group:
                continue
            city_group = static_group["city_mask"]
            city_mask = np.asarray(city_group["data"], dtype=np.float32)
            city_valid = np.asarray(city_group["valid_mask"], dtype=bool)
            city_mask = np.where(city_valid, city_mask, 0.0).astype(np.float32, copy=False)
            spatial_shape = tuple(int(value) for value in city_mask.shape[-2:])

            seed_sequence = np.random.SeedSequence(
                [
                    int(getattr(self, "city_crop_random_seed", 42)) & 0xFFFFFFFF,
                    int(row_index) & 0xFFFFFFFF,
                    int(row["year"]) & 0xFFFFFFFF,
                ]
            )
            rng = np.random.default_rng(seed_sequence)
            for _ in range(samples_per_scene_year):
                crop_box = self._city_crop_box(spatial_shape, rng)
                crop_city = city_mask[:, crop_box.top : crop_box.bottom, crop_box.left : crop_box.right] > 0
                if float(np.mean(crop_city)) < float(getattr(self, "city_crop_min_city_mask_coverage", 0.03)):
                    continue
                for source_name, accumulator in accumulators.items():
                    source_schema = self.source_specs[source_name]
                    group = static_group if source_schema["storage_level"] == "static" else year_group
                    if source_name not in group:
                        continue
                    source_group = group[source_name]
                    self._update_city_crop_source_accumulator(
                        accumulator,
                        source_name,
                        np.asarray(source_group["data"]),
                        np.asarray(source_group["valid_mask"], dtype=bool),
                        city_mask,
                        crop_box,
                    )

    def build(self) -> dict[str, Any]:
        accumulators = self._make_accumulators()
        zero_excluded_accumulators = self._make_zero_excluded_accumulators()
        city_crop_accumulators = self._make_city_crop_accumulators()
        stats_scene_rows = self._stats_scene_rows()
        stats_scene_year_rows = self._stats_scene_year_rows()

        self._update_group_stats(
            stats_scene_rows,
            "static_path",
            desc="Build preprocessing stats: static stores",
            unit="scene",
            accumulators=accumulators,
            zero_excluded_accumulators=zero_excluded_accumulators,
        )

        self._update_group_stats(
            stats_scene_year_rows,
            "year_path",
            desc="Build preprocessing stats: yearly stores",
            unit="scene-year",
            accumulators=accumulators,
            zero_excluded_accumulators=zero_excluded_accumulators,
        )

        self._update_city_crop_accumulators(city_crop_accumulators)

        sources_payload: dict[str, Any] = {}
        for source_name in sorted(self.source_specs):
            source_schema = self.source_specs[source_name]
            rule = self.rules[source_name]
            finalized = _finalize_accumulator(accumulators[source_name])
            city_crop_stats = None
            if source_name in city_crop_accumulators:
                city_crop_stats = _finalize_numeric_stats(city_crop_accumulators[source_name])
            zero_excluded_stats = None
            if source_name in zero_excluded_accumulators:
                zero_excluded_stats = _finalize_numeric_stats(zero_excluded_accumulators[source_name])
            sources_payload[source_name] = {
                "source_name": source_name,
                "modality": source_schema["modality"],
                "storage_level": source_schema["storage_level"],
                "mode": rule["mode"],
                "restore": dict(rule["restore"]),
                "inverse": _inverse_spec_for_rule(rule),
                "divisor": rule.get("divisor"),
                "categories": None if rule["mode"] != "one_hot" else list(rule["categories"]),
                "output_channel_names": _output_channel_names(source_name, rule),
                "stats": finalized["stats"],
                "zero_excluded_stats": zero_excluded_stats,
                "city_crop_stats": city_crop_stats,
                "category_counts": finalized["category_counts"],
                "valid_count": finalized["valid_count"],
            }

        payload = {
            "schema_version": self.schema["schema_version"],
            "preprocessing_version": PREPROCESSING_VERSION,
            "eump_root": str(self.eump_root),
            "stats_scope": self._stats_scope_payload(
                scene_rows=stats_scene_rows,
                scene_year_rows=stats_scene_year_rows,
            ),
            "sources": sources_payload,
        }
        write_json(self.output_path, payload)
        return payload


__all__ = [
    "PREPROCESSING_VERSION",
    "ProcessedSourceArrays",
    "ProcessedPreprocessingBuilder",
    "SourceProcessingAudit",
    "build_preprocessing_rule_registry",
    "load_preprocessing_spec",
    "physical_zero_mask_for_source_arrays",
    "process_physical_scalar_block",
    "process_scene_record",
    "process_source_arrays",
    "restore_processed_raster_block",
    "restore_scene_record",
    "validate_preprocessing_spec",
    "validate_preprocessing_stats_scope",
]
