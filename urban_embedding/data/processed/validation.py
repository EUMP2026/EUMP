from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from .alignment import import_rasterio, import_zarr
from .manifest import (
    load_eump_store_schema,
    read_jsonl,
    scene_years_manifest_path,
    scenes_manifest_path,
)
from .reader import ProcessedSceneStoreReader, normalize_scene_id
from .schema import (
    CITY_MASK_SOURCE,
    CLIMATE_PRECIP_SOURCE,
    PRECIP_TARGET_SOURCE,
    as_int_year_map,
    build_source_specs,
)
from .task_config import ProcessedTaskConfig


def validate_bh8_padding_facts(bh8_root: str | Path) -> dict[str, Any]:
    rasterio, _, _ = import_rasterio()
    bh8_root_path = Path(bh8_root)
    files = sorted(bh8_root_path.glob("BH8_250_*.tif"))
    checks = {
        "GAIA_15_vs_16": (15, 16),
        "GAIA_16_vs_17": (16, 17),
        "GAIA_17_vs_18": (17, 18),
        "GAIA_18_vs_19": (18, 19),
        "POP_17_vs_18": (37, 38),
        "POP_18_vs_19": (38, 39),
        "GAIA_14_vs_15": (14, 15),
        "POP_16_vs_17": (36, 37),
    }
    summary = {
        name: {"equal_all": True, "equal_count": 0, "diff_examples": []}
        for name in checks
    }

    for path in tqdm(files, desc="Validate BH8 padding", unit="file"):
        with rasterio.open(path) as src:
            bands = src.read()

        for check_name, (left_idx, right_idx) in checks.items():
            equal = np.array_equal(bands[left_idx], bands[right_idx])
            if equal:
                summary[check_name]["equal_count"] += 1
                continue
            summary[check_name]["equal_all"] = False
            if len(summary[check_name]["diff_examples"]) < 5:
                diff_pixels = int(np.count_nonzero(bands[left_idx] != bands[right_idx]))
                summary[check_name]["diff_examples"].append(
                    {"file": path.name, "diff_pixels": diff_pixels}
                )

    padded_checks = [
        "GAIA_15_vs_16",
        "GAIA_16_vs_17",
        "GAIA_17_vs_18",
        "GAIA_18_vs_19",
        "POP_17_vs_18",
        "POP_18_vs_19",
    ]
    non_padded_checks = ["GAIA_14_vs_15", "POP_16_vs_17"]

    return {
        "file_count": len(files),
        "checks": summary,
        "all_passed": all(summary[name]["equal_all"] for name in padded_checks)
        and all(not summary[name]["equal_all"] for name in non_padded_checks),
    }


def contains_usage_key(payload: Any) -> bool:
    if isinstance(payload, dict):
        for key, value in payload.items():
            lowered = str(key).lower()
            if lowered == "usage" or lowered == "usage_hint":
                return True
            if contains_usage_key(value):
                return True
    if isinstance(payload, list):
        return any(contains_usage_key(item) for item in payload)
    return False


def normalized_source_schema(spec: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(spec)
    if normalized.get("padding_rule") is not None:
        normalized["padding_rule"] = {
            "kind": normalized["padding_rule"]["kind"],
            "year_map": as_int_year_map(normalized["padding_rule"]["year_map"]),
        }
    return normalized


def validate_eump_store(
    eump_root: str | Path,
    task_config: ProcessedTaskConfig | None = None,
) -> dict[str, Any]:
    eump_root_path = Path(eump_root)
    schema = load_eump_store_schema(eump_root_path)
    scenes_rows = read_jsonl(scenes_manifest_path(eump_root_path))
    scene_year_rows = read_jsonl(scene_years_manifest_path(eump_root_path))
    source_specs = {name: normalized_source_schema(spec) for name, spec in schema["sources"].items()}
    expected_specs = {
        name: normalized_source_schema(spec.to_dict())
        for name, spec in build_source_specs().items()
    }

    errors: list[str] = []
    warnings: list[str] = []

    if contains_usage_key(schema):
        errors.append("schema contains a forbidden usage field")
    if any(contains_usage_key(row) for row in scenes_rows + scene_year_rows):
        errors.append("manifest contains a forbidden usage field")

    for source_name in ("GAIA", "POP"):
        if source_specs[source_name]["raw_materialized_years"] != expected_specs[source_name]["raw_materialized_years"]:
            errors.append(f"{source_name} raw_materialized_years mismatch")
        if source_specs[source_name]["canonical_years"] != expected_specs[source_name]["canonical_years"]:
            errors.append(f"{source_name} canonical_years mismatch")
        if source_specs[source_name]["padding_rule"] != expected_specs[source_name]["padding_rule"]:
            errors.append(f"{source_name} padding_rule mismatch")

    zarr = import_zarr()
    scene_rows_by_id = {normalize_scene_id(row["scene_id"]): row for row in scenes_rows}

    for row in tqdm(scenes_rows, desc="Validate EUMP store: static stores", unit="scene"):
        static_path = eump_root_path / row["static_path"]
        if not static_path.exists():
            errors.append(f"Missing static store: {static_path}")
            continue

        static_group = zarr.open_group(str(static_path), mode="r")
        available_sources = sorted(static_group.group_keys())
        if sorted(row["present_sources"]) != available_sources:
            errors.append(f"Scene {row['scene_id']} present_sources mismatch")
        if CITY_MASK_SOURCE not in available_sources:
            errors.append(f"Scene {row['scene_id']} is missing city_mask")

    for row in tqdm(scene_year_rows, desc="Validate EUMP store: yearly stores", unit="scene-year"):
        year = int(row["year"])
        year_path = eump_root_path / row["year_path"]
        if not year_path.exists():
            errors.append(f"Missing yearly store: {year_path}")
            continue

        year_group = zarr.open_group(str(year_path), mode="r")
        present_sources = sorted(year_group.group_keys())
        if sorted(row["present_sources"]) != present_sources:
            errors.append(f"Scene {row['scene_id']} year {year} present_sources mismatch")

        scene_row = scene_rows_by_id[normalize_scene_id(row["scene_id"])]
        static_group = zarr.open_group(str(eump_root_path / scene_row["static_path"]), mode="r")
        static_shape = tuple(int(value) for value in static_group.attrs["spatial_shape"])
        year_shape = tuple(int(value) for value in year_group.attrs["spatial_shape"])
        if static_shape != year_shape:
            errors.append(f"Scene {row['scene_id']} year {year} spatial_shape mismatch")

        for source_name in present_sources:
            spec = source_specs[source_name]
            if year not in spec["canonical_years"]:
                errors.append(f"{source_name} appears in unsupported year {year}")
            source_group = year_group[source_name]
            data_shape = tuple(int(dim) for dim in np.asarray(source_group["data"]).shape)
            mask_shape = tuple(int(dim) for dim in np.asarray(source_group["valid_mask"]).shape)
            if data_shape != mask_shape:
                errors.append(f"{source_name} data/valid_mask shape mismatch in scene {row['scene_id']} year {year}")

    if task_config is None and scene_year_rows:
        first_year_row = scene_year_rows[0]
        default_inputs = [
            source_name
            for source_name in ("NDVI", "NL", "POP", "water", "BH", "BF", "DEM", "Ta", CLIMATE_PRECIP_SOURCE, "wind")
            if source_name in first_year_row["present_sources"]
            or source_name in scene_rows_by_id[normalize_scene_id(first_year_row["scene_id"])]["present_sources"]
        ]
        target_source = None
        for candidate in ("LST", PRECIP_TARGET_SOURCE):
            if candidate in first_year_row["present_sources"]:
                target_source = candidate
                break
        task_config = ProcessedTaskConfig(
            input_sources=tuple(default_inputs),
            reconstruct_target_source=target_source,
            city_mask_source=CITY_MASK_SOURCE,
        )

    sample_scene_loaded = False
    if task_config is not None and scene_year_rows:
        reader = ProcessedSceneStoreReader(eump_root_path)
        compatible_row = reader.find_compatible_scene_year(task_config)
        if compatible_row is None:
            warnings.append(
                "No scene-year in the exported store satisfies the requested task config without missing sources."
            )
        else:
            reader.load_scene_record(compatible_row["scene_id"], int(compatible_row["year"]), task_config)
            sample_scene_loaded = True

    return {
        "eump_root": str(eump_root_path),
        "schema_version": schema["schema_version"],
        "scene_count": len(scenes_rows),
        "scene_year_count": len(scene_year_rows),
        "errors": errors,
        "warnings": warnings,
        "sample_scene_loaded": sample_scene_loaded,
        "is_valid": not errors,
    }


__all__ = [
    "contains_usage_key",
    "normalized_source_schema",
    "validate_bh8_padding_facts",
    "validate_eump_store",
]
