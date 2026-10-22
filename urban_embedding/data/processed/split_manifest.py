from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from sklearn.model_selection import train_test_split

from urban_embedding.core.paths import CSV_DATA_DIR, EUMP_DATA_DIR

from .manifest import read_json, read_jsonl, scenes_manifest_path, split_manifests_dir, write_json
from .reader import normalize_scene_id


SPLIT_MANIFEST_VERSION = "eump_scene_split_v1"
LEGACY_COMPATIBLE_GENERATOR = "legacy_two_stage_train_test_split"
DEFAULT_LEGACY_SPLIT_MANIFEST_NAME = "legacy_compatible_seed42.json"
DEFAULT_SPLIT_COLUMN_NAME = "split"
SPLIT_LABEL_FIELDS = (
    ("train", "train_scene_ids"),
    ("val", "val_scene_ids"),
    ("test", "test_scene_ids"),
)

REQUIRED_SPLIT_MANIFEST_KEYS = {
    "version",
    "generator",
    "seed",
    "test_size",
    "source_csv_path",
    "year_range",
    "pool_scene_ids",
    "train_scene_ids",
    "val_scene_ids",
    "test_scene_ids",
}


def default_legacy_split_manifest_path(
    eump_root: str | Path = EUMP_DATA_DIR,
    *,
    file_name: str = DEFAULT_LEGACY_SPLIT_MANIFEST_NAME,
) -> Path:
    return split_manifests_dir(eump_root) / file_name


def _normalized_scene_id_list(values: Iterable[str | int]) -> list[str]:
    return [normalize_scene_id(value) for value in values]


def _unique_scene_ids_in_order(values: Iterable[int]) -> list[int]:
    unique = pd.Series(list(values), dtype="int64").drop_duplicates()
    return [int(value) for value in unique.tolist()]


def _sorted_int_list(values: Iterable[int]) -> list[int]:
    return sorted(int(value) for value in values)


def _scene_id_sets(payload: dict[str, Any]) -> tuple[set[str], set[str], set[str], set[str]]:
    train_ids = set(_normalized_scene_id_list(payload["train_scene_ids"]))
    val_ids = set(_normalized_scene_id_list(payload["val_scene_ids"]))
    test_ids = set(_normalized_scene_id_list(payload["test_scene_ids"]))
    pool_ids = set(_normalized_scene_id_list(payload["pool_scene_ids"]))
    return train_ids, val_ids, test_ids, pool_ids


def _contains_year_range(container: tuple[int, int] | None, requested: tuple[int, int]) -> bool:
    if container is None:
        return True
    return int(container[0]) <= int(requested[0]) and int(requested[1]) <= int(container[1])


def validate_split_manifest(
    payload: dict[str, Any],
    *,
    available_scene_ids: Iterable[str | int] | None = None,
    expected_year_range: tuple[int, int] | None = None,
) -> dict[str, Any]:
    missing_keys = sorted(REQUIRED_SPLIT_MANIFEST_KEYS - set(payload))
    if missing_keys:
        missing_list = ", ".join(missing_keys)
        raise ValueError(f"Split manifest is missing required field(s): {missing_list}")

    version = str(payload["version"])
    if version != SPLIT_MANIFEST_VERSION:
        raise ValueError(
            f"Unsupported split manifest version {version!r}; expected {SPLIT_MANIFEST_VERSION!r}"
        )

    train_ids, val_ids, test_ids, pool_ids = _scene_id_sets(payload)
    if not train_ids or not val_ids or not test_ids:
        raise ValueError("Split manifest requires non-empty train_scene_ids, val_scene_ids, and test_scene_ids")
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise ValueError("Split manifest train/val/test scene IDs must be mutually exclusive")
    if train_ids | val_ids | test_ids != pool_ids:
        raise ValueError("Split manifest pool_scene_ids must exactly match the union of train/val/test scene IDs")

    if available_scene_ids is not None:
        available = set(_normalized_scene_id_list(available_scene_ids))
        unknown_ids = sorted(pool_ids - available)
        if unknown_ids:
            unknown_list = ", ".join(unknown_ids[:20])
            raise ValueError(
                "Split manifest contains scene IDs that are unavailable in the current EUMP root: "
                f"{unknown_list}"
            )

    raw_year_range = payload["year_range"]
    normalized_year_range: tuple[int, int] | None
    if raw_year_range is None:
        normalized_year_range = None
    else:
        if not isinstance(raw_year_range, (list, tuple)) or len(raw_year_range) != 2:
            raise ValueError("Split manifest year_range must be null or contain exactly two years")
        normalized_year_range = (int(raw_year_range[0]), int(raw_year_range[1]))

    if expected_year_range is not None:
        requested_year_range = tuple(int(year) for year in expected_year_range)
    else:
        requested_year_range = None

    if requested_year_range is not None and not _contains_year_range(normalized_year_range, requested_year_range):
        raise ValueError(
            "Split manifest year_range does not cover the requested pipeline year_range: "
            f"manifest={normalized_year_range!r} pipeline={requested_year_range!r}"
        )

    return payload


def load_split_manifest(
    path: str | Path,
    *,
    available_scene_ids: Iterable[str | int] | None = None,
    expected_year_range: tuple[int, int] | None = None,
) -> dict[str, Any]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Split manifest does not exist: {manifest_path}")
    payload = read_json(manifest_path)
    return validate_split_manifest(
        payload,
        available_scene_ids=available_scene_ids,
        expected_year_range=expected_year_range,
    )


def split_scene_ids_from_manifest(payload: dict[str, Any]) -> tuple[set[str], set[str], set[str]]:
    train_ids, val_ids, test_ids, _ = _scene_id_sets(payload)
    return train_ids, val_ids, test_ids


def split_label_lookup_from_manifest(payload: dict[str, Any]) -> dict[str, str]:
    validate_split_manifest(payload)
    lookup: dict[str, str] = {}
    for split_name, field_name in SPLIT_LABEL_FIELDS:
        for scene_id in payload[field_name]:
            lookup[normalize_scene_id(scene_id)] = split_name
    return lookup


def _normalize_csv_scene_id(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return normalize_scene_id(int(value))
    return normalize_scene_id(value)


def annotate_table_with_split_labels(
    table: pd.DataFrame,
    split_manifest: dict[str, Any],
    *,
    id_column: str = "ID",
    split_column: str = DEFAULT_SPLIT_COLUMN_NAME,
) -> pd.DataFrame:
    if id_column not in table.columns:
        raise ValueError(f"CSV table must contain an {id_column!r} column to add split labels")
    if not split_column:
        raise ValueError("split_column must be a non-empty column name")

    split_lookup = split_label_lookup_from_manifest(split_manifest)
    annotated = table.copy()
    annotated[split_column] = annotated[id_column].map(
        lambda value: split_lookup.get(_normalize_csv_scene_id(value), pd.NA)
    )
    return annotated


def write_split_annotated_csv(
    *,
    csv_path: str | Path,
    split_manifest: dict[str, Any],
    output_path: str | Path | None = None,
    id_column: str = "ID",
    split_column: str = DEFAULT_SPLIT_COLUMN_NAME,
) -> dict[str, Any]:
    csv_path_obj = Path(csv_path).expanduser().resolve()
    output_path_obj = csv_path_obj if output_path is None else Path(output_path).expanduser().resolve()

    table = pd.read_csv(csv_path_obj)
    annotated = annotate_table_with_split_labels(
        table,
        split_manifest,
        id_column=id_column,
        split_column=split_column,
    )
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    annotated.to_csv(output_path_obj, index=False)

    split_counts = {
        str(split_name): int(count)
        for split_name, count in annotated[split_column].value_counts(dropna=True).sort_index().items()
    }
    unassigned_count = int(annotated[split_column].isna().sum())
    return {
        "source_csv_path": str(csv_path_obj),
        "output_csv_path": str(output_path_obj),
        "split_column": split_column,
        "row_count": int(len(annotated)),
        "assigned_count": int(len(annotated) - unassigned_count),
        "unassigned_count": unassigned_count,
        "split_counts": split_counts,
    }


def build_legacy_compatible_split_manifest(
    *,
    eump_root: str | Path = EUMP_DATA_DIR,
    csv_path: str | Path = Path(CSV_DATA_DIR) / "df_norm_v3.csv",
    output_path: str | Path | None = None,
    random_seed: int = 42,
    test_size: float = 0.3,
    year_range: tuple[int, int] | None = (2003, 2020),
    restrict_to_available_scenes: bool = True,
    sort_output_ids: bool = True,
) -> dict[str, Any]:
    eump_root_path = Path(eump_root).expanduser().resolve()
    csv_path_obj = Path(csv_path).expanduser().resolve()

    table = pd.read_csv(csv_path_obj)
    if "ID" not in table.columns:
        raise ValueError(f"Legacy compatibility CSV must contain an 'ID' column: {csv_path_obj}")
    if year_range is not None:
        if "year" not in table.columns:
            raise ValueError("Legacy compatibility CSV filtering by year_range requires a 'year' column")
        start_year, end_year = (int(year) for year in year_range)
        table = table[(table["year"] >= start_year) & (table["year"] <= end_year)]

    scene_rows = read_jsonl(scenes_manifest_path(eump_root_path))
    available_scene_ids = {int(row["scene_id"]) for row in scene_rows}
    candidate_ids = _unique_scene_ids_in_order(table["ID"].astype(int).tolist())
    if restrict_to_available_scenes:
        pool_scene_ids = [scene_id for scene_id in candidate_ids if scene_id in available_scene_ids]
    else:
        pool_scene_ids = [int(scene_id) for scene_id in candidate_ids]
    if len(pool_scene_ids) < 3:
        raise ValueError(
            "Legacy-compatible split generation requires at least 3 intersecting scene IDs, "
            f"got {len(pool_scene_ids)}"
        )

    train_ids, tmp_ids = train_test_split(
        pool_scene_ids,
        test_size=float(test_size),
        random_state=int(random_seed),
    )
    val_ids, test_ids = train_test_split(
        tmp_ids,
        test_size=0.5,
        random_state=int(random_seed),
    )

    output_pool_scene_ids = _sorted_int_list(pool_scene_ids) if sort_output_ids else [int(scene_id) for scene_id in pool_scene_ids]
    output_train_ids = _sorted_int_list(train_ids) if sort_output_ids else [int(scene_id) for scene_id in train_ids]
    output_val_ids = _sorted_int_list(val_ids) if sort_output_ids else [int(scene_id) for scene_id in val_ids]
    output_test_ids = _sorted_int_list(test_ids) if sort_output_ids else [int(scene_id) for scene_id in test_ids]

    payload = {
        "version": SPLIT_MANIFEST_VERSION,
        "generator": LEGACY_COMPATIBLE_GENERATOR,
        "seed": int(random_seed),
        "test_size": float(test_size),
        "source_csv_path": str(csv_path_obj),
        "year_range": None if year_range is None else [int(year_range[0]), int(year_range[1])],
        "pool_scene_ids": output_pool_scene_ids,
        "train_scene_ids": output_train_ids,
        "val_scene_ids": output_val_ids,
        "test_scene_ids": output_test_ids,
    }
    validate_split_manifest(
        payload,
        available_scene_ids=available_scene_ids if restrict_to_available_scenes else None,
        expected_year_range=year_range,
    )

    target_path = (
        default_legacy_split_manifest_path(eump_root_path)
        if output_path is None
        else Path(output_path).expanduser().resolve()
    )
    write_json(target_path, payload)
    return payload


__all__ = [
    "DEFAULT_SPLIT_COLUMN_NAME",
    "DEFAULT_LEGACY_SPLIT_MANIFEST_NAME",
    "LEGACY_COMPATIBLE_GENERATOR",
    "SPLIT_MANIFEST_VERSION",
    "annotate_table_with_split_labels",
    "build_legacy_compatible_split_manifest",
    "default_legacy_split_manifest_path",
    "load_split_manifest",
    "split_label_lookup_from_manifest",
    "split_scene_ids_from_manifest",
    "validate_split_manifest",
    "write_split_annotated_csv",
]
