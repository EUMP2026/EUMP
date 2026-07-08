from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .schema import (
    NORMALIZATION_FILENAME,
    PREPROCESSING_FILENAME,
    SCENES_MANIFEST_FILENAME,
    SCENE_YEARS_MANIFEST_FILENAME,
    SCHEMA_FILENAME,
)


def schema_path(eump_root: str | Path) -> Path:
    return Path(eump_root) / "manifest" / SCHEMA_FILENAME


def scenes_manifest_path(eump_root: str | Path) -> Path:
    return Path(eump_root) / "manifest" / SCENES_MANIFEST_FILENAME


def scene_years_manifest_path(eump_root: str | Path) -> Path:
    return Path(eump_root) / "manifest" / SCENE_YEARS_MANIFEST_FILENAME


def split_manifests_dir(eump_root: str | Path) -> Path:
    return Path(eump_root) / "manifest" / "splits"


def normalization_stats_path(eump_root: str | Path) -> Path:
    return Path(eump_root) / "manifest" / "stats" / NORMALIZATION_FILENAME


def preprocessing_stats_path(eump_root: str | Path) -> Path:
    return Path(eump_root) / "manifest" / "stats" / PREPROCESSING_FILENAME


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return target


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    return target


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_eump_store_schema(eump_root: str | Path) -> dict[str, Any]:
    return read_json(schema_path(eump_root))


def source_specs_from_schema(schema: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return schema["sources"]


__all__ = [
    "load_eump_store_schema",
    "normalization_stats_path",
    "preprocessing_stats_path",
    "read_json",
    "read_jsonl",
    "scene_years_manifest_path",
    "scenes_manifest_path",
    "schema_path",
    "split_manifests_dir",
    "source_specs_from_schema",
    "write_json",
    "write_jsonl",
]
