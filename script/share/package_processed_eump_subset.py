#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import sys
import tarfile
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from urban_embedding.data.processed.manifest import (
    load_eump_store_schema,
    normalization_stats_path,
    preprocessing_stats_path,
    read_json,
    read_jsonl,
    scene_years_manifest_path,
    scenes_manifest_path,
    schema_path,
    split_manifests_dir,
    write_json,
    write_jsonl,
)
from urban_embedding.data.processed.processing import ProcessedPreprocessingBuilder
from urban_embedding.data.processed.split_manifest import validate_split_manifest
from urban_embedding.data.processed.validation import validate_eump_store


DEFAULT_SPLIT_MANIFEST_NAME = "legacy_compatible_seed42.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Package a filtered processed EUMP subset, for example a 2020-only "
            "shareable data root for downstream Docker research bundles."
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Source processed EUMP root, for example /path/to/processed_v2.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Output subset root directory.",
    )
    parser.add_argument(
        "--year-range",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        default=(2020, 2020),
        help="Inclusive year range to keep. Default: 2020 2020.",
    )
    parser.add_argument(
        "--split-manifest-path",
        type=Path,
        default=None,
        help=(
            "Optional source split manifest to filter into the subset. "
            "Default: <source-root>/manifest/splits/legacy_compatible_seed42.json when it exists."
        ),
    )
    parser.add_argument(
        "--split-manifest-name",
        default=DEFAULT_SPLIT_MANIFEST_NAME,
        help=(
            "Output split manifest file name under manifest/splits/. "
            f"Default: {DEFAULT_SPLIT_MANIFEST_NAME}."
        ),
    )
    parser.add_argument(
        "--stats-split",
        choices=("auto", "all", "train", "val", "test"),
        default="auto",
        help=(
            "Split used when rebuilding preprocessing_v1.json. "
            "Default: auto, meaning train when a split manifest is present, otherwise all."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Worker count passed into preprocessing rebuild. Default: 0.",
    )
    parser.add_argument(
        "--archive-path",
        type=Path,
        default=None,
        help="Optional .tar.gz output path for the packaged subset root.",
    )
    parser.add_argument(
        "--skip-preprocessing-rebuild",
        action="store_true",
        help="Skip preprocessing_v1.json rebuild. Not recommended for a year-filtered package.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip post-build validate_eump_store.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output-root directory or archive.",
    )
    return parser.parse_args(argv)


def _resolve_existing_dir(path: Path | str, *, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{label} directory does not exist: {resolved}")
    return resolved


def _resolve_existing_file(path: Path | str, *, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} file does not exist: {resolved}")
    return resolved


def _copy_tree(source: Path, destination: Path) -> None:
    shutil.copytree(source, destination, dirs_exist_ok=True)


def _filter_scene_year_rows(
    rows: list[dict[str, Any]],
    *,
    year_range: tuple[int, int],
) -> list[dict[str, Any]]:
    start_year, end_year = year_range
    return [
        dict(row)
        for row in rows
        if start_year <= int(row["year"]) <= end_year
    ]


def _filter_scene_rows(
    rows: list[dict[str, Any]],
    *,
    included_scene_ids: set[str],
    years_by_scene_id: dict[str, list[int]],
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for row in rows:
        scene_id = str(row["scene_id"])
        if scene_id not in included_scene_ids:
            continue
        updated = dict(row)
        updated["years"] = sorted(int(year) for year in years_by_scene_id[scene_id])
        filtered.append(updated)
    return filtered


def _filtered_split_manifest(
    payload: dict[str, Any],
    *,
    included_scene_ids: set[str],
    year_range: tuple[int, int],
) -> dict[str, Any]:
    def keep(values: list[Any]) -> list[int]:
        kept = {
            int(value)
            for value in values
            if str(value) in included_scene_ids
        }
        return sorted(kept)

    filtered = dict(payload)
    filtered["pool_scene_ids"] = keep(list(payload["pool_scene_ids"]))
    filtered["train_scene_ids"] = keep(list(payload["train_scene_ids"]))
    filtered["val_scene_ids"] = keep(list(payload["val_scene_ids"]))
    filtered["test_scene_ids"] = keep(list(payload["test_scene_ids"]))
    filtered["year_range"] = [int(year_range[0]), int(year_range[1])]
    return filtered


def _write_subset_readme(
    *,
    output_root: Path,
    year_range: tuple[int, int],
    split_manifest_name: str | None,
    has_preprocessing: bool,
) -> None:
    rebuild_lines = [
        "python script/data/build_eump_preprocessing.py \\",
        "  --eump-root /mnt/eump_2020 \\",
        f"  --year-range {year_range[0]} {year_range[1]} \\",
    ]
    if split_manifest_name is not None:
        rebuild_lines.append(
            f"  --split-manifest-path /mnt/eump_2020/manifest/splits/{split_manifest_name} \\"
        )
    rebuild_lines.append("  --stats-split train")

    lines = [
        "# Processed EUMP Subset",
        "",
        "This package contains a filtered processed EUMP root intended to be mounted",
        "into the Docker research bundle as external data.",
        "",
        f"- included year_range: `{list(year_range)}`",
    ]
    if has_preprocessing:
        lines.append("- preprocessing: `manifest/stats/preprocessing_v1.json`")
    else:
        lines.append("- preprocessing: not included; rebuild it before downstream evaluation")
    if split_manifest_name is not None:
        lines.append(f"- split manifest: `manifest/splits/{split_manifest_name}`")
    lines.extend(
        [
            "",
            "## Important Runtime Note",
            "",
            "When you use this subset with downstream evaluation or custom scripts, add",
            "a Hydra override so the pipeline matches the subset year range:",
            "",
            "```bash",
            f"'pipeline.year_range=[{year_range[0]},{year_range[1]}]'",
            "```",
            "",
            "If you keep the default downstream pipeline year range `2003..2020`,",
            "preprocessing and split-manifest validation will fail against this subset.",
            "",
            "If preprocessing was skipped during packaging, rebuild it inside the bundle with:",
            "",
            "```bash",
            *rebuild_lines,
            "```",
            "",
            "## Mount Example",
            "",
            "```bash",
            "docker run --rm -it --gpus all \\",
            "  -v \"$PWD\":/workspace/urban_embedding_v2 \\",
            "  -v /absolute/path/to/processed_subset:/mnt/eump_2020:ro \\",
            "  -w /workspace/urban_embedding_v2 \\",
            "  <image-tag> \\",
            "  bash",
            "```",
        ]
    )
    (output_root / "DATA_BUNDLE_README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _archive_directory(source_root: Path, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, mode="w:gz") as handle:
        handle.add(source_root, arcname=source_root.name)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    year_range = (int(args.year_range[0]), int(args.year_range[1]))
    if year_range[0] > year_range[1]:
        raise ValueError(f"Invalid year_range: start > end: {year_range}")

    source_root = _resolve_existing_dir(args.source_root, label="source-root")
    output_root = Path(args.output_root).expanduser()
    if not output_root.is_absolute():
        output_root = output_root.resolve()

    split_manifest_path: Path | None
    if args.split_manifest_path is not None:
        split_manifest_path = _resolve_existing_file(args.split_manifest_path, label="split-manifest-path")
    else:
        default_split = source_root / "manifest" / "splits" / DEFAULT_SPLIT_MANIFEST_NAME
        split_manifest_path = default_split if default_split.is_file() else None

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output root already exists: {output_root}. Use --overwrite to replace it."
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=False)

    schema = load_eump_store_schema(source_root)
    scene_rows = read_jsonl(scenes_manifest_path(source_root))
    scene_year_rows = read_jsonl(scene_years_manifest_path(source_root))

    filtered_scene_year_rows = _filter_scene_year_rows(scene_year_rows, year_range=year_range)
    if not filtered_scene_year_rows:
        raise ValueError(
            f"No scene-year rows remain under source-root={source_root} for year_range={year_range}"
        )

    years_by_scene_id: dict[str, list[int]] = defaultdict(list)
    for row in filtered_scene_year_rows:
        years_by_scene_id[str(row["scene_id"])].append(int(row["year"]))
    included_scene_ids = set(years_by_scene_id)
    filtered_scene_rows = _filter_scene_rows(
        scene_rows,
        included_scene_ids=included_scene_ids,
        years_by_scene_id=years_by_scene_id,
    )
    if not filtered_scene_rows:
        raise ValueError(
            f"No scene rows remain under source-root={source_root} for year_range={year_range}"
        )

    for row in filtered_scene_rows:
        source_path = source_root / row["static_path"]
        if not source_path.is_dir():
            raise FileNotFoundError(f"Missing static store referenced by manifest: {source_path}")
        _copy_tree(source_path, output_root / row["static_path"])

    for row in filtered_scene_year_rows:
        source_path = source_root / row["year_path"]
        if not source_path.is_dir():
            raise FileNotFoundError(f"Missing yearly store referenced by manifest: {source_path}")
        _copy_tree(source_path, output_root / row["year_path"])

    schema_payload = dict(schema)
    schema_payload["eump_root"] = output_root.name
    write_json(schema_path(output_root), schema_payload)
    write_jsonl(scenes_manifest_path(output_root), filtered_scene_rows)
    write_jsonl(scene_years_manifest_path(output_root), filtered_scene_year_rows)

    output_split_path: Path | None = None
    if split_manifest_path is not None:
        filtered_split = _filtered_split_manifest(
            read_json(split_manifest_path),
            included_scene_ids=included_scene_ids,
            year_range=year_range,
        )
        try:
            validate_split_manifest(
                filtered_split,
                available_scene_ids=sorted(included_scene_ids),
                expected_year_range=year_range,
            )
        except Exception as exc:
            raise ValueError(
                "Filtered split manifest is not valid for the packaged subset. "
                "This usually means one split became empty after filtering. "
                "Either package a wider year range or rebuild a dedicated split manifest "
                "for the subset."
            ) from exc
        output_split_path = split_manifests_dir(output_root) / args.split_manifest_name
        write_json(output_split_path, filtered_split)

    if not args.skip_preprocessing_rebuild:
        stats_split = args.stats_split
        if stats_split == "auto":
            stats_split = "train" if output_split_path is not None else "all"
        builder = ProcessedPreprocessingBuilder(
            eump_root=output_root,
            output_path=preprocessing_stats_path(output_root),
            split_manifest_path=output_split_path,
            stats_split=stats_split,
            year_range=year_range,
            num_workers=int(args.workers),
        )
        builder.build()

    split_manifest_name = output_split_path.name if output_split_path is not None else None
    _write_subset_readme(
        output_root=output_root,
        year_range=year_range,
        split_manifest_name=split_manifest_name,
        has_preprocessing=not args.skip_preprocessing_rebuild,
    )

    validation_payload: dict[str, Any] | None = None
    if not args.skip_validation:
        validation_payload = validate_eump_store(output_root)
        if not validation_payload["is_valid"]:
            raise ValueError(
                "validate_eump_store failed for the packaged subset: "
                + "; ".join(validation_payload["errors"][:10])
            )

    archive_path: Path | None = None
    if args.archive_path is not None:
        archive_path = Path(args.archive_path).expanduser()
        if not archive_path.is_absolute():
            archive_path = archive_path.resolve()
        if archive_path.exists():
            if not args.overwrite:
                raise FileExistsError(
                    f"Archive already exists: {archive_path}. Use --overwrite to replace it."
                )
            archive_path.unlink()
        _archive_directory(output_root, archive_path)

    summary = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "year_range": [int(year_range[0]), int(year_range[1])],
        "scene_count": len(filtered_scene_rows),
        "scene_year_count": len(filtered_scene_year_rows),
        "split_manifest": None if output_split_path is None else str(output_split_path),
        "preprocessing_path": None if args.skip_preprocessing_rebuild else str(preprocessing_stats_path(output_root)),
        "normalization_path": str(normalization_stats_path(output_root))
        if normalization_stats_path(output_root).exists()
        else None,
        "archive_path": None if archive_path is None else str(archive_path),
        "validation": validation_payload,
    }
    write_json(output_root / "subset_manifest.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
