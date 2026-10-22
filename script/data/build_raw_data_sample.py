#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm


@dataclass(frozen=True)
class RawDatasetRule:
    dataset: str
    glob_pattern: str
    scene_id_pattern: re.Pattern[str]


@dataclass(frozen=True)
class IndexedFile:
    path: Path
    scene_id: int
    dataset: str


RAW_DATASET_RULES = (
    RawDatasetRule("BH8_v1", "*.tif", re.compile(r"^BH8_250_(?P<scene_id>\d+)\.tif$", re.IGNORECASE)),
    RawDatasetRule("VI3_v1", "*.tif", re.compile(r"^VI3_500_(?P<scene_id>\d+)\.tif$", re.IGNORECASE)),
    RawDatasetRule("NL_v1", "*.tif", re.compile(r"^NL_\d{4}_(?P<scene_id>\d+)\.tif$", re.IGNORECASE)),
    RawDatasetRule("land_cover", "*.tif", re.compile(r"^LC_500_(?P<scene_id>\d+)\.tif$", re.IGNORECASE)),
    RawDatasetRule("LCZ", "*.tif", re.compile(r"^LCZ_250_(?P<scene_id>\d+)\.tif$", re.IGNORECASE)),
    RawDatasetRule("PM25", "*.tif", re.compile(r"^PM25_1000_(?P<scene_id>\d+)\.tif$", re.IGNORECASE)),
    RawDatasetRule("road_density", "*/*.tif", re.compile(r"^\d{4}_(?P<scene_id>\d+)_.*_roads_250m\.tif$", re.IGNORECASE)),
    RawDatasetRule("electricity", "*.tif", re.compile(r"^\d{4}_(?P<scene_id>\d+)_.*_masked\.tif$", re.IGNORECASE)),
    RawDatasetRule("GDP", "*.tif", re.compile(r"^gdpTot_6bands_(?P<scene_id>\d+)\.tif$", re.IGNORECASE)),
    RawDatasetRule("lst_v1", "*.tif", re.compile(r"^lst_1000_(?P<scene_id>\d+)\.tif$", re.IGNORECASE)),
    RawDatasetRule("precip_v1", "*.tif", re.compile(r"^Precip_\d{4}_(?P<scene_id>\d+)\.tif$", re.IGNORECASE)),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample aligned city IDs from raw_data into raw_data_sample. "
            "The script first selects scene/city IDs, then copies every known "
            "raw raster file whose filename belongs to one of those IDs while "
            "preserving the original relative folder structure."
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("data/raw_data"),
        help="Input raw raster root. Default: data/raw_data.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/raw_data_sample"),
        help="Output sampled raw raster root. Default: data/raw_data_sample.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=10,
        help="Number of city IDs to sample when --scene-ids is not provided.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible ID sampling.",
    )
    parser.add_argument(
        "--scene-ids",
        nargs="*",
        type=int,
        default=None,
        help="Explicit city/scene IDs to copy. When set, --sample-count is ignored.",
    )
    parser.add_argument(
        "--id-source",
        choices=tuple(rule.dataset for rule in RAW_DATASET_RULES),
        default="BH8_v1",
        help="Dataset used to discover candidate IDs for random sampling. Default: BH8_v1.",
    )
    parser.add_argument(
        "--require-datasets",
        nargs="*",
        default=None,
        choices=tuple(rule.dataset for rule in RAW_DATASET_RULES),
        help=(
            "Optional datasets whose ID sets must all contain a city before it "
            "can be randomly sampled. By default only --id-source is required."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the existing output directory before writing the sample.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.sample_count < 0:
        raise ValueError(f"--sample-count must be >= 0, got {args.sample_count}")
    if args.scene_ids is not None and len(args.scene_ids) == 0:
        raise ValueError("--scene-ids was provided but no IDs were listed")


def prepare_output_root(output_root: Path, overwrite: bool) -> None:
    if output_root.exists():
        if not output_root.is_dir():
            raise NotADirectoryError(f"Output root exists but is not a directory: {output_root}")
        if overwrite:
            shutil.rmtree(output_root)
        elif any(output_root.iterdir()):
            raise FileExistsError(f"Output root is not empty: {output_root}. Use --overwrite to replace it.")
    output_root.mkdir(parents=True, exist_ok=True)


def extract_scene_id(rule: RawDatasetRule, path: Path) -> int | None:
    match = rule.scene_id_pattern.match(path.name)
    if match is None:
        return None
    return int(match.group("scene_id"))


def index_dataset(source_root: Path, rule: RawDatasetRule) -> dict[int, list[Path]]:
    dataset_root = source_root / rule.dataset
    if not dataset_root.exists():
        return {}
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"Dataset path exists but is not a directory: {dataset_root}")

    files_by_scene_id: dict[int, list[Path]] = {}
    for path in sorted(dataset_root.glob(rule.glob_pattern)):
        if not path.is_file():
            continue
        scene_id = extract_scene_id(rule, path)
        if scene_id is None:
            continue
        files_by_scene_id.setdefault(scene_id, []).append(path)
    return files_by_scene_id


def index_raw_data(source_root: Path) -> dict[str, dict[int, list[Path]]]:
    if not source_root.exists():
        raise FileNotFoundError(f"Source root does not exist: {source_root}")
    if not source_root.is_dir():
        raise NotADirectoryError(f"Source root is not a directory: {source_root}")

    indexed: dict[str, dict[int, list[Path]]] = {}
    for rule in tqdm(RAW_DATASET_RULES, desc="Index raw datasets", unit="dataset"):
        indexed[rule.dataset] = index_dataset(source_root, rule)
    return indexed


def select_scene_ids(
    indexed: dict[str, dict[int, list[Path]]],
    explicit_scene_ids: list[int] | None,
    sample_count: int,
    seed: int,
    id_source: str,
    require_datasets: list[str] | None,
) -> list[int]:
    if explicit_scene_ids is not None:
        return sorted(set(explicit_scene_ids))

    required = [id_source]
    if require_datasets is not None:
        required = sorted(set(required).union(require_datasets))

    missing_required = [dataset for dataset in required if dataset not in indexed or not indexed[dataset]]
    if missing_required:
        raise FileNotFoundError(
            "Cannot sample IDs because required datasets have no indexed files: "
            + ", ".join(missing_required)
        )

    candidate_ids = set(indexed[required[0]])
    for dataset in required[1:]:
        candidate_ids.intersection_update(indexed[dataset])

    candidates = sorted(candidate_ids)
    if len(candidates) < sample_count:
        raise ValueError(
            f"Requested {sample_count} IDs but only {len(candidates)} candidate IDs are available "
            f"after requiring datasets: {', '.join(required)}"
        )

    rng = random.Random(seed)
    return sorted(rng.sample(candidates, sample_count))


def collect_selected_files(
    indexed: dict[str, dict[int, list[Path]]],
    selected_scene_ids: list[int],
) -> tuple[list[IndexedFile], dict[str, dict[str, object]], dict[str, list[int]]]:
    selected_id_set = set(selected_scene_ids)
    selected_files: list[IndexedFile] = []
    dataset_summary: dict[str, dict[str, object]] = {}
    missing_ids_by_dataset: dict[str, list[int]] = {}

    for dataset, files_by_scene_id in indexed.items():
        copied_count = 0
        available_ids = sorted(scene_id for scene_id in selected_scene_ids if scene_id in files_by_scene_id)
        missing_ids = sorted(selected_id_set.difference(files_by_scene_id))
        for scene_id in available_ids:
            paths = files_by_scene_id[scene_id]
            copied_count += len(paths)
            selected_files.extend(IndexedFile(path=path, scene_id=scene_id, dataset=dataset) for path in paths)
        dataset_summary[dataset] = {
            "indexed_scene_id_count": len(files_by_scene_id),
            "selected_scene_id_count": len(available_ids),
            "selected_file_count": copied_count,
            "missing_selected_scene_ids": missing_ids,
        }
        if missing_ids:
            missing_ids_by_dataset[dataset] = missing_ids

    selected_files.sort(key=lambda item: item.path.as_posix())
    return selected_files, dataset_summary, missing_ids_by_dataset


def copy_selected_files(selected_files: list[IndexedFile], source_root: Path, output_root: Path) -> None:
    for item in tqdm(selected_files, desc="Copy selected files", unit="file"):
        relative_path = item.path.relative_to(source_root)
        destination = output_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item.path, destination)


def write_manifest(
    output_root: Path,
    source_root: Path,
    selected_scene_ids: list[int],
    selected_files: list[IndexedFile],
    dataset_summary: dict[str, dict[str, object]],
    args: argparse.Namespace,
) -> Path:
    manifest = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "selection": {
            "selected_scene_ids": selected_scene_ids,
            "selected_scene_id_count": len(selected_scene_ids),
            "sample_count": args.sample_count,
            "seed": args.seed,
            "id_source": args.id_source,
            "require_datasets": args.require_datasets,
            "explicit_scene_ids": args.scene_ids,
        },
        "datasets": dataset_summary,
        "copied_files": [
            {
                "dataset": item.dataset,
                "scene_id": item.scene_id,
                "relative_path": item.path.relative_to(source_root).as_posix(),
            }
            for item in selected_files
        ],
    }
    manifest_path = output_root / "sample_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return manifest_path


def main() -> int:
    args = parse_args()
    validate_args(args)

    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    indexed = index_raw_data(source_root)
    selected_scene_ids = select_scene_ids(
        indexed=indexed,
        explicit_scene_ids=args.scene_ids,
        sample_count=args.sample_count,
        seed=args.seed,
        id_source=args.id_source,
        require_datasets=args.require_datasets,
    )
    selected_files, dataset_summary, missing_ids_by_dataset = collect_selected_files(indexed, selected_scene_ids)

    prepare_output_root(output_root, args.overwrite)
    copy_selected_files(selected_files, source_root, output_root)
    manifest_path = write_manifest(
        output_root=output_root,
        source_root=source_root,
        selected_scene_ids=selected_scene_ids,
        selected_files=selected_files,
        dataset_summary=dataset_summary,
        args=args,
    )

    print(f"Selected scene IDs: {', '.join(str(scene_id) for scene_id in selected_scene_ids)}")
    print(f"Copied files: {len(selected_files)}")
    print(f"Sample directory: {output_root}")
    print(f"Manifest: {manifest_path}")
    for dataset, summary in dataset_summary.items():
        print(
            f"- {dataset}: {summary['selected_file_count']} files for "
            f"{summary['selected_scene_id_count']} / {len(selected_scene_ids)} selected IDs"
        )
    if missing_ids_by_dataset:
        print("Datasets with missing selected IDs are recorded in sample_manifest.json.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
