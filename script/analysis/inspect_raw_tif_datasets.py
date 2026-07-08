#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import rasterio


TIFF_SUFFIXES = {".tif", ".tiff"}
DEFAULT_ROAD_DENSITY_YEARS = tuple(str(year) for year in range(2008, 2023))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a raw TIFF dataset directory and summarize file counts, "
            "basic raster metadata, and road_density year coverage."
        )
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Dataset root to inspect, e.g. data/raw_data_sample",
    )
    parser.add_argument(
        "--preview-count",
        type=int,
        default=5,
        help="Number of sample filenames to include for each dataset.",
    )
    parser.add_argument(
        "--road-density-years",
        nargs="*",
        default=list(DEFAULT_ROAD_DENSITY_YEARS),
        help="Expected road_density year folders. Default: 2008-2022.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of a human-readable summary.",
    )
    return parser.parse_args()


def iter_direct_tiffs(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in TIFF_SUFFIXES
    )


def counter_to_sorted_dict(counter: Counter) -> dict[str, int]:
    return {str(key): counter[key] for key in sorted(counter, key=str)}


def normalize_nodata(value) -> str:
    return "None" if value is None else str(value)


def collect_raster_metadata(files: Iterable[Path]) -> dict[str, object]:
    shape_counter: Counter = Counter()
    band_counter: Counter = Counter()
    dtype_counter: Counter = Counter()
    crs_counter: Counter = Counter()
    resolution_counter: Counter = Counter()
    nodata_counter: Counter = Counter()
    driver_counter: Counter = Counter()

    for path in files:
        with rasterio.open(path) as src:
            shape_counter[(src.height, src.width)] += 1
            band_counter[src.count] += 1
            driver_counter[src.driver] += 1
            for dtype in src.dtypes:
                dtype_counter[dtype] += 1
            crs_counter[str(src.crs) if src.crs is not None else "None"] += 1
            resolution_counter[(abs(src.transform.a), abs(src.transform.e))] += 1
            nodata_counter[normalize_nodata(src.nodata)] += 1

    return {
        "shape_counts": counter_to_sorted_dict(shape_counter),
        "band_counts": counter_to_sorted_dict(band_counter),
        "dtype_counts": counter_to_sorted_dict(dtype_counter),
        "crs_counts": counter_to_sorted_dict(crs_counter),
        "resolution_counts": counter_to_sorted_dict(resolution_counter),
        "nodata_counts": counter_to_sorted_dict(nodata_counter),
        "driver_counts": counter_to_sorted_dict(driver_counter),
    }


def inspect_dataset(dataset_dir: Path, preview_count: int, road_density_years: list[str]) -> dict[str, object]:
    info: dict[str, object] = {
        "name": dataset_dir.name,
        "path": str(dataset_dir),
    }

    if dataset_dir.name == "road_density":
        year_dirs = sorted(path for path in dataset_dir.iterdir() if path.is_dir())
        files: list[Path] = []
        year_counts: dict[str, int] = {}
        for year_dir in year_dirs:
            year_files = iter_direct_tiffs(year_dir)
            year_counts[year_dir.name] = len(year_files)
            files.extend(year_files)

        expected_years = set(road_density_years)
        existing_years = set(year_counts)
        info["type"] = "year_subdirs"
        info["year_counts"] = year_counts
        info["missing_expected_years"] = sorted(expected_years - existing_years)
        info["unexpected_years"] = sorted(existing_years - expected_years)
    else:
        files = iter_direct_tiffs(dataset_dir)
        info["type"] = "direct_tiffs"

    info["file_count"] = len(files)
    info["preview_files"] = [path.name for path in files[:preview_count]]

    if files:
        info["raster_metadata"] = collect_raster_metadata(files)
    else:
        info["raster_metadata"] = {}

    return info


def build_summary(root: Path, preview_count: int, road_density_years: list[str]) -> dict[str, object]:
    dataset_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    datasets = [
        inspect_dataset(dataset_dir, preview_count=preview_count, road_density_years=road_density_years)
        for dataset_dir in dataset_dirs
    ]
    return {
        "root": str(root),
        "dataset_count": len(datasets),
        "datasets": datasets,
    }


def format_summary(summary: dict[str, object]) -> str:
    lines: list[str] = []
    lines.append(f"Root: {summary['root']}")
    lines.append(f"Datasets: {summary['dataset_count']}")

    for dataset in summary["datasets"]:
        lines.append("")
        lines.append(f"[{dataset['name']}]")
        lines.append(f"type: {dataset['type']}")
        lines.append(f"file_count: {dataset['file_count']}")

        preview_files = dataset.get("preview_files", [])
        if preview_files:
            lines.append("preview_files:")
            for name in preview_files:
                lines.append(f"  - {name}")

        if dataset["name"] == "road_density":
            lines.append("year_counts:")
            for year, count in dataset.get("year_counts", {}).items():
                lines.append(f"  - {year}: {count}")
            missing_years = dataset.get("missing_expected_years", [])
            unexpected_years = dataset.get("unexpected_years", [])
            lines.append(f"missing_expected_years: {missing_years}")
            lines.append(f"unexpected_years: {unexpected_years}")

        raster_metadata = dataset.get("raster_metadata", {})
        if raster_metadata:
            lines.append("raster_metadata:")
            for key in (
                "shape_counts",
                "band_counts",
                "dtype_counts",
                "crs_counts",
                "resolution_counts",
                "nodata_counts",
                "driver_counts",
            ):
                value = raster_metadata.get(key, {})
                lines.append(f"  {key}: {value}")

    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Root is not a directory: {root}")

    summary = build_summary(
        root=root,
        preview_count=args.preview_count,
        road_density_years=list(args.road_density_years),
    )

    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=True))
    else:
        print(format_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
