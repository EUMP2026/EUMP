from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from tqdm.auto import tqdm

from urban_embedding.core.paths import EUMP_DATA_DIR
from urban_embedding.data.processed.alignment import import_zarr
from urban_embedding.data.processed.manifest import (
    load_eump_store_schema,
    read_jsonl,
    scene_years_manifest_path,
    scenes_manifest_path,
)
from urban_embedding.data.processed.reader import coerce_channel_names, normalize_scene_id
from urban_embedding.data.processed.schema import CITY_MASK_SOURCE


CSV_FIELDS = (
    "row_type",
    "scene_id",
    "year",
    "source",
    "indicator",
    "channel_index",
    "modality",
    "storage_level",
    "used_city_mask",
    "mean",
    "sum",
    "variance",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export raw city-level mean/sum/variance statistics for each EUMP source. "
            "Raster sources are summarized over a center crop or the full scene; scalar "
            "sources are used directly. Yearly rows are followed by per-city year_mean rows."
        )
    )
    parser.add_argument(
        "--eump-root",
        type=Path,
        default=Path(EUMP_DATA_DIR),
        help="EUMP scene-store root. Default: data/processed_v2.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output CSV path. Default: <eump-root>/exports/city_indicator_stats.csv.",
    )
    parser.add_argument(
        "--sources",
        nargs="*",
        default=None,
        help="Optional source names to export. Default: all non-derived schema sources.",
    )
    parser.add_argument(
        "--exclude-sources",
        nargs="*",
        default=(),
        help="Optional source names to exclude from export.",
    )
    parser.add_argument(
        "--scene-ids",
        nargs="*",
        default=None,
        help="Optional scene/city IDs to export. Default: all scenes.",
    )
    parser.add_argument(
        "--year-range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        default=None,
        help="Optional inclusive year range for yearly sources.",
    )
    parser.add_argument(
        "--raster-region",
        choices=("center_crop", "full_scene"),
        default="center_crop",
        help="Raster region to summarize. Default: center_crop.",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=256,
        help="Center crop size for raster sources. Default: 256.",
    )
    parser.add_argument(
        "--use-city-mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use city_mask>0 when summarizing raster sources other than city_mask. Default: true.",
    )
    parser.add_argument(
        "--no-city-mask",
        dest="use_city_mask",
        action="store_false",
        help="Alias for --no-use-city-mask.",
    )
    parser.add_argument(
        "--include-static-year-mean",
        action="store_true",
        help="Also emit a year_mean row for static sources. Default: static sources emit only row_type=static.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output path if it already exists.",
    )
    return parser.parse_args(argv)


def _natural_sort_key(value: Any) -> tuple[int, int | str]:
    text = str(value)
    try:
        return 0, int(text)
    except ValueError:
        return 1, text


def _selected_sources(
    source_specs: dict[str, dict[str, Any]],
    *,
    sources: Iterable[str] | None,
    exclude_sources: Iterable[str],
) -> list[str]:
    exclude = {str(source) for source in exclude_sources}
    if sources is None:
        selected = [
            name
            for name, spec in source_specs.items()
            if not bool(spec.get("is_derived", False))
        ]
    else:
        selected = [str(source) for source in sources]
        unknown = sorted(source for source in selected if source not in source_specs)
        if unknown:
            raise ValueError(f"Unknown source(s): {', '.join(unknown)}")
    return sorted((source for source in selected if source not in exclude), key=_natural_sort_key)


def _center_crop_bounds(height: int, width: int, crop_size: int) -> tuple[int, int, int, int]:
    crop_h = min(int(crop_size), int(height))
    crop_w = min(int(crop_size), int(width))
    top = max(0, (int(height) - crop_h) // 2)
    left = max(0, (int(width) - crop_w) // 2)
    return top, left, top + crop_h, left + crop_w


def _region_bounds(data_shape: tuple[int, ...], *, raster_region: str, crop_size: int) -> tuple[int, int, int, int]:
    height, width = int(data_shape[-2]), int(data_shape[-1])
    if raster_region == "full_scene":
        return 0, 0, height, width
    return _center_crop_bounds(height, width, crop_size)


def _crop_array(array: np.ndarray, bounds: tuple[int, int, int, int]) -> np.ndarray:
    top, left, bottom, right = bounds
    if array.ndim == 2:
        return array[top:bottom, left:right]
    return array[:, top:bottom, left:right]


def _load_city_mask(static_group, bounds: tuple[int, int, int, int]) -> np.ndarray | None:
    if CITY_MASK_SOURCE not in static_group:
        return None
    source_group = static_group[CITY_MASK_SOURCE]
    data = np.asarray(source_group["data"])
    valid_mask = np.asarray(source_group["valid_mask"], dtype=bool)
    if data.ndim != 3 or data.shape[0] < 1:
        return None
    cropped_data = _crop_array(data, bounds)[0]
    cropped_valid = _crop_array(valid_mask, bounds)[0]
    return (cropped_data > 0) & cropped_valid


def _finite_or_valid(data: np.ndarray) -> np.ndarray:
    if np.issubdtype(data.dtype, np.number):
        return np.isfinite(data)
    return np.ones_like(data, dtype=bool)


def _stats_from_values(data: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    channels = int(data.shape[0])
    flat_data = data.reshape(channels, -1).astype(np.float64, copy=False)
    flat_mask = mask.reshape(channels, -1).astype(bool, copy=False)
    valid_count = flat_mask.sum(axis=1).astype(np.float64)
    safe_data = np.where(flat_mask, flat_data, 0.0)
    sums = safe_data.sum(axis=1)
    sum_sq = np.square(safe_data).sum(axis=1)
    means = np.divide(sums, valid_count, out=np.full(channels, np.nan, dtype=np.float64), where=valid_count > 0)
    variances = np.divide(sum_sq, valid_count, out=np.full(channels, np.nan, dtype=np.float64), where=valid_count > 0)
    variances = variances - np.square(means)
    variances = np.where(valid_count > 0, np.maximum(variances, 0.0), np.nan)
    sums = np.where(valid_count > 0, sums, np.nan)
    return means, sums, variances, valid_count


def _float_or_empty(value: float | int | np.floating | np.integer | None) -> str:
    if value is None:
        return ""
    numeric = float(value)
    if math.isnan(numeric):
        return ""
    return f"{numeric:.10g}"


def _row(
    *,
    row_type: str,
    scene_id: Any,
    year: Any,
    source: str,
    indicator: str,
    channel_index: int,
    modality: str,
    storage_level: str,
    used_city_mask: bool,
    mean: float,
    sum_value: float,
    variance: float,
) -> dict[str, Any]:
    return {
        "row_type": row_type,
        "scene_id": str(scene_id),
        "year": "" if year is None else str(year),
        "source": str(source),
        "indicator": str(indicator),
        "channel_index": int(channel_index),
        "modality": str(modality),
        "storage_level": str(storage_level),
        "used_city_mask": bool(used_city_mask),
        "mean": _float_or_empty(mean),
        "sum": _float_or_empty(sum_value),
        "variance": _float_or_empty(variance),
    }


def _source_group_stats_rows(
    *,
    source_group,
    source_name: str,
    source_spec: dict[str, Any],
    scene_id: Any,
    year: Any,
    row_type: str,
    raster_region: str,
    crop_size: int,
    use_city_mask: bool,
    city_mask: np.ndarray | None,
) -> list[dict[str, Any]]:
    data = np.asarray(source_group["data"])
    valid_mask = np.asarray(source_group["valid_mask"], dtype=bool)
    modality = str(source_spec["modality"])
    storage_level = str(source_spec["storage_level"])

    if modality == "scalar":
        names = coerce_channel_names(source_name, dict(source_group.attrs), int(data.shape[0]))
        rows: list[dict[str, Any]] = []
        finite_mask = _finite_or_valid(data)
        valid = valid_mask.astype(bool, copy=False) & finite_mask
        for channel_index, name in enumerate(names):
            if bool(valid[channel_index]):
                value = float(data[channel_index])
                mean = sum_value = value
                variance = 0.0
            else:
                mean = sum_value = variance = float("nan")
            rows.append(
                _row(
                    row_type=row_type,
                    scene_id=scene_id,
                    year=year,
                    source=source_name,
                    indicator=name,
                    channel_index=channel_index,
                    modality=modality,
                    storage_level=storage_level,
                    used_city_mask=False,
                    mean=mean,
                    sum_value=sum_value,
                    variance=variance,
                )
            )
        return rows

    if data.ndim != 3:
        raise ValueError(f"Raster source {source_name!r} must have shape [C,H,W], got {data.shape}")

    bounds = _region_bounds(data.shape, raster_region=raster_region, crop_size=crop_size)
    cropped_data = _crop_array(data, bounds)
    cropped_valid = _crop_array(valid_mask, bounds) & _finite_or_valid(cropped_data)
    names = coerce_channel_names(source_name, dict(source_group.attrs), int(cropped_data.shape[0]))

    effective_city_mask = None
    used_city_mask = False
    if use_city_mask and source_name != CITY_MASK_SOURCE and city_mask is not None:
        effective_city_mask = city_mask.astype(bool, copy=False)
        used_city_mask = True

    if effective_city_mask is None:
        stats_mask = cropped_valid
    else:
        stats_mask = cropped_valid & effective_city_mask[None, :, :]

    means, sums, variances, _valid_counts = _stats_from_values(cropped_data, stats_mask)
    rows = []
    for channel_index, name in enumerate(names):
        rows.append(
            _row(
                row_type=row_type,
                scene_id=scene_id,
                year=year,
                source=source_name,
                indicator=name,
                channel_index=channel_index,
                modality=modality,
                storage_level=storage_level,
                used_city_mask=used_city_mask,
                mean=float(means[channel_index]),
                sum_value=float(sums[channel_index]),
                variance=float(variances[channel_index]),
            )
        )
    return rows


def _numeric_value(row: dict[str, Any], field: str) -> float:
    value = row[field]
    if value == "":
        return float("nan")
    return float(value)


def _year_mean_rows(year_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in year_rows:
        key = (
            row["scene_id"],
            row["source"],
            row["indicator"],
            str(row["channel_index"]),
        )
        grouped[key].append(row)

    mean_rows: list[dict[str, Any]] = []
    for _, rows in sorted(grouped.items(), key=lambda item: item[0]):
        first = rows[0]
        averaged = dict(first)
        averaged["row_type"] = "year_mean"
        averaged["year"] = "mean"
        for field in ("mean", "sum", "variance"):
            values = np.asarray([_numeric_value(row, field) for row in rows], dtype=np.float64)
            if np.isfinite(values).any():
                averaged[field] = _float_or_empty(float(np.nanmean(values)))
            else:
                averaged[field] = ""
        mean_rows.append(averaged)
    return mean_rows


def build_city_indicator_stats(
    *,
    eump_root: str | Path,
    sources: Iterable[str] | None = None,
    exclude_sources: Iterable[str] = (),
    scene_ids: Iterable[str | int] | None = None,
    year_range: tuple[int, int] | None = None,
    raster_region: str = "center_crop",
    crop_size: int = 256,
    use_city_mask: bool = True,
    include_static_year_mean: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eump_root_path = Path(eump_root).expanduser().resolve()
    schema = load_eump_store_schema(eump_root_path)
    source_specs = schema["sources"]
    selected_sources = set(_selected_sources(source_specs, sources=sources, exclude_sources=exclude_sources))
    selected_scene_ids = None if scene_ids is None else {normalize_scene_id(scene_id) for scene_id in scene_ids}
    zarr = import_zarr()

    scene_rows = sorted(
        read_jsonl(scenes_manifest_path(eump_root_path)),
        key=lambda row: _natural_sort_key(row["scene_id"]),
    )
    year_rows = sorted(
        read_jsonl(scene_years_manifest_path(eump_root_path)),
        key=lambda row: (_natural_sort_key(row["scene_id"]), int(row["year"])),
    )

    scene_index = {normalize_scene_id(row["scene_id"]): row for row in scene_rows}
    years_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in year_rows:
        scene_key = normalize_scene_id(row["scene_id"])
        year = int(row["year"])
        if selected_scene_ids is not None and scene_key not in selected_scene_ids:
            continue
        if year_range is not None and not (int(year_range[0]) <= year <= int(year_range[1])):
            continue
        years_by_scene[scene_key].append(row)

    output_rows: list[dict[str, Any]] = []
    yearly_stat_rows: list[dict[str, Any]] = []
    static_stat_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    iterable = [
        row
        for row in scene_rows
        if selected_scene_ids is None or normalize_scene_id(row["scene_id"]) in selected_scene_ids
    ]
    for scene_row in tqdm(iterable, desc="export city indicator stats", unit="scene"):
        scene_key = normalize_scene_id(scene_row["scene_id"])
        static_group = zarr.open_group(str(eump_root_path / scene_row["static_path"]), mode="r")
        present_static = set(scene_row.get("present_sources", ()))
        static_sources = [
            source
            for source in sorted(selected_sources & present_static, key=_natural_sort_key)
            if source_specs[source]["storage_level"] == "static"
        ]

        for source_name in static_sources:
            source_group = static_group[source_name]
            source_spec = source_specs[source_name]
            if source_spec["modality"] == "raster":
                bounds = _region_bounds(np.asarray(source_group["data"]).shape, raster_region=raster_region, crop_size=crop_size)
                city_mask = _load_city_mask(static_group, bounds)
            else:
                city_mask = None
            rows = _source_group_stats_rows(
                source_group=source_group,
                source_name=source_name,
                source_spec=source_spec,
                scene_id=scene_row["scene_id"],
                year=None,
                row_type="static",
                raster_region=raster_region,
                crop_size=crop_size,
                use_city_mask=use_city_mask,
                city_mask=city_mask,
            )
            static_stat_rows.extend(rows)
            output_rows.extend(rows)

        for year_row in years_by_scene.get(scene_key, ()):
            year = int(year_row["year"])
            year_group = zarr.open_group(str(eump_root_path / year_row["year_path"]), mode="r")
            present_yearly = set(year_row.get("present_sources", ()))
            yearly_sources = [
                source
                for source in sorted(selected_sources & present_yearly, key=_natural_sort_key)
                if source_specs[source]["storage_level"] == "yearly"
            ]
            for source_name in yearly_sources:
                source_group = year_group[source_name]
                source_spec = source_specs[source_name]
                if source_spec["modality"] == "raster":
                    bounds = _region_bounds(np.asarray(source_group["data"]).shape, raster_region=raster_region, crop_size=crop_size)
                    city_mask = _load_city_mask(static_group, bounds)
                else:
                    city_mask = None
                rows = _source_group_stats_rows(
                    source_group=source_group,
                    source_name=source_name,
                    source_spec=source_spec,
                    scene_id=year_row["scene_id"],
                    year=year,
                    row_type="year",
                    raster_region=raster_region,
                    crop_size=crop_size,
                    use_city_mask=use_city_mask,
                    city_mask=city_mask,
                )
                yearly_stat_rows.extend(rows)
                output_rows.extend(rows)

            missing_selected = sorted(
                source
                for source in selected_sources
                if source_specs[source]["storage_level"] == "yearly" and source not in present_yearly
            )
            for source in missing_selected:
                skipped.append({"scene_id": scene_key, "year": year, "source": source, "reason": "source_missing"})

    year_mean_rows = _year_mean_rows(yearly_stat_rows)
    output_rows.extend(year_mean_rows)
    if include_static_year_mean:
        static_year_mean = []
        for row in static_stat_rows:
            copied = dict(row)
            copied["row_type"] = "year_mean"
            copied["year"] = "mean"
            static_year_mean.append(copied)
        output_rows.extend(static_year_mean)
    else:
        static_year_mean = []

    summary = {
        "eump_root": str(eump_root_path),
        "sources": sorted(selected_sources, key=_natural_sort_key),
        "scene_count": len(iterable),
        "rows": len(output_rows),
        "static_rows": len(static_stat_rows),
        "year_rows": len(yearly_stat_rows),
        "year_mean_rows": len(year_mean_rows) + len(static_year_mean),
        "raster_region": raster_region,
        "crop_size": int(crop_size),
        "use_city_mask": bool(use_city_mask),
        "skipped_count": len(skipped),
        "skipped_preview": skipped[:20],
    }
    missing_scenes = []
    if selected_scene_ids is not None:
        available = set(scene_index)
        missing_scenes = sorted(selected_scene_ids - available, key=_natural_sort_key)
        if missing_scenes:
            summary["missing_scene_ids"] = missing_scenes
    return output_rows, summary


def write_city_indicator_stats_csv(rows: list[dict[str, Any]], output_path: str | Path, *, overwrite: bool = False) -> Path:
    target = Path(output_path).expanduser().resolve()
    if target.exists() and not overwrite:
        raise FileExistsError(f"Output already exists, pass --overwrite to replace it: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})
    return target


def export_city_indicator_stats(args: argparse.Namespace) -> dict[str, Any]:
    eump_root = args.eump_root.expanduser().resolve()
    output_path = (
        eump_root / "exports" / "city_indicator_stats.csv"
        if args.output_path is None
        else args.output_path.expanduser().resolve()
    )
    year_range = None if args.year_range is None else (int(args.year_range[0]), int(args.year_range[1]))
    rows, summary = build_city_indicator_stats(
        eump_root=eump_root,
        sources=args.sources,
        exclude_sources=args.exclude_sources,
        scene_ids=args.scene_ids,
        year_range=year_range,
        raster_region=args.raster_region,
        crop_size=int(args.crop_size),
        use_city_mask=bool(args.use_city_mask),
        include_static_year_mean=bool(args.include_static_year_mean),
    )
    path = write_city_indicator_stats_csv(rows, output_path, overwrite=bool(args.overwrite))
    return {**summary, "output_path": str(path)}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = export_city_indicator_stats(args)
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
