#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

import numpy as np
import rasterio
from tqdm.auto import tqdm


TIFF_SUFFIXES = {".tif", ".tiff"}
DEFAULT_ROAD_DENSITY_YEARS = tuple(str(year) for year in range(2008, 2023))

# Source resolution groups used before resampling:
# - BH8 is the reference grid
# - other products are grouped by their source resolution before resampling
RESOLUTION_GROUPS = {
    "ref_250m": ["BH8_v1", "LCZ", "road_density"],
    "src_500m": ["VI3_v1", "land_cover"],
    "src_1000m": ["lst_v1", "PM25", "electricity"],
    "src_nl_like": ["NL_v1"],
    "src_precip_like": ["precip_v1"],
}

ROBUST_Z_THRESHOLD = 6.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check current TIFF datasets for nodata consistency, resolution consistency, "
            "and band-level distribution anomalies."
        )
    )
    parser.add_argument(
        "root",
        type=Path,
        nargs="?",
        default=Path("data/raw_data_sample"),
        help="Dataset root to inspect. Default: data/raw_data_sample",
    )
    parser.add_argument(
        "--road-density-years",
        nargs="*",
        default=list(DEFAULT_ROAD_DENSITY_YEARS),
        help="Expected road_density year folders. Default: 2008-2022.",
    )
    parser.add_argument(
        "--resolution-rel-tol",
        type=float,
        default=0.01,
        help="Relative tolerance for group resolution checks. Default: 0.01 (1%%).",
    )
    parser.add_argument(
        "--max-anomaly-examples",
        type=int,
        default=5,
        help="Maximum number of anomaly file examples to keep per band.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON with full band details.",
    )
    return parser.parse_args()


def iter_direct_tiffs(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in TIFF_SUFFIXES
    )


def iter_dataset_files(dataset_dir: Path, road_density_years: list[str]) -> tuple[list[Path], dict[str, int], list[str]]:
    if dataset_dir.name != "road_density":
        return iter_direct_tiffs(dataset_dir), {}, []

    existing_years = sorted(path.name for path in dataset_dir.iterdir() if path.is_dir())
    files: list[Path] = []
    year_counts: dict[str, int] = {}
    for year in existing_years:
        year_dir = dataset_dir / year
        year_files = iter_direct_tiffs(year_dir)
        year_counts[year] = len(year_files)
        files.extend(year_files)

    missing_years = sorted(set(road_density_years) - set(existing_years))
    return sorted(files), year_counts, missing_years


def normalize_nodata(value) -> str:
    return "None" if value is None else str(value)


def is_zero_like_nodata(value) -> bool:
    if value is None:
        return False
    try:
        return math.isfinite(float(value)) and float(value) == 0.0
    except (TypeError, ValueError):
        return False


def build_nodata_masks(
    band: np.ndarray,
    nodata,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    finite_mask = np.isfinite(band)
    declared_nodata_mask = np.zeros_like(finite_mask, dtype=bool)
    excluded_nodata_mask = np.zeros_like(finite_mask, dtype=bool)

    if nodata is None:
        return finite_mask, declared_nodata_mask, excluded_nodata_mask, "no_declared_nodata"

    if isinstance(nodata, (float, np.floating)) and np.isnan(nodata):
        declared_nodata_mask = np.isnan(band)
        excluded_nodata_mask = declared_nodata_mask
        return finite_mask, declared_nodata_mask, excluded_nodata_mask, "exclude_nan_nodata"

    declared_nodata_mask = band == nodata
    if is_zero_like_nodata(nodata):
        return finite_mask, declared_nodata_mask, excluded_nodata_mask, "keep_zero_in_distribution"

    excluded_nodata_mask = declared_nodata_mask
    return finite_mask, declared_nodata_mask, excluded_nodata_mask, "exclude_declared_nodata"


def infer_band_names(src: rasterio.DatasetReader) -> list[str]:
    descriptions = list(src.descriptions or [])
    names: list[str] = []
    for idx in range(src.count):
        desc = descriptions[idx] if idx < len(descriptions) else None
        if desc:
            names.append(desc)
            continue
        band_tags = src.tags(idx + 1)
        if band_tags.get("BandName"):
            names.append(str(band_tags["BandName"]))
            continue
        names.append(f"band_{idx + 1}")
    return names


def robust_z_scores(values: list[float]) -> list[float | None]:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    if len(finite) < 5:
        return [None] * len(values)

    med = float(np.median(finite))
    abs_devs = [abs(value - med) for value in finite]
    mad = float(np.median(abs_devs))
    if mad == 0:
        return [None] * len(values)

    scores: list[float | None] = []
    for value in values:
        if value is None or not math.isfinite(value):
            scores.append(None)
        else:
            scores.append(0.6745 * (value - med) / mad)
    return scores


def summarize_metric(values: list[float | None]) -> dict[str, float | int | None]:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    if not finite:
        return {
            "count": 0,
            "min": None,
            "median": None,
            "max": None,
        }
    return {
        "count": len(finite),
        "min": float(min(finite)),
        "median": float(np.median(finite)),
        "max": float(max(finite)),
    }


def analyze_band_records(records: list[dict[str, object]], max_examples: int) -> dict[str, object]:
    metric_names = [
        "valid_min",
        "valid_p01",
        "valid_p50",
        "valid_p99",
        "valid_max",
        "zero_fraction",
        "declared_nodata_fraction",
        "excluded_nodata_fraction",
        "valid_fraction",
    ]

    metrics_summary = {
        metric: summarize_metric([record[metric] for record in records])
        for metric in metric_names
    }

    anomaly_reasons_by_file: defaultdict[str, list[str]] = defaultdict(list)
    for metric in (
        "valid_p01",
        "valid_p50",
        "valid_p99",
        "zero_fraction",
        "excluded_nodata_fraction",
    ):
        values = [record[metric] for record in records]
        scores = robust_z_scores(values)
        for record, score in zip(records, scores):
            if score is None or abs(score) < ROBUST_Z_THRESHOLD:
                continue
            direction = "high" if score > 0 else "low"
            anomaly_reasons_by_file[str(record["file"])].append(
                f"{metric}:{direction}:robust_z={score:.2f}"
            )

    anomaly_examples = [
        {
            "file": file_name,
            "reasons": reasons,
        }
        for file_name, reasons in sorted(anomaly_reasons_by_file.items())[:max_examples]
    ]

    return {
        "file_count": len(records),
        "metrics": metrics_summary,
        "anomaly_count": len(anomaly_reasons_by_file),
        "anomaly_examples": anomaly_examples,
    }


def check_single_dataset(
    dataset_dir: Path,
    road_density_years: list[str],
    max_anomaly_examples: int,
) -> dict[str, object]:
    files, year_counts, missing_years = iter_dataset_files(dataset_dir, road_density_years)

    nodata_counter: Counter = Counter()
    crs_counter: Counter = Counter()
    nodata_strategy_counter: Counter = Counter()
    xres_values: list[float] = []
    yres_values: list[float] = []
    band_records: defaultdict[str, list[dict[str, object]]] = defaultdict(list)

    for path in tqdm(files, desc=f"Check {dataset_dir.name}", unit="file"):
        with rasterio.open(path) as src:
            nodata = src.nodata
            nodata_counter[normalize_nodata(nodata)] += 1
            crs_counter[str(src.crs) if src.crs is not None else "None"] += 1
            xres_values.append(abs(float(src.transform.a)))
            yres_values.append(abs(float(src.transform.e)))

            band_names = infer_band_names(src)
            for band_idx, band_name in enumerate(band_names, start=1):
                band = src.read(band_idx)
                band = np.asarray(band)
                (
                    finite_mask,
                    declared_nodata_mask,
                    excluded_nodata_mask,
                    nodata_strategy,
                ) = build_nodata_masks(band, nodata)
                nodata_strategy_counter[nodata_strategy] += 1

                valid_mask = finite_mask & ~excluded_nodata_mask
                valid_values = band[valid_mask]

                if valid_values.size:
                    valid_min = float(np.min(valid_values))
                    valid_p01 = float(np.percentile(valid_values, 1))
                    valid_p50 = float(np.percentile(valid_values, 50))
                    valid_p99 = float(np.percentile(valid_values, 99))
                    valid_max = float(np.max(valid_values))
                else:
                    valid_min = valid_p01 = valid_p50 = valid_p99 = valid_max = None

                band_records[band_name].append(
                    {
                        "file": path.name,
                        "valid_min": valid_min,
                        "valid_p01": valid_p01,
                        "valid_p50": valid_p50,
                        "valid_p99": valid_p99,
                        "valid_max": valid_max,
                        "zero_fraction": float(np.mean((band == 0) & finite_mask)),
                        "declared_nodata_fraction": float(np.mean(declared_nodata_mask)),
                        "excluded_nodata_fraction": float(np.mean(excluded_nodata_mask)),
                        "valid_fraction": float(np.mean(valid_mask)),
                    }
                )

    result: dict[str, object] = {
        "dataset": dataset_dir.name,
        "file_count": len(files),
        "nodata_counts": dict(sorted(nodata_counter.items())),
        "nodata_strategies": dict(sorted(nodata_strategy_counter.items())),
        "crs_counts": dict(sorted(crs_counter.items())),
        "warnings": [],
        "band_profiles": {},
    }

    if dataset_dir.name == "road_density":
        result["year_counts"] = year_counts
        if missing_years:
            result["notes"] = {
                "missing_expected_years": missing_years,
            }

    if xres_values and yres_values:
        xres_min = min(xres_values)
        xres_max = max(xres_values)
        yres_min = min(yres_values)
        yres_max = max(yres_values)
        xres_median = median(xres_values)
        yres_median = median(yres_values)
        result["resolution"] = {
            "xres_min": xres_min,
            "xres_median": xres_median,
            "xres_max": xres_max,
            "yres_min": yres_min,
            "yres_median": yres_median,
            "yres_max": yres_max,
            "xres_rel_spread": 0.0 if xres_median == 0 else (xres_max - xres_min) / xres_median,
            "yres_rel_spread": 0.0 if yres_median == 0 else (yres_max - yres_min) / yres_median,
        }

    total_band_anomalies = 0
    anomalous_bands: list[str] = []
    for band_name in sorted(band_records):
        profile = analyze_band_records(band_records[band_name], max_examples=max_anomaly_examples)
        result["band_profiles"][band_name] = profile
        if profile["anomaly_count"] > 0:
            anomalous_bands.append(band_name)
            total_band_anomalies += int(profile["anomaly_count"])

    result["band_summary"] = {
        "band_count": len(result["band_profiles"]),
        "anomalous_bands": anomalous_bands,
        "total_anomaly_files_across_bands": total_band_anomalies,
    }

    return result


def build_group_resolution_checks(
    dataset_results: list[dict[str, object]],
    resolution_rel_tol: float,
) -> list[dict[str, object]]:
    dataset_map = {item["dataset"]: item for item in dataset_results}
    checks: list[dict[str, object]] = []

    for group_name, datasets in RESOLUTION_GROUPS.items():
        present = [
            name for name in datasets
            if name in dataset_map and "resolution" in dataset_map[name]
        ]
        if len(present) < 2:
            continue

        x_medians = {name: dataset_map[name]["resolution"]["xres_median"] for name in present}
        y_medians = {name: dataset_map[name]["resolution"]["yres_median"] for name in present}

        anchor_name = present[0]
        anchor_x = x_medians[anchor_name]
        anchor_y = y_medians[anchor_name]

        mismatches: list[str] = []
        for name in present[1:]:
            x_ok = math.isclose(x_medians[name], anchor_x, rel_tol=resolution_rel_tol)
            y_ok = math.isclose(y_medians[name], anchor_y, rel_tol=resolution_rel_tol)
            if not (x_ok and y_ok):
                mismatches.append(name)

        checks.append(
            {
                "group": group_name,
                "datasets": present,
                "anchor": anchor_name,
                "xres_medians": x_medians,
                "yres_medians": y_medians,
                "status": "ok" if not mismatches else "warning",
                "mismatches": mismatches,
            }
        )

    return checks


def build_report(
    root: Path,
    road_density_years: list[str],
    resolution_rel_tol: float,
    max_anomaly_examples: int,
) -> dict[str, object]:
    dataset_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    datasets = [
        check_single_dataset(
            dataset_dir,
            road_density_years=road_density_years,
            max_anomaly_examples=max_anomaly_examples,
        )
        for dataset_dir in dataset_dirs
    ]
    return {
        "root": str(root),
        "dataset_count": len(datasets),
        "datasets": datasets,
        "group_resolution_checks": build_group_resolution_checks(
            datasets,
            resolution_rel_tol=resolution_rel_tol,
        ),
    }


def format_float(value: float | None) -> str:
    return "None" if value is None else f"{value:.10f}"


def format_metric_triplet(metric: dict[str, object]) -> str:
    return (
        f"[{format_float(metric['min'])}, "
        f"{format_float(metric['median'])}, "
        f"{format_float(metric['max'])}]"
    )


def format_report(report: dict[str, object]) -> str:
    lines: list[str] = []
    lines.append(f"Root: {report['root']}")
    lines.append(f"Datasets: {report['dataset_count']}")

    for item in report["datasets"]:
        lines.append("")
        lines.append(f"[{item['dataset']}]")
        lines.append(f"file_count: {item['file_count']}")
        lines.append(f"nodata_counts: {item['nodata_counts']}")
        lines.append(f"nodata_strategies: {item.get('nodata_strategies', {})}")
        lines.append(f"crs_counts: {item['crs_counts']}")

        resolution = item.get("resolution")
        if resolution:
            lines.append(
                "resolution: "
                f"x={format_metric_triplet({'min': resolution['xres_min'], 'median': resolution['xres_median'], 'max': resolution['xres_max']})} "
                f"y={format_metric_triplet({'min': resolution['yres_min'], 'median': resolution['yres_median'], 'max': resolution['yres_max']})}"
            )
            lines.append(
                "resolution_spread: "
                f"x={resolution['xres_rel_spread']:.6f}, y={resolution['yres_rel_spread']:.6f}"
            )

        if item["dataset"] == "road_density":
            lines.append(f"year_counts: {item.get('year_counts', {})}")
            if item.get("notes"):
                lines.append(f"notes: {item['notes']}")

        band_summary = item.get("band_summary", {})
        lines.append(
            "band_summary: "
            f"band_count={band_summary.get('band_count', 0)}, "
            f"anomalous_bands={band_summary.get('anomalous_bands', [])}, "
            f"total_anomaly_files_across_bands={band_summary.get('total_anomaly_files_across_bands', 0)}"
        )

        anomalous_bands = band_summary.get("anomalous_bands", [])
        for band_name in anomalous_bands:
            profile = item["band_profiles"][band_name]
            lines.append(f"  [band:{band_name}]")
            lines.append(
                "    p01="
                f"{format_metric_triplet(profile['metrics']['valid_p01'])} "
                "p50="
                f"{format_metric_triplet(profile['metrics']['valid_p50'])} "
                "p99="
                f"{format_metric_triplet(profile['metrics']['valid_p99'])}"
            )
            lines.append(
                "    zero_fraction="
                f"{format_metric_triplet(profile['metrics']['zero_fraction'])} "
                "declared_nodata_fraction="
                f"{format_metric_triplet(profile['metrics']['declared_nodata_fraction'])} "
                "excluded_nodata_fraction="
                f"{format_metric_triplet(profile['metrics']['excluded_nodata_fraction'])}"
            )
            for example in profile["anomaly_examples"]:
                lines.append(f"    anomaly_file: {example['file']} -> {example['reasons']}")

        if item["warnings"]:
            lines.append(f"warnings: {item['warnings']}")
        else:
            lines.append("warnings: []")

    if report["group_resolution_checks"]:
        lines.append("")
        lines.append("[group_resolution_checks]")
        for check in report["group_resolution_checks"]:
            lines.append(
                f"- {check['group']}: status={check['status']}, "
                f"anchor={check['anchor']}, mismatches={check['mismatches']}"
            )
            lines.append(f"  xres_medians={check['xres_medians']}")
            lines.append(f"  yres_medians={check['yres_medians']}")

    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Root is not a directory: {root}")

    report = build_report(
        root=root,
        road_density_years=list(args.road_density_years),
        resolution_rel_tol=args.resolution_rel_tol,
        max_anomaly_examples=args.max_anomaly_examples,
    )

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=True))
    else:
        print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
