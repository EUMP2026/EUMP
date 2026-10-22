#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import tempfile
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rasterio
from multiprocessing import get_context

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / f"matplotlib-cache-urban-embedding-{os.getpid()}"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


TIFF_SUFFIXES = {".tif", ".tiff"}
DEFAULT_ROAD_DENSITY_YEARS = tuple(str(year) for year in range(2008, 2023))
SUMMARY_FIELDS = [
    "aggregation_level",
    "dataset",
    "band_name",
    "channel_group",
    "source_band_count",
    "file_count",
    "total_pixels",
    "valid_pixels",
    "finite_pixels",
    "sample_size",
    "declared_nodata_value",
    "nodata_strategy",
    "actual_nan_fraction",
    "actual_inf_fraction",
    "zero_fraction",
    "declared_nodata_fraction",
    "excluded_nodata_fraction",
    "valid_min",
    "sample_p01",
    "sample_p05",
    "sample_p25",
    "sample_p50",
    "sample_p75",
    "sample_p95",
    "sample_p99",
    "valid_max",
    "valid_mean",
    "valid_std",
    "plot_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate per-channel distribution plots and summary statistics for TIFF datasets."
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
        "--out",
        type=Path,
        default=None,
        help=(
            "Output directory. Default: data/channel_distribution_report/<root-name>"
        ),
    )
    parser.add_argument(
        "--road-density-years",
        nargs="*",
        default=list(DEFAULT_ROAD_DENSITY_YEARS),
        help="Expected road_density year folders. Default: 2008-2022.",
    )
    parser.add_argument(
        "--aggregation-level",
        choices=["by_group", "by_band", "both"],
        default="both",
        help="Which aggregation outputs to generate. Default: both.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=max(1, min(4, os.cpu_count() or 1)),
        help=(
            "Number of dataset workers. Use 1 for serial execution. "
            "Default: min(4, cpu_count)."
        ),
    )
    parser.add_argument(
        "--parallel-backend",
        choices=["auto", "process", "thread"],
        default="auto",
        help=(
            "Parallel backend for jobs>1. "
            "auto tries processes first and falls back to threads."
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Emit one progress update every N processed files inside a worker. Default: 1.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress display.",
    )
    parser.add_argument(
        "--max-samples-per-band",
        type=int,
        default=100_000,
        help="Maximum sampled pixels per band for percentile estimation and plotting.",
    )
    parser.add_argument(
        "--max-samples-per-group",
        type=int,
        default=None,
        help=(
            "Maximum sampled pixels per grouped variable for percentile estimation and plotting. "
            "Defaults to --max-samples-per-band."
        ),
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=80,
        help="Histogram bin count for continuous-valued bands. Default: 80.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="PNG DPI. Default: 150.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling. Default: 42.",
    )
    return parser.parse_args()


def iter_direct_tiffs(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in TIFF_SUFFIXES
    )


def iter_dataset_files(
    dataset_dir: Path,
    road_density_years: list[str],
) -> tuple[list[Path], dict[str, int], list[str]]:
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


def collect_dataset_dirs(root: Path) -> list[Path]:
    dataset_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if iter_direct_tiffs(root):
        return [root, *dataset_dirs]
    return dataset_dirs


def normalize_nodata(value) -> str:
    return "None" if value is None else str(value)


def is_zero_like_nodata(value) -> bool:
    if value is None:
        return False
    try:
        return math.isfinite(float(value)) and float(value) == 0.0
    except (TypeError, ValueError):
        return False


def infer_nodata_strategy(nodata) -> str:
    if nodata is None:
        return "no_declared_nodata"
    if isinstance(nodata, (float, np.floating)) and np.isnan(nodata):
        return "exclude_nan_nodata"
    if is_zero_like_nodata(nodata):
        return "keep_zero_in_distribution"
    return "exclude_declared_nodata"


def build_nodata_masks(
    band: np.ndarray,
    nodata,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite_mask = np.isfinite(band)
    declared_nodata_mask = np.zeros_like(finite_mask, dtype=bool)
    excluded_nodata_mask = np.zeros_like(finite_mask, dtype=bool)

    if nodata is None:
        return finite_mask, declared_nodata_mask, excluded_nodata_mask

    if isinstance(nodata, (float, np.floating)) and np.isnan(nodata):
        declared_nodata_mask = np.isnan(band)
        excluded_nodata_mask = declared_nodata_mask
        return finite_mask, declared_nodata_mask, excluded_nodata_mask

    declared_nodata_mask = band == nodata
    if is_zero_like_nodata(nodata):
        return finite_mask, declared_nodata_mask, excluded_nodata_mask

    excluded_nodata_mask = declared_nodata_mask
    return finite_mask, declared_nodata_mask, excluded_nodata_mask


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


def collapse_band_name(band_name: str) -> str:
    if re.fullmatch(r"[Bb]and_\d+", band_name):
        return band_name
    return re.sub(r"_\d+$", "", band_name)


def sanitize_name(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return sanitized or "band"


def format_float(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "None"
    return f"{value:.6f}"


def ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator) / float(denominator)


def sample_chunk(values: np.ndarray, limit: int, rng: np.random.Generator) -> np.ndarray:
    if values.size <= limit:
        return values.astype(np.float64, copy=False)
    indices = rng.choice(values.size, size=limit, replace=False)
    return values[indices].astype(np.float64, copy=False)


def compute_quantiles(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {
            "sample_p01": None,
            "sample_p05": None,
            "sample_p25": None,
            "sample_p50": None,
            "sample_p75": None,
            "sample_p95": None,
            "sample_p99": None,
        }
    return {
        "sample_p01": float(np.percentile(values, 1)),
        "sample_p05": float(np.percentile(values, 5)),
        "sample_p25": float(np.percentile(values, 25)),
        "sample_p50": float(np.percentile(values, 50)),
        "sample_p75": float(np.percentile(values, 75)),
        "sample_p95": float(np.percentile(values, 95)),
        "sample_p99": float(np.percentile(values, 99)),
    }


def is_integer_like(values: np.ndarray) -> bool:
    if values.size == 0:
        return False
    return bool(np.all(np.isclose(values, np.round(values), atol=1e-6)))


def use_bar_plot(values: np.ndarray) -> bool:
    if values.size == 0 or not is_integer_like(values):
        return False
    unique_values = np.unique(values.astype(np.int64))
    return unique_values.size <= 32


class ProgressTracker:
    def __init__(self, total: int, enabled: bool, desc: str) -> None:
        self.total = total
        self.enabled = enabled and total > 0
        self.desc = desc
        self.current = 0
        self.bar = tqdm(total=total, desc=desc, unit="file", dynamic_ncols=True) if self.enabled else None

    def update(self, count: int) -> None:
        if not self.enabled or count <= 0:
            return
        self.current += count
        self.bar.update(count)

    def write(self, message: str) -> None:
        if not self.enabled:
            return
        self.bar.write(message)

    def close(self) -> None:
        if self.bar is not None:
            self.bar.close()


@dataclass
class BandAccumulator:
    aggregation_level: str
    dataset_name: str
    band_name: str
    channel_group: str
    nodata_value: str
    nodata_strategy: str
    sample_limit: int
    per_file_sample_limit: int
    source_band_names: set[str] = field(default_factory=set)
    sample_buffer: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    sampled_candidate_count: int = 0
    file_count: int = 0
    total_pixels: int = 0
    finite_pixels: int = 0
    valid_pixels: int = 0
    nan_pixels: int = 0
    inf_pixels: int = 0
    zero_pixels: int = 0
    declared_nodata_pixels: int = 0
    excluded_nodata_pixels: int = 0
    valid_sum: float = 0.0
    valid_sum_sq: float = 0.0
    valid_min: float | None = None
    valid_max: float | None = None

    def update(
        self,
        band: np.ndarray,
        finite_mask: np.ndarray,
        declared_nodata_mask: np.ndarray,
        excluded_nodata_mask: np.ndarray,
        source_band_name: str,
        rng: np.random.Generator,
    ) -> None:
        valid_mask = finite_mask & ~excluded_nodata_mask
        valid_values = band[valid_mask]

        self.source_band_names.add(source_band_name)
        self.file_count += 1
        self.total_pixels += int(band.size)
        self.finite_pixels += int(np.count_nonzero(finite_mask))
        self.valid_pixels += int(valid_values.size)
        self.nan_pixels += int(np.count_nonzero(np.isnan(band)))
        self.inf_pixels += int(np.count_nonzero(np.isinf(band)))
        self.zero_pixels += int(np.count_nonzero((band == 0) & finite_mask))
        self.declared_nodata_pixels += int(np.count_nonzero(declared_nodata_mask))
        self.excluded_nodata_pixels += int(np.count_nonzero(excluded_nodata_mask))

        if valid_values.size == 0:
            return

        valid_values = valid_values.astype(np.float64, copy=False)
        current_min = float(np.min(valid_values))
        current_max = float(np.max(valid_values))
        self.valid_min = current_min if self.valid_min is None else min(self.valid_min, current_min)
        self.valid_max = current_max if self.valid_max is None else max(self.valid_max, current_max)
        self.valid_sum += float(np.sum(valid_values))
        self.valid_sum_sq += float(np.sum(valid_values * valid_values))

        sampled = sample_chunk(valid_values, self.per_file_sample_limit, rng)
        self._update_sample_buffer(sampled, rng)

    def _update_sample_buffer(
        self,
        sampled_values: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        if sampled_values.size == 0:
            return

        sampled_values = sampled_values.astype(np.float64, copy=False)
        if self.sample_buffer.size < self.sample_limit:
            remaining = self.sample_limit - self.sample_buffer.size
            initial_chunk = sampled_values[:remaining]
            if initial_chunk.size:
                if self.sample_buffer.size == 0:
                    self.sample_buffer = initial_chunk.copy()
                else:
                    self.sample_buffer = np.concatenate([self.sample_buffer, initial_chunk])
                self.sampled_candidate_count += int(initial_chunk.size)
            sampled_values = sampled_values[remaining:]

        if sampled_values.size == 0:
            return

        for value in sampled_values:
            self.sampled_candidate_count += 1
            replace_idx = int(rng.integers(0, self.sampled_candidate_count))
            if replace_idx < self.sample_limit:
                self.sample_buffer[replace_idx] = value

    def finalize_samples(self, rng: np.random.Generator) -> np.ndarray:
        if self.sample_buffer.size == 0:
            return np.array([], dtype=np.float64)
        if self.sample_buffer.size <= self.sample_limit:
            return self.sample_buffer.astype(np.float64, copy=False)
        indices = rng.choice(self.sample_buffer.size, size=self.sample_limit, replace=False)
        return self.sample_buffer[indices].astype(np.float64, copy=False)

    def build_summary(
        self,
        sampled_values: np.ndarray,
        plot_path: Path,
    ) -> dict[str, object]:
        mean = None
        std = None
        if self.valid_pixels > 0:
            mean = self.valid_sum / self.valid_pixels
            variance = max((self.valid_sum_sq / self.valid_pixels) - (mean * mean), 0.0)
            std = math.sqrt(variance)

        quantiles = compute_quantiles(sampled_values)
        return {
            "aggregation_level": self.aggregation_level,
            "dataset": self.dataset_name,
            "band_name": self.band_name,
            "channel_group": self.channel_group,
            "source_band_count": len(self.source_band_names),
            "file_count": self.file_count,
            "total_pixels": self.total_pixels,
            "valid_pixels": self.valid_pixels,
            "finite_pixels": self.finite_pixels,
            "sample_size": int(sampled_values.size),
            "declared_nodata_value": self.nodata_value,
            "nodata_strategy": self.nodata_strategy,
            "actual_nan_fraction": ratio(self.nan_pixels, self.total_pixels),
            "actual_inf_fraction": ratio(self.inf_pixels, self.total_pixels),
            "zero_fraction": ratio(self.zero_pixels, self.total_pixels),
            "declared_nodata_fraction": ratio(self.declared_nodata_pixels, self.total_pixels),
            "excluded_nodata_fraction": ratio(self.excluded_nodata_pixels, self.total_pixels),
            "valid_min": self.valid_min,
            "valid_max": self.valid_max,
            "valid_mean": mean,
            "valid_std": std,
            "plot_path": str(plot_path),
            **quantiles,
        }


def analyze_dataset(
    dataset_dir: Path,
    road_density_years: list[str],
    max_samples_per_band: int,
    max_samples_per_group: int,
    include_by_band: bool,
    include_by_group: bool,
    rng: np.random.Generator,
    progress_callback=None,
    progress_every: int = 1,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    files, year_counts, missing_years = iter_dataset_files(dataset_dir, road_density_years)
    if not files:
        return [], [], {
            "dataset": dataset_dir.name,
            "file_count": 0,
            "year_counts": year_counts,
            "missing_years": missing_years,
        }

    band_per_file_sample_limit = max(1, math.ceil(max_samples_per_band / len(files)))
    group_per_file_sample_limit = max(1, math.ceil(max_samples_per_group / len(files)))
    band_accumulators: dict[str, BandAccumulator] = {}
    group_accumulators: dict[str, BandAccumulator] = {}
    observed_band_names: set[str] = set()
    observed_channel_groups: set[str] = set()
    pending_progress = 0

    for path in files:
        with rasterio.open(path) as src:
            nodata = src.nodata
            nodata_value = normalize_nodata(nodata)
            nodata_strategy = infer_nodata_strategy(nodata)
            band_names = infer_band_names(src)
            for band_idx, band_name in enumerate(band_names, start=1):
                channel_group = collapse_band_name(band_name)
                observed_band_names.add(band_name)
                observed_channel_groups.add(channel_group)
                band = np.asarray(src.read(band_idx))
                finite_mask, declared_nodata_mask, excluded_nodata_mask = build_nodata_masks(
                    band,
                    nodata,
                )
                if include_by_band:
                    band_accumulator = band_accumulators.get(band_name)
                    if band_accumulator is None:
                        band_accumulator = BandAccumulator(
                            aggregation_level="by_band",
                            dataset_name=dataset_dir.name,
                            band_name=band_name,
                            channel_group=channel_group,
                            nodata_value=nodata_value,
                            nodata_strategy=nodata_strategy,
                            sample_limit=max_samples_per_band,
                            per_file_sample_limit=band_per_file_sample_limit,
                        )
                        band_accumulators[band_name] = band_accumulator
                    band_accumulator.update(
                        band=band,
                        finite_mask=finite_mask,
                        declared_nodata_mask=declared_nodata_mask,
                        excluded_nodata_mask=excluded_nodata_mask,
                        source_band_name=band_name,
                        rng=rng,
                    )

                if include_by_group:
                    group_accumulator = group_accumulators.get(channel_group)
                    if group_accumulator is None:
                        group_accumulator = BandAccumulator(
                            aggregation_level="by_group",
                            dataset_name=dataset_dir.name,
                            band_name=channel_group,
                            channel_group=channel_group,
                            nodata_value=nodata_value,
                            nodata_strategy=nodata_strategy,
                            sample_limit=max_samples_per_group,
                            per_file_sample_limit=group_per_file_sample_limit,
                        )
                        group_accumulators[channel_group] = group_accumulator
                    group_accumulator.update(
                        band=band,
                        finite_mask=finite_mask,
                        declared_nodata_mask=declared_nodata_mask,
                        excluded_nodata_mask=excluded_nodata_mask,
                        source_band_name=band_name,
                        rng=rng,
                    )
        if progress_callback is not None:
            pending_progress += 1
            if pending_progress >= max(1, progress_every):
                progress_callback(pending_progress)
                pending_progress = 0

    if progress_callback is not None and pending_progress > 0:
        progress_callback(pending_progress)

    dataset_band_rows: list[dict[str, object]] = []
    dataset_group_rows: list[dict[str, object]] = []
    dataset_meta = {
        "dataset": dataset_dir.name,
        "file_count": len(files),
        "year_counts": year_counts,
        "missing_years": missing_years,
        "band_count": len(observed_band_names),
        "group_count": len(observed_channel_groups),
    }
    for band_name in sorted(band_accumulators):
        dataset_band_rows.append({"_accumulator": band_accumulators[band_name]})
    for group_name in sorted(group_accumulators):
        dataset_group_rows.append({"_accumulator": group_accumulators[group_name]})
    return dataset_band_rows, dataset_group_rows, dataset_meta


def process_dataset_task(
    *,
    dataset_dir: Path,
    out_dir: Path,
    road_density_years: list[str],
    max_samples_per_band: int,
    max_samples_per_group: int,
    include_by_band: bool,
    include_by_group: bool,
    bins: int,
    dpi: int,
    seed: int,
    progress_queue=None,
    progress_callback=None,
    progress_every: int = 1,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    rng = np.random.default_rng(seed)

    def queue_progress_callback(count: int) -> None:
        if progress_queue is not None and count > 0:
            progress_queue.put({"type": "progress", "count": int(count)})

    active_progress_callback = progress_callback
    if active_progress_callback is None and progress_queue is not None:
        active_progress_callback = queue_progress_callback

    dataset_band_rows, dataset_group_rows, dataset_meta = analyze_dataset(
        dataset_dir=dataset_dir,
        road_density_years=road_density_years,
        max_samples_per_band=max_samples_per_band,
        max_samples_per_group=max_samples_per_group,
        include_by_band=include_by_band,
        include_by_group=include_by_group,
        rng=rng,
        progress_callback=active_progress_callback,
        progress_every=progress_every,
    )

    dataset_out_dir = out_dir / dataset_dir.name
    band_plots_dir = dataset_out_dir / "plots_by_band"
    group_plots_dir = dataset_out_dir / "plots_by_group"
    finalized_band_rows: list[dict[str, object]] = []
    finalized_group_rows: list[dict[str, object]] = []

    if include_by_band:
        for placeholder in dataset_band_rows:
            accumulator = placeholder["_accumulator"]
            sampled_values = accumulator.finalize_samples(rng)
            plot_path = band_plots_dir / f"{sanitize_name(accumulator.band_name)}.png"
            row = accumulator.build_summary(sampled_values=sampled_values, plot_path=plot_path)
            plot_band_distribution(
                values=sampled_values,
                row=row,
                plot_path=plot_path,
                bins=bins,
                dpi=dpi,
            )
            finalized_band_rows.append(row)
        write_csv(finalized_band_rows, dataset_out_dir / "summary_by_band.csv")

    if include_by_group:
        for placeholder in dataset_group_rows:
            accumulator = placeholder["_accumulator"]
            sampled_values = accumulator.finalize_samples(rng)
            plot_path = group_plots_dir / f"{sanitize_name(accumulator.band_name)}.png"
            row = accumulator.build_summary(sampled_values=sampled_values, plot_path=plot_path)
            plot_band_distribution(
                values=sampled_values,
                row=row,
                plot_path=plot_path,
                bins=bins,
                dpi=dpi,
            )
            finalized_group_rows.append(row)
        write_csv(finalized_group_rows, dataset_out_dir / "summary_by_group.csv")

    if progress_queue is not None:
        progress_queue.put(
            {
                "type": "dataset_done",
                "dataset": dataset_dir.name,
                "file_count": dataset_meta["file_count"],
                "group_count": dataset_meta.get("group_count", 0),
                "band_count": dataset_meta.get("band_count", 0),
            }
        )

    return dataset_meta, finalized_band_rows, finalized_group_rows


def plot_band_distribution(
    values: np.ndarray,
    row: dict[str, object],
    plot_path: Path,
    bins: int,
    dpi: int,
) -> None:
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for axis in axes:
        axis.grid(alpha=0.2, linewidth=0.5)

    if values.size == 0:
        for axis in axes:
            axis.text(0.5, 0.5, "No valid values", ha="center", va="center", fontsize=12)
            axis.set_axis_off()
    elif use_bar_plot(values):
        unique_values, counts = np.unique(values.astype(np.int64), return_counts=True)
        axes[0].bar(unique_values, counts, width=0.8, color="#2f6c9f")
        axes[0].set_title("Sample Distribution")
        axes[0].set_xlabel("Value")
        axes[0].set_ylabel("Count")

        axes[1].bar(unique_values, counts / counts.sum(), width=0.8, color="#d18232")
        axes[1].set_title("Normalized Frequency")
        axes[1].set_xlabel("Value")
        axes[1].set_ylabel("Share")
    else:
        axes[0].hist(values, bins=bins, color="#2f6c9f", edgecolor="white", linewidth=0.4)
        axes[0].set_title("Full Sample")
        axes[0].set_xlabel("Value")
        axes[0].set_ylabel("Count")

        p01 = row["sample_p01"]
        p99 = row["sample_p99"]
        clipped = values
        if p01 is not None and p99 is not None and p99 > p01:
            clipped = values[(values >= p01) & (values <= p99)]
        axes[1].hist(clipped, bins=bins, color="#d18232", edgecolor="white", linewidth=0.4)
        axes[1].set_title("Central 1%-99% Sample")
        axes[1].set_xlabel("Value")
        axes[1].set_ylabel("Count")

        for axis in axes:
            if row["sample_p50"] is not None:
                axis.axvline(row["sample_p50"], color="#b42318", linestyle="--", linewidth=1.0, label="p50")
            if row["sample_p01"] is not None:
                axis.axvline(row["sample_p01"], color="#1d4ed8", linestyle=":", linewidth=0.9, label="p01")
            if row["sample_p99"] is not None:
                axis.axvline(row["sample_p99"], color="#1d4ed8", linestyle=":", linewidth=0.9, label="p99")
        axes[0].legend(loc="upper right", frameon=False)

    title = f"{row['dataset']} / {row['band_name']}"
    subtitle = (
        f"level={row['aggregation_level']}  files={row['file_count']}  "
        f"source_band_count={row['source_band_count']}  "
        f"valid_pixels={row['valid_pixels']}  sample_size={row['sample_size']}  "
        f"nodata_strategy={row['nodata_strategy']}"
    )
    stats_text = (
        f"actual_nan_fraction={format_float(row['actual_nan_fraction'])}\n"
        f"actual_inf_fraction={format_float(row['actual_inf_fraction'])}\n"
        f"zero_fraction={format_float(row['zero_fraction'])}\n"
        f"declared_nodata_fraction={format_float(row['declared_nodata_fraction'])}\n"
        f"excluded_nodata_fraction={format_float(row['excluded_nodata_fraction'])}\n"
        f"min={format_float(row['valid_min'])}  p01={format_float(row['sample_p01'])}  "
        f"p50={format_float(row['sample_p50'])}  p99={format_float(row['sample_p99'])}  "
        f"max={format_float(row['valid_max'])}\n"
        f"mean={format_float(row['valid_mean'])}  std={format_float(row['valid_std'])}"
    )

    fig.suptitle(title, fontsize=13, y=0.98)
    fig.text(0.5, 0.92, subtitle, ha="center", va="center", fontsize=9)
    fig.text(0.5, 0.02, stats_text, ha="center", va="bottom", fontsize=8, family="monospace")
    fig.tight_layout(rect=[0.02, 0.08, 0.98, 0.9])
    fig.savefig(plot_path, dpi=dpi)
    plt.close(fig)


def write_csv(rows: list[dict[str, object]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_readme(
    out_dir: Path,
    root: Path,
    dataset_metas: list[dict[str, object]],
    band_summary_csv: Path | None,
    group_summary_csv: Path | None,
    max_samples_per_band: int,
    max_samples_per_group: int,
    aggregation_level: str,
) -> Path:
    lines = [
        f"# Channel Distribution Report: {root.name}",
        "",
        f"- Source root: `{root}`",
        f"- Aggregation level: `{aggregation_level}`",
        (
            "- Quantiles in the CSV and plots are estimated from a per-channel sample; "
            "pixel counts, zero fractions, nodata fractions, mean, std, min, and max "
            "are computed from the full data read by the script."
        ),
        (
            "- Declared `nodata=0` is kept inside the distribution to avoid treating "
            "all zeros as invalid for products like NL or VI3."
        ),
        (
            "- `actual_nan_fraction` and `actual_inf_fraction` are measured from real "
            "pixel values in the TIFF array, independent of declared nodata metadata."
        ),
        (
            "- `by_group` collapses year-like suffixes such as `_1`, `_19`, `_2003`, "
            "and `_2022`, while generic names like `Band_1` are kept unchanged."
        ),
        f"- Per-band sample cap: `{max_samples_per_band}`",
        f"- Per-group sample cap: `{max_samples_per_group}`",
        "",
        "## Datasets",
        "",
    ]

    if band_summary_csv is not None:
        lines.insert(3, f"- By-band summary CSV: `{band_summary_csv}`")
    if group_summary_csv is not None:
        insert_at = 4 if band_summary_csv is not None else 3
        lines.insert(insert_at, f"- By-group summary CSV: `{group_summary_csv}`")

    for meta in dataset_metas:
        lines.append(
            f"- `{meta['dataset']}`: files={meta['file_count']}, "
            f"bands={meta.get('band_count', 0)}, groups={meta.get('group_count', 0)}"
        )
        if meta.get("year_counts"):
            lines.append(f"  road_density year_counts={meta['year_counts']}")
        if meta.get("missing_years"):
            lines.append(f"  missing_years={meta['missing_years']}")

    report_root = (Path("data/channel_distribution_report")).resolve()
    try:
        relative_out_dir = out_dir.resolve().relative_to(report_root)
    except ValueError:
        readme_path = out_dir / "README.md"
    else:
        readme_path = (Path("docs/channel_distribution_report") / relative_out_dir / "README.md").resolve()
        readme_path.parent.mkdir(parents=True, exist_ok=True)

    readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return readme_path


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Root is not a directory: {root}")

    out_dir = (
        args.out.expanduser().resolve()
        if args.out is not None
        else (Path("data/channel_distribution_report") / root.name).resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    road_density_years = list(args.road_density_years)
    max_samples_per_group = (
        args.max_samples_per_group
        if args.max_samples_per_group is not None
        else args.max_samples_per_band
    )
    include_by_band = args.aggregation_level in {"by_band", "both"}
    include_by_group = args.aggregation_level in {"by_group", "both"}
    dataset_dirs = collect_dataset_dirs(root)
    dataset_specs: list[dict[str, object]] = []
    total_files = 0
    for index, dataset_dir in enumerate(dataset_dirs):
        files, _, _ = iter_dataset_files(dataset_dir, road_density_years)
        file_count = len(files)
        dataset_specs.append(
            {
                "dataset_dir": dataset_dir,
                "file_count": file_count,
                "seed": args.seed + index,
            }
        )
        total_files += file_count

    max_workers = max(1, min(args.jobs, len(dataset_specs) if dataset_specs else 1))
    progress_tracker = ProgressTracker(
        total=total_files,
        enabled=not args.no_progress,
        desc=f"Scanning {root.name}",
    )
    progress_tracker.write(
        f"Datasets={len(dataset_specs)} jobs={max_workers} aggregation={args.aggregation_level} "
        f"group_sample_cap={max_samples_per_group}"
    )

    all_band_rows: list[dict[str, object]] = []
    all_group_rows: list[dict[str, object]] = []
    dataset_metas: list[dict[str, object]] = []

    try:
        if max_workers == 1:
            for spec in dataset_specs:
                dataset_dir = spec["dataset_dir"]
                progress_tracker.write(
                    f"Start {dataset_dir.name} ({spec['file_count']} files)"
                )
                dataset_meta, finalized_band_rows, finalized_group_rows = process_dataset_task(
                    dataset_dir=dataset_dir,
                    out_dir=out_dir,
                    road_density_years=road_density_years,
                    max_samples_per_band=args.max_samples_per_band,
                    max_samples_per_group=max_samples_per_group,
                    include_by_band=include_by_band,
                    include_by_group=include_by_group,
                    bins=args.bins,
                    dpi=args.dpi,
                    seed=spec["seed"],
                    progress_callback=progress_tracker.update,
                    progress_every=args.progress_every,
                )
                dataset_metas.append(dataset_meta)
                all_band_rows.extend(finalized_band_rows)
                all_group_rows.extend(finalized_group_rows)
                progress_tracker.write(
                    f"Done {dataset_dir.name}: files={dataset_meta['file_count']} "
                    f"groups={dataset_meta.get('group_count', 0)}"
                )
        else:
            def run_parallel(backend: str) -> None:
                if backend == "process":
                    executor_kwargs = {
                        "max_workers": max_workers,
                        "mp_context": get_context("spawn"),
                    }
                    executor_cls = ProcessPoolExecutor
                else:
                    executor_kwargs = {
                        "max_workers": max_workers,
                    }
                    executor_cls = ThreadPoolExecutor

                progress_tracker.write(f"Parallel backend={backend}")
                with executor_cls(**executor_kwargs) as executor:
                    future_map = {
                        executor.submit(
                            process_dataset_task,
                            dataset_dir=spec["dataset_dir"],
                            out_dir=out_dir,
                            road_density_years=road_density_years,
                            max_samples_per_band=args.max_samples_per_band,
                            max_samples_per_group=max_samples_per_group,
                            include_by_band=include_by_band,
                            include_by_group=include_by_group,
                            bins=args.bins,
                            dpi=args.dpi,
                            seed=spec["seed"],
                            progress_every=args.progress_every,
                        ): spec
                        for spec in dataset_specs
                    }
                    pending = set(future_map)
                    while pending:
                        done, pending = wait(
                            pending,
                            timeout=0.2,
                            return_when=FIRST_COMPLETED,
                        )
                        for future in done:
                            spec = future_map[future]
                            dataset_meta, finalized_band_rows, finalized_group_rows = future.result()
                            dataset_metas.append(dataset_meta)
                            all_band_rows.extend(finalized_band_rows)
                            all_group_rows.extend(finalized_group_rows)
                            progress_tracker.update(int(spec["file_count"]))
                            progress_tracker.write(
                                f"Done {dataset_meta['dataset']}: files={dataset_meta['file_count']} "
                                f"groups={dataset_meta.get('group_count', 0)}"
                            )

            if args.parallel_backend == "auto":
                try:
                    run_parallel("process")
                except (PermissionError, OSError) as exc:
                    progress_tracker.write(
                        f"Process backend unavailable ({exc}); fallback to thread backend."
                    )
                    run_parallel("thread")
            else:
                run_parallel(args.parallel_backend)
    finally:
        progress_tracker.close()

    dataset_metas.sort(key=lambda item: item["dataset"])
    all_band_rows.sort(key=lambda item: (item["dataset"], item["band_name"]))
    all_group_rows.sort(key=lambda item: (item["dataset"], item["band_name"]))

    band_summary_csv = out_dir / "band_statistics.csv" if include_by_band else None
    group_summary_csv = out_dir / "group_statistics.csv" if include_by_group else None
    if band_summary_csv is not None:
        write_csv(all_band_rows, band_summary_csv)
    if group_summary_csv is not None:
        write_csv(all_group_rows, group_summary_csv)
    readme_path = write_readme(
        out_dir=out_dir,
        root=root,
        dataset_metas=dataset_metas,
        band_summary_csv=band_summary_csv,
        group_summary_csv=group_summary_csv,
        max_samples_per_band=args.max_samples_per_band,
        max_samples_per_group=max_samples_per_group,
        aggregation_level=args.aggregation_level,
    )

    print(f"Report root: {out_dir}")
    print(f"Markdown summary: {readme_path}")
    if band_summary_csv is not None:
        print(f"Band summary CSV: {band_summary_csv}")
    if group_summary_csv is not None:
        print(f"Group summary CSV: {group_summary_csv}")
    print(f"Datasets processed: {len(dataset_metas)}")
    if include_by_band:
        print(f"Bands processed: {len(all_band_rows)}")
    if include_by_group:
        print(f"Groups processed: {len(all_group_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
