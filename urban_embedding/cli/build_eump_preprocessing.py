from __future__ import annotations

import argparse
import json
from pathlib import Path

from urban_embedding.core.paths import EUMP_DATA_DIR
from urban_embedding.data.processed import ProcessedPreprocessingBuilder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build preprocessing_v1.json for an existing EUMP scene store."
    )
    parser.add_argument(
        "--eump-root",
        type=Path,
        default=Path(EUMP_DATA_DIR),
        help="EUMP scene-store root, e.g. data/processed_v2.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional explicit output JSON path. Default: <eump-root>/manifest/stats/preprocessing_v1.json",
    )
    parser.add_argument(
        "--split-manifest-path",
        type=Path,
        default=None,
        help="Optional split manifest. When set, stats default to the train split instead of the full store.",
    )
    parser.add_argument(
        "--stats-split",
        choices=("auto", "all", "train", "val", "test"),
        default="auto",
        help="Split used for preprocessing stats. 'auto' means train when a split manifest is set, otherwise all.",
    )
    parser.add_argument(
        "--year-range",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        default=None,
        help="Optional inclusive year range for yearly stats. Defaults to the split manifest year_range when available.",
    )
    parser.add_argument(
        "--city-crop-size",
        type=int,
        default=256,
        help="Crop size used to estimate positive city-crop normalization stats.",
    )
    parser.add_argument(
        "--city-crop-samples-per-scene-year",
        type=int,
        default=0,
        help="Number of random city crops sampled per scene-year for legacy city_crop_stats. Default: 0.",
    )
    parser.add_argument(
        "--city-crop-min-city-mask-coverage",
        type=float,
        default=0.03,
        help="Minimum city-mask coverage for a crop to contribute to city_crop_stats.",
    )
    parser.add_argument(
        "--city-crop-random-seed",
        type=int,
        default=42,
        help="Random seed for city_crop_stats sampling.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of worker processes for static/yearly preprocessing stats. Use 0 for serial.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stats_split = args.stats_split
    if stats_split == "auto":
        stats_split = "train" if args.split_manifest_path is not None else "all"
    builder = ProcessedPreprocessingBuilder(
        eump_root=args.eump_root.expanduser().resolve(),
        output_path=None if args.output_path is None else args.output_path.expanduser().resolve(),
        split_manifest_path=None if args.split_manifest_path is None else args.split_manifest_path.expanduser().resolve(),
        stats_split=stats_split,
        year_range=None if args.year_range is None else tuple(args.year_range),
        city_crop_size=args.city_crop_size,
        city_crop_samples_per_scene_year=args.city_crop_samples_per_scene_year,
        city_crop_min_city_mask_coverage=args.city_crop_min_city_mask_coverage,
        city_crop_random_seed=args.city_crop_random_seed,
        num_workers=args.workers,
    )
    payload = builder.build()
    print(json.dumps(payload, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
