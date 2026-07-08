from __future__ import annotations

import argparse
import json
from pathlib import Path

from urban_embedding.core.paths import CSV_DATA_DIR, EUMP_DATA_DIR
from urban_embedding.data.processed import (
    DEFAULT_SPLIT_COLUMN_NAME,
    build_legacy_compatible_split_manifest,
    default_legacy_split_manifest_path,
    write_split_annotated_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a fixed scene-level split manifest compatible with the legacy two-stage split."
    )
    parser.add_argument(
        "--eump-root",
        type=Path,
        default=Path(EUMP_DATA_DIR),
        help="EUMP scene-store root, e.g. data/processed_v1.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path(CSV_DATA_DIR) / "df_norm_v3.csv",
        help="Legacy CSV used to reproduce the old scene split.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional explicit output path. Default: <eump-root>/manifest/splits/legacy_compatible_seed42.json",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used by the legacy split routine.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.3,
        help="Legacy test_size passed into the first train_test_split call.",
    )
    parser.add_argument(
        "--year-range",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        default=(2003, 2020),
        help="Inclusive year range used before deduplicating legacy IDs. Default: 2003 2020.",
    )
    parser.add_argument(
        "--full-csv-pool",
        action="store_true",
        help="Use the full legacy CSV ID pool after year filtering instead of intersecting with scene IDs in eump_root.",
    )
    parser.add_argument(
        "--annotate-csv",
        action="store_true",
        help="Add a train/val/test split column to the legacy CSV using the generated scene-level split.",
    )
    parser.add_argument(
        "--annotated-csv-path",
        type=Path,
        default=None,
        help="Optional path for the split-annotated CSV. Implies --annotate-csv. Default updates --csv-path in place.",
    )
    parser.add_argument(
        "--split-column",
        type=str,
        default=DEFAULT_SPLIT_COLUMN_NAME,
        help=f"Column name used when annotating the CSV. Default: {DEFAULT_SPLIT_COLUMN_NAME}.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    eump_root = args.eump_root.expanduser().resolve()
    csv_path = args.csv_path.expanduser().resolve()
    output_path = (
        default_legacy_split_manifest_path(eump_root)
        if args.output_path is None
        else args.output_path.expanduser().resolve()
    )
    payload = build_legacy_compatible_split_manifest(
        eump_root=eump_root,
        csv_path=csv_path,
        output_path=output_path,
        random_seed=args.seed,
        test_size=args.test_size,
        year_range=tuple(int(year) for year in args.year_range),
        restrict_to_available_scenes=not args.full_csv_pool,
        sort_output_ids=True,
    )
    print_payload = dict(payload)
    if args.annotate_csv or args.annotated_csv_path is not None:
        annotated_csv_path = (
            csv_path
            if args.annotated_csv_path is None
            else args.annotated_csv_path.expanduser().resolve()
        )
        print_payload["csv_annotation"] = write_split_annotated_csv(
            csv_path=csv_path,
            split_manifest=payload,
            output_path=annotated_csv_path,
            split_column=args.split_column,
        )
    print(json.dumps(print_payload, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
