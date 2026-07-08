from __future__ import annotations

import argparse
import json
from pathlib import Path

from urban_embedding.core.paths import CSV_DATA_DIR, EUMP_DATA_DIR, RAW_DATA_DIR
from urban_embedding.data.processed.exporter import ProcessedSceneStoreExporter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export raw raster and scalar sources into the EUMP scene-store "
            "layout backed by local-directory Zarr groups."
        )
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path(RAW_DATA_DIR),
        help="Raw raster root, e.g. data/raw_data",
    )
    parser.add_argument(
        "--csv-root",
        type=Path,
        default=Path(CSV_DATA_DIR),
        help="CSV root, e.g. data/csv_data",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(EUMP_DATA_DIR),
        help="EUMP scene-store root, e.g. data/eump_v1",
    )
    parser.add_argument(
        "--scene-ids",
        nargs="*",
        type=int,
        default=None,
        help="Optional explicit scene IDs to export. Default: discover from BH8_v1.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the existing output root before writing the new processed store.",
    )
    parser.add_argument(
        "--skip-stats",
        action="store_true",
        help="Skip normalization_v1.json generation.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of scene-export worker processes. Use 0 for auto.",
    )
    parser.add_argument(
        "--validate-bh8-padding",
        action="store_true",
        help="Scan BH8_v1 and validate GAIA/POP padding facts before export. Disabled by default.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exporter = ProcessedSceneStoreExporter(
        raw_root=args.raw_root.expanduser().resolve(),
        csv_root=args.csv_root.expanduser().resolve(),
        output_root=args.output_root.expanduser().resolve(),
        overwrite=args.overwrite,
        scene_ids=args.scene_ids,
        compute_stats=not args.skip_stats,
        num_workers=args.workers,
        validate_padding=args.validate_bh8_padding,
    )
    summary = exporter.export()
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
