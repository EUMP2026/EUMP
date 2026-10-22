from __future__ import annotations

import argparse
import json
from pathlib import Path

from urban_embedding.core.paths import EUMP_DATA_DIR, RAW_DATA_DIR
from urban_embedding.data.processed import (
    load_task_config,
    validate_bh8_padding_facts,
    validate_eump_store,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate an exported EUMP scene store, optionally including BH8 padding facts."
    )
    parser.add_argument(
        "--bh8-root",
        type=Path,
        default=Path(RAW_DATA_DIR) / "BH8_v1",
        help="BH8 root to validate against the confirmed GAIA/POP padding facts.",
    )
    parser.add_argument(
        "--eump-root",
        type=Path,
        default=Path(EUMP_DATA_DIR),
        help="EUMP scene-store root, e.g. data/processed_v2.",
    )
    parser.add_argument(
        "--task-config",
        type=Path,
        default=None,
        help="Optional YAML task config used for a sample SceneRecord load.",
    )
    parser.add_argument(
        "--validate-bh8-padding",
        action="store_true",
        help="Also scan BH8_v1 and validate GAIA/POP padding facts. Disabled by default.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    padding_summary = (
        validate_bh8_padding_facts(args.bh8_root.expanduser().resolve())
        if args.validate_bh8_padding
        else {"skipped": True, "reason": "validate_bh8_padding was not requested"}
    )
    task_config = None if args.task_config is None else load_task_config(args.task_config.expanduser().resolve())
    store_summary = validate_eump_store(
        args.eump_root.expanduser().resolve(),
        task_config=task_config,
    )
    summary = {
        "bh8_padding": padding_summary,
        "eump_store": store_summary,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    padding_ok = True if padding_summary.get("skipped") else bool(padding_summary.get("all_passed"))
    return 0 if padding_ok and store_summary["is_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
