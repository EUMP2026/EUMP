#!/usr/bin/env python3
"""Compute manipulate_exp bulk response statistics from an existing NPZ export."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
RUN_PY = EXPERIMENT_DIR / "run.py"
METADATA_KEYS = {
    "scene_ids",
    "years",
    "splits",
    "crop_boxes",
    "order_indices",
    "deltas",
    "channel_names",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-city/per-delta response statistics from an existing "
            "manipulate_exp bulk NPZ without rerunning inference."
        )
    )
    parser.add_argument(
        "npz_path",
        type=Path,
        help="Existing bulk NPZ, e.g. all_cities_ndvi_lst_2020.npz.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="Optional summary.json path. Default: <npz_dir>/summary.json.",
    )
    parser.add_argument(
        "--stats-csv",
        type=Path,
        default=None,
        help="Optional output CSV path. Default: <npz_stem>_stats.csv beside the NPZ.",
    )
    parser.add_argument(
        "--target-name",
        default=None,
        help="Target name override. Default: summary target.name or inferred from array key.",
    )
    parser.add_argument(
        "--source-name",
        default=None,
        help="Source name override. Default: summary perturbation.source_name or inferred from array key.",
    )
    parser.add_argument(
        "--no-update-summary",
        action="store_true",
        help="Write only the stats CSV and leave summary.json unchanged.",
    )
    return parser.parse_args(argv)


def _load_run_module():
    spec = importlib.util.spec_from_file_location("manipulate_exp_run", RUN_PY)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import manipulate run helpers from {RUN_PY}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected summary JSON object at {path}, got {type(payload)!r}")
    return payload


def _array_keys(files: list[str]) -> list[str]:
    return [key for key in files if key not in METADATA_KEYS]


def _infer_target_source_from_key(run_module: Any, array_key: str, deltas: np.ndarray) -> tuple[str, str] | None:
    for delta in deltas:
        suffix = run_module._delta_key_suffix(float(delta))
        if not array_key.endswith(suffix):
            continue
        stem = array_key[: -len(suffix)]
        if "_" not in stem:
            continue
        target_name, source_name = stem.split("_", 1)
        if target_name and source_name:
            return target_name, source_name
    return None


def _resolve_target_source(
    *,
    run_module: Any,
    payload: Any,
    summary: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[str, str]:
    target_name = args.target_name or summary.get("target", {}).get("name")
    source_name = args.source_name or summary.get("perturbation", {}).get("source_name")
    if target_name and source_name:
        return str(target_name), str(source_name)

    keys = _array_keys(list(payload.files))
    if not keys:
        raise ValueError(f"No raster response arrays found in {args.npz_path}")
    inferred = _infer_target_source_from_key(run_module, keys[0], payload["deltas"])
    if inferred is None:
        raise ValueError(
            "Unable to infer target/source from NPZ array keys. "
            "Pass --target-name and --source-name explicitly."
        )
    inferred_target, inferred_source = inferred
    return str(target_name or inferred_target), str(source_name or inferred_source)


class _NpzMapping:
    def __init__(self, payload: Any) -> None:
        self.payload = payload

    def __getitem__(self, key: str) -> np.ndarray:
        return self.payload[key]

    def __contains__(self, key: str) -> bool:
        return key in self.payload.files


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    npz_path = args.npz_path
    if not npz_path.is_absolute():
        npz_path = (PROJECT_ROOT / npz_path).resolve()
    if not npz_path.exists():
        raise FileNotFoundError(f"Bulk NPZ does not exist: {npz_path}")

    summary_path = args.summary_path or (npz_path.parent / "summary.json")
    if not summary_path.is_absolute():
        summary_path = (PROJECT_ROOT / summary_path).resolve()
    stats_csv = args.stats_csv or npz_path.with_name(f"{npz_path.stem}_stats.csv")
    if not stats_csv.is_absolute():
        stats_csv = (PROJECT_ROOT / stats_csv).resolve()

    run_module = _load_run_module()
    summary = _load_summary(summary_path)

    with np.load(str(npz_path), allow_pickle=False) as payload:
        required = {"scene_ids", "years", "splits", "crop_boxes", "deltas"}
        missing = sorted(required.difference(payload.files))
        if missing:
            raise ValueError(f"Bulk NPZ is missing required keys: {missing}")
        target_name, source_name = _resolve_target_source(
            run_module=run_module,
            payload=payload,
            summary=summary,
            args=args,
        )
        stats = run_module._write_bulk_raster_stats_csv(
            output_payload=_NpzMapping(payload),
            stats_csv_path=stats_csv,
            target_name=target_name,
            source_name=source_name,
            deltas=payload["deltas"],
        )
        total_cities = int(payload["scene_ids"].shape[0])

    if not args.no_update_summary:
        summary.setdefault("target", {}).setdefault("kind", "raster")
        summary.setdefault("target", {})["name"] = target_name
        summary.setdefault("perturbation", {})["source_name"] = source_name
        summary.setdefault("data", {})["total_cities"] = total_cities
        summary["stats"] = stats
        summary.setdefault("outputs", {})["npz"] = str(npz_path)
        summary.setdefault("outputs", {})["stats_csv"] = str(stats_csv)
        summary.setdefault("outputs", {})["summary"] = str(summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(run_module._jsonable(summary), indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )

    print(f"Saved stats CSV: {stats_csv}")
    if not args.no_update_summary:
        print(f"Updated summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
