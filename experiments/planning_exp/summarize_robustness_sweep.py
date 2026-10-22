#!/usr/bin/env python3
"""Summarize the cooling-only robustness sweep.

Scans each config directory under an output root for planning_data.pkl, extracts
per-city cooling-only metrics, aggregates them per config, and writes:
  - robustness_summary.csv : one row per config (aggregate stats)
  - robustness_per_city.csv : one row per (config, city)

LST deltas are physical-scale, signed (adjusted - original): negative = cooling.
The mean target cooling equals lst_delta_base, so a robust run has mean
final_delta_lst close to -lst_delta_base.
"""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path
from typing import Any

import numpy as np

EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = EXPERIMENT_DIR / "outputs/cooling_only_robustness_sweep"


def _scalar(value: Any) -> float | None:
    if value is None:
        return None
    arr = np.asarray(value).reshape(-1)
    if arr.size == 0:
        return None
    v = float(arr[0])
    return v if np.isfinite(v) else None


def _get(cfg: dict[str, Any], *keys: str, default: Any = None) -> Any:
    node: Any = cfg
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def _load_package(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f)


def _config_params(package: dict[str, Any]) -> dict[str, Any]:
    cfg = package.get("config", {}) or {}
    lst_target = _get(cfg, "targets", "lst_delta_base")
    epochs = _get(cfg, "cooling_only", "epochs") or _get(cfg, "optimization", "epochs")
    lr = _get(cfg, "cooling_only", "lr")
    if lr is None:
        lr = _get(cfg, "optimization", "lr")
    return {
        "lst_delta_base": lst_target,
        "lr": lr,
        "epochs": epochs,
        "n_cities": len(package.get("cities", []) or []),
    }


def _per_city_rows(label: str, package: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per city from the lightweight cooling_only results summary."""
    rows: list[dict[str, Any]] = []
    for result in package.get("results", []) or []:
        cooling = result.get("cooling_only")
        if not isinstance(cooling, dict):
            continue
        rows.append({
            "config": label,
            "city_id": int(result["city_id"]),
            "signed_delta_lst": _scalar(cooling.get("final_signed_delta_lst")),
            "delta_lst": _scalar(cooling.get("final_delta_lst")),
            "abs_delta_lst": _scalar(cooling.get("final_abs_delta_lst")),
            "delta_ndvi": _scalar(cooling.get("final_delta_ndvi")),
        })
    return rows


def _agg(values: list[float | None]) -> dict[str, float | None]:
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _summary_row(label: str, package: dict[str, Any], city_rows: list[dict[str, Any]]) -> dict[str, Any]:
    params = _config_params(package)
    signed = _agg([r["signed_delta_lst"] for r in city_rows])
    ndvi = _agg([r["delta_ndvi"] for r in city_rows])
    target = params["lst_delta_base"]
    # Robustness: how close mean cooling is to the -lst_delta_base target.
    target_err = None
    if target is not None and signed["mean"] is not None:
        target_err = signed["mean"] - (-float(target))
    return {
        "config": label,
        "lst_delta_base": target,
        "lr": params["lr"],
        "epochs": params["epochs"],
        "n_cities": len(city_rows),
        "mean_signed_delta_lst": signed["mean"],
        "std_signed_delta_lst": signed["std"],
        "min_signed_delta_lst": signed["min"],
        "max_signed_delta_lst": signed["max"],
        "target_err_vs_neg_base": target_err,
        "mean_delta_ndvi": ndvi["mean"],
        "std_delta_ndvi": ndvi["std"],
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--data-file", default="planning_data.pkl")
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--per-city-out", type=Path, default=None)
    args = parser.parse_args(argv)

    root = args.input_root.resolve()
    summary_out = args.summary_out or (root / "robustness_summary.csv")
    per_city_out = args.per_city_out or (root / "robustness_per_city.csv")

    config_dirs = sorted(p for p in root.iterdir() if p.is_dir() and (p / args.data_file).exists())
    if not config_dirs:
        raise SystemExit(f"No '{args.data_file}' found in any subdir of {root}")

    summary_rows: list[dict[str, Any]] = []
    all_city_rows: list[dict[str, Any]] = []
    for cfg_dir in config_dirs:
        label = cfg_dir.name
        package = _load_package(cfg_dir / args.data_file)
        city_rows = _per_city_rows(label, package)
        all_city_rows.extend(city_rows)
        summary_rows.append(_summary_row(label, package, city_rows))
        print(f"{label}: {len(city_rows)} cities")

    summary_fields = [
        "config", "lst_delta_base", "lr", "epochs", "n_cities",
        "mean_signed_delta_lst", "std_signed_delta_lst",
        "min_signed_delta_lst", "max_signed_delta_lst",
        "target_err_vs_neg_base", "mean_delta_ndvi", "std_delta_ndvi",
    ]
    city_fields = ["config", "city_id", "signed_delta_lst", "delta_lst", "abs_delta_lst", "delta_ndvi"]

    _write_csv(summary_out, summary_rows, summary_fields)
    _write_csv(per_city_out, all_city_rows, city_fields)

    print(f"\nWrote summary    : {summary_out}")
    print(f"Wrote per-city   : {per_city_out}")
    print(f"\n{'config':<32} {'lst':>5} {'lr':>8} {'ep':>5} {'mean_dLST':>10} {'std':>7} {'tgt_err':>8}")
    for r in summary_rows:
        print(
            f"{r['config']:<32} {str(r['lst_delta_base']):>5} {str(r['lr']):>8} "
            f"{str(r['epochs']):>5} "
            f"{(r['mean_signed_delta_lst'] if r['mean_signed_delta_lst'] is not None else float('nan')):>10.4f} "
            f"{(r['std_signed_delta_lst'] if r['std_signed_delta_lst'] is not None else float('nan')):>7.4f} "
            f"{(r['target_err_vs_neg_base'] if r['target_err_vs_neg_base'] is not None else float('nan')):>8.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
