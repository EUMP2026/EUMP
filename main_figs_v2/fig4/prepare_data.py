#!/usr/bin/env python3
"""Assemble Fig.4 data from separate joint and cooling-only planning packages."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np

from fig4 import CITY_INFO, _load_pickle, _load_planning_package, prepare_differences

ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = Path(__file__).resolve().parent
PLANNING_OUTPUTS = ROOT / "experiments/planning_exp/outputs"


def load_package(path: Path) -> tuple[dict, dict[int, dict]]:
    package = _load_pickle(path)
    if not isinstance(package, dict) or package.get("schema_version") != 1:
        raise ValueError(f"Expected a schema-v1 planning package: {path}")
    items = package.get("city_data", [])
    cities = {int(item["city_id"]): item for item in items}
    if not cities or len(cities) != len(items) or len(package["cities"]) != len(cities):
        raise ValueError(f"Empty or duplicate city records: {path}")
    if set(map(int, package["cities"])) != set(cities):
        raise ValueError(f"City IDs and spatial records disagree: {path}")
    return package, cities


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def mean(values: np.ndarray, mask: np.ndarray) -> float:
    selected = np.asarray(values)[mask]
    return float(np.nanmean(selected)) if np.any(np.isfinite(selected)) else float("nan")


def export_metrics(package_path: Path) -> list[dict]:
    # Use the renderer's arrays and population percentages so CSV values and
    # figure annotations share the exact raster transformations and masks.
    data = _load_planning_package(package_path)
    cooling_ndvi, cooling_lst, joint_ndvi, joint_lst, _, pop_pct, _ = prepare_differences(data)
    rows = []
    for index, item in enumerate(data):
        mask = item["mask"].astype(bool)
        branches = (
            ("cooling_only", cooling_ndvi, cooling_lst),
            ("joint", joint_ndvi, joint_lst),
        )
        for branch, ndvi, lst in branches:
            original = item["original_congestion"]
            adjusted = item["adjusted_congestion"]
            traffic_reduction = (
                100.0 * (original - adjusted) / original
                if branch == "joint" and original > 0 else None
            )
            rows.append({
                "model": "mae_v3",
                "intervention": branch,
                "city_id": int(item["ID"]),
                "city_name": CITY_INFO.get(int(item["ID"]), {}).get("city", str(item["ID"])),
                "delta_lst_k": mean(lst[index], mask),
                "delta_ndvi_city_mean": mean(ndvi[index], mask),
                "delta_pop_city_total_pct": pop_pct[index] if branch == "joint" else 0.0,
                "traffic_delta_tci": adjusted - original if branch == "joint" else None,
                "traffic_reduction_pct": traffic_reduction,
            })
    write_csv(package_path.parent / "comparable_metrics.csv", rows)
    summary = []
    for branch in ("cooling_only", "joint"):
        subset = [row for row in rows if row["intervention"] == branch]
        record = {"intervention": branch, "cities": len(subset)}
        for metric in (
            "delta_lst_k", "delta_ndvi_city_mean",
            "delta_pop_city_total_pct", "traffic_reduction_pct",
        ):
            values = [
                row[metric] for row in subset
                if row[metric] is not None and np.isfinite(row[metric])
            ]
            record[f"mean_{metric}"] = float(np.mean(values)) if values else None
        summary.append(record)
    write_csv(package_path.parent / "scenario_summary.csv", summary)
    return rows


def assemble(joint_path: Path, cooling_path: Path, output_dir: Path) -> Path:
    joint_package, joint = load_package(joint_path)
    cooling_package, cooling = load_package(cooling_path)
    if set(joint) != set(cooling) or joint_package.get("year") != cooling_package.get("year"):
        raise ValueError("Joint and cooling-only packages must cover the same cities and year")
    spatial = []
    shared_fields = (
        "original_ndvi", "original_lst", "city_mask", "ocean_mask",
        "ndvi_mean", "ndvi_std", "lst_mean", "lst_std",
    )
    for city_id in map(int, joint_package["cities"]):
        joint_data = joint[city_id]["joint"]
        cooling_data = cooling[city_id]["cooling_only"]
        for key in shared_fields:
            if not np.array_equal(joint_data[key], cooling_data[key], equal_nan=True):
                raise ValueError(f"Joint/cooling baseline or preprocessing mismatch: city {city_id}, {key}")
        spatial.append({"city_id": city_id, "joint": joint_data, "cooling_only": cooling_data})
    package = dict(joint_package)
    package["mode"] = "both"
    package["config"] = {**joint_package.get("config", {}), "mode": "both"}
    package["city_data"] = spatial
    joint_results = {int(row["city_id"]): row for row in joint_package.get("results", [])}
    cooling_results = {int(row["city_id"]): row for row in cooling_package.get("results", [])}
    package["results"] = [
        {
            "city_id": item["city_id"],
            "joint": joint_results.get(item["city_id"], {}).get("joint", {}),
            "cooling_only": cooling_results.get(item["city_id"], {}).get("cooling_only", {}),
        }
        for item in spatial
    ]
    sources = {}
    for branch, source in (("joint", joint_path), ("cooling_only", cooling_path)):
        source = source.resolve()
        display = str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source)
        sources[branch] = {"path": display, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    package["config"]["source_packages"] = sources
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "planning_data.pkl"
    temporary = output_path.with_suffix(".pkl.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(package, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(output_path)
    rows = export_metrics(output_path)
    manifest = {
        "schema_version": 1,
        "cities": package["cities"],
        "year": package["year"],
        "sources": sources,
        "shared_baselines_match": True,
        "metric_rows": len(rows),
        "metrics_source": "fig4.prepare_differences raster arrays; equal-city scenario means",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint", type=Path, default=PLANNING_OUTPUTS / "joint_gradient_v2_5cities_current/planning_data.pkl")
    parser.add_argument("--cooling-only", type=Path, default=PLANNING_OUTPUTS / "cooling_only_5cities_1000steps/planning_data.pkl")
    parser.add_argument("--output-dir", type=Path, default=FIGURE_DIR)
    args = parser.parse_args()
    print(f"Wrote Fig.4 data and metric tables: {assemble(args.joint, args.cooling_only, args.output_dir)}")


if __name__ == "__main__":
    main()
