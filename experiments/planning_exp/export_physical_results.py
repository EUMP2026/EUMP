#!/usr/bin/env python3
"""Export v2 joint-gradient planning packages in physical units.

The planning pickles intentionally keep model-space arrays so they can be
reused by the planning code.  This exporter creates a separate, shareable
delivery package with physical NDVI, population, and LST rasters, explicit
deltas, masks, traffic scalars, per-city summaries, and target-level means.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import pickle
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

PACKAGES = {
    "10pct": ROOT / "experiments/planning_exp/outputs/joint_gradient_v2_292cities_traffic10pct/planning_data.pkl",
    "20pct": ROOT / "experiments/planning_exp/outputs/joint_gradient_v2_292cities_traffic20pct/planning_data.pkl",
    "30pct": ROOT / "experiments/planning_exp/outputs/joint_gradient_v2_292cities_traffic30pct/planning_data.pkl",
}

# These are the preprocessing statistics used by the MAE v3 planning
# pipeline.  Traffic/TCI is pass-through and is therefore already physical.
CONVERSIONS = {
    "ndvi": {"mode": "zscore", "mean": 0.5645092747601606, "std": 0.317689319926773},
    "pop": {"mode": "log_zscore", "mean": 0.7041826465122287, "std": 0.3879356525567439},
    "lst": {"mode": "zscore", "mean": 305.32145161793125, "std": 7.508404371966054},
    "traffic": {"mode": "pass_through", "mean": 0.0, "std": 1.0},
}


def _load(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if payload.get("schema_version") != 1 or not isinstance(payload.get("city_data"), list):
        raise ValueError(f"Unsupported planning package: {path}")
    if len(payload["city_data"]) != 292:
        raise ValueError(f"Expected 292 cities in {path}, got {len(payload['city_data'])}")
    return payload


def _physical(name: str, values: np.ndarray) -> np.ndarray:
    spec = CONVERSIONS[name]
    values = np.asarray(values, dtype=np.float32)
    if spec["mode"] == "log_zscore":
        return np.asarray(np.power(10.0, values * spec["std"] + spec["mean"]) - 1.0, dtype=np.float32)
    return np.asarray(values * spec["std"] + spec["mean"], dtype=np.float32)


def _mean(values: np.ndarray, mask: np.ndarray) -> float | None:
    selected = np.asarray(values, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    selected = selected[np.isfinite(selected)]
    return None if selected.size == 0 else float(selected.mean())


def _pct_change(original: float | None, delta: float | None) -> float | None:
    if original is None or delta is None or not np.isfinite(original) or original == 0:
        return None
    return float(100.0 * delta / original)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    fields = list(rows[0].keys())
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _city_name_map(package: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(item["id"]): item for item in package.get("city_metadata", [])}


def export_target(label: str, package_path: Path, output_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    package = _load(package_path)
    target_dir = output_root / f"target_{label}"
    city_dir = target_dir / "cities"
    city_dir.mkdir(parents=True, exist_ok=True)
    names = _city_name_map(package)
    city_rows: list[dict[str, Any]] = []

    for item in package["city_data"]:
        city_id = int(item["city_id"])
        intervention = item.get("joint")
        if not isinstance(intervention, dict):
            raise ValueError(f"City {city_id} has no joint result in {package_path}")

        ndvi0 = _physical("ndvi", intervention["original_ndvi"])
        ndvi1 = _physical("ndvi", intervention["adjusted_ndvi"])
        pop0 = _physical("pop", intervention["original_pop"])
        pop1 = _physical("pop", intervention["adjusted_pop"])
        lst0 = _physical("lst", intervention["original_lst"])
        lst1 = _physical("lst", intervention["adjusted_lst"])
        delta_ndvi = ndvi1 - ndvi0
        delta_pop = pop1 - pop0
        delta_lst = lst1 - lst0

        city_mask = np.asarray(intervention["city_mask"], dtype=bool).squeeze()
        ocean_mask = np.asarray(intervention["ocean_mask"], dtype=bool).squeeze()
        # In the planning package ocean_mask is the inverse of the raster
        # validity mask.  Keep both masks explicit for downstream users.
        valid_mask = ~ocean_mask
        full_image_mask = (
            np.isfinite(ndvi0)
            & np.isfinite(ndvi1)
            & np.isfinite(pop0)
            & np.isfinite(pop1)
            & np.isfinite(lst0)
            & np.isfinite(lst1)
        )
        if city_mask.shape != ndvi0.shape or valid_mask.shape != ndvi0.shape:
            raise ValueError(f"Mask/array shape mismatch for city {city_id}")
        city_valid = city_mask & valid_mask

        traffic0 = float(intervention["original_traffic"])
        traffic1 = float(intervention["adjusted_traffic"])
        traffic_delta = traffic1 - traffic0
        traffic_reduction = float(intervention.get("traffic_drop_pct", -100.0 * traffic_delta / traffic0 if traffic0 else np.nan))

        filename = f"city_{city_id:04d}.npz"
        np.savez_compressed(
            city_dir / filename,
            city_id=np.asarray(city_id, dtype=np.int32),
            city_mask=city_mask,
            ocean_mask=ocean_mask,
            valid_mask=valid_mask,
            original_ndvi=ndvi0,
            delta_ndvi=delta_ndvi,
            adjusted_ndvi=ndvi1,
            original_pop=pop0,
            delta_pop=delta_pop,
            adjusted_pop=pop1,
            original_lst_k=lst0,
            delta_lst_k=delta_lst,
            adjusted_lst_k=lst1,
            original_traffic=np.asarray(traffic0, dtype=np.float32),
            delta_traffic=np.asarray(traffic_delta, dtype=np.float32),
            adjusted_traffic=np.asarray(traffic1, dtype=np.float32),
            traffic_reduction_pct=np.asarray(traffic_reduction, dtype=np.float32),
            pop_delta_city_internal_pct=np.asarray(
                _pct_change(_mean(pop0, city_valid), _mean(delta_pop, city_valid)), dtype=np.float32
            ),
            pop_delta_full_image_pct=np.asarray(
                _pct_change(_mean(pop0, full_image_mask), _mean(delta_pop, full_image_mask)), dtype=np.float32
            ),
            pop_delta_full_valid_raster_pct=np.asarray(
                _pct_change(_mean(pop0, valid_mask), _mean(delta_pop, valid_mask)), dtype=np.float32
            ),
        )

        meta = names.get(city_id, {})
        ndvi_orig_city = _mean(ndvi0, city_valid)
        ndvi_delta_city = _mean(delta_ndvi, city_valid)
        pop_orig_city = _mean(pop0, city_valid)
        pop_delta_city = _mean(delta_pop, city_valid)
        lst_orig_city = _mean(lst0, city_valid)
        lst_delta_city = _mean(delta_lst, city_valid)
        ndvi_orig_valid = _mean(ndvi0, valid_mask)
        ndvi_delta_valid = _mean(delta_ndvi, valid_mask)
        pop_orig_valid = _mean(pop0, valid_mask)
        pop_delta_valid = _mean(delta_pop, valid_mask)
        lst_orig_valid = _mean(lst0, valid_mask)
        lst_delta_valid = _mean(delta_lst, valid_mask)
        ndvi_orig_full = _mean(ndvi0, full_image_mask)
        ndvi_delta_full = _mean(delta_ndvi, full_image_mask)
        pop_orig_full = _mean(pop0, full_image_mask)
        pop_delta_full = _mean(delta_pop, full_image_mask)
        lst_orig_full = _mean(lst0, full_image_mask)
        lst_delta_full = _mean(delta_lst, full_image_mask)
        city_rows.append({
            "target": label,
            "city_id": city_id,
            "city_name": meta.get("name", ""),
            "urban_fraction": meta.get("urban_fraction"),
            "raster_file": f"cities/{filename}",
            "valid_pixel_count": int(valid_mask.sum()),
            "city_valid_pixel_count": int(city_valid.sum()),
            "ndvi_original_city_mean": ndvi_orig_city,
            "ndvi_delta_city_mean": ndvi_delta_city,
            "ndvi_adjusted_city_mean": None if ndvi_orig_city is None or ndvi_delta_city is None else ndvi_orig_city + ndvi_delta_city,
            "ndvi_original_valid_mean": ndvi_orig_valid,
            "ndvi_delta_valid_mean": ndvi_delta_valid,
            "ndvi_original_city_internal_mean": ndvi_orig_city,
            "ndvi_delta_city_internal_mean": ndvi_delta_city,
            "ndvi_adjusted_city_internal_mean": None if ndvi_orig_city is None or ndvi_delta_city is None else ndvi_orig_city + ndvi_delta_city,
            "ndvi_original_full_image_mean": ndvi_orig_full,
            "ndvi_delta_full_image_mean": ndvi_delta_full,
            "pop_original_city_mean": pop_orig_city,
            "pop_delta_city_mean": pop_delta_city,
            "pop_adjusted_city_mean": None if pop_orig_city is None or pop_delta_city is None else pop_orig_city + pop_delta_city,
            "pop_original_valid_mean": pop_orig_valid,
            "pop_delta_valid_mean": pop_delta_valid,
            "pop_delta_city_pct": _pct_change(pop_orig_city, pop_delta_city),
            "pop_original_city_internal_mean": pop_orig_city,
            "pop_delta_city_internal_mean": pop_delta_city,
            "pop_adjusted_city_internal_mean": None if pop_orig_city is None or pop_delta_city is None else pop_orig_city + pop_delta_city,
            "pop_delta_city_internal_pct": _pct_change(pop_orig_city, pop_delta_city),
            "pop_original_full_image_mean": pop_orig_full,
            "pop_delta_full_image_mean": pop_delta_full,
            "pop_delta_full_image_pct": _pct_change(pop_orig_full, pop_delta_full),
            "pop_original_full_valid_raster_mean": pop_orig_valid,
            "pop_delta_full_valid_raster_mean": pop_delta_valid,
            "pop_delta_full_valid_raster_pct": _pct_change(pop_orig_valid, pop_delta_valid),
            "lst_original_city_mean_k": lst_orig_city,
            "lst_delta_city_mean_k": lst_delta_city,
            "lst_adjusted_city_mean_k": None if lst_orig_city is None or lst_delta_city is None else lst_orig_city + lst_delta_city,
            "lst_original_valid_mean_k": lst_orig_valid,
            "lst_delta_valid_mean_k": lst_delta_valid,
            "lst_original_city_internal_mean_k": lst_orig_city,
            "lst_delta_city_internal_mean_k": lst_delta_city,
            "lst_adjusted_city_internal_mean_k": None if lst_orig_city is None or lst_delta_city is None else lst_orig_city + lst_delta_city,
            "lst_original_full_image_mean_k": lst_orig_full,
            "lst_delta_full_image_mean_k": lst_delta_full,
            "original_traffic": traffic0,
            "delta_traffic": traffic_delta,
            "adjusted_traffic": traffic1,
            "traffic_reduction_pct": traffic_reduction,
        })

    _write_csv(target_dir / "city_summary.csv", city_rows)
    manifest = [{"city_id": row["city_id"], "city_name": row["city_name"], "raster_file": row["raster_file"]} for row in city_rows]
    _write_csv(target_dir / "manifest.csv", manifest)

    def avg(field: str) -> float | None:
        vals = np.asarray([row[field] for row in city_rows if row.get(field) is not None], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        return None if vals.size == 0 else float(vals.mean())

    summary = {
        "target": label,
        "planning_package": str(package_path),
        "cities": len(city_rows),
        "mean_ndvi_delta_city": avg("ndvi_delta_city_mean"),
        "mean_ndvi_delta_city_internal": avg("ndvi_delta_city_internal_mean"),
        "mean_pop_delta_city": avg("pop_delta_city_mean"),
        "mean_pop_delta_city_pct": avg("pop_delta_city_pct"),
        "mean_pop_delta_city_internal": avg("pop_delta_city_internal_mean"),
        "mean_pop_delta_city_internal_pct": avg("pop_delta_city_internal_pct"),
        "mean_pop_delta_full_image": avg("pop_delta_full_image_mean"),
        "mean_pop_delta_full_image_pct": avg("pop_delta_full_image_pct"),
        "mean_pop_delta_full_valid_raster": avg("pop_delta_full_valid_raster_mean"),
        "mean_pop_delta_full_valid_raster_pct": avg("pop_delta_full_valid_raster_pct"),
        "mean_lst_delta_city_k": avg("lst_delta_city_mean_k"),
        "mean_lst_delta_city_internal_k": avg("lst_delta_city_internal_mean_k"),
        "mean_traffic_reduction_pct": avg("traffic_reduction_pct"),
        "mean_delta_traffic": avg("delta_traffic"),
        "mask_definition": "city_valid = city_mask & valid_mask; valid_mask = ~ocean_mask",
    }
    (target_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary, city_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments/planning_exp/outputs/physical_results",
    )
    args = parser.parse_args()
    output_root = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for label, package_path in PACKAGES.items():
        summary, _ = export_target(label, package_path, output_root)
        summaries.append(summary)
    _write_csv(output_root / "planning_targets_summary.csv", summaries)
    (output_root / "conversion_spec.json").write_text(
        json.dumps({"source": "MAE v3 preprocessing spec", "variables": CONVERSIONS}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote planning delivery to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
