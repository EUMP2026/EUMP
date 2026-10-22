#!/usr/bin/env python3
"""Summarize and plot city-level predicted responses from planning packages.

The default input is the canonical five-city joint-gradient v2 package.
Use --version NAME=PATH for a selected run or several target-sweep packages.
"""

from __future__ import annotations
import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_V2 = PROJECT_ROOT / "experiments/planning_exp/outputs/joint_gradient_v2_5cities_current"
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments/planning_exp/outputs/planning_distributions"


def _finite(values: Iterable[Any]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    return arr[np.isfinite(arr)]


def _stats(values: Iterable[Any]) -> dict[str, Any]:
    arr = _finite(values)
    if not arr.size:
        return {"n": 0, "mean": None, "std": None, "median": None,
                "p10": None, "p25": None, "p75": None, "p90": None,
                "min": None, "max": None}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _parse_assignment(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name or not path.strip():
        raise argparse.ArgumentTypeError("expected non-empty NAME=PATH")
    return name, Path(path).expanduser()


def _resolve_pickle(path: Path) -> Path:
    if path.is_dir():
        path = path / "planning_data.pkl"
    if not path.is_file():
        raise FileNotFoundError(f"planning result not found: {path}")
    return path


def _display_path(path: Path) -> str:
    """Use repository-relative paths in portable metadata when possible."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _load_planning(path: Path) -> dict[str, Any]:
    with _resolve_pickle(path).open("rb") as handle:
        package = pickle.load(handle)
    if not isinstance(package, dict) or "city_data" not in package:
        raise ValueError(f"unsupported planning_data.pkl schema: {path}")
    return package


def _scalar(value: Any, default: float = 1.0) -> float:
    arr = np.asarray(value)
    return float(arr.reshape(-1)[0]) if arr.size else default


def _valid_mask(joint: dict[str, Any], shape: tuple[int, ...]) -> np.ndarray:
    """Return the exported raster validity mask.

    Newer schemas may include ``valid_mask`` explicitly.  The current
    planning exporter stores the complement as ``ocean_mask`` instead, with
    ``ocean_mask = ~base_mask`` in ``experiments/planning_exp/run.py``.
    """
    if "valid_mask" in joint:
        valid = np.asarray(joint["valid_mask"], dtype=bool).squeeze()
    elif "ocean_mask" in joint:
        valid = ~np.asarray(joint["ocean_mask"], dtype=bool).squeeze()
    else:
        raise KeyError("planning branch contains neither valid_mask nor ocean_mask")
    if valid.shape != shape:
        raise ValueError(f"valid mask shape {valid.shape} does not match raster shape {shape}")
    return valid


def _population_metrics(joint: dict[str, Any]) -> dict[str, float]:
    """Return physical-population changes for valid urban and whole regions.

    Population channels are log10(1 + population), z-scored.  ``ocean_mask``
    in planning exports is the inverse of the valid raster mask (the name is
    historical), so finite, non-ocean pixels define the whole-picture region.
    The urban region is the intersection with ``city_mask``.  Both regional
    percentage changes use the original whole-image population as the common
    denominator.  This is intentionally different from dividing the urban
    change by the urban population itself: the urban and whole-picture values
    then share one reference population and can be compared directly.
    """
    original = np.asarray(joint["original_pop"], dtype=float)
    adjusted = np.asarray(joint["adjusted_pop"], dtype=float)
    pop_mean = _scalar(joint.get("pop_mean", 0.0), 0.0)
    pop_std = _scalar(joint.get("pop_std", 1.0), 1.0)
    original = np.maximum(np.power(10.0, original * pop_std + pop_mean) - 1.0, 0.0)
    adjusted = np.maximum(np.power(10.0, adjusted * pop_std + pop_mean) - 1.0, 0.0)
    valid = np.isfinite(original) & np.isfinite(adjusted)
    valid &= _valid_mask(joint, original.shape)
    city = np.asarray(joint.get("city_mask", np.ones_like(valid)), dtype=bool)
    masks = {"urban": valid & city, "whole": valid}
    full_original = float(np.sum(original[masks["whole"]]))
    denominator = full_original if full_original > 0 else np.nan
    result: dict[str, float] = {}
    for region, mask in masks.items():
        old = float(np.sum(original[mask]))
        new = float(np.sum(adjusted[mask]))
        delta = new - old
        pct = 100.0 * delta / denominator if np.isfinite(denominator) else np.nan
        result[f"delta_pop_{region}_absolute"] = delta
        result[f"delta_pop_{region}_percent"] = pct
        result[f"pop_{region}_original"] = old
        result[f"pop_{region}_adjusted"] = new
    result["pop_full_original_denominator"] = full_original
    return result


def collect_planning(version_paths: dict[str, Path]) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}
    for version, path in version_paths.items():
        package = _load_planning(path)
        metadata[version] = {
            "path": _display_path(_resolve_pickle(path)),
            "cities": package.get("cities"),
            "config": package.get("config", {}),
        }
        for city in package["city_data"]:
            joint = city.get("joint")
            if not isinstance(joint, dict):
                continue
            pop = _population_metrics(joint)
            final_pop = joint.get("final_delta_pop")
            if final_pop is None:
                trajectory = joint.get("delta_pop_traj", [])
                final_pop = trajectory[-1] if trajectory else np.nan
            traffic_drop = joint.get("traffic_drop_pct")
            rows.append({
                "version": version,
                "city_id": int(city.get("city_id", joint.get("city_id"))),
                "delta_lst_k": float(joint["final_delta_lst"]),
                "delta_tci_pct": np.nan if traffic_drop is None else float(traffic_drop),
                # Existing trajectory definition: valid/base-mask percent.
                "delta_pop_traj_percent": float(final_pop),
                **pop,
            })
    frame = pd.DataFrame(rows)
    return frame, metadata


def _import_plt(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = output_dir / ".matplotlib"
    cache.mkdir(exist_ok=True)
    import os
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _histogram(frame: pd.DataFrame, group_col: str, metric: str, title: str,
               xlabel: str, path: Path, bins: int = 10) -> None:
    plt = _import_plt(path.parent)
    fig, ax = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    groups = list(frame[group_col].dropna().astype(str).drop_duplicates())
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, max(1, len(groups))))
    all_values: list[np.ndarray] = []
    for color, group in zip(colors, groups):
        values = _finite(frame.loc[frame[group_col].astype(str) == group, metric])
        if not values.size:
            continue
        all_values.append(values)
    if all_values:
        combined = np.concatenate(all_values)
        if np.ptp(combined) == 0:
            edges = np.linspace(combined[0] - 0.5, combined[0] + 0.5, bins + 1)
        else:
            edges = np.linspace(np.min(combined), np.max(combined), bins + 1)
        for color, group in zip(colors, groups):
            values = _finite(frame.loc[frame[group_col].astype(str) == group, metric])
            if values.size:
                ax.hist(values, bins=edges, alpha=0.48, color=color,
                        edgecolor="white", linewidth=0.6,
                        label=f"{group} (n={values.size}, μ={np.mean(values):.3g})")
    ax.axvline(0.0, color="#444", linestyle="--", linewidth=0.9)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of cities")
    ax.grid(axis="y", alpha=0.25)
    if groups:
        ax.legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_planning(frame: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    metrics = {
        "delta_lst_k": ("Planning: delta LST", "delta LST (K)"),
        "delta_tci_pct": ("Planning: traffic congestion reduction", "delta TCI / traffic reduction (%)"),
        "delta_pop_urban_percent": ("Planning: population change within urban area", "delta POP within urban (%)"),
        "delta_pop_whole_percent": ("Planning: population change in whole picture", "delta POP in whole picture (%)"),
        "delta_pop_urban_absolute": ("Planning: absolute population change within urban area", "delta POP within urban (physical units)"),
        "delta_pop_whole_absolute": ("Planning: absolute population change in whole picture", "delta POP in whole picture (physical units)"),
    }
    summary_rows: list[dict[str, Any]] = []
    for metric, (title, xlabel) in metrics.items():
        _histogram(frame, "version", metric, title, xlabel,
                   output_dir / f"planning_{metric}.png")
        for version, subset in frame.groupby("version", sort=False):
            summary_rows.append({"version": version, "metric": metric,
                                 **_stats(subset[metric])})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "planning_metrics_summary.csv", index=False)
    frame.to_csv(output_dir / "planning_city_metrics.csv", index=False)
    return {"metrics": metrics, "summary": summary.to_dict(orient="records")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="append", type=_parse_assignment, metavar="NAME=PATH",
                        help="planning package(s) to summarize; defaults to the five-city v2 run")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    versions = dict(args.version) if args.version else {"joint_gradient_v2": DEFAULT_V2}
    frame, sources = collect_planning(versions)
    if frame.empty:
        raise ValueError("No joint planning results found in the selected packages")
    result = plot_planning(frame, args.output_dir)
    metadata = {"planning_versions": sources, "planning": result}
    (args.output_dir / "summary.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote planning distributions and tables to {args.output_dir}")


if __name__ == "__main__":
    main()
