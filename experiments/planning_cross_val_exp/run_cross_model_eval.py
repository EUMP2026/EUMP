#!/usr/bin/env python3
"""Evaluate the same planning intervention with the two retained LST models.

The planning package supplies original/adjusted NDVI and population rasters.
For each configured architecture this script builds an independent LST model
and evaluates the four factorial inputs (00, 11, 01, 10). The planning maps
are unchanged between models; this is a predicted-response comparison and does
not evaluate Traffic or causal intervention effects.
"""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import gc
import json
from pathlib import Path
import pickle
import sys
from typing import Any

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = EXPERIMENT_DIR / "config/cross_model_eval_scratch.yaml"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_module():
    import importlib.util

    path = EXPERIMENT_DIR / "model_utils.py"
    spec = importlib.util.spec_from_file_location("planning_cross_val_exp_model_utils", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load planning module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--planning-data", type=Path, default=None)
    parser.add_argument(
        "--planning-variant", choices=("joint", "cooling_only"), default="joint",
        help="Use the joint NDVI+POP map or the NDVI-only cooling map.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--models", default=None, help="Comma-separated config model keys.")
    parser.add_argument("--cities", default=None, help="Comma-separated city IDs.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(_resolve(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("cross_model_eval"), dict):
        raise ValueError(f"Expected cross_model_eval mapping in {path}")
    return deepcopy(payload["cross_model_eval"])


def _load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported planning package at {path}")
    if not isinstance(payload.get("city_data"), list):
        raise ValueError(f"Planning package at {path} has no city_data list")
    return payload


def _model_name_parts(model_name: str) -> tuple[str, str]:
    parts = str(model_name).strip("/\\").split("/", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Model name must be '<group>/<name>', got {model_name!r}")
    return parts[0], parts[1]


def _compose_model(
    *, planning: Any, pipeline: str, overrides: list[str], model_name: str,
    checkpoint_file: str, device: torch.device, num_workers: int,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    hydra_cfg = planning._compose_hydra_config(
        pipeline, [], int(num_workers), pipeline_overrides=[str(value) for value in overrides]
    )
    group, name = _model_name_parts(model_name)
    hydra_cfg["experiment"]["group"] = group
    hydra_cfg["experiment"]["name"] = name
    hydra_cfg["experiment"]["model_name"] = model_name
    hydra_cfg["evaluation"] = {"checkpoint_source": "best_model", "checkpoint_file": checkpoint_file}
    pipeline_spec = planning.PipelineSpec.from_mapping(hydra_cfg["pipeline"])
    planning.validate_eump_downstream_spec(pipeline_spec)
    train_dl, val_dl, test_dl, batch_schema = planning.build_eump_dataloaders(
        data_config=hydra_cfg["data"], pipeline_spec=pipeline_spec, logger=planning.logger,
        random_seed=int(hydra_cfg["experiment"]["random_seed"]), batch_size=1,
        local_rank=0, world_size=1, is_distributed=False, shuffle_train=False,
    )
    target_schema = planning.get_single_downstream_target_schema(batch_schema)
    model = planning._build_model(
        hydra_cfg=hydra_cfg, batch_schema=batch_schema, target_schema=target_schema,
        device=device, freeze_encoder=False,
    )
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    preprocessing = train_dl.dataset.preprocessing_spec
    source_name = target_schema.spec.source
    if source_name is None or source_name not in preprocessing["sources"]:
        raise ValueError(f"Target {target_schema.name!r} has no preprocessing source")
    return model, {
        "loaders": {"train": train_dl, "val": val_dl, "test": test_dl},
        "batch_schema": batch_schema, "target_schema": target_schema,
    }, preprocessing


def _forward_lst(model: torch.nn.Module, raster_input: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        raster_predictions, _ = model(raster_input, None)
    if not raster_predictions:
        raise ValueError("LST model returned no raster prediction")
    return next(iter(raster_predictions.values())).detach().cpu().numpy().squeeze()


def _set_channels(base: torch.Tensor, *, ndvi_idx: int, pop_idx: int,
                  ndvi: np.ndarray, pop: np.ndarray) -> torch.Tensor:
    value = base.clone()
    value[0, ndvi_idx] = torch.as_tensor(ndvi, dtype=value.dtype, device=value.device)
    value[0, pop_idx] = torch.as_tensor(pop, dtype=value.dtype, device=value.device)
    return value


def _pearson(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float | None:
    selected = np.asarray(mask, dtype=bool) & np.isfinite(a) & np.isfinite(b)
    if int(selected.sum()) < 2:
        return None
    x = np.asarray(a, dtype=np.float64)[selected]
    y = np.asarray(b, dtype=np.float64)[selected]
    if np.std(x) <= 0.0 or np.std(y) <= 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _correlation_sums(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> tuple[float, ...]:
    selected = np.asarray(mask, dtype=bool) & np.isfinite(a) & np.isfinite(b)
    x = np.asarray(a, dtype=np.float64)[selected]
    y = np.asarray(b, dtype=np.float64)[selected]
    return (float(x.size), float(x.sum()), float(y.sum()), float((x * x).sum()),
            float((y * y).sum()), float((x * y).sum()))


def _correlation_from_sums(values: tuple[float, ...]) -> float | None:
    count, sum_x, sum_y, sum_xx, sum_yy, sum_xy = values
    if count < 2:
        return None
    x_ss = sum_xx - sum_x * sum_x / count
    y_ss = sum_yy - sum_y * sum_y / count
    denominator = np.sqrt(max(x_ss, 0.0) * max(y_ss, 0.0))
    return None if denominator <= 1.0e-12 else float((sum_xy - sum_x * sum_y / count) / denominator)


def _physical_array(values: np.ndarray, source_spec: dict[str, Any]) -> np.ndarray:
    stats = source_spec.get("stats", {}) or {}
    return (np.asarray(values, dtype=np.float32) * float(stats.get("std", 1.0))
            + float(stats.get("mean", 0.0))).astype(np.float32, copy=False)


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float | None:
    selected = np.asarray(values, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    selected = selected[np.isfinite(selected)]
    return None if selected.size == 0 else float(selected.mean())


def _city_items(package: dict[str, Any], cities: str | None) -> list[dict[str, Any]]:
    items = [item for item in package["city_data"] if isinstance(item, dict)]
    if cities is None:
        return items
    selected = [int(value.strip()) for value in cities.split(",") if value.strip()]
    by_id = {int(item["city_id"]): item for item in items}
    missing = [city for city in selected if city not in by_id]
    if missing:
        raise ValueError(f"Cities missing from planning package: {missing}")
    return [by_id[city] for city in selected]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row if not key.startswith("_")})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _finite_values(rows: list[dict[str, Any]], field: str) -> np.ndarray:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array)]


def _write_comparison_figure(path: Path, *, model_specs: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_rows = {key: [row for row in rows if row["model"] == key] for key in model_specs}
    model_rows = {key: value for key, value in model_rows.items() if value}
    if not model_rows:
        return
    labels = [str(model_specs[key].get("label", key)) for key in model_rows]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    panels = (
        ("delta_lst1_vs_lst0_city", "Combined intervention", "LST change (K)", "#2c7fb8"),
        ("delta_lst10_vs_lst0_city", "NDVI contribution", "LST change (K)", "#31a354"),
        ("delta_lst01_vs_lst0_city", "Population contribution", "LST change (K)", "#756bb1"),
    )
    for axis, (field, title, ylabel, color) in zip(axes, panels):
        values = [_finite_values(model_rows[key], field) for key in model_rows]
        positions = [index + 1 for index, value in enumerate(values) if value.size]
        plotted = [value for value in values if value.size]
        if plotted:
            boxplot = axis.boxplot(
                plotted, positions=positions, widths=0.6,
                tick_labels=[labels[index - 1] for index in positions],
                patch_artist=True, showfliers=False,
            )
            for box in boxplot["boxes"]:
                box.set_facecolor(color)
                box.set_alpha(0.72)
        axis.axhline(0.0, color="#333333", linewidth=0.8, linestyle="--")
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", labelrotation=18)
    figure.suptitle("Planning intervention evaluated with independent LST models", fontsize=15)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _format_report_value(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    number = float(value)
    return "NA" if not np.isfinite(number) else f"{number:.{digits}f}"


def _write_comparison_report(path: Path, summary: list[dict[str, Any]]) -> None:
    variant = summary[0].get("planning_variant", "unknown") if summary else "unknown"
    lines = [
        "# Cross-model LST evaluation", "",
        f"Both models evaluate the same unchanged {variant} planning intervention.",
        "The values are predicted responses from frozen downstream models; they do not establish causal or real-world intervention effects.",
        "",
        "| Model | Cities | LST change (K) | NDVI effect (K) | POP effect (K) | LST MAE | LST RMSE | Spatial r |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {label} | {cities} | {cooling} | {ndvi} | {population} | {mae} | {rmse} | {correlation} |".format(
                label=row["label"], cities=row["cities"],
                cooling=_format_report_value(row["lst_cooling_mean_k"]),
                ndvi=_format_report_value(row["lst_ndvi_effect_mean_k"]),
                population=_format_report_value(row["lst_population_effect_mean_k"]),
                mae=_format_report_value(row["lst_mae_global_pixels"]),
                rmse=_format_report_value(row["lst_rmse_global_pixels"]),
                correlation=_format_report_value(row["lst_spatial_corr_global_pixels"]),
            )
        )
    lines.extend(["", "`LST change` is adjusted minus baseline, so negative values indicate cooling.",
                  "`NDVI effect` and `POP effect` are the corresponding factorial contrasts."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summarize_rows(*, key: str, spec: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(field: str) -> float | None:
        values = [row[field] for row in rows if row.get(field) is not None]
        return None if not values else float(np.mean(np.asarray(values, dtype=np.float64)))

    error_count = sum(int(row.get("_error_count", 0)) for row in rows)
    error_abs_sum = sum(float(row.get("_error_abs_sum", 0.0)) for row in rows)
    error_sq_sum = sum(float(row.get("_error_sq_sum", 0.0)) for row in rows)
    corr_values = (
        sum(float(row["_corr_count"]) for row in rows),
        sum(float(row["_corr_sum_prediction"]) for row in rows),
        sum(float(row["_corr_sum_target"]) for row in rows),
        sum(float(row["_corr_sum_prediction_sq"]) for row in rows),
        sum(float(row["_corr_sum_target_sq"]) for row in rows),
        sum(float(row["_corr_sum_cross"]) for row in rows),
    )
    return {
        "model": key, "label": str(spec.get("label", key)),
        "planning_variant": rows[0].get("planning_variant", "unknown"), "cities": len(rows),
        "lst_cooling_mean_k": mean("delta_lst1_vs_lst0_city"),
        "lst_ndvi_effect_mean_k": mean("delta_lst10_vs_lst0_city"),
        "lst_population_effect_mean_k": mean("delta_lst01_vs_lst0_city"),
        "lst_spatial_corr_mean": mean("lst0_spatial_corr"),
        "lst_spatial_corr_global_pixels": _correlation_from_sums(corr_values),
        "lst_mae_mean_across_cities": mean("lst0_mae"),
        "lst_rmse_mean_across_cities": mean("lst0_rmse"),
        "lst_mae_global_pixels": None if error_count <= 0 else error_abs_sum / error_count,
        "lst_rmse_global_pixels": None if error_count <= 0 else float(np.sqrt(error_sq_sum / error_count)),
    }


def _load_city_sample(planning: Any, loaders: dict[str, Any], city_id: int, year: int) -> Any:
    entry = planning._find_city_entry(loaders, scene_id=city_id, year=year)
    if entry is None:
        raise ValueError(f"City {city_id} year {year} not found in dataset")
    split_name, index = entry
    return loaders[split_name].dataset[index]


def _evaluate_one(*, key: str, spec: dict[str, Any], package: dict[str, Any], output_dir: Path,
                  device: torch.device, num_workers: int, planning: Any, dry_run: bool,
                  cities: str | None, planning_variant: str) -> list[dict[str, Any]]:
    label = str(spec.get("label", key))
    lst_name = str(spec["lst_model_name"])
    lst_file = str(spec.get("lst_checkpoint_file", "eump_downstream_model.pt"))
    lst_path = _resolve("model") / lst_name / lst_file
    print(f"[{key}] LST={lst_path}")
    if dry_run:
        if not lst_path.exists():
            print(f"[{key}] pending: checkpoint will be produced by training")
        return []
    if not lst_path.exists():
        raise FileNotFoundError(f"{key}: required LST checkpoint is missing: {lst_path}")
    lst_model, lst_context, lst_preprocessing = _compose_model(
        planning=planning, pipeline=str(spec["lst_pipeline"]),
        overrides=list(spec.get("lst_overrides", [])), model_name=lst_name,
        checkpoint_file=lst_file, device=device, num_workers=num_workers,
    )
    lst_loaders = lst_context["loaders"]
    lst_schema = lst_context["batch_schema"]
    channel_names = tuple(str(name) for name in lst_schema.raster_input_channel_names)
    ndvi_idx, _, _ = planning._resolve_source_channel(
        source_name="NDVI", preprocessing_spec=lst_preprocessing,
        raster_input_channel_names=channel_names,
    )
    pop_idx, _, _ = planning._resolve_source_channel(
        source_name="POP", preprocessing_spec=lst_preprocessing,
        raster_input_channel_names=channel_names,
    )
    lst_source_name = lst_context["target_schema"].spec.source
    lst_source = lst_preprocessing["sources"][lst_source_name]
    year = int(package.get("year", 2020))
    rows: list[dict[str, Any]] = []
    city_outputs: list[dict[str, Any]] = []
    for item in _city_items(package, cities):
        city_id = int(item["city_id"])
        intervention = item.get(planning_variant)
        if not isinstance(intervention, dict):
            raise ValueError(f"City {city_id} has no {planning_variant} planning result")
        sample = _load_city_sample(planning, lst_loaders, city_id, year)
        raster = torch.from_numpy(sample.raster_inputs).unsqueeze(0).float().to(device)
        city_mask = np.asarray(sample.city_mask).squeeze().astype(bool)
        valid = np.asarray(sample.raster_valid_mask).astype(bool).all(axis=0).squeeze()
        target_name = str(lst_context["target_schema"].name)
        target_valid = np.asarray(sample.raster_targets[target_name][1]).astype(bool).all(axis=0).squeeze()
        valid &= target_valid
        metric_mask = city_mask & valid
        ndvi0 = np.asarray(intervention["original_ndvi"]).squeeze()
        ndvi1 = np.asarray(intervention["adjusted_ndvi"]).squeeze()
        if planning_variant == "joint":
            pop0 = np.asarray(intervention["original_pop"]).squeeze()
            pop1 = np.asarray(intervention["adjusted_pop"]).squeeze()
        else:
            pop0 = raster[0, pop_idx].detach().cpu().numpy()
            pop1 = pop0
        inputs = {
            "00": _set_channels(raster, ndvi_idx=ndvi_idx, pop_idx=pop_idx, ndvi=ndvi0, pop=pop0),
            "11": _set_channels(raster, ndvi_idx=ndvi_idx, pop_idx=pop_idx, ndvi=ndvi1, pop=pop1),
            "01": _set_channels(raster, ndvi_idx=ndvi_idx, pop_idx=pop_idx, ndvi=ndvi0, pop=pop1),
            "10": _set_channels(raster, ndvi_idx=ndvi_idx, pop_idx=pop_idx, ndvi=ndvi1, pop=pop0),
        }
        lst_processed = {name: _forward_lst(lst_model, value) for name, value in inputs.items()}
        lst_physical = {name: _physical_array(value, lst_source) for name, value in lst_processed.items()}
        lst0_mean = _masked_mean(lst_physical["00"], metric_mask)
        lst1_mean = _masked_mean(lst_physical["11"], metric_mask)
        lst01_mean = _masked_mean(lst_physical["01"], metric_mask)
        lst10_mean = _masked_mean(lst_physical["10"], metric_mask)
        target_processed = np.asarray(sample.raster_targets[target_name][0]).squeeze()
        target_physical = _physical_array(target_processed, lst_source)
        baseline_error = lst_physical["00"] - target_physical
        baseline_mse = _masked_mean(baseline_error * baseline_error, valid)
        correlation_sums = _correlation_sums(lst_physical["00"], target_physical, valid)
        finite_error = baseline_error[np.asarray(valid, dtype=bool) & np.isfinite(baseline_error)]
        rows.append({
            "model": key, "label": label, "planning_variant": planning_variant, "city_id": city_id,
            "lst0_city_mean": lst0_mean, "lst1_city_mean": lst1_mean,
            "lst01_city_mean": lst01_mean, "lst10_city_mean": lst10_mean,
            "delta_lst1_vs_lst0_city": None if lst0_mean is None or lst1_mean is None else lst1_mean - lst0_mean,
            "delta_lst01_vs_lst0_city": None if lst0_mean is None or lst01_mean is None else lst01_mean - lst0_mean,
            "delta_lst10_vs_lst0_city": None if lst0_mean is None or lst10_mean is None else lst10_mean - lst0_mean,
            "interaction_city": None if None in (lst0_mean, lst1_mean, lst01_mean, lst10_mean) else lst1_mean - lst10_mean - lst01_mean + lst0_mean,
            "lst0_spatial_corr": _pearson(lst_physical["00"], target_physical, valid),
            "lst1_spatial_corr": _pearson(lst_physical["11"], target_physical, valid),
            "lst0_mae": _masked_mean(np.abs(baseline_error), valid),
            "lst0_rmse": None if baseline_mse is None else float(np.sqrt(baseline_mse)),
            "_corr_count": correlation_sums[0], "_corr_sum_prediction": correlation_sums[1],
            "_corr_sum_target": correlation_sums[2], "_corr_sum_prediction_sq": correlation_sums[3],
            "_corr_sum_target_sq": correlation_sums[4], "_corr_sum_cross": correlation_sums[5],
            "_error_count": int(finite_error.size), "_error_abs_sum": float(np.abs(finite_error).sum()),
            "_error_sq_sum": float((finite_error * finite_error).sum()),
        })
        city_outputs.append({"city_id": city_id, "lst": lst_physical})
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / f"forward_inference_{key}_summary.csv", rows)
    with (output_dir / f"forward_inference_{key}.pkl").open("wb") as handle:
        pickle.dump({
            "schema_version": 3, "model": key, "label": label,
            "planning_variant": planning_variant, "lst_checkpoint": str(lst_path),
            "cities": city_outputs, "rows": rows,
        }, handle)
    (output_dir / "artifacts.json").write_text(json.dumps({
        "model": key, "label": label, "planning_variant": planning_variant,
        "lst_checkpoint": str(lst_path), "city_count": len(rows),
        "scope": "LST-only; unchanged planning intervention",
    }, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    del lst_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = _load_config(args.config)
    models = config.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError("cross_model_eval.models must be a non-empty mapping")
    selected = None if args.models is None else {value.strip() for value in args.models.split(",") if value.strip()}
    unknown = [] if selected is None else sorted(selected.difference(models))
    if unknown:
        raise ValueError(f"Unknown models {unknown}; available={sorted(models)}")
    evaluation_specs = {key: dict(spec) for key, spec in models.items()
                        if selected is None or key in selected}
    package_path = _resolve(args.planning_data or config["planning_data"])
    output_root = _resolve(args.output_dir or config["output_dir"])
    package = _load_pickle(package_path) if not args.dry_run else {"schema_version": 1, "city_data": [], "year": 2020}
    planning = _load_module()
    device = planning._resolve_device(args.device)
    all_rows: list[dict[str, Any]] = []
    for key, spec in evaluation_specs.items():
        all_rows.extend(_evaluate_one(
            key=key, spec=spec, package=package, output_dir=output_root / key,
            device=device, num_workers=args.num_workers, planning=planning,
            dry_run=args.dry_run, cities=args.cities, planning_variant=args.planning_variant,
        ))
    if args.dry_run:
        return 0
    _write_csv(output_root / "cross_model_city_results.csv", all_rows)
    summary = []
    for key, spec in evaluation_specs.items():
        rows = [row for row in all_rows if row["model"] == key]
        if rows:
            summary.append(_summarize_rows(key=key, spec=spec, rows=rows))
    _write_csv(output_root / "cross_model_comparison_summary.csv", summary)
    (output_root / "cross_model_comparison_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    _write_comparison_report(output_root / "cross_model_report.md", summary)
    _write_comparison_figure(output_root / "cross_model_comparison.png",
                             model_specs=evaluation_specs, rows=all_rows)
    print(f"Saved LST-only cross-model results under {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
