#!/usr/bin/env python3
"""LST-only cross-model evaluation of fixed single-target NDVI interventions."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import shutil
import sys
from typing import Any

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "outputs"
COOLING_PACKAGE_PATH = ROOT / "experiments/planning_exp/outputs/cooling_only_robustness_sweep/base_lst1p00_lr1e-3_s1000/planning_data.pkl"
MODEL_CONFIG_PATH = ROOT / "experiments/planning_cross_val_exp/config/cross_model_eval_scratch.yaml"
BACKUP_ROOT = ROOT.parent / f"{ROOT.name}_backups"

SCENARIOS = (
    "gradient_dNDVI_single",
    "uniform_dNDVI_single",
    "hotspot_dNDVI_single",
)
MODELS = ("mae_v3", "one_stage_swin", "convnext_one_stage")
MODEL_LABELS = {
    "mae_v3": "EUMP (MAE v3)",
    "one_stage_swin": "Swin one-stage",
    "convnext_one_stage": "ConvNeXt one-stage",
}
TEST_METRIC_FIELDS = (
    "model", "target", "evaluation_view", "test_count", "r2",
    "rmse_physical", "mae_physical", "metric_scale", "source",
)


def _import_local(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _metadata_path(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _load_maps(reference: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    maps = {}
    for scenario in SCENARIOS:
        values = np.asarray(reference[f"delta_ndvi_{scenario}"], dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f"NDVI intervention contains nonfinite values: {scenario}")
        maps[scenario] = values
    return maps


def _load_metadata(reference: dict[str, np.ndarray]) -> tuple[list[int], list[str], dict[int, dict[str, Any]]]:
    # City IDs, names, and order are stored alongside the intervention maps.
    ids = [int(value) for value in reference["city_ids"]]
    names = [str(value) for value in reference["city_names"]]
    if len(set(ids)) != len(ids) or len(names) != len(ids):
        raise ValueError("Reference archive city IDs/names are inconsistent")
    cooling_package = _read_pickle(COOLING_PACKAGE_PATH)
    cooling_by_id = {int(row["city_id"]): row["cooling_only"] for row in cooling_package["city_data"]}
    missing = sorted(set(ids).difference(cooling_by_id))
    if missing:
        raise ValueError(f"Cooling package is missing city IDs: {missing[:10]}")
    return ids, names, cooling_by_id


def _mean(values: np.ndarray, mask: np.ndarray) -> float:
    selected = np.asarray(values, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    selected = selected[np.isfinite(selected)]
    if selected.size == 0:
        raise ValueError("No finite pixels in the city land metric mask")
    return float(selected.mean())


def _evaluate_independent_model(
    *,
    key: str,
    spec: dict[str, Any],
    city_ids: list[int],
    maps: Any,
    cooling_by_id: dict[int, dict[str, Any]],
    device: torch.device,
    num_workers: int,
    planning: Any,
    cross: Any,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    checkpoint = ROOT / "model" / str(spec["lst_model_name"]) / str(
        spec.get("lst_checkpoint_file", "eump_downstream_model.pt")
    )
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    model, context, preprocessing = cross._compose_model(
        planning=planning,
        pipeline=str(spec["lst_pipeline"]),
        overrides=list(spec.get("lst_overrides", [])),
        model_name=str(spec["lst_model_name"]),
        checkpoint_file=str(spec.get("lst_checkpoint_file", "eump_downstream_model.pt")),
        device=device,
        num_workers=num_workers,
    )
    loaders = context["loaders"]
    schema = context["batch_schema"]
    channel_names = tuple(str(name) for name in schema.raster_input_channel_names)
    ndvi_idx, _, ndvi_source = planning._resolve_source_channel(
        source_name="NDVI",
        preprocessing_spec=preprocessing,
        raster_input_channel_names=channel_names,
    )
    source_name = context["target_schema"].spec.source
    if source_name is None:
        raise ValueError(f"{key}: LST target has no physical source")
    lst_source = preprocessing["sources"][source_name]
    ndvi_stats = ndvi_source.get("stats", {}) or {}
    ndvi_std = float(ndvi_stats.get("std", 1.0))
    if not np.isfinite(ndvi_std) or ndvi_std <= 0.0:
        raise ValueError(f"{key}: invalid NDVI preprocessing std {ndvi_std}")

    year = int(_read_pickle(COOLING_PACKAGE_PATH).get("year", 2020))
    city_arrays: dict[str, list[np.ndarray]] = {"baseline": []}
    city_arrays.update({scenario: [] for scenario in SCENARIOS})
    metric_masks: list[np.ndarray] = []
    max_baseline_error = 0.0
    for index, city_id in enumerate(city_ids):
        entry = planning._find_city_entry(loaders, scene_id=int(city_id), year=year)
        if entry is None:
            raise ValueError(f"{key}: city {city_id} year {year} is absent from model dataset")
        split_name, sample_index = entry
        sample = loaders[split_name].dataset[sample_index]
        raster = torch.from_numpy(sample.raster_inputs).unsqueeze(0).float().to(device)
        city_mask = np.asarray(sample.city_mask).squeeze().astype(bool)
        valid = np.asarray(sample.raster_valid_mask).astype(bool).all(axis=0).squeeze()
        target_name = str(context["target_schema"].name)
        target_valid = np.asarray(sample.raster_targets[target_name][1]).astype(bool).all(axis=0).squeeze()
        valid &= target_valid

        original_processed = np.asarray(cooling_by_id[city_id]["original_ndvi"], dtype=np.float32).squeeze()
        raster_ndvi = raster[0, ndvi_idx].detach().cpu().numpy()
        baseline_error = float(np.max(np.abs(raster_ndvi - original_processed)))
        max_baseline_error = max(max_baseline_error, baseline_error)
        if baseline_error > 2.0e-4:
            raise ValueError(
                f"{key}: city {city_id} baseline NDVI does not match cooling package "
                f"(max processed-space error {baseline_error})"
            )

        metric_mask = city_mask & ~np.asarray(cooling_by_id[city_id]["ocean_mask"], dtype=bool).squeeze() & valid
        metric_masks.append(metric_mask)
        with torch.inference_mode():
            baseline_prediction = cross._forward_lst(model, raster)
            city_arrays["baseline"].append(cross._physical_array(baseline_prediction, lst_source).astype(np.float32))
            for scenario in SCENARIOS:
                physical_delta = np.asarray(maps[scenario][index], dtype=np.float32)
                processed_delta = physical_delta / ndvi_std
                changed = raster.clone()
                changed[0, ndvi_idx] = changed[0, ndvi_idx] + torch.as_tensor(
                    processed_delta, dtype=changed.dtype, device=device
                )
                prediction = cross._forward_lst(model, changed)
                city_arrays[scenario].append(cross._physical_array(prediction, lst_source).astype(np.float32))

        if (index + 1) % 25 == 0 or index + 1 == len(city_ids):
            print(f"[{key}] forwarded {index + 1}/{len(city_ids)} cities", flush=True)

    results = {name: np.stack(values, axis=0) for name, values in city_arrays.items()}
    masks = np.stack(metric_masks, axis=0)
    if float(np.max(np.abs(results["baseline"] - results[SCENARIOS[0]]))) == 0.0:
        raise RuntimeError(f"{key}: intervention predictions unexpectedly equal baseline")
    print(
        f"[{key}] finished; NDVI baseline max error={max_baseline_error:.3g}; "
        f"NDVI std={ndvi_std:.6g}; model={spec['lst_model_name']}",
        flush=True,
    )
    del model, context, loaders
    gc = __import__("gc")
    gc.collect()
    torch.cuda.empty_cache()
    return results, masks


def _test_metric_rows(reference_dir: Path, updated_specs: dict[str, Any]) -> list[dict[str, Any]]:
    with (reference_dir / "test_set_metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = [
            {field: row.get(field, "") for field in TEST_METRIC_FIELDS}
            for row in csv.DictReader(handle)
            if row["target"] == "LST"
        ]
    for row in rows:
        key = next((key for key in updated_specs if MODEL_LABELS[key] == row["model"]), None)
        if key is None:
            continue
        spec = updated_specs[key]
        path = ROOT / "model" / str(spec["lst_model_name"]) / "evaluation/test_metrics.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected_checkpoint = ROOT / "model" / str(spec["lst_model_name"]) / str(
            spec.get("lst_checkpoint_file", "eump_downstream_model.pt")
        )
        if Path(payload["checkpoint"]["path"]).resolve() != expected_checkpoint.resolve():
            raise ValueError(f"Test metrics checkpoint differs from evaluation checkpoint: {path}")
        view = row["evaluation_view"]
        metrics = payload["metrics"][view]
        row.update({
            "test_count": metrics["count"],
            "r2": metrics["r2"],
            "rmse_physical": metrics["rmse"],
            "mae_physical": metrics.get("mae"),
            "metric_scale": payload.get("metric_scale"),
            "source": str(path.relative_to(ROOT)),
        })
    return rows


def _summarize(
    city_ids: list[int],
    names: list[str],
    predictions: dict[str, dict[str, np.ndarray]],
    masks: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    city_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for model in MODELS:
        baseline = predictions[model]["baseline"]
        model_masks = masks[model]
        baseline_means = np.asarray([_mean(baseline[i], model_masks[i]) for i in range(len(city_ids))])
        scenario_means: dict[str, np.ndarray] = {}
        for scenario in SCENARIOS:
            adjusted = predictions[model][scenario]
            means = np.asarray([_mean(adjusted[i], model_masks[i]) for i in range(len(city_ids))])
            delta = means - baseline_means
            scenario_means[scenario] = delta
            for i, city_id in enumerate(city_ids):
                city_rows.append({
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "city_id": city_id,
                    "city_name": names[i],
                    "scenario": scenario,
                    "urban_land_pixel_count": int(model_masks[i].sum()),
                    "baseline_lst_mean_k": float(baseline_means[i]),
                    "scenario_lst_mean_k": float(means[i]),
                    "delta_lst_k": float(delta[i]),
                    "cooling_magnitude_k": float(-delta[i]),
                })
            finite = delta[np.isfinite(delta)]
            summary_rows.append({
                "model": model,
                "model_label": MODEL_LABELS[model],
                "scenario": scenario,
                "cities": int(finite.size),
                "mean_delta_lst_k": float(finite.mean()),
                "median_delta_lst_k": float(np.median(finite)),
                "std_delta_lst_k": float(finite.std()),
                "mean_cooling_magnitude_k": float(-finite.mean()),
                "cities_cooled": int((finite < 0).sum()),
                "cities_cooled_pct": float((finite < 0).mean() * 100.0),
                "mean_urban_land_pixels": float(model_masks.sum(axis=(1, 2)).mean()),
            })

        gradient = scenario_means[SCENARIOS[0]]
        for other in SCENARIOS[1:]:
            delta_advantage = scenario_means[other] - gradient
            row = next(row for row in summary_rows if row["model"] == model and row["scenario"] == SCENARIOS[0])
            row[f"gradient_beats_{other}_cities"] = int((delta_advantage > 0.0).sum())
            row[f"gradient_beats_{other}_pct"] = float((delta_advantage > 0.0).mean() * 100.0)
            row[f"gradient_extra_cooling_vs_{other}_mean_k"] = float(delta_advantage.mean())
    return city_rows, summary_rows


def _write_validation_report(
    output_dir: Path,
    city_ids: list[int],
    names: list[str],
    predictions: dict[str, dict[str, np.ndarray]],
    masks: dict[str, np.ndarray],
) -> None:
    expected_city, expected_summary = _summarize(city_ids, names, predictions, masks)
    checks: dict[str, Any] = {}
    max_city_error = 0.0
    for filename, expected_rows, key_fields in (
        ("cooling_city_results.csv", expected_city, ("model", "city_id", "scenario")),
        ("cooling_summary.csv", expected_summary, ("model", "scenario")),
    ):
        with (output_dir / filename).open(newline="", encoding="utf-8") as handle:
            actual_rows = list(csv.DictReader(handle))
        expected = {tuple(str(row[key]) for key in key_fields): row for row in expected_rows}
        actual_keys = [tuple(row[key] for key in key_fields) for row in actual_rows]
        if len(actual_keys) != len(set(actual_keys)) or set(actual_keys) != set(expected):
            raise ValueError(f"{filename}: missing or duplicate model/city/scenario rows")
        max_error = 0.0
        for actual, key in zip(actual_rows, actual_keys):
            for field, value in expected[key].items():
                if isinstance(value, str):
                    if actual[field] != value:
                        raise ValueError(f"{filename}: {field} differs for {key}")
                else:
                    error = abs(float(actual[field]) - float(value))
                    if not np.isfinite(error) or error > 1.0e-10:
                        raise ValueError(f"{filename}: {field} differs for {key}: {error}")
                    max_error = max(max_error, error)
        checks[filename] = {"rows": len(actual_rows), "matches_prediction_arrays": True, "max_numeric_error": max_error}
        if filename == "cooling_city_results.csv":
            max_city_error = max_error
    with (output_dir / "test_set_metrics.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        test_rows = list(reader)
    expected_views = {(MODEL_LABELS[key], view) for key in MODELS[1:] for view in ("global_pixels", "city_mean")}
    if fields != list(TEST_METRIC_FIELDS) or len(test_rows) != 4 or any(row["target"] != "LST" for row in test_rows):
        raise ValueError("test_set_metrics.csv: expected four LST-only rows and the current schema")
    if {(row["model"], row["evaluation_view"]) for row in test_rows} != expected_views:
        raise ValueError("test_set_metrics.csv: model/test views differ from the LST package")
    checks["test_set_metrics.csv"] = {"rows": 4, "targets": ["LST"], "columns": fields}
    report = {
        "scope": "LST-only",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "city_count": len(city_ids),
        "checks": checks,
        "max_csv_raster_error_k": max_city_error,
        "passed": True,
    }
    (output_dir / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def _write_report(
    path: Path,
    summary_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
) -> None:
    documentation = Path(__file__).resolve().parent / "README.md"
    documentation_link = os.path.relpath(documentation, path.parent)
    lines = [
        "# NDVI Intervention Cross-Model LST Results",
        "",
        f"Method, dependencies, commands, and metric definitions: [experiment README]({documentation_link}).",
        "",
        "## Cooling results",
        "",
        "| Model | NDVI allocation | Mean ΔLST (K) | Median ΔLST (K) | Cities cooled | Gradient vs uniform | Gradient vs hotspot |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    by_model_scenario = {(row["model"], row["scenario"]): row for row in summary_rows}
    for model in MODELS:
        for scenario in SCENARIOS:
            row = by_model_scenario[(model, scenario)]
            gradient_row = by_model_scenario[(model, SCENARIOS[0])]
            uniform_count = gradient_row.get("gradient_beats_uniform_dNDVI_single_cities")
            uniform_pct = gradient_row.get("gradient_beats_uniform_dNDVI_single_pct")
            uniform_adv = gradient_row.get("gradient_extra_cooling_vs_uniform_dNDVI_single_mean_k")
            hotspot_count = gradient_row.get("gradient_beats_hotspot_dNDVI_single_cities")
            hotspot_pct = gradient_row.get("gradient_beats_hotspot_dNDVI_single_pct")
            hotspot_adv = gradient_row.get("gradient_extra_cooling_vs_hotspot_dNDVI_single_mean_k")
            uniform_text = "NA" if uniform_count is None else f"{int(uniform_count)}/292 ({float(uniform_pct):.1f}%), +{float(uniform_adv):.4f} K"
            hotspot_text = "NA" if hotspot_count is None else f"{int(hotspot_count)}/292 ({float(hotspot_pct):.1f}%), +{float(hotspot_adv):.4f} K"
            lines.append(
                f"| {MODEL_LABELS[model]} | {scenario} | {row['mean_delta_lst_k']:.4f} | {row['median_delta_lst_k']:.4f} | {int(row['cities_cooled'])}/292 ({row['cities_cooled_pct']:.1f}%) | {uniform_text} | {hotspot_text} |"
            )
    lines.extend([
        "",
        "The two gradient comparison columns are repeated on each scenario row for readability. They report paired comparisons within the same city: the first value is the number and percentage of cities where gradient has a more negative ΔLST, and the final value is the mean extra cooling magnitude in Kelvin. The complete values are also stored in `cooling_summary.csv`.",
        "",
        "## Independent model test-set accuracy",
        "",
        "LST global-pixel metrics use the complete test set and physical Kelvin. City-mean metrics describe per-city aggregate accuracy. The table and `test_set_metrics.csv` contain four LST records: two views for each of Swin and ConvNeXt. ConvNeXt metrics come from the scratch model's validation-best checkpoint evaluation. These test metrics and the 292-city intervention results describe different evaluation cohorts. Blank MAE fields indicate that the source evaluation did not report that metric.",
        "",
        "| Model | Target | Test view | N | R² | RMSE (physical) | MAE (physical) |",
        "|---|---|---|---:|---:|---:|---:|",
    ])
    for row in test_rows:
        mae_value = row["mae_physical"]
        mae = "NA" if mae_value in (None, "") else f"{float(mae_value):.4f}"
        lines.append(
            f"| {row['model']} | {row['target']} | {row['evaluation_view']} | {row['test_count']} | {float(row['r2']):.4f} | {float(row['rmse_physical']):.4f} | {mae} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--models", default="convnext_one_stage", help="Comma-separated independent model keys to rerun; other predictions are reused.")
    parser.add_argument("--reference-dir", type=Path, default=OUT)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--config", type=Path, default=MODEL_CONFIG_PATH)
    args = parser.parse_args()
    updated_models = [value.strip() for value in args.models.split(",") if value.strip()]
    if not updated_models or len(set(updated_models)) != len(updated_models):
        raise ValueError("--models must contain distinct independent model keys")
    unknown = set(updated_models).difference(("one_stage_swin", "convnext_one_stage"))
    if unknown:
        raise ValueError(f"Unsupported models: {sorted(unknown)}")
    reference_dir = args.reference_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir == reference_dir:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        BACKUP_ROOT.mkdir(parents=True, exist_ok=True)
        backup = BACKUP_ROOT / ("manipulate_exp_before_update_" + stamp)
        shutil.copytree(reference_dir, backup)
        reference_dir = backup
        print(f"Backed up reference package to {backup}", flush=True)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is required; requested {device}, available={torch.cuda.is_available()}")
    torch.cuda.set_device(device)
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(reference_dir / "single_ndvi_lst_predictions.npz", allow_pickle=False) as archive:
        reference = {key: archive[key] for key in archive.files}
    maps = _load_maps(reference)
    city_ids, names, cooling_by_id = _load_metadata(reference)
    if len(city_ids) != 292:
        raise ValueError(f"Expected 292 cities, got {len(city_ids)}")
    if any(np.asarray(maps[key]).shape != (292, 256, 256) for key in SCENARIOS):
        raise ValueError("Task9 single-target map shape mismatch")

    cross = _import_local("task9_cross_model_eval", ROOT / "experiments/planning_cross_val_exp/run_cross_model_eval.py")
    planning = cross._load_module()
    raw_config = yaml.safe_load(args.config.read_text(encoding="utf-8"))["cross_model_eval"]["models"]
    # Preserve the original Task9 identifiers and archive keys while updating
    # the actual ConvNeXt checkpoint to the independently trained scratch model.
    config_keys = {"one_stage_swin": "one_stage_swin", "convnext_one_stage": "convnext_one_stage_scratch"}
    updated_specs = {key: raw_config[config_keys[key]] for key in updated_models}
    predictions: dict[str, dict[str, np.ndarray]] = {}
    masks: dict[str, np.ndarray] = {}
    for key in MODELS:
        predictions[key] = {"baseline": reference[f"lst_baseline_{key}"]}
        predictions[key].update({scenario: reference[f"lst_{key}_{scenario}"] for scenario in SCENARIOS})
        masks[key] = reference[f"metric_mask_{key}"].astype(bool)
        if key not in updated_models:
            print(f"[{key}] reused reference predictions", flush=True)

    for key in updated_models:
        model_predictions, model_masks = _evaluate_independent_model(
            key=key,
            spec=updated_specs[key],
            city_ids=city_ids,
            maps=maps,
            cooling_by_id=cooling_by_id,
            device=device,
            num_workers=args.num_workers,
            planning=planning,
            cross=cross,
        )
        predictions[key] = model_predictions
        masks[key] = model_masks
        if not np.array_equal(model_masks, reference[f"metric_mask_{key}"]):
            raise ValueError(f"{key}: metric mask differs from the reference evaluation")

    for key in ("one_stage_swin", "convnext_one_stage"):
        if not np.array_equal(masks[key], masks["mae_v3"]):
            differing = int(np.count_nonzero(masks[key] != masks["mae_v3"]))
            raise ValueError(f"{key}: city metric masks differ from EUMP in {differing} pixels")

    city_rows, summary_rows = _summarize(city_ids, names, predictions, masks)
    # Retain every existing row and column for models outside the update.
    for filename, generated, key_fields in (
        ("cooling_city_results.csv", city_rows, ("model", "city_id", "scenario")),
        ("cooling_summary.csv", summary_rows, ("model", "scenario")),
    ):
        with (reference_dir / filename).open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or [])
            original_rows = list(reader)
        generated_by_key = {tuple(str(row[field]) for field in key_fields): row for row in generated}
        if len(generated_by_key) != len(generated) or len(original_rows) != len(generated):
            raise ValueError(f"{filename}: reference/generated row counts differ")
        merged = []
        for row in original_rows:
            if row["model"] in updated_models:
                new = generated_by_key[tuple(row[field] for field in key_fields)]
                row = {field: new.get(field, "") for field in fields}
            merged.append(row)
        if filename == "cooling_city_results.csv":
            city_rows = merged
        else:
            summary_rows = merged
        _write_csv(output_dir / filename, merged)
    test_rows = _test_metric_rows(reference_dir, updated_specs)
    _write_csv(output_dir / "test_set_metrics.csv", test_rows)
    prediction_payload = dict(reference)
    for model in updated_models:
        prediction_payload[f"lst_baseline_{model}"] = predictions[model]["baseline"].astype(np.float32)
        for scenario in SCENARIOS:
            prediction_payload[f"lst_{model}_{scenario}"] = predictions[model][scenario].astype(np.float32)
        prediction_payload[f"metric_mask_{model}"] = masks[model].astype(bool)
    np.savez_compressed(output_dir / "single_ndvi_lst_predictions.npz", **prediction_payload)
    metadata = json.loads((reference_dir / "evaluation_metadata.json").read_text(encoding="utf-8"))
    for obsolete_field in ("traffic_model_fold", "historical_tci_metrics", "eump_prediction_source"):
        metadata.pop(obsolete_field, None)
    metadata["models"]["mae_v3"] = "model/eump_full_mae_v3/lst/eump_downstream_model.pt"
    metadata.update({
        "scope": "LST-only",
        "city_count": len(city_ids),
        "package_directory": _metadata_path(output_dir),
        "city_order_source": _metadata_path(output_dir / "single_ndvi_lst_predictions.npz") + "::city_ids,city_names",
        "map_source": _metadata_path(output_dir / "single_ndvi_lst_predictions.npz") + "::delta_ndvi_*",
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "updated_models": updated_models,
        "reference_directory": _metadata_path(reference_dir),
        "model_config": _metadata_path(args.config),
        "input_map_matches_reference_exactly": True,
        "convnext_training": {"encoder_initialization": "scratch", "encoder_source": "convnext_scratch", "freeze_encoder": False, "encoder_and_head_trained_jointly": True, "checkpoint_selection": "validation best"},
        "inference_method": "Task9: add physical NDVI delta / pipeline NDVI std to the observed processed NDVI channel; preserve all other input channels",
    })
    metadata.setdefault("checkpoint_sha256", {})
    for key, spec in updated_specs.items():
        checkpoint = ROOT / "model" / spec["lst_model_name"] / spec.get("lst_checkpoint_file", "eump_downstream_model.pt")
        metadata["models"][key] = str(checkpoint.relative_to(ROOT))
        with checkpoint.open("rb") as handle:
            metadata["checkpoint_sha256"][key] = hashlib.file_digest(handle, "sha256").hexdigest()
    for key, model_predictions in predictions.items():
        if any(not np.isfinite(values).all() for values in model_predictions.values()):
            raise ValueError(f"{key}: LST predictions contain nonfinite values")
    (output_dir / "evaluation_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    # CSV rows retained from the reference are strings; convert numerical report
    # fields without changing their serialized CSV representation.
    report_rows = [{key: (float(value) if key not in ("model", "model_label", "scenario") and value not in (None, "") else value) for key, value in row.items()} for row in summary_rows]
    _write_report(output_dir / "evaluation_report.md", report_rows, test_rows)
    _write_validation_report(output_dir, city_ids, names, predictions, masks)
    print(f"Saved single-target NDVI cross-model results to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
