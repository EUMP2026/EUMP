#!/usr/bin/env python3
"""Run LST prediction under externally supplied physical NDVI scenarios."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import torch
import torch.distributed as dist


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.manipulate_exp import run as base_run
from urban_embedding.core.logging import setup_logger
from urban_embedding.data.processed.dataset import build_eump_dataloaders
from urban_embedding.data.processed.pipeline_spec import (
    PipelineSpec,
    get_single_downstream_target_schema,
    validate_eump_downstream_spec,
)
from urban_embedding.training.distributed import (
    destroy_distributed_process_group,
    distributed_barrier,
    get_rank,
    get_world_size,
    init_distributed_process_group,
    resolve_device as resolve_distributed_device,
)
from urban_embedding.training.regression_metrics import inverse_processed_tensor


DEFAULT_INPUT_NPZ = PROJECT_ROOT / "experiments" / "manipulate_exp" / "all_cities_ge010_hotspot_clustered_ndvi_2020.npz"
DEFAULT_CLUSTER_SCENARIOS = ("baseline_ndvi", "cluster25", "cluster50", "cluster75", "cluster100")
DEFAULT_SCENARIOS = DEFAULT_CLUSTER_SCENARIOS
DEFAULT_COUNTERFACTUAL_SCENARIOS = (
    "ndvi0",
    "ndvi_urban",
    "ndvi_buffer_5km",
    "ndvi_buffer_10km",
    "ndvi_both_5km",
    "ndvi_both_10km",
)
BASELINE_SCENARIO_KEYS = {"baseline_ndvi", "ndvi0"}
EXTRA_METADATA_KEYS = (
    "scene_ids",
    "city_name",
    "country",
    "source_splits",
    "urban_mask",
    "buffer_5km",
    "buffer_10km",
    "urban_n_pixels",
    "buffer_5km_n_actual",
    "buffer_5km_n_theoretical",
    "buffer_5km_ratio",
    "buffer_10km_n_actual",
    "buffer_10km_n_theoretical",
    "buffer_10km_ratio",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Predict raster LST for externally supplied physical NDVI scenarios. "
            "Other EUMP input channels are kept unchanged."
        )
    )
    parser.add_argument("--input-npz", type=Path, default=DEFAULT_INPUT_NPZ)
    parser.add_argument(
        "--meta-csv",
        type=Path,
        default=None,
        help="Optional metadata CSV keyed by scene_id. Defaults to <input_npz>_meta.csv when present.",
    )
    parser.add_argument("--config-path", type=Path, default=base_run.DEFAULT_CONFIG_PATH)
    parser.add_argument("--year", type=int, default=2020)
    parser.add_argument("--source", default="NDVI")
    parser.add_argument(
        "--scenarios",
        default="auto",
        help=(
            "Comma-separated NDVI scenario keys, or 'auto'. Auto supports the "
            "clustered schema and ndvi_counterfactual_scenarios.npz."
        ),
    )
    parser.add_argument("--limit-cities", type=int, default=None)
    parser.add_argument("--checkpoint-source", choices=("best_model", "training_checkpoint"), default=None)
    parser.add_argument("--checkpoint-file", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra config overrides, e.g. data=eump_full/server_v2 pipeline=eump_full/mae_v3/lst",
    )
    return parser.parse_args(argv)


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _default_scenarios_for_payload(keys: set[str]) -> tuple[str, ...]:
    if set(DEFAULT_COUNTERFACTUAL_SCENARIOS).issubset(keys):
        return DEFAULT_COUNTERFACTUAL_SCENARIOS
    if set(DEFAULT_CLUSTER_SCENARIOS).issubset(keys):
        return DEFAULT_CLUSTER_SCENARIOS
    raise ValueError(
        "Could not auto-detect NDVI scenario keys. Provide --scenarios explicitly; "
        f"available keys: {sorted(keys)}"
    )


def _parse_scenarios(
    value: str | tuple[str, ...] | list[str],
    *,
    available_keys: set[str] | None = None,
) -> tuple[str, ...]:
    if isinstance(value, str):
        if value.strip().lower() == "auto":
            if available_keys is None:
                raise ValueError("--scenarios auto requires available_keys")
            scenarios = _default_scenarios_for_payload(available_keys)
        else:
            scenarios = tuple(part.strip() for part in value.split(",") if part.strip())
    else:
        scenarios = tuple(str(part).strip() for part in value if str(part).strip())
    if not scenarios:
        raise ValueError("--scenarios must contain at least one scenario key")
    if scenarios[0] not in BASELINE_SCENARIO_KEYS:
        raise ValueError(
            "The first scenario must be a baseline key "
            f"({sorted(BASELINE_SCENARIO_KEYS)}) so deltas are defined against it"
        )
    if len(set(scenarios)) != len(scenarios):
        raise ValueError(f"Scenario keys must be unique, got {scenarios!r}")
    return scenarios


def _scenario_name(key: str) -> str:
    return "baseline" if key == "baseline_ndvi" else key


def _lst_key_for_scenario(key: str) -> str:
    return f"{_scenario_name(key)}_lst"


def _default_output_dir(input_npz: Path, year: int) -> Path:
    if input_npz.stem == "ndvi_counterfactual_scenarios":
        return Path(f"experiments/manipulate_exp/outputs/ndvi_counterfactual_lst_{int(year)}")
    return Path(f"experiments/manipulate_exp/outputs/custom_ndvi_lst_{int(year)}")


def _default_meta_csv_path(input_npz: Path) -> Path:
    return input_npz.with_name(f"{input_npz.stem}_meta.csv")


def _load_city_metadata(meta_csv_path: Path, selected_ids: np.ndarray) -> dict[str, np.ndarray]:
    by_scene: dict[str, dict[str, str]] = {}
    with meta_csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "scene_id" not in reader.fieldnames:
            raise ValueError(f"Metadata CSV must contain a scene_id column: {meta_csv_path}")
        for row in reader:
            by_scene[str(row["scene_id"])] = row
    missing = [str(scene_id) for scene_id in selected_ids if str(scene_id) not in by_scene]
    if missing:
        preview = ", ".join(missing[:10])
        suffix = "" if len(missing) <= 10 else f", ... ({len(missing)} missing total)"
        raise ValueError(f"Metadata CSV is missing scene_id(s): {preview}{suffix}")
    return {
        "city_name": np.asarray([by_scene[str(scene_id)].get("city_name", "") for scene_id in selected_ids]),
        "country": np.asarray([by_scene[str(scene_id)].get("country", "") for scene_id in selected_ids]),
    }


def _load_custom_ndvi_payload(
    path: Path,
    *,
    scenario_keys: tuple[str, ...],
    limit_cities: int | None = None,
    meta_csv_path: Path | None = None,
) -> dict[str, np.ndarray]:
    path = _resolve_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Custom NDVI NPZ does not exist: {path}")
    with np.load(str(path), allow_pickle=False) as payload:
        id_key = "selected_ids" if "selected_ids" in payload.files else "scene_ids" if "scene_ids" in payload.files else None
        mask_key = "city_mask" if "city_mask" in payload.files else "urban_mask" if "urban_mask" in payload.files else None
        required = {"urban_mask_fraction", *scenario_keys}
        if id_key is None:
            required.add("selected_ids")
        if mask_key is None:
            required.add("city_mask")
        missing = sorted(required.difference(payload.files))
        if missing:
            raise ValueError(f"Custom NDVI NPZ is missing required keys: {missing}")

        selected_ids = np.asarray(payload[id_key]).astype(str)
        n = int(selected_ids.shape[0])
        if n <= 0:
            raise ValueError(f"Custom NDVI NPZ has no {id_key}")
        if len(set(selected_ids.tolist())) != n:
            raise ValueError(f"Custom NDVI NPZ {id_key} must be unique")
        if limit_cities is not None:
            limit = int(limit_cities)
            if limit <= 0:
                raise ValueError("--limit-cities must be > 0 when provided")
            n = min(n, limit)
            selected_ids = selected_ids[:n]

        city_mask = np.asarray(payload[mask_key][:n]).astype(bool, copy=False)
        urban_mask_fraction = np.asarray(payload["urban_mask_fraction"][:n], dtype=np.float32)
        if city_mask.ndim != 3:
            raise ValueError(f"{mask_key} must have shape (N,H,W), got {city_mask.shape}")
        if city_mask.shape[0] != n:
            raise ValueError(f"{mask_key} N={city_mask.shape[0]} does not match {id_key} N={n}")
        if urban_mask_fraction.shape != (n,):
            raise ValueError(f"urban_mask_fraction must have shape ({n},), got {urban_mask_fraction.shape}")
        computed_fraction = city_mask.reshape(n, -1).mean(axis=1).astype(np.float32)
        if not np.allclose(urban_mask_fraction, computed_fraction, atol=1e-7, rtol=0):
            max_diff = float(np.max(np.abs(urban_mask_fraction - computed_fraction)))
            raise ValueError(f"urban_mask_fraction does not match {mask_key} mean; max_abs_diff={max_diff:g}")

        selected_ids_int = (
            np.asarray(payload["selected_ids_int"][:n], dtype=np.int32)
            if "selected_ids_int" in payload.files
            else np.asarray(payload[id_key][:n], dtype=np.int32)
        )
        selected_indices = (
            np.asarray(payload["selected_indices"][:n], dtype=np.int32)
            if "selected_indices" in payload.files
            else np.arange(n, dtype=np.int32)
        )

        result: dict[str, np.ndarray] = {
            "selected_ids": selected_ids,
            "selected_ids_int": selected_ids_int,
            "selected_indices": selected_indices,
            "urban_mask_fraction": urban_mask_fraction,
            "city_mask": city_mask,
            "urban_mask": city_mask,
            "scenario_names": np.asarray([_scenario_name(key) for key in scenario_keys], dtype="<U32"),
        }
        if id_key == "scene_ids" or "scene_ids" in payload.files:
            result["scene_ids"] = np.asarray(payload["scene_ids"][:n], dtype=np.int32)
        if "splits" in payload.files:
            result["source_splits"] = np.asarray(payload["splits"][:n]).astype(str)
        if meta_csv_path is not None:
            result.update(_load_city_metadata(meta_csv_path, selected_ids))
        for key in scenario_keys:
            arr = np.asarray(payload[key][:n], dtype=np.float32)
            if arr.shape != city_mask.shape:
                raise ValueError(f"{key} must have shape {city_mask.shape}, got {arr.shape}")
            if not np.isfinite(arr).all():
                raise ValueError(f"{key} contains NaN or Inf values")
            result[key] = arr

        for key in ("buffer_5km", "buffer_10km"):
            if key in payload.files:
                arr = np.asarray(payload[key][:n]).astype(bool, copy=False)
                if arr.shape != city_mask.shape:
                    raise ValueError(f"{key} must have shape {city_mask.shape}, got {arr.shape}")
                result[key] = arr

        if "urban_n_pixels" in payload.files:
            urban_n_pixels = np.asarray(payload["urban_n_pixels"][:n], dtype=np.int64)
            computed_pixels = city_mask.reshape(n, -1).sum(axis=1).astype(np.int64)
            if not np.array_equal(urban_n_pixels, computed_pixels):
                max_diff = int(np.max(np.abs(urban_n_pixels - computed_pixels)))
                raise ValueError(f"urban_n_pixels does not match {mask_key} sum; max_abs_diff={max_diff}")
            result["urban_n_pixels"] = urban_n_pixels

        for key in (
            "buffer_5km_n_actual",
            "buffer_5km_n_theoretical",
            "buffer_5km_ratio",
            "buffer_10km_n_actual",
            "buffer_10km_n_theoretical",
            "buffer_10km_ratio",
        ):
            if key in payload.files:
                result[key] = np.asarray(payload[key][:n])

        for mask_name, count_name in (
            ("buffer_5km", "buffer_5km_n_actual"),
            ("buffer_10km", "buffer_10km_n_actual"),
        ):
            if mask_name in result and count_name in result:
                expected = result[mask_name].reshape(n, -1).sum(axis=1).astype(np.int64)
                observed = np.asarray(result[count_name], dtype=np.int64)
                if not np.array_equal(observed, expected):
                    max_diff = int(np.max(np.abs(observed - expected)))
                    raise ValueError(f"{count_name} does not match {mask_name} sum; max_abs_diff={max_diff}")

        for key in ("hotspot_quantile", "n_hotspots", "min_hotspot_pixels"):
            if key in payload.files:
                result[key] = np.asarray(payload[key])
        return result


def _output_stem(input_npz: Path, target_name: str, year: int) -> str:
    stem = input_npz.stem
    suffix = f"_{int(year)}"
    target = target_name.lower()
    if stem.endswith(suffix):
        return f"{stem[:-len(suffix)]}_{target}{suffix}"
    return f"{stem}_{target}_{int(year)}"


def _merged_runtime_config(args: argparse.Namespace, *, source_name: str, year: int) -> dict[str, Any]:
    cfg = base_run._load_exp_config(args.config_path)
    cfg["source"] = source_name
    cfg["year"] = int(year)
    cfg["checkpoint_source"] = str(args.checkpoint_source or cfg.get("checkpoint_source") or "best_model")
    cfg["checkpoint_file"] = None if args.checkpoint_file is None else str(args.checkpoint_file)
    cfg["output_dir"] = str(args.output_dir)
    return cfg


def _collect_selected_entries(
    loaders: dict[str, Any],
    *,
    selected_ids: np.ndarray,
    year: int,
) -> list[dict[str, Any]]:
    entry_by_scene: dict[str, dict[str, Any]] = {}
    for entry in base_run._collect_bulk_entries(loaders, year=year, limit_cities=None):
        entry_by_scene[str(entry["scene_id"])] = entry
    missing = [str(scene_id) for scene_id in selected_ids if str(scene_id) not in entry_by_scene]
    if missing:
        preview = ", ".join(missing[:20])
        suffix = "" if len(missing) <= 20 else f", ... ({len(missing)} missing total)"
        raise ValueError(f"Selected scene IDs are missing from dataloaders for year={year}: {preview}{suffix}")
    return [
        {
            **entry_by_scene[str(scene_id)],
            "order_index": int(idx),
        }
        for idx, scene_id in enumerate(selected_ids)
    ]


def _batch_city_mask_array(city_mask_tensor: torch.Tensor) -> np.ndarray:
    arr = city_mask_tensor.detach().cpu().numpy()
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 3:
        raise ValueError(f"Expected batch city_mask shape (B,H,W) or (B,1,H,W), got {arr.shape}")
    return arr > 0


def _replace_raster_channel_with_physical_values(
    *,
    raster_inputs: torch.Tensor,
    raster_valid_mask: torch.Tensor,
    channel_index: int,
    physical_values: np.ndarray,
    source_spec: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    if physical_values.ndim != 3:
        raise ValueError(f"physical_values must have shape (B,H,W), got {physical_values.shape}")
    selected_valid = raster_valid_mask[:, channel_index : channel_index + 1, :, :]
    values = torch.from_numpy(physical_values.astype(np.float32, copy=False)).to(device=device).unsqueeze(1)
    processed_values, _ = base_run._physical_to_processed_tensor(values, selected_valid, source_spec)
    updated = raster_inputs.clone()
    updated[:, channel_index : channel_index + 1, :, :] = processed_values
    return updated


def _concat_chunks(chunks: list[np.ndarray], *, shape_tail: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    if chunks:
        return np.concatenate(chunks, axis=0)
    return np.empty((0, *shape_tail), dtype=dtype)


def _write_rank_shard(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(path), **payload)


def _merge_custom_ndvi_shards(
    *,
    shard_paths: list[Path],
    output_path: Path,
    stats_csv_path: Path,
    summary_path: Path,
    summary: dict[str, Any],
    scenario_keys: tuple[str, ...],
    keep_shards: bool = False,
) -> None:
    loaded = [np.load(str(path), allow_pickle=False) for path in shard_paths]
    try:
        order_indices = np.concatenate([item["order_indices"] for item in loaded], axis=0)
        order = np.argsort(order_indices, kind="stable")
        output_payload: dict[str, np.ndarray] = {}
        keys = [
            "selected_ids",
            "selected_ids_int",
            "selected_indices",
            "years",
            "splits",
            "crop_boxes",
            "urban_mask_fraction",
            "city_mask",
        ]
        keys.extend(key for key in EXTRA_METADATA_KEYS if key in loaded[0].files and key not in keys)
        for key in keys:
            output_payload[key] = np.concatenate([item[key] for item in loaded], axis=0)[order]
        output_payload["scenario_names"] = loaded[0]["scenario_names"]
        for key in scenario_keys:
            output_payload[key] = np.concatenate([item[key] for item in loaded], axis=0)[order]
            output_payload[_lst_key_for_scenario(key)] = np.concatenate(
                [item[_lst_key_for_scenario(key)] for item in loaded], axis=0
            )[order]

        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(output_path), **output_payload)
        stats = _write_custom_ndvi_stats_csv(
            output_payload=output_payload,
            stats_csv_path=stats_csv_path,
            scenario_keys=scenario_keys,
        )
        summary["data"]["total_cities"] = int(output_payload["selected_ids"].shape[0])
        summary["stats"] = stats
        summary["outputs"] = {
            "npz": str(output_path),
            "stats_csv": str(stats_csv_path),
            "summary": str(summary_path),
        }
        summary_path.write_text(json.dumps(base_run._jsonable(summary), indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    finally:
        for item in loaded:
            item.close()
        if not keep_shards:
            for path in shard_paths:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass


def _stat_values(values: np.ndarray) -> dict[str, int | float | None]:
    return base_run._stat_values(values.astype(np.float32, copy=False))


def _write_custom_ndvi_stats_csv(
    *,
    output_payload: dict[str, np.ndarray],
    stats_csv_path: Path,
    scenario_keys: tuple[str, ...],
) -> dict[str, Any]:
    fieldnames = [
        "scene_id",
        "city_name",
        "country",
        "year",
        "split",
        "source_split",
        "scenario",
        "crop_top",
        "crop_left",
        "crop_bottom",
        "crop_right",
        "urban_mask_fraction",
        "urban_n_pixels",
        "buffer_5km_n_actual",
        "buffer_5km_n_theoretical",
        "buffer_5km_ratio",
        "buffer_10km_n_actual",
        "buffer_10km_n_theoretical",
        "buffer_10km_ratio",
        "city_pixel_count",
        "mean_ndvi_city",
        "mean_lst_city",
        "mean_lst_delta_city",
        "mean_cooling_city",
        "median_delta_lst_city",
        "min_delta_lst_city",
        "max_delta_lst_city",
        "std_delta_lst_city",
        "buffer_5km_pixel_count",
        "mean_ndvi_buffer_5km",
        "mean_lst_buffer_5km",
        "mean_lst_delta_buffer_5km",
        "mean_cooling_buffer_5km",
        "buffer_10km_pixel_count",
        "mean_ndvi_buffer_10km",
        "mean_lst_buffer_10km",
        "mean_lst_delta_buffer_10km",
        "mean_cooling_buffer_10km",
    ]
    baseline_lst = output_payload[_lst_key_for_scenario(scenario_keys[0])]
    aggregate: dict[str, Any] = {
        "definition": {
            "lst_delta": "scenario_lst - baseline_lst",
            "cooling": "baseline_lst - scenario_lst; positive means lower prediction than baseline",
            "scope": "city_mask/urban_mask pixels with finite baseline/scenario LST predictions",
        },
        "per_scenario": {},
    }

    def optional_value(key: str, idx: int, cast: Any = float) -> Any:
        if key not in output_payload:
            return ""
        return cast(output_payload[key][idx])

    def masked_region_metrics(
        *,
        prefix: str,
        mask: np.ndarray | None,
        ndvi_values: np.ndarray,
        baseline_values: np.ndarray,
        current_values: np.ndarray,
    ) -> dict[str, Any]:
        if mask is None:
            return {}
        valid = mask & np.isfinite(baseline_values) & np.isfinite(current_values)
        ndvi_valid = mask & np.isfinite(ndvi_values)
        result: dict[str, Any] = {
            f"{prefix}_pixel_count": int(np.count_nonzero(valid)),
            f"mean_ndvi_{prefix}": "",
            f"mean_lst_{prefix}": "",
            f"mean_lst_delta_{prefix}": "",
            f"mean_cooling_{prefix}": "",
        }
        if np.any(ndvi_valid):
            result[f"mean_ndvi_{prefix}"] = float(np.mean(ndvi_values[ndvi_valid]))
        if np.any(valid):
            delta_values = current_values[valid] - baseline_values[valid]
            result[f"mean_lst_{prefix}"] = float(np.mean(current_values[valid]))
            result[f"mean_lst_delta_{prefix}"] = float(np.mean(delta_values))
            result[f"mean_cooling_{prefix}"] = float(np.mean(-delta_values))
        return result

    stats_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for key in scenario_keys:
            scenario = _scenario_name(key)
            lst = output_payload[_lst_key_for_scenario(key)]
            ndvi = output_payload[key]
            city_mean_delta: list[float] = []
            city_mean_cooling: list[float] = []
            city_mean_lst: list[float] = []
            city_mean_ndvi: list[float] = []
            total_city_pixels = 0
            weighted_delta_sum = 0.0
            weighted_cooling_sum = 0.0

            for idx in range(int(lst.shape[0])):
                crop_box = output_payload["crop_boxes"][idx]
                city_mask = output_payload["city_mask"][idx].astype(bool)
                base = baseline_lst[idx]
                current = lst[idx]
                valid = city_mask & np.isfinite(base) & np.isfinite(current)
                ndvi_valid = city_mask & np.isfinite(ndvi[idx])
                pixel_count = int(np.count_nonzero(valid))
                row: dict[str, Any] = {
                    "scene_id": str(output_payload["selected_ids"][idx]),
                    "city_name": optional_value("city_name", idx, str),
                    "country": optional_value("country", idx, str),
                    "year": int(output_payload["years"][idx]),
                    "split": str(output_payload["splits"][idx]),
                    "source_split": "" if "source_splits" not in output_payload else str(output_payload["source_splits"][idx]),
                    "scenario": scenario,
                    "crop_top": int(crop_box[0]),
                    "crop_left": int(crop_box[1]),
                    "crop_bottom": int(crop_box[2]),
                    "crop_right": int(crop_box[3]),
                    "urban_mask_fraction": float(output_payload["urban_mask_fraction"][idx]),
                    "urban_n_pixels": optional_value("urban_n_pixels", idx, int),
                    "buffer_5km_n_actual": optional_value("buffer_5km_n_actual", idx, int),
                    "buffer_5km_n_theoretical": optional_value("buffer_5km_n_theoretical", idx, int),
                    "buffer_5km_ratio": optional_value("buffer_5km_ratio", idx, float),
                    "buffer_10km_n_actual": optional_value("buffer_10km_n_actual", idx, int),
                    "buffer_10km_n_theoretical": optional_value("buffer_10km_n_theoretical", idx, int),
                    "buffer_10km_ratio": optional_value("buffer_10km_ratio", idx, float),
                    "city_pixel_count": pixel_count,
                    "mean_ndvi_city": "",
                    "mean_lst_city": "",
                    "mean_lst_delta_city": "",
                    "mean_cooling_city": "",
                    "median_delta_lst_city": "",
                    "min_delta_lst_city": "",
                    "max_delta_lst_city": "",
                    "std_delta_lst_city": "",
                }
                if np.any(ndvi_valid):
                    row["mean_ndvi_city"] = float(np.mean(ndvi[idx][ndvi_valid]))
                if pixel_count > 0:
                    delta_values = current[valid] - base[valid]
                    cooling_values = -delta_values
                    delta_stats = _stat_values(delta_values)
                    mean_delta = float(delta_stats["mean"])
                    mean_cooling = float(np.mean(cooling_values))
                    mean_lst = float(np.mean(current[valid]))
                    row.update(
                        {
                            "mean_lst_city": mean_lst,
                            "mean_lst_delta_city": mean_delta,
                            "mean_cooling_city": mean_cooling,
                            "median_delta_lst_city": delta_stats["median"],
                            "min_delta_lst_city": delta_stats["min"],
                            "max_delta_lst_city": delta_stats["max"],
                            "std_delta_lst_city": delta_stats["std"],
                        }
                    )
                    city_mean_delta.append(mean_delta)
                    city_mean_cooling.append(mean_cooling)
                    city_mean_lst.append(mean_lst)
                    if row["mean_ndvi_city"] != "":
                        city_mean_ndvi.append(float(row["mean_ndvi_city"]))
                    total_city_pixels += pixel_count
                    weighted_delta_sum += float(np.sum(delta_values))
                    weighted_cooling_sum += float(np.sum(cooling_values))
                if "buffer_5km" in output_payload:
                    row.update(masked_region_metrics(
                        prefix="buffer_5km",
                        mask=output_payload["buffer_5km"][idx].astype(bool),
                        ndvi_values=ndvi[idx],
                        baseline_values=base,
                        current_values=current,
                    ))
                if "buffer_10km" in output_payload:
                    row.update(masked_region_metrics(
                        prefix="buffer_10km",
                        mask=output_payload["buffer_10km"][idx].astype(bool),
                        ndvi_values=ndvi[idx],
                        baseline_values=base,
                        current_values=current,
                    ))
                writer.writerow(row)

            aggregate["per_scenario"][scenario] = {
                "city_count": int(lst.shape[0]),
                "valid_city_count": int(len(city_mean_delta)),
                "city_pixel_count": int(total_city_pixels),
                "city_mean_ndvi": _stat_values(np.asarray(city_mean_ndvi, dtype=np.float32)),
                "city_mean_lst": _stat_values(np.asarray(city_mean_lst, dtype=np.float32)),
                "city_mean_lst_delta": _stat_values(np.asarray(city_mean_delta, dtype=np.float32)),
                "city_mean_cooling": _stat_values(np.asarray(city_mean_cooling, dtype=np.float32)),
                "pixel_weighted_mean_lst_delta": (
                    None if total_city_pixels <= 0 else float(weighted_delta_sum / total_city_pixels)
                ),
                "pixel_weighted_mean_cooling": (
                    None if total_city_pixels <= 0 else float(weighted_cooling_sum / total_city_pixels)
                ),
            }
    return aggregate


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    source_name = str(args.source)
    if source_name.upper() != "NDVI":
        raise ValueError("run_custom_ndvi.py currently supports only --source NDVI")

    input_npz = _resolve_path(args.input_npz)
    with np.load(str(input_npz), allow_pickle=False) as payload_for_keys:
        available_keys = set(payload_for_keys.files)
    scenario_keys = _parse_scenarios(args.scenarios, available_keys=available_keys)
    if args.output_dir is None:
        args.output_dir = _default_output_dir(input_npz, int(args.year))
    meta_csv_path = _resolve_path(args.meta_csv) if args.meta_csv is not None else _default_meta_csv_path(input_npz)
    if not meta_csv_path.exists():
        meta_csv_path = None
    custom_payload = _load_custom_ndvi_payload(
        input_npz,
        scenario_keys=scenario_keys,
        limit_cities=args.limit_cities,
        meta_csv_path=meta_csv_path,
    )
    target_year = int(args.year)

    cfg_merged = _merged_runtime_config(args, source_name=source_name, year=target_year)

    hydra_cfg = base_run._compose_hydra_config(args, num_workers=int(args.num_workers))
    base_run._apply_checkpoint_overrides(hydra_cfg, cfg_merged)
    hydra_cfg["pipeline"]["year_range"] = [target_year, target_year]
    hydra_cfg["evaluation"]["split"] = "test"

    distributed_requested = int(os.environ.get("WORLD_SIZE", "1")) > 1
    initialized_distributed = False
    if distributed_requested and not (dist.is_available() and dist.is_initialized()):
        init_distributed_process_group()
        initialized_distributed = True

    try:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        rank = get_rank()
        world_size = get_world_size()
        if distributed_requested and str(args.device).strip().lower() == "auto":
            device = resolve_distributed_device(local_rank)
        else:
            device = base_run._resolve_device(str(args.device))

        logger = setup_logger(log_dir=None, rank=rank, log_level=str(hydra_cfg["logging"]["level"]))
        base_run._configure_torch_multiprocessing(logger, rank=rank)
        if rank == 0:
            logger.info(
                f"custom_ndvi_LST: input={input_npz} year={target_year} "
                f"scenarios={','.join(scenario_keys)} source={source_name} "
                f"world_size={world_size}"
            )

        pipeline_spec = PipelineSpec.from_mapping(hydra_cfg["pipeline"])
        validate_eump_downstream_spec(pipeline_spec)
        train_dl, val_dl, test_dl, batch_schema = build_eump_dataloaders(
            data_config=hydra_cfg["data"],
            pipeline_spec=pipeline_spec,
            logger=logger,
            random_seed=int(hydra_cfg["experiment"]["random_seed"]),
            batch_size=int(args.batch_size),
            local_rank=local_rank,
            world_size=world_size,
            is_distributed=False,
            shuffle_train=False,
        )
        loaders = {"train": train_dl, "val": val_dl, "test": test_dl}
        preprocessing_spec = train_dl.dataset.preprocessing_spec
        channel_index, channel_name, source_spec = base_run._resolve_source_channel(
            source_name=source_name,
            preprocessing_spec=preprocessing_spec,
            raster_input_channel_names=batch_schema.raster_input_channel_names,
        )

        target_schema = get_single_downstream_target_schema(batch_schema)
        target_name = target_schema.name
        target_kind = base_run._target_kind(batch_schema, target_name)
        if target_kind != "raster" or target_name.lower() != "lst":
            raise ValueError(f"run_custom_ndvi.py expects raster LST target, got target={target_name!r} kind={target_kind!r}")
        target_source_name = target_schema.spec.source
        if target_source_name is None:
            raise ValueError(f"Downstream target {target_name!r} has no source name in the pipeline spec")
        target_source_spec = preprocessing_spec["sources"][target_source_name]

        model, checkpoint_path, checkpoint_source, _ = base_run._build_model_and_load_checkpoint(
            hydra_cfg=hydra_cfg,
            batch_schema=batch_schema,
            target_schema=target_schema,
            device=device,
        )
        if rank == 0:
            logger.info(f"Loaded checkpoint: {checkpoint_path}")
            logger.info(
                f"Resolved source {source_name!r} -> channel {channel_name!r} "
                f"index={channel_index} mode={source_spec['mode']}"
            )

        selected_entries = _collect_selected_entries(
            loaders,
            selected_ids=custom_payload["selected_ids"],
            year=target_year,
        )
        rank_entries = selected_entries[rank::world_size]
        indices_by_split: dict[str, list[int]] = {"train": [], "val": [], "test": []}
        order_by_sample: dict[tuple[str, str, int], int] = {}
        for entry in rank_entries:
            split_name = str(entry["split"])
            indices_by_split[split_name].append(int(entry["index"]))
            order_by_sample[(split_name, str(entry["scene_id"]), int(entry["year"]))] = int(entry["order_index"])

        output_dir = base_run._resolve_output_base(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        shard_dir = output_dir / "_custom_ndvi_shards"
        if rank == 0 and shard_dir.exists():
            shutil.rmtree(shard_dir)
        distributed_barrier(local_rank)
        shard_dir.mkdir(parents=True, exist_ok=True)

        crop_size = pipeline_spec.crop_size
        if isinstance(crop_size, tuple):
            crop_h, crop_w = int(crop_size[0]), int(crop_size[1])
        else:
            crop_h = crop_w = int(crop_size)

        meta_chunks: dict[str, list[np.ndarray]] = {
            "order_indices": [],
            "selected_ids": [],
            "selected_ids_int": [],
            "selected_indices": [],
            "years": [],
            "splits": [],
            "crop_boxes": [],
            "urban_mask_fraction": [],
            "city_mask": [],
        }
        extra_meta_keys = [key for key in EXTRA_METADATA_KEYS if key in custom_payload and key not in meta_chunks]
        for key in extra_meta_keys:
            meta_chunks[key] = []
        ndvi_chunks: dict[str, list[np.ndarray]] = {key: [] for key in scenario_keys}
        lst_chunks: dict[str, list[np.ndarray]] = {_lst_key_for_scenario(key): [] for key in scenario_keys}

        with torch.no_grad():
            for split_name in ("train", "val", "test"):
                split_loader = base_run._selected_subset_loader(
                    loaders[split_name],
                    indices_by_split[split_name],
                    batch_size=int(args.batch_size),
                )
                if split_loader is None:
                    continue
                for batch in split_loader:
                    batch_size = int(batch.raster_inputs.shape[0])
                    raster_inputs = batch.raster_inputs.to(device)
                    scalar_inputs = batch.scalar_inputs.to(device)
                    raster_valid = batch.raster_valid_mask.to(device)
                    batch_city_mask = batch.city_mask.to(device)
                    batch_mask_np = _batch_city_mask_array(batch_city_mask)

                    batch_order_indices: list[int] = []
                    years: list[int] = []
                    splits: list[str] = []
                    crop_boxes: list[tuple[int, int, int, int]] = []
                    for meta in batch.meta:
                        scene_id = str(meta["scene_id"])
                        year = int(meta["year"])
                        order_index = order_by_sample[(split_name, scene_id, year)]
                        batch_order_indices.append(order_index)
                        years.append(year)
                        splits.append(split_name)
                        crop_boxes.append(tuple(int(value) for value in meta.get("crop_box", (0, 0, crop_h, crop_w))))

                    batch_indices_np = np.asarray(batch_order_indices, dtype=np.int64)
                    expected_mask = custom_payload["city_mask"][batch_indices_np]
                    if expected_mask.shape != batch_mask_np.shape or not np.array_equal(expected_mask, batch_mask_np):
                        bad_scene = str(custom_payload["selected_ids"][int(batch_indices_np[0])])
                        raise ValueError(f"city_mask mismatch between custom NPZ and dataloader near scene_id={bad_scene}")

                    meta_chunks["order_indices"].append(batch_indices_np)
                    meta_chunks["selected_ids"].append(custom_payload["selected_ids"][batch_indices_np])
                    meta_chunks["selected_ids_int"].append(custom_payload["selected_ids_int"][batch_indices_np])
                    meta_chunks["selected_indices"].append(custom_payload["selected_indices"][batch_indices_np])
                    meta_chunks["years"].append(np.asarray(years, dtype=np.int32))
                    meta_chunks["splits"].append(np.asarray(splits, dtype="<U8"))
                    meta_chunks["crop_boxes"].append(np.asarray(crop_boxes, dtype=np.int32))
                    meta_chunks["urban_mask_fraction"].append(custom_payload["urban_mask_fraction"][batch_indices_np])
                    meta_chunks["city_mask"].append(expected_mask.astype(bool, copy=False))
                    for key in extra_meta_keys:
                        meta_chunks[key].append(custom_payload[key][batch_indices_np])

                    for key in scenario_keys:
                        ndvi_np = custom_payload[key][batch_indices_np].astype(np.float32, copy=False)
                        scenario_inputs = _replace_raster_channel_with_physical_values(
                            raster_inputs=raster_inputs,
                            raster_valid_mask=raster_valid,
                            channel_index=channel_index,
                            physical_values=ndvi_np,
                            source_spec=source_spec,
                            device=device,
                        )
                        raster_pred, _ = model(scenario_inputs, scalar_inputs)
                        lst_physical = inverse_processed_tensor(raster_pred[target_name], target_source_spec)
                        lst_np = lst_physical.detach().cpu().numpy().astype(np.float32, copy=False)
                        if lst_np.ndim == 4 and lst_np.shape[1] == 1:
                            lst_np = lst_np[:, 0]
                        if lst_np.shape != (batch_size, crop_h, crop_w):
                            raise ValueError(f"Unexpected LST prediction shape {lst_np.shape}; expected {(batch_size, crop_h, crop_w)}")
                        ndvi_chunks[key].append(ndvi_np)
                        lst_chunks[_lst_key_for_scenario(key)].append(lst_np)

        shard_payload: dict[str, np.ndarray] = {
            "order_indices": _concat_chunks(meta_chunks["order_indices"], shape_tail=(), dtype=np.dtype(np.int64)),
            "selected_ids": _concat_chunks(meta_chunks["selected_ids"], shape_tail=(), dtype=np.dtype("<U32")),
            "selected_ids_int": _concat_chunks(meta_chunks["selected_ids_int"], shape_tail=(), dtype=np.dtype(np.int32)),
            "selected_indices": _concat_chunks(meta_chunks["selected_indices"], shape_tail=(), dtype=np.dtype(np.int32)),
            "years": _concat_chunks(meta_chunks["years"], shape_tail=(), dtype=np.dtype(np.int32)),
            "splits": _concat_chunks(meta_chunks["splits"], shape_tail=(), dtype=np.dtype("<U8")),
            "crop_boxes": _concat_chunks(meta_chunks["crop_boxes"], shape_tail=(4,), dtype=np.dtype(np.int32)),
            "urban_mask_fraction": _concat_chunks(meta_chunks["urban_mask_fraction"], shape_tail=(), dtype=np.dtype(np.float32)),
            "city_mask": _concat_chunks(meta_chunks["city_mask"], shape_tail=(crop_h, crop_w), dtype=np.dtype(bool)),
            "scenario_names": np.asarray([_scenario_name(key) for key in scenario_keys], dtype="<U32"),
        }
        for key in extra_meta_keys:
            first = custom_payload[key]
            shape_tail = tuple(int(v) for v in first.shape[1:])
            shard_payload[key] = _concat_chunks(
                meta_chunks[key],
                shape_tail=shape_tail,
                dtype=np.asarray(first).dtype,
            )
        for key, chunks in ndvi_chunks.items():
            shard_payload[key] = _concat_chunks(chunks, shape_tail=(crop_h, crop_w), dtype=np.dtype(np.float32))
        for key, chunks in lst_chunks.items():
            shard_payload[key] = _concat_chunks(chunks, shape_tail=(crop_h, crop_w), dtype=np.dtype(np.float32))

        shard_path = shard_dir / f"rank{rank:05d}-of-{world_size:05d}.npz"
        _write_rank_shard(shard_path, shard_payload)
        logger.info(f"Rank {rank} wrote shard: {shard_path} ({int(shard_payload['selected_ids'].shape[0])} rows)")
        distributed_barrier(local_rank)

        if rank == 0:
            shard_paths = [shard_dir / f"rank{idx:05d}-of-{world_size:05d}.npz" for idx in range(world_size)]
            missing = [str(path) for path in shard_paths if not path.exists()]
            if missing:
                raise FileNotFoundError(f"Missing custom NDVI shard(s): {missing}")
            stem = _output_stem(input_npz, target_name, target_year)
            output_path = output_dir / f"{stem}.npz"
            stats_csv_path = output_dir / f"{stem}_stats.csv"
            summary_path = output_dir / "summary.json"
            summary = {
                "experiment": {
                    "group": str(hydra_cfg["experiment"]["group"]),
                    "name": str(hydra_cfg["experiment"]["name"]),
                    "model_name": str(hydra_cfg["experiment"]["model_name"]),
                    "pretrain_model": str(hydra_cfg["experiment"]["pretrain_model"]),
                    "checkpoint_path": checkpoint_path,
                    "checkpoint_source": checkpoint_source,
                },
                "data": {
                    "year": target_year,
                    "eump_root": str(hydra_cfg["data"]["eump_root"]),
                    "preprocessing_path": str(hydra_cfg["data"]["preprocessing_path"]),
                    "split_manifest_path": hydra_cfg["data"].get("split_manifest_path"),
                    "pipeline_year_range": list(pipeline_spec.year_range),
                    "center_ratio": float(pipeline_spec.center_ratio),
                    "total_cities": 0,
                },
                "custom_ndvi": {
                    "input_npz": str(input_npz),
                    "meta_csv": None if meta_csv_path is None else str(meta_csv_path),
                    "scenario_keys": list(scenario_keys),
                    "scenario_names": [_scenario_name(key) for key in scenario_keys],
                    "scale": "physical",
                    "schema": (
                        "counterfactual_buffer"
                        if set(DEFAULT_COUNTERFACTUAL_SCENARIOS).issubset(set(scenario_keys))
                        else "clustered_hotspot"
                    ),
                    "metadata_keys": [key for key in EXTRA_METADATA_KEYS if key in custom_payload],
                    "hotspot_quantile": None if "hotspot_quantile" not in custom_payload else float(custom_payload["hotspot_quantile"]),
                    "n_hotspots": None if "n_hotspots" not in custom_payload else int(custom_payload["n_hotspots"]),
                    "min_hotspot_pixels": None if "min_hotspot_pixels" not in custom_payload else int(custom_payload["min_hotspot_pixels"]),
                },
                "source": {
                    "name": source_name,
                    "channel_name": channel_name,
                    "channel_index": channel_index,
                    "mode": str(source_spec["mode"]),
                },
                "target": {
                    "name": target_name,
                    "kind": target_kind,
                    "source": target_source_name,
                    "scale": "physical",
                },
                "runtime": {
                    "distributed": world_size > 1,
                    "world_size": int(world_size),
                    "batch_size_per_rank": int(args.batch_size),
                    "num_workers_per_rank": int(getattr(train_dl, "num_workers", 0)),
                    "limit_cities": args.limit_cities,
                },
            }
            _merge_custom_ndvi_shards(
                shard_paths=shard_paths,
                output_path=output_path,
                stats_csv_path=stats_csv_path,
                summary_path=summary_path,
                summary=summary,
                scenario_keys=scenario_keys,
            )
            logger.info(f"Merged custom NDVI output: {output_path}")
        distributed_barrier(local_rank)
    finally:
        if initialized_distributed:
            destroy_distributed_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
