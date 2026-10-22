#!/usr/bin/env python3
"""
Forward LST counterfactual inference from planning_data.pkl.

For each city, reload the original multi-channel raster input from the EUMP
dataset, replace only NDVI / POP channels using the joint planning result, and
evaluate four LST forward combinations:

    LST0  = f(NDVI0, POP0)
    LST1  = f(NDVI1, POP1)
    LST01 = f(NDVI0, POP1)
    LST10 = f(NDVI1, POP0)
"""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import importlib.util
import pickle
from pathlib import Path
import sys
from typing import Any

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_PLANNING_DATA = (
    EXPERIMENT_DIR / "outputs/joint_gradient_v2_5cities_current/planning_data.pkl"
)
DEFAULT_CONFIG_PATH = EXPERIMENT_DIR / "config.yaml"
DEFAULT_DATA_OVERRIDE = "data=eump_full/server_v2"
SOURCE_RESULT = "joint"


class RemapNumpyUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run LST forward counterfactual inference from planning_data.pkl."
    )
    parser.add_argument(
        "--planning-data",
        type=Path,
        default=DEFAULT_PLANNING_DATA,
        help=f"Input planning_data.pkl. Default: {DEFAULT_PLANNING_DATA}",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Fallback planning config if the package does not contain config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: parent directory of --planning-data.",
    )
    parser.add_argument("--output-file", default="forward_inference.pkl")
    parser.add_argument("--summary-file", default="forward_inference_summary.csv")
    parser.add_argument("--cities", default=None, help="Optional comma-separated city order.")
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    args, overrides = parser.parse_known_args(argv)
    return args, overrides


def _resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return RemapNumpyUnpickler(f).load()


def _load_yaml_config(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        raw = yaml.safe_load(f)
    return raw.get("planning_exp", raw)


def _load_planning_run_module() -> Any:
    module_path = EXPERIMENT_DIR / "run.py"
    spec = importlib.util.spec_from_file_location("planning_exp_run", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load planning run module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["planning_exp_run"] = module
    spec.loader.exec_module(module)
    return module


def _validate_package(package: Any, path: Path) -> dict[str, Any]:
    if not isinstance(package, dict):
        raise ValueError(f"{path} must contain a dict package.")
    if package.get("schema_version") != 1:
        raise ValueError(
            f"Unsupported schema_version={package.get('schema_version')!r}; expected 1."
        )
    city_data = package.get("city_data")
    if not isinstance(city_data, list):
        raise ValueError(f"{path} is missing list field 'city_data'.")
    return package


def _package_config(package: dict[str, Any], config_path: Path) -> dict[str, Any]:
    cfg = package.get("config")
    if isinstance(cfg, dict) and cfg:
        return deepcopy(cfg)
    return _load_yaml_config(config_path)


def _data_overrides(package: dict[str, Any], cli_overrides: list[str]) -> list[str]:
    if cli_overrides:
        return list(cli_overrides)
    run_args = package.get("run_args")
    if isinstance(run_args, dict):
        data_override = run_args.get("data_override")
        if isinstance(data_override, list) and data_override:
            return [str(item) for item in data_override]
        if isinstance(data_override, str) and data_override:
            return [data_override]
    return [DEFAULT_DATA_OVERRIDE]


def _parse_cities(value: str | None) -> list[int] | None:
    if value is None:
        return None
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def _select_city_items(
    city_data: list[dict[str, Any]],
    cities: list[int] | None,
) -> list[dict[str, Any]]:
    by_city: dict[int, dict[str, Any]] = {}
    for item in city_data:
        if not isinstance(item, dict) or "city_id" not in item:
            raise ValueError("Each city_data item must be a dict with city_id.")
        city_id = int(item["city_id"])
        if city_id in by_city:
            raise ValueError(f"Duplicate city_data entry for city {city_id}")
        by_city[city_id] = item

    if cities is None:
        return [item for item in city_data if isinstance(item, dict)]

    missing = [city_id for city_id in cities if city_id not in by_city]
    if missing:
        raise ValueError(f"Requested city IDs are missing from planning_data.pkl: {missing}")
    return [by_city[city_id] for city_id in cities]


def _array2d(data: dict[str, Any], key: str, *, city_id: int) -> np.ndarray:
    if key not in data:
        raise ValueError(f"City {city_id} {SOURCE_RESULT} result is missing field {key!r}.")
    arr = np.asarray(data[key]).squeeze()
    if arr.ndim != 2:
        raise ValueError(
            f"City {city_id} field {key!r} must be 2D after squeeze, got {arr.shape}."
        )
    return arr


def _scalar(data: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = data.get(key, default)
    arr = np.asarray(value)
    if arr.size == 0:
        return default
    return float(arr.reshape(-1)[0])


def _finite_masked_values(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid = mask.astype(bool) & np.isfinite(values)
    return np.asarray(values, dtype=np.float64)[valid]


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    selected = _finite_masked_values(values, mask)
    if selected.size == 0:
        return float("nan")
    return float(np.mean(selected))


def _physical_mean(lst_processed: np.ndarray, mask: np.ndarray, mean: float, std: float) -> float:
    return _masked_mean(np.asarray(lst_processed, dtype=np.float64) * std + mean, mask)


def _physical_delta_mean(a: np.ndarray, b: np.ndarray, mask: np.ndarray, std: float) -> float:
    return _masked_mean((np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)) * std, mask)


def _max_abs_diff(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    selected = _finite_masked_values(diff, mask)
    if selected.size == 0:
        return float("nan")
    return float(np.max(np.abs(selected)))


def _validate_channel_array(arr: np.ndarray, shape: tuple[int, int], *, city_id: int, key: str) -> None:
    if tuple(arr.shape) != tuple(shape):
        raise ValueError(
            f"City {city_id} field {key!r} has shape {arr.shape}, expected {shape}."
        )


def _set_channel(
    base_input: Any,
    *,
    ch_ndvi: int,
    ch_pop: int,
    ndvi: np.ndarray,
    pop: np.ndarray,
) -> Any:
    import torch

    x = base_input.clone()
    device = x.device
    dtype = x.dtype
    x[0, ch_ndvi] = torch.as_tensor(ndvi, dtype=dtype, device=device)
    x[0, ch_pop] = torch.as_tensor(pop, dtype=dtype, device=device)
    return x


def _predict_lst(lst_model: Any, raster_input: Any) -> np.ndarray:
    import torch

    output_size = tuple(int(v) for v in raster_input.shape[-2:])
    with torch.no_grad():
        embedding = lst_model.encoder.encode(raster_input, bool_masked_pos=None)
        pred = list(lst_model.raster_heads.values())[0](embedding, output_size)
    return pred.detach().cpu().numpy().squeeze()


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "city_id",
        "lst0_city_mean",
        "lst1_city_mean",
        "lst01_city_mean",
        "lst10_city_mean",
        "delta_lst1_vs_lst0_city",
        "delta_lst01_vs_lst0_city",
        "delta_lst10_vs_lst0_city",
        "interaction_city",
        "lst0_global_mean",
        "lst1_global_mean",
        "lst01_global_mean",
        "lst10_global_mean",
        "max_abs_diff_lst0_vs_planning_original",
        "max_abs_diff_lst1_vs_planning_adjusted",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    args, cli_overrides = parse_args(argv)
    import torch

    planning_data_path = _resolve_path(args.planning_data)
    output_dir = _resolve_path(args.output_dir) if args.output_dir else planning_data_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    package = _validate_package(_load_pickle(planning_data_path), planning_data_path)
    cfg = _package_config(package, _resolve_path(args.config_path))
    overrides = _data_overrides(package, cli_overrides)
    year = int(args.year if args.year is not None else package.get("year", cfg.get("year", 2020)))
    selected_items = _select_city_items(package["city_data"], _parse_cities(args.cities))
    selected_cities = [int(item["city_id"]) for item in selected_items]

    planning_run = _load_planning_run_module()
    device = planning_run._resolve_device(args.device)
    hydra_cfg_lst = planning_run._compose_hydra_config(
        cfg["lst_pipeline"], overrides, int(args.num_workers)
    )
    planning_run._apply_checkpoint_overrides(hydra_cfg_lst, cfg)
    random_seed = int(hydra_cfg_lst["experiment"]["random_seed"])
    planning_run.set_reproducible(random_seed)

    pipeline_spec_lst = planning_run.PipelineSpec.from_mapping(hydra_cfg_lst["pipeline"])
    planning_run.validate_eump_downstream_spec(pipeline_spec_lst)
    train_dl, val_dl, test_dl, batch_schema_lst = planning_run.build_eump_dataloaders(
        data_config=hydra_cfg_lst["data"],
        pipeline_spec=pipeline_spec_lst,
        logger=planning_run.logger,
        random_seed=random_seed,
        batch_size=1,
        local_rank=0,
        world_size=1,
        is_distributed=False,
        shuffle_train=False,
    )
    loaders = {"train": train_dl, "val": val_dl, "test": test_dl}
    preprocessing_spec = train_dl.dataset.preprocessing_spec
    raster_input_channel_names = tuple(
        str(n) for n in batch_schema_lst.raster_input_channel_names
    )

    ch_ndvi, _, _ = planning_run._resolve_source_channel(
        source_name=cfg["update_channels"]["lst_step"],
        preprocessing_spec=preprocessing_spec,
        raster_input_channel_names=raster_input_channel_names,
    )
    ch_pop, _, _ = planning_run._resolve_source_channel(
        source_name=cfg["update_channels"]["traffic_step"],
        preprocessing_spec=preprocessing_spec,
        raster_input_channel_names=raster_input_channel_names,
    )
    lst_target_name = str(hydra_cfg_lst["pipeline"]["targets"]["raster"][0]["source"])

    target_schema_lst = planning_run.get_single_downstream_target_schema(batch_schema_lst)
    lst_model = planning_run._build_model(
        hydra_cfg=hydra_cfg_lst,
        batch_schema=batch_schema_lst,
        target_schema=target_schema_lst,
        device=device,
        freeze_encoder=False,
    )
    for param in lst_model.parameters():
        param.requires_grad = False
    lst_model.eval()

    print(
        f"Forward inference: cities={selected_cities}, year={year}, "
        f"device={device}, overrides={overrides}"
    )

    results: list[dict[str, Any]] = []
    city_outputs: list[dict[str, Any]] = []

    for idx, item in enumerate(selected_items):
        city_id = int(item["city_id"])
        joint = item.get(SOURCE_RESULT)
        if not isinstance(joint, dict):
            raise ValueError(
                f"City {city_id} is missing city_data[*].{SOURCE_RESULT}; "
                "forward inference requires joint planning results."
            )
        print(f"[{idx + 1}/{len(selected_items)}] city {city_id}")

        entry = planning_run._find_city_entry(loaders, scene_id=city_id, year=year)
        if entry is None:
            raise ValueError(f"City {city_id} year {year} not found in current dataset.")
        split_name, sample_index = entry
        sample = loaders[split_name].dataset[sample_index]

        raster_inputs = torch.from_numpy(sample.raster_inputs).unsqueeze(0).float().to(device)
        city_mask = np.asarray(sample.city_mask).squeeze().astype(bool)
        raster_valid = np.asarray(sample.raster_valid_mask).astype(bool)
        valid_mask = raster_valid.all(axis=0).squeeze()
        lst_valid = (
            np.asarray(sample.raster_targets[lst_target_name][1])
            .astype(bool)
            .all(axis=0)
            .squeeze()
        )
        valid_mask = valid_mask & lst_valid
        metric_mask = city_mask & valid_mask
        if not bool(np.any(metric_mask)):
            raise ValueError(f"City {city_id} has an empty city valid metric mask.")
        if not bool(np.any(valid_mask)):
            raise ValueError(f"City {city_id} has an empty global valid mask.")

        shape = tuple(int(v) for v in raster_inputs.shape[-2:])
        ndvi0 = _array2d(joint, "original_ndvi", city_id=city_id)
        ndvi1 = _array2d(joint, "adjusted_ndvi", city_id=city_id)
        pop0 = _array2d(joint, "original_pop", city_id=city_id)
        pop1 = _array2d(joint, "adjusted_pop", city_id=city_id)
        planning_original_lst = _array2d(joint, "original_lst", city_id=city_id)
        planning_adjusted_lst = _array2d(joint, "adjusted_lst", city_id=city_id)
        for key, arr in (
            ("original_ndvi", ndvi0),
            ("adjusted_ndvi", ndvi1),
            ("original_pop", pop0),
            ("adjusted_pop", pop1),
            ("original_lst", planning_original_lst),
            ("adjusted_lst", planning_adjusted_lst),
        ):
            _validate_channel_array(arr, shape, city_id=city_id, key=key)

        lst0 = _predict_lst(
            lst_model,
            _set_channel(raster_inputs, ch_ndvi=ch_ndvi, ch_pop=ch_pop, ndvi=ndvi0, pop=pop0),
        )
        lst1 = _predict_lst(
            lst_model,
            _set_channel(raster_inputs, ch_ndvi=ch_ndvi, ch_pop=ch_pop, ndvi=ndvi1, pop=pop1),
        )
        lst01 = _predict_lst(
            lst_model,
            _set_channel(raster_inputs, ch_ndvi=ch_ndvi, ch_pop=ch_pop, ndvi=ndvi0, pop=pop1),
        )
        lst10 = _predict_lst(
            lst_model,
            _set_channel(raster_inputs, ch_ndvi=ch_ndvi, ch_pop=ch_pop, ndvi=ndvi1, pop=pop0),
        )

        lst_mean = _scalar(joint, "lst_mean", 0.0)
        lst_std = _scalar(joint, "lst_std", 1.0)
        interaction = lst1 - lst10 - lst01 + lst0
        result = {
            "city_id": city_id,
            "lst0_city_mean": _physical_mean(lst0, metric_mask, lst_mean, lst_std),
            "lst1_city_mean": _physical_mean(lst1, metric_mask, lst_mean, lst_std),
            "lst01_city_mean": _physical_mean(lst01, metric_mask, lst_mean, lst_std),
            "lst10_city_mean": _physical_mean(lst10, metric_mask, lst_mean, lst_std),
            "delta_lst1_vs_lst0_city": _physical_delta_mean(
                lst1, lst0, metric_mask, lst_std
            ),
            "delta_lst01_vs_lst0_city": _physical_delta_mean(
                lst01, lst0, metric_mask, lst_std
            ),
            "delta_lst10_vs_lst0_city": _physical_delta_mean(
                lst10, lst0, metric_mask, lst_std
            ),
            "interaction_city": _masked_mean(interaction * lst_std, metric_mask),
            "lst0_global_mean": _physical_mean(lst0, valid_mask, lst_mean, lst_std),
            "lst1_global_mean": _physical_mean(lst1, valid_mask, lst_mean, lst_std),
            "lst01_global_mean": _physical_mean(lst01, valid_mask, lst_mean, lst_std),
            "lst10_global_mean": _physical_mean(lst10, valid_mask, lst_mean, lst_std),
            "max_abs_diff_lst0_vs_planning_original": _max_abs_diff(
                lst0, planning_original_lst, valid_mask
            ),
            "max_abs_diff_lst1_vs_planning_adjusted": _max_abs_diff(
                lst1, planning_adjusted_lst, valid_mask
            ),
        }
        print(
            f"[city {city_id}] consistency: "
            f"LST0 diff={result['max_abs_diff_lst0_vs_planning_original']:.6g}, "
            f"LST1 diff={result['max_abs_diff_lst1_vs_planning_adjusted']:.6g}"
        )
        results.append(result)
        city_outputs.append({
            "city_id": city_id,
            "lst0": lst0,
            "lst1": lst1,
            "lst01": lst01,
            "lst10": lst10,
            "city_mask": city_mask,
            "ocean_mask": ~valid_mask,
            "lst_mean": lst_mean,
            "lst_std": lst_std,
        })

    output_package = {
        "schema_version": 1,
        "source_planning_data": str(planning_data_path),
        "source_result": SOURCE_RESULT,
        "year": year,
        "cities": selected_cities,
        "config": {
            "planning": cfg,
            "forward_inference": {
                "data_overrides": overrides,
                "device": str(device),
                "num_workers": int(args.num_workers),
            },
        },
        "results": results,
        "city_data": city_outputs,
    }

    output_path = output_dir / str(args.output_file)
    with output_path.open("wb") as f:
        pickle.dump(output_package, f)
    summary_path = output_dir / str(args.summary_file)
    _write_summary_csv(summary_path, results)
    print(f"Saved forward inference package: {output_path}")
    print(f"Saved forward inference summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
