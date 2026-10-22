#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import statistics
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from urban_embedding.training.evaluation import normalize_evaluation_config, run_evaluation
from urban_embedding.training.runner import normalize_runtime_config, run_training


CONFIG_DIR = REPO_ROOT / "configs"
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = EXPERIMENT_DIR / "config.yaml"

FIXED_TEST_TASKS = {"lst", "precip"}
CV_TASKS = {"traffic", "pm25"}
SUPPORTED_TASKS = FIXED_TEST_TASKS | CV_TASKS
SUPPORTED_METHODS = {"ours", "one_stage", "aef", "scalar_mlp"}

OURS_PIPELINES = {
    "lst": "eump_full/mae_v3/lst",
    "precip": "eump_full/mae_v3/precip",
}
AEF_PIPELINES = {
    "lst": "eump_full/aef_mlp/lst",
    "precip": "eump_full/aef_mlp/precip",
}
SCALAR_MLP_PIPELINES = {
    "lst": "eump_full/scalar_mlp/lst",
    "precip": "eump_full/scalar_mlp/precip",
}
ONE_STAGE_PIPELINES = {
    "lst": "eump_full/mae_v3_one_stage/lst",
    "precip": "eump_full/mae_v3_one_stage/precip",
}
SCALAR_MLP_METHODS = {"scalar_mlp"}
CV_TARGETS = {
    "traffic": {
        "target_name": "Traffic",
        "target_pipeline": "eump_full/mae_v3/traffic",
        "output_prefix": "traffic_cv",
        "aef_mode": "aef-mlp-traffic",
        "scalar_mlp_mode": "mlp-traffic",
    },
    "pm25": {
        "target_name": "PM25_mean",
        "target_pipeline": "eump_full/mae_v3/pm25",
        "output_prefix": "pm25_cv",
        "aef_mode": "aef-mlp-pm25",
        "scalar_mlp_mode": "mlp-pm25",
    },
}
CV_MODEL_BY_METHOD = {
    "ours": "swin_embed_head_128_d05",
    "one_stage": "one_stage_swin_l_unfrozen",
    "aef": "aef_mlp_128_d05",
    "scalar_mlp": "scalar_mlp_128_d05",
}
METHOD_BY_CV_MODEL = {value: key for key, value in CV_MODEL_BY_METHOD.items() if key != "one_stage"}

SUMMARY_COLUMNS = (
    "method",
    "task",
    "protocol",
    "split",
    "target",
    "scope",
    "year_start",
    "year_end",
    "fold",
    "fold_count",
    "count",
    "count_std",
    "mae",
    "mae_std",
    "mse",
    "mse_std",
    "rmse",
    "rmse_std",
    "r2",
    "r2_std",
    "experiment_group",
    "experiment_name",
    "checkpoint_path",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EUMP mae_v3 versus AEF comparison experiments.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=None, help="Override the run output directory.")
    parser.add_argument("--tasks", default=None, help="Comma-separated tasks: lst,precip,traffic,pm25.")
    parser.add_argument(
        "--methods",
        default=None,
        help="Comma-separated methods: ours,one_stage,aef,scalar_mlp.",
    )
    parser.add_argument("--stages", default=None, help="Comma-separated stages: train,eval,collect.")
    parser.add_argument("--continue-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--worker-stage", choices=("train", "eval"), default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-config", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("overrides", nargs="*", help="Hydra overrides forwarded to composed configs.")
    return parser.parse_args(argv)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _comma_list(value: str | None) -> list[str] | None:
    if value is None or not str(value).strip():
        return None
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _truthy(value: str | None) -> bool:
    return str(value or "").lower() in {"1", "true", "yes", "on"}


def _load_compare_config(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(payload, dict):
        raise TypeError(f"AEF compare config must be a mapping: {path}")
    cfg = payload.get("aef_compare", payload)
    if not isinstance(cfg, dict):
        raise TypeError(f"aef_compare section must be a mapping: {path}")
    return cfg


def _selected_values(defaults: list[Any], selected: str | None) -> list[str]:
    parsed = _comma_list(selected)
    return parsed if parsed is not None else [str(value) for value in defaults]


def _merge_overrides(default_overrides: list[Any], cli_overrides: list[str]) -> list[str]:
    cli_has_data = any(override.startswith("data=") or override.startswith("+data=") for override in cli_overrides)
    merged: list[str] = []
    for override in default_overrides:
        text = str(override)
        if cli_has_data and text.startswith("data="):
            continue
        merged.append(text)
    merged.extend(cli_overrides)
    return merged


def _year_range(compare_cfg: dict[str, Any], task: str) -> list[int]:
    ranges = compare_cfg.get("year_ranges", {})
    raw = ranges.get(task, ranges.get("default", [2017, 2020]))
    values = [int(value) for value in _as_list(raw)]
    if len(values) != 2:
        raise ValueError(f"aef_compare.year_ranges.{task} must contain [START, END]")
    if values[0] > values[1]:
        raise ValueError(f"Invalid year_range for task={task}: {values}")
    return values


def _year_override(compare_cfg: dict[str, Any], task: str) -> str:
    start, end = _year_range(compare_cfg, task)
    return f"pipeline.year_range=[{start},{end}]"


def _compose_config(config_name: str, *, pipeline: str, overrides: list[str]) -> dict[str, Any]:
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name=config_name, overrides=[f"pipeline={pipeline}", *overrides])
    payload = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(payload, dict):
        raise TypeError(f"Composed config must be a mapping for pipeline={pipeline}")
    return payload


def _write_yaml(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(payload), path)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def _config_path(run_dir: Path, method: str, task: str, label: str) -> Path:
    return run_dir / "configs" / method / f"{task}_{label}.yaml"


def _cv_config_path(run_dir: Path, task: str) -> Path:
    return run_dir / "configs" / "cv" / f"{task}_cv.yaml"


def _method_from_cv_model(compare_cfg: dict[str, Any], task: str, model: str) -> str:
    one_stage_model = _cv_model_name(compare_cfg, "one_stage", task)
    if model == one_stage_model:
        return "one_stage"
    return METHOD_BY_CV_MODEL[model]


def _method_group(compare_cfg: dict[str, Any], method: str) -> str:
    if method == "ours":
        return str(compare_cfg["ours_group"])
    if method == "one_stage":
        return str(compare_cfg["one_stage_group"])
    if method == "aef":
        return str(compare_cfg["aef_group"])
    if method == "scalar_mlp":
        return str(compare_cfg["scalar_mlp_group"])
    raise ValueError(f"Unsupported method: {method}")


def _fixed_pipeline(method: str, task: str) -> str:
    if method == "ours":
        return OURS_PIPELINES[task]
    if method == "one_stage":
        return ONE_STAGE_PIPELINES[task]
    if method == "aef":
        return AEF_PIPELINES[task]
    if method in SCALAR_MLP_METHODS:
        return SCALAR_MLP_PIPELINES[task]
    raise ValueError(f"Unsupported method: {method}")


def _apply_smoke_runtime(cfg: dict[str, Any], compare_cfg: dict[str, Any]) -> None:
    smoke = compare_cfg.get("smoke", {})
    downstream = cfg.setdefault("downstream", {})
    pipeline = cfg.setdefault("pipeline", {})
    if "epochs" in downstream:
        downstream["epochs"] = int(smoke.get("epochs", 1))
    downstream["batch_size"] = min(int(downstream.get("batch_size", smoke.get("batch_size", 8))), int(smoke.get("batch_size", 8)))
    if "selection" in downstream and isinstance(downstream["selection"], dict):
        downstream["selection"]["smoothing_window"] = 1
    pipeline["num_workers"] = int(smoke.get("num_workers", 0))
    pipeline["pin_memory"] = bool(smoke.get("pin_memory", False))
    pipeline["persistent_workers"] = bool(smoke.get("persistent_workers", False))
    pipeline["prefetch_factor"] = int(smoke.get("prefetch_factor", 2))
    pipeline["cache_scenes"] = False
    pipeline["preload_scenes"] = False
    cache_cfg = pipeline.get("city_scalar_cache")
    if isinstance(cache_cfg, dict):
        cache_cfg["num_workers"] = int(smoke.get("num_workers", 0))
        cache_cfg["batch_size"] = min(int(cache_cfg.get("batch_size", 1024)), 64)


def _apply_fixed_overrides(
    cfg: dict[str, Any],
    *,
    compare_cfg: dict[str, Any],
    method: str,
    task: str,
    split: str | None,
    smoke: bool,
) -> dict[str, Any]:
    resolved = deepcopy(cfg)
    year_start, year_end = _year_range(compare_cfg, task)
    resolved.setdefault("experiment", {})["group"] = _method_group(compare_cfg, method)
    resolved.setdefault("pipeline", {})["year_range"] = [year_start, year_end]
    resolved.setdefault("logging", {})["progress_bar"] = _truthy(os.environ.get("LOG_PROGRESS_BAR", "0"))
    if method == "ours":
        downstream = resolved.setdefault("downstream", {})
        downstream["encoder_model_name"] = str(compare_cfg["pretrain_model_name"])
        downstream["encoder_checkpoint_file"] = str(compare_cfg["encoder_checkpoint_file"])
    elif method == "one_stage":
        resolved.setdefault("experiment", {})["pretrain_model"] = str(
            compare_cfg.get("one_stage_pretrain_model", "swin_L")
        )
        resolved.setdefault("experiment", {})["scratch_backbone"] = bool(
            compare_cfg.get("one_stage_scratch_backbone", False)
        )
    elif method == "aef":
        feature_source = (
            resolved.setdefault("pipeline", {})
            .setdefault("city_scalar_cache", {})
            .setdefault("feature_source", {})
        )
        feature_source["year_range"] = [year_start, year_end]
    elif method == "scalar_mlp":
        pass
    if smoke:
        _apply_smoke_runtime(resolved, compare_cfg)
    if split is None:
        return normalize_runtime_config(resolved)
    evaluation = resolved.setdefault("evaluation", {})
    evaluation["split"] = split
    evaluation["checkpoint_source"] = str(compare_cfg.get("evaluation", {}).get("checkpoint_source", "best_model"))
    evaluation["metric_scale"] = str(compare_cfg.get("evaluation", {}).get("metric_scale", "physical"))
    evaluation["center_ratio"] = float(compare_cfg.get("evaluation", {}).get("center_ratio", 1.0))
    evaluation["save_json"] = True
    return normalize_evaluation_config(resolved)


def compose_and_write_fixed_config(
    *,
    compare_cfg: dict[str, Any],
    method: str,
    task: str,
    stage: str,
    split: str | None,
    overrides: list[str],
    run_dir: Path,
    smoke: bool,
) -> tuple[dict[str, Any], Path]:
    if task not in FIXED_TEST_TASKS:
        raise ValueError(f"Fixed-test config only supports {sorted(FIXED_TEST_TASKS)}, got {task!r}")
    config_name = "train" if stage == "train" else "evaluate"
    cfg = _compose_config(config_name, pipeline=_fixed_pipeline(method, task), overrides=overrides)
    resolved = _apply_fixed_overrides(
        cfg,
        compare_cfg=compare_cfg,
        method=method,
        task=task,
        split=split,
        smoke=smoke,
    )
    label = "train" if split is None else f"eval_{split}"
    path = _config_path(run_dir, method, task, label)
    _write_yaml(path, resolved)
    return resolved, path


def _cv_experiment_group(
    compare_cfg: dict[str, Any],
    task: str,
    *,
    one_stage_only: bool = False,
    aef_only: bool = False,
    scalar_mlp_only: bool = False,
) -> str:
    if one_stage_only and task == "traffic":
        return str(compare_cfg.get("one_stage_traffic_cv_group", "eump_aef_compare_traffic_cv_one_stage"))
    if one_stage_only and task == "pm25":
        return str(compare_cfg.get("one_stage_pm25_cv_group", "eump_aef_compare_pm25_cv_one_stage"))
    if aef_only and task == "traffic":
        return str(compare_cfg.get("aef_traffic_cv_group", "eump_aef_compare_traffic_cv_aef"))
    if aef_only and task == "pm25":
        return str(compare_cfg.get("aef_pm25_cv_group", "eump_aef_compare_pm25_cv_aef"))
    if scalar_mlp_only and task == "traffic":
        return str(compare_cfg.get("scalar_mlp_traffic_cv_group", "eump_aef_compare_traffic_cv_scalar_mlp"))
    if scalar_mlp_only and task == "pm25":
        return str(compare_cfg.get("scalar_mlp_pm25_cv_group", "eump_aef_compare_pm25_cv_scalar_mlp"))
    if task == "traffic":
        return str(compare_cfg["traffic_cv_group"])
    if task == "pm25":
        return str(compare_cfg["pm25_cv_group"])
    raise ValueError(f"Unsupported CV task: {task}")


def _cv_run_name(compare_cfg: dict[str, Any], task: str) -> str:
    cv_cfg = compare_cfg.get("cv", {})
    if task == "traffic":
        return str(cv_cfg.get("traffic_run_name", "traffic_cv_seed42"))
    if task == "pm25":
        return str(cv_cfg.get("pm25_run_name", "pm25_cv_seed42"))
    raise ValueError(f"Unsupported CV task: {task}")


def _cv_results_path(
    compare_cfg: dict[str, Any],
    task: str,
    *,
    one_stage_only: bool = False,
    aef_only: bool = False,
    scalar_mlp_only: bool = False,
) -> Path:
    output_prefix = CV_TARGETS[task]["output_prefix"]
    return (
        REPO_ROOT
        / "log"
        / _cv_experiment_group(
            compare_cfg,
            task,
            one_stage_only=one_stage_only,
            aef_only=aef_only,
            scalar_mlp_only=scalar_mlp_only,
        )
        / _cv_run_name(compare_cfg, task)
        / f"{output_prefix}_results.jsonl"
    )


def _cv_result_path_candidates(
    compare_cfg: dict[str, Any],
    task: str,
    *,
    method: str,
    selected_methods: list[str],
) -> list[Path]:
    default_path = _cv_results_path(compare_cfg, task, scalar_mlp_only=False)
    if method == "one_stage":
        one_stage_path = _cv_results_path(compare_cfg, task, one_stage_only=True)
        if selected_methods == ["one_stage"]:
            return [one_stage_path, default_path]
        return [default_path, one_stage_path]
    if method == "aef":
        aef_path = _cv_results_path(compare_cfg, task, aef_only=True)
        if selected_methods == ["aef"]:
            return [aef_path, default_path]
        return [aef_path, default_path]
    if method != "scalar_mlp":
        return [default_path]
    scalar_path = _cv_results_path(compare_cfg, task, scalar_mlp_only=True)
    if selected_methods == ["scalar_mlp"]:
        return [scalar_path, default_path]
    return [default_path, scalar_path]


def _cv_model_name(compare_cfg: dict[str, Any], method: str, task: str) -> str:
    if method == "one_stage":
        base = str(compare_cfg.get("one_stage_model_name", CV_MODEL_BY_METHOD["one_stage"]))
        return f"{base}_{task}"
    return CV_MODEL_BY_METHOD[method]


def _one_stage_cv_model_config(compare_cfg: dict[str, Any], task: str) -> dict[str, Any]:
    return {
        "name": _cv_model_name(compare_cfg, "one_stage", task),
        "kind": "launcher",
        "mode": task,
        "launch": str(compare_cfg["one_stage_launch"]),
        "overrides": [
            f"experiment.pretrain_model={compare_cfg.get('one_stage_pretrain_model', 'swin_L')}",
            f"++experiment.scratch_backbone={_hydra_scalar(bool(compare_cfg.get('one_stage_scratch_backbone', False)))}",
        ],
    }


def _hydra_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return str(value)
    return str(value)


def _cv_results_cover_methods(compare_cfg: dict[str, Any], task: str, methods: list[str]) -> bool:
    for method in methods:
        model_name = _cv_model_name(compare_cfg, method, task)
        found = False
        for path in _cv_result_path_candidates(
            compare_cfg,
            task,
            method=method,
            selected_methods=methods,
        ):
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    payload = json.loads(line)
                    if str(payload.get("model", "")) == model_name:
                        found = True
                        break
            if found:
                break
        if not found:
            return False
    return True


def _cv_models(compare_cfg: dict[str, Any], *, task: str, methods: list[str], smoke: bool) -> list[dict[str, Any]]:
    cv_cfg = compare_cfg.get("cv", {})
    smoke_cfg = compare_cfg.get("smoke", {})
    year_start, year_end = _year_range(compare_cfg, task)
    models: list[dict[str, Any]] = []
    if "ours" in methods:
        head_cfg = dict(cv_cfg.get("head", {}))
        if smoke:
            head_cfg["epochs"] = int(smoke_cfg.get("cv_epochs", 1))
            head_cfg["batch_size"] = int(smoke_cfg.get("cv_batch_size", 8))
        models.append(
            {
                "name": CV_MODEL_BY_METHOD["ours"],
                "kind": "swin_embedding_head",
                "scalar_hidden_dim": int(head_cfg.get("scalar_hidden_dim", 128)),
                "dropout": float(head_cfg.get("dropout", 0.5)),
                "lr": float(head_cfg.get("lr", 1.0e-4)),
                "weight_decay": float(head_cfg.get("weight_decay", 1.0e-2)),
                "batch_size": int(head_cfg.get("batch_size", 64)),
                "epochs": int(head_cfg.get("epochs", 80)),
                "loss": str(head_cfg.get("loss", "mae")),
                "selection_metric": str(head_cfg.get("selection_metric", "val_loss")),
            }
        )
    if "one_stage" in methods:
        models.append(_one_stage_cv_model_config(compare_cfg, task))
    if "aef" in methods:
        aef_cfg = dict(cv_cfg.get("aef", {}))
        hidden_dims = _as_list(aef_cfg.get("mlp_hidden_dims", [128]))
        hidden_text = "[" + ",".join(str(int(value)) for value in hidden_dims) + "]"
        models.append(
            {
                "name": CV_MODEL_BY_METHOD["aef"],
                "kind": "launcher",
                "mode": CV_TARGETS[task]["aef_mode"],
                "launch": str(compare_cfg["aef_launch"]),
                "overrides": [
                    f"model.mlp_hidden_dims={hidden_text}",
                    f"model.dropout={float(aef_cfg.get('dropout', 0.5))}",
                    f"downstream.feature_standardization={aef_cfg.get('feature_standardization', 'none')}",
                    f"pipeline.city_scalar_cache.feature_source.year_range=[{year_start},{year_end}]",
                ],
            }
        )
    if "scalar_mlp" in methods:
        scalar_cfg = dict(cv_cfg.get("scalar_mlp", {}))
        hidden_dims = _as_list(scalar_cfg.get("mlp_hidden_dims", [128]))
        hidden_text = "[" + ",".join(str(int(value)) for value in hidden_dims) + "]"
        models.append(
            {
                "name": CV_MODEL_BY_METHOD["scalar_mlp"],
                "kind": "launcher",
                "mode": CV_TARGETS[task]["scalar_mlp_mode"],
                "launch": str(compare_cfg["scalar_mlp_launch"]),
                "overrides": [
                    f"model.mlp_hidden_dims={hidden_text}",
                    f"model.dropout={float(scalar_cfg.get('dropout', 0.5))}",
                    f"downstream.feature_standardization={scalar_cfg.get('feature_standardization', 'none')}",
                ],
            }
        )
    return models


def _cv_target_pipeline(compare_cfg: dict[str, Any], task: str, methods: list[str]) -> str:
    if methods == ["one_stage"]:
        return f"{compare_cfg['one_stage_launch']}/{task}"
    return str(CV_TARGETS[task]["target_pipeline"])


def compose_and_write_cv_config(
    *,
    compare_cfg: dict[str, Any],
    task: str,
    methods: list[str],
    run_dir: Path,
    smoke: bool,
) -> tuple[dict[str, Any], Path]:
    if task not in CV_TASKS:
        raise ValueError(f"CV config only supports {sorted(CV_TASKS)}, got {task!r}")
    cv_cfg = compare_cfg.get("cv", {})
    smoke_cfg = compare_cfg.get("smoke", {})
    year_start, year_end = _year_range(compare_cfg, task)
    folds = int(cv_cfg.get("folds", 5))
    embedding_batch_size = int(cv_cfg.get("embedding_batch_size", 64))
    if smoke:
        folds = int(smoke_cfg.get("cv_folds", min(2, folds)))
        embedding_batch_size = int(smoke_cfg.get("cv_embedding_batch_size", 8))
    payload = {
        "city_scalar_cv": {
            "target_name": CV_TARGETS[task]["target_name"],
            "output_prefix": CV_TARGETS[task]["output_prefix"],
            "folds": folds,
            "fold_seed": int(cv_cfg.get("fold_seed", 42)),
            "experiment_group": _cv_experiment_group(
                compare_cfg,
                task,
                one_stage_only=methods == ["one_stage"],
                aef_only=methods == ["aef"],
                scalar_mlp_only=methods == ["scalar_mlp"],
            ),
            "run_name": _cv_run_name(compare_cfg, task),
            "fold_manifest_dir": None,
            "embedding_cache_dir": None,
            "target_pipeline": _cv_target_pipeline(compare_cfg, task, methods),
            "embedding": {
                "model_name": str(compare_cfg["pretrain_model_name"]),
                "checkpoint_file": str(compare_cfg["encoder_checkpoint_file"]),
                "checkpoint_source": "auto",
                "pipeline": f"{compare_cfg['base_pipeline_suite']}/pretrain",
                "years": [year_start, year_end],
                "batch_size": embedding_batch_size,
                "workers_per_gpu": int(cv_cfg.get("workers_per_gpu", 4)),
                "torch_threads": int(cv_cfg.get("torch_threads", 1)),
                "overwrite": False,
                "overrides": [],
            },
            "models": _cv_models(compare_cfg, task=task, methods=methods, smoke=smoke),
        }
    }
    path = _cv_config_path(run_dir, task)
    _write_yaml(path, payload)
    return payload, path


def _printable_command(cmd: list[str], env: dict[str, str] | None = None) -> str:
    prefix = [] if env is None else [f"{key}={value}" for key, value in env.items()]
    return "+ " + " ".join(shlex.quote(part) for part in [*prefix, *cmd])


def run_command(
    cmd: list[str],
    *,
    dry_run: bool,
    commands_log: Path,
    env: dict[str, str] | None = None,
) -> None:
    line = _printable_command(cmd, env=env)
    print(line)
    commands_log.parent.mkdir(parents=True, exist_ok=True)
    with commands_log.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    if dry_run:
        return
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    subprocess.run(cmd, cwd=REPO_ROOT, env=merged_env, check=True)


def _gpu_env(compare_cfg: dict[str, Any], *, stage: str) -> tuple[str, str, str]:
    runner_cfg = compare_cfg.get("runner", {})
    if stage == "eval" or stage == "cv":
        gpu = os.environ.get("EVAL_GPU", "0")
        nproc = os.environ.get("EVAL_NPROC", "1")
        master_port = os.environ.get("EVAL_MASTER_PORT", str(runner_cfg.get("eval_master_port", 29501)))
    else:
        gpu = os.environ.get("GPU", "0")
        nproc = os.environ.get("NPROC")
        if nproc is None:
            nproc = str(max(1, len([part for part in gpu.split(",") if part.strip()])))
        master_port = os.environ.get("MASTER_PORT", str(runner_cfg.get("master_port", 29500)))
    return gpu, str(nproc), str(master_port)


def _worker_command(stage: str, config_path: Path, *, compare_cfg: dict[str, Any]) -> tuple[list[str], dict[str, str]]:
    gpu, nproc, master_port = _gpu_env(compare_cfg, stage=stage)
    runner_cfg = compare_cfg.get("runner", {})
    cmd = [
        "torchrun",
        f"--nproc_per_node={nproc}",
        f"--nnodes={os.environ.get('NNODES', '1')}",
        f"--node_rank={os.environ.get('NODE_RANK', '0')}",
        f"--master_addr={os.environ.get('MASTER_ADDR', str(runner_cfg.get('master_addr', '127.0.0.1')))}",
        f"--master_port={master_port}",
        "experiments/eump_aef_compare/run.py",
        "--worker-stage",
        stage,
        "--worker-config",
        str(config_path),
    ]
    return cmd, {
        "CUDA_VISIBLE_DEVICES": gpu,
        "EUMP_TORCH_SHARING_STRATEGY": os.environ.get(
            "EUMP_TORCH_SHARING_STRATEGY",
            str(runner_cfg.get("torch_sharing_strategy", "file_system")),
        ),
    }


def _cv_command(
    config_path: Path,
    *,
    compare_cfg: dict[str, Any],
    task: str,
    overrides: list[str],
) -> tuple[list[str], dict[str, str]]:
    gpu, nproc, _ = _gpu_env(compare_cfg, stage="cv")
    cv_cfg = compare_cfg.get("cv", {})
    forced_overrides = [*overrides, _year_override(compare_cfg, task)]
    cmd = [
        "python",
        "script/experiments/run_city_scalar_cv.py",
        "--config",
        str(config_path),
        *forced_overrides,
    ]
    return cmd, {
        "GPU": gpu,
        "NPROC": nproc,
        "CUDA_VISIBLE_DEVICES": gpu,
        "LOG_PROGRESS_BAR": os.environ.get("LOG_PROGRESS_BAR", "0"),
        "EUMP_TORCH_SHARING_STRATEGY": os.environ.get(
            "EUMP_TORCH_SHARING_STRATEGY",
            str(compare_cfg.get("runner", {}).get("torch_sharing_strategy", "file_system")),
        ),
        "OMP_NUM_THREADS": str(cv_cfg.get("torch_threads", 1)),
        "MKL_NUM_THREADS": str(cv_cfg.get("torch_threads", 1)),
    }


def _fixed_experiment_name(method: str, task: str) -> str:
    if method == "ours":
        return task
    if method == "one_stage":
        return task
    if method == "aef":
        return f"aef_mlp_{task}"
    if method == "scalar_mlp":
        return f"mlp_{task}"
    raise ValueError(f"Unsupported method: {method}")


def _fixed_model_file(compare_cfg: dict[str, Any], method: str, task: str, cfg: dict[str, Any]) -> Path:
    group = _method_group(compare_cfg, method)
    name = str(cfg["experiment"]["name"])
    save_file = str(cfg["downstream"]["save_file"])
    return REPO_ROOT / "model" / group / name / save_file


def _fixed_metrics_file(compare_cfg: dict[str, Any], method: str, task: str, split: str) -> Path:
    group = _method_group(compare_cfg, method)
    name = _fixed_experiment_name(method, task)
    return REPO_ROOT / "model" / group / name / "evaluation" / f"{split}_metrics.json"


def _metric_value(metrics: dict[str, Any], key: str) -> Any:
    value = metrics.get(key)
    return None if value == "" else value


def _metric_rows_from_eval_summary(
    summary: dict[str, Any],
    *,
    method: str,
    task: str,
    protocol: str,
    year_range: list[int],
) -> list[dict[str, Any]]:
    experiment = summary.get("experiment", {})
    checkpoint = summary.get("checkpoint", {})
    rows: list[dict[str, Any]] = []
    for scope, metrics in summary.get("metrics", {}).items():
        row = {
            "method": method,
            "task": task,
            "protocol": protocol,
            "split": summary.get("split"),
            "target": summary.get("target"),
            "scope": scope,
            "year_start": int(year_range[0]),
            "year_end": int(year_range[1]),
            "fold": "",
            "fold_count": "",
            "count_std": "",
            "mae_std": "",
            "mse_std": "",
            "rmse_std": "",
            "r2_std": "",
            "experiment_group": experiment.get("group"),
            "experiment_name": experiment.get("name"),
            "checkpoint_path": checkpoint.get("path"),
        }
        for key in ("count", "mae", "mse", "rmse", "r2"):
            row[key] = _metric_value(metrics, key)
        rows.append(row)
    return rows


def _mean(values: list[float]) -> float | None:
    return None if not values else float(statistics.mean(values))


def _std(values: list[float]) -> float | None:
    return None if len(values) < 2 else float(statistics.stdev(values))


def _metric_rows_from_cv_results(
    *,
    compare_cfg: dict[str, Any],
    task: str,
    methods: list[str],
    eval_splits: list[str],
) -> list[dict[str, Any]]:
    allowed_splits = {str(split) for split in eval_splits}
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for method in methods:
        model_name = _cv_model_name(compare_cfg, method, task)
        for path in _cv_result_path_candidates(
            compare_cfg,
            task,
            method=method,
            selected_methods=methods,
        ):
            if not path.exists():
                continue
            found = False
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    payload = json.loads(line)
                    split = str(payload.get("split", ""))
                    model = str(payload.get("model", ""))
                    if split not in allowed_splits or model != model_name:
                        continue
                    payload = dict(payload)
                    payload["_source_experiment_group"] = path.parent.parent.name
                    target = str(payload.get("target", CV_TARGETS[task]["target_name"]))
                    grouped.setdefault((model, split, target), []).append(payload)
                    found = True
            if found:
                break

    year_start, year_end = _year_range(compare_cfg, task)
    rows: list[dict[str, Any]] = []
    for (model, split, target), payloads in sorted(grouped.items()):
        method = _method_from_cv_model(compare_cfg, task, model)
        folds = {
            str(payload.get("fold"))
            for payload in payloads
            if payload.get("fold") not in (None, "")
        }
        row = {
            "method": method,
            "task": task,
            "protocol": "city_kfold",
            "split": split,
            "target": target,
            "scope": "global",
            "year_start": year_start,
            "year_end": year_end,
            "fold": "kfold_mean",
            "fold_count": len(folds) if folds else len(payloads),
            "experiment_group": payloads[0].get("_source_experiment_group", ""),
            "experiment_name": f"{model}_kfold_mean",
            "checkpoint_path": "",
        }
        for key in ("count", "mae", "mse", "rmse", "r2"):
            values = [float(payload[key]) for payload in payloads if payload.get(key) is not None]
            row[key] = _mean(values)
            row[f"{key}_std"] = _std(values)
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...] = SUMMARY_COLUMNS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key, "") for key in columns})


def _headline_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    fallback: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        if str(row.get("split")) != "test":
            continue
        key = (str(row["method"]), str(row["task"]), str(row["protocol"]), str(row["split"]))
        if row.get("protocol") == "city_kfold" and row.get("fold") == "kfold_mean":
            selected[key] = row
            continue
        if row.get("protocol") == "fixed_test":
            if row.get("scope") == "city_mean":
                selected[key] = row
            elif key not in fallback:
                fallback[key] = row
    for key, row in fallback.items():
        selected.setdefault(key, row)
    return [selected[key] for key in sorted(selected)]


def collect_results(
    *,
    compare_cfg: dict[str, Any],
    methods: list[str],
    tasks: list[str],
    eval_splits: list[str],
    run_dir: Path,
) -> list[dict[str, Any]]:
    raw_rows: list[dict[str, Any]] = []
    for task in tasks:
        if task in FIXED_TEST_TASKS:
            for method in methods:
                for split in eval_splits:
                    metrics_path = _fixed_metrics_file(compare_cfg, method, task, split)
                    if not metrics_path.exists():
                        continue
                    summary = json.loads(metrics_path.read_text(encoding="utf-8"))
                    raw_rows.extend(
                        _metric_rows_from_eval_summary(
                            summary,
                            method=method,
                            task=task,
                            protocol="fixed_test",
                            year_range=_year_range(compare_cfg, task),
                        )
                    )
        elif task in CV_TASKS:
            raw_rows.extend(
                _metric_rows_from_cv_results(
                    compare_cfg=compare_cfg,
                    task=task,
                    methods=methods,
                    eval_splits=eval_splits,
                )
            )

    results_path = run_dir / "results.jsonl"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as handle:
        for row in raw_rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    _write_csv(run_dir / "summary_raw.csv", raw_rows)
    summary_rows = _headline_rows(raw_rows)
    _write_csv(run_dir / "summary.csv", summary_rows)
    return summary_rows


def worker_main(args: argparse.Namespace) -> int:
    if args.worker_config is None:
        raise ValueError("--worker-config is required in worker mode")
    payload = OmegaConf.to_container(OmegaConf.load(args.worker_config), resolve=True)
    if args.worker_stage == "train":
        run_training(payload, source="aef_compare")
        return 0
    if args.worker_stage == "eval":
        run_evaluation(payload, source="aef_compare")
        return 0
    raise ValueError(f"Unsupported worker stage: {args.worker_stage!r}")


def _validate_selection(values: list[str], supported: set[str], *, label: str) -> None:
    unknown = sorted(set(values) - supported)
    if unknown:
        raise ValueError(f"Unsupported {label}: {', '.join(unknown)}; supported={sorted(supported)}")


def run_compare(args: argparse.Namespace) -> int:
    compare_cfg = _load_compare_config(args.config)
    methods = _selected_values(_as_list(compare_cfg.get("default_methods", ["ours", "aef"])), args.methods)
    tasks = _selected_values(_as_list(compare_cfg.get("default_tasks", ["lst", "precip", "traffic", "pm25"])), args.tasks)
    eval_splits = _selected_values(_as_list(compare_cfg.get("eval_splits", ["test"])), None)
    stages = set(_selected_values(["train", "eval", "collect"], args.stages))
    _validate_selection(methods, SUPPORTED_METHODS, label="method(s)")
    _validate_selection(tasks, SUPPORTED_TASKS, label="task(s)")
    _validate_selection(list(stages), {"train", "eval", "collect"}, label="stage(s)")

    run_name = str(compare_cfg["run_name"])
    if args.smoke:
        run_name = f"{run_name}_{compare_cfg.get('smoke', {}).get('run_name_suffix', 'smoke')}"
    run_dir = args.output_dir or EXPERIMENT_DIR / "outputs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    dry_run = bool(args.dry_run or _truthy(os.environ.get("DRY_RUN")))
    overrides = _merge_overrides(_as_list(compare_cfg.get("data_overrides")), list(args.overrides))
    commands_log = run_dir / "commands.log"
    if not args.continue_run:
        for path in (commands_log, run_dir / "results.jsonl", run_dir / "summary_raw.csv", run_dir / "summary.csv"):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")

    runtime_config = {
        "config": str(args.config),
        "run_name": run_name,
        "methods": methods,
        "tasks": tasks,
        "eval_splits": eval_splits,
        "stages": sorted(stages),
        "smoke": bool(args.smoke),
        "overrides": overrides,
    }
    _write_yaml(run_dir / "runtime_config.yaml", runtime_config)
    _write_json(run_dir / "manifest.json", {"methods": methods, "tasks": tasks, "year_ranges": compare_cfg.get("year_ranges", {})})

    for task in tasks:
        if task in FIXED_TEST_TASKS:
            for method in methods:
                if "train" in stages:
                    train_cfg, train_path = compose_and_write_fixed_config(
                        compare_cfg=compare_cfg,
                        method=method,
                        task=task,
                        stage="train",
                        split=None,
                        overrides=overrides,
                        run_dir=run_dir,
                        smoke=args.smoke,
                    )
                    model_file = _fixed_model_file(compare_cfg, method, task, train_cfg)
                    if args.skip_existing and model_file.exists():
                        print(f"Skip existing train method={method} task={task}: {model_file}")
                    else:
                        cmd, env = _worker_command("train", train_path, compare_cfg=compare_cfg)
                        run_command(cmd, dry_run=dry_run, commands_log=commands_log, env=env)

                if "eval" in stages:
                    for split in eval_splits:
                        _, eval_path = compose_and_write_fixed_config(
                            compare_cfg=compare_cfg,
                            method=method,
                            task=task,
                            stage="eval",
                            split=split,
                            overrides=overrides,
                            run_dir=run_dir,
                            smoke=args.smoke,
                        )
                        metrics_file = _fixed_metrics_file(compare_cfg, method, task, split)
                        if args.skip_existing and metrics_file.exists():
                            print(f"Skip existing eval method={method} task={task} split={split}: {metrics_file}")
                        else:
                            cmd, env = _worker_command("eval", eval_path, compare_cfg=compare_cfg)
                            run_command(cmd, dry_run=dry_run, commands_log=commands_log, env=env)
            continue

        if task in CV_TASKS and "train" in stages:
            _, cv_path = compose_and_write_cv_config(
                compare_cfg=compare_cfg,
                task=task,
                methods=methods,
                run_dir=run_dir,
                smoke=args.smoke,
            )
            if args.skip_existing and _cv_results_cover_methods(compare_cfg, task, methods):
                print(f"Skip existing CV task={task}: selected model rows already exist")
            else:
                cmd, env = _cv_command(cv_path, compare_cfg=compare_cfg, task=task, overrides=overrides)
                run_command(cmd, dry_run=dry_run, commands_log=commands_log, env=env)

    if "collect" in stages and not dry_run:
        rows = collect_results(
            compare_cfg=compare_cfg,
            methods=methods,
            tasks=tasks,
            eval_splits=eval_splits,
            run_dir=run_dir,
        )
        print(f"Collected {len(rows)} headline row(s): {run_dir / 'summary.csv'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    os.chdir(REPO_ROOT)
    if args.worker_stage is not None:
        return worker_main(args)
    return run_compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
