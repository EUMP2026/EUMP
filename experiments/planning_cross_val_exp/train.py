#!/usr/bin/env python3
"""Train and evaluate the retained ConvNeXt one-stage LST model.

The ConvNeXt encoder and LST head are trained jointly from scratch for the
cross-planning predicted-response comparison.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_DIR = EXPERIMENT_DIR / "config"
DEFAULT_CONFIG_NAME = "convnext_one_stage_lst_scratch.yaml"
DEFAULT_DATA_OVERRIDE = "data=eump_full/server_v2"
DEFAULT_GROUP = "planning_cross_val_exp_convnext_one_stage_scratch"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("lst", "all"), default="all")
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--config-name", default=DEFAULT_CONFIG_NAME)
    parser.add_argument("--nproc", type=int, default=4)
    parser.add_argument("--lst-gpus", default="0,1,2,3")
    parser.add_argument("--data-override", default=DEFAULT_DATA_OVERRIDE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("overrides", nargs="*", help="Extra Hydra overrides forwarded to the job.")
    return parser.parse_args(argv)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Experiment config must be a mapping: {path}")
    return payload


def _as_hydra_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_as_hydra_value(item) for item in value) + "]"
    text = str(value)
    if any(char in text for char in (" ", "#", "'", '"')):
        return repr(text)
    return text


def _override_list(payload: dict[str, Any]) -> list[str]:
    overrides = payload.get("overrides", {})
    if isinstance(overrides, list):
        return [str(item) for item in overrides]
    if not isinstance(overrides, dict):
        raise TypeError("config overrides must be a mapping or a list")
    return [f"{key}={_as_hydra_value(value)}" for key, value in overrides.items()]


def _load_lst_config(config_dir: Path, config_name: str) -> tuple[str, list[str]]:
    path = _resolve(config_dir) / config_name
    payload = _load_yaml(path)
    base_pipeline = payload.get("base_pipeline")
    if not isinstance(base_pipeline, str) or not base_pipeline.strip():
        raise ValueError(f"{path} must define a non-empty base_pipeline")
    return base_pipeline, _override_list(payload)


def _validate_cuda(args: argparse.Namespace) -> None:
    if args.dry_run:
        return
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --dry-run to inspect the command only.")
    gpu_count = len([value for value in args.lst_gpus.split(",") if value.strip()])
    if gpu_count != int(args.nproc):
        raise ValueError(
            f"LST GPU list contains {gpu_count} devices but --nproc={args.nproc}: {args.lst_gpus}"
        )


def _command_text(command: list[str], env: dict[str, str]) -> str:
    prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
    return "+ " + (prefix + " " if prefix else "") + " ".join(shlex.quote(part) for part in command)


def _run_job(*, command: list[str], env: dict[str, str], log_path: Path, dry_run: bool) -> None:
    print(_command_text(command, env))
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        subprocess.run(
            command, cwd=REPO_ROOT, env={**os.environ, **env},
            stdout=log_handle, stderr=subprocess.STDOUT, check=True,
        )


def _build_command(*, python_bin: str, nproc: int, base_pipeline: str,
                   overrides: list[str], data_override: str,
                   extra_overrides: list[str], resume: bool,
                   epochs: int | None) -> list[str]:
    command = [
        python_bin, "-m", "torch.distributed.run", f"--nproc_per_node={int(nproc)}",
        "--nnodes=1", "--node_rank=0", "--master_addr=127.0.0.1",
        "--master_port=29531", "main/train_eump.py", f"pipeline={base_pipeline}",
        data_override, *overrides,
    ]
    if resume:
        command.append("downstream.resume=true")
    if epochs is not None:
        command.append(f"downstream.epochs={int(epochs)}")
    command.extend(extra_overrides)
    return command


def _build_evaluation_command(*, python_bin: str, base_pipeline: str,
                              overrides: list[str], data_override: str,
                              extra_overrides: list[str]) -> list[str]:
    return [
        python_bin, "main/evaluate_eump.py", f"pipeline={base_pipeline}",
        data_override, *overrides, "evaluation.split=test",
        "evaluation.metric_scale=physical", "evaluation.checkpoint_source=best_model",
        "+evaluation.extended_metrics=true", "+evaluation.extended_spatial_metrics=true",
        "logging.progress_bar=false", *extra_overrides,
    ]


def _copy_labeled_metrics(*, model_name: str, output_name: str) -> None:
    source = REPO_ROOT / "model" / model_name / "evaluation/test_metrics.json"
    output_dir = EXPERIMENT_DIR / "outputs/metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(source.read_text(encoding="utf-8"))
    (output_dir / output_name).write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )


def _override_value(overrides: list[str], key: str, default: str) -> str:
    prefix = f"{key}="
    for override in reversed(overrides):
        if override.startswith(prefix):
            value = override[len(prefix):].strip().strip("'\"")
            if value:
                return value
    return default


def _model_name_from_overrides(overrides: list[str]) -> str:
    group = _override_value(overrides, "experiment.group", DEFAULT_GROUP)
    name = _override_value(overrides, "experiment.name", "lst")
    return f"{group}/{name}"


def _safe_filename(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    os.chdir(REPO_ROOT)
    _validate_cuda(args)
    lst_base, lst_overrides = _load_lst_config(args.config_dir, args.config_name)
    model_name = _model_name_from_overrides(lst_overrides)
    run_slug = _safe_filename(model_name)
    common_env = {"EUMP_TORCH_SHARING_STRATEGY": "file_system", "LOG_PROGRESS_BAR": "0"}
    logs_dir = EXPERIMENT_DIR / "outputs/logs"
    command = _build_command(
        python_bin=args.python_bin, nproc=args.nproc, base_pipeline=lst_base,
        overrides=lst_overrides, data_override=args.data_override,
        extra_overrides=args.overrides, resume=args.resume, epochs=args.epochs,
    )
    _run_job(
        command=command,
        env={**common_env, "CUDA_VISIBLE_DEVICES": args.lst_gpus},
        log_path=logs_dir / f"{run_slug}.log", dry_run=args.dry_run,
    )
    if not args.skip_eval:
        eval_command = _build_evaluation_command(
            python_bin=args.python_bin, base_pipeline=lst_base,
            overrides=lst_overrides, data_override=args.data_override,
            extra_overrides=args.overrides,
        )
        _run_job(
            command=eval_command,
            env={**common_env, "CUDA_VISIBLE_DEVICES": args.lst_gpus},
            log_path=logs_dir / f"{run_slug}_eval.log", dry_run=args.dry_run,
        )
        if not args.dry_run:
            _copy_labeled_metrics(
                model_name=model_name,
                output_name=f"{run_slug}_test_metrics.json",
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
