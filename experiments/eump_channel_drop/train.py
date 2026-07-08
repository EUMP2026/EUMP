#!/usr/bin/env python
"""EUMP MAE v3 channel-drop AUGMENTED TRAINING.

Re-trains the 4 downstream tasks (lst / precip / pm25 / traffic) on top of the
SAME frozen ``eump_full_mae_v3`` pretrain encoder, but with
``pipeline.input_channel_drop`` enabled during head training so the head learns
to tolerate 1-2 missing encoder input channels. Hyperparameters are copied
verbatim from mae_v3 -- the ONLY difference vs the clean mainline is the
channel-drop augmentation.

Outputs land in a dedicated experiment group
(``train.trained_group`` in config.yaml) so the clean mae_v3 artifacts are
never overwritten. After training, point ``run.py`` at that group via
``trained_group`` (config) to run the drop robustness eval on these models.

Mechanics
---------
- ``lst`` / ``precip``: a single end-to-end head training via
  ``run_eump_full_train.sh <task>`` (frozen encoder, mae_v3 pipeline).
- ``pm25`` / ``traffic``: end-to-end per CV fold via
  ``run_city_scalar_cv.py`` with a launcher-only model list. A launcher-only
  CV needs no pooled-embedding cache (``needs_embedding=False``), so every fold
  trains the frozen head directly -- the same training style as lst/precip.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omegaconf import OmegaConf

EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = EXPERIMENT_DIR / "config.yaml"
FIXED_TASKS = ("lst", "precip")
CV_TASKS = ("pm25", "traffic")
SUPPORTED_TASKS = (*FIXED_TASKS, *CV_TASKS)
# Mode that maps to the single frozen swin_downstream job (NOT the *_cv recursion).
DOWNSTREAM_MODE = {"pm25": "pm25-downstream", "traffic": "traffic-downstream"}
CV_TARGET_NAME = {"pm25": "PM25_mean", "traffic": "Traffic"}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return list(value) if isinstance(value, list) else [value]


def _comma_list(value: str | None) -> list[str] | None:
    if value is None or not str(value).strip():
        return None
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _load_config(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(payload, dict) or "channel_drop" not in payload:
        raise TypeError(f"channel_drop config must contain a 'channel_drop' section: {path}")
    return payload["channel_drop"]


def _year_range(cfg: dict[str, Any], task: str) -> list[int]:
    ranges = cfg.get("year_ranges", {})
    raw = ranges.get(task, ranges.get("default", [2003, 2020]))
    years = [int(year) for year in raw]
    if len(years) != 2:
        raise ValueError(f"year_ranges for task {task!r} must be [START, END]")
    return years


def _drop_train_overrides(cfg: dict[str, Any]) -> list[str]:
    """Pipeline overrides enabling channel-drop augmentation during training."""
    drop = cfg.get("channel_drop", {})
    train = cfg.get("train", {})
    overrides = [
        "++pipeline.input_channel_drop.enabled=true",
        f"++pipeline.input_channel_drop.min_drop={int(drop.get('min_drop', 1))}",
        f"++pipeline.input_channel_drop.max_drop={int(drop.get('max_drop', 2))}",
        f"++pipeline.input_channel_drop.seed={int(train.get('train_seed', 12345))}",
    ]
    droppable = drop.get("droppable_channels")
    if droppable is not None:
        joined = ",".join(str(name) for name in droppable)
        overrides.append(f"++pipeline.input_channel_drop.droppable_channels=[{joined}]")
    return overrides


def _printable(cmd: list[str], env: dict[str, str] | None = None) -> str:
    prefix = [] if env is None else [f"{k}={v}" for k, v in env.items()]
    return "+ " + " ".join(shlex.quote(p) for p in [*prefix, *cmd])


def run_command(cmd: list[str], *, env: dict[str, str], dry_run: bool, commands_log: Path) -> None:
    line = _printable(cmd, env=env)
    print(line)
    with commands_log.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    if dry_run:
        return
    merged_env = dict(os.environ)
    merged_env.update(env)
    subprocess.run(cmd, cwd=REPO_ROOT, env=merged_env, check=True)


def _common_train_overrides(cfg: dict[str, Any], task: str) -> list[str]:
    """Overrides shared by every drop-augmented training command."""
    train = cfg.get("train", {})
    trained_group = str(train["trained_group"])
    encoder_model_name = str(train["encoder_model_name"])
    year_start, year_end = _year_range(cfg, task)
    return [
        *_as_list(cfg.get("data_overrides")),
        f"experiment.group={trained_group}",
        # Frozen encoder must still resolve to the clean mae_v3 pretrain, not the
        # new group (downstream profile uses ${experiment.group}/pretrain).
        f"downstream.encoder_model_name={encoder_model_name}",
        f"pipeline.year_range=[{year_start},{year_end}]",
        *_drop_train_overrides(cfg),
    ]


def write_cv_train_config(*, cfg: dict[str, Any], task: str, run_dir: Path) -> Path:
    """A launcher-only CV config: every fold trains the frozen head end-to-end.

    Launcher-only models make run_city_scalar_cv set needs_embedding=False, so no
    pooled-embedding cache is built -- each fold runs run_eump_full_train.sh.
    """
    train = cfg.get("train", {})
    trained_group = str(train["trained_group"])
    cv = cfg.get("cv", {})
    year_start, year_end = _year_range(cfg, task)
    mode = DOWNSTREAM_MODE[task]
    launch = str(train.get("cv_launch", "eump_full/mae_v3"))
    # variant.overrides are forwarded to BOTH the per-fold train and eval, so the
    # fold model is trained AND scored with channel-drop active. (experiment.group
    # is already injected by run_launcher_variant, so it is omitted here.)
    variant_overrides = [
        f"downstream.encoder_model_name={train['encoder_model_name']}",
        f"pipeline.year_range=[{year_start},{year_end}]",
        *_drop_train_overrides(cfg),
    ]
    cv_payload: dict[str, Any] = {
        "target_name": CV_TARGET_NAME[task],
        "folds": int(cv.get("folds", 5)),
        "fold_seed": int(cv.get("fold_seed", 42)),
        "experiment_group": trained_group,
        "run_name": f"{task}_cv_drop_trained",
        "output_prefix": f"{task}_cv",
        "target_pipeline": f"{cfg['base_pipeline_suite']}/{task}",
        "fold_manifest_dir": None,
        "embedding_cache_dir": None,
        "models": [
            {
                "name": f"swin_downstream_drop_trained_{task}",
                "kind": "launcher",
                "mode": mode,
                "launch": launch,
                "overrides": variant_overrides,
            }
        ],
    }
    path = run_dir / f"{task}_cv_drop_trained.yaml"
    OmegaConf.save(config=OmegaConf.create({f"{task}_cv": cv_payload}), f=str(path))
    return path


def _gpu_nproc() -> tuple[str, str]:
    gpu = os.environ.get("GPU", "0")
    nproc = os.environ.get("NPROC") or str(max(1, len([p for p in gpu.split(",") if p.strip()])))
    return gpu, nproc


def train_fixed_task(cfg: dict[str, Any], task: str, *, dry_run: bool, commands_log: Path) -> None:
    """lst / precip: a single end-to-end frozen-head training."""
    overrides = _common_train_overrides(cfg, task)
    cmd = ["bash", "script/training/run_eump_full_train.sh", task, *overrides]
    run_command(cmd, env={}, dry_run=dry_run, commands_log=commands_log)


def train_cv_task(cfg: dict[str, Any], task: str, *, dry_run: bool, commands_log: Path) -> None:
    """pm25 / traffic: end-to-end per CV fold via launcher-only CV (no embedding)."""
    run_dir = EXPERIMENT_DIR / "outputs" / str(cfg.get("run_name", "eump_mae_v3_channel_drop")) / "train"
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = write_cv_train_config(cfg=cfg, task=task, run_dir=run_dir)
    gpu, nproc = _gpu_nproc()
    model_name = f"swin_downstream_drop_trained_{task}"
    cmd = [
        sys.executable,
        "script/experiments/run_city_scalar_cv.py",
        "--config",
        str(config_path),
        "--models",
        model_name,
        *_as_list(cfg.get("data_overrides")),
    ]
    runner = cfg.get("runner", {})
    env = {
        "GPU": gpu,
        "CUDA_VISIBLE_DEVICES": gpu,
        "NPROC": nproc,
        "LOG_PROGRESS_BAR": os.environ.get("LOG_PROGRESS_BAR", "0"),
        "EUMP_TORCH_SHARING_STRATEGY": os.environ.get(
            "EUMP_TORCH_SHARING_STRATEGY", str(runner.get("torch_sharing_strategy", "file_system"))
        ),
    }
    run_command(cmd, env=env, dry_run=dry_run, commands_log=commands_log)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EUMP MAE v3 channel-drop augmented downstream training."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--tasks", default=None, help="Comma list subset of lst,precip,pm25,traffic. Default: all."
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = _load_config(args.config)
    if "train" not in cfg:
        raise KeyError("config.yaml channel_drop section is missing the 'train' block.")
    tasks = _comma_list(args.tasks) or _as_list(cfg.get("default_tasks", list(SUPPORTED_TASKS)))
    unknown = sorted(set(tasks) - set(SUPPORTED_TASKS))
    if unknown:
        raise ValueError(f"Unsupported task(s): {unknown}; supported={list(SUPPORTED_TASKS)}")
    dry_run = bool(args.dry_run or os.environ.get("DRY_RUN", "0") == "1")

    run_dir = EXPERIMENT_DIR / "outputs" / str(cfg.get("run_name", "eump_mae_v3_channel_drop"))
    run_dir.mkdir(parents=True, exist_ok=True)
    commands_log = run_dir / "train_commands.log"
    commands_log.write_text("", encoding="utf-8")

    print(f"Channel-drop augmented training -> group={cfg['train']['trained_group']}")
    print(f"Tasks: {tasks}")
    if any(task in CV_TASKS for task in tasks):
        print(
            "NOTE: pm25/traffic train end-to-end per CV fold (no embedding cache); "
            "this is much slower than pooled-embedding CV but is required for "
            "per-epoch channel-drop augmentation."
        )
    for task in tasks:
        if task in FIXED_TASKS:
            train_fixed_task(cfg, task, dry_run=dry_run, commands_log=commands_log)
        else:
            train_cv_task(cfg, task, dry_run=dry_run, commands_log=commands_log)

    print(f"Training commands logged to: {commands_log}")
    print(
        "Next: run drop eval against these models with\n"
        "  python experiments/eump_channel_drop/run.py "
        "(set channel_drop.trained_group in config.yaml first)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
