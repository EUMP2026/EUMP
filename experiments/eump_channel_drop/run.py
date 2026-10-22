#!/usr/bin/env python
"""Evaluate channel-drop-trained MAE v3 heads under input channel drop.

LST/precipitation use evaluation workers. PM2.5/Traffic score existing CV
checkpoints through launcher evaluation. ``--collect-only``
rebuilds tables from saved evaluation summaries and per-fold records.
"""
from __future__ import annotations

import argparse
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

CONFIG_DIR = REPO_ROOT / "configs"
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = EXPERIMENT_DIR / "config.yaml"
FIXED_TASKS = ("lst", "precip")
CV_TASKS = ("pm25", "traffic")
SUPPORTED_TASKS = (*FIXED_TASKS, *CV_TASKS)
# CV eval-only launcher wiring -- mirrors train.py so the eval re-scores the very
# per-fold checkpoints that train.py produced (no embedding cache, no retrain).
CV_DOWNSTREAM_MODE = {"pm25": "pm25-downstream", "traffic": "traffic-downstream"}
CV_TARGET_NAME = {"pm25": "PM25_mean", "traffic": "Traffic"}
CV_MODEL_NAME = {task: f"swin_downstream_drop_trained_{task}" for task in CV_TASKS}
SUMMARY_COLUMNS = (
    "task", "task_kind", "variant", "seed", "fold", "split", "scope",
    "rmse", "mae", "mse", "r2", "count",
)


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
    raw = ranges.get(task, ranges.get("default", [2017, 2020]))
    years = [int(year) for year in raw]
    if len(years) != 2:
        raise ValueError(f"year_ranges for task {task!r} must be [START, END]")
    return years


def _channel_drop_overrides(cfg: dict[str, Any], *, seed: int, enabled: bool) -> list[str]:
    if not enabled:
        # Input channel drop defaults to disabled in the task pipeline.
        return []
    drop = cfg.get("channel_drop", {})
    overrides = [
        "++pipeline.input_channel_drop.enabled=true",
        f"++pipeline.input_channel_drop.min_drop={int(drop.get('min_drop', 1))}",
        f"++pipeline.input_channel_drop.max_drop={int(drop.get('max_drop', 2))}",
        f"++pipeline.input_channel_drop.seed={int(seed)}",
    ]
    droppable = drop.get("droppable_channels")
    if droppable is not None:
        joined = ",".join(str(name) for name in droppable)
        overrides.append(f"++pipeline.input_channel_drop.droppable_channels=[{joined}]")
    return overrides


def _variant_runs(cfg: dict[str, Any]) -> list[tuple[str, int, bool]]:
    """Return (variant_label, seed, drop_enabled) runs to execute."""
    runs: list[tuple[str, int, bool]] = []
    if bool(cfg.get("include_baseline", True)):
        runs.append(("baseline", 0, False))
    base_seed = int(cfg.get("channel_drop", {}).get("base_seed", 12345))
    for repeat in range(int(cfg.get("repeats", 1))):
        runs.append((f"drop_seed{repeat}", base_seed + repeat * 101, True))
    return runs


def _compose(config_name: str, *, pipeline: str, overrides: list[str]) -> dict[str, Any]:
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name=config_name, overrides=[f"pipeline={pipeline}", *overrides])
    return OmegaConf.to_container(cfg, resolve=True)


def _eval_group(cfg: dict[str, Any]) -> str:
    """Group containing the channel-drop-trained downstream weights."""
    return str(cfg["train"]["trained_group"])


def compose_fixed_eval_config(
    *,
    cfg: dict[str, Any],
    task: str,
    split: str,
    seed: int,
    enabled: bool,
    overrides: list[str],
) -> dict[str, Any]:
    pipeline = f"{cfg['base_pipeline_suite']}/{task}"
    drop_overrides = _channel_drop_overrides(cfg, seed=seed, enabled=enabled)
    resolved = _compose("evaluate", pipeline=pipeline, overrides=[*overrides, *drop_overrides])
    year_start, year_end = _year_range(cfg, task)
    eval_group = _eval_group(cfg)
    # Load the channel-drop-trained downstream weights read-only.
    resolved.setdefault("experiment", {})["group"] = eval_group
    resolved["experiment"]["name"] = task
    resolved["experiment"]["model_name"] = f"{eval_group}/{task}"
    # Downstream heads and the shared frozen encoder live in separate groups.
    resolved.setdefault("downstream", {})["encoder_model_name"] = str(cfg["pretrain_model_name"])
    resolved.setdefault("pipeline", {})["year_range"] = [year_start, year_end]
    # Rebuild the head exactly as the deployed checkpoint was trained (the repo
    # pipeline config may have drifted from the saved weights).
    model_overrides = cfg.get("model_overrides", {}).get(task, {})
    if model_overrides:
        resolved.setdefault("model", {}).update(dict(model_overrides))
    evaluation = resolved.setdefault("evaluation", {})
    evaluation["split"] = split
    evaluation["checkpoint_source"] = str(cfg.get("evaluation", {}).get("checkpoint_source", "best_model"))
    evaluation["metric_scale"] = str(cfg.get("evaluation", {}).get("metric_scale", "physical"))
    evaluation["center_ratio"] = float(cfg.get("evaluation", {}).get("center_ratio", 1.0))
    # Never write into model/ — this experiment owns its outputs.
    evaluation["save_json"] = False
    return normalize_evaluation_config(resolved)


def _metric_rows_from_eval_summary(
    summary: dict[str, Any], *, task: str, variant: str, seed: int, split: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metrics_by_scope = summary.get("metrics", {})
    for scope, metrics in metrics_by_scope.items():
        if not isinstance(metrics, dict):
            continue
        rows.append(
            {
                "task": task,
                "task_kind": "downstream",
                "variant": variant,
                "seed": seed,
                "split": split,
                "scope": scope,
                "rmse": metrics.get("rmse"),
                "mae": metrics.get("mae"),
                "mse": metrics.get("mse"),
                "r2": metrics.get("r2"),
                "count": metrics.get("count"),
            }
        )
    return rows


def _printable(cmd: list[str], env: dict[str, str] | None = None) -> str:
    prefix = [] if env is None else [f"{k}={v}" for k, v in env.items()]
    return "+ " + " ".join(shlex.quote(p) for p in [*prefix, *cmd])


def run_command(cmd: list[str], *, env: dict[str, str], dry_run: bool, commands_log: Path) -> None:
    line = _printable(cmd, env=env)
    print(line)
    commands_log.parent.mkdir(parents=True, exist_ok=True)
    with commands_log.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    if dry_run:
        return
    merged_env = os.environ.copy()
    merged_env.update(env)
    subprocess.run(cmd, cwd=REPO_ROOT, env=merged_env, check=True)


def write_cv_config(
    *, cfg: dict[str, Any], task: str, seed: int, enabled: bool, run_dir: Path, variant: str
) -> Path:
    """Re-evaluate channel-drop-trained PM2.5/Traffic heads for each CV fold.

    The launcher loads the per-fold checkpoints produced by train.py and
    applies the requested corruption seed before the frozen encoder.
    """
    cv = cfg.get("cv", {})
    train = cfg.get("train", {})
    year_start, year_end = _year_range(cfg, task)
    group = _eval_group(cfg)
    drop_overrides = _channel_drop_overrides(cfg, seed=seed, enabled=enabled)

    variant_overrides = [
        f"downstream.encoder_model_name={cfg['pretrain_model_name']}",
        f"pipeline.year_range=[{year_start},{year_end}]",
        *drop_overrides,
    ]
    cv_payload = {
        "target_name": CV_TARGET_NAME[task],
        "folds": int(cv.get("folds", 5)),
        "fold_seed": int(cv.get("fold_seed", 42)),
        "experiment_group": group,
        "run_name": f"{task}_cv_{variant}",
        "output_prefix": f"{task}_cv",
        "target_pipeline": f"{cfg['base_pipeline_suite']}/{task}",
        "fold_manifest_dir": None,
        "embedding_cache_dir": None,
        "models": [
            {
                "name": CV_MODEL_NAME[task],
                "kind": "launcher",
                "mode": CV_DOWNSTREAM_MODE[task],
                "launch": str(train.get("cv_launch", cfg["base_pipeline_suite"])),
                "eval_only": True,
                "overrides": variant_overrides,
            }
        ],
    }
    path = run_dir / f"{task}_cv_{variant}.yaml"
    OmegaConf.save(config=OmegaConf.create({f"{task}_cv": cv_payload}), f=str(path))
    return path


def _cv_command(
    config_path: Path, *, cfg: dict[str, Any], task: str, variant: str, overrides: list[str]
) -> tuple[list[str], dict[str, str]]:
    runner = cfg.get("runner", {})
    gpu = os.environ.get("GPU", "0")
    nproc = os.environ.get("NPROC") or str(max(1, len([p for p in gpu.split(",") if p.strip()])))
    cmd = [
        sys.executable,
        "script/experiments/run_city_scalar_cv.py",
        "--config",
        str(config_path),
        "--models",
        CV_MODEL_NAME[task],
        # Forward data/Hydra overrides (e.g. data=eump_full/server_v2) so the CV
        # subprocess resolves the same dataset as the in-process lst/precip eval.
        *overrides,
    ]
    env = {
        "GPU": gpu,
        "CUDA_VISIBLE_DEVICES": gpu,
        "NPROC": str(nproc),
        "LOG_PROGRESS_BAR": os.environ.get("LOG_PROGRESS_BAR", "0"),
        "EUMP_TORCH_SHARING_STRATEGY": os.environ.get(
            "EUMP_TORCH_SHARING_STRATEGY", str(runner.get("torch_sharing_strategy", "file_system"))
        ),
    }
    return cmd, env


def _worker_eval_command(config_path: Path, *, cfg: dict[str, Any]) -> tuple[list[str], dict[str, str]]:
    runner = cfg.get("runner", {})
    gpu = os.environ.get("EVAL_GPU", "0")
    nproc = os.environ.get("EVAL_NPROC", "1")
    master_port = os.environ.get("EVAL_MASTER_PORT", str(runner.get("master_port", 29501)))
    cmd = [
        "torchrun",
        f"--nproc_per_node={nproc}",
        f"--nnodes={os.environ.get('NNODES', '1')}",
        f"--node_rank={os.environ.get('NODE_RANK', '0')}",
        f"--master_addr={os.environ.get('MASTER_ADDR', str(runner.get('master_addr', '127.0.0.1')))}",
        f"--master_port={master_port}",
        "experiments/eump_channel_drop/run.py",
        "--worker-config",
        str(config_path),
    ]
    env = {
        "CUDA_VISIBLE_DEVICES": gpu,
        "EUMP_TORCH_SHARING_STRATEGY": os.environ.get(
            "EUMP_TORCH_SHARING_STRATEGY", str(runner.get("torch_sharing_strategy", "file_system"))
        ),
    }
    return cmd, env


def _archive_cv_outputs(cv_log_dir: Path, run_dir: Path, *, task: str, variant: str) -> None:
    """Copy the raw CV products into this experiment's outputs.

    run_city_scalar_cv writes {prefix}_results.jsonl / _fold_metrics.csv /
    _summary.csv under log/<group>/<run_name>/. lst/precip already drop their
    per-run summaries into run_dir, so mirror the CV detail there too, prefixed
    by variant so the 5 drop seeds never overwrite each other.
    """
    import shutil

    archive_dir = run_dir / "cv_detail"
    archive_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("results.jsonl", "fold_metrics.csv", "summary.csv"):
        src = cv_log_dir / f"{task}_cv_{suffix}"
        if src.exists():
            shutil.copy2(src, archive_dir / f"{task}_{variant}_{suffix}")


def _read_cv_rows(results_path: Path, *, task: str, variant: str, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not results_path.exists():
        return rows
    for line in results_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        rows.append(
            {
                "task": task,
                "task_kind": "city_scalar_cv",
                "variant": variant,
                "seed": seed,
                "fold": payload.get("fold"),
                "split": payload.get("split"),
                "scope": "global",
                "rmse": payload.get("rmse"),
                "mae": payload.get("mae"),
                "mse": payload.get("mse"),
                "r2": payload.get("r2"),
                "count": payload.get("count"),
            }
        )
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    grouped: dict[tuple, list[dict[str, Any]]] = {}
    for row in rows:
        if str(row.get("variant", "")).startswith("baseline"):
            continue
        grouped.setdefault((row["task"], row["split"], row["scope"]), []).append(row)
    out: list[dict[str, Any]] = []
    for (task, split, scope), values in sorted(grouped.items(), key=lambda kv: tuple(str(p) for p in kv[0])):
        # Average CV folds within each corruption repeat before summarizing
        # repeats. Five folds and five seeds represent five corruption runs.
        by_variant: dict[str, list[dict[str, Any]]] = {}
        for value in values:
            by_variant.setdefault(str(value.get("variant", "")), []).append(value)
        ordered = [by_variant[name] for name in sorted(by_variant)]
        for metric in ("rmse", "mae", "r2"):
            nums = [
                statistics.fmean(float(v[metric]) for v in repeat if v.get(metric) is not None)
                for repeat in ordered
                if any(v.get(metric) is not None for v in repeat)
            ]
            base = {"task": task, "split": split, "scope": scope, "metric": metric}
            # Single-test result: the first drop seed only.
            out.append(
                {
                    **base,
                    "kind": "single",
                    "n_runs": 1 if nums else 0,
                    "mean": nums[0] if nums else "",
                    "std": "",
                }
            )
            # Multi-test aggregate: mean/std across all drop seeds.
            out.append(
                {
                    **base,
                    "kind": "aggregate",
                    "n_runs": len(nums),
                    "mean": statistics.fmean(nums) if nums else "",
                    "std": statistics.stdev(nums) if len(nums) > 1 else (0.0 if nums else ""),
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["task", "split", "scope", "metric", "kind", "n_runs", "mean", "std"]
        )
        writer.writeheader()
        writer.writerows(out)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EUMP MAE v3 inference-time channel-drop robustness suite.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=None, help="Override the evaluation output directory.")
    parser.add_argument("--tasks", default=None, help="Comma list subset of lst,precip,pm25,traffic. Default: all.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--collect-only", action="store_true", help="Rebuild tables from saved results without inference.")
    parser.add_argument("--worker-config", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("overrides", nargs="*", help="Extra Hydra overrides forwarded to composed configs.")
    return parser.parse_args(argv)


def worker_main(args: argparse.Namespace) -> int:
    payload = OmegaConf.to_container(OmegaConf.load(args.worker_config), resolve=True)
    summary = run_evaluation(payload, source="channel_drop")
    if summary is not None and int(os.environ.get("RANK", "0")) == 0:
        out_path = Path(str(payload["evaluation"]["worker_summary_path"]))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    return 0


def run_suite(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config)
    tasks = _comma_list(args.tasks) or _as_list(cfg.get("default_tasks", list(SUPPORTED_TASKS)))
    unknown = sorted(set(tasks) - set(SUPPORTED_TASKS))
    if unknown:
        raise ValueError(f"Unsupported task(s): {unknown}; supported={list(SUPPORTED_TASKS)}")
    splits = _as_list(cfg.get("eval_splits", ["test"]))
    overrides = [*_as_list(cfg.get("data_overrides")), *list(args.overrides)]
    dry_run = bool(args.dry_run or os.environ.get("DRY_RUN", "0") == "1")

    run_dir = args.output_dir or (
        EXPERIMENT_DIR
        / "outputs"
        / str(cfg.get("run_name", "eump_mae_v3_channel_drop"))
        / _eval_group(cfg)
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    commands_log = run_dir / "commands.log"
    if not args.collect_only:
        commands_log.write_text("", encoding="utf-8")
    rows: list[dict[str, Any]] = []

    for variant, seed, enabled in _variant_runs(cfg):
        for task in tasks:
            if task in FIXED_TASKS:
                for split in splits:
                    summary_path = run_dir / f"{task}_{variant}_{split}_summary.json"
                    if args.collect_only:
                        if not summary_path.is_file():
                            raise FileNotFoundError(summary_path)
                        summary = json.loads(summary_path.read_text(encoding="utf-8"))
                        rows.extend(_metric_rows_from_eval_summary(
                            summary, task=task, variant=variant, seed=seed, split=split
                        ))
                        continue
                    resolved = compose_fixed_eval_config(
                        cfg=cfg, task=task, split=split, seed=seed, enabled=enabled, overrides=overrides
                    )
                    resolved["evaluation"]["worker_summary_path"] = str(summary_path)
                    config_path = run_dir / f"{task}_{variant}_{split}.yaml"
                    OmegaConf.save(config=OmegaConf.create(resolved), f=str(config_path))
                    cmd, env = _worker_eval_command(config_path, cfg=cfg)
                    run_command(cmd, env=env, dry_run=dry_run, commands_log=commands_log)
                    if summary_path.exists():
                        summary = json.loads(summary_path.read_text(encoding="utf-8"))
                        rows.extend(
                            _metric_rows_from_eval_summary(
                                summary, task=task, variant=variant, seed=seed, split=split
                            )
                        )
            else:
                if args.collect_only:
                    results_path = run_dir / "cv_detail" / f"{task}_{variant}_results.jsonl"
                    if not results_path.is_file():
                        raise FileNotFoundError(results_path)
                    rows.extend(_read_cv_rows(results_path, task=task, variant=variant, seed=seed))
                    continue
                config_path = write_cv_config(
                    cfg=cfg, task=task, seed=seed, enabled=enabled, run_dir=run_dir, variant=variant
                )
                cmd, env = _cv_command(config_path, cfg=cfg, task=task, variant=variant, overrides=overrides)
                run_command(cmd, env=env, dry_run=dry_run, commands_log=commands_log)
                group = _eval_group(cfg)
                cv_log_dir = REPO_ROOT / "log" / group / f"{task}_cv_{variant}"
                results_path = cv_log_dir / f"{task}_cv_results.jsonl"
                rows.extend(_read_cv_rows(results_path, task=task, variant=variant, seed=seed))
                # Archive the raw CV products (results/fold-metrics/summary) into
                # this experiment's outputs so pm25/traffic detail lands beside
                # lst/precip instead of only living under log/<group>/...
                _archive_cv_outputs(cv_log_dir, run_dir, task=task, variant=variant)

    if dry_run:
        return 0
    _write_jsonl(run_dir / "results.jsonl", rows)
    _write_summary_csv(run_dir / "summary.csv", rows)
    print(f"Saved channel-drop results: {run_dir / 'results.jsonl'}")
    print(f"Saved channel-drop summary: {run_dir / 'summary.csv'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker_config is not None:
        return worker_main(args)
    return run_suite(args)


if __name__ == "__main__":
    raise SystemExit(main())




