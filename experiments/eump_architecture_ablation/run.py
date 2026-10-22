#!/usr/bin/env python
"""Train, evaluate, and collect Swin-UNet, ConvNeXt-UNet, and MoCo.

Training/evaluation use the shared launchers. Collection reads existing task
and CV metrics and verifies the ConvNeXt downstream encoder parameters.
"""
from __future__ import annotations

import argparse
import csv
from functools import lru_cache
import os
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
EXPERIMENT_DIR = Path(__file__).resolve().parent


VARIANTS: dict[str, dict[str, object]] = {
    "swin_unet": {
        "family": "architecture",
        "launch": "eump_full/architecture_v1",
        "train_modes": ["swin-unet"],
        "eval_modes": ["swin-unet"],
        "experiment_group": "eump_full_architecture_v1_swin_unet",
        "checkpoint_file": "eump_reconstruction_best_checkpoint.pt",
        "embedding_pipeline": "eump_full/architecture_v1/swin_unet/pretrain",
    },
    "convnext_unet": {
        "family": "architecture",
        "launch": "eump_full/architecture_v1",
        "train_modes": ["convnext-unet"],
        "eval_modes": ["convnext-unet"],
        "experiment_group": "eump_full_architecture_v1_convnext_unet",
        "checkpoint_file": "eump_reconstruction_best_checkpoint.pt",
        "embedding_pipeline": "eump_full/architecture_v1/convnext_unet/pretrain",
    },
    "contrastive_v2": {
        "family": "pretrain_objective",
        "launch": "eump_full/contrastive_v2",
        "train_modes": ["all"],
        "eval_modes": ["all"],
        "experiment_group": "eump_full_contrastive_v2",
        "checkpoint_file": "eump_contrastive_best_checkpoint.pt",
        "embedding_pipeline": "eump_full/contrastive_v2/pretrain",
        "result_variant": "moco",
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EUMP architecture / pretrain-objective ablation")
    parser.add_argument(
        "--variants",
        default=",".join(VARIANTS.keys()),
        help="Comma-separated variant names. Default runs all variants.",
    )
    parser.add_argument("--stages", default="train,eval,collect", help="Comma-separated stages: train,eval,collect.")
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_DIR / "outputs")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra overrides forwarded to both train and eval invocations.",
    )
    return parser.parse_args(argv)


def _comma_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


@lru_cache(maxsize=None)
def checkpoint_encoder_kind(path: Path) -> str:
    """Identify the stored encoder from parameter names without GPU allocation."""
    import torch

    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    state = payload.get("model_state_dict", payload.get("state_dict", payload))
    keys = tuple(str(key).removeprefix("module.") for key in state)
    if any(key.startswith(("encoder.stages.", "backbone.stages.")) for key in keys):
        return "convnext"
    if any("swin.swinv2." in key for key in keys):
        return "swin"
    return "unknown"


def collect_variant_results(names: list[str], output_dir: Path) -> None:
    from experiments.eump_ablation_suite.run import (
        DEFAULT_CONFIG_PATH, METRIC_COLUMNS, _load_suite_config, build_variants,
        collect_results,
    )

    cfg = _load_suite_config(DEFAULT_CONFIG_PATH)
    variants = build_variants(
        cfg, selected_families=["architecture"],
        selected_variants=[str(VARIANTS[name].get("result_variant", name)) for name in names],
    )
    rows = collect_results(suite_cfg=cfg, variants=variants, tasks=["lst", "precip", "traffic", "pm25"],
                           eval_splits=["val", "test"], run_dir=output_dir)
    comparable = [row for row in rows if row["split"] == "test" and
                  row["scope"] == ("city_mean" if row["task"] == "lst" else "global")]
    with (output_dir / "comparable_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(comparable)
    print(f"Collected {len(rows)} metric rows and {len(comparable)} comparable rows: {output_dir}")


def _run(cmd: list[str], *, dry_run: bool, env: dict[str, str] | None = None) -> None:
    assignments = [f"{key}={value}" for key, value in (env or {}).items()]
    shown = " ".join(shlex.quote(x) for x in [*assignments, *cmd])
    print(f"+ {shown}", flush=True)
    if dry_run:
        return
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    subprocess.run(cmd, check=True, env=merged_env)


def run_variant(
    name: str,
    spec: dict[str, object],
    *,
    stages: list[str],
    overrides: list[str],
    dry_run: bool,
) -> None:
    launch = str(spec["launch"])
    train_modes = list(spec["train_modes"])
    eval_modes = list(spec["eval_modes"])
    passthrough = list(overrides)

    if "train" in stages:
        for mode in train_modes:
            cmd = [
                "bash",
                "script/training/run_eump_full_train.sh",
                mode,
                f"launch={launch}",
            ] + passthrough
            _run(cmd, dry_run=dry_run)

    if "eval" in stages:
        for mode in eval_modes:
            cmd = [
                "bash",
                "script/training/run_eump_full_eval.sh",
                mode,
                f"launch={launch}",
            ] + passthrough
            _run(cmd, dry_run=dry_run)

    if "train" in stages:
        _run_cv(name, spec, overrides=overrides, dry_run=dry_run)


def _run_cv(
    name: str,
    spec: dict[str, object],
    *,
    overrides: list[str],
    dry_run: bool,
) -> None:
    experiment_group = str(spec["experiment_group"])
    checkpoint_file = str(spec["checkpoint_file"])
    embedding_pipeline = str(spec["embedding_pipeline"])
    encoder_model_name = f"{experiment_group}/pretrain"

    data_config = "eump_full/server_v2"
    for override in overrides:
        if override.startswith("data="):
            data_config = override.split("=", 1)[1]

    cv_tasks = [
        {
            "script": "script/experiments/run_current_traffic_cv.sh",
            "task": "traffic_cv",
        },
        {
            "script": "script/experiments/run_current_pm25_cv.sh",
            "task": "pm25_cv",
        },
    ]

    for cv in cv_tasks:
        print(f"\n--- {name}: {cv['task']} ---", flush=True)
        env = {
            "ENCODER_MODEL_NAME": encoder_model_name,
            "CHECKPOINT_FILE": checkpoint_file,
            "EMBEDDING_PIPELINE": embedding_pipeline,
            "EXPERIMENT_GROUP": experiment_group,
            "RUN_NAME": f"{cv['task']}_mae_v3_seed42",
        }
        cmd = ["bash", cv["script"], "--data", data_config, "--models", "swin"]
        cmd.extend(override for override in overrides if not override.startswith("data="))
        if dry_run:
            cmd.append("--dry-run")
        _run(cmd, dry_run=dry_run, env=env)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    variants = _comma_list(args.variants)
    stages = _comma_list(args.stages)
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        print(f"Unknown variants: {unknown}; valid: {list(VARIANTS)}", file=sys.stderr)
        return 2
    unknown_stages = [s for s in stages if s not in {"train", "eval", "collect"}]
    if unknown_stages:
        print(f"Unknown stages: {unknown_stages}; valid: ['train', 'eval', 'collect']", file=sys.stderr)
        return 2

    os.chdir(REPO_ROOT)
    overrides = list(args.overrides)
    if not any(value.lstrip("+").startswith("data=") for value in overrides):
        overrides.insert(0, "data=eump_full/server_v2")
    for variant_name in variants:
        spec = VARIANTS[variant_name]
        print(f"\n=== variant: {variant_name} (family={spec['family']}) ===", flush=True)
        run_variant(
            variant_name,
            spec,
            stages=stages,
            overrides=overrides,
            dry_run=args.dry_run,
        )
    if "collect" in stages and not args.dry_run:
        collect_variant_results(variants, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

