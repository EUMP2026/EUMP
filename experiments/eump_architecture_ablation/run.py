#!/usr/bin/env python
"""Minimal orchestrator for the architecture + pretrain-objective ablation.

Compares the ``mae_v3`` Swin MAE baseline against:

- ``architecture/swin_unet``: same Swin encoder, UNet-style decoder.
- ``architecture/convnext_unet``: ConvNeXt encoder, UNet-style decoder.
- ``pretrain_objective/contrastive_v2``: MoCo momentum-encoder contrastive pretrain.

Each variant runs through ``script/training/run_eump_full_train.sh`` and
``script/training/run_eump_full_eval.sh`` so resource controls (``GPU``,
``NPROC``, ``EVAL_GPU``, ``EVAL_NPROC``) flow through the existing launcher
contract. Use ``--smoke`` to exercise the full chain on a single GPU with
1 epoch and tiny batches.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


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
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EUMP architecture / pretrain-objective ablation")
    parser.add_argument(
        "--variants",
        default=",".join(VARIANTS.keys()),
        help="Comma-separated variant names. Default runs all variants.",
    )
    parser.add_argument("--stages", default="train,eval", help="Comma-separated stages: train,eval.")
    parser.add_argument("--smoke", action="store_true", help="Run a 1-epoch tiny smoke test.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra overrides forwarded to both train and eval invocations.",
    )
    return parser.parse_args(argv)


def _comma_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _smoke_overrides() -> list[str]:
    return [
        "++reconstruct.epochs=1",
        "++reconstruct.checkpoint_every_n_epochs=1",
        "++downstream.epochs=1",
        "++downstream.checkpoint_every_n_epochs=1",
        "++contrastive.epochs=1",
        "++contrastive.checkpoint_every_n_epochs=1",
    ]


def _run(cmd: list[str], *, dry_run: bool, env: dict[str, str] | None = None) -> None:
    shown = " ".join(shlex.quote(x) for x in cmd)
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
    smoke: bool,
    dry_run: bool,
) -> None:
    launch = str(spec["launch"])
    train_modes = list(spec["train_modes"])
    eval_modes = list(spec["eval_modes"])
    passthrough = list(overrides) + (_smoke_overrides() if smoke else [])

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
        _run_cv(name, spec, overrides=overrides, smoke=smoke, dry_run=dry_run)


def _run_cv(
    name: str,
    spec: dict[str, object],
    *,
    overrides: list[str],
    smoke: bool,
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
            "RUN_NAME": f"{cv['task']}_seed42",
        }
        cmd = ["bash", cv["script"], "--data", data_config, "--models", "swin"]
        if smoke:
            env["HEAD_EPOCHS"] = "1"
        if dry_run:
            cmd.append("--dry-run")
        _run(cmd, dry_run=False, env=env)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    variants = _comma_list(args.variants)
    stages = _comma_list(args.stages)
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        print(f"Unknown variants: {unknown}; valid: {list(VARIANTS)}", file=sys.stderr)
        return 2
    unknown_stages = [s for s in stages if s not in {"train", "eval"}]
    if unknown_stages:
        print(f"Unknown stages: {unknown_stages}; valid: ['train', 'eval']", file=sys.stderr)
        return 2

    os.chdir(REPO_ROOT)
    for variant_name in variants:
        spec = VARIANTS[variant_name]
        print(f"\n=== variant: {variant_name} (family={spec['family']}) ===", flush=True)
        run_variant(
            variant_name,
            spec,
            stages=stages,
            overrides=list(args.overrides),
            smoke=args.smoke,
            dry_run=args.dry_run,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

