#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shlex
import statistics
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from urban_embedding.core.paths import MODEL_DIR, resolve_model_artifact_path
from urban_embedding.core.reproducibility import set_reproducible
from urban_embedding.data.processed.dataset import build_eump_dataloaders
from urban_embedding.data.processed.manifest import write_json
from urban_embedding.data.processed.pipeline_spec import PipelineSpec, get_single_downstream_target_schema
from urban_embedding.data.processed.split_manifest import load_split_manifest
from urban_embedding.training.regression_metrics import inverse_processed_tensor
from urban_embedding.training.runner import normalize_runtime_config
from urban_embedding.training.schedulers import build_warmup_cosine_scheduler, normalize_warmup_cosine_scheduler_config


CONFIG_DIR = REPO_ROOT / "configs"
DEFAULT_TARGET_NAME = "Traffic"
DEFAULT_OUTPUT_PREFIX = "traffic_cv"
SPLIT_ORDER = ("train", "val", "test")


@dataclass(frozen=True, slots=True)
class FoldSpec:
    index: int
    path: Path
    train_scene_ids: tuple[str, ...]
    val_scene_ids: tuple[str, ...]
    test_scene_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CityScalarRows:
    embeddings: np.ndarray
    targets: np.ndarray
    split_names: np.ndarray
    scene_ids: np.ndarray
    sample_ids: np.ndarray
    years: np.ndarray


class _QuietLogger:
    def info(self, *args, **kwargs) -> None:
        return None

    def warning(self, *args, **kwargs) -> None:
        print(*args, file=sys.stderr, **kwargs)


class PooledEmbeddingScalarHead(nn.Module):
    def __init__(self, *, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(int(input_dim)),
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run city-scalar fixed-test CV experiments.")
    parser.add_argument("--config", type=Path, default=Path("configs/run/eump_full/traffic_cv.yaml"))
    parser.add_argument("--models", default=None, help="Comma-separated model names from the CV config models list.")
    parser.add_argument("--folds", default=None, help="Comma-separated fold indices, e.g. 0,1.")
    parser.add_argument("--cv-folds", type=int, default=None, help="Override the configured number of CV folds.")
    parser.add_argument("--fold-seed", type=int, default=None, help="Override the configured CV fold seed.")
    parser.add_argument("--experiment-group", default=None, help="Override the configured experiment group.")
    parser.add_argument("--run-name", default=None, help="Override the configured CV run name.")
    parser.add_argument("--output-prefix", default=None, help="Override the configured output prefix.")
    parser.add_argument("--prepare-embeddings-only", action="store_true")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides forwarded to composed configs and launcher commands.")
    return parser.parse_args(argv)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    return [value]


def _comma_set(value: str | None) -> set[str] | None:
    if value is None or not str(value).strip():
        return None
    return {item.strip() for item in str(value).split(",") if item.strip()}


def _comma_int_set(value: str | None) -> set[int] | None:
    raw = _comma_set(value)
    if raw is None:
        return None
    return {int(item) for item in raw}


def _natural_scene_sort_key(value: Any) -> tuple[int, int | str]:
    text = str(value)
    try:
        return 0, int(text)
    except ValueError:
        return 1, text


def _sorted_scene_ids(values: Any) -> tuple[str, ...]:
    return tuple(sorted((str(value) for value in values), key=_natural_scene_sort_key))


def _hash_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()[:20]


def _embedding_cache_cli_overrides(overrides: list[str]) -> list[str]:
    fingerprint_overrides: list[str] = []
    for override in overrides:
        text = str(override)
        key = text.split("=", 1)[0].lstrip("+")
        if key == "data" or key.startswith("data."):
            continue
        fingerprint_overrides.append(text)
    return fingerprint_overrides


def _normalize_embedding_overrides(overrides: Any) -> list[str]:
    normalized: list[str] = []
    for override in _as_list(overrides):
        text = str(override)
        if text == "++pipeline.inputs.optional_sources=[]":
            continue
        if text.startswith("experiment.scratch_backbone="):
            text = f"++{text}"
        elif text == "pipeline.targets.scalar=[]":
            text = "++pipeline.targets.scalar=[]"
        elif text.startswith("pipeline.targets.input_reconstruction.source_losses="):
            value = text.split("=", 1)[1]
            normalized.append("~pipeline.targets.input_reconstruction.source_losses")
            if value != "{}":
                normalized.append(f"++pipeline.targets.input_reconstruction.source_losses={value}")
            continue
        elif text.startswith("pipeline.targets.input_reconstruction.source_weights="):
            value = text.split("=", 1)[1]
            normalized.append("~pipeline.targets.input_reconstruction.source_weights")
            if value != "{}":
                normalized.append(f"++pipeline.targets.input_reconstruction.source_weights={value}")
            continue
        normalized.append(text)
    return normalized


def _safe_token(value: str) -> str:
    token = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value))
    token = "_".join(part for part in token.split("_") if part)
    return token or "target"


def _normalize_loaded_city_scalar_cv_config(cfg: dict[str, Any], *, section_name: str) -> dict[str, Any]:
    normalized = deepcopy(cfg)
    if "target_name" not in normalized:
        normalized["target_name"] = DEFAULT_TARGET_NAME if section_name == "traffic_cv" else ""
    if not str(normalized.get("target_name", "")).strip():
        raise ValueError("City-scalar CV config requires target_name")
    if "output_prefix" not in normalized:
        target_token = _safe_token(str(normalized["target_name"]))
        normalized["output_prefix"] = DEFAULT_OUTPUT_PREFIX if target_token == "traffic" else f"{target_token}_cv"
    if "target_pipeline" not in normalized and str(normalized["target_name"]) == DEFAULT_TARGET_NAME:
        normalized["target_pipeline"] = "eump_full/mae_v3/traffic"
    embedding = normalized.get("embedding")
    if isinstance(embedding, dict):
        embedding["overrides"] = _normalize_embedding_overrides(embedding.get("overrides"))
        model_name = str(embedding.get("model_name", ""))
        if model_name.startswith("eump_ablation_") and int(embedding.get("batch_size", 64)) == 256:
            embedding["batch_size"] = 64
    return normalized


def _printable_command(cmd: list[str], env: dict[str, str] | None = None) -> str:
    prefix: list[str] = []
    if env:
        prefix = [f"{key}={value}" for key, value in env.items()]
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


def load_city_scalar_cv_config(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(payload, dict):
        raise TypeError(f"City-scalar CV config must be a mapping: {path}")
    section_name = "city_scalar_cv"
    cfg = payload
    for candidate in ("city_scalar_cv", "traffic_cv", "pm25_cv"):
        section = payload.get(candidate)
        if isinstance(section, dict):
            section_name = candidate
            cfg = section
            break
    if not isinstance(cfg, dict):
        raise TypeError(f"{section_name} config section must be a mapping: {path}")
    return _normalize_loaded_city_scalar_cv_config(cfg, section_name=section_name)


def load_traffic_cv_config(path: Path) -> dict[str, Any]:
    return load_city_scalar_cv_config(path)


def compose_runtime_config(*, pipeline: str, overrides: list[str]) -> dict[str, Any]:
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="train", overrides=[f"pipeline={pipeline}", *overrides])
    return normalize_runtime_config(cfg)


def resolve_default_path(value: Any, *, default: Path) -> Path:
    if value is None or str(value).strip() == "":
        return default
    path = Path(str(value))
    return path if path.is_absolute() else (REPO_ROOT / path)


def generate_fold_manifests(
    *,
    split_manifest_path: Path,
    output_dir: Path,
    folds: int,
    seed: int,
    output_prefix: str = DEFAULT_OUTPUT_PREFIX,
) -> tuple[FoldSpec, ...]:
    original = load_split_manifest(split_manifest_path)
    original_train = _sorted_scene_ids(original["train_scene_ids"])
    original_val = _sorted_scene_ids(original["val_scene_ids"])
    original_test = _sorted_scene_ids(original["test_scene_ids"])
    dev_scene_ids = list(_sorted_scene_ids((*original_train, *original_val)))
    if int(folds) < 2:
        raise ValueError(f"{output_prefix}.folds must be >= 2, got {folds}")
    if int(folds) > len(dev_scene_ids):
        raise ValueError(f"{output_prefix}.folds={folds} exceeds dev scene count={len(dev_scene_ids)}")

    rng = np.random.default_rng(int(seed))
    shuffled = np.asarray(dev_scene_ids, dtype=object)
    rng.shuffle(shuffled)
    chunks = [list(chunk) for chunk in np.array_split(shuffled, int(folds))]
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_specs: list[FoldSpec] = []
    dev_set = set(dev_scene_ids)
    for fold_index, val_ids_raw in enumerate(chunks):
        val_ids = _sorted_scene_ids(val_ids_raw)
        train_ids = _sorted_scene_ids(dev_set.difference(val_ids))
        test_ids = original_test
        pool_ids = _sorted_scene_ids((*train_ids, *val_ids, *test_ids))
        payload = deepcopy(original)
        payload.update(
            {
                "generator": f"{output_prefix}_train_val_kfold_v1",
                "seed": int(seed),
                "fold_index": int(fold_index),
                "folds": int(folds),
                "source_split_manifest_path": str(split_manifest_path),
                "source_train_count": len(original_train),
                "source_val_count": len(original_val),
                "fixed_test_count": len(original_test),
                "pool_scene_ids": list(pool_ids),
                "train_scene_ids": list(train_ids),
                "val_scene_ids": list(val_ids),
                "test_scene_ids": list(test_ids),
            }
        )
        fold_path = output_dir / f"fold_{fold_index}.json"
        write_json(fold_path, payload)
        fold_specs.append(
            FoldSpec(
                index=fold_index,
                path=fold_path,
                train_scene_ids=train_ids,
                val_scene_ids=val_ids,
                test_scene_ids=test_ids,
            )
        )
    return tuple(fold_specs)


def write_fold_summary(
    path: Path,
    folds: tuple[FoldSpec, ...],
    *,
    target_name: str | None = None,
    target_counts: dict[int, dict[str, int]] | None = None,
) -> None:
    target_counts = {} if target_counts is None else target_counts
    payload = {
        "target": target_name,
        "folds": [
            {
                "fold": fold.index,
                "path": str(fold.path),
                "train_scene_ids": list(fold.train_scene_ids),
                "val_scene_ids": list(fold.val_scene_ids),
                "test_scene_ids": list(fold.test_scene_ids),
                "train_scene_count": len(fold.train_scene_ids),
                "val_scene_count": len(fold.val_scene_ids),
                "test_scene_count": len(fold.test_scene_ids),
                "valid_target_rows": target_counts.get(fold.index, {}),
            }
            for fold in folds
        ]
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def resolve_embedding_output_path(
    *,
    cfg: dict[str, Any],
    data_config: dict[str, Any],
    cache_dir: Path,
    output_prefix: str = DEFAULT_OUTPUT_PREFIX,
) -> Path:
    model_name = str(cfg["model_name"])
    checkpoint_file = str(cfg["checkpoint_file"])
    checkpoint_path = Path(str(resolve_model_artifact_path(model_name, checkpoint_file)))
    embedding_overrides = _normalize_embedding_overrides(cfg.get("overrides"))
    cli_overrides = [str(override) for override in _as_list(cfg.get("cli_overrides"))]
    preprocessing_path = Path(str(data_config.get("preprocessing_path", "")))
    if not preprocessing_path.is_absolute():
        preprocessing_path = REPO_ROOT / preprocessing_path
    eump_root = Path(str(data_config.get("eump_root", "")))
    if not eump_root.is_absolute():
        eump_root = REPO_ROOT / eump_root
    payload = {
        "kind": f"{output_prefix}_swin_pooled_embedding_cache",
        "model_name": model_name,
        "checkpoint_file": checkpoint_file,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _hash_file(checkpoint_path),
        "checkpoint_source": str(cfg.get("checkpoint_source", "auto")),
        "pipeline": str(cfg["pipeline"]),
        "eump_root": str(eump_root),
        "preprocessing_path": str(preprocessing_path),
        "preprocessing_sha256": _hash_file(preprocessing_path),
        "years": list(resolve_embedding_years(cfg)),
        "pooled": True,
    }
    if embedding_overrides:
        payload["overrides"] = embedding_overrides
    if cli_overrides:
        payload["cli_overrides"] = cli_overrides
    return cache_dir / f"{output_prefix}_swin_pooled_{_stable_hash(payload)}.npz"


def resolve_embedding_years(cfg: dict[str, Any]) -> tuple[int, int]:
    raw_years = _as_list(cfg.get("years", [2017, 2020]))
    if len(raw_years) == 2:
        start, end = int(raw_years[0]), int(raw_years[1])
    else:
        start, end = min(int(year) for year in raw_years), max(int(year) for year in raw_years)
    if start > end:
        raise ValueError(f"Invalid embedding years: {raw_years!r}")
    return start, end


def ensure_embedding_cache(
    *,
    cfg: dict[str, Any],
    data_overrides: list[str],
    output_path: Path,
    commands_log: Path,
    dry_run: bool,
) -> None:
    if output_path.exists() and not bool(cfg.get("overwrite", False)):
        print(f"Reuse Swin embedding cache: {output_path}")
        return
    start_year, end_year = resolve_embedding_years(cfg)
    embedding_overrides = _normalize_embedding_overrides(cfg.get("overrides"))
    hydra_overrides = [
        "data.runtime_split_manifest_path=null",
        *data_overrides,
        f"pipeline={cfg['pipeline']}",
        *embedding_overrides,
    ]
    gpu = os.environ.get("GPU") or os.environ.get("CUDA_VISIBLE_DEVICES") or "0"
    nproc = os.environ.get("NPROC")
    if nproc is None:
        nproc = str(max(1, len([part for part in gpu.split(",") if part.strip()])))
    cmd = [
        "torchrun",
        f"--nproc_per_node={nproc}",
        f"--nnodes={os.environ.get('NNODES', '1')}",
        f"--node_rank={os.environ.get('NODE_RANK', '0')}",
        f"--master_addr={os.environ.get('MASTER_ADDR', '127.0.0.1')}",
        f"--master_port={os.environ.get('MASTER_PORT', '29500')}",
        "script/experiments/export_eump_embeddings.py",
        "--year-range",
        str(start_year),
        str(end_year),
        "--model-name",
        str(cfg["model_name"]),
        "--checkpoint-file",
        str(cfg["checkpoint_file"]),
        "--checkpoint-source",
        str(cfg.get("checkpoint_source", "auto")),
        "--batch-size",
        str(int(cfg.get("batch_size", 256))),
        "--workers-per-gpu",
        str(int(cfg.get("workers_per_gpu", 4))),
        "--torch-threads",
        str(int(cfg.get("torch_threads", 1))),
        "--output-path",
        str(output_path),
    ]
    if bool(cfg.get("overwrite", False)):
        cmd.append("--overwrite")
    cmd.extend(hydra_overrides)
    run_command(
        cmd,
        dry_run=dry_run,
        commands_log=commands_log,
        env={
            "CUDA_VISIBLE_DEVICES": gpu,
            "EUMP_TORCH_SHARING_STRATEGY": os.environ.get("EUMP_TORCH_SHARING_STRATEGY", "file_system"),
            "OMP_NUM_THREADS": str(cfg.get("torch_threads", 1)),
            "MKL_NUM_THREADS": str(cfg.get("torch_threads", 1)),
        },
    )


def load_embedding_table(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Swin embedding cache not found: {path}")
    payload = np.load(path, allow_pickle=False)
    embeddings = payload["embeddings"].astype(np.float32, copy=False)
    if embeddings.ndim != 2:
        raise ValueError(f"City-scalar CV requires pooled [N, C] embeddings, got shape={embeddings.shape}")
    scene_ids = payload["scene_ids"].astype(str, copy=False)
    years = payload["years"].astype(np.int32, copy=False)
    table: dict[str, np.ndarray] = {}
    for idx, (scene_id, year) in enumerate(zip(scene_ids, years)):
        key = f"{scene_id}:{int(year)}"
        if key in table:
            raise ValueError(f"Duplicate embedding row key in {path}: {key}")
        table[key] = embeddings[idx]
    return table


def target_only_pipeline_config(pipeline_config: dict[str, Any]) -> dict[str, Any]:
    config = deepcopy(pipeline_config)
    config["inputs"] = {"sources": []}
    config["broadcast_scalar"] = False
    config["include_city_mask_as_input"] = config.get("city_mask_source") not in (None, "")
    config["cache_scenes"] = False
    config["preload_scenes"] = False
    config["pin_memory"] = False
    config["persistent_workers"] = False
    config["prefetch_factor"] = 1
    config["num_workers"] = 0
    return config


def _target_schema_and_source_spec(dataloader, *, target_name: str):
    dataset = getattr(dataloader, "dataset", None)
    batch_schema = getattr(dataset, "batch_schema", None)
    preprocessing_spec = getattr(dataset, "preprocessing_spec", None)
    if batch_schema is None or preprocessing_spec is None:
        raise ValueError("Unable to resolve target schema/preprocessing spec from dataloader")
    target_schema = get_single_downstream_target_schema(batch_schema)
    if target_schema.name != target_name:
        raise ValueError(
            f"Configured target_name={target_name!r} does not match pipeline target {target_schema.name!r}"
        )
    if target_schema.name not in {schema.name for schema in batch_schema.scalar_targets}:
        raise ValueError(
            f"City-scalar CV requires a scalar target, got {target_schema.name!r} "
            f"with target_type={target_schema.spec.target_type!r}"
        )
    source_name = target_schema.spec.source
    if source_name is None:
        raise ValueError(f"Target {target_schema.name!r} requires a source")
    source_spec = preprocessing_spec["sources"].get(source_name)
    if source_spec is None:
        raise KeyError(f"Source {source_name!r} is missing from preprocessing spec")
    return target_schema, source_spec


def _valid_target_count_for_batch(batch, *, target_name: str) -> int:
    if target_name not in batch.scalar_targets:
        available = ", ".join(sorted(batch.scalar_targets))
        raise KeyError(f"Target {target_name!r} is missing from scalar targets. Available scalar targets: {available}")
    target_block = batch.scalar_targets[target_name]
    target_data = target_block.data.detach().cpu().numpy().astype(np.float32, copy=False)
    valid_mask = target_block.valid_mask.detach().cpu().numpy().astype(bool, copy=False)
    valid_rows = valid_mask.all(axis=1) & np.isfinite(target_data).all(axis=1)
    return int(np.count_nonzero(valid_rows))


def count_valid_target_rows_for_fold(
    *,
    data_config: dict[str, Any],
    pipeline_config: dict[str, Any],
    fold: FoldSpec,
    random_seed: int,
    target_name: str,
) -> dict[str, int]:
    fold_data_config = dict(data_config)
    fold_data_config["runtime_split_manifest_path"] = str(fold.path)
    pipeline_spec = PipelineSpec.from_mapping(target_only_pipeline_config(pipeline_config))
    train_dl, val_dl, test_dl, _ = build_eump_dataloaders(
        data_config=fold_data_config,
        pipeline_spec=pipeline_spec,
        logger=_QuietLogger(),
        random_seed=int(random_seed),
        batch_size=1024,
        local_rank=0,
        world_size=1,
        is_distributed=False,
        shuffle_train=False,
    )
    _target_schema_and_source_spec(train_dl, target_name=target_name)
    counts: dict[str, int] = {}
    for split_name, dataloader in zip(SPLIT_ORDER, (train_dl, val_dl, test_dl)):
        split_count = 0
        for batch in dataloader:
            split_count += _valid_target_count_for_batch(batch, target_name=target_name)
        counts[split_name] = int(split_count)
    if counts.get("train", 0) <= 0 or counts.get("val", 0) <= 0:
        raise ValueError(
            f"Fold {fold.index} has no valid {target_name} rows in train/val under the fixed city split: "
            f"{counts}. Reduce folds, use a different seed, or inspect target coverage; city folds were not changed."
        )
    return counts


def load_city_scalar_rows_for_fold(
    *,
    data_config: dict[str, Any],
    pipeline_config: dict[str, Any],
    fold: FoldSpec,
    embedding_table: dict[str, np.ndarray],
    random_seed: int,
    target_name: str,
) -> tuple[CityScalarRows, dict[str, Any]]:
    fold_data_config = dict(data_config)
    fold_data_config["runtime_split_manifest_path"] = str(fold.path)
    pipeline_spec = PipelineSpec.from_mapping(target_only_pipeline_config(pipeline_config))
    train_dl, val_dl, test_dl, _ = build_eump_dataloaders(
        data_config=fold_data_config,
        pipeline_spec=pipeline_spec,
        logger=_QuietLogger(),
        random_seed=int(random_seed),
        batch_size=1024,
        local_rank=0,
        world_size=1,
        is_distributed=False,
        shuffle_train=False,
    )
    target_schema, source_spec = _target_schema_and_source_spec(train_dl, target_name=target_name)
    embedding_rows: list[np.ndarray] = []
    targets: list[float] = []
    split_names: list[str] = []
    scene_ids: list[str] = []
    sample_ids: list[str] = []
    years: list[int] = []
    missing_embedding_keys: list[str] = []

    for split_name, dataloader in zip(SPLIT_ORDER, (train_dl, val_dl, test_dl)):
        for batch in dataloader:
            target_block = batch.scalar_targets[target_name]
            target_data = target_block.data.detach().cpu().numpy().astype(np.float32, copy=False)
            valid_mask = target_block.valid_mask.detach().cpu().numpy().astype(bool, copy=False)
            for row_idx, meta in enumerate(batch.meta):
                if not bool(valid_mask[row_idx].all()) or not bool(np.isfinite(target_data[row_idx]).all()):
                    continue
                scene_id = str(meta.get("scene_id", ""))
                year = int(meta.get("year"))
                key = f"{scene_id}:{year}"
                embedding = embedding_table.get(key)
                if embedding is None:
                    missing_embedding_keys.append(key)
                    continue
                embedding_rows.append(embedding)
                targets.append(float(target_data[row_idx, 0]))
                split_names.append(split_name)
                scene_ids.append(scene_id)
                sample_ids.append("" if meta.get("sample_id") is None else str(meta.get("sample_id")))
                years.append(year)

    if missing_embedding_keys:
        preview = ", ".join(missing_embedding_keys[:10])
        suffix = "" if len(missing_embedding_keys) <= 10 else f", ... ({len(missing_embedding_keys)} total)"
        raise ValueError(f"Missing embedding rows for {target_name} targets: {preview}{suffix}")
    if not embedding_rows:
        raise ValueError(f"Fold {fold.index} produced no valid {target_name} rows")
    return (
        CityScalarRows(
            embeddings=np.stack(embedding_rows, axis=0).astype(np.float32, copy=False),
            targets=np.asarray(targets, dtype=np.float32).reshape(-1, 1),
            split_names=np.asarray(split_names, dtype=str),
            scene_ids=np.asarray(scene_ids, dtype=str),
            sample_ids=np.asarray(sample_ids, dtype=str),
            years=np.asarray(years, dtype=np.int32),
        ),
        {
            "target_name": target_schema.name,
            "source_name": target_schema.spec.source,
            "source_spec": source_spec,
        },
    )


def regression_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    pred = prediction.reshape(-1).astype(np.float64)
    tgt = target.reshape(-1).astype(np.float64)
    if tgt.size <= 0:
        return {"count": 0, "mae": None, "mse": None, "rmse": None, "r2": None}
    residual = pred - tgt
    mse = float(np.mean(residual ** 2))
    mae = float(np.mean(np.abs(residual)))
    ss_res = float(np.sum(residual ** 2))
    centered = tgt - float(np.mean(tgt))
    ss_tot = float(np.sum(centered ** 2))
    r2 = None if ss_tot <= 0 else 1.0 - ss_res / ss_tot
    return {
        "count": int(tgt.size),
        "mae": mae,
        "mse": mse,
        "rmse": math.sqrt(mse),
        "r2": None if r2 is None else float(r2),
    }


def inverse_processed_array(values: np.ndarray, source_spec: dict[str, Any]) -> np.ndarray:
    tensor = torch.as_tensor(values.astype(np.float32, copy=False))
    restored = inverse_processed_tensor(tensor, source_spec)
    return restored.detach().cpu().numpy().astype(np.float32, copy=False)


def evaluate_head(
    *,
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    device: torch.device,
    target_source_spec: dict[str, Any],
) -> tuple[float, np.ndarray, dict[str, Any]]:
    model.eval()
    predictions: list[np.ndarray] = []
    losses: list[float] = []
    counts: list[int] = []
    with torch.no_grad():
        for start in range(0, int(x.shape[0]), int(batch_size)):
            xb = x[start:start + int(batch_size)].to(device)
            yb = y[start:start + int(batch_size)].to(device)
            pred = model(xb)
            loss = torch.mean(torch.abs(pred - yb))
            losses.append(float(loss.detach().cpu().item()) * int(xb.shape[0]))
            counts.append(int(xb.shape[0]))
            predictions.append(pred.detach().cpu().numpy())
    prediction_array = np.concatenate(predictions, axis=0) if predictions else np.empty((0, 1), dtype=np.float32)
    total = sum(losses) / max(1, sum(counts))
    target_array = y.detach().cpu().numpy().astype(np.float32, copy=False)
    physical_prediction = inverse_processed_array(prediction_array, target_source_spec)
    physical_target = inverse_processed_array(target_array, target_source_spec)
    return total, prediction_array, regression_metrics(physical_prediction, physical_target)


def split_tensors(rows: CityScalarRows, split_name: str, *, target_name: str) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    mask = rows.split_names == split_name
    if int(mask.sum()) <= 0:
        raise ValueError(f"City-scalar CV split {split_name!r} has no valid {target_name} rows")
    return (
        torch.as_tensor(rows.embeddings[mask], dtype=torch.float32),
        torch.as_tensor(rows.targets[mask], dtype=torch.float32),
        mask,
    )


def train_swin_embedding_head(
    *,
    variant: dict[str, Any],
    fold: FoldSpec,
    rows: CityScalarRows,
    target_context: dict[str, Any],
    output_prefix: str,
    output_root: Path,
    random_seed: int,
    dry_run: bool,
    results: list[dict[str, Any]],
) -> None:
    model_name = str(variant["name"])
    target_name = str(target_context["target_name"])
    target_source_spec = dict(target_context["source_spec"])
    fold_name = f"{model_name}_fold{fold.index}"
    model_dir = Path(MODEL_DIR) / output_root.name / str(output_prefix) / fold_name
    checkpoint_path = model_dir / "swin_embedding_scalar_head.pt"
    evaluation_dir = model_dir / "evaluation"
    print(f"Run Swin embedding-head city-scalar CV target={target_name} model={model_name} fold={fold.index}")
    if dry_run:
        print(f"+ train_swin_embedding_head model={model_name} fold={fold.index} checkpoint={checkpoint_path}")
        return

    seed = int(random_seed) + int(fold.index) * 1009
    set_reproducible(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_x, train_y, _ = split_tensors(rows, "train", target_name=target_name)
    val_x, val_y, _ = split_tensors(rows, "val", target_name=target_name)
    test_x, test_y, _ = split_tensors(rows, "test", target_name=target_name)
    input_dim = int(train_x.shape[1])
    model = PooledEmbeddingScalarHead(
        input_dim=input_dim,
        hidden_dim=int(variant.get("scalar_hidden_dim", 128)),
        dropout=float(variant.get("dropout", 0.5)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(variant.get("lr", 1.0e-4)),
        weight_decay=float(variant.get("weight_decay", 1.0e-2)),
    )
    batch_size = int(variant.get("batch_size", 64))
    epochs = int(variant.get("epochs", 80))
    scheduler_cfg = normalize_warmup_cosine_scheduler_config(variant.get("scheduler"))
    scheduler, _ = build_warmup_cosine_scheduler(
        optimizer=optimizer,
        base_lr=float(variant.get("lr", 1.0e-4)),
        num_batches=math.ceil(int(train_x.shape[0]) / max(1, batch_size)),
        accumulation_steps=1,
        epochs=epochs,
        warmup_fraction=scheduler_cfg.warmup_fraction,
        lr_start_factor=scheduler_cfg.lr_start_factor,
        lr_end_factor=scheduler_cfg.lr_end_factor,
    )
    dataset = TensorDataset(train_x, train_y)
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    best_metric = float("inf")
    best_epoch = None
    best_state = None
    model_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: list[float] = []
        train_counts: list[int] = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = torch.mean(torch.abs(pred - yb))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            train_losses.append(float(loss.detach().cpu().item()) * int(xb.shape[0]))
            train_counts.append(int(xb.shape[0]))

        train_loss = sum(train_losses) / max(1, sum(train_counts))
        val_loss, _, val_metrics = evaluate_head(
            model=model,
            x=val_x,
            y=val_y,
            batch_size=batch_size,
            device=device,
            target_source_spec=target_source_spec,
        )
        selection_metric = val_loss if str(variant.get("selection_metric", "val_loss")) == "val_loss" else val_metrics["rmse"]
        if selection_metric is not None and float(selection_metric) < best_metric:
            best_metric = float(selection_metric)
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save(
                {
                    "model_state_dict": best_state,
                    "model": {
                        "kind": "swin_embedding_scalar_head",
                        "input_dim": input_dim,
                        "scalar_hidden_dim": int(variant.get("scalar_hidden_dim", 128)),
                        "dropout": float(variant.get("dropout", 0.5)),
                    },
                    "target": target_name,
                    "target_source": target_context.get("source_name"),
                    "fold": int(fold.index),
                    "best_epoch": int(best_epoch),
                    "best_metric": float(best_metric),
                    "variant": variant,
                },
                checkpoint_path,
            )
        if epoch == 1 or epoch == epochs or epoch % 10 == 0:
            val_r2 = "null" if val_metrics["r2"] is None else f"{val_metrics['r2']:.6f}"
            print(
                "Swin embedding CV "
                f"target={target_name} "
                f"model={model_name} fold={fold.index} epoch={epoch}/{epochs} "
                f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
                f"val_rmse={val_metrics['rmse']:.6f} val_r2={val_r2}"
            )

    if best_state is None:
        raise RuntimeError(f"No best state selected for model={model_name} fold={fold.index}")
    model.load_state_dict(best_state)
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_x, split_y in (("train", train_x, train_y), ("val", val_x, val_y), ("test", test_x, test_y)):
        loss, _, metrics = evaluate_head(
            model=model,
            x=split_x,
            y=split_y,
            batch_size=batch_size,
            device=device,
            target_source_spec=target_source_spec,
        )
        summary = {
            "experiment": {
                "group": str(output_root.name),
                "name": f"{output_prefix}/{fold_name}",
                "model_name": f"{output_root.name}/{output_prefix}/{fold_name}",
            },
            "checkpoint": {
                "source": "best_model",
                "file": checkpoint_path.name,
                "path": str(checkpoint_path),
            },
            "split": split_name,
            "target": target_name,
            "target_source": target_context.get("source_name"),
            "target_kind": "swin_embedding_city_scalar",
            "model_kind": "swin_embedding_head",
            "training_target_scale": "processed",
            "metric_scale": "physical",
            "fold": int(fold.index),
            "best_epoch": int(best_epoch),
            "loss": {"val_selected": float(best_metric), "current_split_mae": float(loss)},
            "metrics": {"global": metrics, "city_mean": metrics},
        }
        (evaluation_dir / f"{split_name}_metrics.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        results.append(result_row_from_metrics(model_name=model_name, fold=fold.index, split=split_name, summary=summary))


def evaluate_pretrained_embedding_head(
    *,
    variant: dict[str, Any],
    fold: FoldSpec,
    rows: CityScalarRows,
    target_context: dict[str, Any],
    output_prefix: str,
    random_seed: int,
    dry_run: bool,
    results: list[dict[str, Any]],
) -> None:
    """Score an EXISTING pooled-embedding head (no training).

    Loads a head checkpoint trained elsewhere (e.g. the mainline eump_full_mae_v3
    CV head) and evaluates it against the embedding rows for this fold. Pair it
    with a drop-augmented embedding cache (drop injected at export time) to get
    the clean-arm robustness number under the SAME drop pipeline the launcher arm
    uses -- the symmetric clean-vs-drop comparison without retraining.

    variant keys: name, source_group (group holding the head), head_name (the head
    model dir name under <source_group>/<output_prefix>/), plus the usual head dims
    for the fallback when the checkpoint lacks a saved 'model' block.
    """
    model_name = str(variant["name"])
    source_group = str(variant["source_group"])
    head_name = str(variant.get("head_name", model_name))
    target_name = str(target_context["target_name"])
    target_source_spec = dict(target_context["source_spec"])
    fold_name = f"{head_name}_fold{fold.index}"
    model_dir = Path(MODEL_DIR) / source_group / str(output_prefix) / fold_name
    checkpoint_path = model_dir / "swin_embedding_scalar_head.pt"
    evaluation_dir = (
        Path(MODEL_DIR) / source_group / str(output_prefix) / f"{model_name}_fold{fold.index}" / "evaluation"
    )
    print(
        f"Eval pretrained embedding-head target={target_name} model={model_name} "
        f"source={source_group} fold={fold.index} checkpoint={checkpoint_path}"
    )
    if dry_run:
        print(f"+ evaluate_pretrained_embedding_head model={model_name} fold={fold.index} checkpoint={checkpoint_path}")
        return
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Pretrained head checkpoint not found for model={model_name} fold={fold.index}: {checkpoint_path}"
        )

    set_reproducible(int(random_seed) + int(fold.index) * 1009)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    payload = torch.load(checkpoint_path, map_location="cpu")
    head_cfg = dict(payload.get("model", {}))
    train_x, _, _ = split_tensors(rows, "train", target_name=target_name)
    input_dim = int(head_cfg.get("input_dim", int(train_x.shape[1])))
    model = PooledEmbeddingScalarHead(
        input_dim=input_dim,
        hidden_dim=int(head_cfg.get("scalar_hidden_dim", variant.get("scalar_hidden_dim", 128))),
        dropout=float(head_cfg.get("dropout", variant.get("dropout", 0.5))),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    batch_size = int(variant.get("batch_size", 64))
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    for split_name in SPLIT_ORDER:
        split_x, split_y, _ = split_tensors(rows, split_name, target_name=target_name)
        loss, _, metrics = evaluate_head(
            model=model,
            x=split_x,
            y=split_y,
            batch_size=batch_size,
            device=device,
            target_source_spec=target_source_spec,
        )
        summary = {
            "experiment": {
                "group": source_group,
                "name": f"{output_prefix}/{model_name}_fold{fold.index}",
                "model_name": f"{source_group}/{output_prefix}/{fold_name}",
            },
            "checkpoint": {
                "source": "pretrained",
                "file": checkpoint_path.name,
                "path": str(checkpoint_path),
            },
            "split": split_name,
            "target": target_name,
            "target_source": target_context.get("source_name"),
            "target_kind": "swin_embedding_city_scalar",
            "model_kind": "swin_embedding_head_eval",
            "training_target_scale": "processed",
            "metric_scale": "physical",
            "fold": int(fold.index),
            "loss": {"current_split_mae": float(loss)},
            "metrics": {"global": metrics, "city_mean": metrics},
        }
        (evaluation_dir / f"{split_name}_metrics.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        results.append(result_row_from_metrics(model_name=model_name, fold=fold.index, split=split_name, summary=summary))


def result_row_from_metrics(*, model_name: str, fold: int, split: str, summary: dict[str, Any]) -> dict[str, Any]:
    metrics = summary.get("metrics", {}).get("global", {})
    return {
        "model": model_name,
        "fold": int(fold),
        "split": split,
        "target": summary.get("target"),
        "target_kind": summary.get("target_kind"),
        "model_kind": summary.get("model_kind"),
        "count": metrics.get("count"),
        "mae": metrics.get("mae"),
        "mse": metrics.get("mse"),
        "rmse": metrics.get("rmse"),
        "r2": metrics.get("r2"),
        "checkpoint_path": summary.get("checkpoint", {}).get("path"),
    }


def run_launcher_variant(
    *,
    variant: dict[str, Any],
    fold: FoldSpec,
    cli_overrides: list[str],
    experiment_group: str,
    output_prefix: str,
    commands_log: Path,
    dry_run: bool,
    results: list[dict[str, Any]],
    eval_only: bool = False,
) -> None:
    model_name = str(variant["name"])
    fold_experiment_name = f"{output_prefix}/{model_name}_fold{fold.index}"
    mode = str(variant["mode"])
    launch = str(variant["launch"])
    base_overrides = [
        *cli_overrides,
        f"launch={launch}",
        f"data.runtime_split_manifest_path={fold.path}",
        f"experiment.group={experiment_group}",
        f"experiment.name={fold_experiment_name}",
        *_as_list(variant.get("overrides")),
    ]
    # eval_only re-scores a checkpoint trained in a prior run (e.g. the
    # channel-drop robustness eval, which loads the drop-trained per-fold models
    # read-only and applies inference-time channel drop via variant.overrides).
    if not eval_only:
        train_cmd = ["bash", "script/training/run_eump_full_train.sh", mode, *base_overrides]
        run_command(train_cmd, dry_run=dry_run, commands_log=commands_log)
    for split_name in SPLIT_ORDER:
        eval_cmd = [
            "bash",
            "script/training/run_eump_full_eval.sh",
            mode,
            *base_overrides,
            f"evaluation.split={split_name}",
            "evaluation.checkpoint_source=best_model",
        ]
        run_command(eval_cmd, dry_run=dry_run, commands_log=commands_log)
        metrics_path = Path(MODEL_DIR) / experiment_group / fold_experiment_name / "evaluation" / f"{split_name}_metrics.json"
        if metrics_path.exists():
            summary = json.loads(metrics_path.read_text(encoding="utf-8"))
            summary["model_kind"] = "launcher"
            if launch == "eump_full/mae_v3_one_stage":
                summary["target_kind"] = "one_stage_city_scalar_cv"
            results.append(result_row_from_metrics(model_name=model_name, fold=fold.index, split=split_name, summary=summary))
        elif not dry_run:
            raise FileNotFoundError(
                "City-scalar CV evaluation finished but metrics JSON was not found: "
                f"{metrics_path}. Check the evaluation log for model={model_name} fold={fold.index} split={split_name}."
            )


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def write_fold_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "fold", "split", "target", "target_kind", "model_kind", "count", "mae", "mse", "rmse", "r2", "checkpoint_path"]
    sorted_rows = sorted(rows, key=lambda row: (str(row["model"]), int(row["fold"]), str(row["split"])))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted_rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["model"]), str(row["split"])), []).append(row)
    summary_rows: list[dict[str, Any]] = []
    for (model, split), values in sorted(grouped.items()):
        rmse_values = [float(row["rmse"]) for row in values if row.get("rmse") is not None]
        r2_values = [float(row["r2"]) for row in values if row.get("r2") is not None]
        mae_values = [float(row["mae"]) for row in values if row.get("mae") is not None]
        summary_rows.append(
            {
                "model": model,
                "split": split,
                "folds": len(values),
                "rmse_mean": statistics.fmean(rmse_values) if rmse_values else "",
                "rmse_std": statistics.stdev(rmse_values) if len(rmse_values) > 1 else 0.0 if rmse_values else "",
                "r2_mean": statistics.fmean(r2_values) if r2_values else "",
                "r2_std": statistics.stdev(r2_values) if len(r2_values) > 1 else 0.0 if r2_values else "",
                "mae_mean": statistics.fmean(mae_values) if mae_values else "",
                "mae_std": statistics.stdev(mae_values) if len(mae_values) > 1 else 0.0 if mae_values else "",
            }
        )
    fieldnames = ["model", "split", "folds", "rmse_mean", "rmse_std", "r2_mean", "r2_std", "mae_mean", "mae_std"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def persist_results(*, log_root: Path, rows: list[dict[str, Any]], output_prefix: str) -> tuple[Path, Path, Path]:
    results_path = log_root / f"{output_prefix}_results.jsonl"
    fold_metrics_path = log_root / f"{output_prefix}_fold_metrics.csv"
    summary_path = log_root / f"{output_prefix}_summary.csv"
    append_jsonl(results_path, rows)
    write_fold_metrics_csv(fold_metrics_path, rows)
    write_summary_csv(summary_path, rows)
    return results_path, fold_metrics_path, summary_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    os.chdir(REPO_ROOT)
    dry_run = os.environ.get("DRY_RUN", "0") == "1"
    cfg = load_city_scalar_cv_config(args.config)
    if args.cv_folds is not None:
        cfg["folds"] = int(args.cv_folds)
    if args.fold_seed is not None:
        cfg["fold_seed"] = int(args.fold_seed)
    if args.experiment_group is not None:
        cfg["experiment_group"] = str(args.experiment_group)
    if args.run_name is not None:
        cfg["run_name"] = str(args.run_name)
    if args.output_prefix is not None:
        cfg["output_prefix"] = str(args.output_prefix)
    target_name = str(cfg["target_name"])
    output_prefix = str(cfg.get("output_prefix", DEFAULT_OUTPUT_PREFIX))
    if "target_pipeline" not in cfg:
        raise ValueError(f"City-scalar CV config for target={target_name!r} requires target_pipeline")
    target_runtime = compose_runtime_config(
        pipeline=str(cfg["target_pipeline"]),
        overrides=args.overrides,
    )
    data_config = dict(target_runtime["data"])
    split_manifest_path = Path(str(data_config["split_manifest_path"]))
    if not split_manifest_path.is_absolute():
        split_manifest_path = REPO_ROOT / split_manifest_path
    eump_root = Path(str(data_config["eump_root"]))
    if not eump_root.is_absolute():
        eump_root = REPO_ROOT / eump_root
    folds = int(cfg.get("folds", 5))
    fold_seed = int(cfg.get("fold_seed", 42))
    experiment_group = str(cfg.get("experiment_group", "eump_traffic_cv"))
    run_name = str(cfg.get("run_name", f"{output_prefix}_seed{fold_seed}"))
    log_root = REPO_ROOT / "log" / experiment_group / run_name
    commands_log = log_root / "commands.log"
    if commands_log.exists() and not dry_run:
        commands_log.unlink()
    fold_dir = resolve_default_path(
        cfg.get("fold_manifest_dir"),
        default=eump_root / "manifest" / "splits" / f"{output_prefix}_seed{fold_seed}",
    )
    embedding_cache_dir = resolve_default_path(
        cfg.get("embedding_cache_dir"),
        default=eump_root / "manifest" / "cache" / "swin_embeddings",
    )
    fold_specs = generate_fold_manifests(
        split_manifest_path=split_manifest_path,
        output_dir=fold_dir,
        folds=folds,
        seed=fold_seed,
        output_prefix=output_prefix,
    )
    write_fold_summary(log_root / "folds.json", fold_specs, target_name=target_name)
    selected_folds = _comma_int_set(args.folds)
    if selected_folds is not None:
        unknown = sorted(selected_folds.difference({fold.index for fold in fold_specs}))
        if unknown:
            raise ValueError(f"Unknown fold index: {unknown}")
        fold_specs = tuple(fold for fold in fold_specs if fold.index in selected_folds)

    models = _as_list(cfg.get("models"))
    selected_models = _comma_set(args.models)
    if selected_models is not None:
        known = {str(model.get("name")) for model in models}
        unknown = sorted(selected_models.difference(known))
        if unknown:
            raise ValueError(f"Unknown city-scalar CV model(s): {unknown}; available={sorted(known)}")
        models = [model for model in models if str(model.get("name")) in selected_models]
    names = [str(model.get("name")) for model in models]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate city-scalar CV model names: {duplicates}")

    fold_target_counts: dict[int, dict[str, int]] = {}
    if not dry_run and not args.prepare_embeddings_only:
        for fold in fold_specs:
            fold_target_counts[fold.index] = count_valid_target_rows_for_fold(
                data_config=data_config,
                pipeline_config=target_runtime["pipeline"],
                fold=fold,
                random_seed=int(target_runtime["experiment"]["random_seed"]),
                target_name=target_name,
            )
        write_fold_summary(
            log_root / "folds.json",
            fold_specs,
            target_name=target_name,
            target_counts=fold_target_counts,
        )

    needs_embedding = any(
        str(model.get("kind")) in {"swin_embedding_head", "swin_embedding_head_eval"} for model in models
    )
    embedding_cfg: dict[str, Any] | None = None
    embedding_output_path: Path | None = None
    if needs_embedding or args.prepare_embeddings_only:
        embedding_cfg = dict(cfg.get("embedding", {}))
        if not embedding_cfg:
            raise ValueError(
                "City-scalar CV requires an embedding section when using "
                "swin_embedding_head models or --prepare-embeddings-only."
            )
        embedding_cfg["cli_overrides"] = _embedding_cache_cli_overrides(args.overrides)
        embedding_output_path = resolve_embedding_output_path(
            cfg=embedding_cfg,
            data_config=data_config,
            cache_dir=embedding_cache_dir,
            output_prefix=output_prefix,
        )
        ensure_embedding_cache(
            cfg=embedding_cfg,
            data_overrides=args.overrides,
            output_path=embedding_output_path,
            commands_log=commands_log,
            dry_run=dry_run,
        )
    if args.prepare_embeddings_only:
        print(f"Embedding cache path: {embedding_output_path}")
        return 0

    embedding_table: dict[str, np.ndarray] | None = None
    if needs_embedding and not dry_run:
        if embedding_output_path is None:
            raise RuntimeError("Embedding output path was not resolved")
        embedding_table = load_embedding_table(embedding_output_path)

    all_results: list[dict[str, Any]] = []
    results_path = log_root / f"{output_prefix}_results.jsonl"
    fold_metrics_path = log_root / f"{output_prefix}_fold_metrics.csv"
    summary_path = log_root / f"{output_prefix}_summary.csv"
    for variant in models:
        kind = str(variant.get("kind"))
        if kind not in {"swin_embedding_head", "swin_embedding_head_eval", "launcher"}:
            raise ValueError(f"Unsupported city-scalar CV model kind={kind!r} for model={variant.get('name')!r}")
        for fold in fold_specs:
            if kind in {"swin_embedding_head", "swin_embedding_head_eval"}:
                if dry_run:
                    rows = None
                else:
                    if embedding_table is None:
                        raise RuntimeError("Embedding table was not loaded")
                    rows, target_context = load_city_scalar_rows_for_fold(
                        data_config=data_config,
                        pipeline_config=target_runtime["pipeline"],
                        fold=fold,
                        embedding_table=embedding_table,
                        random_seed=int(target_runtime["experiment"]["random_seed"]),
                        target_name=target_name,
                    )
                if kind == "swin_embedding_head_eval":
                    if rows is None:
                        print(f"+ evaluate_pretrained_embedding_head model={variant['name']} fold={fold.index}")
                    else:
                        evaluate_pretrained_embedding_head(
                            variant=variant,
                            fold=fold,
                            rows=rows,
                            target_context=target_context,
                            output_prefix=output_prefix,
                            random_seed=int(target_runtime["experiment"]["random_seed"]),
                            dry_run=dry_run,
                            results=all_results,
                        )
                elif rows is None:
                    print(f"+ train_swin_embedding_head model={variant['name']} fold={fold.index}")
                else:
                    train_swin_embedding_head(
                        variant=variant,
                        fold=fold,
                        rows=rows,
                        target_context=target_context,
                        output_prefix=output_prefix,
                        output_root=Path(experiment_group),
                        random_seed=int(target_runtime["experiment"]["random_seed"]),
                        dry_run=dry_run,
                        results=all_results,
                    )
            else:
                run_launcher_variant(
                    variant=variant,
                    fold=fold,
                    cli_overrides=args.overrides,
                    experiment_group=experiment_group,
                    output_prefix=output_prefix,
                    commands_log=commands_log,
                    dry_run=dry_run,
                    results=all_results,
                    eval_only=bool(variant.get("eval_only", False)),
                )
            if all_results or dry_run:
                persist_results(log_root=log_root, rows=all_results, output_prefix=output_prefix)

    results_path, fold_metrics_path, summary_path = persist_results(
        log_root=log_root,
        rows=all_results,
        output_prefix=output_prefix,
    )
    print(f"Saved city-scalar CV results: {results_path}")
    print(f"Saved city-scalar CV fold metrics: {fold_metrics_path}")
    print(f"Saved city-scalar CV summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
