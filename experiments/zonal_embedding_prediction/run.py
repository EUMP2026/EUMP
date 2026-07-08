#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
except ModuleNotFoundError:
    torch = None
    nn = None
    F = None
    DataLoader = None
    TensorDataset = None


CONFIG_DIR = REPO_ROOT / "configs"
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = EXPERIMENT_DIR / "config.yaml"
SPLIT_ORDER = ("train", "val", "test")
LCZ_COLUMNS = tuple(f"LCZ_{index}" for index in range(1, 18))
IDENTIFIER_COLUMNS = {"id", "ID", "split"}
YEAR_COLUMN_RE = re.compile(r"^(?P<group>.+)_(?P<year>\d{4})$")
EPS = 1.0e-12


def resolve_model_artifact_path(model_name: str, file_name: str | None) -> str | None:
    if file_name is None:
        return None
    if os.path.isabs(str(file_name)):
        return str(file_name)
    return str(REPO_ROOT / "model" / str(model_name) / str(file_name))


@dataclass(frozen=True, slots=True)
class RegressionTargetSpec:
    name: str
    kind: str
    group: str
    column: str | None = None
    year_to_column: dict[int, str] | None = None


@dataclass(frozen=True, slots=True)
class NormalizationStat:
    target: str
    group: str
    kind: str
    mean: float
    std: float
    count: int
    std_was_clamped: bool


@dataclass(frozen=True, slots=True)
class ZonalArrays:
    embeddings: np.ndarray
    regression_targets: np.ndarray
    regression_mask: np.ndarray
    lcz_targets: np.ndarray
    lcz_mask: np.ndarray
    split_names: np.ndarray
    scene_ids: np.ndarray
    years: np.ndarray
    sample_kinds: np.ndarray


if nn is not None:

    class SwinZonalMultitaskHead(nn.Module):
        def __init__(self, *, input_dim: int, hidden_dim: int, dropout: float, regression_dim: int) -> None:
            super().__init__()
            self.trunk = nn.Sequential(
                nn.LayerNorm(int(input_dim)),
                nn.Linear(int(input_dim), int(hidden_dim)),
                nn.GELU(),
                nn.Dropout(float(dropout)),
            )
            self.regression_head = nn.Linear(int(hidden_dim), int(regression_dim))
            self.lcz_head = nn.Linear(int(hidden_dim), len(LCZ_COLUMNS))

        def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            hidden = self.trunk(features)
            return self.regression_head(hidden), self.lcz_head(hidden)

else:

    class SwinZonalMultitaskHead:
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("PyTorch is required to train the zonal embedding prediction head")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Swin-embedding MLP head on zonal CSV targets.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--prepare-embeddings-only", action="store_true")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides forwarded to composed configs and embedding export.")
    return parser.parse_args(argv)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()[:20]


def _hash_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_path(path: Any, *, base: Path = REPO_ROOT) -> Path:
    resolved = Path(str(path)).expanduser()
    return resolved if resolved.is_absolute() else base / resolved


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


def load_experiment_config(path: Path) -> dict[str, Any]:
    try:
        from omegaconf import OmegaConf
    except ModuleNotFoundError as exc:
        raise RuntimeError("OmegaConf is required to read the zonal experiment config") from exc
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(payload, dict):
        raise TypeError(f"Experiment config must be a mapping: {path}")
    config = payload.get("zonal_embedding_prediction", payload)
    if not isinstance(config, dict):
        raise TypeError(f"zonal_embedding_prediction config must be a mapping: {path}")
    return config


def compose_runtime_config(*, pipeline: str, overrides: list[str]) -> dict[str, Any]:
    try:
        from hydra import compose, initialize_config_dir
    except ModuleNotFoundError as exc:
        raise RuntimeError("Hydra is required to compose the EUMP runtime config") from exc
    from urban_embedding.training.runner import normalize_runtime_config

    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="train", overrides=[f"pipeline={pipeline}", *overrides])
    return normalize_runtime_config(cfg)


def resolve_year_range(value: Any, *, default: tuple[int, int]) -> tuple[int, int]:
    values = _as_list(value)
    if not values:
        return default
    if len(values) != 2:
        raise ValueError(f"Year range must have exactly two values, got {value!r}")
    start, end = int(values[0]), int(values[1])
    if start > end:
        raise ValueError(f"Year range start must be <= end, got {(start, end)!r}")
    return start, end


def inclusive_years(value: Any, *, default: tuple[int, int]) -> tuple[int, ...]:
    start, end = resolve_year_range(value, default=default)
    return tuple(range(start, end + 1))


def normalize_scene_id(value: Any) -> str:
    text = str(value).strip()
    if text == "":
        raise ValueError("Empty scene ID")
    try:
        as_float = float(text)
    except ValueError:
        return text
    if math.isfinite(as_float) and as_float.is_integer():
        return str(int(as_float))
    return text


def read_zonal_csv(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, encoding="utf-8-sig", dtype={"ID": str, "id": str, "split": str})
    required = {"ID", "split"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"Zonal CSV {path} is missing required columns: {missing}")
    missing_lcz = [column for column in LCZ_COLUMNS if column not in table.columns]
    if missing_lcz:
        raise ValueError(f"Zonal CSV {path} is missing LCZ distribution columns: {missing_lcz}")
    table = table.copy()
    table["ID"] = table["ID"].map(normalize_scene_id)
    duplicate_ids = sorted(table.loc[table["ID"].duplicated(), "ID"].unique().tolist())
    if duplicate_ids:
        raise ValueError(f"Zonal CSV {path} contains duplicate ID values: {duplicate_ids[:20]}")
    table["split"] = table["split"].astype(str).str.strip().str.lower()
    unknown_splits = sorted(set(table["split"]).difference(SPLIT_ORDER))
    if unknown_splits:
        raise ValueError(f"Zonal CSV {path} has unsupported split values: {unknown_splits}")
    return table


def numeric_series(table: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(table[column], errors="coerce")


def finite_values(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64, copy=True)
    return values[np.isfinite(values)]


def is_numeric_target_column(table: pd.DataFrame, column: str) -> bool:
    return finite_values(table[column]).size > 0


def split_year_column(column: str) -> tuple[str, int] | None:
    match = YEAR_COLUMN_RE.fullmatch(str(column))
    if match is None:
        return None
    return match.group("group"), int(match.group("year"))


def infer_regression_targets(
    table: pd.DataFrame,
    *,
    dynamic_years: tuple[int, ...],
    exclude_groups: set[str],
) -> tuple[RegressionTargetSpec, ...]:
    static_targets: list[RegressionTargetSpec] = []
    dynamic_by_group: dict[str, dict[int, str]] = {}
    group_order: list[str] = []
    dynamic_year_set = set(int(year) for year in dynamic_years)

    for column in table.columns:
        if column in IDENTIFIER_COLUMNS or column in LCZ_COLUMNS:
            continue
        year_info = split_year_column(column)
        if year_info is None:
            if is_numeric_target_column(table, column):
                static_targets.append(
                    RegressionTargetSpec(name=column, kind="static", group=column, column=column)
                )
            continue

        group, year = year_info
        if group in exclude_groups or year not in dynamic_year_set:
            continue
        if not is_numeric_target_column(table, column):
            continue
        if group not in dynamic_by_group:
            dynamic_by_group[group] = {}
            group_order.append(group)
        dynamic_by_group[group][int(year)] = column

    dynamic_targets = [
        RegressionTargetSpec(
            name=group,
            kind="dynamic",
            group=group,
            year_to_column=dict(sorted(dynamic_by_group[group].items())),
        )
        for group in group_order
    ]
    if not static_targets and not dynamic_targets:
        raise ValueError("No regression targets were selected from the zonal CSV")
    return tuple([*static_targets, *dynamic_targets])


def lcz_distribution_from_row(row: pd.Series) -> np.ndarray | None:
    values: list[float] = []
    for column in LCZ_COLUMNS:
        raw = row.get(column)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value) or value < 0:
            return None
        values.append(value)
    array = np.asarray(values, dtype=np.float32)
    total = float(array.sum())
    if total <= 0:
        return None
    return array / total


def compute_normalization_stats(
    table: pd.DataFrame,
    targets: tuple[RegressionTargetSpec, ...],
) -> dict[str, NormalizationStat]:
    train = table[table["split"] == "train"]
    stats: dict[str, NormalizationStat] = {}
    for target in targets:
        if target.kind == "static":
            assert target.column is not None
            values = finite_values(train[target.column])
        elif target.kind == "dynamic":
            columns = list((target.year_to_column or {}).values())
            pieces = [finite_values(train[column]) for column in columns]
            values = np.concatenate(pieces, axis=0) if pieces else np.empty((0,), dtype=np.float64)
        else:
            raise ValueError(f"Unsupported target kind: {target.kind!r}")

        if values.size <= 0:
            raise ValueError(f"Target {target.name!r} has no finite train values for normalization")
        mean = float(np.mean(values))
        std = float(np.std(values))
        clamped = False
        if not math.isfinite(std) or std <= EPS:
            std = 1.0
            clamped = True
        stats[target.name] = NormalizationStat(
            target=target.name,
            group=target.group,
            kind=target.kind,
            mean=mean,
            std=std,
            count=int(values.size),
            std_was_clamped=clamped,
        )
    return stats


def target_manifest_payload(
    *,
    table: pd.DataFrame,
    targets: tuple[RegressionTargetSpec, ...],
    stats: dict[str, NormalizationStat],
    dynamic_years: tuple[int, ...],
    static_embedding_year: int,
    exclude_groups: set[str],
) -> dict[str, Any]:
    split_counts = {split: int((table["split"] == split).sum()) for split in SPLIT_ORDER}
    target_entries: list[dict[str, Any]] = []
    for target in targets:
        if target.kind == "static":
            valid_count = int(finite_values(table[target.column]).size) if target.column is not None else 0
            target_entries.append(
                {
                    "name": target.name,
                    "kind": target.kind,
                    "group": target.group,
                    "column": target.column,
                    "valid_values": valid_count,
                    "normalization": asdict(stats[target.name]),
                }
            )
        else:
            year_to_column = target.year_to_column or {}
            valid_by_year = {
                str(year): int(finite_values(table[column]).size)
                for year, column in year_to_column.items()
            }
            target_entries.append(
                {
                    "name": target.name,
                    "kind": target.kind,
                    "group": target.group,
                    "years": sorted(int(year) for year in year_to_column),
                    "year_to_column": {str(year): column for year, column in sorted(year_to_column.items())},
                    "missing_years": [
                        int(year) for year in dynamic_years if int(year) not in year_to_column
                    ],
                    "valid_values_by_year": valid_by_year,
                    "normalization": asdict(stats[target.name]),
                }
            )
    return {
        "csv_rows": int(table.shape[0]),
        "split_counts": split_counts,
        "dynamic_years": [int(year) for year in dynamic_years],
        "static_embedding_year": int(static_embedding_year),
        "excluded_groups": sorted(exclude_groups),
        "regression_target_count": len(targets),
        "regression_targets": target_entries,
        "lcz": {
            "name": "LCZ_distribution",
            "kind": "distribution",
            "columns": list(LCZ_COLUMNS),
            "valid_rows": int(sum(lcz_distribution_from_row(row) is not None for _, row in table.iterrows())),
        },
    }


def resolve_embedding_output_path(
    *,
    cfg: dict[str, Any],
    data_config: dict[str, Any],
    cache_dir: Path,
    dynamic_years: tuple[int, ...],
) -> Path:
    checkpoint_path = Path(str(resolve_model_artifact_path(str(cfg["model_name"]), str(cfg["checkpoint_file"]))))
    preprocessing_path = _resolve_path(data_config.get("preprocessing_path", data_config.get("eump_root")))
    eump_root = _resolve_path(data_config.get("eump_root"))
    payload = {
        "kind": "zonal_prediction_swin_pooled_embedding_cache",
        "model_name": str(cfg["model_name"]),
        "checkpoint_file": str(cfg["checkpoint_file"]),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _hash_file(checkpoint_path),
        "checkpoint_source": str(cfg.get("checkpoint_source", "auto")),
        "pipeline": str(cfg["pipeline"]),
        "eump_root": str(eump_root),
        "preprocessing_path": str(preprocessing_path),
        "preprocessing_sha256": _hash_file(preprocessing_path),
        "years": [int(min(dynamic_years)), int(max(dynamic_years))],
        "pooled": True,
    }
    return cache_dir / f"zonal_swin_pooled_{_stable_hash(payload)}.npz"


def ensure_embedding_cache(
    *,
    cfg: dict[str, Any],
    data_overrides: list[str],
    output_path: Path,
    dynamic_years: tuple[int, ...],
    commands_log: Path,
    dry_run: bool,
) -> None:
    if output_path.exists() and not bool(cfg.get("overwrite", False)):
        print(f"Reuse Swin embedding cache: {output_path}")
        return
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
        str(int(min(dynamic_years))),
        str(int(max(dynamic_years))),
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
        "data.runtime_split_manifest_path=null",
        *data_overrides,
        f"pipeline={cfg['pipeline']}",
    ]
    if bool(cfg.get("overwrite", False)):
        cmd.insert(-len(data_overrides) - 2, "--overwrite")
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
        raise ValueError(f"Zonal prediction requires pooled [N, C] embeddings, got shape={embeddings.shape}")
    scene_ids = payload["scene_ids"].astype(str, copy=False)
    years = payload["years"].astype(np.int32, copy=False)
    table: dict[str, np.ndarray] = {}
    for idx, (scene_id, year) in enumerate(zip(scene_ids, years)):
        key = f"{normalize_scene_id(scene_id)}:{int(year)}"
        if key in table:
            raise ValueError(f"Duplicate embedding row key in {path}: {key}")
        table[key] = embeddings[idx]
    return table


def _empty_regression_row(target_count: int) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.zeros((int(target_count),), dtype=np.float32),
        np.zeros((int(target_count),), dtype=bool),
    )


def build_zonal_arrays(
    *,
    table: pd.DataFrame,
    targets: tuple[RegressionTargetSpec, ...],
    stats: dict[str, NormalizationStat],
    embedding_table: dict[str, np.ndarray],
    dynamic_years: tuple[int, ...],
    static_embedding_year: int,
) -> ZonalArrays:
    target_index = {target.name: index for index, target in enumerate(targets)}
    embeddings: list[np.ndarray] = []
    regression_targets: list[np.ndarray] = []
    regression_masks: list[np.ndarray] = []
    lcz_targets: list[np.ndarray] = []
    lcz_masks: list[bool] = []
    split_names: list[str] = []
    scene_ids: list[str] = []
    years: list[int] = []
    sample_kinds: list[str] = []
    missing_embedding_keys: list[str] = []

    for _, row in table.iterrows():
        scene_id = normalize_scene_id(row["ID"])
        split_name = str(row["split"])

        static_values, static_mask = _empty_regression_row(len(targets))
        for target in targets:
            if target.kind != "static":
                continue
            assert target.column is not None
            raw_value = pd.to_numeric(pd.Series([row.get(target.column)]), errors="coerce").iloc[0]
            if pd.isna(raw_value) or not math.isfinite(float(raw_value)):
                continue
            stat = stats[target.name]
            static_values[target_index[target.name]] = (float(raw_value) - stat.mean) / stat.std
            static_mask[target_index[target.name]] = True

        lcz_target = lcz_distribution_from_row(row)
        lcz_valid = lcz_target is not None
        if lcz_target is None:
            lcz_target = np.zeros((len(LCZ_COLUMNS),), dtype=np.float32)

        if bool(static_mask.any()) or lcz_valid:
            key = f"{scene_id}:{int(static_embedding_year)}"
            embedding = embedding_table.get(key)
            if embedding is None:
                missing_embedding_keys.append(key)
            else:
                embeddings.append(embedding)
                regression_targets.append(static_values)
                regression_masks.append(static_mask)
                lcz_targets.append(lcz_target.astype(np.float32, copy=False))
                lcz_masks.append(bool(lcz_valid))
                split_names.append(split_name)
                scene_ids.append(scene_id)
                years.append(int(static_embedding_year))
                sample_kinds.append("static")

        for year in dynamic_years:
            values, mask = _empty_regression_row(len(targets))
            for target in targets:
                if target.kind != "dynamic":
                    continue
                column = (target.year_to_column or {}).get(int(year))
                if column is None:
                    continue
                raw_value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
                if pd.isna(raw_value) or not math.isfinite(float(raw_value)):
                    continue
                stat = stats[target.name]
                values[target_index[target.name]] = (float(raw_value) - stat.mean) / stat.std
                mask[target_index[target.name]] = True

            if not bool(mask.any()):
                continue
            key = f"{scene_id}:{int(year)}"
            embedding = embedding_table.get(key)
            if embedding is None:
                missing_embedding_keys.append(key)
                continue
            embeddings.append(embedding)
            regression_targets.append(values)
            regression_masks.append(mask)
            lcz_targets.append(np.zeros((len(LCZ_COLUMNS),), dtype=np.float32))
            lcz_masks.append(False)
            split_names.append(split_name)
            scene_ids.append(scene_id)
            years.append(int(year))
            sample_kinds.append("dynamic")

    if missing_embedding_keys:
        unique_missing = list(dict.fromkeys(missing_embedding_keys))
        preview = ", ".join(unique_missing[:10])
        suffix = "" if len(unique_missing) <= 10 else f", ... ({len(unique_missing)} total)"
        raise ValueError(f"Missing embedding rows for zonal samples: {preview}{suffix}")
    if not embeddings:
        raise ValueError("No valid zonal samples were joined with Swin embeddings")

    return ZonalArrays(
        embeddings=np.stack(embeddings, axis=0).astype(np.float32, copy=False),
        regression_targets=np.stack(regression_targets, axis=0).astype(np.float32, copy=False),
        regression_mask=np.stack(regression_masks, axis=0).astype(bool, copy=False),
        lcz_targets=np.stack(lcz_targets, axis=0).astype(np.float32, copy=False),
        lcz_mask=np.asarray(lcz_masks, dtype=bool),
        split_names=np.asarray(split_names, dtype=str),
        scene_ids=np.asarray(scene_ids, dtype=str),
        years=np.asarray(years, dtype=np.int32),
        sample_kinds=np.asarray(sample_kinds, dtype=str),
    )


def split_tensor_dataset(arrays: ZonalArrays, split_name: str) -> tuple[TensorDataset, np.ndarray]:
    mask = arrays.split_names == split_name
    if int(mask.sum()) <= 0:
        raise ValueError(f"Zonal split {split_name!r} has no valid samples")
    dataset = TensorDataset(
        torch.as_tensor(arrays.embeddings[mask], dtype=torch.float32),
        torch.as_tensor(arrays.regression_targets[mask], dtype=torch.float32),
        torch.as_tensor(arrays.regression_mask[mask], dtype=torch.bool),
        torch.as_tensor(arrays.lcz_targets[mask], dtype=torch.float32),
        torch.as_tensor(arrays.lcz_mask[mask], dtype=torch.bool),
    )
    return dataset, mask


def compute_loss(
    *,
    regression_pred: torch.Tensor,
    lcz_logits: torch.Tensor,
    regression_target: torch.Tensor,
    regression_mask: torch.Tensor,
    lcz_target: torch.Tensor,
    lcz_mask: torch.Tensor,
    regression_weight: float,
    lcz_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    branch_losses: list[torch.Tensor] = []
    branch_weights: list[float] = []
    report: dict[str, float] = {}

    if bool(regression_mask.any().detach().cpu()) and float(regression_weight) > 0:
        mask = regression_mask.to(dtype=regression_pred.dtype)
        reg_loss = (torch.abs(regression_pred - regression_target) * mask).sum() / mask.sum().clamp_min(1.0)
        branch_losses.append(reg_loss * float(regression_weight))
        branch_weights.append(float(regression_weight))
        report["loss/regression"] = float(reg_loss.detach().cpu().item())
    else:
        report["loss/regression"] = math.nan

    if bool(lcz_mask.any().detach().cpu()) and float(lcz_weight) > 0:
        selected_logits = lcz_logits[lcz_mask]
        selected_target = lcz_target[lcz_mask]
        log_prob = F.log_softmax(selected_logits, dim=1)
        lcz_loss = -(selected_target * log_prob).sum(dim=1).mean()
        branch_losses.append(lcz_loss * float(lcz_weight))
        branch_weights.append(float(lcz_weight))
        report["loss/lcz"] = float(lcz_loss.detach().cpu().item())
    else:
        report["loss/lcz"] = math.nan

    if not branch_losses:
        total = regression_pred.sum() * 0.0
    else:
        total = sum(branch_losses) / max(EPS, sum(branch_weights))
    report["loss/total"] = float(total.detach().cpu().item())
    return total, report


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


def lcz_metrics(logits: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    if target.shape[0] <= 0:
        return {
            "count": 0,
            "ce": None,
            "kl": None,
            "dist_mae": None,
            "dist_rmse": None,
            "top1_acc": None,
        }
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    prob = exp / np.clip(exp.sum(axis=1, keepdims=True), a_min=EPS, a_max=None)
    log_prob = np.log(np.clip(prob, a_min=EPS, a_max=1.0))
    ce = -np.sum(target * log_prob, axis=1)
    safe_target = np.clip(target, a_min=EPS, a_max=1.0)
    kl = np.sum(target * (np.log(safe_target) - log_prob), axis=1)
    diff = prob - target
    return {
        "count": int(target.shape[0]),
        "ce": float(np.mean(ce)),
        "kl": float(np.mean(kl)),
        "dist_mae": float(np.mean(np.abs(diff))),
        "dist_rmse": float(math.sqrt(float(np.mean(diff ** 2)))),
        "top1_acc": float(np.mean(np.argmax(prob, axis=1) == np.argmax(target, axis=1))),
    }


def evaluate_model(
    *,
    model: nn.Module,
    dataset: TensorDataset,
    batch_size: int,
    device: torch.device,
    regression_weight: float,
    lcz_weight: float,
) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False)
    model.eval()
    total_weighted_loss = 0.0
    sample_count = 0
    regression_predictions: list[np.ndarray] = []
    lcz_logits_parts: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            x, reg_target, reg_mask, lcz_target, lcz_mask = (item.to(device) for item in batch)
            reg_pred, lcz_logits = model(x)
            loss, _ = compute_loss(
                regression_pred=reg_pred,
                lcz_logits=lcz_logits,
                regression_target=reg_target,
                regression_mask=reg_mask,
                lcz_target=lcz_target,
                lcz_mask=lcz_mask,
                regression_weight=regression_weight,
                lcz_weight=lcz_weight,
            )
            total_weighted_loss += float(loss.detach().cpu().item()) * int(x.shape[0])
            sample_count += int(x.shape[0])
            regression_predictions.append(reg_pred.detach().cpu().numpy())
            lcz_logits_parts.append(lcz_logits.detach().cpu().numpy())
    return {
        "loss_total": total_weighted_loss / max(1, sample_count),
        "regression_pred_norm": np.concatenate(regression_predictions, axis=0),
        "lcz_logits": np.concatenate(lcz_logits_parts, axis=0),
    }


def split_metrics(
    *,
    split_name: str,
    targets: tuple[RegressionTargetSpec, ...],
    stats: dict[str, NormalizationStat],
    dataset: TensorDataset,
    evaluation: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _, reg_target, reg_mask, lcz_target, lcz_mask = (tensor.detach().cpu().numpy() for tensor in dataset.tensors)
    reg_pred_norm = evaluation["regression_pred_norm"]
    rows: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {"loss_total": float(evaluation["loss_total"]), "regression": {}, "lcz": {}}

    for index, target in enumerate(targets):
        valid = reg_mask[:, index].astype(bool)
        stat = stats[target.name]
        pred_raw = reg_pred_norm[valid, index] * stat.std + stat.mean
        target_raw = reg_target[valid, index] * stat.std + stat.mean
        metric = regression_metrics(pred_raw, target_raw)
        metrics["regression"][target.name] = metric
        rows.append(
            {
                "split": split_name,
                "target": target.name,
                "kind": target.kind,
                "count": metric["count"],
                "mae": metric["mae"],
                "mse": metric["mse"],
                "rmse": metric["rmse"],
                "r2": metric["r2"],
                "ce": "",
                "kl": "",
                "dist_mae": "",
                "dist_rmse": "",
                "top1_acc": "",
            }
        )

    valid_lcz = lcz_mask.astype(bool)
    lcz_metric = lcz_metrics(evaluation["lcz_logits"][valid_lcz], lcz_target[valid_lcz])
    metrics["lcz"] = lcz_metric
    rows.append(
        {
            "split": split_name,
            "target": "LCZ_distribution",
            "kind": "distribution",
            "count": lcz_metric["count"],
            "mae": "",
            "mse": "",
            "rmse": "",
            "r2": "",
            "ce": lcz_metric["ce"],
            "kl": lcz_metric["kl"],
            "dist_mae": lcz_metric["dist_mae"],
            "dist_rmse": lcz_metric["dist_rmse"],
            "top1_acc": lcz_metric["top1_acc"],
        }
    )
    return metrics, rows


def write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "split",
        "target",
        "kind",
        "count",
        "mae",
        "mse",
        "rmse",
        "r2",
        "ce",
        "kl",
        "dist_mae",
        "dist_rmse",
        "top1_acc",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def save_predictions(
    *,
    path: Path,
    arrays: ZonalArrays,
    split_mask: np.ndarray,
    targets: tuple[RegressionTargetSpec, ...],
    stats: dict[str, NormalizationStat],
    evaluation: dict[str, Any],
) -> None:
    reg_pred_norm = evaluation["regression_pred_norm"].astype(np.float32, copy=False)
    reg_target_norm = arrays.regression_targets[split_mask].astype(np.float32, copy=False)
    means = np.asarray([stats[target.name].mean for target in targets], dtype=np.float32)
    stds = np.asarray([stats[target.name].std for target in targets], dtype=np.float32)
    reg_pred_raw = reg_pred_norm * stds.reshape(1, -1) + means.reshape(1, -1)
    reg_target_raw = reg_target_norm * stds.reshape(1, -1) + means.reshape(1, -1)
    logits = evaluation["lcz_logits"].astype(np.float32, copy=False)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    prob = np.exp(shifted)
    prob = prob / np.clip(prob.sum(axis=1, keepdims=True), a_min=EPS, a_max=None)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        scene_ids=arrays.scene_ids[split_mask],
        years=arrays.years[split_mask],
        sample_kinds=arrays.sample_kinds[split_mask],
        target_names=np.asarray([target.name for target in targets], dtype=np.str_),
        regression_pred_norm=reg_pred_norm,
        regression_target_norm=reg_target_norm,
        regression_pred_raw=reg_pred_raw.astype(np.float32, copy=False),
        regression_target_raw=reg_target_raw.astype(np.float32, copy=False),
        regression_mask=arrays.regression_mask[split_mask],
        lcz_columns=np.asarray(LCZ_COLUMNS, dtype=np.str_),
        lcz_prob=prob.astype(np.float32, copy=False),
        lcz_target=arrays.lcz_targets[split_mask],
        lcz_mask=arrays.lcz_mask[split_mask],
    )


def train_model(
    *,
    cfg: dict[str, Any],
    arrays: ZonalArrays,
    targets: tuple[RegressionTargetSpec, ...],
    stats: dict[str, NormalizationStat],
    run_dir: Path,
    runtime_config: dict[str, Any],
) -> dict[str, Any]:
    require_torch()
    from urban_embedding.training.schedulers import (
        build_warmup_cosine_scheduler,
        normalize_warmup_cosine_scheduler_config,
    )

    training_cfg = dict(cfg.get("training", {}))
    model_cfg = dict(cfg.get("model", {}))
    seed = int(runtime_config.get("experiment", {}).get("random_seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_dataset, train_mask = split_tensor_dataset(arrays, "train")
    val_dataset, val_mask = split_tensor_dataset(arrays, "val")
    test_dataset, test_mask = split_tensor_dataset(arrays, "test")

    model = SwinZonalMultitaskHead(
        input_dim=int(arrays.embeddings.shape[1]),
        hidden_dim=int(model_cfg.get("hidden_dim", 256)),
        dropout=float(model_cfg.get("dropout", 0.3)),
        regression_dim=len(targets),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.get("lr", 1.0e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1.0e-2)),
    )
    batch_size = int(training_cfg.get("batch_size", 256))
    epochs = int(training_cfg.get("epochs", 100))
    regression_weight = float(training_cfg.get("regression_weight", 1.0))
    lcz_weight = float(training_cfg.get("lcz_weight", 1.0))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    scheduler_cfg = normalize_warmup_cosine_scheduler_config(training_cfg.get("scheduler"))
    scheduler, scheduler_metadata = build_warmup_cosine_scheduler(
        optimizer=optimizer,
        base_lr=float(training_cfg.get("lr", 1.0e-4)),
        num_batches=max(1, len(train_loader)),
        accumulation_steps=1,
        epochs=epochs,
        warmup_fraction=scheduler_cfg.warmup_fraction,
        lr_start_factor=scheduler_cfg.lr_start_factor,
        lr_end_factor=scheduler_cfg.lr_end_factor,
    )

    model_dir = run_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / "swin_zonal_multitask_head.pt"
    best_metric = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    log_every = max(1, int(training_cfg.get("log_every_n_epochs", 10)))

    for epoch in range(1, epochs + 1):
        model.train()
        weighted_loss = 0.0
        sample_count = 0
        for batch in train_loader:
            x, reg_target, reg_mask, lcz_target, lcz_mask = (item.to(device) for item in batch)
            optimizer.zero_grad()
            reg_pred, lcz_logits = model(x)
            loss, _ = compute_loss(
                regression_pred=reg_pred,
                lcz_logits=lcz_logits,
                regression_target=reg_target,
                regression_mask=reg_mask,
                lcz_target=lcz_target,
                lcz_mask=lcz_mask,
                regression_weight=regression_weight,
                lcz_weight=lcz_weight,
            )
            loss.backward()
            grad_clip = training_cfg.get("grad_clip_norm", 1.0)
            if grad_clip is not None and float(grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
            optimizer.step()
            scheduler.step()
            weighted_loss += float(loss.detach().cpu().item()) * int(x.shape[0])
            sample_count += int(x.shape[0])

        train_loss = weighted_loss / max(1, sample_count)
        val_eval = evaluate_model(
            model=model,
            dataset=val_dataset,
            batch_size=batch_size,
            device=device,
            regression_weight=regression_weight,
            lcz_weight=lcz_weight,
        )
        val_loss = float(val_eval["loss_total"])
        if val_loss < best_metric:
            best_metric = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save(
                {
                    "model_state_dict": best_state,
                    "model": {
                        "kind": "swin_zonal_multitask_head",
                        "input_dim": int(arrays.embeddings.shape[1]),
                        "hidden_dim": int(model_cfg.get("hidden_dim", 256)),
                        "dropout": float(model_cfg.get("dropout", 0.3)),
                        "regression_dim": len(targets),
                        "lcz_dim": len(LCZ_COLUMNS),
                    },
                    "target_names": [target.name for target in targets],
                    "normalization_stats": {name: asdict(stat) for name, stat in stats.items()},
                    "best_epoch": int(best_epoch),
                    "best_metric": float(best_metric),
                    "scheduler": scheduler_metadata,
                    "config": cfg,
                },
                checkpoint_path,
            )
        if epoch == 1 or epoch == epochs or epoch % log_every == 0:
            print(
                "Zonal embedding train "
                f"epoch={epoch}/{epochs} train_loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f} best_val_loss={best_metric:.6f}"
            )

    if best_state is None:
        raise RuntimeError("No best checkpoint was selected")
    model.load_state_dict(best_state)

    metrics_payload: dict[str, Any] = {
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_metric),
        "splits": {},
    }
    metric_rows: list[dict[str, Any]] = []
    for split_name, dataset, split_mask in (
        ("train", train_dataset, train_mask),
        ("val", val_dataset, val_mask),
        ("test", test_dataset, test_mask),
    ):
        evaluation = evaluate_model(
            model=model,
            dataset=dataset,
            batch_size=batch_size,
            device=device,
            regression_weight=regression_weight,
            lcz_weight=lcz_weight,
        )
        split_payload, rows = split_metrics(
            split_name=split_name,
            targets=targets,
            stats=stats,
            dataset=dataset,
            evaluation=evaluation,
        )
        metrics_payload["splits"][split_name] = split_payload
        metric_rows.extend(rows)
        save_predictions(
            path=run_dir / f"predictions_{split_name}.npz",
            arrays=arrays,
            split_mask=split_mask,
            targets=targets,
            stats=stats,
            evaluation=evaluation,
        )

    (run_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    write_metrics_csv(run_dir / "metrics.csv", metric_rows)
    return metrics_payload


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def save_yaml(path: Path, payload: Any) -> None:
    try:
        from omegaconf import OmegaConf
    except ModuleNotFoundError as exc:
        raise RuntimeError("OmegaConf is required to write runtime_config.yaml") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(OmegaConf.to_yaml(OmegaConf.create(payload), resolve=True), encoding="utf-8")


def require_torch() -> None:
    if torch is None or nn is None or F is None or DataLoader is None or TensorDataset is None:
        raise RuntimeError(
            "PyTorch is required for this operation. Use the jwk_EUMP environment on the server "
            "for embedding export and head training."
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    os.chdir(REPO_ROOT)
    dry_run = os.environ.get("DRY_RUN", "0") == "1"
    cfg = load_experiment_config(args.config)
    dynamic_years = inclusive_years(cfg.get("dynamic_years"), default=(2003, 2020))
    static_embedding_year = int(cfg.get("static_embedding_year", 2018))
    if static_embedding_year not in dynamic_years:
        raise ValueError(
            f"static_embedding_year={static_embedding_year} must be inside dynamic_years={dynamic_years}"
        )
    embedding_cfg = dict(cfg.get("embedding", {}))
    runtime_config = compose_runtime_config(pipeline=str(embedding_cfg["pipeline"]), overrides=args.overrides)

    output_root = _resolve_path(cfg.get("output_root", EXPERIMENT_DIR / "outputs"))
    run_name = str(cfg.get("run_name", "zonal_embedding_mae_v2"))
    run_dir = output_root / run_name
    commands_log = run_dir / "commands.log"
    if commands_log.exists() and not dry_run:
        commands_log.unlink()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "cache" / "swin_embeddings").mkdir(parents=True, exist_ok=True)

    csv_path = _resolve_path(cfg.get("csv_path", "data/csv_data/zonal_merged_2020.csv"))
    zonal_table = read_zonal_csv(csv_path)
    exclude_groups = {str(group) for group in _as_list(cfg.get("exclude_groups"))}
    targets = infer_regression_targets(
        zonal_table,
        dynamic_years=dynamic_years,
        exclude_groups=exclude_groups,
    )
    stats = compute_normalization_stats(zonal_table, targets)
    manifest = target_manifest_payload(
        table=zonal_table,
        targets=targets,
        stats=stats,
        dynamic_years=dynamic_years,
        static_embedding_year=static_embedding_year,
        exclude_groups=exclude_groups,
    )
    save_yaml(run_dir / "runtime_config.yaml", runtime_config)
    save_json(run_dir / "target_manifest.json", manifest)
    save_json(run_dir / "normalization_stats.json", {name: asdict(stat) for name, stat in stats.items()})

    embedding_output_path = resolve_embedding_output_path(
        cfg=embedding_cfg,
        data_config=runtime_config["data"],
        cache_dir=run_dir / "cache" / "swin_embeddings",
        dynamic_years=dynamic_years,
    )
    ensure_embedding_cache(
        cfg=embedding_cfg,
        data_overrides=args.overrides,
        output_path=embedding_output_path,
        dynamic_years=dynamic_years,
        commands_log=commands_log,
        dry_run=dry_run,
    )
    if args.prepare_embeddings_only:
        print(f"Embedding cache path: {embedding_output_path}")
        return 0
    if dry_run:
        print(f"Dry run ready. Output directory: {run_dir}")
        print(
            "Selected targets: "
            f"regression={len(targets)} lcz_distribution=1 "
            f"static_year={static_embedding_year} dynamic_years={dynamic_years[0]}-{dynamic_years[-1]}"
        )
        return 0

    require_torch()
    embedding_table = load_embedding_table(embedding_output_path)
    arrays = build_zonal_arrays(
        table=zonal_table,
        targets=targets,
        stats=stats,
        embedding_table=embedding_table,
        dynamic_years=dynamic_years,
        static_embedding_year=static_embedding_year,
    )
    print(
        "Prepared zonal embedding dataset: "
        f"samples={arrays.embeddings.shape[0]} input_dim={arrays.embeddings.shape[1]} "
        f"regression_targets={len(targets)} lcz_rows={int(arrays.lcz_mask.sum())}"
    )
    metrics = train_model(
        cfg=cfg,
        arrays=arrays,
        targets=targets,
        stats=stats,
        run_dir=run_dir,
        runtime_config=runtime_config,
    )
    print(f"Saved zonal embedding model: {run_dir / 'model' / 'swin_zonal_multitask_head.pt'}")
    print(f"Saved zonal embedding metrics: {run_dir / 'metrics.csv'}")
    print(f"Best val loss: {metrics['best_val_loss']:.6f} at epoch {metrics['best_epoch']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
