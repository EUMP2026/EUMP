#!/usr/bin/env python
"""Collect scatter-plot data for main figure 2 (downstream prediction quality).

Runs inference on the held-out test set for a 3x4 grid of
(model x task) cells and stores per-sample observed vs predicted values
(physical units) into a SINGLE npz file consumed by ``fig2_v1.py``.

Grid
----
Rows (models):
    eump      -> EUMP, two-stage frozen-encoder embedding head (mae_v3)
    eump_e2e  -> Supervised NN, one-stage Swin (supervised, encoder unfrozen)
    hand      -> Handcraft scalar MLP baseline

Columns (tasks):
    lst           -> raster target (point = city-mask mean per city-year)
    precip        -> scalar target (point = per city-year sample)
    tci, pm25     -> scalar targets, 5-fold CV (point = per (city, year) sample)

For the two CV tasks the BEST fold is selected per cell by the highest
test r2 recorded in each fold's ``evaluation/test_metrics.json``.

Three inference paths are used, mirroring the production evaluators:
    * raster / scalar swin downstream  -> EumpDownstreamEvaluator internals
    * EUMP embedding-head CV            -> run_city_scalar_cv helpers
    * handcraft scalar MLP             -> EumpScalarMlpEvaluator internals

This script must run on the server (processed data + checkpoints live there).
Run single-process (no torchrun); it evaluates on one GPU / CPU.

Usage
-----
    python main_figs_v2/fig2/collect_data.py \
        --output main_figs_v2/fig2/fig2_data.npz [--cells lst:eump,pm25:hand]
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from hydra import compose, initialize_config_dir

from urban_embedding.core.paths import MODEL_DIR, resolve_model_artifact_path
from urban_embedding.core.reproducibility import set_reproducible
from urban_embedding.training.eump.registry import RuntimeContext, resolve_evaluation_runtime
from urban_embedding.training.evaluation import normalize_evaluation_config
from urban_embedding.training.regression_metrics import (
    raster_city_mean_metric_tensors,
    scalar_metric_tensors,
)

CONFIG_DIR = REPO_ROOT / "configs"


# --------------------------------------------------------------------------- #
# Grid specification
# --------------------------------------------------------------------------- #

TASKS = ("lst", "precip", "tci", "pm25")
MODELS = ("eump", "eump_e2e", "hand")

# Display names baked into the data file (fig2_v1.py reads these back).
TASK_LABELS = {"lst": "LST", "precip": "Precip", "tci": "TCI", "pm25": "PM2.5"}
MODEL_LABELS = {"eump": "EUMP", "eump_e2e": "Supervised NN", "hand": "Hand"}
CV_TASKS = {"tci", "pm25"}

# CV fold manifest directory (under eump_root/manifest/splits) and prefix per task.
CV_OUTPUT_PREFIX = {"tci": "traffic_cv", "pm25": "pm25_cv"}
CV_FOLD_SEED = 42


@dataclass(frozen=True, slots=True)
class CellSpec:
    task: str
    model: str
    # how to run inference: "raster_downstream", "scalar_downstream",
    # "embedding_head_cv", "scalar_mlp"
    path: str
    pipeline: str | None          # hydra pipeline= override (None for embedding_head_cv)
    model_name: str               # model dir; "{fold}" placeholder for CV cells
    pretrain_model: str = "swin_L"
    one_stage: bool = False
    encoder_model_name: str | None = None  # for two-stage frozen LST/Precip
    # CV-only: target pipeline for embedding-head row + target name
    cv_target_pipeline: str | None = None
    cv_target_name: str | None = None
    cv_embedding_pipeline: str | None = None
    cv_embedding_model_name: str | None = None
    cv_embedding_years: tuple[int, int] | None = None
    cv_head_dir_name: str | None = None  # fold dir base name, "{fold}" appended
    # scalar-MLP only: feature standardization the checkpoint was trained with.
    # Non-CV mlp_lst/mlp_precip used the pipeline default (train_split_zscore);
    # the CV scalar_mlp baselines used the launcher override feature_cache_train_split.
    scalar_mlp_feature_standardization: str | None = None


def build_grid() -> dict[tuple[str, str], CellSpec]:
    grid: dict[tuple[str, str], CellSpec] = {}

    # ---- LST raster / Precip scalar downstream ---------------------------- #
    grid[("lst", "eump")] = CellSpec(
        task="lst", model="eump", path="raster_downstream",
        pipeline="eump_full/mae_v3/lst",
        model_name="eump_full_mae_v3/lst",
        encoder_model_name="eump_full_mae_v3/pretrain",
    )
    grid[("lst", "eump_e2e")] = CellSpec(
        task="lst", model="eump_e2e", path="raster_downstream",
        pipeline="eump_full/mae_v3_one_stage/lst",
        model_name="eump_ablation_v2_one_stage_swin_l_unfrozen/lst",
        one_stage=True,
    )
    grid[("lst", "hand")] = CellSpec(
        task="lst", model="hand", path="scalar_mlp",
        pipeline="eump_full/scalar_mlp/lst",
        model_name="eump_scalar_mlp/mlp_lst",
    )

    grid[("precip", "eump")] = CellSpec(
        task="precip", model="eump", path="scalar_downstream",
        pipeline="eump_full/mae_v3/precip",
        model_name="eump_full_mae_v3/precip",
        encoder_model_name="eump_full_mae_v3/pretrain",
    )
    grid[("precip", "eump_e2e")] = CellSpec(
        task="precip", model="eump_e2e", path="scalar_downstream",
        pipeline="eump_full/mae_v3_one_stage/precip",
        model_name="eump_ablation_v2_one_stage_swin_l_unfrozen/precip",
        one_stage=True,
    )
    grid[("precip", "hand")] = CellSpec(
        task="precip", model="hand", path="scalar_mlp",
        pipeline="eump_full/scalar_mlp/precip",
        model_name="eump_scalar_mlp/mlp_precip",
    )

    # ---- TCI (traffic) / PM2.5: scalar, 5-fold CV ------------------------- #
    grid[("tci", "eump")] = CellSpec(
        task="tci", model="eump", path="embedding_head_cv",
        pipeline=None,
        model_name="eump_full_mae_v3/traffic_cv/swin_embed_head_128_d05_fold{fold}",
        cv_target_pipeline="eump_full/mae_v3/traffic",
        cv_target_name="Traffic",
        cv_embedding_pipeline="eump_full/mae_v3/pretrain",
        cv_embedding_model_name="eump_full_mae_v3/pretrain",
        cv_embedding_years=(2017, 2020),
        cv_head_dir_name="swin_embed_head_128_d05",
    )
    grid[("tci", "eump_e2e")] = CellSpec(
        task="tci", model="eump_e2e", path="scalar_downstream",
        pipeline="eump_full/mae_v3_one_stage/traffic",
        model_name=(
            "eump_ablation_v2_one_stage_swin_l_unfrozen/traffic_cv/"
            "one_stage_swin_l_unfrozen_traffic_fold{fold}"
        ),
        one_stage=True,
    )
    grid[("tci", "hand")] = CellSpec(
        task="tci", model="hand", path="scalar_mlp",
        pipeline="eump_full/scalar_mlp/traffic",
        model_name="eump_scalar_mlp/traffic_cv/scalar_mlp_128_d05_fold{fold}",
        scalar_mlp_feature_standardization="feature_cache_train_split",
    )

    grid[("pm25", "eump")] = CellSpec(
        task="pm25", model="eump", path="embedding_head_cv",
        pipeline=None,
        model_name="eump_full_mae_v3/pm25_cv/swin_embed_head_128_d05_fold{fold}",
        cv_target_pipeline="eump_full/mae_v3/pm25",
        cv_target_name="PM25_mean",
        cv_embedding_pipeline="eump_full/mae_v3/pretrain",
        cv_embedding_model_name="eump_full_mae_v3/pretrain",
        cv_embedding_years=(2015, 2019),
        cv_head_dir_name="swin_embed_head_128_d05",
    )
    grid[("pm25", "eump_e2e")] = CellSpec(
        task="pm25", model="eump_e2e", path="scalar_downstream",
        pipeline="eump_full/mae_v3_one_stage/pm25",
        model_name=(
            "eump_ablation_v2_one_stage_swin_l_unfrozen/pm25_cv/"
            "one_stage_swin_l_unfrozen_pm25_fold{fold}"
        ),
        one_stage=True,
    )
    grid[("pm25", "hand")] = CellSpec(
        task="pm25", model="hand", path="scalar_mlp",
        pipeline="eump_full/scalar_mlp/pm25",
        model_name="eump_scalar_mlp/pm25_cv/scalar_mlp_128_d05_fold{fold}",
        scalar_mlp_feature_standardization="feature_cache_train_split",
    )

    return grid


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

class _Logger:
    def info(self, *a, **k):
        return None

    def warning(self, *a, **k):
        print("WARN:", *a, file=sys.stderr)


def _device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _compose_eval_cfg(
    *,
    pipeline: str,
    mutate: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Compose the ``evaluate`` config for a pipeline, mutate the raw container,
    then normalize.

    ``experiment.downstream/reconstruct/model_name/scratch_backbone`` are NOT
    set here -- ``normalize_evaluation_config`` infers them from ``pipeline.kind``
    and from ``experiment.group``/``experiment.name``. Setting them via hydra
    overrides fails because the composed ``experiment`` struct lacks those keys.
    """
    from omegaconf import OmegaConf

    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="evaluate", overrides=[f"pipeline={pipeline}"])
    container = OmegaConf.to_container(cfg, resolve=True)
    if mutate is not None:
        mutate(container)
    return normalize_evaluation_config(container)


def _split_group_name(model_name: str) -> tuple[str, str]:
    group, _, name = model_name.partition("/")
    return group, name


def _select_best_fold(model_name_template: str, folds: int = 5) -> tuple[int, dict[str, Any]]:
    """Pick the fold with highest test r2 from its evaluation/test_metrics.json."""
    best_fold = None
    best_r2 = -float("inf")
    per_fold: dict[int, float] = {}
    for fold in range(folds):
        model_name = model_name_template.format(fold=fold)
        metrics_path = Path(resolve_model_artifact_path(model_name, "evaluation/test_metrics.json"))
        if not metrics_path.exists():
            per_fold[fold] = float("nan")
            continue
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics = payload.get("metrics", {})
        scope = metrics.get("global", metrics.get("global_pixels", {}))
        r2 = scope.get("r2")
        per_fold[fold] = float("nan") if r2 is None else float(r2)
        if r2 is not None and float(r2) > best_r2:
            best_r2 = float(r2)
            best_fold = fold
    if best_fold is None:
        raise RuntimeError(f"No fold with valid test r2 for {model_name_template}")
    return best_fold, {"per_fold_test_r2": per_fold, "best_test_r2": best_r2}


def _metrics_from_arrays(y: np.ndarray, yhat: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(y, dtype=np.float64).ravel()
    yhat = np.asarray(yhat, dtype=np.float64).ravel()
    n = int(min(y.size, yhat.size))
    y, yhat = y[:n], yhat[:n]
    resid = yhat - y
    rmse = float(np.sqrt(np.mean(resid ** 2))) if n else float("nan")
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) if n else 0.0
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"count": n, "rmse": rmse, "r2": r2}


# --------------------------------------------------------------------------- #
# Capture loops (one per inference path)
# --------------------------------------------------------------------------- #

def _capture_swin_downstream(evaluator) -> tuple[np.ndarray, np.ndarray]:
    """Run the swin downstream model and return (y, yhat) in physical units.

    Raster targets -> one point per city (city-mean). Scalar targets -> per sample.
    """
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    schema = evaluator.target_schema
    is_raster = evaluator.target_kind == "raster"
    loader = evaluator._selected_loader()
    evaluator.model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(evaluator.device)
            raster_pred, scalar_pred = evaluator.model(batch.raster_inputs, batch.scalar_inputs)
            if is_raster:
                prediction = raster_pred[schema.name]
                block = batch.raster_targets[schema.name]
                mp = evaluator._metric_tensor(prediction)
                mt = evaluator._metric_tensor(block.data)
                p, t = raster_city_mean_metric_tensors(
                    prediction=mp, target=mt,
                    valid_mask=block.valid_mask, city_mask=batch.city_mask,
                    mask_strategy=schema.spec.mask,
                )
            else:
                prediction = scalar_pred[schema.name]
                block = batch.scalar_targets[schema.name]
                mp = evaluator._metric_tensor(prediction)
                mt = evaluator._metric_tensor(block.data)
                p, t = scalar_metric_tensors(
                    prediction=mp, target=mt,
                    valid_mask=block.valid_mask, mask_strategy=schema.spec.mask,
                )
            if p.numel():
                preds.append(p.detach().cpu().numpy().ravel())
                targets.append(t.detach().cpu().numpy().ravel())
    if not preds:
        raise RuntimeError("swin downstream capture produced no samples")
    return np.concatenate(targets), np.concatenate(preds)


def _capture_scalar_mlp(evaluator) -> tuple[np.ndarray, np.ndarray]:
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    loader = evaluator._selected_loader()
    evaluator.model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(evaluator.device)
            prediction = evaluator._physical_metric_tensor(evaluator.model(batch.features))
            target = evaluator._physical_metric_tensor(batch.target)
            preds.append(prediction.detach().cpu().numpy().ravel())
            targets.append(target.detach().cpu().numpy().ravel())
    if not preds:
        raise RuntimeError("scalar mlp capture produced no samples")
    return np.concatenate(targets), np.concatenate(preds)


# --------------------------------------------------------------------------- #
# Per-cell runners
# --------------------------------------------------------------------------- #

def _run_swin_downstream_cell(spec: CellSpec, *, fold: int | None) -> tuple[np.ndarray, np.ndarray, dict]:
    model_name = spec.model_name.format(fold=fold) if fold is not None else spec.model_name
    group, name = _split_group_name(model_name)
    fold_manifest = str(_cv_fold_manifest_path(spec.task, fold)) if fold is not None else None

    def mutate(cfg: dict[str, Any]) -> None:
        exp = cfg.setdefault("experiment", {})
        exp["group"] = group
        exp["name"] = name
        ev = cfg.setdefault("evaluation", {})
        ev["split"] = "test"
        ev["metric_scale"] = "physical"
        ev["checkpoint_source"] = "best_model"
        ev["save_json"] = False
        down = cfg.setdefault("downstream", {})
        # one-stage pipeline configs already set encoder_source=swin_pretrained /
        # freeze_encoder=false; only the two-stage frozen cells need the encoder dir.
        if not spec.one_stage and spec.encoder_model_name is not None:
            down["encoder_model_name"] = spec.encoder_model_name
        if fold_manifest is not None:
            cfg.setdefault("data", {})["runtime_split_manifest_path"] = fold_manifest

    cfg = _compose_eval_cfg(pipeline=spec.pipeline, mutate=mutate)
    set_reproducible(int(cfg["experiment"]["random_seed"]))
    context = RuntimeContext(
        logger=_Logger(), device=_device(), local_rank=0, world_size=1,
        log_dir=str(REPO_ROOT / "log" / "fig2_collect"),
        random_seed=int(cfg["experiment"]["random_seed"]),
    )
    runtime = resolve_evaluation_runtime(cfg)
    evaluator = runtime.evaluator_factory(cfg, context)
    y, yhat = _capture_swin_downstream(evaluator)
    meta = {"model_name": model_name, "checkpoint_path": evaluator.checkpoint_path,
            "target": evaluator.target_schema.name, "target_kind": evaluator.target_kind}
    return y, yhat, meta


def _run_scalar_mlp_cell(spec: CellSpec, *, fold: int | None) -> tuple[np.ndarray, np.ndarray, dict]:
    model_name = spec.model_name.format(fold=fold) if fold is not None else spec.model_name
    group, name = _split_group_name(model_name)
    fold_manifest = str(_cv_fold_manifest_path(spec.task, fold)) if fold is not None else None

    def mutate(cfg: dict[str, Any]) -> None:
        exp = cfg.setdefault("experiment", {})
        exp["group"] = group
        exp["name"] = name
        ev = cfg.setdefault("evaluation", {})
        ev["split"] = "test"
        ev["metric_scale"] = "physical"
        ev["checkpoint_source"] = "best_model"
        ev["save_json"] = False
        if spec.scalar_mlp_feature_standardization is not None:
            cfg.setdefault("downstream", {})["feature_standardization"] = (
                spec.scalar_mlp_feature_standardization
            )
        if fold_manifest is not None:
            cfg.setdefault("data", {})["runtime_split_manifest_path"] = fold_manifest

    cfg = _compose_eval_cfg(pipeline=spec.pipeline, mutate=mutate)
    set_reproducible(int(cfg["experiment"]["random_seed"]))
    context = RuntimeContext(
        logger=_Logger(), device=_device(), local_rank=0, world_size=1,
        log_dir=str(REPO_ROOT / "log" / "fig2_collect"),
        random_seed=int(cfg["experiment"]["random_seed"]),
    )
    runtime = resolve_evaluation_runtime(cfg)
    evaluator = runtime.evaluator_factory(cfg, context)
    y, yhat = _capture_scalar_mlp(evaluator)
    meta = {"model_name": model_name, "checkpoint_path": evaluator.checkpoint_path,
            "target": evaluator.target_schema.name, "target_kind": "city_scalar"}
    return y, yhat, meta


def _cv_fold_manifest_path(task: str, fold: int) -> Path:
    """Resolve the fold manifest produced by run_city_scalar_cv (server side)."""
    eump_root = _eump_root()
    prefix = CV_OUTPUT_PREFIX[task]
    return eump_root / "manifest" / "splits" / f"{prefix}_seed{CV_FOLD_SEED}" / f"fold_{fold}.json"


def _eump_root() -> Path:
    from omegaconf import OmegaConf

    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="evaluate", overrides=["pipeline=eump_full/mae_v3/lst"])
    data = OmegaConf.to_container(cfg, resolve=True)["data"]
    root = Path(str(data["eump_root"]))
    return root if root.is_absolute() else (REPO_ROOT / root)


def _run_embedding_head_cv_cell(spec: CellSpec, *, fold: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """EUMP embedding-head CV: reuse run_city_scalar_cv to score the saved head."""
    import importlib.util

    cv_mod_path = REPO_ROOT / "script" / "experiments" / "run_city_scalar_cv.py"
    mod_name = "_fig2_run_city_scalar_cv"
    cv_spec = importlib.util.spec_from_file_location(mod_name, cv_mod_path)
    cv = importlib.util.module_from_spec(cv_spec)
    # Register before exec so @dataclass(slots=True) can resolve cls.__module__.
    sys.modules[mod_name] = cv
    cv_spec.loader.exec_module(cv)

    target_runtime = cv.compose_runtime_config(
        pipeline=spec.cv_target_pipeline, overrides=["data=eump_full/server_v2"]
    )
    data_config = dict(target_runtime["data"])
    eump_root = _eump_root()

    fold_manifest = _cv_fold_manifest_path(spec.task, fold)
    fold_obj = cv.FoldSpec(index=fold, path=fold_manifest,
                           train_scene_ids=(), val_scene_ids=(), test_scene_ids=())

    embedding_cfg = {
        "model_name": spec.cv_embedding_model_name,
        "checkpoint_file": "eump_reconstruction_best_checkpoint.pt",
        "checkpoint_source": "auto",
        "pipeline": spec.cv_embedding_pipeline,
        "years": list(spec.cv_embedding_years),
        "overrides": [],
        "cli_overrides": [],
    }
    embedding_cache_dir = eump_root / "manifest" / "cache" / "swin_embeddings"
    emb_path = cv.resolve_embedding_output_path(
        cfg=embedding_cfg, data_config=data_config, cache_dir=embedding_cache_dir,
        output_prefix=CV_OUTPUT_PREFIX[spec.task],
    )
    # Generate pooled embeddings from the configured encoder when no matching
    # checkpoint/preprocessing cache exists.
    if not emb_path.exists():
        print(f"   embedding cache missing, regenerating: {emb_path.name}", flush=True)
        commands_log = REPO_ROOT / "log" / "fig2_collect" / "commands.log"
        cv.ensure_embedding_cache(
            cfg=embedding_cfg,
            data_overrides=["data=eump_full/server_v2"],
            output_path=emb_path,
            commands_log=commands_log,
            dry_run=False,
        )
    embedding_table = cv.load_embedding_table(emb_path)

    rows, target_context = cv.load_city_scalar_rows_for_fold(
        data_config=data_config, pipeline_config=target_runtime["pipeline"],
        fold=fold_obj, embedding_table=embedding_table,
        random_seed=int(target_runtime["experiment"]["random_seed"]),
        target_name=spec.cv_target_name,
    )

    head_dir = Path(MODEL_DIR) / spec.model_name.format(fold=fold)
    ckpt = head_dir / "swin_embedding_scalar_head.pt"
    payload = torch.load(ckpt, map_location="cpu")
    head_cfg = dict(payload.get("model", {}))
    device = _device()
    train_x, _, _ = cv.split_tensors(rows, "train", target_name=spec.cv_target_name)
    model = cv.PooledEmbeddingScalarHead(
        input_dim=int(head_cfg.get("input_dim", int(train_x.shape[1]))),
        hidden_dim=int(head_cfg.get("scalar_hidden_dim", 128)),
        dropout=float(head_cfg.get("dropout", 0.5)),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])

    test_x, test_y, _ = cv.split_tensors(rows, "test", target_name=spec.cv_target_name)
    source_spec = dict(target_context["source_spec"])
    model.eval()
    with torch.no_grad():
        pred = model(test_x.to(device)).detach().cpu().numpy()
    yhat = cv.inverse_processed_array(pred, source_spec).ravel()
    y = cv.inverse_processed_array(test_y.detach().cpu().numpy(), source_spec).ravel()
    meta = {"model_name": spec.model_name.format(fold=fold), "checkpoint_path": str(ckpt),
            "target": spec.cv_target_name, "target_kind": "swin_embedding_city_scalar"}
    return y, yhat, meta


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def run_cell(spec: CellSpec) -> dict[str, Any]:
    fold = None
    fold_info: dict[str, Any] = {}
    if spec.task in CV_TASKS:
        # best fold by test r2 of the actual model dir for this cell
        fold, fold_info = _select_best_fold(spec.model_name)

    if spec.path == "raster_downstream" or spec.path == "scalar_downstream":
        y, yhat, meta = _run_swin_downstream_cell(spec, fold=fold)
    elif spec.path == "scalar_mlp":
        y, yhat, meta = _run_scalar_mlp_cell(spec, fold=fold)
    elif spec.path == "embedding_head_cv":
        y, yhat, meta = _run_embedding_head_cv_cell(spec, fold=fold)
    else:
        raise ValueError(f"Unknown path {spec.path!r}")

    stats = _metrics_from_arrays(y, yhat)
    meta.update({
        "task": spec.task, "model": spec.model,
        "task_label": TASK_LABELS[spec.task], "model_label": MODEL_LABELS[spec.model],
        "path": spec.path, "fold": fold, **stats, **fold_info,
    })
    return {"y": y.astype(np.float32), "yhat": yhat.astype(np.float32), "meta": meta}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect fig2 scatter data.")
    p.add_argument("--output", type=Path, default=Path(__file__).resolve().with_name("fig2_data.npz"))
    p.add_argument("--cells", default=None,
                   help="Comma-separated task:model subset, e.g. 'lst:eump,pm25:hand'. Default: all 12.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    grid = build_grid()

    if args.cells:
        selected = []
        for token in args.cells.split(","):
            task, model = token.strip().split(":")
            selected.append((task, model))
    else:
        selected = [(t, m) for t in TASKS for m in MODELS]

    arrays: dict[str, np.ndarray] = {}
    metas: list[dict[str, Any]] = []
    for task, model in selected:
        spec = grid[(task, model)]
        print(f"[fig2] collecting task={task} model={model} path={spec.path} ...", flush=True)
        result = run_cell(spec)
        key = f"{task}__{model}"
        arrays[f"{key}__y"] = result["y"]
        arrays[f"{key}__yhat"] = result["yhat"]
        metas.append(result["meta"])
        m = result["meta"]
        print(f"   -> n={m['count']} r2={m['r2']:.4f} rmse={m['rmse']:.4f} fold={m['fold']}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # meta stored as a JSON unicode string array (no pickle) -> release-safe.
    np.savez_compressed(
        args.output,
        meta_json=np.array(json.dumps(metas), dtype="U"),
        tasks=np.array(list(TASKS)),
        models=np.array(list(MODELS)),
        task_labels=np.array([TASK_LABELS[t] for t in TASKS]),
        model_labels=np.array([MODEL_LABELS[m] for m in MODELS]),
        **arrays,
    )
    print(f"[fig2] saved {args.output} ({len(metas)} cells)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
