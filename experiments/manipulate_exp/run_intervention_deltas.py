#!/usr/bin/env python3
"""Forward-inference of precomputed physical intervention deltas → ΔLST / ΔTCI.

Reads ``task7_step3_intervention_deltas_v3.npz`` (5 cities, physical-space delta
maps for NDVI greening and POP change scenarios), applies each delta to the
corresponding input channel in *physical* space, and predicts:

* raster LST via the mae_v3 LST downstream head (shared frozen encoder), and
* scalar TCI (traffic congestion index) via a PooledEmbeddingScalarHead loaded
  from a traffic CV fold checkpoint.

Scenarios (12 total):
    baseline                                   (no delta)
    greening-only:  gradient_dNDVI / uniform_dNDVI / hotspot_dNDVI   (→ ΔLST)
    pop-only:       gradient_dPOP / smooth_dPOP                       (→ ΔTCI, ΔLST side-effect)
    dual:           {gradient,uniform,hotspot} dNDVI × {gradient,smooth} dPOP
                                                                      (→ ΔLST and ΔTCI)

Every saved array is in PHYSICAL units (LST in Kelvin, NDVI as a reflectance
index, POP in persons, TCI as the unitless congestion index). The npz/summary
carry ``value_space="physical"`` tags so downstream consumers never confuse
these with processed/normalized values.

The 5 cities match experiments/planning_exp; the LST head and the traffic fold2
head reproduce the planning_exp baseline predictors, so baseline LST/TCI here
equal the prior planning baselines for these cities.

Usage (server, single GPU):
    conda activate jwk_EUMP
    python experiments/manipulate_exp/run_intervention_deltas.py \
        data=eump_full/server_v2
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hydra import compose, initialize_config_dir

from experiments.manipulate_exp import run as base_run
from experiments.planning_exp.run import _PooledEmbeddingScalarHead
from urban_embedding.core.logging import setup_logger
from urban_embedding.core.reproducibility import set_reproducible
from urban_embedding.data.processed.dataset import build_eump_dataloaders
from urban_embedding.data.processed.pipeline_spec import (
    PipelineSpec,
    get_single_downstream_target_schema,
    validate_eump_downstream_spec,
)
from urban_embedding.training.evaluation import normalize_evaluation_config
from urban_embedding.training.regression_metrics import inverse_processed_tensor


CONFIG_DIR = str(PROJECT_ROOT / "configs")
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_NPZ = EXPERIMENT_DIR / "task7_step3_intervention_deltas_v3.npz"
DEFAULT_OUTPUT_DIR = "experiments/manipulate_exp/outputs/intervention_deltas_v3"
DEFAULT_TRAFFIC_CHECKPOINT = (
    PROJECT_ROOT
    / "model/eump_full_mae_v3/traffic_cv/swin_embed_head_128_d05_fold2/swin_embedding_scalar_head.pt"
)

NDVI_KEYS = ("gradient_dNDVI_single", "uniform_dNDVI_single", "hotspot_dNDVI_single", "gradient_dNDVI_dual", "uniform_dNDVI_dual", "hotspot_dNDVI_dual")
POP_KEYS = ("gradient_dPOP_dual", "smooth_dPOP_dual")

logger = setup_logger(log_dir=None, rank=0)


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

def _build_scenarios() -> list[dict[str, Any]]:
    """Return scenario specs: name, optional ndvi delta key, optional pop delta key."""
    scenarios: list[dict[str, Any]] = [
        {"name": "baseline", "ndvi_key": None, "pop_key": None, "kind": "baseline"},
    ]
    scenarios.append(
        {"name": "gradient_dNDVI_single", "ndvi_key": "gradient_dNDVI_single", "pop_key": None, "kind": "greening_only"}
    )
    scenarios.append(
        {"name": "uniform_dNDVI_single", "ndvi_key": "uniform_dNDVI_single", "pop_key": None, "kind": "greening_only"}
    )
    scenarios.append(
        {"name": "hotspot_dNDVI_single", "ndvi_key": "hotspot_dNDVI_single", "pop_key": None, "kind": "greening_only"}
    )
    scenarios.append(
        {
            "name": f"gradient_dNDVI_dual__gradient_dPOP_dual",
            "ndvi_key": "gradient_dNDVI_dual",
            "pop_key": "gradient_dPOP_dual",
            "kind": "dual",
        }
    )
    scenarios.append(
        {
            "name": f"uniform_dNDVI_dual__smooth_dPOP_dual",
            "ndvi_key": "uniform_dNDVI_dual",
            "pop_key": "smooth_dPOP_dual",
            "kind": "dual",
        }
    )
    scenarios.append(
        {
            "name": f"hotspot_dNDVI_dual__smooth_dPOP_dual",
            "ndvi_key": "hotspot_dNDVI_dual",
            "pop_key": "smooth_dPOP_dual",
            "kind": "dual",
        }
    )
    return scenarios


# ---------------------------------------------------------------------------
# CLI / config
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Forward-inference of precomputed physical intervention deltas."
    )
    parser.add_argument("--input-npz", type=Path, default=DEFAULT_INPUT_NPZ)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--year", type=int, default=2020)
    parser.add_argument("--lst-pipeline", default="eump_full/mae_v3/lst")
    parser.add_argument(
        "--traffic-checkpoint",
        type=Path,
        default=DEFAULT_TRAFFIC_CHECKPOINT,
        help="PooledEmbeddingScalarHead checkpoint for TCI (traffic) prediction.",
    )
    parser.add_argument("--traffic-hidden-dim", type=int, default=128)
    parser.add_argument("--traffic-dropout", type=float, default=0.5)
    parser.add_argument("--ndvi-source", default="NDVI")
    parser.add_argument("--pop-source", default="POP")
    parser.add_argument("--checkpoint-source", choices=("best_model", "training_checkpoint"), default="best_model")
    parser.add_argument("--checkpoint-file", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--mask-mismatch-tol",
        type=float,
        default=0.02,
        help="Max allowed fraction of differing city_mask pixels before aborting.",
    )
    args, overrides = parser.parse_known_args(argv)
    args.overrides = overrides
    return args


def _compose_hydra_config(pipeline: str, overrides: list[str], num_workers: int) -> dict[str, Any]:
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        raw = compose(config_name="evaluate", overrides=[f"pipeline={pipeline}"] + overrides)
    cfg = normalize_evaluation_config(raw)
    cfg["pipeline"]["num_workers"] = num_workers
    cfg["pipeline"]["center_ratio"] = 1.0
    return cfg


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


# ---------------------------------------------------------------------------
# NPZ loading
# ---------------------------------------------------------------------------

def _load_intervention_npz(path: Path) -> dict[str, np.ndarray]:
    path = _resolve_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Intervention NPZ does not exist: {path}")
    required = {"ids", "names", "city_mask", "ocean_mask", *NDVI_KEYS, *POP_KEYS}
    with np.load(str(path), allow_pickle=False) as payload:
        missing = sorted(required.difference(payload.files))
        if missing:
            raise ValueError(f"Intervention NPZ is missing keys: {missing}")
        ids = np.asarray(payload["ids"]).astype(np.int64)
        n = int(ids.shape[0])
        city_mask = np.asarray(payload["city_mask"]).astype(bool)
        ocean_mask = np.asarray(payload["ocean_mask"]).astype(bool)
        if city_mask.shape[0] != n or city_mask.ndim != 3:
            raise ValueError(f"city_mask must be (N,H,W) with N={n}, got {city_mask.shape}")
        result: dict[str, np.ndarray] = {
            "ids": ids,
            "names": np.asarray(payload["names"]).astype(str),
            "city_mask": city_mask,
            "ocean_mask": ocean_mask,
        }
        for key in (*NDVI_KEYS, *POP_KEYS):
            arr = np.asarray(payload[key]).astype(np.float32)
            if arr.shape != city_mask.shape:
                raise ValueError(f"{key} must have shape {city_mask.shape}, got {arr.shape}")
            if not np.isfinite(arr).all():
                raise ValueError(f"{key} contains NaN or Inf")
            result[key] = arr
        return result


# ---------------------------------------------------------------------------
# Perturbation in physical space
# ---------------------------------------------------------------------------

def _apply_physical_delta(
    *,
    raster_inputs: torch.Tensor,
    raster_valid_mask: torch.Tensor,
    channel_index: int,
    source_spec: dict[str, Any],
    delta_physical: torch.Tensor,
    min_value: float | None,
    max_value: float | None,
) -> torch.Tensor:
    """Return a clone of raster_inputs with one channel perturbed by a physical delta map.

    delta_physical is (H, W) in physical units; applied only on valid pixels.
    """
    selected = raster_inputs[:, channel_index : channel_index + 1, :, :]
    selected_valid = raster_valid_mask[:, channel_index : channel_index + 1, :, :]
    source_before = inverse_processed_tensor(selected, source_spec)
    delta = delta_physical.to(device=source_before.device, dtype=source_before.dtype)
    while delta.dim() < source_before.dim():
        delta = delta.unsqueeze(0)
    source_after = source_before.clone()
    valid = selected_valid.bool()
    source_after = torch.where(valid, source_before + delta, source_before)
    if min_value is not None or max_value is not None:
        source_after = torch.clamp(source_after, min=min_value, max=max_value)
    processed_after, _ = base_run._physical_to_processed_tensor(
        source_after, selected_valid, source_spec
    )
    perturbed = raster_inputs.clone()
    perturbed[:, channel_index : channel_index + 1, :, :] = processed_after
    return perturbed


def _physical_channel_map(
    raster_inputs: torch.Tensor,
    channel_index: int,
    source_spec: dict[str, Any],
) -> np.ndarray:
    selected = raster_inputs[:, channel_index : channel_index + 1, :, :]
    phys = inverse_processed_tensor(selected, source_spec)
    return phys.squeeze().detach().cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def _masked_stats(values: np.ndarray, mask: np.ndarray) -> dict[str, float | int | None]:
    valid = mask.astype(bool) & np.isfinite(values)
    if not np.any(valid):
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None, "std": None}
    sel = values[valid]
    return {
        "count": int(sel.size),
        "mean": float(np.mean(sel)),
        "median": float(np.median(sel)),
        "min": float(np.min(sel)),
        "max": float(np.max(sel)),
        "std": float(np.std(sel)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = base_run._resolve_device(str(args.device))
    output_base = base_run._resolve_output_base(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    year = int(args.year)

    payload = _load_intervention_npz(args.input_npz)
    ids = payload["ids"]
    names = payload["names"]
    n_cities = int(ids.shape[0])
    scenarios = _build_scenarios()
    logger.info(
        f"intervention_deltas: cities={ids.tolist()} names={names.tolist()} "
        f"scenarios={len(scenarios)} year={year}"
    )

    hydra_cfg = _compose_hydra_config(str(args.lst_pipeline), args.overrides, int(args.num_workers))
    hydra_cfg["evaluation"]["checkpoint_source"] = str(args.checkpoint_source)
    if args.checkpoint_file:
        hydra_cfg["evaluation"]["checkpoint_file"] = str(args.checkpoint_file)
    random_seed = int(hydra_cfg["experiment"]["random_seed"])
    set_reproducible(random_seed)

    pipeline_spec = PipelineSpec.from_mapping(hydra_cfg["pipeline"])
    validate_eump_downstream_spec(pipeline_spec)

    train_dl, val_dl, test_dl, batch_schema = build_eump_dataloaders(
        data_config=hydra_cfg["data"],
        pipeline_spec=pipeline_spec,
        logger=logger,
        random_seed=random_seed,
        batch_size=1,
        local_rank=0,
        world_size=1,
        is_distributed=False,
        shuffle_train=False,
    )
    loaders = {"train": train_dl, "val": val_dl, "test": test_dl}
    preprocessing_spec = train_dl.dataset.preprocessing_spec
    raster_input_channel_names = tuple(str(c) for c in batch_schema.raster_input_channel_names)

    ch_ndvi, ndvi_name, ndvi_spec = base_run._resolve_source_channel(
        source_name=str(args.ndvi_source),
        preprocessing_spec=preprocessing_spec,
        raster_input_channel_names=raster_input_channel_names,
    )
    ch_pop, pop_name, pop_spec = base_run._resolve_source_channel(
        source_name=str(args.pop_source),
        preprocessing_spec=preprocessing_spec,
        raster_input_channel_names=raster_input_channel_names,
    )

    target_schema = get_single_downstream_target_schema(batch_schema)
    lst_target_name = target_schema.name
    if base_run._target_kind(batch_schema, lst_target_name) != "raster":
        raise ValueError(f"LST pipeline must have a raster target, got {lst_target_name!r}")
    lst_source_name = target_schema.spec.source
    lst_target_spec = preprocessing_spec["sources"][lst_source_name]

    # LST downstream model (shared encoder + raster head).
    lst_model, lst_ckpt_path, _, _ = base_run._build_model_and_load_checkpoint(
        hydra_cfg=hydra_cfg,
        batch_schema=batch_schema,
        target_schema=target_schema,
        device=device,
    )
    lst_model.eval()
    encoder = lst_model.encoder

    # TCI (traffic) scalar head from a CV fold checkpoint.
    traffic_ckpt_path = _resolve_path(Path(str(args.traffic_checkpoint)))
    if not traffic_ckpt_path.exists():
        raise FileNotFoundError(f"Traffic checkpoint does not exist: {traffic_ckpt_path}")
    embedding_channels = encoder.config.embed_dim * (2 ** (encoder.config.num_layers - 1))
    traffic_head = _PooledEmbeddingScalarHead(
        input_dim=embedding_channels,
        hidden_dim=int(args.traffic_hidden_dim),
        dropout=float(args.traffic_dropout),
    ).to(device)
    ckpt = torch.load(str(traffic_ckpt_path), map_location=device)
    traffic_head.load_state_dict(ckpt.get("model_state_dict", ckpt))
    traffic_head.eval()
    logger.info(f"LST checkpoint: {lst_ckpt_path}")
    logger.info(f"TCI checkpoint: {traffic_ckpt_path}")
    logger.info(
        f"Channels: {ndvi_name}={ch_ndvi} ({ndvi_spec['mode']}), "
        f"{pop_name}={ch_pop} ({pop_spec['mode']}); "
        f"LST source={lst_source_name} ({lst_target_spec['mode']})"
    )

    crop_size = pipeline_spec.crop_size
    crop_h, crop_w = (int(crop_size[0]), int(crop_size[1])) if isinstance(crop_size, tuple) else (int(crop_size), int(crop_size))

    # Accumulators (city-ordered to match npz ids).
    out: dict[str, list[Any]] = {scn["name"] + "_lst": [] for scn in scenarios}
    out.update({scn["name"] + "_tci": [] for scn in scenarios})
    out.update({scn["name"] + "_ndvi": [] for scn in scenarios})
    out.update({scn["name"] + "_pop": [] for scn in scenarios})
    valid_masks: list[np.ndarray] = []
    csv_rows: list[dict[str, Any]] = []

    start = time.time()
    with torch.no_grad():
        for i in range(n_cities):
            scene_id = int(ids[i])
            city_name = str(names[i])
            entry = base_run._find_scene_year_entry(
                loaders, scene_id=str(scene_id), year=year, search_order=["test", "val", "train"]
            )
            if entry is None:
                raise ValueError(f"Scene {scene_id} year {year} not found in any split")
            split_name, sample_index = entry
            sample = loaders[split_name].dataset[sample_index]

            raster_inputs = torch.from_numpy(sample.raster_inputs).unsqueeze(0).float().to(device)
            raster_valid = torch.from_numpy(sample.raster_valid_mask.astype(bool)).unsqueeze(0).to(device)
            dl_city_mask = sample.city_mask.squeeze().astype(bool)
            output_size = (int(raster_inputs.shape[-2]), int(raster_inputs.shape[-1]))

            # Guard: npz city_mask must match dataloader crop city_mask.
            npz_city_mask = payload["city_mask"][i]
            if npz_city_mask.shape != dl_city_mask.shape:
                raise ValueError(
                    f"[{scene_id}] city_mask shape mismatch npz={npz_city_mask.shape} "
                    f"dataloader={dl_city_mask.shape}"
                )
            mismatch_frac = float(np.mean(npz_city_mask != dl_city_mask))
            if mismatch_frac > float(args.mask_mismatch_tol):
                raise ValueError(
                    f"[{scene_id}] city_mask differs from dataloader by "
                    f"{mismatch_frac:.4f} (> tol {args.mask_mismatch_tol}); crop misaligned"
                )
            if mismatch_frac > 0:
                logger.warning(f"[{scene_id}] city_mask mismatch fraction {mismatch_frac:.5f} (within tol)")

            # valid pixels across all input channels and the LST target.
            valid_inputs = raster_valid.squeeze(0).all(dim=0).cpu().numpy().astype(bool)
            lst_valid = sample.raster_targets[lst_source_name][1].all(axis=0).squeeze().astype(bool)
            valid_mask = valid_inputs & lst_valid
            valid_masks.append(valid_mask)

            city_mask = npz_city_mask  # use the npz mask as the canonical urban mask

            ndvi_deltas = {k: torch.from_numpy(payload[k][i]).to(device) for k in NDVI_KEYS}
            pop_deltas = {k: torch.from_numpy(payload[k][i]).to(device) for k in POP_KEYS}

            for scn in scenarios:
                cur = raster_inputs
                if scn["ndvi_key"] is not None:
                    cur = _apply_physical_delta(
                        raster_inputs=cur,
                        raster_valid_mask=raster_valid,
                        channel_index=ch_ndvi,
                        source_spec=ndvi_spec,
                        delta_physical=ndvi_deltas[scn["ndvi_key"]],
                        min_value=-1.0,
                        max_value=1.0,
                    )
                if scn["pop_key"] is not None:
                    cur = _apply_physical_delta(
                        raster_inputs=cur,
                        raster_valid_mask=raster_valid,
                        channel_index=ch_pop,
                        source_spec=pop_spec,
                        delta_physical=pop_deltas[scn["pop_key"]],
                        min_value=0.0,
                        max_value=None,
                    )

                embedding = encoder.encode(cur, bool_masked_pos=None)
                lst_pred = list(lst_model.raster_heads.values())[0](embedding, output_size)
                lst_phys = inverse_processed_tensor(lst_pred, lst_target_spec)
                lst_np = lst_phys.squeeze().detach().cpu().numpy().astype(np.float32)
                tci_val = float(traffic_head(embedding).squeeze().item())

                ndvi_phys = _physical_channel_map(cur, ch_ndvi, ndvi_spec)
                pop_phys = _physical_channel_map(cur, ch_pop, pop_spec)

                out[scn["name"] + "_lst"].append(lst_np)
                out[scn["name"] + "_tci"].append(tci_val)
                out[scn["name"] + "_ndvi"].append(ndvi_phys)
                out[scn["name"] + "_pop"].append(pop_phys)

            logger.info(f"[{i+1}/{n_cities}] {scene_id} {city_name} ({split_name}) done")

        # Build derived deltas (scenario - baseline) and per-city/scenario stats.
        baseline_lst = np.stack(out["baseline_lst"], axis=0)  # (N,H,W) physical Kelvin
        baseline_tci = np.asarray(out["baseline_tci"], dtype=np.float32)  # (N,) physical TCI

        for scn in scenarios:
            name = scn["name"]
            scen_lst = np.stack(out[name + "_lst"], axis=0)
            scen_tci = np.asarray(out[name + "_tci"], dtype=np.float32)
            dlst = scen_lst - baseline_lst  # physical Kelvin; cooling = -dlst
            dtci = scen_tci - baseline_tci
            out[name + "_dlst"] = list(dlst)
            out[name + "_dtci"] = list(dtci)

            for i in range(n_cities):
                cm = payload["city_mask"][i]
                vm = valid_masks[i]
                urban = cm & vm
                cooling_stats = _masked_stats(-dlst[i], urban)  # cooling = baseline - scenario
                dlst_stats = _masked_stats(dlst[i], urban)
                applied_dndvi = _masked_stats(
                    out[name + "_ndvi"][i] - out["baseline_ndvi"][i], urban
                )
                applied_dpop_sum = float(
                    np.sum((out[name + "_pop"][i] - out["baseline_pop"][i])[urban])
                ) if np.any(urban) else 0.0
                csv_rows.append(
                    {
                        "scene_id": int(ids[i]),
                        "city_name": str(names[i]),
                        "scenario": name,
                        "kind": scn["kind"],
                        "ndvi_key": scn["ndvi_key"] or "",
                        "pop_key": scn["pop_key"] or "",
                        "urban_pixel_count": int(np.count_nonzero(urban)),
                        "mean_dlst_city": dlst_stats["mean"],
                        "mean_cooling_city": cooling_stats["mean"],
                        "median_cooling_city": _masked_stats(-dlst[i], urban)["median"],
                        "std_dlst_city": dlst_stats["std"],
                        "min_dlst_city": dlst_stats["min"],
                        "max_dlst_city": dlst_stats["max"],
                        "mean_applied_dndvi_city": applied_dndvi["mean"],
                        "sum_applied_dpop_city": applied_dpop_sum,
                        "baseline_tci": float(baseline_tci[i]),
                        "scenario_tci": float(scen_tci[i]),
                        "delta_tci": float(dtci[i]),
                    }
                )

    elapsed = time.time() - start

    # -----------------------------------------------------------------------
    # Write outputs (all PHYSICAL units).
    # -----------------------------------------------------------------------
    npz_payload: dict[str, np.ndarray] = {
        "ids": ids,
        "names": names,
        "city_mask": payload["city_mask"],
        "ocean_mask": payload["ocean_mask"],
        "valid_mask": np.stack(valid_masks, axis=0),
        "value_space": np.asarray("physical"),
        "scenario_names": np.asarray([s["name"] for s in scenarios], dtype="<U64"),
        "scenario_kinds": np.asarray([s["kind"] for s in scenarios], dtype="<U16"),
    }
    for scn in scenarios:
        name = scn["name"]
        npz_payload[f"lst_{name}"] = np.stack(out[name + "_lst"], axis=0).astype(np.float32)  # Kelvin
        npz_payload[f"dlst_{name}"] = np.stack(out[name + "_dlst"], axis=0).astype(np.float32)  # Kelvin
        npz_payload[f"ndvi_{name}"] = np.stack(out[name + "_ndvi"], axis=0).astype(np.float32)  # index
        npz_payload[f"pop_{name}"] = np.stack(out[name + "_pop"], axis=0).astype(np.float32)  # persons
        npz_payload[f"tci_{name}"] = np.asarray(out[name + "_tci"], dtype=np.float32)  # unitless
        npz_payload[f"dtci_{name}"] = np.asarray(out[name + "_dtci"], dtype=np.float32)

    npz_path = output_base / "intervention_predictions.npz"
    np.savez_compressed(str(npz_path), **npz_payload)
    logger.info(f"Saved predictions: {npz_path}")

    csv_path = output_base / "intervention_stats.csv"
    fieldnames = list(csv_rows[0].keys()) if csv_rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)
    logger.info(f"Saved stats CSV: {csv_path}")

    summary = {
        "experiment": "intervention_deltas_v3",
        "value_space": "physical",
        "units": {
            "lst": "Kelvin",
            "dlst": "Kelvin (scenario - baseline); cooling = -dlst",
            "ndvi": "reflectance index",
            "pop": "persons (per pixel)",
            "tci": "unitless traffic congestion index",
        },
        "input_npz": str(_resolve_path(args.input_npz)),
        "cities": [{"scene_id": int(ids[i]), "name": str(names[i])} for i in range(n_cities)],
        "year": year,
        "scenarios": scenarios,
        "checkpoints": {
            "lst": str(lst_ckpt_path),
            "tci": str(traffic_ckpt_path),
            "tci_note": "traffic CV fold2 (aligns with planning_exp baseline)",
        },
        "sources": {
            "ndvi": {"channel": ndvi_name, "index": ch_ndvi, "mode": str(ndvi_spec["mode"])},
            "pop": {"channel": pop_name, "index": ch_pop, "mode": str(pop_spec["mode"])},
            "lst": {"source": lst_source_name, "mode": str(lst_target_spec["mode"])},
        },
        "crop": {"height": crop_h, "width": crop_w, "center_ratio": float(pipeline_spec.center_ratio)},
        "outputs": {"npz": str(npz_path), "stats_csv": str(csv_path)},
        "elapsed_seconds": elapsed,
    }
    summary_path = output_base / "summary.json"
    summary_path.write_text(
        json.dumps(base_run._jsonable(summary), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    logger.info(f"Saved summary: {summary_path} ({elapsed:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
