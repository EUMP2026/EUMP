#!/usr/bin/env python3
"""
experiments/planning_exp/run.py

Urban planning optimization: freeze encoder + downstream heads, treat selected
input channels (NDVI, POP) as learnable parameters, optimize them to achieve
target reductions in LST and traffic congestion.

Usage:
    python experiments/planning_exp/run.py \
        data=eump_full/server_v2

    python experiments/planning_exp/run.py \
        --cities 782,339,216 --epochs 3000 --mode cooling_only \
        data=eump_full/server_v2
"""

from __future__ import annotations

import argparse
import json
import math
from numbers import Number
import os
import pickle
import time
from pathlib import Path
import sys
from typing import Any

import yaml
from hydra import compose, initialize_config_dir
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from urban_embedding.core.logging import setup_logger
from urban_embedding.core.paths import MODEL_DIR, resolve_model_artifact_path
from urban_embedding.core.reproducibility import set_reproducible
from urban_embedding.data.processed.dataset import build_eump_dataloaders
from urban_embedding.data.processed.pipeline_spec import (
    PipelineSpec,
    get_single_downstream_target_schema,
    validate_eump_downstream_spec,
)
from urban_embedding.models.eump.swin.downstream import build_eump_downstream_model
from urban_embedding.training.checkpointing import load_model_state_for_evaluation
from urban_embedding.training.evaluation import normalize_evaluation_config
from urban_embedding.training.regression_metrics import inverse_processed_tensor
CONFIG_DIR = str(PROJECT_ROOT / "configs")
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = EXPERIMENT_DIR / "config.yaml"
SUPPORTED_SOURCE_MODES = {"pass_through", "divide", "zscore", "log_zscore"}
LINEAR_LST_POP_WEIGHT_MODES = {"linear", "direct", "population", "pop"}
LOG10P_LST_POP_WEIGHT_MODES = {"log10p", "log10_1p", "log_10_1p", "log"}
ALL_LST_TARGET_REGIONS = {"all", "full", "global", "raster"}
CITY_LST_TARGET_REGIONS = {"city", "city_only", "urban"}
ABSOLUTE_TRAFFIC_DELTA_MODES = {"absolute", "abs"}
PERCENT_TRAFFIC_DELTA_MODES = {"percent", "percentage", "pct", "relative"}

logger = setup_logger(log_dir=None, rank=0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Urban planning optimization via input-channel gradient descent."
    )
    parser.add_argument(
        "--config-path", type=Path, default=DEFAULT_CONFIG_PATH,
        help=f"Path to experiment config.yaml. Default: {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument("--cities", default=None, help="Comma-separated city IDs.")
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--mode", choices=("joint", "cooling_only", "both"), default=None)
    parser.add_argument(
        "--gradient-mode",
        choices=("source_separated", "joint"),
        default=None,
        help=(
            "Gradient policy for joint planning: source_separated keeps the "
            "current NDVI<-LST and POP<-Traffic updates; joint backpropagates "
            "the summed LST+Traffic objective once and updates both channels."
        ),
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--lst-delta-base",
        type=float,
        default=None,
        help="Override planning_exp.targets.lst_delta_base, in physical LST units.",
    )
    parser.add_argument(
        "--lst-delta-pop-weighting",
        choices=("linear", "log10p"),
        default=None,
        help="Override planning_exp.targets.lst_delta_pop_weighting.",
    )
    parser.add_argument(
        "--lst-target-region",
        choices=("all", "city"),
        default=None,
        help="Override planning_exp.targets.lst_target_region.",
    )
    parser.add_argument(
        "--lst-delta-normalize-city-mean",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override planning_exp.targets.lst_delta_normalize_city_mean. "
            "When true, the city-mask mean target cooling equals lst_delta_base."
        ),
    )
    parser.add_argument(
        "--traffic-delta",
        default=None,
        help=(
            "Override planning_exp.targets.traffic_delta. Numeric values use "
            "traffic_delta_mode; strings like '10%' use percent mode."
        ),
    )
    parser.add_argument(
        "--traffic-delta-mode",
        choices=("absolute", "percent"),
        default=None,
        help="Override planning_exp.targets.traffic_delta_mode.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--no-city-npz",
        action="store_true",
        help="Disable per-city city_<id>.npz export.",
    )
    parser.add_argument(
        "--no-data-export",
        action="store_true",
        help="Disable standard planning_data.pkl export.",
    )
    parser.add_argument(
        "--data-file",
        default=None,
        help="Override standard planning data package filename.",
    )

    args, overrides = parser.parse_known_args(argv)
    args.overrides = overrides
    return args


def _load_experiment_config(args: argparse.Namespace) -> dict[str, Any]:
    with open(args.config_path, "r") as f:
        raw = yaml.safe_load(f)
    cfg = raw.get("planning_exp", raw)

    if args.cities is not None:
        cfg["cities"] = [int(c.strip()) for c in args.cities.split(",")]
    if args.year is not None:
        cfg["year"] = args.year
    if args.mode is not None:
        cfg["mode"] = args.mode
    if args.gradient_mode is not None:
        cfg.setdefault("optimization", {})["gradient_mode"] = args.gradient_mode
    if args.epochs is not None:
        cfg["optimization"]["epochs"] = args.epochs
        cfg.setdefault("cooling_only", {})["epochs"] = args.epochs
    if args.lr is not None:
        cfg["optimization"]["lr"] = args.lr
        cfg.setdefault("cooling_only", {})["lr"] = args.lr
    if args.lst_delta_base is not None:
        cfg["targets"]["lst_delta_base"] = float(args.lst_delta_base)
    if args.lst_delta_pop_weighting is not None:
        cfg["targets"]["lst_delta_pop_weighting"] = args.lst_delta_pop_weighting
    if args.lst_target_region is not None:
        cfg["targets"]["lst_target_region"] = args.lst_target_region
    if args.lst_delta_normalize_city_mean is not None:
        cfg["targets"]["lst_delta_normalize_city_mean"] = bool(
            args.lst_delta_normalize_city_mean
        )
    if args.traffic_delta is not None:
        cfg["targets"]["traffic_delta"] = args.traffic_delta
    if args.traffic_delta_mode is not None:
        cfg["targets"]["traffic_delta_mode"] = args.traffic_delta_mode
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
    if args.no_city_npz:
        cfg.setdefault("outputs", {})["save_city_npz"] = False
    if args.no_data_export:
        cfg.setdefault("data_export", {})["enabled"] = False
    if args.data_file is not None:
        cfg.setdefault("data_export", {})["data_file"] = args.data_file
    return cfg


# ---------------------------------------------------------------------------
# Hydra & model building
# ---------------------------------------------------------------------------

def _compose_hydra_config(pipeline: str, overrides: list[str], num_workers: int) -> dict[str, Any]:
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        raw = compose(config_name="evaluate", overrides=[f"pipeline={pipeline}"] + overrides)
    cfg = normalize_evaluation_config(raw)
    cfg["pipeline"]["num_workers"] = num_workers
    cfg["pipeline"]["center_ratio"] = 1.0
    return cfg


def _resolve_device(requested: str) -> torch.device:
    if str(requested).strip().lower() == "auto":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return torch.device(requested)


def _resolve_output_base(path_value: Any) -> Path:
    output_base = Path(str(path_value))
    if not output_base.is_absolute():
        output_base = (PROJECT_ROOT / output_base).resolve()
    return output_base


def _resolve_source_channel(
    *,
    source_name: str,
    preprocessing_spec: dict[str, Any],
    raster_input_channel_names: tuple[str, ...],
) -> tuple[int, str, dict[str, Any]]:
    source_spec = preprocessing_spec["sources"].get(source_name)
    if source_spec is None:
        raise KeyError(f"Source {source_name!r} not found in preprocessing spec")
    channel_names = tuple(str(n) for n in source_spec.get("output_channel_names", ()))
    if len(channel_names) != 1:
        raise ValueError(f"Source {source_name!r} must have exactly one channel, got {channel_names!r}")
    channel_name = channel_names[0]
    if channel_name not in raster_input_channel_names:
        raise ValueError(f"Channel {channel_name!r} not in raster inputs {raster_input_channel_names!r}")
    return raster_input_channel_names.index(channel_name), channel_name, source_spec


def _apply_checkpoint_overrides(hydra_cfg: dict[str, Any], cfg: dict[str, Any]) -> None:
    checkpoint_source = str(cfg.get("checkpoint_source", "best_model"))
    hydra_cfg["evaluation"]["checkpoint_source"] = checkpoint_source
    checkpoint_file = cfg.get("checkpoint_file")
    if checkpoint_file:
        hydra_cfg["evaluation"]["checkpoint_file"] = str(checkpoint_file)


def _build_model(
    *,
    hydra_cfg: dict[str, Any],
    batch_schema: Any,
    target_schema: Any,
    device: torch.device,
    freeze_encoder: bool,
) -> nn.Module:
    encoder_source = str(hydra_cfg["downstream"]["encoder_source"])
    encoder_checkpoint_path = None
    if encoder_source in {
        "reconstruction_checkpoint",
        "convnext_reconstruction_checkpoint",
        "convnext_pretrained",
    }:
        encoder_checkpoint_path = resolve_model_artifact_path(
            str(hydra_cfg["downstream"]["encoder_model_name"]),
            str(hydra_cfg["downstream"]["encoder_checkpoint_file"]),
        )
    model = build_eump_downstream_model(
        model_config=dict(hydra_cfg["model"]),
        batch_schema=batch_schema,
        target_schema=target_schema,
        base_swin_path=str(Path(MODEL_DIR) / str(hydra_cfg["experiment"]["pretrain_model"])),
        encoder_source=encoder_source,
        encoder_checkpoint_path=encoder_checkpoint_path,
        load_pretrained=not bool(hydra_cfg["experiment"]["scratch_backbone"]),
        freeze_encoder=freeze_encoder,
        map_location=device,
    ).to(device)

    checkpoint_source = str(hydra_cfg["evaluation"]["checkpoint_source"])
    checkpoint_file = (
        str(hydra_cfg["evaluation"]["checkpoint_file"])
        if hydra_cfg["evaluation"]["checkpoint_file"]
        else (
            str(hydra_cfg["downstream"]["save_file"])
            if checkpoint_source == "best_model"
            else str(hydra_cfg["downstream"]["checkpoint_file"])
        )
    )
    load_model_state_for_evaluation(
        model=model,
        model_name=str(hydra_cfg["experiment"]["model_name"]),
        checkpoint_source=checkpoint_source,
        checkpoint_file=checkpoint_file,
        map_location=device,
    )
    model.eval()
    return model


def _check_planning_prerequisites(cfg: dict[str, Any], hydra_cfg_lst: dict[str, Any]) -> None:
    """Fail early with actionable messages if required model files are missing."""
    missing: list[str] = []

    # Pretrain encoder checkpoint
    encoder_model_name = str(hydra_cfg_lst["downstream"]["encoder_model_name"])
    encoder_checkpoint_file = str(hydra_cfg_lst["downstream"]["encoder_checkpoint_file"])
    encoder_path = Path(MODEL_DIR) / encoder_model_name / encoder_checkpoint_file
    if not encoder_path.exists():
        missing.append(
            f"Pretrain encoder checkpoint not found: {encoder_path}\n"
            "  -> Run the mae_v3 pretrain stage first:\n"
            "     bash script/training/run_eump_full_train.sh pretrain launch=eump_full/mae_v3 data=eump_full/server_v2"
        )

    # LST downstream model
    lst_model_name = str(hydra_cfg_lst["experiment"]["model_name"])
    lst_checkpoint_file = str(hydra_cfg_lst["downstream"]["save_file"])
    lst_path = Path(MODEL_DIR) / lst_model_name / lst_checkpoint_file
    if not lst_path.exists():
        missing.append(
            f"LST downstream model not found: {lst_path}\n"
            "  -> Run the mae_v3 lst downstream stage first:\n"
            "     bash script/training/run_eump_full_train.sh lst launch=eump_full/mae_v3 data=eump_full/server_v2"
        )

    # Traffic CV fold checkpoint
    traffic_cv_cfg = cfg.get("traffic_cv", {})
    traffic_checkpoint = traffic_cv_cfg.get("checkpoint")
    if traffic_checkpoint is not None:
        traffic_path = Path(str(traffic_checkpoint))
        if not traffic_path.is_absolute():
            traffic_path = PROJECT_ROOT / traffic_path
        if not traffic_path.exists():
            missing.append(
                f"Traffic CV fold checkpoint not found: {traffic_path}\n"
                "  -> Run the mae_v3 traffic CV first:\n"
                "     bash script/experiments/run_current_traffic_cv.sh --data eump_full/server_v2\n"
                "  -> Then update traffic_cv.checkpoint in config.yaml to point at the desired fold."
            )

    if missing:
        raise FileNotFoundError(
            "planning_exp prerequisites missing:\n\n" + "\n\n".join(missing)
        )


def build_planning_models(
    cfg: dict[str, Any],
    hydra_cfg_lst: dict[str, Any],
    batch_schema_lst: Any,
    device: torch.device,
) -> tuple[nn.Module, nn.Module, nn.Module]:
    """Build shared encoder + LST model + Traffic scalar head.

    LST model is built with freeze_encoder=False so gradients flow to input,
    but all encoder params are manually set to requires_grad=False.
    Traffic head is a PooledEmbeddingScalarHead loaded from a CV fold checkpoint.
    """
    _check_planning_prerequisites(cfg, hydra_cfg_lst)
    target_schema_lst = get_single_downstream_target_schema(batch_schema_lst)

    lst_model = _build_model(
        hydra_cfg=hydra_cfg_lst,
        batch_schema=batch_schema_lst,
        target_schema=target_schema_lst,
        device=device,
        freeze_encoder=False,
    )
    for param in lst_model.encoder.parameters():
        param.requires_grad = False
    for param in lst_model.raster_heads.parameters():
        param.requires_grad = False

    encoder = lst_model.encoder

    traffic_cv_cfg = cfg.get("traffic_cv", {})
    traffic_checkpoint = traffic_cv_cfg.get("checkpoint")
    if traffic_checkpoint is None:
        raise ValueError("planning_exp.traffic_cv.checkpoint must be set for joint mode")
    traffic_checkpoint_path = Path(str(traffic_checkpoint))
    if not traffic_checkpoint_path.is_absolute():
        traffic_checkpoint_path = PROJECT_ROOT / traffic_checkpoint_path

    embedding_channels = lst_model.encoder.config.embed_dim * (
        2 ** (lst_model.encoder.config.num_layers - 1)
    )
    hidden_dim = int(traffic_cv_cfg.get("hidden_dim", 128))
    dropout = float(traffic_cv_cfg.get("dropout", 0.5))
    traffic_head = _PooledEmbeddingScalarHead(
        input_dim=embedding_channels, hidden_dim=hidden_dim, dropout=dropout,
    ).to(device)

    checkpoint_data = torch.load(str(traffic_checkpoint_path), map_location=device)
    state_dict = checkpoint_data.get("model_state_dict", checkpoint_data)
    traffic_head.load_state_dict(state_dict)
    traffic_head.eval()
    for param in traffic_head.parameters():
        param.requires_grad = False

    logger.info(f"Loaded traffic head from: {traffic_checkpoint_path}")
    return encoder, lst_model, traffic_head


class _PooledEmbeddingScalarHead(nn.Module):
    def __init__(self, *, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.net = nn.Sequential(
            nn.LayerNorm(int(input_dim)),
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(embedding).flatten(1)
        return self.net(pooled)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def tv_loss(
    x: torch.Tensor,
    beta: int = 2,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return TV over valid neighboring edges only.

    The planning variable can be updated over the complete valid raster, so
    regularization must use the same valid decision region.  Otherwise edges
    crossing invalid/background pixels contribute gradients to valid pixels
    even though those neighboring pixels are not decision variables.
    """
    dx = torch.abs(x[1:, :] - x[:-1, :]) ** beta
    dy = torch.abs(x[:, 1:] - x[:, :-1]) ** beta
    if valid_mask is None:
        return dx.mean() + dy.mean()

    valid = valid_mask.to(device=x.device, dtype=torch.bool).squeeze()
    if valid.shape != x.shape:
        raise ValueError(
            f"valid_mask shape {tuple(valid.shape)} does not match TV input shape {tuple(x.shape)}"
        )
    valid_dx = valid[1:, :] & valid[:-1, :]
    valid_dy = valid[:, 1:] & valid[:, :-1]
    dx_count = int(valid_dx.sum().item())
    dy_count = int(valid_dy.sum().item())
    if dx_count == 0 and dy_count == 0:
        return x.sum() * 0.0
    # Keep the original ``dx.mean() + dy.mean()`` scale, while excluding
    # edges that cross invalid pixels.  Computing the two directions
    # separately also avoids changing the relative horizontal/vertical
    # weighting for non-square rasters or irregular masks.
    dx_mean = dx[valid_dx].mean() if dx_count else x.sum() * 0.0
    dy_mean = dy[valid_dy].mean() if dy_count else x.sum() * 0.0
    return dx_mean + dy_mean


def one_sided_decrease_penalty(
    cur_input: torch.Tensor,
    original_input: torch.Tensor,
    channel_idx: int,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    diff = original_input[:, channel_idx, :, :] - cur_input[:, channel_idx, :, :]
    penalty = torch.clamp(diff, min=0)
    if valid_mask is None:
        return penalty.mean()
    valid = valid_mask.to(device=penalty.device, dtype=torch.bool).squeeze()
    if valid.shape != penalty.shape[-2:]:
        raise ValueError(
            f"valid_mask shape {tuple(valid.shape)} does not match penalty raster shape "
            f"{tuple(penalty.shape[-2:])}"
        )
    if not bool(valid.any()):
        return penalty.sum() * 0.0
    return penalty[:, valid].mean()


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------

def _masked_avg(values: np.ndarray, mask: np.ndarray) -> float:
    mask_sum = mask.sum()
    if mask_sum == 0:
        return 0.0
    return float(np.sum(values * mask) / mask_sum)


def _build_input_adamw_optimizer(
    *,
    cur_input: torch.Tensor,
    opt_cfg: dict[str, Any],
    lr: float,
) -> torch.optim.Optimizer | None:
    if not bool(opt_cfg.get("use_adamw_step", True)):
        return None
    adamw_lr = lr * float(opt_cfg.get("adamw_lr_multiplier", 5.0))
    weight_decay = float(opt_cfg.get("weight_decay", 0.0))
    return torch.optim.AdamW([cur_input], lr=adamw_lr, weight_decay=weight_decay)


def _apply_input_update(
    *,
    cur_input: torch.Tensor,
    masked_grad: torch.Tensor,
    update_mask: torch.Tensor,
    lr: float,
    use_sign_step: bool,
    optimizer: torch.optim.Optimizer | None,
) -> None:
    step = masked_grad.sign() if use_sign_step else masked_grad
    with torch.no_grad():
        cur_input.sub_(lr * step)
    if optimizer is not None:
        after_manual_step = cur_input.detach().clone()
        cur_input.grad = masked_grad.clone()
        optimizer.step()
        optimizer.zero_grad()
        with torch.no_grad():
            active = update_mask.to(dtype=torch.bool, device=cur_input.device)
            cur_input[~active] = after_manual_step[~active]


def _canonical_lst_pop_weighting(mode: object) -> str:
    mode_text = str(mode).strip().lower()
    if mode_text in LINEAR_LST_POP_WEIGHT_MODES:
        return "linear"
    if mode_text in LOG10P_LST_POP_WEIGHT_MODES:
        return "log10p"
    allowed = sorted(LINEAR_LST_POP_WEIGHT_MODES | LOG10P_LST_POP_WEIGHT_MODES)
    raise ValueError(f"Unsupported lst_delta_pop_weighting={mode!r}; expected one of {allowed}")


def _lst_pop_weight(pop_physical: torch.Tensor, targets_cfg: dict[str, Any]) -> torch.Tensor:
    weighting = _canonical_lst_pop_weighting(targets_cfg.get("lst_delta_pop_weighting", "linear"))
    if weighting == "log10p":
        return torch.log10(torch.clamp(pop_physical, min=0.0) + 1.0)
    return pop_physical


def _canonical_lst_target_region(region: object) -> str:
    region_text = str(region).strip().lower()
    if region_text in ALL_LST_TARGET_REGIONS:
        return "all"
    if region_text in CITY_LST_TARGET_REGIONS:
        return "city"
    allowed = sorted(ALL_LST_TARGET_REGIONS | CITY_LST_TARGET_REGIONS)
    raise ValueError(f"Unsupported lst_target_region={region!r}; expected one of {allowed}")


def _city_mask_tensor(
    city_mask: np.ndarray | torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.as_tensor(city_mask, device=device, dtype=torch.bool).squeeze()
    if mask.ndim != 2:
        raise ValueError(f"city_mask must be 2D after squeeze, got shape {tuple(mask.shape)}")
    if not bool(mask.any()):
        raise ValueError("city_mask is empty; cannot use city-only LST target mode")
    return mask


def _normalize_lst_delta_factor_to_city_mean(
    factor: torch.Tensor,
    city_mask: np.ndarray | torch.Tensor | None,
    valid_mask: np.ndarray | torch.Tensor | None = None,
) -> torch.Tensor:
    if city_mask is None:
        raise ValueError(
            "targets.lst_delta_normalize_city_mean=true requires city_mask in compute_targets."
        )
    mask = _city_mask_tensor(city_mask, device=factor.device)
    if valid_mask is not None:
        valid = torch.as_tensor(valid_mask, device=factor.device, dtype=torch.bool).squeeze()
        if valid.shape != mask.shape:
            raise ValueError(
                f"valid_mask shape {tuple(valid.shape)} does not match city_mask shape "
                f"{tuple(mask.shape)} in LST target normalization"
            )
        mask = mask & valid
        if not bool(mask.any()):
            raise ValueError("City valid mask is empty in LST target normalization.")
    city_mean = factor[mask].mean()
    if not bool(city_mean > 0):
        raise ValueError("City mean LST target factor must be positive for normalization.")
    return factor / city_mean


def _lst_loss_mask(
    *,
    cfg: dict[str, Any],
    city_mask: np.ndarray,
    base_mask: torch.Tensor,
) -> torch.Tensor | None:
    region = _canonical_lst_target_region(cfg["targets"].get("lst_target_region", "all"))
    if region == "all":
        # "all" means the complete valid raster, not invalid/background
        # pixels whose target tensor is filled with a placeholder value.
        return base_mask.to(dtype=torch.bool)
    city_mask_t = _city_mask_tensor(city_mask, device=base_mask.device)
    if city_mask_t.shape != base_mask.shape:
        raise ValueError(
            f"city_mask shape {tuple(city_mask_t.shape)} does not match "
            f"base_mask shape {tuple(base_mask.shape)}"
        )
    loss_mask = city_mask_t & base_mask.to(dtype=torch.bool)
    if not bool(loss_mask.any()):
        raise ValueError("LST city loss mask is empty after intersecting with base_mask.")
    return loss_mask


def _mse_loss_with_optional_mask(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    if mask is None:
        return F.mse_loss(pred, target)
    mask = mask.to(device=pred.device, dtype=torch.bool)
    while mask.ndim < pred.ndim:
        mask = mask.unsqueeze(0)
    mask = mask.expand_as(pred)
    if not bool(mask.any()):
        raise ValueError("Masked MSE received an empty mask.")
    return F.mse_loss(pred[mask], target[mask])


def _canonical_traffic_delta_mode(mode: object) -> str:
    mode_text = str(mode).strip().lower()
    if mode_text in ABSOLUTE_TRAFFIC_DELTA_MODES:
        return "absolute"
    if mode_text in PERCENT_TRAFFIC_DELTA_MODES:
        return "percent"
    allowed = sorted(ABSOLUTE_TRAFFIC_DELTA_MODES | PERCENT_TRAFFIC_DELTA_MODES)
    raise ValueError(f"Unsupported traffic_delta_mode={mode!r}; expected one of {allowed}")


def _parse_traffic_delta(targets_cfg: dict[str, Any]) -> tuple[str, float]:
    raw_delta = targets_cfg["traffic_delta"]
    raw_mode = targets_cfg.get("traffic_delta_mode", "absolute")

    if isinstance(raw_delta, dict):
        raw_mode = raw_delta.get("mode", raw_delta.get("type", raw_mode))
        if "value" not in raw_delta:
            raise ValueError(
                "targets.traffic_delta mapping must include a 'value' field, "
                "for example {'mode': 'percent', 'value': 0.1}."
            )
        raw_delta = raw_delta["value"]

    mode = _canonical_traffic_delta_mode(raw_mode)
    if isinstance(raw_delta, str):
        delta_text = raw_delta.strip()
        if delta_text.endswith("%"):
            mode = "percent"
            delta = float(delta_text[:-1].strip()) / 100.0
        else:
            delta = float(delta_text)
    else:
        delta = float(raw_delta)

    if mode == "percent" and not 0.0 <= delta <= 1.0:
        raise ValueError(
            "Percent traffic_delta must be between 0 and 1, or written as a "
            "percentage string such as '10%'."
        )
    return mode, delta


def compute_targets(
    original_lst: torch.Tensor,
    original_traffic: torch.Tensor,
    pop_physical: torch.Tensor,
    cfg: dict[str, Any],
    lst_source_spec: dict[str, Any],
    city_mask: np.ndarray | torch.Tensor | None = None,
    valid_mask: np.ndarray | torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    targets_cfg = cfg["targets"]
    stats = lst_source_spec.get("stats", {})
    lst_std = float(stats.get("std", 1.0)) if stats else 1.0

    delta_base = float(targets_cfg["lst_delta_base"]) / lst_std
    pop_scale = float(targets_cfg["lst_delta_pop_scale"])
    pop_weight = _lst_pop_weight(pop_physical, targets_cfg)
    pop_weight_max = pop_weight.max()
    if pop_weight_max > 0:
        delta_factor = 1.0 + pop_scale * pop_weight / pop_weight_max
    else:
        delta_factor = torch.ones_like(pop_physical)
    if bool(targets_cfg.get("lst_delta_normalize_city_mean", False)):
        delta_factor = _normalize_lst_delta_factor_to_city_mean(
            delta_factor,
            city_mask,
            valid_mask=valid_mask,
        )
    target_lst = original_lst - delta_base * delta_factor

    traffic_delta_mode, traffic_delta = _parse_traffic_delta(targets_cfg)
    if traffic_delta_mode == "percent":
        target_traffic = original_traffic * (1.0 - traffic_delta)
    else:
        target_traffic = original_traffic - traffic_delta
    target_traffic = torch.clamp(target_traffic, min=0.0)

    return target_lst, target_traffic


def optimize_city(
    *,
    cur_input: torch.Tensor,
    original_input: torch.Tensor,
    city_mask: np.ndarray,
    base_mask: torch.Tensor,
    target_lst: torch.Tensor,
    target_traffic: torch.Tensor,
    original_lst: torch.Tensor,
    original_traffic: torch.Tensor,
    lst_model: nn.Module,
    traffic_model: nn.Module,
    cfg: dict[str, Any],
    ch_ndvi: int,
    ch_pop: int,
    ndvi_source_spec: dict[str, Any],
    pop_source_spec: dict[str, Any],
    lst_target_spec: dict[str, Any],
    city_id: int,
) -> dict[str, Any]:
    opt_cfg = cfg["optimization"]
    loss_cfg = cfg["loss"]
    epochs = int(opt_cfg["epochs"])
    lr = float(opt_cfg["lr"])
    use_sign = bool(opt_cfg["use_sign_step"])
    gradient_mode = str(opt_cfg.get("gradient_mode", "source_separated")).strip().lower()
    if gradient_mode not in {"source_separated", "joint"}:
        raise ValueError(
            "optimization.gradient_mode must be 'source_separated' or 'joint', "
            f"got {gradient_mode!r}"
        )
    joint_optimizer = _build_input_adamw_optimizer(cur_input=cur_input, opt_cfg=opt_cfg, lr=lr)

    ndvi_max = original_input[0, ch_ndvi].max()
    pop_min = original_input[0, ch_pop].min()

    lst_stats = lst_target_spec.get("stats", {})
    lst_mean = float(lst_stats.get("mean", 0.0)) if lst_stats else 0.0
    lst_std = float(lst_stats.get("std", 1.0)) if lst_stats else 1.0
    ndvi_stats = ndvi_source_spec.get("stats", {})
    ndvi_mean = float(ndvi_stats.get("mean", 0.0)) if ndvi_stats else 0.0
    ndvi_std = float(ndvi_stats.get("std", 1.0)) if ndvi_stats else 1.0
    pop_stats = pop_source_spec.get("stats", {})
    pop_std = float(pop_stats.get("std", 1.0)) if pop_stats else 1.0
    pop_mean = float(pop_stats.get("mean", 0.0)) if pop_stats else 0.0
    logger.info(f"Stats: ndvi_std={ndvi_std}, pop_std={pop_std}, pop_mean={pop_mean}, lst_std={lst_std}")

    data = {
        "delta_lst_traj": [], "delta_traffic_traj": [],
        "pixel_loss_traj": [], "reg_loss_traj": [],
        "penalty_loss_traj": [], "total_lst_loss_traj": [],
        "traffic_loss_traj": [], "reg_traffic_traj": [],
        "pop_sum_loss_traj": [], "total_traffic_loss_traj": [],
        "delta_ndvi_traj": [], "delta_pop_traj": [],
    }

    output_size = tuple(int(v) for v in cur_input.shape[-2:])
    lst_pixel_loss_mask = _lst_loss_mask(cfg=cfg, city_mask=city_mask, base_mask=base_mask)
    lst_metric_mask = city_mask & base_mask.detach().cpu().numpy().astype(bool)

    for epoch in range(epochs):
        if joint_optimizer is not None:
            joint_optimizer.zero_grad()
        if cur_input.grad is not None:
            cur_input.grad.zero_()

        # --- LST objective ---
        embedding = lst_model.encoder.encode(cur_input, bool_masked_pos=None)
        lst_pred = list(lst_model.raster_heads.values())[0](embedding, output_size)
        pixel_loss = float(loss_cfg["scale_pixel_loss"]) * _mse_loss_with_optional_mask(
            lst_pred, target_lst, lst_pixel_loss_mask
        )
        reg_loss = float(loss_cfg["lambda_reg_lst"]) * tv_loss(
            cur_input[0, ch_ndvi] - original_input[0, ch_ndvi],
            valid_mask=base_mask,
        )
        penalty_loss = float(loss_cfg["lambda_one_sided"]) * one_sided_decrease_penalty(
            cur_input,
            original_input,
            ch_ndvi,
            valid_mask=base_mask,
        )
        lst_loss = pixel_loss + reg_loss + penalty_loss

        # --- Traffic objective ---
        traffic_embedding = lst_model.encoder.encode(cur_input, bool_masked_pos=None)
        traffic_pred = traffic_model(traffic_embedding)

        traffic_loss = float(loss_cfg["scale_traffic_loss"]) * nn.MSELoss()(
            traffic_pred, target_traffic
        )
        reg_traffic = float(loss_cfg["lambda_reg_tr"]) * tv_loss(
            cur_input[0, ch_pop] - original_input[0, ch_pop],
            valid_mask=base_mask,
        )
        phys_pop_cur = torch.pow(10.0, cur_input[0, ch_pop] * pop_std + pop_mean) - 1.0
        phys_pop_orig = torch.pow(10.0, original_input[0, ch_pop] * pop_std + pop_mean) - 1.0
        pop_sum = torch.abs(
            phys_pop_cur[base_mask].sum() - phys_pop_orig[base_mask].sum()
        )
        pop_sum_loss = float(loss_cfg["lambda_pop_sum"]) * pop_sum
        traffic_total = traffic_loss + reg_traffic + pop_sum_loss

        # The default policy intentionally keeps the original source-separated
        # intervention.  The optional joint policy backpropagates the summed
        # objective once, so both LST and Traffic losses contribute to both
        # NDVI and POP gradients.
        if gradient_mode == "joint":
            (lst_loss + traffic_total).backward()
            total_grad = cur_input.grad.detach().clone()
            assembled_grad = torch.zeros_like(cur_input)
            assembled_grad[0, ch_ndvi] = total_grad[0, ch_ndvi]
            assembled_grad[0, ch_pop] = total_grad[0, ch_pop]
        else:
            lst_loss.backward()
            ndvi_grad = cur_input.grad[0, ch_ndvi].detach().clone()
            if cur_input.grad is not None:
                cur_input.grad.zero_()
            traffic_total.backward()
            pop_grad = cur_input.grad[0, ch_pop].detach().clone()
            assembled_grad = torch.zeros_like(cur_input)
            assembled_grad[0, ch_ndvi] = ndvi_grad
            assembled_grad[0, ch_pop] = pop_grad

        # --- Apply one masked update to NDVI and POP ---
        if cur_input.grad is not None:
            cur_input.grad.zero_()
        active_mask = base_mask.to(device=assembled_grad.device, dtype=assembled_grad.dtype)
        assembled_grad[0, ch_ndvi] = assembled_grad[0, ch_ndvi] * active_mask
        assembled_grad[0, ch_pop] = assembled_grad[0, ch_pop] * active_mask
        joint_mask = torch.zeros_like(cur_input, dtype=torch.bool)
        joint_mask[0, ch_ndvi] = base_mask
        joint_mask[0, ch_pop] = base_mask
        _apply_input_update(
            cur_input=cur_input,
            masked_grad=assembled_grad,
            update_mask=joint_mask,
            lr=lr,
            use_sign_step=use_sign,
            optimizer=joint_optimizer,
        )
        with torch.no_grad():
            ndvi_ch = cur_input[0, ch_ndvi]
            ndvi_ch[base_mask] = torch.clamp(ndvi_ch[base_mask], max=ndvi_max)
            pop_ch = cur_input[0, ch_pop]
            pop_ch[base_mask] = torch.clamp(pop_ch[base_mask], min=pop_min)

        if cur_input.grad is not None:
            cur_input.grad.zero_()

        # --- Record trajectories ---
        with torch.no_grad():
            current_embedding = lst_model.encoder.encode(cur_input, bool_masked_pos=None)
            current_lst_pred = list(lst_model.raster_heads.values())[0](
                current_embedding, output_size
            )
            current_traffic_pred = traffic_model(current_embedding)
            eval_pixel_loss = float(loss_cfg["scale_pixel_loss"]) * _mse_loss_with_optional_mask(
                current_lst_pred, target_lst, lst_pixel_loss_mask
            )
            eval_reg_loss = float(loss_cfg["lambda_reg_lst"]) * tv_loss(
                cur_input[0, ch_ndvi] - original_input[0, ch_ndvi],
                valid_mask=base_mask,
            )
            eval_penalty_loss = float(loss_cfg["lambda_one_sided"]) * one_sided_decrease_penalty(
                cur_input,
                original_input,
                ch_ndvi,
                valid_mask=base_mask,
            )
            eval_lst_loss = eval_pixel_loss + eval_reg_loss + eval_penalty_loss
            eval_traffic_loss = float(loss_cfg["scale_traffic_loss"]) * nn.MSELoss()(
                current_traffic_pred, target_traffic
            )
            eval_reg_traffic = float(loss_cfg["lambda_reg_tr"]) * tv_loss(
                cur_input[0, ch_pop] - original_input[0, ch_pop],
                valid_mask=base_mask,
            )
            det_lst = lst_std * _masked_avg(
                current_lst_pred.cpu().numpy().squeeze() - original_lst.cpu().numpy().squeeze(),
                lst_metric_mask,
            )
            det_traffic = float((current_traffic_pred - original_traffic).item())
            det_ndvi = ndvi_std * _masked_avg(
                (cur_input[0, ch_ndvi] - original_input[0, ch_ndvi]).cpu().numpy(),
                lst_metric_mask,
            )
            log_pop_orig = original_input[0, ch_pop] * pop_std + pop_mean
            log_pop_cur = cur_input[0, ch_pop] * pop_std + pop_mean
            phys_pop_orig = (10.0 ** log_pop_orig - 1.0).clamp(min=0)
            phys_pop_cur = (10.0 ** log_pop_cur - 1.0).clamp(min=0)
            pop_orig_sum = phys_pop_orig[base_mask].sum().item()
            eval_pop_sum_loss = float(loss_cfg["lambda_pop_sum"]) * torch.abs(
                phys_pop_cur[base_mask].sum() - phys_pop_orig[base_mask].sum()
            )
            eval_traffic_total = eval_traffic_loss + eval_reg_traffic + eval_pop_sum_loss
            det_pop = (
                (phys_pop_cur[base_mask].sum().item() - pop_orig_sum) / pop_orig_sum * 100.0
            ) if pop_orig_sum > 0 else 0.0

        data["delta_lst_traj"].append(det_lst)
        data["delta_traffic_traj"].append(det_traffic)
        data["pixel_loss_traj"].append(float(eval_pixel_loss.item()))
        data["reg_loss_traj"].append(float(eval_reg_loss.item()))
        data["penalty_loss_traj"].append(float(eval_penalty_loss.item()))
        data["total_lst_loss_traj"].append(float(eval_lst_loss.item()))
        data["traffic_loss_traj"].append(float(eval_traffic_loss.item()))
        data["reg_traffic_traj"].append(float(eval_reg_traffic.item()))
        data["pop_sum_loss_traj"].append(float(eval_pop_sum_loss.item()))
        data["total_traffic_loss_traj"].append(float(eval_traffic_total.item()))
        data["delta_ndvi_traj"].append(float(det_ndvi))
        data["delta_pop_traj"].append(float(det_pop))

        if epoch % 50 == 0 or epoch == epochs - 1:
            logger.info(
                f"[city {city_id}] epoch {epoch}: "
                f"lst_loss={eval_lst_loss.item():.4f} "
                f"traffic_loss={eval_traffic_total.item():.4f} "
                f"delta_lst={det_lst:.4f} delta_traffic={det_traffic:.4f}"
            )

    with torch.no_grad():
        embedding = lst_model.encoder.encode(cur_input, bool_masked_pos=None)
        adjusted_lst = list(lst_model.raster_heads.values())[0](embedding, output_size)
        adjusted_traffic = traffic_model(embedding)

    final_signed_det_lst = lst_std * _masked_avg(
        adjusted_lst.cpu().numpy().squeeze() - original_lst.cpu().numpy().squeeze(),
        lst_metric_mask,
    )
    final_abs_det_lst = abs(final_signed_det_lst)
    final_det_traffic = float((adjusted_traffic - original_traffic).item())
    original_traffic_value = float(original_traffic.item())
    target_traffic_value = float(target_traffic.item())
    adjusted_traffic_value = float(adjusted_traffic.item())
    if original_traffic_value > 0:
        traffic_drop_pct = (
            (original_traffic_value - adjusted_traffic_value) / original_traffic_value * 100.0
        )
        target_traffic_drop_pct = (
            (original_traffic_value - target_traffic_value) / original_traffic_value * 100.0
        )
    else:
        traffic_drop_pct = None
        target_traffic_drop_pct = None

    data["original_ndvi"] = original_input[0, ch_ndvi].cpu().numpy()
    data["adjusted_ndvi"] = cur_input[0, ch_ndvi].detach().cpu().numpy()
    data["original_pop"] = original_input[0, ch_pop].cpu().numpy()
    data["adjusted_pop"] = cur_input[0, ch_pop].detach().cpu().numpy()
    data["original_lst"] = original_lst.cpu().numpy().squeeze()
    data["adjusted_lst"] = adjusted_lst.cpu().numpy().squeeze()
    data["target_lst"] = target_lst.cpu().numpy().squeeze()
    data["lst_mean"] = np.array(lst_mean)
    data["lst_std"] = np.array(lst_std)
    data["ndvi_mean"] = np.array(ndvi_mean)
    data["ndvi_std"] = np.array(ndvi_std)
    data["pop_std"] = np.array(pop_std)
    data["pop_mean"] = np.array(pop_mean)
    data["ndvi_source_mode"] = np.array(str(ndvi_source_spec.get("mode", "")))
    data["pop_source_mode"] = np.array(str(pop_source_spec.get("mode", "")))
    data["original_traffic"] = original_traffic_value
    data["target_traffic"] = target_traffic_value
    data["adjusted_traffic"] = adjusted_traffic_value
    data["city_mask"] = city_mask
    data["ocean_mask"] = ~base_mask.cpu().numpy()
    data["city_id"] = int(city_id)
    data["final_signed_delta_lst"] = final_signed_det_lst
    data["final_delta_lst"] = final_signed_det_lst
    data["final_abs_delta_lst"] = final_abs_det_lst
    data["final_delta_traffic"] = final_det_traffic
    if traffic_drop_pct is not None:
        data["traffic_drop_pct"] = traffic_drop_pct
    if target_traffic_drop_pct is not None:
        data["target_traffic_drop_pct"] = target_traffic_drop_pct

    logger.info(
        f"[city {city_id}] DONE: delta_lst={final_signed_det_lst:.4f} "
        f"(abs={final_abs_det_lst:.4f}), "
        f"delta_traffic={final_det_traffic:.4f}"
    )
    return data


def optimize_cooling_only(
    *,
    cur_input: torch.Tensor,
    original_input: torch.Tensor,
    city_mask: np.ndarray,
    base_mask: torch.Tensor,
    target_lst: torch.Tensor,
    original_lst: torch.Tensor,
    lst_model: nn.Module,
    cfg: dict[str, Any],
    ch_ndvi: int,
    ndvi_source_spec: dict[str, Any],
    lst_target_spec: dict[str, Any],
    city_id: int,
) -> dict[str, Any]:
    cooling_cfg = cfg.get("cooling_only", {})
    loss_cfg = cfg["loss"]
    epochs = int(cooling_cfg.get("epochs") or cfg["optimization"]["epochs"])
    lr_val = cooling_cfg.get("lr")
    lr = float(lr_val) if lr_val is not None else float(cfg["optimization"]["lr"])
    opt_cfg = cfg["optimization"]
    use_sign = bool(opt_cfg["use_sign_step"])
    optimizer = _build_input_adamw_optimizer(cur_input=cur_input, opt_cfg=opt_cfg, lr=lr)

    ndvi_max = original_input[0, ch_ndvi].max()
    lst_stats = lst_target_spec.get("stats", {})
    lst_mean = float(lst_stats.get("mean", 0.0)) if lst_stats else 0.0
    lst_std = float(lst_stats.get("std", 1.0)) if lst_stats else 1.0
    ndvi_stats = ndvi_source_spec.get("stats", {})
    ndvi_mean = float(ndvi_stats.get("mean", 0.0)) if ndvi_stats else 0.0
    ndvi_std = float(ndvi_stats.get("std", 1.0)) if ndvi_stats else 1.0
    output_size = tuple(int(v) for v in cur_input.shape[-2:])
    lst_pixel_loss_mask = _lst_loss_mask(cfg=cfg, city_mask=city_mask, base_mask=base_mask)
    lst_metric_mask = city_mask & base_mask.detach().cpu().numpy().astype(bool)

    data: dict[str, Any] = {
        "delta_lst_traj": [], "pixel_loss_traj": [],
        "reg_loss_traj": [], "penalty_loss_traj": [],
        "total_lst_loss_traj": [], "delta_ndvi_traj": [],
    }
    for epoch in range(epochs):
        if optimizer is not None:
            optimizer.zero_grad()
        if cur_input.grad is not None:
            cur_input.grad.zero_()

        embedding = lst_model.encoder.encode(cur_input, bool_masked_pos=None)
        lst_pred = list(lst_model.raster_heads.values())[0](embedding, output_size)

        pixel_loss = float(loss_cfg["scale_pixel_loss"]) * _mse_loss_with_optional_mask(
            lst_pred, target_lst, lst_pixel_loss_mask
        )
        reg_loss = float(loss_cfg["lambda_reg_lst"]) * tv_loss(
            cur_input[0, ch_ndvi] - original_input[0, ch_ndvi],
            valid_mask=base_mask,
        )
        penalty_loss = float(loss_cfg["lambda_one_sided"]) * one_sided_decrease_penalty(
            cur_input,
            original_input,
            ch_ndvi,
            valid_mask=base_mask,
        )
        lst_loss = pixel_loss + reg_loss + penalty_loss
        lst_loss.backward()

        g = cur_input.grad
        if g is not None:
            grad_mask = torch.zeros_like(cur_input, dtype=torch.bool)
            grad_mask[0, ch_ndvi] = base_mask
            masked_grad = g * grad_mask.to(g.dtype)
            _apply_input_update(
                cur_input=cur_input,
                masked_grad=masked_grad,
                update_mask=grad_mask,
                lr=lr,
                use_sign_step=use_sign,
                optimizer=optimizer,
            )
            with torch.no_grad():
                ndvi_ch = cur_input[0, ch_ndvi]
                ndvi_ch[base_mask] = torch.clamp(ndvi_ch[base_mask], max=ndvi_max)

        if cur_input.grad is not None:
            cur_input.grad.zero_()

        with torch.no_grad():
            current_embedding = lst_model.encoder.encode(cur_input, bool_masked_pos=None)
            current_lst_pred = list(lst_model.raster_heads.values())[0](
                current_embedding, output_size
            )
            eval_pixel_loss = float(loss_cfg["scale_pixel_loss"]) * _mse_loss_with_optional_mask(
                current_lst_pred, target_lst, lst_pixel_loss_mask
            )
            eval_reg_loss = float(loss_cfg["lambda_reg_lst"]) * tv_loss(
                cur_input[0, ch_ndvi] - original_input[0, ch_ndvi],
                valid_mask=base_mask,
            )
            eval_penalty_loss = float(loss_cfg["lambda_one_sided"]) * one_sided_decrease_penalty(
                cur_input,
                original_input,
                ch_ndvi,
                valid_mask=base_mask,
            )
            eval_lst_loss = eval_pixel_loss + eval_reg_loss + eval_penalty_loss
            det_lst = lst_std * _masked_avg(
                current_lst_pred.cpu().numpy().squeeze() - original_lst.cpu().numpy().squeeze(),
                lst_metric_mask,
            )
        det_ndvi = ndvi_std * _masked_avg(
            (cur_input[0, ch_ndvi] - original_input[0, ch_ndvi]).detach().cpu().numpy(),
            lst_metric_mask,
        )
        data["delta_lst_traj"].append(det_lst)
        data["pixel_loss_traj"].append(float(eval_pixel_loss.item()))
        data["reg_loss_traj"].append(float(eval_reg_loss.item()))
        data["penalty_loss_traj"].append(float(eval_penalty_loss.item()))
        data["total_lst_loss_traj"].append(float(eval_lst_loss.item()))
        data["delta_ndvi_traj"].append(float(det_ndvi))

        if epoch % 50 == 0 or epoch == epochs - 1:
            logger.info(
                f"[cooling city {city_id}] epoch {epoch}: "
                f"lst_loss={eval_lst_loss.item():.4f} delta_lst={det_lst:.4f}"
            )

    with torch.no_grad():
        embedding = lst_model.encoder.encode(cur_input, bool_masked_pos=None)
        adjusted_lst = list(lst_model.raster_heads.values())[0](embedding, output_size)

    final_signed_det_lst = lst_std * _masked_avg(
        adjusted_lst.cpu().numpy().squeeze() - original_lst.cpu().numpy().squeeze(),
        lst_metric_mask,
    )
    final_abs_det_lst = abs(final_signed_det_lst)

    data["original_ndvi"] = original_input[0, ch_ndvi].cpu().numpy()
    data["adjusted_ndvi"] = cur_input[0, ch_ndvi].detach().cpu().numpy()
    data["original_lst"] = original_lst.cpu().numpy().squeeze()
    data["adjusted_lst"] = adjusted_lst.cpu().numpy().squeeze()
    data["target_lst"] = target_lst.cpu().numpy().squeeze()
    data["lst_mean"] = np.array(lst_mean)
    data["lst_std"] = np.array(lst_std)
    data["ndvi_mean"] = np.array(ndvi_mean)
    data["ndvi_std"] = np.array(ndvi_std)
    data["ndvi_source_mode"] = np.array(str(ndvi_source_spec.get("mode", "")))
    data["city_mask"] = city_mask
    data["ocean_mask"] = ~base_mask.cpu().numpy()
    data["city_id"] = int(city_id)
    data["final_signed_delta_lst"] = final_signed_det_lst
    data["final_delta_lst"] = final_signed_det_lst
    data["final_abs_delta_lst"] = final_abs_det_lst

    logger.info(
        f"[cooling city {city_id}] DONE: delta_lst={final_signed_det_lst:.4f} "
        f"(abs={final_abs_det_lst:.4f})"
    )
    return data


# ---------------------------------------------------------------------------
# Data loading & city lookup
# ---------------------------------------------------------------------------

def _find_city_entry(
    loaders: dict[str, DataLoader],
    *,
    scene_id: int,
    year: int,
) -> tuple[str, int] | None:
    for split_name in ("test", "val", "train"):
        dataset = loaders[split_name].dataset
        rows = getattr(dataset, "rows", None)
        if rows is None:
            continue
        for index, row in enumerate(rows):
            if int(row.get("scene_id", -1)) == scene_id and int(row.get("year", -1)) == year:
                return split_name, index
    return None


def _get_city_mask_from_batch(batch: Any) -> np.ndarray:
    raster_inputs = batch.raster_inputs
    if hasattr(batch, "city_mask"):
        return batch.city_mask.squeeze().cpu().numpy().astype(bool)
    mask_channel = raster_inputs[0, -1]
    return (mask_channel > 0).cpu().numpy()


def _last_traj_value(data: dict[str, Any], key: str) -> float | None:
    values = data.get(key)
    if values is None:
        return None
    arr = np.asarray(values).reshape(-1)
    if arr.size == 0:
        return None
    value = float(arr[-1])
    return value if math.isfinite(value) else None


def _mode_result_summary(data: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "final_signed_delta_lst": data.get("final_signed_delta_lst"),
        "final_delta_lst": data.get("final_delta_lst"),
        "final_abs_delta_lst": data.get("final_abs_delta_lst"),
    }
    final_delta_ndvi = _last_traj_value(data, "delta_ndvi_traj")
    if final_delta_ndvi is not None:
        result["final_delta_ndvi"] = final_delta_ndvi
    final_delta_pop = _last_traj_value(data, "delta_pop_traj")
    if final_delta_pop is not None:
        result["final_delta_pop"] = final_delta_pop
    for key in (
        "final_delta_traffic",
        "original_traffic",
        "target_traffic",
        "adjusted_traffic",
        "target_traffic_drop_pct",
        "traffic_drop_pct",
    ):
        if key in data:
            result[key] = data.get(key)
    return result


def _write_planning_data_export(
    *,
    output_base: Path,
    mode: str,
    year: int,
    cities: list[int],
    cfg: dict[str, Any],
    results: list[dict[str, Any]],
    city_data_items: list[dict[str, Any]],
    data_cfg: dict[str, Any],
) -> dict[str, str]:
    if not city_data_items:
        return {}
    data_path = output_base / str(data_cfg.get("data_file", "planning_data.pkl"))
    package = {
        "schema_version": 1,
        "mode": mode,
        "year": year,
        "cities": [int(city) for city in cities],
        "config": cfg,
        "results": results,
        "city_data": city_data_items,
    }
    with open(data_path, "wb") as f:
        pickle.dump(package, f)

    return {"planning_data": str(data_path)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _mode_has_joint(mode: str) -> bool:
    return mode in {"joint", "both"}


def _mode_has_cooling_only(mode: str) -> bool:
    return mode in {"cooling_only", "both"}


def _save_city_npz(*, output_base: Path, city_id: int, data: dict[str, Any]) -> Path:
    npz_path = output_base / f"city_{city_id}.npz"
    save_arrays = {
        k: np.asarray(v)
        for k, v in data.items()
        if isinstance(v, (np.ndarray, list, tuple, Number, np.bool_))
    }
    save_arrays["city_id"] = np.array(city_id)
    np.savez_compressed(str(npz_path), **save_arrays)
    return npz_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = _load_experiment_config(args)
    device = _resolve_device(args.device)
    output_base = _resolve_output_base(cfg["output_dir"])
    output_base.mkdir(parents=True, exist_ok=True)

    mode = str(cfg.get("mode", "joint")).strip().lower()
    if mode not in {"joint", "cooling_only", "both"}:
        raise ValueError("planning_exp.mode must be one of: joint, cooling_only, both")
    cfg["mode"] = mode
    cities = cfg["cities"]
    year = int(cfg.get("year", 2020))
    data_cfg = dict(cfg.get("data_export", {}))
    data_export_enabled = bool(data_cfg.get("enabled", True))
    save_city_npz = bool(cfg.get("outputs", {}).get("save_city_npz", True))

    logger.info(f"Planning experiment: mode={mode}, cities={cities}, year={year}")
    logger.info(f"Output: {output_base}")
    logger.info(
        "Exports: "
        f"planning_data={'enabled' if data_export_enabled else 'disabled'}, "
        f"city_npz={'enabled' if save_city_npz else 'disabled'}"
    )

    hydra_cfg_lst = _compose_hydra_config(cfg["lst_pipeline"], args.overrides, args.num_workers)
    _apply_checkpoint_overrides(hydra_cfg_lst, cfg)
    random_seed = int(hydra_cfg_lst["experiment"]["random_seed"])
    set_reproducible(random_seed)
    logger.info(f"Random seed: {random_seed}")

    pipeline_spec_lst = PipelineSpec.from_mapping(hydra_cfg_lst["pipeline"])
    validate_eump_downstream_spec(pipeline_spec_lst)

    train_dl, val_dl, test_dl, batch_schema_lst = build_eump_dataloaders(
        data_config=hydra_cfg_lst["data"],
        pipeline_spec=pipeline_spec_lst,
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
    raster_input_channel_names = tuple(
        str(n) for n in batch_schema_lst.raster_input_channel_names
    )

    ch_ndvi, _, ndvi_spec = _resolve_source_channel(
        source_name=cfg["update_channels"]["lst_step"],
        preprocessing_spec=preprocessing_spec,
        raster_input_channel_names=raster_input_channel_names,
    )
    ch_pop, _, pop_spec = _resolve_source_channel(
        source_name=cfg["update_channels"]["traffic_step"],
        preprocessing_spec=preprocessing_spec,
        raster_input_channel_names=raster_input_channel_names,
    )

    lst_target_name = str(hydra_cfg_lst["pipeline"]["targets"]["raster"][0]["source"])
    lst_target_spec = preprocessing_spec["sources"].get(lst_target_name, {})

    logger.info(f"Channels: NDVI={ch_ndvi}, POP={ch_pop}")

    encoder, lst_model, traffic_head = build_planning_models(
        cfg=cfg,
        hydra_cfg_lst=hydra_cfg_lst,
        batch_schema_lst=batch_schema_lst,
        device=device,
    )

    results: list[dict[str, Any]] = []
    city_data_items: list[dict[str, Any]] = []
    start_time = time.time()

    for idx, city_id in enumerate(cities):
        logger.info(f"Processing city {city_id} ({idx + 1}/{len(cities)})")
        entry = _find_city_entry(loaders, scene_id=int(city_id), year=year)
        if entry is None:
            logger.warning(f"City {city_id} year {year} not found, skipping")
            continue

        split_name, sample_index = entry
        dataset = loaders[split_name].dataset
        sample = dataset[sample_index]

        raster_inputs = torch.from_numpy(sample.raster_inputs).unsqueeze(0).float().to(device)
        city_mask = sample.city_mask.squeeze().astype(bool)
        raster_valid = sample.raster_valid_mask.astype(bool)
        valid_mask = raster_valid.all(axis=0).squeeze()

        original_input = raster_inputs.clone().detach()

        with torch.no_grad():
            output_size = tuple(int(v) for v in raster_inputs.shape[-2:])
            embedding = encoder.encode(raster_inputs, bool_masked_pos=None)
            original_lst = list(lst_model.raster_heads.values())[0](embedding, output_size)
            if _mode_has_joint(mode):
                original_traffic = traffic_head(embedding)
            else:
                original_traffic = torch.zeros(1, device=device)

        lst_valid = sample.raster_targets[lst_target_name][1].all(axis=0).squeeze().astype(bool)
        valid_mask = valid_mask & lst_valid
        base_mask = torch.from_numpy(valid_mask).to(device)

        pop_channel = original_input[0, ch_pop]
        pop_physical = inverse_processed_tensor(pop_channel.unsqueeze(0), pop_spec).squeeze(0)

        target_lst, target_traffic = compute_targets(
            original_lst,
            original_traffic,
            pop_physical,
            cfg,
            lst_target_spec,
            city_mask,
            valid_mask=base_mask,
        )

        joint_data: dict[str, Any] | None = None
        cooling_data: dict[str, Any] | None = None

        if _mode_has_joint(mode):
            joint_input = original_input.clone().detach().requires_grad_(True)
            joint_data = optimize_city(
                cur_input=joint_input,
                original_input=original_input,
                city_mask=city_mask,
                base_mask=base_mask,
                target_lst=target_lst,
                target_traffic=target_traffic,
                original_lst=original_lst,
                original_traffic=original_traffic,
                lst_model=lst_model,
                traffic_model=traffic_head,
                cfg=cfg,
                ch_ndvi=ch_ndvi,
                ch_pop=ch_pop,
                pop_source_spec=pop_spec,
                ndvi_source_spec=ndvi_spec,
                lst_target_spec=lst_target_spec,
                city_id=city_id,
            )

        if _mode_has_cooling_only(mode):
            cooling_input = original_input.clone().detach().requires_grad_(True)
            cooling_data = optimize_cooling_only(
                cur_input=cooling_input,
                original_input=original_input,
                city_mask=city_mask,
                base_mask=base_mask,
                target_lst=target_lst,
                original_lst=original_lst,
                lst_model=lst_model,
                cfg=cfg,
                ch_ndvi=ch_ndvi,
                ndvi_source_spec=ndvi_spec,
                lst_target_spec=lst_target_spec,
                city_id=city_id,
            )

        if save_city_npz:
            npz_data = joint_data if joint_data is not None else cooling_data
            if npz_data is not None:
                npz_path = _save_city_npz(
                    output_base=output_base,
                    city_id=int(city_id),
                    data=npz_data,
                )
                source_mode = "joint" if joint_data is not None else "cooling_only"
                logger.info(f"Saved {source_mode} per-city npz: {npz_path}")

        result_item: dict[str, Any] = {"city_id": int(city_id)}
        city_data_item: dict[str, Any] = {"city_id": int(city_id)}
        if joint_data is not None:
            result_item["joint"] = _mode_result_summary(joint_data)
            city_data_item["joint"] = joint_data
        if cooling_data is not None:
            result_item["cooling_only"] = _mode_result_summary(cooling_data)
            city_data_item["cooling_only"] = cooling_data

        results.append(result_item)
        city_data_items.append(city_data_item)

    elapsed = time.time() - start_time
    data_outputs = (
        _write_planning_data_export(
            output_base=output_base,
            mode=mode,
            year=year,
            cities=[int(c) for c in cities],
            cfg=cfg,
            results=results,
            city_data_items=city_data_items,
            data_cfg=data_cfg,
        )
        if data_export_enabled
        else {}
    )
    for key, path in data_outputs.items():
        logger.info(f"Saved {key}: {path}")
    summary = {
        "mode": mode,
        "cities": [int(c) for c in cities],
        "year": year,
        "config": cfg,
        "results": results,
        "data_outputs": data_outputs,
        "elapsed_seconds": elapsed,
    }
    summary_path = output_base / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Summary saved: {summary_path} ({elapsed:.1f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
