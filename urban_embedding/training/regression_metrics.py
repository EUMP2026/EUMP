from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch

from .losses import build_loss_mask


def empty_regression_stats(*, device: torch.device) -> torch.Tensor:
    return torch.zeros(4, dtype=torch.float64, device=device)


def accumulate_regression_stats(stats: torch.Tensor, prediction: torch.Tensor, target: torch.Tensor) -> None:
    if prediction.numel() == 0 or target.numel() == 0:
        return
    prediction_flat = prediction.reshape(-1).to(dtype=torch.float64)
    target_flat = target.reshape(-1).to(dtype=torch.float64)
    if prediction_flat.shape != target_flat.shape:
        raise ValueError(
            f"Prediction shape {tuple(prediction_flat.shape)} does not match target shape {tuple(target_flat.shape)}"
        )
    stats[0] += float(target_flat.numel())
    stats[1] += target_flat.sum()
    stats[2] += (target_flat * target_flat).sum()
    diff = prediction_flat - target_flat
    stats[3] += (diff * diff).sum()


def summarize_regression_stats(
    stats: torch.Tensor,
    *,
    variance_epsilon: float = 1.0e-12,
) -> tuple[dict[str, int | float | None], bool]:
    count = int(round(float(stats[0].detach().cpu().item())))
    if count <= 0:
        return {"count": 0, "rmse": None, "r2": None}, False

    sum_target = float(stats[1].detach().cpu().item())
    sum_target_sq = float(stats[2].detach().cpu().item())
    sum_sq_error = float(stats[3].detach().cpu().item())
    rmse = math.sqrt(max(sum_sq_error, 0.0) / count)
    sst = sum_target_sq - (sum_target * sum_target) / count
    if sst <= variance_epsilon:
        return {"count": count, "rmse": rmse, "r2": None}, True
    return {"count": count, "rmse": rmse, "r2": 1.0 - (sum_sq_error / sst)}, False


def inverse_processed_tensor(values: torch.Tensor, source_spec: Mapping[str, Any]) -> torch.Tensor:
    mode = str(source_spec["mode"])
    if mode == "pass_through":
        return values
    if mode == "divide":
        return values * float(source_spec["divisor"])
    if mode in {"zscore", "log_zscore"}:
        stats = source_spec.get("stats")
        if not stats:
            raise ValueError(f"Missing preprocessing stats for source {source_spec.get('source_name')!r}")
        restored = values * float(stats["std"]) + float(stats["mean"])
        if mode == "log_zscore":
            restored = torch.pow(restored.new_tensor(10.0), restored) - 1.0
        return restored
    raise ValueError(
        "EUMP regression metrics support only continuous preprocessing modes "
        f"'pass_through', 'divide', 'zscore', and 'log_zscore', got {mode!r}"
    )


def _evaluation_mask(
    *,
    mask_strategy: str,
    valid_mask: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
    city_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    effective_valid = valid_mask.bool() & torch.isfinite(target) & torch.isfinite(prediction)
    loss_mask = build_loss_mask(
        mask_strategy,
        effective_valid,
        city_mask=city_mask,
        masked_patch=None,
    )
    if loss_mask is None:
        return effective_valid
    return loss_mask > 0


def scalar_metric_tensors(
    *,
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    mask_strategy: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = _evaluation_mask(
        mask_strategy=mask_strategy,
        valid_mask=valid_mask,
        prediction=prediction,
        target=target,
        city_mask=None,
    )
    return prediction[mask], target[mask]


def raster_global_metric_tensors(
    *,
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    city_mask: torch.Tensor,
    mask_strategy: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = _evaluation_mask(
        mask_strategy=mask_strategy,
        valid_mask=valid_mask,
        prediction=prediction,
        target=target,
        city_mask=city_mask,
    )
    return prediction[mask], target[mask]


def raster_city_mean_metric_tensors(
    *,
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    city_mask: torch.Tensor,
    mask_strategy: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = _evaluation_mask(
        mask_strategy=mask_strategy,
        valid_mask=valid_mask,
        prediction=prediction,
        target=target,
        city_mask=city_mask,
    )
    city = city_mask > 0
    while city.dim() < mask.dim():
        city = city.unsqueeze(1)
    city = city.expand_as(mask)
    city_masked = mask & city
    weights = city_masked.to(dtype=prediction.dtype)
    support = weights.sum(dim=(-2, -1))
    valid = support > 0
    if not bool(valid.any().detach().cpu()):
        empty = prediction.new_empty((0,))
        return empty, empty

    pred_mean = (prediction * weights).sum(dim=(-2, -1)) / support.clamp_min(1.0e-8)
    target_mean = (target * weights).sum(dim=(-2, -1)) / support.clamp_min(1.0e-8)
    return pred_mean[valid], target_mean[valid]


__all__ = [
    "accumulate_regression_stats",
    "empty_regression_stats",
    "inverse_processed_tensor",
    "raster_city_mean_metric_tensors",
    "raster_global_metric_tensors",
    "scalar_metric_tensors",
    "summarize_regression_stats",
]
