from __future__ import annotations

import torch
import torch.nn.functional as F


SUPPORTED_LOSSES = {"mse", "mae", "huber", "bce", "ce", "kl", "hurdle_mse", "hurdle_mae", "hurdle_huber"}
SUPPORTED_MASKS = {
    "none",
    "valid",
    "city",
    "masked",
    "valid_and_city",
    "valid_and_masked",
    "city_and_masked",
    "valid_and_city_and_masked",
}


def unreduced_loss(kind: str, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    kind = kind.lower()
    if kind == "mse":
        return (prediction - target) ** 2
    if kind == "mae":
        return torch.abs(prediction - target)
    if kind == "huber":
        return F.huber_loss(prediction, target, delta=1.0, reduction="none")
    if kind == "bce":
        return F.binary_cross_entropy_with_logits(prediction, target, reduction="none")
    if kind == "ce":
        if prediction.dim() < 2 or prediction.shape[1] < 2:
            raise ValueError("CE loss requires prediction to have at least 2 class channels")
        if target.shape == prediction.shape:
            target = target.argmax(dim=1)
        elif target.dim() == prediction.dim() and target.shape[1] == 1:
            target = target[:, 0]
        loss = F.cross_entropy(prediction, target.long(), reduction="none")
        return loss.unsqueeze(1)
    if kind == "kl":
        if prediction.dim() < 2 or prediction.shape[1] < 2:
            raise ValueError("KL loss requires prediction to have at least 2 distribution channels")
        if target.shape != prediction.shape:
            raise ValueError(f"KL loss requires target shape {prediction.shape}, got {target.shape}")
        if bool((target < -1e-6).any().detach().cpu()):
            raise ValueError("KL loss target distribution must be non-negative")
        return F.kl_div(F.log_softmax(prediction, dim=1), target, reduction="none").sum(dim=1, keepdim=True)
    if kind in {"hurdle_mse", "hurdle_mae", "hurdle_huber"}:
        raise ValueError("Hurdle losses require logits and are handled by urban_embedding.training.eump_loss")
    raise ValueError(f"Unsupported loss kind {kind!r}; expected one of {sorted(SUPPORTED_LOSSES)}")


def _align_mask(mask: torch.Tensor, loss: torch.Tensor) -> torch.Tensor:
    while mask.dim() < loss.dim():
        mask = mask.unsqueeze(1)
    if mask.dim() != loss.dim():
        raise ValueError(f"Mask rank {mask.dim()} is incompatible with loss rank {loss.dim()}")
    if mask.shape == loss.shape:
        return mask
    if mask.dim() >= 2 and mask.shape[0] == loss.shape[0] and mask.shape[2:] == loss.shape[2:]:
        if mask.shape[1] == 1:
            return mask.expand_as(loss)
        if loss.shape[1] == 1:
            return mask.any(dim=1, keepdim=True).to(dtype=mask.dtype)
    return torch.broadcast_to(mask, loss.shape)


def build_loss_mask(
    strategy: str,
    valid_mask: torch.Tensor | None,
    *,
    city_mask: torch.Tensor | None = None,
    masked_patch: torch.Tensor | None = None,
) -> torch.Tensor | None:
    strategy = strategy.lower()
    if strategy not in SUPPORTED_MASKS:
        raise ValueError(f"Unsupported mask strategy {strategy!r}; expected one of {sorted(SUPPORTED_MASKS)}")
    if strategy == "none":
        return None

    mask: torch.Tensor | None = None
    if "valid" in strategy:
        if valid_mask is None:
            raise ValueError(f"Mask strategy {strategy!r} requires valid_mask")
        mask = valid_mask.bool()
    if "city" in strategy:
        if city_mask is None:
            raise ValueError(f"Mask strategy {strategy!r} requires city_mask")
        city = city_mask > 0
        mask = city if mask is None else (mask & city)
    if "masked" in strategy:
        if masked_patch is not None:
            patch = masked_patch.bool()
            mask = patch if mask is None else (mask & patch)
    return None if mask is None else mask.float()


def reduce_weighted_loss(loss: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is None:
        return loss.mean()
    aligned_mask = _align_mask(mask.float(), loss)
    weighted = loss * aligned_mask
    denom = aligned_mask.sum().clamp_min(1e-8)
    return weighted.sum() / denom


def compute_weighted_loss(
    kind: str,
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    return reduce_weighted_loss(unreduced_loss(kind, prediction, target), mask=mask)


__all__ = [
    "SUPPORTED_LOSSES",
    "SUPPORTED_MASKS",
    "build_loss_mask",
    "compute_weighted_loss",
    "reduce_weighted_loss",
    "unreduced_loss",
]
