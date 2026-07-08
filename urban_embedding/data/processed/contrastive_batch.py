from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(slots=True)
class ContrastivePair:
    """One contrastive training sample: two stochastic views of the same crop.

    Both views derive from the exact same materialized crop tensor so the
    positive pair is defined by ``(scene_id, year, crop_box)`` plus
    independent stochastic corruptions. No targets are carried; contrastive
    pretraining has no reconstruction aux.
    """

    view_a: torch.Tensor  # (C, H, W)
    view_b: torch.Tensor  # (C, H, W)
    valid_mask_a: torch.Tensor  # (C, H, W) bool
    valid_mask_b: torch.Tensor  # (C, H, W) bool
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ContrastiveBatch:
    """Batched contrastive pair. Views stacked separately for the trainer."""

    view_a: torch.Tensor  # (B, C, H, W)
    view_b: torch.Tensor  # (B, C, H, W)
    valid_mask_a: torch.Tensor  # (B, C, H, W)
    valid_mask_b: torch.Tensor  # (B, C, H, W)
    meta: list[dict[str, Any]] = field(default_factory=list)

    def to(self, device: torch.device | str, non_blocking: bool = True) -> "ContrastiveBatch":
        return ContrastiveBatch(
            view_a=self.view_a.to(device, non_blocking=non_blocking),
            view_b=self.view_b.to(device, non_blocking=non_blocking),
            valid_mask_a=self.valid_mask_a.to(device, non_blocking=non_blocking),
            valid_mask_b=self.valid_mask_b.to(device, non_blocking=non_blocking),
            meta=self.meta,
        )


__all__ = ["ContrastiveBatch", "ContrastivePair"]
