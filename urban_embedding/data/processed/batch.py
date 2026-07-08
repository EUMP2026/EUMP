from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(slots=True)
class TensorBlock:
    data: torch.Tensor
    valid_mask: torch.Tensor
    names: tuple[str, ...] = ()

    def to(self, device: torch.device | str, non_blocking: bool = True) -> "TensorBlock":
        return TensorBlock(
            data=self.data.to(device, non_blocking=non_blocking),
            valid_mask=self.valid_mask.to(device, non_blocking=non_blocking),
            names=self.names,
        )


@dataclass(slots=True)
class ProcessedModelBatch:
    raster_inputs: torch.Tensor
    raster_valid_mask: torch.Tensor
    scalar_inputs: torch.Tensor
    scalar_valid_mask: torch.Tensor
    city_mask: torch.Tensor
    raster_targets: dict[str, TensorBlock] = field(default_factory=dict)
    scalar_targets: dict[str, TensorBlock] = field(default_factory=dict)
    meta: list[dict[str, Any]] = field(default_factory=list)

    def to(self, device: torch.device | str, non_blocking: bool = True) -> "ProcessedModelBatch":
        raster_inputs = self.raster_inputs.to(device, non_blocking=non_blocking)
        raster_valid_mask = self.raster_valid_mask.to(device, non_blocking=non_blocking)
        scalar_inputs = self.scalar_inputs.to(device, non_blocking=non_blocking)
        scalar_valid_mask = self.scalar_valid_mask.to(device, non_blocking=non_blocking)
        city_mask = self.city_mask.to(device, non_blocking=non_blocking)

        raster_targets: dict[str, TensorBlock] = {}
        for name, block in self.raster_targets.items():
            if block.data is self.raster_inputs and block.valid_mask is self.raster_valid_mask:
                raster_targets[name] = TensorBlock(
                    data=raster_inputs,
                    valid_mask=raster_valid_mask,
                    names=block.names,
                )
            else:
                raster_targets[name] = block.to(device, non_blocking=non_blocking)

        return ProcessedModelBatch(
            raster_inputs=raster_inputs,
            raster_valid_mask=raster_valid_mask,
            scalar_inputs=scalar_inputs,
            scalar_valid_mask=scalar_valid_mask,
            city_mask=city_mask,
            raster_targets=raster_targets,
            scalar_targets={
                name: block.to(device, non_blocking=non_blocking)
                for name, block in self.scalar_targets.items()
            },
            meta=self.meta,
        )


__all__ = ["ProcessedModelBatch", "TensorBlock"]
