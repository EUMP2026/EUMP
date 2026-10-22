from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn

from urban_embedding.models.backbones.swinv2_masked import SwinWithConv

from .projection import ProjectionHead


class EumpSwinMoCoModel(nn.Module):
    """Swin encoder with MoCo-style momentum target for contrastive pretraining.

    Online branch: backbone → pool → projection (has gradient).
    Momentum branch: EMA copy of online (no gradient, updated after each step).
    """

    def __init__(
        self,
        model_path: str,
        input_channels: int,
        *,
        load_pretrained: bool = True,
        projection_hidden_dim: int = 2048,
        projection_dim: int = 128,
        momentum: float = 0.996,
    ) -> None:
        super().__init__()
        self.backbone = SwinWithConv(
            model_path=model_path,
            input_channels=input_channels,
            output_channels=0,
            load_pretrained=load_pretrained,
        )
        self.config = self.backbone.config
        num_features = self.config.embed_dim * (2 ** (self.config.num_layers - 1))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = ProjectionHead(
            in_channels=num_features,
            hidden_dim=int(projection_hidden_dim),
            projection_dim=int(projection_dim),
        )

        self.momentum_backbone = deepcopy(self.backbone)
        self.momentum_projection = deepcopy(self.projection)
        for p in self.momentum_backbone.parameters():
            p.requires_grad = False
        for p in self.momentum_projection.parameters():
            p.requires_grad = False
        self.momentum = momentum

    def encode(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.backbone.encode(input_tensor, bool_masked_pos=None)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        embedding = self.backbone.encode(input_tensor, bool_masked_pos=None)
        pooled = self.pool(embedding).flatten(1)
        return self.projection(pooled)

    @torch.no_grad()
    def forward_momentum(self, input_tensor: torch.Tensor) -> torch.Tensor:
        embedding = self.momentum_backbone.encode(input_tensor, bool_masked_pos=None)
        pooled = self.pool(embedding).flatten(1)
        return self.momentum_projection(pooled)

    @torch.no_grad()
    def update_momentum(self) -> None:
        m = self.momentum
        for p_o, p_m in zip(self.backbone.parameters(), self.momentum_backbone.parameters()):
            p_m.data.mul_(m).add_(p_o.data, alpha=1 - m)
        for p_o, p_m in zip(self.projection.parameters(), self.momentum_projection.parameters()):
            p_m.data.mul_(m).add_(p_o.data, alpha=1 - m)


__all__ = ["EumpSwinMoCoModel"]
