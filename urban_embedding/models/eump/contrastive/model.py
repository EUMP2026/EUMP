from __future__ import annotations

import torch
import torch.nn as nn

from urban_embedding.models.backbones.swinv2_masked import SwinWithConv

from .projection import ProjectionHead


class EumpSwinContrastiveModel(nn.Module):
    """Swin encoder + global-avg-pool + projection head for SimCLR-style pretraining.

    The encoder is identical to :class:`EumpSwinReconstructionModel` (same
    ``SwinWithConv`` backbone, same ``conv_layer`` input projection), so the
    resulting checkpoint is directly consumable by the frozen-encoder
    downstream loader. The projection head is only used at pretrain time and
    is saved under a separate key so the downstream loader skips it.

    ``forward`` accepts the full concatenated batch of stacked views
    ``(2*B, C, H, W)``; the trainer is responsible for the view split.
    """

    def __init__(
        self,
        model_path: str,
        input_channels: int,
        *,
        load_pretrained: bool = True,
        projection_hidden_dim: int = 2048,
        projection_dim: int = 128,
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

    def encode(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.backbone.encode(input_tensor, bool_masked_pos=None)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        embedding = self.backbone.encode(input_tensor, bool_masked_pos=None)
        pooled = self.pool(embedding).flatten(1)
        return self.projection(pooled)


__all__ = ["EumpSwinContrastiveModel"]
