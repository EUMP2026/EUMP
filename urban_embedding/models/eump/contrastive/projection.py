from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """SimCLR-style projection head: LN -> Linear -> GELU -> Linear -> L2-norm.

    Operates on the globally-pooled encoder embedding. Output is L2-normalized
    so the downstream NT-Xent loss can use cosine similarity directly.
    """

    def __init__(self, *, in_channels: int, hidden_dim: int, projection_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        z = self.net(embedding)
        return F.normalize(z, dim=-1)


__all__ = ["ProjectionHead"]
