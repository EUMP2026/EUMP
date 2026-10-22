from __future__ import annotations

import torch
import torch.nn as nn


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm over channel dim for 4D tensors (B, C, H, W)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        return x.permute(0, 3, 1, 2).contiguous()


class ConvNeXtBlock(nn.Module):
    """ConvNeXt v1 block: DWConv 7x7 -> LN -> 1x1 -> GELU -> 1x1, with residual."""

    def __init__(self, dim: int, *, mlp_ratio: float = 4.0, drop_path: float = 0.0) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim, eps=1e-6)
        self.pwconv1 = nn.Conv2d(dim, hidden, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(hidden, dim, kernel_size=1)
        self.drop_path = nn.Identity() if drop_path <= 0.0 else nn.Dropout2d(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return residual + self.drop_path(x)


class _Downsample(nn.Module):
    def __init__(self, *, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.norm = LayerNorm2d(in_dim, eps=1e-6)
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.norm(x))


class ConvNeXtEncoder(nn.Module):
    """4-stage ConvNeXt encoder mirroring the Swin encode() contract.

    Input: ``(B, input_channels, H, W)`` -> ``conv_layer`` reshapes to 3-channel
    for compatibility with the Swin checkpoint key layout, then a 4x4 stride-4
    patch embedding brings spatial down to ``H/4, W/4``. Three downsample
    stages (each 2x) yield a bottleneck of ``(B, C_deep, H/32, W/32)``,
    matching Swin encoder stride.
    """

    def __init__(
        self,
        *,
        input_channels: int,
        depths: tuple[int, int, int, int] = (3, 3, 27, 3),
        dims: tuple[int, int, int, int] = (192, 384, 768, 1536),
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv_layer = nn.Conv2d(
            in_channels=input_channels,
            out_channels=3,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0], eps=1e-6),
        )
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        total_blocks = sum(depths)
        drop_path_schedule = [drop_path_rate * i / max(1, total_blocks - 1) for i in range(total_blocks)]
        block_cursor = 0
        for stage_idx, (depth, dim) in enumerate(zip(depths, dims)):
            blocks = nn.Sequential(
                *[
                    ConvNeXtBlock(dim=dim, drop_path=drop_path_schedule[block_cursor + i])
                    for i in range(depth)
                ]
            )
            self.stages.append(blocks)
            block_cursor += depth
            if stage_idx < len(depths) - 1:
                self.downsamples.append(_Downsample(in_dim=dim, out_dim=dims[stage_idx + 1]))

        self.num_features = int(dims[-1])
        self.encoder_stride = 32
        self.patch_size = 4

    def encode(
        self,
        input_tensor: torch.Tensor,
        bool_masked_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.conv_layer(input_tensor)
        x = self.patch_embed(x)
        for stage_idx, stage in enumerate(self.stages):
            x = stage(x)
            if stage_idx < len(self.downsamples):
                x = self.downsamples[stage_idx](x)
        return x


__all__ = ["ConvNeXtEncoder", "ConvNeXtBlock", "LayerNorm2d"]

