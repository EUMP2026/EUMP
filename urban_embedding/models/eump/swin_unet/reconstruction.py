from __future__ import annotations

import torch
import torch.nn as nn

from urban_embedding.data.processed.pipeline_spec import ProcessedBatchSchema
from urban_embedding.models.backbones.swinv2_masked import SwinWithConv
from urban_embedding.models.eump.swin.reconstruction import build_eump_reconstruction_output_routing


class _UNetUpBlock(nn.Module):
    def __init__(self, *, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.up(x))


class _UNetDecoder(nn.Module):
    """Symmetric UNet-style decoder upsampling bottleneck back to input resolution.

    Encoder output (B, C_deep, H/32, W/32) -> (B, out_channels, H, W) via 5 stages
    of 2x nearest upsampling plus double conv. Channels halve each stage down to
    ``min_channels``.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        stages: int = 5,
        min_channels: int = 64,
    ) -> None:
        super().__init__()
        schedule = [in_channels]
        for _ in range(stages):
            schedule.append(max(min_channels, schedule[-1] // 2))
        self.blocks = nn.ModuleList(
            _UNetUpBlock(in_channels=schedule[i], out_channels=schedule[i + 1])
            for i in range(stages)
        )
        self.final_conv = nn.Conv2d(schedule[-1], out_channels, kernel_size=1)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        x = embedding
        for block in self.blocks:
            x = block(x)
        return self.final_conv(x)


def _validate_schema_input_reconstruction_only(batch_schema: ProcessedBatchSchema) -> None:
    if batch_schema.scalar_targets:
        raise ValueError(
            "Swin-UNet reconstruction does not support scalar targets; "
            "configure pipeline without `targets.scalar` entries."
        )
    routing = build_eump_reconstruction_output_routing(batch_schema)
    if routing.raster_classification_channels > 0:
        raise ValueError(
            "Swin-UNet reconstruction does not support raster classification targets"
        )
    if routing.raster_regression_channels <= 0:
        raise ValueError(
            "Swin-UNet reconstruction requires at least one raster regression target"
        )
    if routing.raster_regression_channels != routing.raster_output_channels:
        raise ValueError(
            "Swin-UNet reconstruction must route every raster channel to regression; "
            f"got regression_channels={routing.raster_regression_channels} "
            f"vs output_channels={routing.raster_output_channels}"
        )


class EumpSwinUnetReconstructionModel(nn.Module):
    """Swin encoder + UNet-style symmetric decoder for MAE reconstruction.

    Restricted to input-reconstruction style pretraining: only raster
    regression targets (typically just ``input_reconstruction``), no scalar
    targets, no auxiliary classification heads. This keeps the decoder a
    single stack so the comparison with the Swin PixelShuffle baseline is
    clean.

    Encoder interface matches ``SwinWithConv`` so frozen-encoder downstream
    training can load the checkpoint unchanged.
    """

    def __init__(
        self,
        model_path: str,
        input_channels: int,
        *,
        batch_schema: ProcessedBatchSchema,
        load_pretrained: bool = True,
        decoder_stages: int = 5,
        decoder_min_channels: int = 64,
        **_ignored: object,
    ) -> None:
        super().__init__()
        _validate_schema_input_reconstruction_only(batch_schema)
        routing = build_eump_reconstruction_output_routing(batch_schema)

        self.output_routing = routing
        self.raster_output_channels = routing.raster_output_channels
        self.backbone = SwinWithConv(
            model_path=model_path,
            input_channels=input_channels,
            output_channels=0,
            load_pretrained=load_pretrained,
        )
        self.config = self.backbone.config
        num_features = self.config.embed_dim * (2 ** (self.config.num_layers - 1))
        self.decoder = _UNetDecoder(
            in_channels=num_features,
            out_channels=routing.raster_output_channels,
            stages=int(decoder_stages),
            min_channels=int(decoder_min_channels),
        )

    def forward(
        self,
        input_tensor: torch.Tensor,
        bool_masked_pos: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        embedding = self.backbone.encode(input_tensor, bool_masked_pos=bool_masked_pos)
        raster_prediction = self.decoder(embedding)
        return raster_prediction, {}


__all__ = ["EumpSwinUnetReconstructionModel"]

