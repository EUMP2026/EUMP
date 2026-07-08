from __future__ import annotations

import torch
import torch.nn as nn

from urban_embedding.data.processed.pipeline_spec import ProcessedBatchSchema
from urban_embedding.models.eump.swin.reconstruction import build_eump_reconstruction_output_routing
from urban_embedding.models.eump.swin_unet.reconstruction import _UNetDecoder

from .encoder import ConvNeXtEncoder


def _validate_schema_input_reconstruction_only(batch_schema: ProcessedBatchSchema) -> None:
    if batch_schema.scalar_targets:
        raise ValueError(
            "ConvNeXt-UNet reconstruction does not support scalar targets; "
            "configure pipeline without `targets.scalar` entries."
        )
    routing = build_eump_reconstruction_output_routing(batch_schema)
    if routing.raster_classification_channels > 0:
        raise ValueError("ConvNeXt-UNet reconstruction does not support raster classification targets")
    if routing.raster_regression_channels <= 0:
        raise ValueError("ConvNeXt-UNet reconstruction requires at least one raster regression target")
    if routing.raster_regression_channels != routing.raster_output_channels:
        raise ValueError(
            "ConvNeXt-UNet reconstruction must route every raster channel to regression; "
            f"got regression_channels={routing.raster_regression_channels} "
            f"vs output_channels={routing.raster_output_channels}"
        )


class EumpConvNextUnetReconstructionModel(nn.Module):
    """ConvNeXt encoder + UNet-style symmetric decoder for MAE reconstruction.

    Designed to be a pure encoder-architecture swap of Swin-UNet: same decoder
    stack, same input-reconstruction-only contract, same ``(B, C_deep, H/32,
    W/32)`` bottleneck. Only raster regression targets are supported; scalar
    and auxiliary heads are intentionally omitted to keep the comparison
    clean.

    Masking strategy at pretrain time: the trainer zero-fills masked input
    patches before calling the encoder (ConvNeXt is fully convolutional, so
    there is no token-drop equivalent).
    """

    def __init__(
        self,
        input_channels: int,
        *,
        batch_schema: ProcessedBatchSchema,
        depths: tuple[int, int, int, int] = (3, 3, 27, 3),
        dims: tuple[int, int, int, int] = (192, 384, 768, 1536),
        drop_path_rate: float = 0.0,
        decoder_stages: int = 5,
        decoder_min_channels: int = 64,
        **_ignored: object,
    ) -> None:
        super().__init__()
        _validate_schema_input_reconstruction_only(batch_schema)
        routing = build_eump_reconstruction_output_routing(batch_schema)

        self.output_routing = routing
        self.raster_output_channels = routing.raster_output_channels
        self.backbone = ConvNeXtEncoder(
            input_channels=input_channels,
            depths=depths,
            dims=dims,
            drop_path_rate=float(drop_path_rate),
        )

        from types import SimpleNamespace
        self.config = SimpleNamespace(
            patch_size=self.backbone.patch_size,
            encoder_stride=self.backbone.encoder_stride,
        )
        self.decoder = _UNetDecoder(
            in_channels=self.backbone.num_features,
            out_channels=routing.raster_output_channels,
            stages=int(decoder_stages),
            min_channels=int(decoder_min_channels),
        )

    def encode(
        self,
        input_tensor: torch.Tensor,
        bool_masked_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.backbone.encode(input_tensor, bool_masked_pos=bool_masked_pos)

    def forward(
        self,
        input_tensor: torch.Tensor,
        bool_masked_pos: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        embedding = self.backbone.encode(input_tensor, bool_masked_pos=bool_masked_pos)
        raster_prediction = self.decoder(embedding)
        return raster_prediction, {}


__all__ = ["EumpConvNextUnetReconstructionModel"]
