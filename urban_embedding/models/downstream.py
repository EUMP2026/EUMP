"""Compatibility exports for Swin downstream models."""

from urban_embedding.models.eump.swin.downstream import (
    RASTER_HEAD_TYPES,
    RASTER_UPSAMPLE_MODES,
    EumpLightDualHeadModel,
    EncoderLoadReport,
    PooledScalarHead,
    ProgressiveUpsampleRasterHead,
    SegFormerStyleUpsampleHead,
    build_eump_downstream_model,
    load_eump_reconstruction_encoder_weights,
)

__all__ = [
    "EumpLightDualHeadModel",
    "EncoderLoadReport",
    "PooledScalarHead",
    "ProgressiveUpsampleRasterHead",
    "RASTER_HEAD_TYPES",
    "RASTER_UPSAMPLE_MODES",
    "SegFormerStyleUpsampleHead",
    "build_eump_downstream_model",
    "load_eump_reconstruction_encoder_weights",
]
