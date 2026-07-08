"""Compatibility exports for Swin reconstruction models."""

from urban_embedding.models.eump.swin.reconstruction import (
    CLASSIFICATION_LOSSES,
    REGRESSION_LOSSES,
    SCALAR_HEAD_MODES,
    EumpReconstructionOutputRouting,
    EumpSwinReconstructionModel,
    RasterClassificationHead,
    RasterPixelShuffleHead,
    RasterRegressionHead,
    ScalarPooledHead,
    SwinWithConv,
    build_eump_reconstruction_output_routing,
)

__all__ = [
    "CLASSIFICATION_LOSSES",
    "EumpReconstructionOutputRouting",
    "EumpSwinReconstructionModel",
    "REGRESSION_LOSSES",
    "RasterClassificationHead",
    "RasterPixelShuffleHead",
    "RasterRegressionHead",
    "SCALAR_HEAD_MODES",
    "ScalarPooledHead",
    "SwinWithConv",
    "build_eump_reconstruction_output_routing",
]
