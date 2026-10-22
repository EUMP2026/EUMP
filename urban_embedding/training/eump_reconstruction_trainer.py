"""Compatibility exports for the Swin reconstruction trainer."""

from urban_embedding.training.eump.swin_reconstruction.trainer import (
    GRAD_NORM_DEBUG_PARAMETER_SCOPES,
    EumpSwinReconstructionTrainer,
    GradNormDebugConfig,
    normalize_grad_norm_debug_config,
    normalize_scalar_head_mode,
)

__all__ = [
    "EumpSwinReconstructionTrainer",
    "GRAD_NORM_DEBUG_PARAMETER_SCOPES",
    "GradNormDebugConfig",
    "normalize_grad_norm_debug_config",
    "normalize_scalar_head_mode",
]
