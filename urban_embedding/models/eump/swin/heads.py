"""Reusable Swin downstream heads."""

from .downstream import (
    PooledScalarHead,
    ProgressiveUpsampleRasterHead,
    SegFormerStyleUpsampleHead,
)

__all__ = [
    "PooledScalarHead",
    "ProgressiveUpsampleRasterHead",
    "SegFormerStyleUpsampleHead",
]
