"""EUMP training runtime packages."""

from .registry import (
    CITY_SCALAR_MLP_MODEL_KIND,
    RuntimeContext,
    SWIN_DOWNSTREAM_LIGHT_MODEL_KIND,
    SWIN_RECONSTRUCTION_MODEL_KIND,
    TRAINING_RUNTIME_SPECS,
    TrainingRuntimeSpec,
    infer_eump_model_kind,
    normalize_runtime_identity,
    resolve_evaluation_runtime,
    resolve_training_runtime,
    supported_runtime_keys,
)

__all__ = [
    "CITY_SCALAR_MLP_MODEL_KIND",
    "RuntimeContext",
    "SWIN_DOWNSTREAM_LIGHT_MODEL_KIND",
    "SWIN_RECONSTRUCTION_MODEL_KIND",
    "TRAINING_RUNTIME_SPECS",
    "TrainingRuntimeSpec",
    "infer_eump_model_kind",
    "normalize_runtime_identity",
    "resolve_evaluation_runtime",
    "resolve_training_runtime",
    "supported_runtime_keys",
]
