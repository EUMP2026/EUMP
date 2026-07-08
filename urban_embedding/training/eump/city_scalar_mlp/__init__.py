"""City-level scalar MLP training and evaluation runtime."""

from .evaluator import EumpScalarMlpEvaluator
from .features import (
    CityLevelFeatureLayout,
    CityLevelTensors,
    SourceFeatureSlice,
    StandardizationStats,
    build_city_level_feature_layout,
    build_city_level_features,
    build_city_level_target,
    compute_city_level_standardization_stats,
    valid_city_level_sample_mask,
)
from .trainer import EumpScalarMlpTrainer

__all__ = [
    "CityLevelFeatureLayout",
    "CityLevelTensors",
    "EumpScalarMlpEvaluator",
    "EumpScalarMlpTrainer",
    "SourceFeatureSlice",
    "StandardizationStats",
    "build_city_level_feature_layout",
    "build_city_level_features",
    "build_city_level_target",
    "compute_city_level_standardization_stats",
    "valid_city_level_sample_mask",
]
