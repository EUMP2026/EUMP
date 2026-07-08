from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


ACTIVATIONS = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
}


def _normalize_hidden_dims(value: Any) -> tuple[int, ...]:
    if value is None:
        return (128,)
    if isinstance(value, int):
        dims = (int(value),)
    else:
        dims = tuple(int(dim) for dim in value)
    if any(dim <= 0 for dim in dims):
        raise ValueError(f"model.mlp_hidden_dims must contain positive integers, got {dims!r}")
    return dims


def _activation(name: str) -> nn.Module:
    normalized = str(name).lower()
    if normalized not in ACTIVATIONS:
        raise ValueError(f"Unsupported model.activation={normalized!r}; expected one of {sorted(ACTIVATIONS)}")
    return ACTIVATIONS[normalized]()


class EumpScalarMlpModel(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...] = (128,),
        dropout: float = 0.3,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"Scalar MLP input_dim must be positive, got {input_dim}")
        if output_dim <= 0:
            raise ValueError(f"Scalar MLP output_dim must be positive, got {output_dim}")
        if dropout < 0 or dropout >= 1:
            raise ValueError(f"model.dropout must be in [0, 1), got {dropout}")

        layers: list[nn.Module] = []
        current_dim = int(input_dim)
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, int(hidden_dim)))
            layers.append(_activation(activation))
            layers.append(nn.Dropout(float(dropout)))
            current_dim = int(hidden_dim)
        layers.append(nn.Linear(current_dim, int(output_dim)))
        self.net = nn.Sequential(*layers)
        self.register_buffer("feature_mean", torch.zeros(input_dim, dtype=torch.float32))
        self.register_buffer("feature_std", torch.ones(input_dim, dtype=torch.float32))
        self.register_buffer("target_mean", torch.zeros(output_dim, dtype=torch.float32))
        self.register_buffer("target_std", torch.ones(output_dim, dtype=torch.float32))

    def set_standardization(
        self,
        *,
        feature_mean: torch.Tensor,
        feature_std: torch.Tensor,
        target_mean: torch.Tensor,
        target_std: torch.Tensor,
    ) -> None:
        self.feature_mean.copy_(feature_mean.to(device=self.feature_mean.device, dtype=self.feature_mean.dtype))
        self.feature_std.copy_(feature_std.to(device=self.feature_std.device, dtype=self.feature_std.dtype).clamp_min(1.0e-8))
        self.target_mean.copy_(target_mean.to(device=self.target_mean.device, dtype=self.target_mean.dtype))
        self.target_std.copy_(target_std.to(device=self.target_std.device, dtype=self.target_std.dtype).clamp_min(1.0e-8))

    def standardize_features(self, features: torch.Tensor) -> torch.Tensor:
        return (features - self.feature_mean) / self.feature_std

    def standardize_target(self, target: torch.Tensor) -> torch.Tensor:
        return (target - self.target_mean) / self.target_std

    def unstandardize_target(self, target: torch.Tensor) -> torch.Tensor:
        return target * self.target_std + self.target_mean

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(self.standardize_features(features))


def build_eump_scalar_mlp_model(
    *,
    model_config: dict[str, Any],
    input_dim: int,
    output_dim: int,
) -> EumpScalarMlpModel:
    hidden_dims = _normalize_hidden_dims(model_config.get("mlp_hidden_dims", (128,)))
    return EumpScalarMlpModel(
        input_dim=int(input_dim),
        output_dim=int(output_dim),
        hidden_dims=hidden_dims,
        dropout=float(model_config.get("dropout", 0.3)),
        activation=str(model_config.get("activation", "gelu")),
    )


__all__ = ["EumpScalarMlpModel", "build_eump_scalar_mlp_model"]
