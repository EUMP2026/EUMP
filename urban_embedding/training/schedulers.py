from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch.optim.lr_scheduler import LambdaLR


WARMUP_COSINE_SCHEDULER_KIND = "warmup_cosine_v1"
DEFAULT_WARMUP_FRACTION = 0.2
DEFAULT_LR_START_FACTOR = 0.08
DEFAULT_LR_END_FACTOR = 0.2


@dataclass(frozen=True, slots=True)
class WarmupCosineSchedulerConfig:
    warmup_fraction: float = DEFAULT_WARMUP_FRACTION
    lr_start_factor: float = DEFAULT_LR_START_FACTOR
    lr_end_factor: float = DEFAULT_LR_END_FACTOR


def normalize_warmup_cosine_scheduler_config(payload: Mapping[str, object] | None) -> WarmupCosineSchedulerConfig:
    if payload is None:
        payload = {}
    elif not isinstance(payload, Mapping):
        raise TypeError(f"scheduler config must be a mapping, got {type(payload)!r}")
    else:
        payload = dict(payload)

    if "peak_lr_factor" in payload:
        raise ValueError("scheduler config no longer supports peak_lr_factor; the configured lr is the max lr")
    if "lr_start_factor" in payload and "start_lr_factor" in payload:
        raise ValueError("scheduler config cannot set both lr_start_factor and legacy start_lr_factor")
    end_aliases = [key for key in ("lr_end_factor", "end_factor", "final_lr_factor") if key in payload]
    if len(end_aliases) > 1:
        raise ValueError(f"scheduler config cannot set multiple end factor aliases: {end_aliases}")
    lr_start_factor = payload.get("lr_start_factor", payload.get("start_lr_factor", DEFAULT_LR_START_FACTOR))
    lr_end_factor = payload.get(
        "lr_end_factor",
        payload.get("end_factor", payload.get("final_lr_factor", DEFAULT_LR_END_FACTOR)),
    )

    config = WarmupCosineSchedulerConfig(
        warmup_fraction=float(payload.get("warmup_fraction", DEFAULT_WARMUP_FRACTION)),
        lr_start_factor=float(lr_start_factor),
        lr_end_factor=float(lr_end_factor),
    )
    _validate_scheduler_factors(
        warmup_fraction=config.warmup_fraction,
        lr_start_factor=config.lr_start_factor,
        lr_end_factor=config.lr_end_factor,
    )
    return config


def optimizer_steps_per_epoch(num_batches: int, accumulation_steps: int) -> int:
    num_batches = int(num_batches)
    accumulation_steps = int(accumulation_steps)
    if num_batches < 1:
        raise ValueError(f"Scheduler requires at least one train batch, got {num_batches}")
    if accumulation_steps < 1:
        raise ValueError(f"accumulation_steps must be >= 1, got {accumulation_steps}")
    return math.ceil(num_batches / accumulation_steps)


def _validate_epochs(epochs: int) -> int:
    epochs = int(epochs)
    if epochs < 1:
        raise ValueError(f"Scheduler epochs must be >= 1, got {epochs}")
    return epochs


def _validate_scheduler_factors(
    *,
    warmup_fraction: float,
    lr_start_factor: float,
    lr_end_factor: float,
) -> None:
    if not 0 < warmup_fraction <= 1:
        raise ValueError(f"warmup_fraction must be in (0, 1], got {warmup_fraction}")
    if not 0 < lr_start_factor <= 1:
        raise ValueError(f"lr_start_factor must be in (0, 1], got {lr_start_factor}")
    if not 0 <= lr_end_factor <= 1:
        raise ValueError(f"lr_end_factor must be in [0, 1], got {lr_end_factor}")


def warmup_cosine_metadata(
    *,
    base_lr: float,
    num_batches: int,
    accumulation_steps: int,
    epochs: int,
    warmup_fraction: float = DEFAULT_WARMUP_FRACTION,
    lr_start_factor: float | None = None,
    lr_end_factor: float | None = None,
    end_factor: float | None = None,
    start_lr_factor: float | None = None,
    final_lr_factor: float | None = None,
) -> dict[str, Any]:
    epochs = _validate_epochs(epochs)
    steps_per_epoch = optimizer_steps_per_epoch(num_batches, accumulation_steps)
    total_steps = steps_per_epoch * epochs
    if base_lr <= 0:
        raise ValueError(f"base_lr must be > 0, got {base_lr}")
    if lr_start_factor is not None and start_lr_factor is not None:
        raise ValueError("Cannot set both lr_start_factor and legacy start_lr_factor")
    end_values = [
        value
        for value in (lr_end_factor, end_factor, final_lr_factor)
        if value is not None
    ]
    if len(end_values) > 1:
        raise ValueError("Cannot set multiple lr end factor aliases")
    lr_start_factor = DEFAULT_LR_START_FACTOR if lr_start_factor is None and start_lr_factor is None else (
        start_lr_factor if lr_start_factor is None else lr_start_factor
    )
    lr_end_factor = DEFAULT_LR_END_FACTOR if not end_values else end_values[0]
    _validate_scheduler_factors(
        warmup_fraction=float(warmup_fraction),
        lr_start_factor=float(lr_start_factor),
        lr_end_factor=float(lr_end_factor),
    )
    peak_lr = float(base_lr)
    start_lr = float(base_lr) * float(lr_start_factor)
    final_lr = float(base_lr) * float(lr_end_factor)
    if total_steps == 1:
        warmup_steps = 1
    else:
        warmup_steps = min(total_steps, max(2, math.ceil(total_steps * float(warmup_fraction))))

    return {
        "kind": WARMUP_COSINE_SCHEDULER_KIND,
        "base_lr": float(base_lr),
        "lr_start_factor": float(lr_start_factor),
        "lr_end_factor": float(lr_end_factor),
        "start_lr": float(start_lr),
        "peak_lr": float(peak_lr),
        "final_lr": float(final_lr),
        "warmup_fraction": float(warmup_fraction),
        "warmup_steps": int(warmup_steps),
        "num_batches": int(num_batches),
        "accumulation_steps": int(accumulation_steps),
        "optimizer_steps_per_epoch": int(steps_per_epoch),
        "total_steps": int(total_steps),
        "n_epochs": int(epochs),
    }


def _lr_at_step(step: int, metadata: Mapping[str, Any]) -> float:
    total_steps = int(metadata["total_steps"])
    warmup_steps = int(metadata["warmup_steps"])
    start_lr = float(metadata["start_lr"])
    peak_lr = float(metadata["peak_lr"])
    final_lr = float(metadata["final_lr"])
    step = min(max(int(step), 0), total_steps - 1)

    if total_steps == 1:
        return start_lr
    if step < warmup_steps:
        progress = step / max(1, warmup_steps - 1)
        return start_lr + (peak_lr - start_lr) * progress

    decay_steps = max(1, total_steps - warmup_steps)
    progress = (step - warmup_steps + 1) / decay_steps
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return final_lr + (peak_lr - final_lr) * cosine


def build_warmup_cosine_scheduler(
    *,
    optimizer: torch.optim.Optimizer,
    base_lr: float,
    num_batches: int,
    accumulation_steps: int,
    epochs: int,
    warmup_fraction: float = DEFAULT_WARMUP_FRACTION,
    lr_start_factor: float | None = None,
    lr_end_factor: float | None = None,
    end_factor: float | None = None,
    start_lr_factor: float | None = None,
    final_lr_factor: float | None = None,
) -> tuple[LambdaLR, dict[str, Any]]:
    metadata = warmup_cosine_metadata(
        base_lr=base_lr,
        num_batches=num_batches,
        accumulation_steps=accumulation_steps,
        epochs=epochs,
        warmup_fraction=warmup_fraction,
        lr_start_factor=lr_start_factor,
        lr_end_factor=lr_end_factor,
        end_factor=end_factor,
        start_lr_factor=start_lr_factor,
        final_lr_factor=final_lr_factor,
    )
    peak_lr = float(metadata["peak_lr"])
    for group in optimizer.param_groups:
        group["lr"] = peak_lr
        group["initial_lr"] = peak_lr

    def lr_lambda(step: int) -> float:
        return _lr_at_step(step, metadata) / peak_lr

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    return scheduler, metadata


def scheduler_metadata_matches(saved: Mapping[str, Any] | None, expected: Mapping[str, Any]) -> bool:
    if not isinstance(saved, Mapping):
        return False
    for key, expected_value in expected.items():
        if key not in saved:
            if key == "lr_start_factor" and "start_lr_factor" in saved:
                if math.isclose(float(saved["start_lr_factor"]), float(expected_value), rel_tol=1e-12, abs_tol=1e-15):
                    continue
            if key == "lr_end_factor" and "final_lr_factor" in saved:
                if math.isclose(float(saved["final_lr_factor"]), float(expected_value), rel_tol=1e-12, abs_tol=1e-15):
                    continue
            if key == "lr_start_factor" and "start_lr" in saved and "base_lr" in saved:
                saved_value = float(saved["start_lr"]) / float(saved["base_lr"])
                if math.isclose(saved_value, float(expected_value), rel_tol=1e-12, abs_tol=1e-15):
                    continue
            if key == "lr_end_factor" and "final_lr" in saved and "base_lr" in saved:
                saved_value = float(saved["final_lr"]) / float(saved["base_lr"])
                if math.isclose(saved_value, float(expected_value), rel_tol=1e-12, abs_tol=1e-15):
                    continue
            return False
        saved_value = saved[key]
        if isinstance(expected_value, float):
            if not math.isclose(float(saved_value), expected_value, rel_tol=1e-12, abs_tol=1e-15):
                return False
        elif expected_value is None:
            if saved_value is not None:
                return False
        else:
            if saved_value != expected_value:
                return False
    return True


__all__ = [
    "DEFAULT_LR_END_FACTOR",
    "DEFAULT_LR_START_FACTOR",
    "DEFAULT_WARMUP_FRACTION",
    "WarmupCosineSchedulerConfig",
    "WARMUP_COSINE_SCHEDULER_KIND",
    "build_warmup_cosine_scheduler",
    "normalize_warmup_cosine_scheduler_config",
    "optimizer_steps_per_epoch",
    "scheduler_metadata_matches",
    "warmup_cosine_metadata",
]
