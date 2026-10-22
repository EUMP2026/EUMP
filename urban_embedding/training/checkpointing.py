from __future__ import annotations

import os
from typing import Any

import torch

from urban_embedding.core.reproducibility import capture_rng_state, restore_rng_state
from urban_embedding.core.paths import MODEL_DIR, resolve_model_artifact_path

from .distributed import get_model, get_rank
from .schedulers import scheduler_metadata_matches


def save_model_state(model_name: str, model, file_name: str = "model.pt") -> str:
    save_path = resolve_model_artifact_path(model_name, file_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    return save_path


def save_training_checkpoint(
    trainer,
    epoch: int,
    val_loss: float,
    best_val_loss: float,
    val_loss_list: list[float],
    test_loss_list: list[float],
) -> str:
    checkpoint_path = resolve_model_artifact_path(trainer.model_name, trainer.checkpoint_file_name)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "val_loss": float(val_loss),
        "best_val_loss": float(best_val_loss),
        "n_epochs": int(trainer.n_epochs),
        "model_state_dict": get_model(trainer.model).state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "scheduler_state_dict": trainer.scheduler.state_dict(),
        "val_loss_list": list(val_loss_list),
        "test_loss_list": list(test_loss_list),
        "global_step": int(getattr(trainer, "global_step", 0)),
        "rng_state": capture_rng_state(),
    }
    scheduler_metadata = _trainer_scheduler_metadata(trainer)
    if scheduler_metadata is not None:
        checkpoint["scheduler_metadata"] = dict(scheduler_metadata)
    trainer_state_getter = getattr(trainer, "training_state_dict", None)
    if callable(trainer_state_getter):
        trainer_state = trainer_state_getter()
        if trainer_state:
            checkpoint["trainer_state"] = dict(trainer_state)
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def _trainer_scheduler_metadata(trainer) -> dict[str, Any] | None:
    getter = getattr(trainer, "current_scheduler_metadata", None)
    if callable(getter):
        metadata = getter()
    else:
        metadata = getattr(trainer, "scheduler_metadata", None)
    if metadata is None:
        return None
    return dict(metadata)


def _compatible_scheduler_epochs(trainer, checkpoint: dict[str, Any]) -> int | None:
    expected_metadata = _trainer_scheduler_metadata(trainer)
    if expected_metadata is None:
        return None
    saved_metadata = checkpoint.get("scheduler_metadata")
    if scheduler_metadata_matches(saved_metadata, expected_metadata):
        return None

    metadata_for_epochs = getattr(trainer, "scheduler_metadata_for_epochs", None)
    if callable(metadata_for_epochs) and isinstance(saved_metadata, dict) and "n_epochs" in saved_metadata:
        saved_scheduler_epochs = int(saved_metadata["n_epochs"])
        if scheduler_metadata_matches(saved_metadata, metadata_for_epochs(saved_scheduler_epochs)):
            return saved_scheduler_epochs
    return -1


def load_training_checkpoint(trainer) -> dict[str, Any] | None:
    resume_requested = getattr(trainer, "resume", False) or bool(getattr(trainer, "resume_checkpoint", None))
    if not resume_requested:
        return None

    checkpoint_ref = trainer.resume_checkpoint or trainer.checkpoint_file_name
    checkpoint_path = resolve_model_artifact_path(trainer.model_name, checkpoint_ref)
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=trainer.device)
    get_model(trainer.model).load_state_dict(checkpoint["model_state_dict"])
    trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    trainer.global_step = int(checkpoint.get("global_step", getattr(trainer, "global_step", 0)))
    completed_epoch = int(checkpoint["epoch"])
    trainer.start_epoch = completed_epoch + 1
    trainer.best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
    saved_epochs = int(checkpoint.get("n_epochs", trainer.n_epochs))
    remaining_epochs = trainer.n_epochs - completed_epoch

    compatible_scheduler_epochs = _compatible_scheduler_epochs(trainer, checkpoint)
    scheduler_compatible = compatible_scheduler_epochs is None or compatible_scheduler_epochs > 0

    if remaining_epochs <= 0:
        if get_rank() == 0:
            trainer.logger.warning(
                f"Resume checkpoint is already at epoch {completed_epoch}, "
                f"while current n_epochs={trainer.n_epochs}. No training steps remain."
            )
        trainer.start_epoch = trainer.n_epochs + 1
    elif saved_epochs == trainer.n_epochs and scheduler_compatible:
        if compatible_scheduler_epochs is not None and compatible_scheduler_epochs > 0:
            trainer.reset_scheduler(compatible_scheduler_epochs)
        trainer.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    else:
        trainer.reset_scheduler(remaining_epochs)
        if get_rank() == 0:
            reason = (
                "scheduler metadata changed or is missing"
                if saved_epochs == trainer.n_epochs
                else "total epoch count changed"
            )
            trainer.logger.warning(
                f"Resume requested with incompatible scheduler state ({reason}). "
                f"Checkpoint used n_epochs={saved_epochs}, current config uses n_epochs={trainer.n_epochs}. "
                f"Starting a fresh warmup-cosine schedule for the remaining {remaining_epochs} epoch(s)."
            )

    if get_rank() == 0:
        trainer.logger.info(f"Resumed training state from {checkpoint_path} at epoch {checkpoint['epoch']}")

    trainer_state_loader = getattr(trainer, "load_training_state_dict", None)
    trainer_state = checkpoint.get("trainer_state")
    if callable(trainer_state_loader):
        trainer_state_loader(trainer_state if isinstance(trainer_state, dict) else {})
    restore_rng_state(checkpoint.get("rng_state"))

    return checkpoint


def load_model_state_for_evaluation(
    *,
    model,
    model_name: str,
    checkpoint_source: str,
    checkpoint_file: str,
    map_location: torch.device | str = "cpu",
) -> str:
    checkpoint_path = resolve_model_artifact_path(model_name, checkpoint_file)
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Evaluation checkpoint not found: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location=map_location)
    if checkpoint_source == "best_model":
        state_dict = payload
    elif checkpoint_source == "training_checkpoint":
        state_dict = payload["model_state_dict"] if isinstance(payload, dict) else None
        if state_dict is None:
            raise KeyError(
                "Evaluation checkpoint_source='training_checkpoint' requires a checkpoint with "
                "'model_state_dict'"
            )
    else:
        raise ValueError(
            "evaluation.checkpoint_source must be 'best_model' or 'training_checkpoint', "
            f"got {checkpoint_source!r}"
        )

    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected model state dict mapping, got {type(state_dict)!r}")
    get_model(model).load_state_dict(state_dict)
    return checkpoint_path


__all__ = [
    "MODEL_DIR",
    "load_model_state_for_evaluation",
    "load_training_checkpoint",
    "save_model_state",
    "save_training_checkpoint",
]
