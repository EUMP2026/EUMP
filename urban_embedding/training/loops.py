from __future__ import annotations

from collections.abc import Mapping
import os
import time

import numpy as np
import torch
import torch.distributed as dist
from tqdm.auto import tqdm

from urban_embedding.core.paths import MODEL_DIR

from .checkpointing import load_training_checkpoint, save_model_state, save_training_checkpoint
from .distributed import get_model, get_rank, get_world_size


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def fit(trainer, test: bool = False) -> None:
    def reduce_scalar(value):
        if not dist.is_available() or not dist.is_initialized():
            return float(value.item() if torch.is_tensor(value) else value)
        if torch.is_tensor(value):
            reduced = value.detach().float().to(trainer.device)
        else:
            reduced = torch.tensor(float(value), device=trainer.device)
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        reduced /= get_world_size()
        return reduced.item()

    def reduce_metrics(value):
        if isinstance(value, Mapping):
            return {name: reduce_scalar(metric) for name, metric in value.items()}
        return reduce_scalar(value)

    def total_loss(value) -> float:
        if isinstance(value, Mapping):
            return float(value["loss/total"])
        return float(value)

    def format_epoch_line(phase: str, value) -> str:
        formatter = getattr(trainer, "format_epoch_log", None)
        if isinstance(value, Mapping) and formatter is not None:
            return formatter(phase, value)
        return f"{phase} Loss: {float(value):.6f}"

    def compute_selection_metrics(epoch: int) -> dict[str, float] | None:
        compute = getattr(trainer, "compute_selection_metrics", None)
        if callable(compute):
            metrics = compute(epoch)
            if metrics is None:
                return None
            return dict(metrics)
        return None

    def process_epoch_selection(epoch: int, val_metrics) -> dict[str, object]:
        selector = getattr(trainer, "process_epoch_selection", None)
        if callable(selector):
            return dict(selector(epoch=epoch, val_metrics=val_metrics))

        save_best_model = bool(getattr(trainer, "save_best_model_enabled", True))
        save_best_checkpoint = bool(getattr(trainer, "save_best_checkpoint_enabled", False))
        val_loss = total_loss(val_metrics)
        if (save_best_model or save_best_checkpoint) and val_loss < trainer.best_val_loss:
            trainer.best_val_loss = val_loss
            return {
                "save_best": True,
                "message": f"Best validation artifact saved with val loss {val_loss:.6f}",
                "extra_metrics": {},
            }
        return {"save_best": False, "message": None, "extra_metrics": {}}

    def should_save_training_checkpoint(epoch: int) -> bool:
        interval = int(getattr(trainer, "checkpoint_every_n_epochs", 1))
        if interval <= 0:
            raise ValueError(f"checkpoint_every_n_epochs must be > 0, got {interval}")
        return epoch >= int(trainer.n_epochs) or epoch % interval == 0

    start_time = time.time()
    resume_state = load_training_checkpoint(trainer)
    test_loss_list = list((resume_state or {}).get("test_loss_list", []))
    val_loss_list = list((resume_state or {}).get("val_loss_list", []))
    start_epoch = trainer.start_epoch
    show_progress = bool(getattr(trainer, "progress_bar_enabled", False)) and get_rank() == 0
    remaining_epochs = max(0, trainer.n_epochs - start_epoch + 1)
    epoch_bar = (
        tqdm(total=remaining_epochs, desc="epoch progress", unit="epoch", dynamic_ncols=True, leave=True)
        if show_progress and remaining_epochs > 0
        else None
    )

    try:
        for epoch in range(start_epoch, trainer.n_epochs + 1):
            if get_rank() == 0:
                trainer.logger.info(f"## EPOCH {epoch} ##")

            train_metrics = reduce_metrics(trainer.train_epoch(epoch))
            val_metrics = reduce_metrics(trainer.validate_epoch(epoch))
            val_loss = total_loss(val_metrics)
            selection_metrics = compute_selection_metrics(epoch)
            if selection_metrics is not None:
                val_metrics = dict(val_metrics)
                val_metrics.update(selection_metrics)
            selection_result = process_epoch_selection(epoch, val_metrics)
            extra_metrics = selection_result.get("extra_metrics", {})
            if extra_metrics:
                val_metrics = dict(val_metrics)
                val_metrics.update(extra_metrics)

            if get_rank() == 0:
                trainer.logger.info(
                    f"{format_epoch_line('TRAIN', train_metrics)} | {format_epoch_line('VAL', val_metrics)}"
                )

            if bool(selection_result.get("save_best", False)):
                if get_rank() == 0:
                    try:
                        if bool(getattr(trainer, "save_best_model_enabled", True)):
                            save_model_state(
                                model_name=trainer.model_name,
                                model=get_model(trainer.model),
                                file_name=trainer.save_file_name,
                            )
                        if bool(getattr(trainer, "save_best_checkpoint_enabled", False)):
                            checkpoint_path = save_model_state(
                                model_name=trainer.model_name,
                                model=get_model(trainer.model),
                                file_name=getattr(trainer, "best_checkpoint_file_name", "best_model.pt"),
                            )
                            trainer.logger.info(f"Saved best validation checkpoint to {checkpoint_path}")
                        message = selection_result.get("message")
                        if message:
                            trainer.logger.info(str(message))
                    except Exception as exc:
                        trainer.logger.error(f"Error saving model: {exc}")

            if test:
                test_metrics = reduce_metrics(trainer.test_epoch(epoch))
                test_loss = total_loss(test_metrics)
                if get_rank() == 0:
                    val_loss_list.append(val_loss)
                    test_loss_list.append(test_loss)
                    trainer.logger.info(format_epoch_line("TEST", test_metrics))

            if get_rank() == 0 and should_save_training_checkpoint(epoch):
                checkpoint_path = save_training_checkpoint(
                    trainer=trainer,
                    epoch=epoch,
                    val_loss=val_loss,
                    best_val_loss=trainer.best_val_loss,
                    val_loss_list=val_loss_list,
                    test_loss_list=test_loss_list,
                )
                trainer.logger.info(f"Saved training checkpoint to {checkpoint_path}")

            if get_rank() == 0:
                if epoch_bar is not None:
                    completed_epochs = epoch - start_epoch + 1
                    total_elapsed = time.time() - start_time
                    average_epoch_seconds = total_elapsed / max(1, completed_epochs)
                    epoch_bar.update(1)
                    epoch_bar.set_description(f"epoch {epoch}/{trainer.n_epochs}")
                    postfix = {
                        "train_loss": f"{total_loss(train_metrics):.6f}",
                        "val_loss": f"{val_loss:.6f}",
                        "elapsed": _format_duration(total_elapsed),
                        "eta": _format_duration(average_epoch_seconds * max(0, trainer.n_epochs - epoch)),
                    }
                    if test:
                        postfix["test_loss"] = f"{test_loss_list[-1]:.6f}"
                    epoch_bar.set_postfix(postfix)

        if get_rank() == 0:
            if test:
                np.save(os.path.join(MODEL_DIR, trainer.model_name, "test_loss.npy"), np.array(test_loss_list))
                np.save(os.path.join(MODEL_DIR, trainer.model_name, "val_loss.npy"), np.array(val_loss_list))

            if bool(getattr(trainer, "save_final_model_enabled", False)):
                save_model_state(
                    model_name=trainer.model_name,
                    model=get_model(trainer.model),
                    file_name=trainer.save_file_name,
                )

            duration_minutes = (time.time() - start_time) / 60
            formatter = getattr(trainer, "format_training_summary", None)
            if callable(formatter):
                trainer.logger.info(formatter(duration_minutes))
            else:
                trainer.logger.info(
                    f"Best Val Loss: {trainer.best_val_loss:.6f}, Total Training Time: {duration_minutes:.2f} mins"
                )
    finally:
        if epoch_bar is not None:
            epoch_bar.close()
        close = getattr(trainer, "close", None)
        if close is not None:
            close()


__all__ = ["fit"]
