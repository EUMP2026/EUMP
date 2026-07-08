from __future__ import annotations

from pathlib import Path


class TensorBoardScalarWriter:
    """Rank-aware TensorBoard scalar writer with explicit dependency errors."""

    def __init__(self, *, log_dir: str | None, enabled: bool, rank: int) -> None:
        self.enabled = bool(enabled) and int(rank) == 0
        self.writer = None
        if not self.enabled:
            return
        if not log_dir:
            raise ValueError("TensorBoard logging is enabled but no log_dir was provided")

        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            raise RuntimeError(
                "TensorBoard logging is enabled, but the tensorboard package is not installed. "
                "Install project requirements or set pipeline.loss.tensorboard=false."
            ) from exc

        self.writer = SummaryWriter(str(Path(log_dir) / "tensorboard"))

    def add_scalars(
        self,
        *,
        phase: str,
        level: str,
        scalars: dict[str, float],
        step: int,
    ) -> None:
        if self.writer is None:
            return
        prefix = f"{phase}/{level}"
        for name, value in scalars.items():
            self.writer.add_scalar(f"{prefix}/{name}", float(value), int(step))

    def flush(self) -> None:
        if self.writer is not None:
            self.writer.flush()

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None


__all__ = ["TensorBoardScalarWriter"]
