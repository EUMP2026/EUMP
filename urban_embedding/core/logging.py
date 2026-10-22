from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml
from loguru import logger


def setup_logger(log_dir: str | None = None, rank: int = 0, log_level: str = "INFO"):
    """Configure a rank-aware Loguru logger."""
    logger.remove()

    console_level = log_level if rank == 0 else "WARNING"
    logger.add(
        sys.stdout,
        level=console_level,
        colorize=True,
        enqueue=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level:<8}</level> | "
            "{message}"
        ),
    )

    if log_dir and rank == 0:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_path / "train.log",
            level="DEBUG",
            enqueue=True,
            backtrace=False,
            diagnose=False,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
        )

    return logger


def save_runtime_config(log_dir: str | None, config: dict[str, Any]) -> str | None:
    """Persist the resolved runtime config for reproducibility."""
    if not log_dir:
        return None

    output_dir = Path(log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "runtime_config.yaml"

    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False, allow_unicode=True)

    return str(config_path)


__all__ = ["save_runtime_config", "setup_logger"]
