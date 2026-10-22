from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from urban_embedding.training.evaluation import run_evaluation


CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs")


@hydra.main(config_path=CONFIG_DIR, config_name="evaluate", version_base=None)
def main(cfg: DictConfig):
    run_evaluation(cfg, source="hydra")


if __name__ == "__main__":
    main()
