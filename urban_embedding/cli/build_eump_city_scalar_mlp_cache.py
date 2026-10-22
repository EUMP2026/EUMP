from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from urban_embedding.core.logging import setup_logger
from urban_embedding.core.reproducibility import set_reproducible
from urban_embedding.training.eump.city_scalar_mlp.cache import ensure_city_scalar_cache_files
from urban_embedding.training.runner import normalize_runtime_config, resolve_log_dir


CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs")


@hydra.main(config_path=CONFIG_DIR, config_name="train", version_base=None)
def main(cfg: DictConfig):
    config = normalize_runtime_config(cfg)
    log_dir = resolve_log_dir(config)
    logger = setup_logger(log_dir=log_dir, rank=0, log_level=config["logging"]["level"])
    set_reproducible(config["experiment"]["random_seed"])
    feature_cache, target_cache, _, target_schema, feature_layout, _ = ensure_city_scalar_cache_files(
        data_config=config["data"],
        pipeline_config=config["pipeline"],
        logger=logger,
        random_seed=config["experiment"]["random_seed"],
        local_rank=0,
        log_dir=log_dir,
    )
    logger.info(
        "City-scalar MLP cache ready "
        f"target={target_schema.name} features={feature_layout.feature_dim} "
        f"feature_cache={feature_cache.path} target_cache={target_cache.path}"
    )


if __name__ == "__main__":
    main()
