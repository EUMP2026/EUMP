# Documentation

Technical reference for `urban_embedding_v2`.
Experiment-specific notes live in each `experiments/<name>/EXPERIMENT.md`.

## Architecture And Code Layout

- [codebase_layout.md](architecture/codebase_layout.md)
  Canonical file tree, model/trainer registry boundaries, CLI wrappers, and
  extension policy for new EUMP models.

## Data

- [raw_data_desc.md](data/raw_data_desc.md)
  Raw source catalog, canonical year rules, and `BH8` padding facts.

- [data_description.md](data/data_description.md)
  `eump_v1` storage layout, source coverage, chunking, and reader behavior.

- [scene_data_contract.md](data/scene_data_contract.md)
  Persisted-store contract, in-memory `SceneRecord`, and crop contract.

## Config Reference

- [eump_config_usage.md](config/eump_config_usage.md)
  Hydra config layering, inputs, targets, masks, optional inputs/targets,
  `pipeline.kind`/`model.kind` runtime selection, and common train/eval commands.
