# Documentation

Technical reference for `urban_embedding_v2`.
Experiment-specific methods, commands, and outputs are documented in each
`experiments/<name>/EXPERIMENT.md`, or
[`experiments/manipulate_exp/README.md`](../experiments/manipulate_exp/README.md).
See the [repository README](../README.md) for the experiment index.

## Environment

Run commands from the repository root. The dependency versions in
[`requirements.txt`](../requirements.txt) target Python 3.12 and PyTorch 2.6.0
with torchvision 0.21.0. On Linux x86-64, the standard PyTorch wheel includes
CUDA 12.4 runtime dependencies; GPU execution requires a compatible NVIDIA
driver.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements cover training, data processing, and figure rendering,
including Rasterio, GeoPandas, and Cartopy. Arial is optional for matching the
published figure typography; Matplotlib uses an available fallback if the font
is absent. Model weights and processed data are separate runtime inputs; their
paths are specified by the selected configuration and experiment document.

## Architecture And Code Layout

- [codebase_layout.md](architecture/codebase_layout.md)
  Canonical file tree, model/trainer registry boundaries, CLI wrappers, and
  extension policy for new EUMP models.

## Data

- [raw_data_desc.md](data/raw_data_desc.md)
  Raw source catalog, canonical year rules, and `BH8` padding facts.

- [data_description.md](data/data_description.md)
  `data/processed_v2` preparation, storage layout, fixed splits, source coverage,
  chunking, and reader behavior.

- [mae_v3_data_description.csv](data/mae_v3_data_description.csv)
  MAE v3 source roles and separate train, validation, and test sample counts.

- [scene_data_contract.md](data/scene_data_contract.md)
  Persisted-store contract, in-memory `SceneRecord`, and crop contract.

## Config Reference

- [eump_config_usage.md](config/eump_config_usage.md)
  Hydra config layering, inputs, targets, masks, optional inputs/targets,
  `pipeline.kind`/`model.kind` runtime selection, and common train/eval commands.
