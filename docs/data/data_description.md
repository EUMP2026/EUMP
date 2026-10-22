# Data Description

## Scope

This document describes the processed data layout for `urban_embedding_v2`
and reader/exporter behavior.

Primary code references:

- `urban_embedding/data/processed/schema.py`
- `urban_embedding/data/processed/manifest.py`
- `urban_embedding/data/processed/reader.py`
- `urban_embedding/data/processed/exporter.py`
- `urban_embedding/data/processed/validation.py`
- `urban_embedding/data/processed/task_config.py`
- `urban_embedding/cli/export_eump_store.py`
- `urban_embedding/cli/validate_eump_store.py`
- `raw_data_desc.md`
- `scene_data_contract.md`

## Storage And Defaults

- the authoritative processed storage target is `data/processed_v2/`
- the store format is local-directory `Zarr v2`
- the persisted storage unit is `scene_id + year`, not a fixed crop patch
- raster sources are stored as aligned raw values plus `valid_mask`, not as
  normalized training tensors
- source roles such as input or target are not stored in schema or zarr attrs;
  they are chosen by a task config at read time
- `Usage` is intentionally absent from `schema.json`, zarr attrs, and manifest
  files
- `Area` and `pop_sum` are runtime-derived reader sources, not persisted zarr
  datasets
- store readers and exporters live in `urban_embedding.data.processed`
- the default Hydra trainer uses `data=eump_full/server_v2` and the MAE v3
  pretraining pipeline; downstream tasks are selected through the launch suite

## Root Layout

```text
data/processed_v2/
  manifest/
    schema.json
    scenes.jsonl
    scene_years.jsonl
    splits/
      legacy_compatible_seed42.json
    stats/
      normalization_v1.json
      preprocessing_v1.json
  scenes/
    <scene_id>/
      static.zarr/
      years/
        <year>.zarr/
```

## Preparation And Fixed Splits

The data-processing launcher reads rasters from `data/raw_data/` and scalar
tables from `data/csv_data/`, and writes `data/processed_v2/` by default:

```bash
# Export aligned scenes, validate, build the split, then fit preprocessing
bash script/data/run_eump_full_data_processing.sh full

# Validate an existing processed store
bash script/data/run_eump_full_data_processing.sh validate

# Build split and preprocessing metadata for an existing processed store
bash script/data/run_eump_full_data_processing.sh split,preprocess
```

These commands require the raw inputs or an existing processed store as
appropriate. The launcher accepts path and worker settings through environment
variables; use `DRY_RUN=1` to inspect commands before processing.

The fixed manifest `manifest/splits/legacy_compatible_seed42.json` contains
1,076 scenes: 753 training, 161 validation, and 162 test. Preprocessing is fitted
only on the training scenes over 2003–2020. Keep the exact split and
`manifest/stats/preprocessing_v1.json` with the scene store for reproducible
training and inference.

The [MAE v3 source table](mae_v3_data_description.csv) reports train,
validation, and test counts separately. Static-source counts are scenes;
dynamic-source counts are source-present scene-years within the recipe's year
range. Traffic and `PM25_mean` counts are finite scalar city-year observations.
These counts describe source coverage before task-specific crop sampling and
pixel-validity filtering. A source can be present in a manifest while its
values have an all-false `valid_mask`.

## Persisted Source Set

`data/processed_v2` persists the following schema sources:

- static raster:
  `BH`, `BF`, `water`, `DEM`, `LCZ`, `city_mask`
- yearly raster:
  `NDVI`, `NL`, `POP`, `LC`, `GAIA`, `PM25`, `road_density`, `Electricity`,
  `GDP`, `LST`, `precip`
- yearly scalar:
  `Ta`, `precip_bg`, `wind`, `Traffic`, `GHG_mean`, `GHG_emission`, `PM25_mean`

Runtime-derived reader-only sources:

- `Area`
- `pop_sum`

Naming note:

- `precip` means the raster precipitation source from `raw_data/precip_v1`.
  The default `mae_v3` downstream task aggregates this raster to a
  crop-level scalar target.
- `precip_bg` means the scalar background precipitation source from
  `csv_data/clim_values.csv`
- `PM25` is the raster schema source; `PM25_mean` is the scalar downstream
  PM2.5 target.
- `mae_v3` keeps `PM25` in the processed store but removes it from effective
  MAE pretrain aux supervision; the paper evaluates PM2.5 as a downstream task.
- `GHG_mean` and `GHG_emission` are present in the store schema but are not
  used as downstream prediction targets in the current paper experiments.

## Spatial Resolution And Grid Alignment

Raw source resolution is heterogeneous:

- `250m`:
  `BH`, `BF`, `water`, `DEM`, `LCZ`, `POP`, `GAIA`, `road_density`
- `500m`:
  `NDVI`, `LC`
- `1000m`:
  `NL`, `PM25`, `Electricity`, `GDP`, `LST`
- `4000m`:
  `precip`
- scalar:
  `Ta`, `precip_bg`, `wind`, `Traffic`, `PM25_mean`, `GHG_mean`, `GHG_emission`

Persisted raster arrays are aligned to a common `BH8_250m` reference grid per
scene.

Meaning:

- `raw_resolution` in `schema.json` preserves source provenance
- zarr raster shapes follow the aligned `BH8` scene grid
- all raster arrays for the same scene share the same `[H, W]`

## Per-Source Storage Format

Each source is stored under its own zarr group.

Raster source layout:

```text
/<source>/data        [C, H, W]
/<source>/valid_mask  [C, H, W]
```

Scalar source layout:

```text
/<source>/data        [D]
/<source>/valid_mask  [D]
```

Conventions:

- raster channels currently use `C = 1` per source
- scalar sources are stored as scene-level or year-level vectors
- `data` and `valid_mask` must have identical shapes

## Chunking And Compression

Raster chunk target:

- `(1, 256, 256)`

Scalar chunk target:

- full-length single chunk

Compression:

- `Blosc(cname="zstd", clevel=5, shuffle=SHUFFLE)`

Important detail:

- chunking is by maximum chunk size, not by four-way equal partitioning
- zarr does not pad `H` or `W` to a multiple of `256`
- the last chunk on each spatial axis may be smaller

Example:

- if a raster shape is `[1, 513, 512]`, chunk count is
  `ceil(1/1) * ceil(513/256) * ceil(512/256) = 1 * 3 * 2 = 6`
- if a raster shape is `[1, 513, 637]`, chunk count is
  `1 * 3 * 3 = 9`

This is why some zarr arrays appear to have 4 files, some 6, some 9, and some
more. The count depends on scene shape, not on a fixed split template.

## Valid Mask And Raw Stats

The persisted store writes raw aligned source values plus a per-source
`valid_mask`. `normalization_v1.json` is a raw-value statistics artifact over
valid persisted values. Training normalization uses `preprocessing_v1.json`.

Current raw-stat recomputation rules:

- values must pass the stored `valid_mask`
- values must be finite
- values are also rechecked against the source's schema `nodata` rule
- `POP=0`, `BH=0`, `BF=0`, and `DEM=0` are valid physical zeros and must stay
  in both raw stats and preprocessing stats
- for extreme float nodata sentinels such as `Electricity`, approximate
  documented values like `-3.4E+38` are matched by threshold-style comparison
  so actual pixels around `-3.402823E+38` are excluded

`preprocessing_v1.json` is generated separately by
[`urban_embedding/cli/build_eump_preprocessing.py`](../../urban_embedding/cli/build_eump_preprocessing.py)
and contains the source-driven normalization statistics used by
[`urban_embedding/data/processed/processing.py`](../../urban_embedding/data/processed/processing.py).
For full-data training, generate the fixed split manifest first and build this
file with `--split-manifest-path ... --stats-split train --year-range 2003 2020` so
the recorded z-score state is fitted from train scenes and training years only.
The output includes `stats_scope`; loaders reject stale all-data preprocessing
stats when `data.split_manifest_path` is configured.

For `log_zscore` and `zscore` sources, the preprocessing manifest also stores
`zero_excluded_stats`. These stats use the same prepared value space as global
`stats`, but exclude restored physical values equal to zero. Full pretrain uses
them for hurdle sources so the positive regression branch and positive scalar
targets are normalized by the nonzero distribution, while physical zeros remain
valid classification negatives. The manifest can also contain
`city_crop_stats` if city-crop sampling is explicitly enabled, but that field is
diagnostic compatibility metadata rather than a default training normalization
layer.

## Canonical Year Rules

Most sources persist all raw years that exist and align cleanly. Two sources
need explicit canonicalization because the raw `BH8` raster contains
materialized tail padding:

- `GAIA`
  raw materialized years: `2003-2022`
  canonical years: `2003-2018`
  padding rule:
  `2019 -> 2018, 2020 -> 2018, 2021 -> 2018, 2022 -> 2018`
- `POP`
  raw materialized years: `2003-2022`
  canonical years: `2003-2020`
  padding rule:
  `2021 -> 2020, 2022 -> 2020`

The exporter records this fact in `schema.json` but does not write those padded
years as fake zarr sources.

## Availability Semantics

The processed store does not synthesize missing sources.

Rules:

- if a raw source is absent for a `scene_id`, that source is simply absent from
  the corresponding zarr group
- if a year is outside a source's canonical range, that source is absent for
  that year
- readers should return explicit unavailable or raise a clear error for missing
  task-required sources
- implicit fill, temporal padding, crop sampling, and normalization belong to
  the reading layer, not to the persisted store

## Reader Layer

`urban_embedding.data.processed` provides:

- processed schema loading
- manifest loading
- `ProcessedSceneStoreReader`
- task-config-driven `SceneRecord` assembly
- runtime-derived `Area` and `pop_sum`
- explicit task-compatibility checks for missing sources
- first-stage processed training pipeline specs, torch batch materialization,
  Dataset/DataLoader support, and preprocessing-manifest compatibility checks
- processed reconstruction and single-target processed downstream batch
  materialization through `ProcessedModelBatch`
- default processed training configs keep `city_mask` as a loss-mask field and
  also append it to encoder raster inputs
- processed reconstruction can keep sample-years when optional aux targets are
  missing; those targets are represented by invalid `NaN` placeholders and are
  excluded from loss aggregation
- optional inputs are configured separately with
  `pipeline.inputs.optional_sources`; missing optional input channels are
  zero-filled with all-false valid masks, preserving fixed encoder channel
  layouts without making downstream targets optional

The intended read contract is:

1. choose `eump_root`
2. load `schema.json`, `scenes.jsonl`, and `scene_years.jsonl`
3. provide a task config that maps source names to model roles
4. materialize a `SceneRecord`
5. perform crop sampling and normalization at runtime
6. materialize model-specific torch batches from the processed pipeline config

## Training Relationship

The default Hydra training path uses MAE v3 reconstruction. Downstream tasks
and comparison models are selected through launch suites; see the
[configuration reference](../config/eump_config_usage.md) for commands and the
[runtime registry](../architecture/codebase_layout.md) for supported models.
The store and preprocessing contract are shared by reconstruction,
contrastive, supervised Swin/ConvNeXt, and scalar MLP pipelines.

## Related Files

- [Codebase layout](../architecture/codebase_layout.md)
- [Raw source catalog](raw_data_desc.md)
- [Scene data contract](scene_data_contract.md)
