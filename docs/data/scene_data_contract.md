# Scene Data Contract

This document defines the in-memory scene contract and its persisted
backing-store contract for `urban_embedding_v2`.

## Core Principle

The stable storage unit should be the full scene or tile, not a cropped patch.

Reason:

- crop locations are random at training time
- one full scene should support many different crops across epochs
- loading full arrays once and slicing repeatedly is cheaper than repeatedly
  opening many small files

## Runtime Structure

### `SceneRecord`

The long-lived in-memory object is a full-scene record:

- `scene_id`
- `time_id`
- `inputs.static_raster`
- `inputs.dynamic_raster`
- `inputs.scalar`
- `inputs.city_mask`
- `targets.reconstruct_raster`
- `targets.aux_raster`
- `targets.aux_scalar`
- `targets.prediction_raster`
- `targets.crop_scalar_specs`
- `meta`

All raster arrays use `[C, H, W]`.
All scalar arrays use `[D]`.

Typed interfaces live in:

- [`urban_embedding/data/contracts.py`](../../urban_embedding/data/contracts.py)
- [`urban_embedding/data/crop.py`](../../urban_embedding/data/crop.py)

### `CropSample`

The training-time sample is generated from a full scene plus a runtime crop
box:

- `CropBox(top, left, height, width)`, with derived `bottom` and `right`
  properties; serialized crop metadata uses `(top, left, bottom, right)`
- cropped raster inputs and targets
- unchanged scalar inputs
- copied scene metadata plus crop metadata

Crop helpers live in:

- [`urban_embedding/data/crop.py`](../../urban_embedding/data/crop.py)

### `ProcessedModelBatch`

The processed training batch exposes model inputs, targets, and masks:

- `raster_inputs`
- `raster_valid_mask`
- `scalar_inputs`
- `scalar_valid_mask`
- `city_mask`
- `raster_targets` and `scalar_targets`, dictionaries of `TensorBlock` values
  containing `data`, `valid_mask`, and `names`
- `meta`

Current processed pipeline configs keep `city_mask` as the separate
`batch.city_mask` weighting/masking field and also append it to
`raster_inputs` through `pipeline.include_city_mask_as_input: true`.

Batch helpers live in:

- [`urban_embedding/data/processed/batch.py`](../../urban_embedding/data/processed/batch.py)
- [`urban_embedding/data/processed/dataset.py`](../../urban_embedding/data/processed/dataset.py)
- [`urban_embedding/data/processed/pipeline_spec.py`](../../urban_embedding/data/processed/pipeline_spec.py)

## Persisted Backing Store

The current persisted backing store is `data/processed_v2/`, using local
directory `Zarr v2`.

Layout:

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

Each scene has:

- one `static.zarr`
- zero or more yearly stores at `years/<year>.zarr`

### Per-source storage

Raster sources:

```text
/<source>/data        [C, H, W]
/<source>/valid_mask  [C, H, W]
```

Scalar sources:

```text
/<source>/data        [D]
/<source>/valid_mask  [D]
```

Store facts:

- persisted rasters are aligned to the `BH8_250m` scene grid
- normalized training values are not stored
- source roles such as input or target are not stored
- missing sources remain missing; the store does not synthesize them

Stats facts:

- `normalization_v1.json` is a raw-value stats artifact over values that pass
  `valid_mask`, finite checks, and the source schema `nodata` rule
- `preprocessing_v1.json` is the current preprocessing manifest used for
  source-driven normalization, categorical expansion, and inverse restore
- full-data `preprocessing_v1.json` must be generated after the split manifest
  and records `stats_scope` for train-split, configured-year-range fitting;
  loaders reject stale all-data preprocessing stats when a split manifest is
  configured
- `preprocessing_v1.json` stores `zero_excluded_stats` for `log_zscore` and
  `zscore` sources; these stats use the same prepared value space as `stats`
  but exclude restored physical zero values
- `preprocessing_v1.json` can additionally store `city_crop_stats` when
  city-crop sampling is explicitly enabled
- true physical zeros, including `POP=0`, `BH=0`, `BF=0`, and `DEM=0`, are
  valid values and are retained
- extreme float nodata sentinels, especially `Electricity` around
  `-3.402823E+38`, must be excluded before statistics or preprocessing

### Chunking

Raster chunks use a maximum target of `(1, 256, 256)`.
Scalar sources use one full-length chunk.

The final chunk on a spatial axis may be smaller than `256`; zarr does not pad
scene dimensions to a multiple of the chunk size.

## Task Configuration Boundary

The persisted store is task-agnostic.

Task-specific meaning lives outside the store:

- input sources
- optional input sources
- target source
- auxiliary targets
- prediction targets
- normalization policy
- crop policy
- whether a reader chooses to perform any explicit padding

These choices belong to the reading layer and task config, not to
`schema.json`, zarr attrs, or manifest rows.

Current task-config loader:

- [`urban_embedding/data/processed/task_config.py`](../../urban_embedding/data/processed/task_config.py)

Current processed training pipeline config:

- [`urban_embedding/data/processed/pipeline_spec.py`](../../urban_embedding/data/processed/pipeline_spec.py)
- [`configs/pipeline/eump_full/mae_v3/pretrain.yaml`](../../configs/pipeline/eump_full/mae_v3/pretrain.yaml)
- [`configs/pipeline/eump_full/mae_v3/lst.yaml`](../../configs/pipeline/eump_full/mae_v3/lst.yaml)
- [`configs/pipeline/eump_full/scalar_mlp/lst.yaml`](../../configs/pipeline/eump_full/scalar_mlp/lst.yaml)

## Runtime-Derived Sources

Two reader-level sources are intentionally derived at read time instead of
persisted to zarr:

- `Area`
- `pop_sum`

They are computed from already-persisted sources and exposed through the reader
interface when requested by task config.

## Mask Semantics

Keep these masks separate:

- `valid_mask`
  whether a raster or scalar value is present and should participate
- `city_mask`
  semantic or weighting information that identifies the city region within a
  scene or crop

These should not be merged into one mask because they serve different roles in
the training pipeline.

## Availability Semantics

When a source is not available for a given scene or year:

- it is absent from the corresponding zarr group
- the manifest records it as missing
- the reader surfaces explicit unavailable behavior for task-required sources
  rather than silently filling values
- a source listed in `pipeline.inputs.optional_sources` is the exception:
  it keeps the sample-year compatible, materializes a zero-filled input block,
  and sets that block's `valid_mask` to all false

This applies to:

- genuinely missing raw data
- years outside a source's canonical range
- years intentionally dropped during canonicalization, such as padded `GAIA` and
  `POP` tail years from raw `BH8`

Optional input behavior belongs to the task config, not to the persisted store.
The source must still be known in `schema.json`; optional input only means that
some scene-years may be missing the source. Missing optional inputs are recorded
in `SceneRecord.meta["optional_missing_input_sources"]`.

Required inputs, downstream prediction targets, and non-optional aux targets
still filter incompatible sample-years or raise a clear error when loaded
directly.

## Crop Semantics

Crop sampling remains a runtime concern.

Recommended rule:

- persist the full aligned scene
- sample crop boxes at read time
- materialize crop tensors from the loaded `SceneRecord`

Current default crop size in the training code remains `256 x 256`.

## Code References

- `urban_embedding/data/contracts.py`
- `urban_embedding/data/crop.py`
- `urban_embedding/data/processed/reader.py`
- `urban_embedding/data/processed/exporter.py`
- `urban_embedding/data/processed/validation.py`
