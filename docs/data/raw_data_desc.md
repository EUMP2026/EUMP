# Raw Source Catalog

This document records the factual raw-source catalog and how each source maps
into `eump_scene_store_v1`.

Implementation references:

- `urban_embedding/data/processed/schema.py`
- `urban_embedding/data/processed/exporter.py`
- `urban_embedding/data/processed/validation.py`
- `configs/pipeline/eump_full/mae_v3/pretrain.yaml`

## Conventions

- `Default Data Role` and `Swin Output Targets` are task-config-derived
  summaries, not raw metadata or processed schema fields.
- `Swin Output Targets` records the current Swin training framework only:
  `eump_full/mae_v3/pretrain` plus Swin downstream `lst`, `precip`, `traffic`,
  and PM2.5 downstream tasks. It intentionally excludes scalar MLP baseline configs.
- `raw_resolution` records the nominal source resolution on disk. Persisted
  raster arrays in `data/processed_v2` are aligned to the `BH8_250m` reference grid.
- `Area` and `pop_sum` are runtime-derived reader features. They are available
  through the processed-store reader but are not materialized as on-disk zarr
  sources.
- `precip` is the yearly raster source from `raw_data/precip_v1`.
  The default `mae_v3` downstream task predicts its crop-level scalar mean.
- `precip_bg` is the yearly scalar climate source from
  `csv_data/clim_values.csv` column family `PRECTOTCORR`.
- `PM25` is the raster schema source name even though display text may appear
  as `PM2.5`. `PM25_mean` is the persisted yearly scalar source from
  `csv_data/pm25_mean.csv`.
- The processed schema and `preprocessing_v1.json` must describe the same source
  set; the loader validates their compatibility.
- `mae_v3` defines its pretrain recipe directly and disables effective
  PM2.5 aux supervision with `aux_raster_target_weights.PM25=0.0` and
  `loss.aux_source_weights.PM25=0.0`. The PM2.5 downstream target is the scalar
  `PM25_mean`, rather than the `PM25` raster.
- `road_density` is an aux target in the default `mae_v3` pretrain. In
  input-channel ablations it can also be used as an optional input via
  `pipeline.inputs.optional_sources`, preserving `2003-2007` samples with
  zero-filled data and all-false valid masks.
- `No data` records the schema nodata rule used by the processed exporter. `/`
  means no explicit nodata sentinel beyond finite-value checks.
- `Normalization` records the `preprocessing_v1` strategy, rather than the
  raw-value `normalization_v1.json` summary artifact.
- For `log10(1 + x) + z-score`, invalid values are excluded before statistics:
  values must pass `valid_mask`, be finite after restore, and satisfy
  `restored > -1`.
- `POP=0`, `BH=0`, `BF=0`, and `DEM=0` are treated as valid physical zeros,
  not nodata.
- `Electricity` raw files document nodata approximately as `-3.4E+38`, while
  actual pixels may store the nearby float32 sentinel around
  `-3.402823E+38`. The processed exporter treats extreme float nodata values
  with threshold-style matching instead of exact equality, so these sentinel
  values are excluded from `valid_mask` and raw-stat recomputation.

## Persisted Sources

The loss column follows the MAE v3 reconstruction and downstream recipes.
`hurdle_mae` combines binary presence classification with MAE regression of
positive values. Categorical rasters use CE, while their crop distributions
use KL. MAE pretraining uses a global auxiliary-loss multiplier of 0.25;
target-specific weights are shown where they differ from 1. The source catalog covers all stored years;
the paper recipe's year ranges and split counts are listed in the
[MAE v3 source table](mae_v3_data_description.csv).

| Source | Raw reso | Raw materialized years | Canonical years in `processed_v2` | Modality | Storage level | Raw source dir | No data | Normalization | Notes | Loss | Default Data Role | Swin Output Targets |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `BH` | 250 | static | static | raster | static | `raw_data/BH8_v1` | / | `log10(1 + x)` + z-score | aligned to `BH8_250m`; zero is valid | `hurdle_mae`; crop MAE | Input/Target | pretrain `input_reconstruction/BH` hurdle slice; `BH_city_positive_share` (w=0); `BH_city_positive_mean` |
| `BF` | 250 | static | static | raster | static | `raw_data/BH8_v1` | / | `log10(1 + x)` + z-score | aligned to `BH8_250m`; zero is valid | `hurdle_mae`; crop MAE | Input/Target | pretrain `input_reconstruction/BF` hurdle slice; `BF_city_positive_share` (w=0); `BF_city_positive_mean` |
| `water` | 250 | static | static | raster | static | `raw_data/BH8_v1` | / | `x / 100` | binary-like values `0` or `100` | BCE | Input/Target | pretrain `input_reconstruction/water`; `water_city_share` |
| `DEM` | 250 | static | static | raster | static | `raw_data/BH8_v1` | / | `log10(1 + x)` + z-score | aligned to `BH8_250m`; zero is valid | MAE | Input/Target | pretrain `input_reconstruction/DEM`; `DEM_city_mean` |
| `LCZ` | 250 | static | static | raster | static | `raw_data/LCZ` | `0` | one-hot over categories `1..17` | static raster when present | CE / KL | Target | pretrain raster aux `LCZ`; `LCZ_city_distribution` |
| `city_mask` | 250 | static | static | raster | static | `raw_data/BH8_v1` | / | pass-through | semantic mask source; binary values | BCE | Input/Target | pretrain `input_reconstruction/city_mask`; `city_mask_crop_share` |
| `NDVI` | 500 | 2003-2022 | 2003-2022 | raster | yearly | `raw_data/VI3_v1` | `0` | restore `x = raw / 10000 - 1`, then z-score | resampled to `BH8_250m`; raw storage scale is `(x + 1) * 10000`; bounded signed index, so no log transform | MAE | Input/Target | pretrain `input_reconstruction/NDVI`; `NDVI_city_mean` |
| `NL` | 1000 | 2000-2021 | 2000-2021 | raster | yearly | `raw_data/NL_v1` | / | `log10(1 + x)` + z-score | resampled to `BH8_250m`; zero is valid | MAE | Input/Target | pretrain `input_reconstruction/NL`; `NL_city_mean` |
| `POP` | 250 | 2003-2022 | 2003-2020 | raster | yearly | `raw_data/BH8_v1` | / | `log10(1 + x)` + z-score | zero population is valid; raw tail years are materialized padding and canonical store drops 2021-2022 | `hurdle_mae`; crop MAE | Input/Target | pretrain `input_reconstruction/POP` hurdle slice; `POP_city_positive_share` (w=0); `POP_city_positive_mean` |
| `LC` | 500 | 2003-2022 | 2003-2022 | raster | yearly | `raw_data/land_cover` | `0` | one-hot over categories `1..17` | resampled to `BH8_250m` | CE / KL | Target | pretrain raster aux `LC`; `LC_city_distribution` |
| `GAIA` | 250 | 2003-2022 | 2003-2018 | raster | yearly | `raw_data/BH8_v1` | / | `log10(1 + x)` + z-score | raw tail years are materialized padding and canonical store drops 2019-2022 | `hurdle_mae`; crop MAE | Target | pretrain raster aux `GAIA`; `GAIA_city_positive_share` (w=0); `GAIA_city_positive_mean` |
| `PM25` | 1000 | 2017-2022 | 2017-2022 | raster | yearly | `raw_data/PM25` | / | `log10(1 + x)` + z-score | display name may be shown as `PM2.5`; effective pretrain aux is disabled in `mae_v3` | MAE (w=0) | Disabled auxiliary target | PM2.5 raster auxiliary supervision is disabled in `mae_v3` |
| `road_density` | 250 | 2008-2022 | 2008-2022 | raster | yearly | `raw_data/road_density` | `-9999` | pass-through | binary-like values after valid filtering | BCE | Target | pretrain raster aux `road_density`; `road_density_city_share` |
| `Electricity` | 1000 | 2003-2019 | 2003-2019 | raster | yearly | `raw_data/electricity` | approx. `-3.4E+38`; actual pixels may be around `-3.402823E+38` | `log10(1 + x)` + z-score | extreme float nodata is matched by threshold rather than exact equality | MAE | Target | pretrain raster aux `Electricity`; `Electricity_city_mean` |
| `GDP` | 1000 | 2000, 2005, 2010, 2015, 2020, 2024 | 2000, 2005, 2010, 2015, 2020, 2024 | raster | yearly | `raw_data/GDP` | `NaN` | `log10(1 + x)` + z-score | treated as a normal yearly raster source | `hurdle_mae`; crop MAE | Target | pretrain raster aux `GDP`; `GDP_city_positive_share` (w=0); `GDP_city_positive_mean` |
| `LST` | 1000 | 2003-2022 | 2003-2022 | raster | yearly | `raw_data/lst_v1` | `0` | restore `x = raw / 50`, then z-score | resampled to `BH8_250m`; raw storage scale is `50 * x` | MAE | Predict | downstream `lst` raster target `LST` |
| `precip` | 4000 | 2003-2024 | 2003-2024 | raster | yearly | `raw_data/precip_v1` | / | `log10(1 + x)` + z-score | raster precipitation source, resampled to `BH8_250m`; default downstream aggregates the crop to a scalar mean | MAE | Predict | downstream `precip` crop-scalar target |
| `Ta` | scalar | 2003-2022 | 2003-2022 | scalar | yearly | `csv_data/clim_values.csv` | / | z-score | raw source name `TLML` | MAE | Input/Target | pretrain scalar `input_reconstruction/Ta`; broadcast raster slice has w=0 |
| `precip_bg` | scalar | 2003-2022 | 2003-2022 | scalar | yearly | `csv_data/clim_values.csv` | / | z-score | raw source name `PRECTOTCORR` | MAE | Input/Target | pretrain scalar `input_reconstruction/precip_bg`; broadcast raster slice has w=0 |
| `wind` | scalar | 2003-2022 | 2003-2022 | scalar | yearly | `csv_data/clim_values.csv` | / | z-score | raw source name `SPEED` | MAE | Input/Target | pretrain scalar `input_reconstruction/wind`; broadcast raster slice has w=0 |
| `Traffic` | scalar | 2017-2021 | 2017-2021 | scalar | yearly | `csv_data/traffic.csv` | `NaN` | pass-through | yearly scalar | MAE | Predict | downstream `traffic` scalar target `Traffic` |
| `PM25_mean` | scalar | 2015-2019 | 2015-2019 | scalar | yearly | `csv_data/pm25_mean.csv` (`PM25_mean_YYYY`) | `NaN` | `log10(1 + x)` + z-score | sparse yearly PM2.5 mean scalar extracted from `zonal_merged_2020.csv` | MAE | Predict | downstream `pm25` scalar target `PM25_mean` (fixed-test CV) |

The schema additionally supports `GHG_mean` and `GHG_emission` scalar sources;
these are outside the paper's four downstream targets. See the
[processed source set](data_description.md#persisted-source-set).

## Runtime-Derived Sources

These sources are supported by the reader layer but are not written into zarr
stores.

| Source | Derived from | Modality | Availability | Notes | Swin Output Targets |
| --- | --- | --- | --- | --- | --- |
| `Area` | `city_mask` | scalar | static | scene-level city area summary | not configured as a Swin output target |
| `pop_sum` | `POP`, `city_mask` | scalar | yearly | year-level population sum over valid city pixels | not configured as a Swin output target |

## `BH8` Padding Rules

The materialized `BH8` tail bands repeat the last available year:

- `GAIA_15 == GAIA_16 == GAIA_17 == GAIA_18 == GAIA_19`
- `POP_17 == POP_18 == POP_19`
- `GAIA_14 != GAIA_15` for most samples
- `POP_16 != POP_17` for most samples

This supports the canonicalization rule used in `data/processed_v2`:

- `GAIA`
  raw materialized years: `2003-2022`
  canonical years: `2003-2018`
- `POP`
  raw materialized years: `2003-2022`
  canonical years: `2003-2020`

The processed store records these padding facts in schema metadata but does not
materialize the padded years as fake yearly sources.
