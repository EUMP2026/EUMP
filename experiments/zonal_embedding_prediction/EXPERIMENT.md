# Zonal Embedding Prediction

Uses pooled embeddings from the frozen `mae_v3` Swin encoder to train a
lightweight multi-task MLP head that predicts city-level zonal variables from
`data/csv_data/zonal_merged_2020.csv`.

## Target rules

- CSV must contain `ID` (unique) and `split` (`train`/`val`/`test`).
- The following variable groups are excluded as prediction targets (current
  downstream tasks and deprecated tasks): `LST_*`, `Precip_*`, `Traffic_*`,
  `PM25_*`, `PM25_mean_*`, `GHG_mean_*`, `GHG_sum_*`.
- `LCZ_1..LCZ_17` are treated as a 17-dim distribution (row-normalised to
  sum=1, soft cross-entropy loss).
- Static variables (e.g. `zone_area_km2`, `BF`, `DEM`, `BH_*`, `water`) use
  the `static_embedding_year` (default 2018) embedding only.
- Dynamic annual variables use the same-year embedding over `dynamic_years`
  (default 2003–2020).

## Normalisation

- Z-score stats are computed from train split only.
- Annual variable groups (e.g. `GAIA_2003..GAIA_2020`) share one mean/std.
- Static variables are normalised per column.
- Training loss uses normalised scale; evaluation metrics are denormalised back
  to CSV original scale. LCZ uses soft cross-entropy, not z-score.

## Prerequisites

`model/eump_full_mae_v3/pretrain/eump_reconstruction_best_checkpoint.pt` must
exist. Run the `mae_v3` pretrain stage first if it is missing.

## Commands

Full run:

```bash
GPU=0,1,2,3,4,5,6,7 NPROC=8 LOG_PROGRESS_BAR=0 \
python experiments/zonal_embedding_prediction/run.py data=eump_full/server_v2
```

Export embeddings only:

```bash
GPU=0,1,2,3,4,5,6,7 NPROC=8 LOG_PROGRESS_BAR=0 \
python experiments/zonal_embedding_prediction/run.py \
  --prepare-embeddings-only data=eump_full/server_v2
```

Dry run:

```bash
DRY_RUN=1 GPU=0 NPROC=1 LOG_PROGRESS_BAR=0 \
python experiments/zonal_embedding_prediction/run.py data=eump_full/server_v2
```

## Outputs

```text
experiments/zonal_embedding_prediction/outputs/<run_name>/
  runtime_config.yaml
  target_manifest.json
  normalization_stats.json
  cache/swin_embeddings/*.npz
  model/swin_zonal_multitask_head.pt
  metrics.json
  metrics.csv
  predictions_{train,val,test}.npz
```
