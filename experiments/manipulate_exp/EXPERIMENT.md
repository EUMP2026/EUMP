# EUMP Input Manipulation Experiment

## Objective

Measure how a downstream predictor (e.g. LST or traffic) responds to controlled
physical-space changes in a single input channel (e.g. NDVI or POP).

The first use case is:
- **City**: Beijing (scene_id = 782)
- **Year**: 2020
- **Perturbation**: NDVI + 0.1 (within city mask)
- **Predictor**: mae_v3 LST downstream model
- **Output**: spatial delta_lst map = prediction_after − prediction_before

The bulk use case is:
- **Cities**: all compatible 2020 cities across train + val + test
- **Perturbations**: NDVI + 0.1, +0.3, +0.5
- **Mask scope**: either all valid pixels in the center crop, or only pixels inside
  `city_mask`
- **Output**: one `.npz` with `lst_ndvi01`, `lst_ndvi03`, and `lst_ndvi05`

## Methodology

1. Load the mae_v3 LST downstream checkpoint (frozen Swin-L encoder + light head).
2. Force center crop (`pipeline.center_ratio=1.0`) before building the data loader.
3. In physical space, add `delta` to the selected input channel inside `city_mask`.
   Clip with source defaults when no explicit bounds are provided: NDVI `[-1, 1]`,
   POP `[0, +inf]`.
4. Convert the perturbed physical values back to processed space.
5. Run the downstream model on the original and perturbed batches.
6. Convert both predictions to physical space via `inverse_processed_tensor`.
7. Compute `delta_pred = prediction_after − prediction_before`.
8. For raster targets, also report `cooling = prediction_before − prediction_after`;
   positive cooling means the predicted target is lower after perturbation.
9. Single-scene raster mode also renders a PNG; bulk mode writes arrays only.

## Inputs

- Processed EUMP scene store (default evaluate config uses `data=eump_full/server_v2`)
- `preprocessing_v1.json` (must have been generated with `build_eump_preprocessing.py`)
- mae_v3 LST downstream checkpoint
  (`model/eump_full_mae_v3/lst/eump_downstream_model.pt` by default)

## Outputs

Single-scene outputs are written to `output_dir/<scene_id>_<year>/`:

| File | Contents |
|---|---|
| `delta_lst_<id>_<year>.npz` | NumPy arrays: `delta_prediction`, `pred_before`, `pred_after`, `input_before`, `input_after`, `city_mask`, `perturbation_mask`, `target` (if available), `target_valid_mask` |
| `delta_lst_<id>_<year>.png` | 3-panel matplotlib figure: input before, input after, delta prediction |
| `summary_<id>_<year>.json` | Experiment metadata, model/checkpoint info, perturbation params, output statistics |

Bulk outputs are written to `output_dir/`:

| File | Contents |
|---|---|
| `all_cities_ndvi_lst_2020.npz` | `scene_ids`, `years`, `splits`, `crop_boxes`, `deltas`, `channel_names`, `lst_ndvi01`, `lst_ndvi03`, `lst_ndvi05` |
| `all_cities_ndvi_lst_2020_stats.csv` | Per-city/per-delta response statistics, including city-mask mean cooling |
| `summary.json` | Checkpoint, data, perturbation, target, runtime metadata, and aggregate response statistics |

Each `lst_ndviXX` array has shape `(N, 5, 256, 256)` and channel order:
`ndvi0`, `ndvi1`, `lst0`, `lst1`, `city_mask`.

For the stats CSV, `mean_delta_target_city` is `lst1 - lst0`, while
`mean_cooling_city` is `lst0 - lst1`; therefore positive `mean_cooling_city`
means average predicted LST decreases inside `city_mask`.

## Extending To Other Channels and Predictors

Change the source and predictor via command-line overrides:

```bash
# Perturb POP instead of NDVI
python experiments/manipulate_exp/run.py \
  --source POP --delta 100 --min-value 0 \
  --scene-id 782 --year 2020 \
  data=eump_full/server_v2 pipeline=eump_full/mae_v3/lst

# Predict precip instead of LST
python experiments/manipulate_exp/run.py \
  --source NDVI --delta 0.1 \
  --scene-id 782 --year 2020 \
  data=eump_full/server_v2 pipeline=eump_full/mae_v3/precip

# Different city or year
python experiments/manipulate_exp/run.py \
  --scene-id 123 --year 2015 --delta 0.1 \
  data=eump_full/server_v2 pipeline=eump_full/mae_v3/lst
```

## Commands

**Compile check (no data needed):**

```bash
conda run -n EUMP python -m compileall experiments/manipulate_exp/run.py
```

**Counterfactual NDVI buffer scenarios → LST export (mae_v3):**

This reads `ndvi_counterfactual_scenarios.npz` and predicts physical LST for
six NDVI scenarios: `ndvi0`, `ndvi_urban`, `ndvi_buffer_5km`,
`ndvi_buffer_10km`, `ndvi_both_5km`, and `ndvi_both_10km`.

```bash
experiments/manipulate_exp/run_ndvi_counterfactual_lst.sh
```

Multi-GPU example:

```bash
GPU=0,1,2,3,4,5,6,7 NPROC=8 BATCH_SIZE=4 \
experiments/manipulate_exp/run_ndvi_counterfactual_lst.sh
```

Outputs default to:

```text
experiments/manipulate_exp/outputs/ndvi_counterfactual_lst_2020/
```

The merged NPZ keeps `urban_mask`, `buffer_5km`, `buffer_10km`, all six input
NDVI arrays, and matching physical LST prediction arrays with `*_lst` suffixes.

**Local smoke run (raw_sample data — use a scene_id present in that sample):**

```bash
conda run -n EUMP python experiments/manipulate_exp/run.py \
  --scene-id <sample_scene_id> --year 2020 \
  data=eump_full/raw_sample pipeline=eump_full/mae_v3/lst
```

**Server run (full data, Beijing):**

```bash
conda activate EUMP
python experiments/manipulate_exp/run.py \
  --scene-id 782 --year 2020 --delta 0.1 \
  data=eump_full/server_v2 pipeline=eump_full/mae_v3/lst
```

**Bulk all-city NDVI → LST export, full valid crop (8 GPUs):**

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 conda run -n EUMP \
  torchrun --nproc_per_node=8 experiments/manipulate_exp/run.py \
  --all-cities \
  --year 2020 \
  --source NDVI \
  --deltas 0.1,0.3,0.5 \
  --mask-scope all \
  --output-dir experiments/manipulate_exp/outputs/ndvi_lst_2020 \
  data=eump_full/server_v2 pipeline=eump_full/mae_v3/lst
```

**Bulk all-city NDVI → LST export, urban-mask only (8 GPUs):**

This is the current urban-mask task. It writes to an independent folder so it
does not overwrite the previous full-crop export in `ndvi_lst_2020`.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 conda run -n EUMP \
  torchrun --nproc_per_node=8 experiments/manipulate_exp/run.py \
  --all-cities \
  --year 2020 \
  --source NDVI \
  --deltas 0.1,0.3,0.5 \
  --mask-scope city \
  --output-dir experiments/manipulate_exp/outputs/ndvi_lst_2020_city_mask \
  data=eump_full/server_v2 pipeline=eump_full/mae_v3/lst
```

If the default torchrun port is occupied, set a separate port:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MASTER_PORT=29521 conda run -n EUMP \
  torchrun --nproc_per_node=8 experiments/manipulate_exp/run.py \
  --all-cities \
  --year 2020 \
  --source NDVI \
  --deltas 0.1,0.3,0.5 \
  --mask-scope city \
  --output-dir experiments/manipulate_exp/outputs/ndvi_lst_2020_city_mask \
  data=eump_full/server_v2 pipeline=eump_full/mae_v3/lst
```

Urban-mask output files:

| File | Contents |
|---|---|
| `experiments/manipulate_exp/outputs/ndvi_lst_2020_city_mask/all_cities_ndvi_lst_2020.npz` | `scene_ids`, `years`, `splits`, `crop_boxes`, `deltas`, `channel_names`, `lst_ndvi01`, `lst_ndvi03`, `lst_ndvi05` |
| `experiments/manipulate_exp/outputs/ndvi_lst_2020_city_mask/all_cities_ndvi_lst_2020_stats.csv` | Per-city/per-delta city-mask response statistics |
| `experiments/manipulate_exp/outputs/ndvi_lst_2020_city_mask/summary.json` | Metadata with `perturbation.mask_scope = city` plus aggregate response statistics |

**Bulk smoke test:**

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n EUMP python experiments/manipulate_exp/run.py \
  --all-cities --year 2020 --source NDVI --deltas 0.1 \
  --limit-cities 4 \
  --mask-scope all \
  data=eump_full/server_v2 pipeline=eump_full/mae_v3/lst
```

**Compute stats only from an existing bulk NPZ:**

Use this when the bulk inference arrays already exist and you only need the
per-city cooling/statistics CSV plus updated `summary.json`.

```bash
conda run -n EUMP python experiments/manipulate_exp/compute_bulk_stats.py \
  experiments/manipulate_exp/outputs/ndvi_lst_2020_city_mask/all_cities_ndvi_lst_2020.npz
```

**Custom output directory:**

```bash
python experiments/manipulate_exp/run.py \
  --scene-id 782 --year 2020 \
  --output-dir experiments/manipulate_exp/outputs/beijing \
  data=eump_full/server_v2
```

## Notes

- Single-scene mode searches the requested split first, then falls back to the other
  splits if the target scene/year is elsewhere.
- Bulk mode combines train, val, and test rows for the requested year, then shards
  deterministically across ranks when launched with `torchrun`.
- The script exits with a clear error if the target scene_id/year is not found in the
  selected split and year_range.
- `delta` is additive in physical units (NDVI is typically in [-1, 1]; delta=0.1 is a 10%
  reflectance-index shift).
- The perturbation is applied only within `city_mask` by default (`--mask-scope city`).
  In `--all-cities` mode the default is `--mask-scope all`.
  Pass `--mask-scope city` explicitly for the urban-mask export; otherwise bulk
  mode intentionally uses all valid crop pixels.
