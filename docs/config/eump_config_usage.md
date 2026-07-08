# EUMP Config Usage

## Config Layers

Do not pass long ad-hoc Hydra overrides. Use a launch suite and a mode:

```bash
bash script/training/run_eump_full_train.sh all   launch=eump_full/mae_v3 data=eump_full/server_v2
bash script/training/run_eump_full_train.sh lst   launch=eump_full/mae_v3 data=eump_full/server_v2
bash script/training/run_eump_full_train.sh pm25  launch=eump_full/mae_v3 data=eump_full/server_v2
EVAL_GPU=0 EVAL_NPROC=1 EVAL_MASTER_PORT=29501 LOG_PROGRESS_BAR=0 \
  bash script/training/run_eump_full_eval.sh lst launch=eump_full/mae_v3 \
    data=eump_full/server_v2 evaluation.checkpoint_source=best_model
```

| Config group | Purpose | Examples |
| --- | --- | --- |
| `data` | processed scene store, split manifest, preprocessing stats | `eump_full/server_v2`, `eump_full/raw_sample` |
| `launch` | maps a mode to a list of runnable jobs | `eump_full/mae_v3`, `eump_full/scalar_mlp` |
| `pipeline` | the actual Hydra-composed runnable config for a single task | `eump_full/mae_v3/pretrain`, `eump_full/mae_v3/lst` |
| `model` | model architecture and hyperparameters | `eump/swin/reconstruction`, `eump/city_scalar_mlp` |
| `eval` | evaluation options and checkpoint loading | `eump/downstream` |

Available train/eval modes per launch suite:

| Launch suite | Train modes | Eval modes |
| --- | --- | --- |
| `eump_full/mae_v3` | `all`, `pretrain`, `lst`, `precip`, `pm25`, `traffic` | `all`, `lst`, `precip`, `pm25`, `traffic` |
| `eump_full/scalar_mlp` | `all`, `mlp-cache`, `mlp-lst`, `mlp-precip`, `mlp-pm25`, `mlp-traffic` | `all`, `mlp-lst`, `mlp-precip`, `mlp-pm25`, `mlp-traffic` |
| `eump_full/aef_mlp` | `all`, `aef-mlp-lst`, `aef-mlp-precip`, `aef-mlp-pm25`, `aef-mlp-traffic` | `all`, `aef-mlp-lst`, `aef-mlp-precip`, `aef-mlp-pm25`, `aef-mlp-traffic` |
| `eump_full/architecture_v1` | `all`, `swin-unet`, `convnext-unet`, `pretrain`, `lst`, `precip` | `all`, `swin-unet`, `convnext-unet`, `lst`, `precip` |
| `eump_full/contrastive_v2` | `all`, `pretrain`, `lst`, `precip` | `all`, `lst`, `precip` |

`mae_v3` train mode `pm25` expands to `pm25_cv` (city-scalar CV); `traffic`
expands to `traffic_cv`. Eval modes `pm25`/`traffic` evaluate the saved CV fold
checkpoints.

`launch` configs are wrapper-only — they expand modes to pipeline names and are
never composed by Hydra directly. The shell wrappers reject `pipeline=...`
overrides to prevent mixing the two layers. Always pass `launch=` and a mode.

## Current Baseline: mae_v3

`eump_full/mae_v3` is the current Swin MAE baseline. Key properties:

- Same backbone, input channels, MAE reconstruction, and training schedule as mae_v2.
- `pipeline.loss.aux_weight = 0.25`
- PM2.5 raster aux and crop-scalar aux are **disabled** as pretrain targets
  (`aux_raster_target_weights.PM25=0.0`, `aux_source_weights.PM25=0.0`).
- PM2.5 is exposed as explicit downstream tasks: `pm25` (scalar `PM25_mean`,
  years 2015–2019) and `pm25-city`.
- `precip` is a scalar crop-level downstream target by default; the legacy
  raster version is available as `precip-raster`.

Pipeline configs under `mae_v3/` are self-contained and only inherit shared
profile/target building blocks from `configs/pipeline/eump_full/_shared`.

## Runtime Registry

Training/evaluation dispatch is resolved by `(pipeline.kind, model.kind)`:

| `pipeline.kind` | `model.kind` | Runtime |
| --- | --- | --- |
| `eump_reconstruction` | `swin_reconstruction` | Swin MAE pretrain |
| `eump_reconstruction` | `swin_unet_reconstruction` | Swin encoder + UNet decoder |
| `eump_reconstruction` | `convnext_unet_reconstruction` | ConvNeXt encoder + UNet decoder |
| `eump_contrastive` | `swin_contrastive` | Swin MoCo contrastive pretrain |
| `eump_downstream` | `swin_downstream_light` | Frozen Swin encoder + downstream head |
| `eump_city_scalar_downstream` | `city_scalar_mlp` | City-level scalar MLP baseline |

`eump_scalar_mlp_downstream` is a compatibility alias for `eump_city_scalar_downstream`.

## Downstream Targets

One prediction target per downstream config:

- **Raster**: `LST`; legacy raster `precip` via `precip-raster`
- **Scalar**: `Traffic`, `PM25_mean` (years 2015–2019), `GHG_mean`, `GHG_emission`
- **Crop-scalar**: default `precip`, legacy `PM25_city_mean`, city-level MLP targets

Pretrain auxiliary targets (in addition to input reconstruction):

- Scalar input reconstruction: `Ta`, `precip_bg`, `wind`
- Aux raster/crop-scalar: `LC`, `LCZ`, `GAIA`, `road_density`, `Electricity`, `GDP`
- PM2.5 aux effective weight is 0 in `mae_v3`

Common loss and mask types:

| Name | Use |
| --- | --- |
| `mae`, `mse`, `huber` | continuous regression |
| `hurdle_mae`, `hurdle_mse`, `hurdle_huber` | zero-inflated continuous raster |
| `bce` | binary/share logits |
| `valid` | source valid mask only |
| `valid_and_masked` | source valid mask and MAE masked patches only |
| `valid_and_city` | source valid mask inside city mask |

## Optional Availability

Two kinds of optional — do not mix:

| Setting | Scope | Behavior when missing |
| --- | --- | --- |
| target `optional: true` | reconstruction aux target only | sample-year kept; target invalid/NaN; no loss contribution |
| `pipeline.inputs.optional_sources` | input source only | sample-year kept; channel zero-filled; valid mask all-false |

Ordinary downstream prediction targets remain required. Missing targets are
never silently filled.

## City-Scalar CV

PM2.5 and Traffic use fixed-test K-fold CV rather than a single train/val/test
downstream run. The original `test` cities are held out; `train+val` cities are
split into K folds.

```bash
bash script/experiments/run_current_traffic_cv.sh data=eump_full/server_v2
bash script/experiments/run_current_pm25_cv.sh   data=eump_full/server_v2
```

CV outputs:

```text
log/<experiment.group>/traffic_cv_seed42/traffic_cv_results.jsonl
log/<experiment.group>/pm25_cv_seed42/PM25_mean_cv_results.jsonl
model/<experiment.group>/Traffic_cv/swin_embed_head_128_d05_fold<k>/
model/<experiment.group>/PM25_mean_cv/PM25_mean_cv_fold<k>/
```

## Checkpoints And Outputs

Training artifacts:

```text
model/<experiment.group>/<experiment.name>/
log/<experiment.group>/<experiment.name>/<date>/<time>/
```

| File | Contents |
| --- | --- |
| `eump_reconstruction_best_checkpoint.pt` | pretrain best checkpoint; default encoder source for downstream |
| `eump_downstream_model.pt` | downstream best/final evaluation model |
| `eump_downstream_checkpoint.pt` | downstream resume checkpoint |

Evaluation metrics:

```text
model/<experiment.group>/<experiment.name>/evaluation/<split>_metrics.json
```

## Architecture And Contrastive Comparison Suites

These are launched the same way as the main MAE suite:

```bash
# Swin-UNet or ConvNeXt-UNet
GPU=0,1,2,3,4,5,6,7 NPROC=8 LOG_PROGRESS_BAR=0 \
  bash script/training/run_eump_full_train.sh swin-unet \
    launch=eump_full/architecture_v1 data=eump_full/server_v2

# MoCo contrastive pretrain
GPU=0,1,2,3,4,5,6,7 NPROC=8 LOG_PROGRESS_BAR=0 \
  bash script/training/run_eump_full_train.sh all \
    launch=eump_full/contrastive_v2 data=eump_full/server_v2
```

See `experiments/eump_architecture_ablation/EXPERIMENT.md` for full details.

## Domain Transfer

```bash
GPU=0,1,2,3,4,5,6,7 NPROC=8 EVAL_GPU=0 EVAL_NPROC=1 EVAL_MASTER_PORT=29501 LOG_PROGRESS_BAR=0 \
  python experiments/eump_domain_transfer/run.py \
    --scenarios climate_acd_to_b,gdp_nonpoor_to_poor,income_hic_lic_lmic_to_umic \
    data=eump_full/server_v2
```

Transfer uses the global `mae_v3` pretrain checkpoint directly and does not run
source-domain pretrain. Each scenario trains on the full source domain and
evaluates on the target domain using the final downstream checkpoint.

See `experiments/eump_domain_transfer/EXPERIMENT.md` for scenario definitions
and city class counts.

## Dry Run

```bash
DRY_RUN=1 GPU=0 NPROC=1 LOG_PROGRESS_BAR=0 \
  bash script/training/run_eump_full_train.sh all \
    launch=eump_full/mae_v3 data=eump_full/server_v2
```
