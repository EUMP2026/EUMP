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

The training and evaluation launchers default to `eump_full/mae_v3`.
The default data configuration is `eump_full/server_v2`, whose scene-store
root is `data/processed_v2/`.

Swin model construction also reads the local Hugging Face-format backbone
directory `model/${experiment.pretrain_model}/`, which is `model/swin_L/` for
the main MAE v3 suite. This directory must contain `config.json`; runs using
the standard pretrained initialization also need its backbone weights
(`pytorch_model.bin` or the corresponding supported safetensors file).
Downstream runs load their selected EUMP encoder checkpoint separately. The
backbone files, trained checkpoints, and data store are external runtime
artifacts and are excluded from Git.

Available train/eval modes per launch suite:

| Launch suite | Train modes | Eval modes |
| --- | --- | --- |
| `eump_full/mae_v3` | `all`, `pretrain`, `lst`, `precip`, `pm25`, `traffic` | `all`, `lst`, `precip`, `pm25`, `traffic` |
| `eump_full/mae_v3_one_stage` | `all`, `lst`, `precip`, `pm25`, `traffic` | `all`, `lst`, `precip`, `pm25`, `traffic` |
| `eump_full/scalar_mlp` | `all`, `mlp-cache`, `mlp-lst`, `mlp-precip`, `mlp-pm25`, `mlp-pm25-city`, `mlp-ghg`, `mlp-ghg-sum`, `mlp-traffic` | `all`, `mlp-lst`, `mlp-precip`, `mlp-pm25`, `mlp-pm25-city`, `mlp-ghg`, `mlp-ghg-sum`, `mlp-traffic` |
| `eump_full/aef_mlp` | `all`, `aef-mlp-cache`, `aef-mlp-lst`, `aef-mlp-precip`, `aef-mlp-pm25`, `aef-mlp-pm25-city`, `aef-mlp-traffic` | `all`, `aef-mlp-lst`, `aef-mlp-precip`, `aef-mlp-pm25`, `aef-mlp-pm25-city`, `aef-mlp-traffic` |
| `eump_full/architecture_v1` | `all`, `swin-unet`, `convnext-unet`, `pretrain`, `lst`, `precip` | `all`, `swin-unet`, `convnext-unet`, `lst`, `precip` |
| `eump_full/contrastive_v2` | `all`, `pretrain`, `lst`, `precip` | `all`, `lst`, `precip` |

`mae_v3` train mode `pm25` expands to `pm25_cv` (city-scalar CV); `traffic`
expands to `traffic_cv`. Eval modes `pm25`/`traffic` evaluate the saved CV fold
checkpoints.

`mae_v3_one_stage` trains the supervised Swin encoder and prediction head
jointly (`downstream.encoder_source=swin_pretrained`,
`downstream.freeze_encoder=false`), starting from the standard Swin backbone
initialization without an EUMP MAE stage. The `*-pm25-city` modes are aliases
for their respective `*-pm25` jobs. Scalar MLP `all` also includes the GHG mean
task; `mlp-ghg` and `mlp-ghg-sum` are supported additional tasks outside the
four paper targets.

`launch` configs are wrapper-only — they expand modes to pipeline names and are
never composed by Hydra directly. The shell wrappers reject `pipeline=...`
overrides to prevent mixing the two layers. Always pass `launch=` and a mode.

## Current Baseline: mae_v3

`eump_full/mae_v3` is the current Swin MAE baseline. Key properties:

- Swin-L backbone with MAE input reconstruction and auxiliary supervision.
- `pipeline.loss.aux_weight = 0.25`
- PM2.5 raster aux and crop-scalar aux are **disabled** as pretrain targets
  (`aux_raster_target_weights.PM25=0.0`, `aux_source_weights.PM25=0.0`).
- PM2.5 uses `pm25` (scalar `PM25_mean`, years 2015–2019).
- `precip` predicts a scalar crop-level precipitation target.

Pipeline configs under `mae_v3/` are self-contained and only inherit shared
profile/target building blocks from `configs/pipeline/eump_full/_shared`.

## Runtime Registry

Training/evaluation dispatch is resolved by `(pipeline.kind, model.kind)`:

| `pipeline.kind` | `model.kind` | Runtime |
| --- | --- | --- |
| `eump_reconstruction` | `swin_reconstruction` | Swin MAE pretrain |
| `eump_reconstruction` | `swin_unet_reconstruction` | Swin encoder + UNet decoder |
| `eump_reconstruction` | `convnext_unet_reconstruction` | ConvNeXt encoder + UNet decoder |
| `eump_contrastive` | `swin_moco` | Swin momentum-encoder contrastive pretrain |
| `eump_downstream` | `swin_downstream_light` | Swin or ConvNeXt encoder + downstream head; frozen or jointly trained |
| `eump_city_scalar_downstream` | `city_scalar_mlp` | City-level scalar MLP baseline |

`eump_scalar_mlp_downstream` is a compatibility alias for `eump_city_scalar_downstream`.

## Downstream Targets

One prediction target per downstream config:

- **Raster**: `LST`
- **Scalar**: `Traffic`, `PM25_mean` (years 2015–2019)
- **Crop-scalar**: `precip`, city-level MLP targets

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

CV output directories and table prefixes use lowercase `traffic_cv` and
`pm25_cv`. For the main MAE v3 launcher:

```text
log/<experiment.group>/traffic_cv_seed42/traffic_cv_results.jsonl
log/<experiment.group>/pm25_cv_seed42/pm25_cv_results.jsonl
model/<experiment.group>/traffic_cv/swin_embed_head_128_d05_fold<k>/swin_embedding_scalar_head.pt
model/<experiment.group>/pm25_cv/swin_embed_head_128_d05_fold<k>/swin_embedding_scalar_head.pt
```

Here `experiment.group` defaults to `eump_full_mae_v3`. The standalone scripts
above default to groups `eump_traffic_cv` and `eump_pm25_cv`, with log run names
`traffic_cv_mae_v3_seed42` and `pm25_cv_mae_v3_seed42`, respectively. Their
checkpoint subdirectories use the same lowercase names.

## Checkpoints And Outputs

Training artifacts:

```text
model/<experiment.group>/<experiment.name>/
log/<experiment.group>/<experiment.name>/<date>/<time>/
```

| File | Contents |
| --- | --- |
| `eump_reconstruction_best_checkpoint.pt` | validation-best MAE checkpoint; encoder selected by the main downstream and planning runs |
| `eump_reconstruction_model.pt` | final MAE model weights |
| `eump_downstream_model.pt` | validation-best downstream model, saved to the configured `save_file` |
| `eump_downstream_checkpoint.pt` | resumable downstream training state |

These names describe the main MAE/downstream training path. Domain-transfer
runs use final checkpoints as specified below. Architecture-specific
checkpoint selection is documented in the
[architecture experiment](../../experiments/eump_architecture_ablation/EXPERIMENT.md).

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

See the [architecture experiment](../../experiments/eump_architecture_ablation/EXPERIMENT.md)
for the four-task comparison and the reported MoCo encoder selection.

## Domain Transfer

Two protocols are configured separately:

```bash
# Global MAE v3 encoder; no scenario-specific MAE stage
python experiments/eump_domain_transfer/run.py \
  --config experiments/eump_domain_transfer/config.yaml \
  data=eump_full/server_v2

# Source-domain MAE for Tropical and GDP Poor, then frozen-encoder downstream
python experiments/eump_domain_transfer/run.py \
  --config experiments/eump_domain_transfer/config_source_pretrain.yaml \
  data=eump_full/server_v2
```

`config.yaml` uses the global validation-best MAE v3 checkpoint and six
transfer scenarios. Its `pretrain.enabled=false` disables only the
scenario-specific MAE stage. `config_source_pretrain.yaml` enables source-domain
MAE initialized from the standard pretrained `swin_L` backbone, followed by
frozen-encoder downstream training. Both protocols cover LST, precipitation,
PM2.5, and Traffic. Scenario-specific MAE and downstream stages use their final
checkpoints, with no validation or target-based checkpoint selection.

See the [domain-transfer experiment](../../experiments/eump_domain_transfer/EXPERIMENT.md)
for scenario definitions, required inputs, execution options, and table exports.

## Dry Run

```bash
DRY_RUN=1 GPU=0 NPROC=1 LOG_PROGRESS_BAR=0 \
  bash script/training/run_eump_full_train.sh all \
    launch=eump_full/mae_v3 data=eump_full/server_v2
```
