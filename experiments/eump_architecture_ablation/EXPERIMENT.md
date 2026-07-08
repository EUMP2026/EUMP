# EUMP Architecture + Pretrain-Objective Ablation

Compares three alternatives to the `mae_v3` Swin MAE baseline:

| Variant | Family | What changes vs mae_v3 |
| --- | --- | --- |
| `swin_unet` | architecture | Swin encoder kept, decoder replaced with UNet-style symmetric upsampler |
| `convnext_unet` | architecture | Encoder swapped to ConvNeXt-L (params ~Swin-L), same UNet decoder |
| `contrastive_v2` | pretrain_objective | Same Swin encoder, MoCo momentum-encoder contrastive pretrain |

Everything else stays fixed: data (`eump_full/server_v2`), input channels,
year range `2003–2020`, crop size `256`, downstream tasks `lst/precip/traffic/pm25`,
downstream hyperparameters.

Both UNet variants use input-reconstruction only at pretrain time (aux targets
disabled). Only the `(B, C, H/32, W/32)` bottleneck feeds downstream heads —
UNet skip connections are not propagated.

## Commands

Full run (all variants):

```bash
GPU=0,1,2,3,4,5,6,7 NPROC=8 EVAL_GPU=0 EVAL_NPROC=1 LOG_PROGRESS_BAR=0 \
python experiments/eump_architecture_ablation/run.py data=eump_full/server_v2
```

Single variant:

```bash
GPU=0,1,2,3,4,5,6,7 NPROC=8 LOG_PROGRESS_BAR=0 \
python experiments/eump_architecture_ablation/run.py \
  --variants swin_unet --stages train data=eump_full/server_v2
```

Smoke test:

```bash
GPU=0 NPROC=1 EVAL_GPU=0 EVAL_NPROC=1 LOG_PROGRESS_BAR=0 \
python experiments/eump_architecture_ablation/run.py --smoke data=eump_full/server_v2
```

## Collecting results

Results are collected via the ablation suite's `architecture` family, which
references these experiment groups via `reference_group`. Run after this
experiment completes:

```bash
python experiments/eump_ablation_suite/run.py \
  --stages collect --families architecture --continue-run data=eump_full/server_v2
```

## Artifacts

```text
model/eump_full_architecture_v1_swin_unet/pretrain/eump_reconstruction_best_checkpoint.pt
model/eump_full_architecture_v1_convnext_unet/pretrain/eump_reconstruction_best_checkpoint.pt
model/eump_full_contrastive_v2/pretrain/eump_contrastive_best_checkpoint.pt
```

Downstream checkpoints: `model/<group>/<task>/eump_downstream_model.pt`

CV outputs:

```text
log/<group>/traffic_cv_mae_v3_seed42/traffic_cv_results.jsonl
log/<group>/pm25_cv_mae_v3_seed42/PM25_mean_cv_results.jsonl
model/<group>/Traffic_cv/swin_embed_head_128_d05_fold<k>/
model/<group>/PM25_mean_cv/PM25_mean_cv_fold<k>/
```
