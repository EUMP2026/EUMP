# EUMP Architecture and Pretraining Objective Comparison

This experiment supplies the architecture and contrastive-pretraining rows
of the EUMP ablation comparison on LST, precipitation, PM2.5, and Traffic.
Each variant follows its own reconstruction or contrastive training recipe.
The [ablation suite](../eump_ablation_suite/EXPERIMENT.md) provides the
mainline EUMP reference and the other pretraining and input ablations.

| CLI variant | Result identifier | Encoder and pretraining |
| --- | --- | --- |
| `swin_unet` | `swin_unet` | Swin-L with a UNet-style reconstruction decoder |
| `convnext_unet` | `convnext_unet` | ConvNeXt-L with a UNet-style reconstruction decoder |
| `contrastive_v2` | `moco` | Swin-L with MoCo momentum-encoder contrastive pretraining |

Both UNet variants train input reconstruction without auxiliary targets.
Their decoder outputs use ordinary MAE, including the building, population,
water, and city-mask reconstruction channels. MoCo uses a contrastive
objective with a momentum encoder. The variant pipeline YAML files define
the complete settings.

| Variant | Encoder initialization | Default pretraining epochs |
| --- | --- | --- |
| Swin-UNet | Pretrained Swin-L | 200 |
| ConvNeXt-UNet | Random ConvNeXt-L | 200 |
| MoCo | Pretrained Swin-L | 100 |

MoCo uses pretrained Swin-L initialization, AdamW (learning rate `1e-4`,
weight decay `0.05`), global batch size 128, temperature `0.2`, momentum
`0.996`, and a 2048-to-128 projection head. The default budget is 100 epochs
with a 100-epoch warmup/cosine schedule and 10% warmup. The reported comparison
uses the epoch-31 checkpoint as the selected best model, stored at
`model/eump_full_contrastive_v2/pretrain/eump_contrastive_best_checkpoint.pt`.
The epoch-31 checkpoint was selected to avoid representation collapse.
All four downstream tasks use this frozen encoder.

The reconstruction decoders progressively upsample the bottleneck without
multi-scale encoder skip connections. Downstream heads consume only the
encoder bottleneck. The encoder is frozen after pretraining. ConvNeXt
downstream configurations explicitly select
`model.backbone_kind=convnext` and
`downstream.encoder_source=convnext_reconstruction_checkpoint`.

## Dependencies and evaluation

Use `requirements.txt`, compatible PyTorch/CUDA, and the
`data=eump_full/server_v2` processed scene store at `data/processed_v2/`,
including schema/scene manifests, Zarr arrays, preprocessing statistics,
and the scene split manifest. Inputs use the configured 11 channels including
city mask. Pretraining and LST/precipitation cover 2003–2020; Traffic CV uses
2017–2020 and PM2.5 CV uses 2015–2019.

LST/precipitation use the fixed scene train/validation/test assignments.
PM2.5/Traffic train five pooled-embedding regression heads with train/val
cities re-partitioned at seed 42 and the original test cities fixed.
MAE encoders use `eump_reconstruction_best_checkpoint.pt`; contrastive
encoders use `eump_contrastive_best_checkpoint.pt`. Downstream models and
CV heads use validation-selected weights.

| Variant | Model/log group |
| --- | --- |
| Swin-UNet | `eump_full_architecture_v1_swin_unet` |
| ConvNeXt-UNet | `eump_full_architecture_v1_convnext_unet` |
| MoCo | `eump_full_contrastive_v2` |

Pretraining files are under `model/<group>/pretrain/`; fixed-task models use
`model/<group>/<task>/eump_downstream_model.pt`. CV records are under
`log/<group>/{traffic,pm25}_cv_mae_v3_seed42/`, with per-fold heads under
`model/<group>/{traffic,pm25}_cv/swin_embed_head_128_d05_fold<k>/`.

## Run

Run from the repository root:

```bash
GPU=0,1,2,3,4,5,6,7 NPROC=8 EVAL_GPU=0 EVAL_NPROC=1 LOG_PROGRESS_BAR=0 \
python experiments/eump_architecture_ablation/run.py data=eump_full/server_v2
```

The default stages are `train,eval,collect`. Training runs pretraining,
fixed-task heads, and scalar CV. Evaluation runs the fixed-task evaluations;
scalar CV evaluates its folds during the training stage. Select variants
with `--variants swin_unet,convnext_unet,contrastive_v2`.

To run only the first 31 MoCo pretraining epochs while preserving the
100-epoch learning-rate schedule:

```bash
GPU=0,1,2,3,4,5,6,7 NPROC=8 LOG_PROGRESS_BAR=0 \
bash script/training/run_eump_full_train.sh pretrain \
  launch=eump_full/contrastive_v2 data=eump_full/server_v2 \
  contrastive.epochs=31 contrastive.scheduler_epochs=100
```

Inspect commands without invoking the launchers or writing CV runtime files:

```bash
python experiments/eump_architecture_ablation/run.py --dry-run \
  data=eump_full/server_v2
```

Rebuild tables without training or inference:

```bash
python experiments/eump_architecture_ablation/run.py --stages collect
```

## Outputs and metric selection

The `outputs/` directory contains `summary.csv`, `summary_by_family.csv`,
`results.jsonl`, `comparable_metrics.csv`, and `validation_report.json`.
`--output-dir` selects another destination. The collector reads raw model
and CV evaluation records rather than a hand-maintained summary table.

`summary.csv` preserves all collected splits/scopes. The comparable table
selects test LST `city_mean` and other-task `global` metrics in physical
units. CV means/stds summarize fold metrics. A complete comparison contains
12 comparable rows: three variants by four tasks.

ConvNeXt fixed-task records are included only when the evaluated checkpoint
contains ConvNeXt encoder parameters. The validation report lists excluded
checkpoint mismatches. Generate compatible model/evaluation records before
using a complete architecture comparison; missing rows are never filled from
a differently parameterized model. The same check applies when the ablation
suite collects these references.

`outputs/`, logs, embedding caches, and model weights are excluded from Git.

The entry point defaults to `data=eump_full/server_v2` for training,
evaluation, and CV; an explicit CLI `data=` override selects another store.
