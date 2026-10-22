# EUMP Pretraining and Input Ablations

This experiment evaluates how MAE masking, auxiliary supervision, encoder
size/initialization, input channels, and supervised one-stage training affect
LST, precipitation, PM2.5, and Traffic prediction. `run.py` uses `config_v2.yaml`.
The base pipeline is `eump_full/mae_v3`.

## Experiment families

| Family | Variants | Training protocol |
| --- | --- | --- |
| `center1_noaux` | Mask ratios 0, 0.2, 0.4, 0.6, 0.8 | Center crops, input reconstruction without auxiliary targets |
| `comparison` | `noaux`, `swin_t`, `scratch_init` | No auxiliary targets, Swin-T, or random-initialized Swin-L MAE, respectively |
| `comparison` | `mae_v3` | Collect the independently trained mainline MAE v3 reference |
| `input_groups` | `group_1` through `group_5` | Increasing input coverage, without auxiliary targets |
| `one_stage` | `swin_l_unfrozen` | Train encoder and downstream head jointly from ImageNet-initialized Swin-L; skip MAE training |
| `architecture` | Swin-UNet, ConvNeXt-UNet, MoCo | Collect the [architecture experiment](../eump_architecture_ablation/EXPERIMENT.md) |

The architecture and contrastive training configurations are documented in
the linked experiment.

Input groups are:

| Group | Encoder input sources | City-mask input |
| --- | --- | --- |
| G1 | NDVI, POP | No |
| G2 | NDVI, NL, POP | No |
| G3 | G2 + water, DEM | Yes |
| G4 | G3 + BH, BF | Yes |
| G5 | G4 + Ta, precip_bg, wind | Yes |

Except for `scratch_init`, MAE variants use the configured pretrained Swin
initialization. `scratch_init` still performs MAE pretraining. The one-stage
variant skips MAE but uses ImageNet initialization. Reference variants are
collected without training or evaluation jobs in this suite. Explicit
`experiment_group` values in the YAML determine model storage independently
of the family names used in tables.

## Data, models, and metrics

Install the repository dependencies from `requirements.txt`. The default
`data=eump_full/server_v2` configuration requires the `data/processed_v2/`
scene store, its schema/scene manifests, Zarr arrays,
`manifest/stats/preprocessing_v1.json`, and
`manifest/splits/legacy_compatible_seed42.json`.

LST and precipitation use the fixed scene train/validation/test assignments.
PM2.5 and Traffic use five folds over the training/validation cities, with
the original test cities fixed and fold seed 42. MAE variants use a frozen
encoder and pooled-embedding scalar heads for these CV tasks; the one-stage
variant trains the encoder and head jointly within each fold.

MAE encoders use `eump_reconstruction_best_checkpoint.pt`; fixed-task
predictions use validation-selected `eump_downstream_model.pt` files. Scalar
CV heads use their configured validation selection. PM2.5 CV covers 2015–2019
and Traffic CV 2017–2020; the MAE and fixed-task pipelines cover 2003–2020.
Metrics are restored to physical units. LST includes `global_pixels` and
`city_mean`; scalar tasks report `global`. CV rows give fold means and sample
standard deviations, not metrics from an ensemble of predictions.

The mainline reference requires
`model/eump_full_mae_v3/pretrain/eump_reconstruction_best_checkpoint.pt`, its
fixed-task evaluation records, and its CV records. Architecture references
require the corresponding model and log groups documented in the linked
experiment. Raw CV paths are selected by their explicitly configured run names.

## Run

Run from the repository root:

```bash
GPU=0,1,2,3,4,5,6,7 NPROC=8 EVAL_GPU=0 EVAL_NPROC=1 LOG_PROGRESS_BAR=0 \
python experiments/eump_ablation_suite/run.py \
  --continue-run --skip-existing data=eump_full/server_v2
```

Select a subset with `--families`, `--variants`, `--tasks`, or `--eval-splits`.
Stages are `train`, `eval`, and `collect`. Scalar CV training also evaluates
its folds; the `eval` stage handles the fixed-task evaluation jobs.

Collect existing metrics without training or inference:

```bash
python experiments/eump_ablation_suite/run.py \
  --stages collect --continue-run data=eump_full/server_v2
```

Inspect resolved configurations and commands in a separate directory:

```bash
python experiments/eump_ablation_suite/run.py --dry-run \
  --output-dir /tmp/eump_ablation_commands data=eump_full/server_v2
```

## Outputs

The default result directory is `outputs/eump_ablation_v2_full/`:

- `summary.csv` and `results.jsonl`: per-variant, task, split, and scope metrics.
- `summary_by_family.csv`: descriptive means/stds across variants within a family; these are not repeat-run uncertainty estimates.
- `variant_manifest.json`, `runtime_config.yaml`, and `configs/`: selected variants and resolved execution configurations.
- `validation_report.json`: collected row count and excluded checkpoint mismatches. ConvNeXt fixed-task records require ConvNeXt encoder parameters.

Models are saved under `model/<experiment_group>/<task>/`. CV results are
under `log/<experiment_group>/traffic_cv_seed42/` and `pm25_cv_seed42/`;
architecture references use the explicitly configured `*_cv_mae_v3_seed42/`
run names. Scalar embedding-head checkpoints use
`model/<group>/<task>_cv/<head_name>_fold<k>/`; launcher CV variants record
their exact checkpoint paths in the raw results.

Optional figures use the same test-set rows, with LST `city_mean`:

```bash
python experiments/eump_ablation_suite/visualize_v2.py
```

Generated configurations, tables, figures, and all `outputs/` content are
excluded from Git. Data and model weights must be obtained separately.
