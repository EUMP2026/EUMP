# EUMP, AEF, and Scalar Baseline Comparison

This experiment compares four prediction methods on LST, precipitation,
PM2.5, and Traffic using the restricted common study period in `config.yaml`.

| Method | Input and training |
| --- | --- |
| `ours` | Frozen MAE v3 encoder with a trained task head |
| `one_stage` | ImageNet-initialized Swin-L encoder and task head trained jointly, without MAE pretraining |
| `aef` | MLP using AEF features |
| `scalar_mlp` | MLP using processed scalar features |

The one-stage Swin model uses `scratch_backbone=false`; it is not a
random-initialized backbone experiment. Each method has a separate model
and CV output group specified in the YAML.

## Data and evaluation

Install `requirements.txt` and use a compatible PyTorch/CUDA environment.
`data=eump_full/server_v2` requires the `data/processed_v2/` scene store,
schema/scene manifests, Zarr arrays, preprocessing statistics, and the
`legacy_compatible_seed42.json` split manifest. AEF inference/training also
requires the feature store selected by the `eump_full/aef_mlp` pipelines.
The default AEF files are `data/AEF/AEF_<year>.csv`, keyed by `ID`.
Scalar MLP uses the feature cache built from the processed data.

The EUMP encoder is loaded from
`model/eump_full_mae_v3/pretrain/eump_reconstruction_best_checkpoint.pt`.
The encoder stays frozen for `ours`. Fixed-task models and scalar CV heads
use validation-selected weights; no test metrics select checkpoints.

LST and precipitation use the fixed train/validation/test split. PM2.5 and
Traffic use five folds over the training/validation cities with the original
test set fixed, seed 42. EUMP CV uses pooled-embedding heads; one-stage and
MLP CV models use the shared launcher. Year ranges are 2017–2020 for
LST/precipitation/Traffic and 2017–2019 for PM2.5.

All reported errors are in the physical target scale. The headline table
selects `city_mean` for a fixed-task model when available, and otherwise its
reported aggregate scope. Consequently precipitation uses `city_mean` for
AEF/Scalar MLP and `global` for the scalar EUMP/Swin predictor. LST uses
`city_mean`. CV uses `global`, with the mean and sample standard deviation
of the five fold metrics. This is not an ensemble prediction score.

Keep each row's sample count: required feature availability can give methods
different evaluation coverage. `summary_raw.csv` preserves all reported
scopes, including LST pixel metrics and MLP aggregate metrics.

## Run

Run from the repository root:

```bash
GPU=0,1,2,3,4,5,6,7 NPROC=8 EVAL_GPU=0 EVAL_NPROC=1 LOG_PROGRESS_BAR=0 \
python experiments/eump_aef_compare/run.py \
  --continue-run --skip-existing data=eump_full/server_v2
```

Use `--methods`, `--tasks`, and `--stages` for subsets. The `train` stage
includes both training and evaluation of scalar CV folds. The `eval` stage
runs fixed-task evaluations. Use `--output-dir` for separate subset reports;
collection writes the selected methods/tasks to that directory.
For one-stage CV, configure evaluation devices
for the unfrozen Swin-L memory requirement, for example:

```bash
GPU=0,1,2,3,4,5,6,7 NPROC=8 EVAL_GPU=0,1,2,3,4,5,6,7 EVAL_NPROC=8 LOG_PROGRESS_BAR=0 \
python experiments/eump_aef_compare/run.py \
  --methods one_stage --tasks traffic,pm25 --stages train \
  --continue-run data=eump_full/server_v2
```

Rebuild tables from existing evaluation records:

```bash
python experiments/eump_aef_compare/run.py \
  --stages collect --continue-run data=eump_full/server_v2
```

Inspect commands and configurations without starting workers:

```bash
python experiments/eump_aef_compare/run.py --dry-run \
  --output-dir /tmp/eump_aef_commands data=eump_full/server_v2
```

## Outputs

`outputs/eump_aef_compare_mae_v3_aef/` contains:

- `summary.csv`: one test row per method/task, 16 rows for a complete run.
- `summary_raw.csv` and `results.jsonl`: all collected test metric scopes.
- `manifest.json`, `runtime_config.yaml`, and `configs/`: selected methods, year ranges, and resolved configurations.
- `commands.log`: executed or inspected worker commands.

Fixed-task checkpoints are under `model/<method_group>/<task>/`. Scalar CV
records are under
`log/<cv_experiment_group>/{traffic,pm25}_cv_mae_v3_seed42/`; model locations
and their group identifiers are recorded in the raw CV results. The YAML
lists the EUMP, one-stage, AEF, and Scalar MLP groups separately.

Generated outputs, feature caches, logs, and model weights are excluded from
Git and distributed separately from the source code.
