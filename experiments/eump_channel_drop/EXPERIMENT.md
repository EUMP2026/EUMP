# EUMP Input Channel-Drop Robustness

This experiment measures prediction robustness when one or two processed
encoder input channels are zero-filled per crop. Downstream heads are trained
with channel-drop augmentation on the frozen MAE v3 encoder, then evaluated
under repeated inference-time channel drop for LST, precipitation, PM2.5,
and Traffic.

Zero filling occurs in processed model space. It does not imply that the
physical variable takes value zero. All 11 input channels, including the
city mask, are eligible by default. Draws are deterministic for the seed,
scene ID, year, and crop box. `droppable_channels` restricts eligible channels.

## Models and evaluation

All tasks require
`model/eump_full_mae_v3/pretrain/eump_reconstruction_best_checkpoint.pt`.
LST/precipitation load validation-selected downstream models at
`model/eump_full_mae_v3_channel_drop_trained/<task>/eump_downstream_model.pt`.

For PM2.5/Traffic, launcher evaluation loads the per-fold downstream heads
under `model/eump_full_mae_v3_channel_drop_trained/{pm25,traffic}_cv/`
and applies input corruption before encoder inference. Evaluation scores
existing checkpoints without retraining heads. `channel_drop.train.trained_group`
selects the downstream model group for both training and evaluation.

`train.py` trains the drop-augmented heads with the encoder frozen. Scalar
CV uses the launcher so corruption can be applied before encoder inference.
The training seed is 12345. Validation and CV evaluation use the configured
channel-drop pipeline as well.

Use the repository dependencies and the `data=eump_full/server_v2` processed
scene store, schema/scene manifests, Zarr data, preprocessing statistics,
and `legacy_compatible_seed42.json` split manifest. LST/precipitation cover
2003–2020; PM2.5 CV covers 2015–2019; Traffic CV covers 2017–2020.
Scalar CV keeps test cities fixed and uses five train/val folds, seed 42.

## Repeated corruption metrics

The default run uses five corruption seeds, `12345 + 101 * repeat_index`.
Metrics are restored to physical target scale. LST reports `global_pixels`
and `city_mean`; scalar tasks report `global`.

For CV tasks, each corruption repeat first averages the five fold metrics.
The `single` row reports the first corruption repeat, including all its
folds. The `aggregate` row reports the mean and sample standard deviation
across five corruption repeats, with `n_runs=5`. The standard deviation
therefore measures sensitivity to corruption draws after fold averaging;
it is not a standard deviation over 25 independent training runs.

`include_baseline=false` is the default. Enabling it adds uncorrupted
reference records to `results.jsonl`; the corruption summary excludes them.
CV raw records also include training/validation splits, whereas the fixed
LST/precipitation evaluation defaults to the test split. Use test rows for
the robustness comparison. Missing source metrics remain blank.

## Run

Run from the repository root. Train drop-augmented heads:

```bash
GPU=0,1,2,3,4,5,6,7 NPROC=8 LOG_PROGRESS_BAR=0 \
python experiments/eump_channel_drop/train.py data=eump_full/server_v2
```

Evaluate the trained heads:

```bash
GPU=0,1,2,3,4,5,6,7 NPROC=8 EVAL_GPU=0 EVAL_NPROC=1 LOG_PROGRESS_BAR=0 \
bash experiments/eump_channel_drop/run_channel_drop_suite.sh
```

Or invoke the Python evaluation entry point directly:

```bash
python experiments/eump_channel_drop/run.py data=eump_full/server_v2
```

Select tasks with `--tasks lst,precip,pm25,traffic`; the shell suite accepts
`TASKS=lst,precip`. GPU count must match `NPROC`. Fixed-task evaluation uses
`EVAL_GPU`, while scalar CV launcher evaluation uses `GPU/NPROC`.

Rebuild tables from saved evaluation records without inference:

```bash
python experiments/eump_channel_drop/run.py --collect-only
```

Inspect configurations/commands in a separate output directory:

```bash
python experiments/eump_channel_drop/run.py --dry-run \
  --output-dir /tmp/eump_channel_drop_commands data=eump_full/server_v2
python experiments/eump_channel_drop/train.py --dry-run data=eump_full/server_v2
```

## Outputs

Evaluation writes under
`outputs/eump_mae_v3_channel_drop/eump_full_mae_v3_channel_drop_trained/`:

- `results.jsonl`: task, variant, seed, fold, split, scope, and metric records.
- `summary.csv`: single-repeat and aggregate metrics with explicit `n_runs`.
- `<task>_<variant>_<split>_summary.json`: raw fixed-task evaluation summaries.
- `cv_detail/`: per-variant fold metrics and raw CV result records.
- Resolved YAML files and `commands.log`.

Drop-augmented CV training configurations are under
`outputs/eump_mae_v3_channel_drop/train/`. Model files are under
`model/eump_full_mae_v3_channel_drop_trained/`, and training CV records under
`log/eump_full_mae_v3_channel_drop_trained/{traffic,pm25}_cv_drop_trained/`.
Generated outputs, logs, embedding caches, and weights are excluded from Git.
