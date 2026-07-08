# EUMP MAE v3 — Inference-Time Input Channel Drop

Probes robustness of the trained `eump_full_mae_v3` model when 1–2 encoder
input channels are randomly zero-filled per crop sample at inference time.
Two arms are compared:

- **clean arm**: mainline `eump_full_mae_v3` weights scored under channel drop.
- **drop-trained arm**: downstream heads retrained with channel-drop augmentation
  on top of the same frozen `mae_v3` encoder (`train.py`), then scored under drop.

`lst`/`precip` run in-process eval (weights loaded read-only, metrics written to
`outputs/`). `pm25`/`traffic` shell out to `run_city_scalar_cv.py` under a
dedicated experiment group so CV artifacts never collide with the clean baseline.

Drop draws are deterministic per `(seed, scene_id, year, crop_box)`. The suite
runs `repeats` seeds (default 5); `summary.csv` reports a `single` row (first
seed) and an `aggregate` row (mean/std across all seeds) per task/split/scope.

## Commands

**Step 1 — train drop-augmented downstream heads** (skip to step 2 for clean-arm only):

```bash
GPU=0,1,2,3,4,5,6,7 NPROC=8 LOG_PROGRESS_BAR=0 \
python experiments/eump_channel_drop/train.py data=eump_full/server_v2
```

**Step 2 — run both eval arms**:

```bash
GPU=0,1,2,3,4,5,6,7 NPROC=8 EVAL_GPU=0 EVAL_NPROC=1 LOG_PROGRESS_BAR=0 \
bash experiments/eump_channel_drop/run_channel_drop_suite.sh
```

Or run each arm separately:

```bash
# drop-trained arm
GPU=0,1,2,3,4,5,6,7 NPROC=8 EVAL_GPU=0 EVAL_NPROC=1 LOG_PROGRESS_BAR=0 \
python experiments/eump_channel_drop/run.py data=eump_full/server_v2

# clean arm
GPU=0,1,2,3,4,5,6,7 NPROC=8 EVAL_GPU=0 EVAL_NPROC=1 LOG_PROGRESS_BAR=0 \
python experiments/eump_channel_drop/run.py --trained-group clean data=eump_full/server_v2
```

Dry run:

```bash
DRY_RUN=1 python experiments/eump_channel_drop/run.py --tasks lst,pm25
```

## Outputs

```text
experiments/eump_channel_drop/outputs/eump_mae_v3_channel_drop/
  <eval_group>/
    commands.log
    results.jsonl
    summary.csv        ← single + aggregate rows per task/split/scope/metric
    <task>_<variant>_<split>.yaml
    cv_detail/         ← per-variant CV fold metrics for pm25/traffic
```

Drop-trained model checkpoints: `model/eump_full_mae_v3_channel_drop_trained/<task>/`

CV outputs: `log/eump_full_mae_v3_channel_drop_trained/{traffic,pm25}_cv_drop_trained/`

## Config

Key knobs in `config.yaml`: `repeats`, `include_baseline`,
`channel_drop.{min_drop, max_drop, base_seed, droppable_channels}`,
`train.trained_group`, `cv.head.*`.
