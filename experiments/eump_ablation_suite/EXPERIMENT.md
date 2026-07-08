# EUMP Ablation Suite

Reproducible EUMP Swin MAE pretraining ablations from the `mae_v3` suite.
`run.py` defaults to `experiments/eump_ablation_suite/config_v2.yaml`.

The suite composes existing runnable EUMP configs, applies experiment-specific
overrides, writes resolved configs under this directory, and launches workers
through the `run_training` / `run_evaluation` framework. Training workers use
`GPU/NPROC`; eval workers use `EVAL_GPU/EVAL_NPROC` (default: 1 GPU).

## Ablation Families

- **`center1_noaux`**: center-crop pretraining (`center_ratio=1.0`) without aux
  losses, varying MAE mask ratio over `0`, `0.2`, `0.4`, `0.6`, `0.8`.
- **`comparison`**: baseline anchors. `noaux`, `swin_t`, and `scratch_init` are
  trained in-suite. `swin_t` uses Swin-T; `scratch_init` uses Swin-L without
  ImageNet weights. `mae_v3` is **reference-only** — collected from the existing
  `eump_full_mae_v3` group, never retrained here.
- **`input_groups`**: progressively increases encoder input channels without aux
  losses. G1–G5:
  - G1: `NDVI`, `POP`
  - G2: G1 + `NL`
  - G3: G2 + `water`, `DEM`, `city_mask`
  - G4: G3 + `BH`, `BF`
  - G5: full default inputs (all sources, `city_mask`)
- **`one_stage`**: supervised one-stage baseline. Skips MAE pretraining, starts
  from base Swin-L weights, unfreezes encoder. `lst` and `precip` use the normal
  downstream runtime; `pm25` and `traffic` use fixed-test city-scalar CV.
- **`architecture`** (reference-only): re-collects results from
  `experiments/eump_architecture_ablation/` via `reference_group`. Run that
  experiment first; empty group → empty summary rows, not an error.

## Commands

Default full run:

```bash
GPU=0,1,2,3,4,5,6,7 NPROC=8 EVAL_GPU=0 EVAL_NPROC=1 LOG_PROGRESS_BAR=0 \
python experiments/eump_ablation_suite/run.py \
  --continue-run --skip-existing data=eump_full/server_v2
```

Dry run:

```bash
DRY_RUN=1 GPU=0 NPROC=1 LOG_PROGRESS_BAR=0 \
python experiments/eump_ablation_suite/run.py data=eump_full/server_v2
```

Collect existing metrics only (resolves all `reference_group` rows):

```bash
python experiments/eump_ablation_suite/run.py \
  --stages collect --continue-run data=eump_full/server_v2
```

Smoke test:

```bash
GPU=0 NPROC=1 LOG_PROGRESS_BAR=0 \
python experiments/eump_ablation_suite/run.py \
  --smoke --families center1_noaux --variants center1_mask60_noaux \
  --tasks lst,precip,traffic data=eump_full/server_v2
```

## Outputs

```text
experiments/eump_ablation_suite/outputs/<run_name>/
  results.jsonl
  summary.csv
  summary_by_family.csv
  configs/<variant>/*.yaml
```

Model checkpoints: `model/<experiment.group>/<experiment.name>/`

City-scalar CV outputs:

```text
log/<experiment.group>/traffic_cv_seed42/
log/<experiment.group>/pm25_cv_seed42/
model/<experiment.group>/traffic_cv/traffic_cv_fold<k>/
model/<experiment.group>/PM25_mean_cv/PM25_mean_cv_fold<k>/
```

The `one_stage` family saves downstream checkpoints under
`model/eump_ablation_v2_one_stage_swin_l_unfrozen/<task>/` and CV outputs under
`log/eump_ablation_v2_one_stage_swin_l_unfrozen/{traffic,pm25}_cv_seed42/`.
