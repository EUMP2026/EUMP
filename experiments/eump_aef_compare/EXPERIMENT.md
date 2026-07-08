# EUMP AEF Compare

Compares four methods on all four downstream tasks:

- `ours`: frozen `mae_v3` encoder + task head. Reuses `model/eump_full_mae_v3/pretrain/eump_reconstruction_best_checkpoint.pt`.
- `one_stage`: unfrozen Swin-L from ImageNet weights (`mae_v3_one_stage` pipeline).
- `aef`: AEF MLP baseline (`eump_full/aef_mlp` pipeline).
- `scalar_mlp`: processed scalar MLP baseline (`eump_full/scalar_mlp` pipeline).

`lst` and `precip` use fixed train/val/test split. `traffic` and `pm25` use 5-fold city-scalar CV (seed 42). Year ranges: `2017–2020` for lst/precip/traffic; `2017–2019` for pm25.

## Commands

Full run:

```bash
GPU=0,1,2,3,4,5,6,7 NPROC=8 EVAL_GPU=0 EVAL_NPROC=1 LOG_PROGRESS_BAR=0 \
python experiments/eump_aef_compare/run.py data=eump_full/server_v2
```

Single method or task subset:

```bash
GPU=0,1,2,3,4,5,6,7 NPROC=8 EVAL_GPU=0 EVAL_NPROC=1 LOG_PROGRESS_BAR=0 \
python experiments/eump_aef_compare/run.py \
  --methods aef --tasks lst,precip --stages train,eval \
  --continue-run data=eump_full/server_v2
```

For one-stage `traffic` or `pm25` CV, use `EVAL_NPROC=8` to avoid OOM with the unfrozen Swin-L:

```bash
GPU=0,1,2,3,4,5,6,7 NPROC=8 EVAL_GPU=0,1,2,3,4,5,6,7 EVAL_NPROC=8 LOG_PROGRESS_BAR=0 \
python experiments/eump_aef_compare/run.py \
  --methods one_stage --tasks traffic,pm25 --stages train \
  --continue-run data=eump_full/server_v2
```

Collect existing results:

```bash
python experiments/eump_aef_compare/run.py --stages collect --continue-run data=eump_full/server_v2
```

Dry run:

```bash
DRY_RUN=1 GPU=0 NPROC=1 EVAL_GPU=0 EVAL_NPROC=1 LOG_PROGRESS_BAR=0 \
python experiments/eump_aef_compare/run.py data=eump_full/server_v2
```

## Outputs

```text
experiments/eump_aef_compare/outputs/<run_name>/
  results.jsonl
  summary_raw.csv
  summary.csv          ← headline rows: city_mean for lst, global for precip/traffic/pm25
  configs/{ours,one_stage,aef,scalar_mlp,cv}/*.yaml
```

Fixed-test model checkpoints: `model/<experiment.group>/<task>/`

CV outputs follow the same layout as the ablation suite:

```text
log/<cv_experiment_group>/traffic_cv_mae_v3_seed42/
log/<cv_experiment_group>/pm25_cv_mae_v3_seed42/
model/<cv_experiment_group>/traffic_cv/traffic_cv_fold<k>/
model/<cv_experiment_group>/PM25_mean_cv/PM25_mean_cv_fold<k>/
```
