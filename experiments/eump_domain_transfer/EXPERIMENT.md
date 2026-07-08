# EUMP Domain Transfer

Tests cross-domain transferability of downstream models on all four tasks.
Domain labels come from `data/csv_data/city_classes.csv`. Three models are
compared: `swin` (frozen `mae_v3` encoder + task head), `scalar_mlp`, `aef_mlp`.

The experiment uses its own train/test split — all source-domain cities for
training, all target-domain cities for testing. The original `train/val/test`
manifest is not used. There is no validation set and no best-checkpoint
selection; all results come from the final training checkpoint.

Swin downstream encoder reuses the global pretrain checkpoint:
`model/eump_full_mae_v3/pretrain/eump_reconstruction_best_checkpoint.pt`.
Per-scenario Swin pretraining is supported (`pretrain.enabled=true`) but
disabled by default — only the downstream task head is trained.

All models are restricted to `downstream_year_range: [2017, 2020]`. PM2.5
naturally falls to `2017–2019` because the target only covers up to 2019.

## Transfer Scenarios

| Scenario | Column | Train domain | Test domain |
| --- | --- | --- | --- |
| `climate_acd_to_b` | `climate_group` | `A+C+D` | `B` |
| `gdp_nonpoor_to_poor` | `gdp_class` | `Rich+Medium` | `Poor` |
| `income_hic_lic_lmic_to_umic` | `INCM_CMI` | `HIC+LIC+LMIC` | `UMIC` |
| `climate_abc_to_d` | `climate_group` | `A+B+C` | `D` |
| `climate_abd_to_c` | `climate_group` | `A+B+D` | `C` |
| `climate_bcd_to_a` | `climate_group` | `B+C+D` | `A` |

Cities with blank or unlisted class values are excluded from the relevant scenario.

## Commands

Full run (all 6 scenarios, all models):

```bash
GPU=0,1,2,3,4,5,6,7 NPROC=8 EVAL_GPU=0 EVAL_NPROC=1 LOG_PROGRESS_BAR=0 \
python experiments/eump_domain_transfer/run.py data=eump_full/server_v2
```

Priority scenarios only:

```bash
GPU=0,1,2,3,4,5,6,7 NPROC=8 EVAL_GPU=0 EVAL_NPROC=1 LOG_PROGRESS_BAR=0 \
python experiments/eump_domain_transfer/run.py \
  --scenarios climate_acd_to_b,gdp_nonpoor_to_poor,income_hic_lic_lmic_to_umic \
  data=eump_full/server_v2
```

Re-run eval only (reuse existing final checkpoints):

```bash
GPU=0,1,2,3,4,5,6,7 NPROC=8 EVAL_GPU=0 EVAL_NPROC=1 LOG_PROGRESS_BAR=0 \
python experiments/eump_domain_transfer/run.py \
  --eval-only --continue-run data=eump_full/server_v2
```

Dry run:

```bash
DRY_RUN=1 GPU=0,1,2,3,4,5,6,7 NPROC=8 EVAL_GPU=0 EVAL_NPROC=1 LOG_PROGRESS_BAR=0 \
python experiments/eump_domain_transfer/run.py \
  --scenarios climate_acd_to_b data=eump_full/server_v2
```

Override training hyperparameters:

```bash
# All tasks
python experiments/eump_domain_transfer/run.py ... training.epochs=50

# Single task
python experiments/eump_domain_transfer/run.py ... training.task_epochs.lst=50

# Single model+task
python experiments/eump_domain_transfer/run.py ... training.model_task_epochs.scalar_mlp.precip=120
```

## Outputs

```text
experiments/eump_domain_transfer/outputs/<run_name>/
  commands.log
  runtime_config.yaml
  domain_manifest.json
  results.jsonl
  summary.csv
  model/<scenario>/<task>/eump_downstream_final_checkpoint.pt
  model/scalar_mlp/<scenario>/<task>/eump_scalar_mlp_final_checkpoint.pt
  model/aef_mlp/<scenario>/<task>/eump_scalar_mlp_final_checkpoint.pt
```

Metrics: LST reports `global_pixels` and `city_mean`; precip/PM2.5/Traffic
report `global`. `summary.csv` includes model, scenario, source/target domain,
task, scope, RMSE, MAE, R², count, city count, and checkpoint path.

## Visualization

After all models are done:

```bash
python experiments/eump_domain_transfer/visualize_transferability.py \
  --run-dir experiments/eump_domain_transfer/outputs/<run_name>
```

Outputs land in `<run_name>/figures/`: RMSE/R² bar charts per target domain,
RMSE-ratio and R²-drop heatmaps, rank summary, and `transferability_report.md`.
