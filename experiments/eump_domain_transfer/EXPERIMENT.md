# EUMP Domain Transfer

This experiment evaluates downstream transfer across city domains with two
MAE encoder protocols and four tasks: LST, precipitation, PM2.5, and Traffic.

## Protocols

| Protocol | Configuration | Source-domain MAE stage | Downstream Swin encoder |
| --- | --- | --- | --- |
| Global MAE encoder | `config.yaml` | `pretrain.enabled=false` | Global MAE v3 validation-best checkpoint |
| Source-domain MAE encoder | `config_source_pretrain.yaml` | `pretrain.enabled=true` | Final source-domain MAE checkpoint |

`pretrain.enabled=false` skips scenario-specific MAE training and loads
`model/eump_full_mae_v3/pretrain/eump_reconstruction_best_checkpoint.pt`.
`pretrain.enabled=true` starts from the standard pretrained `swin_L`
initialization and trains the MAE using source-domain cities only. Both
protocols freeze the encoder during downstream head training.

Source-domain MAE training excludes target cities from optimizer updates and
checkpoint selection. The global MAE checkpoint uses the full pretraining
dataset and therefore does not impose the same source-only restriction.

## Scenarios and evaluation

Domain labels come from `data/csv_data/city_classes.csv`.

| Scenario | Class column | Source | Target | Global MAE | Source-domain MAE |
| --- | --- | --- | --- | --- | --- |
| `climate_acd_to_b` | `climate_group` | A+C+D | B | Yes | — |
| `gdp_nonpoor_to_poor` | `gdp_class` | Rich+Medium | Poor | Yes | Yes |
| `income_hic_lic_lmic_to_umic` | `INCM_CMI` | HIC+LIC+LMIC | UMIC | Yes | — |
| `climate_abc_to_d` | `climate_group` | A+B+C | D | Yes | — |
| `climate_abd_to_c` | `climate_group` | A+B+D | C | Yes | — |
| `climate_bcd_to_a` | `climate_group` | B+C+D | A / Tropical | Yes | Yes |

Cities with blank or unlisted class values are excluded from that scenario.

The global-MAE configuration trains `swin`, `scalar_mlp`, and `aef_mlp`.
The source-domain-MAE configuration trains `swin`; model comparisons use the
matching MLP results from the global-MAE run.

All source-domain cities train the downstream predictor and all target-domain
cities evaluate transfer. Domain membership defines the training/evaluation
split instead of the scene store's original train/validation/test assignments.
There is no validation set or target-based checkpoint selection;
downstream evaluation uses the final model. Source-domain metrics evaluate
the training domain, so target/source degradation includes both domain shift
and train-to-test generalization.

Downstream inputs are restricted to 2017–2020; usable PM2.5 observations end
in 2019. Source-domain MAE training and reconstruction evaluation use available
years 2003–2020. Every result row records its year range and sample coverage.
MLP and Swin coverage can differ because their required inputs differ.

## Dependencies

Run commands from the repository root with the dependencies in
`requirements.txt` and a compatible PyTorch/CUDA environment. Required data
and model files are:

- The `data/processed_v2/` scene store, including scene/schema manifests, Zarr data, `manifest/stats/preprocessing_v1.json`, and `manifest/splits/legacy_compatible_seed42.json`.
- `data/csv_data/city_classes.csv`, containing city IDs and the class columns listed above.
- The global MAE v3 checkpoint for `config.yaml`, or the standard `swin_L` initialization weights for source-domain MAE training.
- The input feature data required by Scalar MLP and AEF MLP when those baselines are selected.

Data and model files are distributed separately from source code. Their paths
are set by the data/pipeline configurations and the two experiment YAML files.
Both YAML files default to `data=eump_full/server_v2`; an explicit CLI `data=`
override selects another processed store.

## Training and evaluation

Global MAE encoder:

```bash
GPU=0,1,2,3,4,5,6,7 NPROC=8 EVAL_GPU=0 EVAL_NPROC=1 LOG_PROGRESS_BAR=0 \
python experiments/eump_domain_transfer/run.py \
  --config experiments/eump_domain_transfer/config.yaml \
  data=eump_full/server_v2
```

Source-domain MAE encoder:

```bash
GPU=0,1,2,3,4,5,6,7 NPROC=8 EVAL_GPU=0 EVAL_NPROC=1 LOG_PROGRESS_BAR=0 \
python experiments/eump_domain_transfer/run.py \
  --config experiments/eump_domain_transfer/config_source_pretrain.yaml \
  data=eump_full/server_v2
```

Source-domain MAE training runs for 200 epochs with mask ratio 0.6. Downstream
training runs for 100 epochs for LST/PM2.5 and 50 for precipitation/Traffic;
the MLP precipitation heads use 100 epochs. The YAML files specify the other
training parameters.

Add `--continue-run --skip-pretrain` to reuse completed source-domain MAE
checkpoints and continue downstream training. `--skip-pretrain` keeps the
source-domain encoder protocol. Add `--eval-only --continue-run` to evaluate
existing final downstream models with either configuration.

Use `--scenarios`, `--models`, and `--tasks` to select subsets. Training-duration
overrides are `training.epochs`, `training.task_epochs.<task>`, and
`training.model_task_epochs.<model>.<task>`. `DRY_RUN=1` writes configurations
and commands without starting workers; use a separate `output_root` for this
inspection.

## Checkpoints and run outputs

| Configuration | Run directory under `outputs/` | Scenarios | Models | Summary rows |
| --- | --- | ---: | --- | ---: |
| `config.yaml` | `eump_domain_transfer_mae_v3_city_classes_multi_transfer` | 6 | Swin, Scalar MLP, AEF MLP | 156 |
| `config_source_pretrain.yaml` | `eump_domain_transfer_mae_v3_source_pretrain_climate_tropic_gdp_poor` | 2 | Swin | 20 |

Each run writes `summary.csv`, `results.jsonl`, `domain_manifest.json`,
`runtime_config.yaml`, model files, and training logs. Swin outputs use:

```text
model/<scenario>/<task>/eump_downstream_final_model.pt
model/<scenario>/<task>/eump_downstream_final_checkpoint.pt
```

MLP outputs use
`model/<model>/<scenario>/<task>/eump_scalar_mlp_final_{model,checkpoint}.pt`.
The source-domain MAE protocol additionally writes:

```text
model/<scenario>/pretrain/eump_reconstruction_final_model.pt
model/<scenario>/pretrain/eump_reconstruction_final_checkpoint.pt
model/<scenario>/pretrain/pretrain_metrics.json
pretrain_summary.csv
```

Downstream encoder initialization loads the final reconstruction checkpoint;
downstream evaluation loads the final downstream model. The reconstruction
metrics use the processed-space training objective, including continuous,
hurdle, binary, and categorical targets.

## Comparable tables

Export both protocols and the supplementary comparisons from the raw metric
summaries:

```bash
python experiments/eump_domain_transfer/export_results.py
```

Output paths below are relative to `outputs/delivery/`:

| File | Rows | Contents |
| --- | ---: | --- |
| `without_source_pretrain/summary.csv` | 156 | Complete global-MAE downstream metrics |
| `without_source_pretrain/comparable_metrics.csv` | 144 | Six scenarios, three models, four tasks, source and target |
| `with_source_pretrain/summary.csv` | 20 | Complete source-domain-MAE downstream metrics |
| `with_source_pretrain/comparable_metrics.csv` | 16 | Two scenarios, Swin, four tasks, source and target |
| `comparisons/comparable_metrics.csv` | 32 | Source/global MAE Swin and MLP target comparisons, with references and metric deltas |
| `comparisons/pretrain_protocol_comparison.csv` | 10 | Source/global MAE Swin comparison, including both LST scopes |
| `comparisons/pretrain_reconstruction_transfer.csv` | 12 | Total, reconstruction-group, and auxiliary-group losses with target/source ratios |

For cross-model LST comparison, Swin uses `city_mean` and the city-level MLP
predictors use `global`. Other tasks use `global`. Full `summary.csv` files
also preserve LST `global_pixels`. Physical downstream metrics and
processed-space reconstruction losses are exported separately.

Each protocol includes target/source degradation tables and a manifest;
the global-MAE protocol also includes model rank tables. `export_manifest.json`
records summary SHA256 values, verified checkpoint counts, and exported row
counts. The exporter checks that source/global Swin target sample counts,
city counts, and row counts agree. MLP comparison rows keep their individual
coverage fields.

`outputs/`, including generated tables, logs, and checkpoints, is excluded
from Git. To regenerate figures as well as tables, run:

```bash
python experiments/eump_domain_transfer/visualize_transferability.py \
  --run-dir experiments/eump_domain_transfer/outputs/eump_domain_transfer_mae_v3_city_classes_multi_transfer
```

The figure script uses the same metric-selection rules as the table exporter.
