# Urban Planning Optimization Experiment

Optimizes urban input features (NDVI, population density) to achieve target
reductions in LST and traffic. The pretrained encoder and downstream heads are
frozen; selected input channels are treated as learnable parameters.

## Prerequisites

Run commands from the repository root with `requirements.txt` installed and
a compatible PyTorch/CUDA environment. Use `data=eump_full/server_v2` with the
`data/processed_v2/` scene store, including scene/schema manifests, Zarr data,
`manifest/stats/preprocessing_v1.json`, and
`manifest/splits/legacy_compatible_seed42.json`.

The following model artifacts must exist before running this experiment:

- `model/eump_full_mae_v3/pretrain/eump_reconstruction_best_checkpoint.pt`
- `model/eump_full_mae_v3/lst/eump_downstream_model.pt`
- A traffic CV fold checkpoint. Since traffic uses K-fold CV, multiple fold
  models are available; one fold is selected for planning. The default is fold 2:
  `model/eump_full_mae_v3/traffic_cv/swin_embed_head_128_d05_fold2/swin_embedding_scalar_head.pt`
  Change `traffic_cv.checkpoint` in `config.yaml` to use a different fold.

Both the MAE encoder and the LST downstream model above contain
validation-best weights. The LST `eump_downstream_model.pt` file is the
configured best-model save, rather than the resumable training-state
checkpoint. The shared model setup loads the traffic head in every mode,
including `cooling_only`, so all three files are required.

The joint-gradient policy uses one backward pass on `LST_loss + Traffic_loss`
so both objectives update NDVI and population. The execution modes are:

- `joint`: optimize NDVI and population with both objectives.
- `cooling_only`: optimize NDVI for LST reduction.
- `both`: run joint and cooling-only from the original input for Fig.4.

Each loss step applies a manual sign-gradient update followed by an AdamW
raw-gradient step (default). Soft planning terms encourage one-sided NDVI
changes and population-total conservation.

See `config.yaml` for the full parameter reference.

## Key config parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mode` | `joint` | `joint`, `cooling_only`, or `both` |
| `optimization.epochs` | `1000` | Optimization iterations |
| `optimization.lr` | `0.001` | Learning rate |
| `targets.lst_delta_base` | `1.0` | Base LST cooling target (K) |
| `targets.lst_delta_pop_weighting` | `linear` | Population weighting: `linear` or `log10p` |
| `targets.lst_target_region` | `city` | LST pixel loss region: `all` or `city` |
| `targets.lst_delta_normalize_city_mean` | `true` | Normalize so city-mean cooling = `lst_delta_base` |
| `targets.traffic_delta` | `0.2` | Traffic reduction target |
| `targets.traffic_delta_mode` | `percent` | `absolute` or `percent` |

## Commands

Joint-gradient run (5 cities):

```bash
python experiments/planning_exp/run.py \
  --mode joint --gradient-mode joint \
  --cities 782,339,216,475,261 \
  data=eump_full/server_v2
```

Both modes (required for Fig4):

```bash
python experiments/planning_exp/run.py --mode both --cities 782,339,216,475,261 \
  --gradient-mode joint \
  data=eump_full/server_v2
```

Large-city multi-GPU statistics:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python experiments/planning_exp/run_urbanfrac_city_stats.py \
  --num-workers 0 --gpus 0,1,2,3,4,5,6,7 --mode both \
  --gradient-mode joint
```

Robustness sweep over other hyperparameters:

```bash
bash experiments/planning_exp/run_robustness_sweep.sh
python experiments/planning_exp/summarize_robustness_sweep.py
```

Forward inference counterfactuals (LST decomposition):

```bash
python experiments/planning_exp/run_forward_inference.py \
  --planning-data experiments/planning_exp/outputs/joint_gradient_v2_5cities_current/planning_data.pkl \
  data=eump_full/server_v2
```

Fig4 visualization:

```bash
python main_figs_v2/fig4/prepare_data.py
python main_figs_v2/fig4/fig4.py \
  --input-dir main_figs_v2/fig4 \
  --output main_figs_v2/fig4/fig4.png
```

`prepare_data.py` combines the five-city joint and cooling-only packages after
checking their original inputs, masks, and preprocessing statistics. An existing
`--mode both` package can be passed directly to the renderer. See the
[Fig.4 document](../../main_figs_v2/fig4/README.md) for inputs and metric tables.

Export the three 292-city target-sweep packages in physical units:

```bash
python experiments/planning_exp/export_physical_results.py
```

Summarize and plot the five-city joint planning responses:

```bash
python experiments/planning_exp/plot_distributions.py
```

Select one or more other packages with repeated `--version NAME=PATH`:

```bash
python experiments/planning_exp/plot_distributions.py \
  --version traffic10pct=experiments/planning_exp/outputs/joint_gradient_v2_292cities_traffic10pct \
  --version traffic20pct=experiments/planning_exp/outputs/joint_gradient_v2_292cities_traffic20pct \
  --version traffic30pct=experiments/planning_exp/outputs/joint_gradient_v2_292cities_traffic30pct \
  --output-dir experiments/planning_exp/outputs/target_sweep_distributions
```

## Outputs

```text
outputs/<run_name>/
  planning_data.pkl       ← canonical output; consumed by Fig4 and statistics scripts
  summary.json
  city_<id>.npz           ← optional per-city arrays (disable with --no-city-npz)
  figures/
```

The large-city statistics script writes one merged v2 package per target under
`outputs/joint_gradient_v2_292cities_traffic<target>/planning_data.pkl`.

`export_physical_results.py` writes to `outputs/physical_results/`, with one
`target_<target>/` directory per 292-city run. Each contains physical NDVI,
POP, and LST arrays and their deltas in `cities/city_<id>.npz`, masks,
`city_summary.csv`, `manifest.csv`, and `summary.json`. The root contains
`planning_targets_summary.csv` and the MAE v3 physical-conversion parameters
in `conversion_spec.json`. `--output-dir` selects another destination. City
metrics distinguish the valid urban mask, all valid pixels, and the full
finite raster; field names identify each scope.

`plot_distributions.py` writes `planning_city_metrics.csv`,
`planning_metrics_summary.csv`, `summary.json`, and six distribution plots
under `outputs/planning_distributions/` by default. It reports predicted LST
and Traffic changes plus absolute and percentage population changes in the
urban and whole valid regions. Both population percentage metrics use the
original whole-region population sum as the denominator. This differs from
the city-internal population percentage in Fig.4 and the physical export.
Distribution summaries include sample count, mean, sample standard deviation,
median, percentiles, and extrema; each city has equal weight. Both the default
five-city run and explicit `--version` inputs write only to the selected output
directory. Generated output directories are excluded from Git.

## planning_data.pkl schema

```text
{
  "schema_version": 1,
  "mode": "joint" | "cooling_only" | "both",
  "year": 2020,
  "cities": [782, 339, ...],
  "config": {...},
  "results": [                    ← lightweight per-city summary
    {
      "city_id": 782,
      "joint": {...},
      "cooling_only": {...},
    }, ...
  ],
  "city_data": [                  ← full per-city arrays and trajectories
    {
      "city_id": 782,
      "joint": {...},
      "cooling_only": {...},
    }, ...
  ]
}
```

Spatial arrays (`original_ndvi`, `adjusted_ndvi`, `original_lst`, etc.) are in
processed model space, not physical units. Restore with:

```text
lst_physical    = lst_processed * lst_std + lst_mean
ndvi_physical   = ndvi_processed * ndvi_std + ndvi_mean
pop_physical    = 10 ** (pop_processed * pop_std + pop_mean) - 1   # log-zscore default
```

For differences the mean cancels: `delta_lst_physical = (adjusted - original) * lst_std`.
`final_delta_lst` and `delta_lst_traj` are already in physical-scale differences
(negative = cooling). `delta_pop_traj` is already a percentage change.

`run_forward_inference.py` outputs (`forward_inference.pkl`, `forward_inference_summary.csv`)
store `lst0/lst1/lst01/lst10` in processed scale; city/global means and deltas
in the CSV are restored to physical scale.

## Result packages

| Directory under `outputs/` | Scope | Use |
| --- | --- | --- |
| `joint_gradient_v2_5cities_current` | 5 cities, 1000 steps | Joint optimization for Fig.4 |
| `cooling_only_5cities_1000steps` | 5 cities, 1000 steps | NDVI-only cooling comparison for Fig.4 |
| `joint_gradient_v2_292cities_traffic10pct` | 292 cities, 10% traffic target | Joint target sweep |
| `joint_gradient_v2_292cities_traffic20pct` | 292 cities, 20% traffic target | Joint target sweep and cross-model evaluation input |
| `joint_gradient_v2_292cities_traffic30pct` | 292 cities, 30% traffic target | Joint target sweep |
| `cooling_only_robustness_sweep` | Cooling target, learning-rate, and iteration settings | Cooling robustness analysis |

All joint packages use `optimization.gradient_mode: joint`. Across the same
292 cities, the target sweep produces:

| Traffic target | Mean predicted LST change (K) | Mean predicted Traffic reduction |
| ---: | ---: | ---: |
| 10% | -0.8984 | 6.3631% |
| 20% | -0.8855 | 13.3818% |
| 30% | -0.8732 | 20.1285% |

Traffic reductions are relative changes. Targets enter the optimization loss;
they are not imposed by post-processing. These results are model-predicted
responses and do not establish real-world or causal intervention effects.

The five-city metrics are in
[planning_city_metrics.csv](outputs/planning_distributions/planning_city_metrics.csv).
Fig.4 code and its data package are in [main_figs_v2/fig4](../../main_figs_v2/fig4).
For evaluation with Swin and scratch ConvNeXt, use the commands in
[the cross-model experiment](../planning_cross_val_exp/EXPERIMENT.md), which
write to `experiments/planning_cross_val_exp/outputs/cross_model_evaluation_lst_scratch/`.
