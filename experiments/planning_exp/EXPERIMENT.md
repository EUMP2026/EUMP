# Urban Planning Optimization Experiment

Optimizes urban input features (NDVI, population density) to achieve target
reductions in LST and traffic. The pretrained encoder and downstream heads are
frozen; selected input channels are treated as learnable parameters.

## Prerequisites

The following artifacts must exist before running this experiment:

- `model/eump_full_mae_v3/pretrain/eump_reconstruction_best_checkpoint.pt`
- `model/eump_full_mae_v3/lst/eump_downstream_model.pt`
- A traffic CV fold checkpoint. Since traffic uses K-fold CV, multiple fold
  models are available; one fold is selected for planning. The default is fold 2:
  `model/eump_full_mae_v3/Traffic_cv/swin_embed_head_128_d05_fold2/swin_embedding_scalar_head.pt`
  Change `traffic_cv.checkpoint` in `config.yaml` to use a different fold.

Two optimization modes:

- **`joint`**: jointly optimizes NDVI (LST gradient) and POP (traffic gradient).
- **`cooling_only`**: optimizes NDVI for LST reduction only.
- **`both`**: runs joint and cooling-only from the original input; required for Fig4.

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

Default run (joint, 5 cities):

```bash
python experiments/planning_exp/run.py data=eump_full/server_v2
```

Both modes (required for Fig4):

```bash
python experiments/planning_exp/run.py --mode both --cities 782,339,216,475,261 \
  data=eump_full/server_v2
```

Large-city multi-GPU statistics:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python experiments/planning_exp/run_urbanfrac_city_stats.py \
  --num-workers 0 --gpus 0,1,2,3,4,5,6,7 --mode both
```

Robustness sweep over LST cooling targets:

```bash
bash experiments/planning_exp/run_lst_delta_base_sweep.sh

# Summarize sweep
python experiments/planning_exp/summarize_lst_delta_base_sweep.py \
  --input-root experiments/planning_exp/outputs/lst_delta_base_joint_sweep_5cities
```

Robustness sweep over other hyperparameters:

```bash
bash experiments/planning_exp/run_robustness_sweep.sh
python experiments/planning_exp/summarize_robustness_sweep.py
```

Forward inference counterfactuals (LST decomposition):

```bash
python experiments/planning_exp/run_forward_inference.py \
  --planning-data experiments/planning_exp/outputs/planning_data.pkl \
  data=eump_full/server_v2
```

Fig4 visualization:

```bash
python experiments/planning_exp/visualize_fig4.py \
  --input-dir experiments/planning_exp/outputs \
  --output experiments/planning_exp/outputs/fig4.png
```

## Outputs

```text
outputs/<run_name>/
  planning_data.pkl       ← canonical output; consumed by Fig4 and statistics scripts
  summary.json
  city_<id>.npz           ← optional per-city arrays (disable with --no-city-npz)
  figures/
```

The LST delta sweep writes one `run.py` output folder per target under
`outputs/lst_delta_base_joint_sweep_5cities/lst_delta_<value>/`.

The large-city statistics script writes a merged package to
`outputs/urbanfrac_cities_ge0p1_stats/planning_data.pkl`.

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

`forward_inference.py` outputs (`forward_inference.pkl`, `forward_inference_summary.csv`)
store `lst0/lst1/lst01/lst10` in processed scale; city/global means and deltas
in the CSV are restored to physical scale.

