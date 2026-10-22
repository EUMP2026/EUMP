# planning_cross_val_exp

This experiment evaluates fixed planning interventions with two
independently trained LST predictors:

- `one_stage_swin`: `eump_ablation_v2_one_stage_swin_l_unfrozen/lst`
- `convnext_one_stage_scratch`: `planning_cross_val_exp_convnext_one_stage_scratch/lst`

The ConvNeXt encoder and LST head are trained jointly from scratch.
Evaluation uses the validation-best checkpoint.

The same planning intervention is passed unchanged to both models. The
evaluator reports factorial LST responses for the original input (`00`), the
full intervention (`11`), and the two single-channel contrasts (`01`, `10`).
Negative LST changes indicate a predicted cooling response. These are
model-internal predicted responses and do not establish real-world or causal
intervention effects.

The separate [gradient/uniform/hotspot greening experiment](../manipulate_exp/README.md)
evaluates fixed NDVI-only maps with EUMP, Swin, and scratch ConvNeXt. Its
scenario definitions, input maps, and paired city win rates are documented
there. The factorial columns here describe joint-planning contrasts.

## Dependencies

Use the repository's Python dependencies and the `data=eump_full/server_v2`
processed scene store, including preprocessing and split metadata. Evaluation
requires both model directories listed above, each containing
`eump_downstream_model.pt`, and the joint planning package at
`experiments/planning_exp/outputs/joint_gradient_v2_292cities_traffic20pct/planning_data.pkl`.
See [the planning experiment](../planning_exp/EXPERIMENT.md) for generating
that package. `config/cross_model_eval_scratch.yaml` specifies the model and
data paths and the output directory.
The training and evaluation entry points use these scratch configurations by
default.

## Reproduce

Inspect the commands without starting a job:

```bash
python experiments/planning_cross_val_exp/train.py \
  --config-name convnext_one_stage_lst_scratch.yaml --dry-run
python experiments/planning_cross_val_exp/run_cross_model_eval.py \
  --config experiments/planning_cross_val_exp/config/cross_model_eval_scratch.yaml \
  --dry-run
```

Train/evaluate the ConvNeXt LST model:

```bash
python experiments/planning_cross_val_exp/train.py \
  --config-name convnext_one_stage_lst_scratch.yaml --phase all \
  --nproc 4 --lst-gpus 0,1,2,3
```

This model uses `downstream.encoder_source=convnext_scratch` and
`experiment.scratch_backbone=true`; no reconstruction checkpoint is loaded.
Its model path is
`model/planning_cross_val_exp_convnext_one_stage_scratch/lst/`.

Evaluate Swin and scratch ConvNeXt on the v2 joint-gradient package:

```bash
python experiments/planning_cross_val_exp/run_cross_model_eval.py \
  --planning-variant joint \
  --config experiments/planning_cross_val_exp/config/cross_model_eval_scratch.yaml \
  --device cuda --num-workers 0
```

Evaluate the NDVI-only cooling baseline from the same planning package:

```bash
python experiments/planning_cross_val_exp/run_cross_model_eval.py \
  --planning-variant cooling_only \
  --output-dir experiments/planning_cross_val_exp/outputs/cross_model_evaluation_lst_scratch/cooling_only \
  --device cuda --num-workers 0
```

Joint results are written to
`outputs/cross_model_evaluation_lst_scratch/`; cooling-only results use its
`cooling_only/` subdirectory. Each variant contains only LST tables and
predictions for the two models:

- `cross_model_city_results.csv`: one row per model and city;
- `cross_model_comparison_summary.{csv,json}`: city-level aggregates;
- `one_stage_swin/` and `convnext_one_stage_scratch/`: model-specific CSV, pickle, and checkpoint
  metadata;
- `cross_model_comparison.png` and `cross_model_report.md`: LST-only visual and
  summary report.

The evaluator and the [NDVI-only comparison](../manipulate_exp/README.md) share
the LST loading and data helpers in `model_utils.py`. Intervention optimization
is implemented by [planning_exp](../planning_exp/EXPERIMENT.md).
