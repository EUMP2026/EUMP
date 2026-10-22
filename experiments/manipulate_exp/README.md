# NDVI Intervention Cross-Model LST Evaluation

## Scope and metric

This experiment evaluates fixed gradient, uniform, and hotspot NDVI interventions with three LST predictors for 292 cities. Population and all other input channels are held fixed. The intervention keys are `gradient_dNDVI_single`, `uniform_dNDVI_single`, and `hotspot_dNDVI_single`. City IDs/names, city order, metric masks, and the physical NDVI intervention maps are stored in `outputs/single_ndvi_lst_predictions.npz`. The ConvNeXt one-stage encoder and LST head are trained jointly from scratch, without pretraining; evaluation uses the validation-best checkpoint. The `convnext_one_stage` identifier refers to this model in the CSV and NPZ files.

## Models

- EUMP: `model/eump_full_mae_v3/lst/eump_downstream_model.pt`.
- Swin one-stage: `model/eump_ablation_v2_one_stage_swin_l_unfrozen/lst/eump_downstream_model.pt`.
- ConvNeXt one-stage: `model/planning_cross_val_exp_convnext_one_stage_scratch/lst/eump_downstream_model.pt`.

`outputs/evaluation_metadata.json` records checkpoint provenance and the scratch ConvNeXt checkpoint SHA256.

## Calculation

City cooling is the mean physical LST over `city_mask & ~ocean_mask & valid_mask`, computed separately for each city. `delta_lst_k = scenario mean - baseline mean`; negative values indicate cooling. Cross-city summaries give each city equal weight. `gradient_extra_cooling_vs_*` is the paired mean of `delta_lst_other - delta_lst_gradient`, so positive values favor gradient.

Physical NDVI deltas are divided by each model pipeline's NDVI standard deviation and added to the processed NDVI input channel. Baseline NDVI is checked against the cooling-only planning package for every city. All other input channels remain fixed.

The greening budgets match over each city's entire image; urban-only allocations may differ. Reported cooling values are model-predicted responses and do not establish causal or real-world intervention effects.

## Cooling results

| Model | NDVI allocation | Mean ΔLST (K) | Median ΔLST (K) | Cities cooled | Gradient vs uniform | Gradient vs hotspot |
|---|---|---:|---:|---:|---|---|
| EUMP (MAE v3) | gradient_dNDVI_single | -0.9085 | -0.9441 | 292/292 (100.0%) | 292/292 (100.0%), +0.6735 K | 292/292 (100.0%), +0.5884 K |
| EUMP (MAE v3) | uniform_dNDVI_single | -0.2350 | -0.2183 | 285/292 (97.6%) | 292/292 (100.0%), +0.6735 K | 292/292 (100.0%), +0.5884 K |
| EUMP (MAE v3) | hotspot_dNDVI_single | -0.3201 | -0.3193 | 287/292 (98.3%) | 292/292 (100.0%), +0.6735 K | 292/292 (100.0%), +0.5884 K |
| Swin one-stage | gradient_dNDVI_single | -0.9754 | -0.9047 | 280/292 (95.9%) | 275/292 (94.2%), +0.6338 K | 234/292 (80.1%), +0.3330 K |
| Swin one-stage | uniform_dNDVI_single | -0.3416 | -0.3051 | 244/292 (83.6%) | 275/292 (94.2%), +0.6338 K | 234/292 (80.1%), +0.3330 K |
| Swin one-stage | hotspot_dNDVI_single | -0.6424 | -0.5960 | 263/292 (90.1%) | 275/292 (94.2%), +0.6338 K | 234/292 (80.1%), +0.3330 K |
| ConvNeXt one-stage | gradient_dNDVI_single | -1.1078 | -1.0232 | 281/292 (96.2%) | 269/292 (92.1%), +0.5731 K | 221/292 (75.7%), +0.2678 K |
| ConvNeXt one-stage | uniform_dNDVI_single | -0.5346 | -0.5150 | 275/292 (94.2%) | 269/292 (92.1%), +0.5731 K | 221/292 (75.7%), +0.2678 K |
| ConvNeXt one-stage | hotspot_dNDVI_single | -0.8399 | -0.7717 | 281/292 (96.2%) | 269/292 (92.1%), +0.5731 K | 221/292 (75.7%), +0.2678 K |

The two gradient comparison columns are repeated on each scenario row for readability. They report paired comparisons within the same city: the first value is the number and percentage of cities where gradient has a more negative ΔLST, and the final value is the mean extra cooling magnitude in Kelvin. The complete values are also stored in `cooling_summary.csv`.

## Independent model test-set accuracy

LST global-pixel metrics use the complete test set and physical Kelvin. City-mean metrics describe per-city aggregate accuracy. The table and `test_set_metrics.csv` contain four LST records: two views for each of Swin and ConvNeXt. ConvNeXt metrics come from the scratch model's validation-best checkpoint evaluation. These test metrics and the 292-city intervention results describe different evaluation cohorts. Blank MAE fields indicate that the source evaluation did not report that metric.

| Model | Target | Test view | N | R² | RMSE (physical) | MAE (physical) |
|---|---|---|---:|---:|---:|---:|
| Swin one-stage | LST | global_pixels | 168666655 | 0.8392 | 2.9755 | NA |
| Swin one-stage | LST | city_mean | 2916 | 0.7918 | 2.4831 | NA |
| ConvNeXt one-stage | LST | global_pixels | 168666655 | 0.8268 | 3.0883 | 2.2637 |
| ConvNeXt one-stage | LST | city_mean | 2916 | 0.8006 | 2.4303 | NA |

## Files

Generated data and evaluation reports are stored in the ignored `outputs/` directory:

- `cooling_city_results.csv`: one row per model, city, and NDVI scenario.
- `cooling_summary.csv`: 292-city equal-weight summaries and paired gradient comparisons.
- `single_ndvi_lst_predictions.npz`: physical NDVI delta maps, baseline/scenario LST prediction grids, city IDs/names, and metric masks.
- `test_set_metrics.csv`: four LST test records; columns are `model,target,evaluation_view,test_count,r2,rmse_physical,mae_physical,metric_scale,source`.
- `evaluation_metadata.json`: source files, checkpoint identifiers, evaluation device, and mask definition.
- `validation_report.json`: current row counts, LST test schema, and agreement between the saved tables and prediction arrays.
- `evaluation_report.md`: cooling and independent test-set metric tables regenerated by the export script.

`run_single_ndvi_cross_model.py` is the source entry point in the experiment directory.

## Reproduce the evaluation

Run from the repository root. Required inputs are the processed scene store and preprocessing/split metadata, the selected model checkpoints and configuration, the cooling-only planning package in `experiments/planning_exp/outputs/cooling_only_robustness_sweep/base_lst1p00_lr1e-3_s1000/planning_data.pkl`, and the result files listed above. The prediction NPZ supplies the three physical NDVI intervention maps, city order, metric masks, and EUMP predictions. Download or copy these data files into `experiments/manipulate_exp/outputs/` before running the command.

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/manipulate_exp/run_single_ndvi_cross_model.py \
  --models one_stage_swin,convnext_one_stage --device cuda:0 --num-workers 0
```

Use `--models convnext_one_stage` to evaluate only ConvNeXt. EUMP and unselected model predictions are read from the reference package. Both the reference and output directories default to `experiments/manipulate_exp/outputs/`. Use `--reference-dir` and `--output-dir` to select other directories.

## Data files

The prediction NPZ, result CSVs, evaluation metadata, and validation report are excluded from Git by the local `.gitignore`. The source code and this README describe the required files and how to regenerate the evaluation outputs.
