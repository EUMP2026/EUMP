# Repository Agent Notes

## Public documentation

Keep one primary document per experiment: `EXPERIMENT.md`, or `README.md` for
the manipulation experiment. Merge method, dependency, command, output, and
metric definitions into that document; link to related experiments rather
than duplicating their results. Public documentation should describe the
reproducible experiment, without local cleanup, migration, historical-output,
or backup narratives. Keep generated README templates consistent with the
primary documents.

## Planning model release and reproducibility

The retained planning line is the v2 joint-gradient experiment. It uses one
backward pass on the LST and traffic objectives, with both objectives updating
the NDVI and population intervention channels. The canonical five-city run is
defined by `experiments/planning_exp/config.yaml` and
`experiments/planning_exp/run.py`.

Uploading only the MAE pretrain and LST downstream weights is **not sufficient**
to reproduce planning. The current run also requires the traffic head selected
in `planning_exp/config.yaml`, the processed EUMP data store, and the exact
preprocessing and split metadata.

The current model dependencies are:

```text
model/eump_full_mae_v3/pretrain/eump_reconstruction_best_checkpoint.pt
model/eump_full_mae_v3/lst/eump_downstream_model.pt
model/eump_full_mae_v3/traffic_cv/swin_embed_head_128_d05_fold2/swin_embedding_scalar_head.pt
```

The checkpoint semantics are important:

- `eump_reconstruction_best_checkpoint.pt` is the validation-best MAE-v3
  encoder checkpoint. `eump_reconstruction_model.pt` is the final pretraining
  model and is not the encoder selected by planning.
- `eump_downstream_model.pt` is the validation-best LST downstream model. Its
  filename is the configured `save_file`; it is not the final training-state
  checkpoint. `eump_downstream_checkpoint.pt` is a resumable training
  checkpoint and is not used by planning.
- The traffic fold-2 file is a separate pooled embedding regression head. It
  is required for the current joint planning run and is not contained in the
  LST downstream file.
- `run.py` currently builds the traffic head through the shared model setup for
  every planning mode, so the configured traffic checkpoint is also required
  by the present `cooling_only` entry point even though that objective does not
  optimize traffic.

An exact planning rerun additionally needs:

1. The repository code at the release commit, including `run.py`, the EUMP
   model/data modules, and the Hydra configuration tree.
2. The `data=eump_full/server_v2` processed scene store, currently resolved by
   `configs/data/eump_full/server_v2.yaml` to `data/processed_v2/`.
3. The processed store's `manifest/schema.json`, scene manifests, Zarr scene
   data, `manifest/stats/preprocessing_v1.json`, and
   `manifest/splits/legacy_compatible_seed42.json`.
4. The five-city IDs in `planning_exp/config.yaml`, or the 292-city IDs in the
   planning package used for the selected run.
5. The Python dependencies and compatible PyTorch/CUDA environment from the
   repository requirements and release notes.

For the current five-city Fig.4 package, the minimal reproducible command is:

```bash
python experiments/planning_exp/run.py \
  --mode both --gradient-mode joint \
  --cities 782,339,216,475,261 \
  data=eump_full/server_v2
```

This command requires the three model files above. It writes a
`planning_data.pkl` package under the selected output directory; Fig.4 is then
rendered with `main_figs_v2/fig4/fig4.py`.

Planning physical exports and city-distribution statistics are implemented by
`experiments/planning_exp/export_physical_results.py` and
`plot_distributions.py`, with generated artifacts under the experiment's
ignored `outputs/` directory. Fig.4 assembly is limited to
`main_figs_v2/fig4/prepare_data.py`; its CSV metrics use the renderer's raster
transformations. Keep documentation in the planning `EXPERIMENT.md` and
the Fig.4 `README.md`.

For a lightweight release that only reproduces the published figures, upload
the already-generated `planning_data.pkl` packages and the rendered Fig.4
assets in addition to the code. That reproduces visualization and summary
tables, but it does not replace the processed data and model files needed to
rerun the optimization on new cities or settings.

Do not commit model weights, processed data, Hugging Face caches, or release
staging bundles to Git. Keep those artifacts in Hugging Face or external
storage and record their revisions/checksums in the release metadata.

## Cross-planning LST release

The retained cross-planning experiment is
`experiments/planning_cross_val_exp/`. It evaluates the same unchanged planning
intervention with two independently trained LST models:

```text
model/eump_ablation_v2_one_stage_swin_l_unfrozen/lst/eump_downstream_model.pt
model/planning_cross_val_exp_convnext_one_stage_scratch/lst/eump_downstream_model.pt
```

The current ConvNeXt encoder and LST head were trained jointly from scratch
with `downstream.encoder_source=convnext_scratch` and
`experiment.scratch_backbone=true`. No pretraining checkpoint was loaded;
the downstream model file contains the validation-best weights. Use
`config/convnext_one_stage_lst_scratch.yaml` for training and
`config/cross_model_eval_scratch.yaml` for cross-planning evaluation.

`model_utils.py` provides the shared LST model/data helpers; planning optimization
is owned by `planning_exp/run.py`.
`run_cross_model_eval.py` is LST-only. It writes factorial predicted LST
responses for `one_stage_swin` and `convnext_one_stage_scratch`. Keep its
evaluation outputs LST-only. The planning package supplies
the original and adjusted NDVI/POP maps, and must be generated with the
planning dependencies described above. The cross-model comparison is a
model-internal predicted-response analysis and should not be described as a
causal or real-world intervention evaluation.

The separate gradient/uniform/hotspot greening comparison is exported in
`experiments/manipulate_exp/`. Its ConvNeXt results use the scratch
checkpoint above with the `convnext_one_stage` CSV identifier and NPZ keys.
This directory is LST-only: retain the
cooling tables, LST prediction archive, four Swin/ConvNeXt LST test-metric
records, metadata, validation report, and export script. Visualization code
and rendered figures are excluded from this delivery directory.
Do not introduce Traffic/TCI rows or fold columns. Match documentation to the
files actually retained by the user. The current paired city win rates are
Swin 94.18% / 80.14% and scratch
ConvNeXt 92.12% / 75.68% against uniform / hotspot, respectively. These rates
belong to the NDVI-only intervention comparison, not to the factorial
joint-planning tables.

`experiments/manipulate_exp/README.md` and
`run_single_ndvi_cross_model.py` are the retained manipulation experiment.
Generated manipulation artifacts belong in `experiments/manipulate_exp/outputs/`,
which is the default reference and output directory.
The physical intervention maps are embedded in the local prediction NPZ;
runtime inference must use the local archive rather than temporary delivery paths.
Keep the script, README, and local `.gitignore` in Git; exclude generated
CSVs, prediction NPZs, evaluation metadata, and validation reports.
In-place evaluation backups go outside the repository
under `<repository-parent>/<repository-name>_backups/`.

## Domain-transfer protocols

`experiments/eump_domain_transfer/` supports both protocols:

- `config.yaml`: `pretrain.enabled=false`, six scenarios, Swin with the global
  MAE v3 best checkpoint plus Scalar MLP and AEF MLP baselines.
- `config_source_pretrain.yaml`: `pretrain.enabled=true`, source-domain MAE
  training for Tropical and GDP Poor, followed by frozen-encoder downstream
  training. The MAE stage starts from the standard pretrained `swin_L`
  initialization. It is not random-initialized backbone training.

The disabled setting means no scenario-specific MAE stage; do not describe it
as a Swin model without pretrained weights. Both protocols retain LST,
precipitation, PM2.5, and Traffic. The global run and the source-pretraining run
use stable output names ending in `mae_v3_city_classes_multi_transfer` and
`mae_v3_source_pretrain_climate_tropic_gdp_poor`, respectively. Source MAE and
all domain-transfer downstream models use their final checkpoints; there is
no validation or target-based checkpoint selection in these runs.

Use `export_results.py` to rebuild both protocols' comparable tables and the
two supplementary comparisons directly from raw summaries. The canonical
exports are under ignored `outputs/delivery/without_source_pretrain/`,
`with_source_pretrain/`, and `comparisons/`. Keep processed-space reconstruction
losses separate from physical downstream metrics. Domain-transfer table
generation is owned by this exporter.
Keep backups, checkpoints, and all generated result bundles out of Git.

## Other reproducibility experiments

The remaining experiment entry points are `eump_ablation_suite`,
`eump_architecture_ablation`, `eump_aef_compare`, and `eump_channel_drop`.
Keep their generated outputs ignored and use the existing collectors to
rebuild tables without training. Architecture collection checks the actual
ConvNeXt encoder parameter names before including fixed-task metrics.
ConvNeXt downstream pipelines must specify `backbone_kind=convnext` and
`encoder_source=convnext_reconstruction_checkpoint`. The retained architecture
experiment includes Swin-UNet, ConvNeXt-UNet, and MoCo, with matching launcher,
collector, and documentation entries.
The architecture/pretraining rows follow method-specific recipes alongside
the dedicated masking, auxiliary-supervision, and initialization ablations.
MoCo defaults to 100 training epochs and a 100-epoch scheduler horizon
(`contrastive.epochs=100`, `contrastive.scheduler_epochs=100`). The reported
comparison uses the epoch-31 encoder retained as
`eump_contrastive_best_checkpoint.pt`; all four downstream tasks use this
encoder. Override `contrastive.epochs=31` to run only the schedule prefix.
Retain the existing MoCo implementation and reported results. Keep exploratory
contrastive experiments under ignored `tmp/` with separate artifacts. The
architecture entry point excludes smoke training; use `--dry-run` to inspect
commands.
AEF comparison headline scopes follow `_headline_rows`; preserve sample
coverage per method. Channel-drop repeat statistics first average CV folds
within each corruption seed, then summarize the five corruption seeds.
Do not report the 25 fold/seed combinations as 25 independent repeats.
Channel-drop retains only the augmented downstream group
`eump_full_mae_v3_channel_drop_trained`, selected by
`channel_drop.train.trained_group` for training and evaluation. Its public
documentation and launcher describe this single experiment.
