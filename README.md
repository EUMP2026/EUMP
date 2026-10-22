# EUMP: From Features to Futures

Official code for the paper **"From Features to Futures: Embedding Cities for Predictive Modeling and Scenario Planning"**.

## Abstract

Developing sustainable, livable cities in the context of climate change and complex urban dynamics is increasingly dependent on accurate urban modeling and agile, data-informed planning. Inspired by large pre-trained AI models, we propose a shift toward embedding-based urban modeling and planning (EUMP) that learns low-dimensional representations from large-scale unlabeled geospatial data, and embeds heterogeneous urban processes within a unified latent space. Across 1,076 cities worldwide, the learned embeddings captured more urban predictive information and transferred more effectively than handcrafted predictors and general-purpose Earth embeddings. Relative to handcrafted predictors, EUMP improved R² by 0.05–0.41 in predicting land surface temperature, precipitation, traffic congestion and PM2.5 concentration. More importantly, EUMP can generate pixel-level intervention scenarios that advance towards predefined planning targets. Counterfactual analyses suggested that urban greening was more effective at cooling dry cities than humid ones, and that peri-urban greening could cool urban centers. A community-in-the-loop extension positions EUMP as an AI-assisted decision-support tool for fine-grained urban planning and management.

## Requirements

- Python 3.12, PyTorch 2.6, CUDA 12.4
- Install dependencies: `pip install -r requirements.txt`
- Alternatively, build and run with the provided `Dockerfile`

## Data

The commands below use `data=eump_full/server_v2`, which resolves the
processed scene store to `data/processed_v2/`. See
[`docs/data/data_description.md`](docs/data/data_description.md) for the full
store layout and source descriptions. Data download instructions will be
provided upon publication.

## Training, Evaluation, and Planning

In this repository, `mae_v3` is the configuration name for the **EUMP** framework
described in the paper, used as `launch=eump_full/mae_v3` in the commands below.

The workflow has three stages: **EUMP pretraining → downstream adaptation →
scenario planning**. First, EUMP learns urban representations through masked
input reconstruction and auxiliary supervision. Next, the validation-best
encoder is frozen while task-specific prediction heads are trained for LST,
precipitation, PM2.5, and traffic. Finally, planning freezes the encoder and
prediction heads and backpropagates the LST and traffic objectives to update
NDVI and population inputs, producing model-predicted intervention responses.

Run the following commands from the repository root. Training and evaluation
use `script/training/run_eump_full_train.sh` and
`script/training/run_eump_full_eval.sh`, with a launch suite and a task mode.
Adjust `GPU` and `NPROC` to the available hardware.

```bash
# 1. EUMP pretraining
GPU=0,1,2,3,4,5,6,7 NPROC=8 LOG_PROGRESS_BAR=0 \
  bash script/training/run_eump_full_train.sh pretrain \
    launch=eump_full/mae_v3 data=eump_full/server_v2

# 2. Downstream head training (LST; use precip for precipitation)
GPU=0,1,2,3,4,5,6,7 NPROC=8 LOG_PROGRESS_BAR=0 \
  bash script/training/run_eump_full_train.sh lst \
    launch=eump_full/mae_v3 data=eump_full/server_v2

# City-scalar cross-validation (PM2.5 and Traffic)
bash script/training/run_eump_full_train.sh pm25 \
  launch=eump_full/mae_v3 data=eump_full/server_v2
bash script/training/run_eump_full_train.sh traffic \
  launch=eump_full/mae_v3 data=eump_full/server_v2

# Evaluation
EVAL_GPU=0 EVAL_NPROC=1 EVAL_MASTER_PORT=29501 LOG_PROGRESS_BAR=0 \
  bash script/training/run_eump_full_eval.sh lst \
    launch=eump_full/mae_v3 data=eump_full/server_v2 \
    evaluation.checkpoint_source=best_model

# 3. Five-city planning: joint LST/traffic and NDVI-only cooling scenarios
python experiments/planning_exp/run.py \
  --mode both --gradient-mode joint \
  --cities 782,339,216,475,261 \
  data=eump_full/server_v2
```

Planning requires the validation-best EUMP encoder, the validation-best LST
model, and the Traffic CV fold-2 head under `model/eump_full_mae_v3/`, together
with the processed data and preprocessing/split metadata. The pretraining,
LST, and Traffic commands above generate these model artifacts. They can also
be supplied as released checkpoints. The planning command exports
`planning_data.pkl` under `experiments/planning_exp/outputs/<run_name>/` for
Fig.4 rendering and physical-result summaries. These outputs describe predicted
responses and do not establish real-world or causal intervention effects.

See [`docs/config/eump_config_usage.md`](docs/config/eump_config_usage.md) for
all launch suites and modes, and the
[planning experiment document](experiments/planning_exp/EXPERIMENT.md) for
checkpoint paths, optimization settings, and output definitions.

## Experiments

| Experiment | Description |
| --- | --- |
| [`experiments/eump_ablation_suite/`](experiments/eump_ablation_suite/EXPERIMENT.md) | Input channel and masking ablations |
| [`experiments/eump_architecture_ablation/`](experiments/eump_architecture_ablation/EXPERIMENT.md) | Swin/ConvNeXt UNet reconstruction and MoCo pretraining comparisons |
| [`experiments/eump_aef_compare/`](experiments/eump_aef_compare/EXPERIMENT.md) | EUMP, one-stage Swin, AEF, and scalar baselines on four prediction tasks |
| [`experiments/eump_channel_drop/`](experiments/eump_channel_drop/EXPERIMENT.md) | Robustness to missing input channels with channel-drop-augmented downstream heads |
| [`experiments/eump_domain_transfer/`](experiments/eump_domain_transfer/EXPERIMENT.md) | Transfer across city classes with global and source-domain MAE pretraining protocols |
| [`experiments/manipulate_exp/`](experiments/manipulate_exp/README.md) | Fixed NDVI interventions evaluated with EUMP, Swin, and scratch ConvNeXt LST models |
| [`experiments/planning_exp/`](experiments/planning_exp/EXPERIMENT.md) | Gradient-based scenario planning via backpropagation through the embedding space |
| [`experiments/planning_cross_val_exp/`](experiments/planning_cross_val_exp/EXPERIMENT.md) | Fixed planning interventions evaluated with independent Swin and scratch ConvNeXt LST predictors |

## Citation

```bibtex
@article{eump[year],
  title   = {From Features to Futures: Embedding Cities for Predictive Modeling and Scenario Planning},
  year    = {[year]},
  journal = {[venue]},
}
```

## License

See LICENSE for details.
