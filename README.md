# EUMP: From Features to Futures

Official code for the paper **"From Features to Futures: Embedding Cities for Predictive Modeling and Scenario Planning"**.

## Abstract

Developing sustainable, livable cities in the context of climate change and complex urban dynamics is increasingly dependent on accurate urban modeling and agile, data-informed planning. Inspired by large pre-trained AI models, we propose a shift toward embedding-based urban modeling and planning (EUMP) that learns low-dimensional representations from large-scale unlabeled geospatial data, and embeds heterogeneous urban processes within a unified latent space. Across 1,076 cities worldwide, the learned embeddings captured more urban predictive information and transferred more effectively than handcrafted predictors and general-purpose Earth embeddings. Relative to handcrafted predictors, EUMP improved R² by 0.05–0.41 in predicting land surface temperature, precipitation, traffic congestion and PM2.5 concentration. More importantly, EUMP enables the integration of prediction and scenario planning to quantify trade-offs among planning goals. Counterfactual analyses suggested that urban greening was more effective at cooling dry cities than humid ones, but population dispersal for congestion relief could partly offset this cooling benefit. A community-in-the-loop extension positions EUMP as an AI-assisted decision-support tool for fine-grained urban planning and management.

## Requirements

- Python 3.12, PyTorch 2.6, CUDA 12.4
- Install dependencies: `pip install -r requirements.txt`
- Alternatively, build and run with the provided `Dockerfile`

## Data

The processed scene store is expected at `data/eump_v1/`. See
[`docs/data/data_description.md`](docs/data/data_description.md) for the full
store layout and source descriptions. Data download instructions will be
provided upon publication.

## Training and Evaluation

All runs go through `script/training/run_eump_full_train.sh` and
`script/training/run_eump_full_eval.sh`. Pass a launch suite and a mode:

```bash
# Pretrain
GPU=0,1,2,3,4,5,6,7 NPROC=8 LOG_PROGRESS_BAR=0 \
  bash script/training/run_eump_full_train.sh pretrain \
    launch=eump_full/mae_v3 data=eump_full/server_v2

# Downstream (lst / precip / pm25 / traffic)
GPU=0,1,2,3,4,5,6,7 NPROC=8 LOG_PROGRESS_BAR=0 \
  bash script/training/run_eump_full_train.sh lst \
    launch=eump_full/mae_v3 data=eump_full/server_v2

# City-scalar cross-validation (PM2.5 and Traffic)
bash script/experiments/run_current_pm25_cv.sh    data=eump_full/server_v2
bash script/experiments/run_current_traffic_cv.sh data=eump_full/server_v2

# Evaluation
EVAL_GPU=0 EVAL_NPROC=1 EVAL_MASTER_PORT=29501 LOG_PROGRESS_BAR=0 \
  bash script/training/run_eump_full_eval.sh lst \
    launch=eump_full/mae_v3 data=eump_full/server_v2 \
    evaluation.checkpoint_source=best_model
```

See [`docs/config/eump_config_usage.md`](docs/config/eump_config_usage.md) for
all available launch suites and modes.

## Experiments

| Experiment | Description |
| --- | --- |
| [`experiments/eump_ablation_suite/`](experiments/eump_ablation_suite/EXPERIMENT.md) | Input channel and masking ablations |
| [`experiments/eump_architecture_ablation/`](experiments/eump_architecture_ablation/EXPERIMENT.md) | Swin MAE vs UNet decoder vs MoCo contrastive |
| [`experiments/eump_domain_transfer/`](experiments/eump_domain_transfer/EXPERIMENT.md) | Cross-domain transferability across city classes |
| [`experiments/manipulate_exp/`](experiments/manipulate_exp/EXPERIMENT.md) | Scenario planning via embedding-space manipulation |
| [`experiments/planning_exp/`](experiments/planning_exp/EXPERIMENT.md) | Gradient-based scenario planning via backpropagation through the embedding space |
| [`experiments/zonal_embedding_prediction/`](experiments/zonal_embedding_prediction/EXPERIMENT.md) | Zonal embedding prediction |

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
