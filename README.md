# EUMP: From Features to Futures

Official code for the paper **"From Features to Futures: Embedding Cities for Predictive Modeling and Scenario Planning"**.

## Abstract

Developing sustainable, livable cities in the context of climate change and complex urban dynamics is increasingly dependent on accurate urban modeling and agile, data-informed planning. Inspired by the success of large pre-trained AI models, we propose a shift toward embedding-based urban modeling that learns generalizable low-dimensional representations and latent dependencies from large-scale unlabeled geospatial data. More importantly, our unified framework enables the integration of prediction and scenario planning such that urban heat island effect and traffic congestion were jointly mitigated through manipulating green space and population dispersal in the embedding space. We further propose a 'community-in-the-loop' framework that positions our model within AI-based urban planning processes. This study marks a critical step toward realizing the transformative role of AI in fine-grained urban planning and management.

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
