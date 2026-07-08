# Codebase Layout

The key runtime boundary:

- `pipeline.kind` describes the data and target contract.
- `model.kind` describes the model architecture.
- `urban_embedding/training/eump/registry.py` maps `(pipeline.kind, model.kind)`
  to trainer/evaluator factories.

## Package Structure

```text
urban_embedding/
  cli/                  # user-facing entry points called from main/ and script/
  core/                 # paths, logging, reproducibility
  data/
    processed/          # scene-store reader, manifests, batch schema, pipeline_spec
  models/
    backbones/swinv2_masked.py
    eump/
      swin/{reconstruction,downstream,heads}.py
      swin_unet/reconstruction.py
      convnext_unet/{encoder,reconstruction}.py
      contrastive/{model,projection}.py
      city_scalar_mlp/model.py
  training/
    runner.py / evaluation.py / loops.py / checkpointing.py
    eump/
      registry.py
      swin_reconstruction/trainer.py
      swin_downstream/{trainer,evaluator}.py
      swin_contrastive/{trainer,loss}.py
      city_scalar_mlp/{trainer,evaluator,features,cache}.py
```

## Entry Points

- `main/train_eump.py` — Hydra training entry, delegates to `urban_embedding.cli.train_eump`
- `main/evaluate_eump.py` — Hydra evaluation entry, delegates to `urban_embedding.cli.evaluate_eump`
- `script/training/run_eump_full_train.sh` — multi-GPU training launcher
- `script/training/run_eump_full_eval.sh` — evaluation launcher (single-GPU by default)

## Runtime Registry

| `pipeline.kind` | `model.kind` | Runtime |
| --- | --- | --- |
| `eump_reconstruction` | `swin_reconstruction` | Swin MAE pretrain |
| `eump_reconstruction` | `swin_unet_reconstruction` | Swin encoder + UNet decoder |
| `eump_reconstruction` | `convnext_unet_reconstruction` | ConvNeXt encoder + UNet decoder |
| `eump_contrastive` | `swin_contrastive` | Swin MoCo contrastive pretrain |
| `eump_downstream` | `swin_downstream_light` | Frozen Swin encoder + light head |
| `eump_city_scalar_downstream` | `city_scalar_mlp` | City-level scalar MLP baseline |

`eump_scalar_mlp_downstream` is a compatibility alias for `eump_city_scalar_downstream`.

## Hydra Config Layout

```text
configs/
  train.yaml / evaluate.yaml
  data/eump_full/
    server_v2.yaml          # full processed scene store (server)
    raw_sample.yaml         # local smoke-test sample
  launch/eump_full/
    mae_v3.yaml             # current baseline — maps modes to pipeline list
    scalar_mlp.yaml
    aef_mlp.yaml
    architecture_v1.yaml
    contrastive_v2.yaml
  pipeline/eump_full/
    _shared/                # shared profile and target building blocks
    mae_v3/                 # pretrain, lst, precip, pm25, traffic
    scalar_mlp/             # lst, precip, pm25, traffic
    aef_mlp/                # lst, precip, pm25, traffic
    architecture_v1/        # swin_unet, convnext_unet
    contrastive_v2/         # pretrain
  model/eump/
    swin/{reconstruction,downstream_light}.yaml
    city_scalar_mlp.yaml
```

`launch/` configs are wrapper-only — they map training modes (`all`, `lst`,
`pm25`, ...) to ordered job lists. `pipeline/` configs are what Hydra actually
composes. Pass `launch=eump_full/mae_v3` to the shell wrappers; do not pass
`pipeline=` directly.
