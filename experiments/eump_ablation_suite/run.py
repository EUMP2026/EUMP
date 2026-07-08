#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import statistics
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from urban_embedding.data.crop import sample_random_crop_box
from urban_embedding.data.processed.pipeline_spec import PipelineSpec
from urban_embedding.data.processed.reader import ProcessedSceneStoreReader
from urban_embedding.data.processed.split_manifest import load_split_manifest
from urban_embedding.training.evaluation import normalize_evaluation_config, run_evaluation
from urban_embedding.training.runner import normalize_runtime_config, run_training


CONFIG_DIR = REPO_ROOT / "configs"
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = EXPERIMENT_DIR / "config_v2.yaml"
TASK_PIPELINES = {
    "lst": "lst",
    "precip": "precip",
    "pm25": "pm25",
    "traffic": "traffic",
    "ghg": "ghg",
}
TRAFFIC_CV_TASK = "traffic"
PM25_CV_TASK = "pm25"
ONE_STAGE_FAMILY = "one_stage"
METRIC_COLUMNS = (
    "family",
    "variant",
    "task",
    "fold",
    "fold_count",
    "split",
    "target",
    "target_kind",
    "scope",
    "rmse",
    "rmse_std",
    "mae",
    "mae_std",
    "mse",
    "mse_std",
    "r2",
    "r2_std",
    "count",
    "count_std",
    "experiment_group",
    "experiment_name",
    "model_path",
    "checkpoint_path",
)
TRAFFIC_CV_RUN_NAME_DEFAULT = "traffic_cv_seed42"


@dataclass(frozen=True, slots=True)
class VariantSpec:
    family: str
    name: str
    payload: dict[str, Any]

    @property
    def label(self) -> str:
        return f"{self.family}/{self.name}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EUMP MAE ablation-suite experiments.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--families", default=None, help="Comma-separated family names.")
    parser.add_argument("--variants", default=None, help="Comma-separated variant names.")
    parser.add_argument("--tasks", default=None, help="Comma-separated downstream tasks.")
    parser.add_argument("--eval-splits", default=None, help="Comma-separated eval splits.")
    parser.add_argument("--stages", default=None, help="Comma-separated stages: train,eval,collect.")
    parser.add_argument("--continue-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--worker-stage", choices=("train", "eval"), default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-config", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("overrides", nargs="*", help="Hydra overrides forwarded to composed runtime configs.")
    return parser.parse_args(argv)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _comma_list(value: str | None) -> list[str] | None:
    if value is None or not str(value).strip():
        return None
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _truthy(value: str | None) -> bool:
    return str(value or "").lower() in {"1", "true", "yes", "on"}


def _sanitize_segment(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value).strip())
    return cleaned.strip("_-") or "unnamed"


def _resolve_path(value: Any, *, base: Path = REPO_ROOT) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else base / path


def _load_suite_config(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(payload, dict):
        raise TypeError(f"Ablation config must be a mapping: {path}")
    cfg = payload.get("ablation_suite", payload)
    if not isinstance(cfg, dict):
        raise TypeError(f"ablation_suite section must be a mapping: {path}")
    return cfg


def _selected_values(defaults: list[Any], selected: str | None) -> list[str]:
    parsed = _comma_list(selected)
    if parsed is not None:
        return parsed
    return [str(value) for value in defaults]


def _selected_tasks(suite_cfg: dict[str, Any], *, selected_families: list[str], selected: str | None) -> list[str]:
    parsed = _comma_list(selected)
    if parsed is not None:
        return parsed
    if len(selected_families) == 1:
        family_cfg = suite_cfg.get("families", {}).get(selected_families[0], {})
        family_tasks = family_cfg.get("default_tasks") if isinstance(family_cfg, dict) else None
        if family_tasks:
            return [str(value) for value in _as_list(family_tasks)]
    return [str(value) for value in _as_list(suite_cfg.get("default_tasks"))]


def _merge_overrides(default_overrides: list[Any], cli_overrides: list[str]) -> list[str]:
    cli_has_data = any(override.startswith("data=") or override.startswith("+data=") for override in cli_overrides)
    merged: list[str] = []
    for override in default_overrides:
        text = str(override)
        if cli_has_data and text.startswith("data="):
            continue
        merged.append(text)
    merged.extend(cli_overrides)
    return merged


def _variant_group(prefix: str, variant: VariantSpec) -> str:
    return _sanitize_segment(f"{prefix}_{variant.family}_{variant.name}")


def _is_reference_variant(variant: VariantSpec) -> bool:
    return bool(variant.payload.get("reference_group"))


def _reference_group(variant: VariantSpec) -> str:
    return str(variant.payload["reference_group"])


def _has_pretrain_source(variant: VariantSpec) -> bool:
    return bool(variant.payload.get("pretrain_source_group"))


def _pretrain_source_group(variant: VariantSpec) -> str:
    return str(variant.payload["pretrain_source_group"])


def _is_one_stage_variant(variant: VariantSpec) -> bool:
    return variant.family == ONE_STAGE_FAMILY or bool(variant.payload.get("one_stage", False))


def _variant_pipeline_suite(suite_cfg: dict[str, Any], variant: VariantSpec) -> str:
    return str(variant.payload.get("pipeline_suite", suite_cfg["base_pipeline_suite"]))


def _uses_pretrain(variant: VariantSpec) -> bool:
    return not _is_one_stage_variant(variant)


def _uses_traffic_cv(variant: VariantSpec, task: str) -> bool:
    if task != TRAFFIC_CV_TASK:
        return False
    return bool(variant.payload.get("traffic_cv", True))


def _uses_pm25_cv(variant: VariantSpec, task: str) -> bool:
    if task != PM25_CV_TASK:
        return False
    return bool(variant.payload.get("pm25_cv", True))


def build_variants(
    suite_cfg: dict[str, Any],
    *,
    selected_families: list[str],
    selected_variants: list[str] | None,
) -> list[VariantSpec]:
    families_cfg = suite_cfg.get("families")
    if not isinstance(families_cfg, dict):
        raise TypeError("ablation_suite.families must be a mapping")
    selected_variant_set = None if selected_variants is None else set(selected_variants)
    variants: list[VariantSpec] = []
    for family in selected_families:
        if family not in families_cfg:
            supported = ", ".join(sorted(families_cfg))
            raise ValueError(f"Unknown ablation family {family!r}. Supported families: {supported}")
        raw_variants = families_cfg[family].get("variants")
        if not isinstance(raw_variants, dict):
            raise TypeError(f"ablation_suite.families.{family}.variants must be a mapping")
        for name, payload in raw_variants.items():
            if selected_variant_set is not None and str(name) not in selected_variant_set:
                continue
            if not isinstance(payload, dict):
                raise TypeError(f"Variant payload must be a mapping: {family}/{name}")
            variants.append(VariantSpec(family=family, name=str(name), payload=deepcopy(payload)))
    if selected_variant_set is not None:
        found = {variant.name for variant in variants}
        missing = sorted(selected_variant_set - found)
        if missing:
            raise ValueError(f"Requested variant(s) were not found in selected families: {', '.join(missing)}")
    if not variants:
        raise ValueError("No ablation variants selected")
    return variants


def _compose_config(config_name: str, *, pipeline: str, overrides: list[str]) -> dict[str, Any]:
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name=config_name, overrides=[f"pipeline={pipeline}", *overrides])
    payload = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(payload, dict):
        raise TypeError(f"Composed config must be a mapping for pipeline={pipeline}")
    return payload


def _scalar_sources_for_inputs(input_sources: list[str]) -> list[str]:
    return [source for source in ("Ta", "precip_bg", "wind") if source in input_sources]


def _source_losses_for_inputs(input_sources: list[str], *, include_city_mask: bool) -> dict[str, str]:
    losses: dict[str, str] = {}
    for source_name in ("POP", "BH", "BF"):
        if source_name in input_sources:
            losses[source_name] = "hurdle_mae"
    if "water" in input_sources:
        losses["water"] = "bce"
    if include_city_mask:
        losses["city_mask"] = "bce"
    return losses


def _source_weights_for_inputs(input_sources: list[str]) -> dict[str, float]:
    return {source_name: 0.0 for source_name in _scalar_sources_for_inputs(input_sources)}


def _apply_input_group(cfg: dict[str, Any], variant: VariantSpec, *, is_pretrain: bool) -> None:
    inputs = variant.payload.get("inputs")
    if not isinstance(inputs, dict):
        return
    sources = [str(source) for source in _as_list(inputs.get("sources"))]
    optional_sources = [str(source) for source in _as_list(inputs.get("optional_sources"))]
    include_city_mask = bool(inputs.get("include_city_mask_as_input", False))

    pipeline = cfg.setdefault("pipeline", {})
    pipeline.setdefault("inputs", {})
    pipeline["inputs"]["sources"] = sources
    if optional_sources:
        pipeline["inputs"]["optional_sources"] = optional_sources
    else:
        pipeline["inputs"].pop("optional_sources", None)
    pipeline["include_city_mask_as_input"] = include_city_mask
    pipeline["city_mask_source"] = "city_mask"

    if not is_pretrain:
        return

    pipeline.setdefault("loss", {})
    pipeline["loss"]["aux_weight"] = 0.0
    targets = pipeline.setdefault("targets", {})
    input_reconstruction = targets.setdefault("input_reconstruction", {})
    input_reconstruction["enabled"] = True
    input_reconstruction["group"] = "reconstruct"
    input_reconstruction["loss"] = "mae"
    input_reconstruction["weight"] = 1.0
    input_reconstruction["mask"] = "valid_and_masked"
    input_reconstruction["scalar_sources"] = _scalar_sources_for_inputs(sources)
    input_reconstruction["scalar_loss"] = "mae"
    input_reconstruction["scalar_weight"] = 1.0
    input_reconstruction["scalar_mask"] = "valid"
    input_reconstruction["source_losses"] = _source_losses_for_inputs(sources, include_city_mask=include_city_mask)
    input_reconstruction["source_weights"] = _source_weights_for_inputs(sources)
    targets["raster"] = []
    targets["scalar"] = []
    targets["crop_scalar"] = []


def _apply_smoke_overrides(cfg: dict[str, Any], suite_cfg: dict[str, Any], *, smoke_split_path: Path | None) -> None:
    smoke_cfg = suite_cfg.get("smoke", {})
    pipeline = cfg.setdefault("pipeline", {})
    pipeline["num_workers"] = int(smoke_cfg.get("num_workers", 0))
    pipeline["pin_memory"] = bool(smoke_cfg.get("pin_memory", False))
    pipeline["persistent_workers"] = bool(smoke_cfg.get("persistent_workers", False))
    pipeline["prefetch_factor"] = int(smoke_cfg.get("prefetch_factor", 2))
    data = cfg.setdefault("data", {})
    if smoke_split_path is not None:
        data["runtime_split_manifest_path"] = str(smoke_split_path)

    if "reconstruct" in cfg:
        cfg["reconstruct"]["epochs"] = int(smoke_cfg.get("epochs", 1))
        cfg["reconstruct"]["batch_size"] = int(smoke_cfg.get("pretrain_batch_size", 8))
        cfg["reconstruct"]["checkpoint_every_n_epochs"] = 1
    if "downstream" in cfg:
        cfg["downstream"]["epochs"] = int(smoke_cfg.get("epochs", 1))
        cfg["downstream"]["batch_size"] = int(smoke_cfg.get("downstream_batch_size", 8))
        cfg["downstream"]["checkpoint_every_n_epochs"] = 1
        selection = cfg["downstream"].setdefault("selection", {})
        if isinstance(selection, dict):
            selection["smoothing_window"] = int(smoke_cfg.get("downstream_selection_smoothing_window", 1))


def apply_variant_config(
    cfg: dict[str, Any],
    *,
    suite_cfg: dict[str, Any],
    variant: VariantSpec,
    task: str | None,
    is_pretrain: bool,
    is_eval: bool,
    smoke: bool,
    smoke_split_path: Path | None,
) -> dict[str, Any]:
    cfg = deepcopy(cfg)
    group = _variant_group(str(suite_cfg["experiment_group_prefix"]), variant)
    experiment = cfg.setdefault("experiment", {})
    experiment["group"] = group
    experiment["name"] = "pretrain" if is_pretrain else str(task)
    experiment["pretrain_model"] = str(variant.payload.get("pretrain_model", "swin_L"))
    experiment["scratch_backbone"] = bool(variant.payload.get("scratch_backbone", False))

    if "mask_ratio" in variant.payload and "reconstruct" in cfg:
        cfg["reconstruct"]["mask_ratio"] = float(variant.payload["mask_ratio"])

    if "center_ratio" in variant.payload:
        center_ratio = float(variant.payload["center_ratio"])
        experiment["center_ratio"] = center_ratio
        cfg.setdefault("pipeline", {})["center_ratio"] = center_ratio

    _apply_input_group(cfg, variant, is_pretrain=is_pretrain)

    if variant.payload.get("no_aux") and is_pretrain and not variant.payload.get("inputs"):
        pipeline = cfg.setdefault("pipeline", {})
        pipeline.setdefault("loss", {})["aux_weight"] = 0.0
        targets = pipeline.setdefault("targets", {})
        targets["raster"] = []
        targets["crop_scalar"] = []
        aux_weights = pipeline.get("aux_raster_target_weights")
        if isinstance(aux_weights, dict):
            for key in aux_weights:
                aux_weights[key] = 0.0

    if not is_pretrain and "downstream" in cfg:
        if _is_one_stage_variant(variant):
            cfg["downstream"]["encoder_source"] = "swin_pretrained"
            cfg["downstream"]["encoder_model_name"] = None
            cfg["downstream"]["encoder_checkpoint_file"] = None
            cfg["downstream"]["freeze_encoder"] = False
        elif _has_pretrain_source(variant):
            source_group = _pretrain_source_group(variant)
            source_checkpoint = _model_file(source_group, "pretrain", str(suite_cfg["encoder_checkpoint_file"]))
            if source_checkpoint.exists():
                cfg["downstream"]["encoder_model_name"] = f"{source_group}/pretrain"
            else:
                cfg["downstream"]["encoder_model_name"] = f"{group}/pretrain"
            cfg["downstream"]["encoder_checkpoint_file"] = str(suite_cfg["encoder_checkpoint_file"])
        else:
            cfg["downstream"]["encoder_model_name"] = f"{group}/pretrain"
            cfg["downstream"]["encoder_checkpoint_file"] = str(suite_cfg["encoder_checkpoint_file"])

    if is_eval:
        evaluation = cfg.setdefault("evaluation", {})
        eval_cfg = suite_cfg.get("evaluation", {})
        evaluation["checkpoint_source"] = str(eval_cfg.get("checkpoint_source", "best_model"))
        evaluation["metric_scale"] = str(eval_cfg.get("metric_scale", "physical"))
        evaluation["center_ratio"] = float(eval_cfg.get("center_ratio", 1.0))

    cfg.setdefault("logging", {})["progress_bar"] = _truthy(os.environ.get("LOG_PROGRESS_BAR", "1"))

    if smoke:
        _apply_smoke_overrides(cfg, suite_cfg, smoke_split_path=smoke_split_path)
    return normalize_evaluation_config(cfg) if is_eval else normalize_runtime_config(cfg)


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=OmegaConf.create(payload), f=str(path), resolve=True)


def _pretrain_aux_weight(cfg: dict[str, Any]) -> float:
    return float(cfg["pipeline"]["loss"]["aux_weight"])


def _validate_pretrain_aux_policy(
    *,
    suite_cfg: dict[str, Any],
    variant: VariantSpec,
    base_cfg: dict[str, Any],
    resolved_cfg: dict[str, Any],
) -> None:
    base_aux_weight = _pretrain_aux_weight(base_cfg)
    expected_base_aux_weight = suite_cfg.get("base_pretrain_aux_weight")
    if expected_base_aux_weight is not None:
        expected = float(expected_base_aux_weight)
        if abs(base_aux_weight - expected) > 1e-12:
            raise ValueError(
                "Base pretrain aux weight drifted from the ablation-suite expectation: "
                f"expected={expected} actual={base_aux_weight}. "
                f"Update {suite_cfg['base_pipeline_suite']}/pretrain or "
                "ablation_suite.base_pretrain_aux_weight intentionally."
            )

    if str(suite_cfg.get("base_pipeline_suite")) == "eump_full/mae_v3":
        aux_raster_weights = base_cfg.get("pipeline", {}).get("aux_raster_target_weights", {})
        aux_source_weights = base_cfg.get("pipeline", {}).get("loss", {}).get("aux_source_weights", {})
        if abs(float(aux_raster_weights.get("PM25", 1.0))) > 1e-12:
            raise ValueError("mae_v3 ablation base must disable PM25 raster aux weight")
        if abs(float(aux_source_weights.get("PM25", 1.0))) > 1e-12:
            raise ValueError("mae_v3 ablation base must disable PM25 source-level aux weight")

    resolved_aux_weight = _pretrain_aux_weight(resolved_cfg)
    if variant.family == "input_groups" or variant.payload.get("no_aux"):
        if abs(resolved_aux_weight) > 1e-12:
            raise ValueError(
                f"No-aux variant {variant.name!r} must disable aux loss, "
                f"got aux_weight={resolved_aux_weight}"
            )
        return

    if abs(resolved_aux_weight - base_aux_weight) > 1e-12:
        raise ValueError(
            f"Variant {variant.label!r} must inherit base pretrain aux_weight={base_aux_weight}, "
            f"got {resolved_aux_weight}"
        )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")


def _printable_command(cmd: list[str], env: dict[str, str] | None = None) -> str:
    prefix = [] if env is None else [f"{key}={value}" for key, value in env.items()]
    return "+ " + " ".join(shlex.quote(part) for part in [*prefix, *cmd])


def _gpu_env(stage: str, runner_cfg: dict[str, Any]) -> tuple[str, int, str]:
    runner_profile = "eval" if stage == "eval" else "train"
    if runner_profile == "eval":
        gpu = os.environ.get("EVAL_GPU", "0")
        nproc = os.environ.get("EVAL_NPROC", "1")
        master_port = os.environ.get("EVAL_MASTER_PORT", str(runner_cfg.get("eval_master_port", "29501")))
    else:
        gpu = os.environ.get("GPU", "0")
        nproc = os.environ.get("NPROC")
        if nproc is None:
            nproc = str(len([item for item in gpu.split(",") if item.strip()]))
        master_port = os.environ.get("MASTER_PORT", "29500")
    return gpu, int(nproc), master_port


def _worker_command(stage: str, config_path: Path, *, suite_cfg: dict[str, Any]) -> tuple[list[str], dict[str, str]]:
    runner_cfg = suite_cfg.get("runner", {})
    gpu, nproc, master_port = _gpu_env(stage, runner_cfg)
    cmd = [
        "torchrun",
        "--nproc_per_node",
        str(nproc),
        "--nnodes",
        os.environ.get("NNODES", "1"),
        "--node_rank",
        os.environ.get("NODE_RANK", "0"),
        "--master_addr",
        os.environ.get("MASTER_ADDR", str(runner_cfg.get("master_addr", "127.0.0.1"))),
        "--master_port",
        master_port,
        str(Path(__file__).resolve()),
        "--worker-stage",
        stage,
        "--worker-config",
        str(config_path),
    ]
    env = {
        "CUDA_VISIBLE_DEVICES": gpu,
        "EUMP_TORCH_SHARING_STRATEGY": os.environ.get(
            "EUMP_TORCH_SHARING_STRATEGY",
            str(runner_cfg.get("torch_sharing_strategy", "file_system")),
        ),
    }
    return cmd, env


def _python_command() -> str:
    return sys.executable or "python"


def _traffic_cv_run_name(suite_cfg: dict[str, Any]) -> str:
    traffic_cfg = suite_cfg.get("traffic_cv", {})
    return str(traffic_cfg.get("run_name", TRAFFIC_CV_RUN_NAME_DEFAULT))


def _traffic_cv_results_path(group: str, suite_cfg: dict[str, Any]) -> Path:
    primary = REPO_ROOT / "log" / group / _traffic_cv_run_name(suite_cfg) / "traffic_cv_results.jsonl"
    if primary.exists():
        return primary
    alt_pattern = list((REPO_ROOT / "log" / group).glob("traffic_cv_*/traffic_cv_results.jsonl"))
    if alt_pattern:
        return sorted(alt_pattern)[-1]
    return primary


def _traffic_cv_config_path(run_dir: Path, variant: VariantSpec) -> Path:
    return _config_paths(run_dir, variant, "traffic_cv")


def _traffic_cv_model_config(suite_cfg: dict[str, Any], *, smoke: bool) -> dict[str, Any]:
    traffic_cfg = suite_cfg.get("traffic_cv", {})
    head_cfg = dict(traffic_cfg.get("head", {}))
    if smoke:
        smoke_cfg = suite_cfg.get("smoke", {})
        head_cfg["epochs"] = int(smoke_cfg.get("traffic_cv_epochs", 1))
        head_cfg["batch_size"] = int(smoke_cfg.get("traffic_cv_batch_size", head_cfg.get("batch_size", 64)))
    return {
        "name": str(traffic_cfg.get("model_name", "swin_embed_head_128_d05")),
        "kind": "swin_embedding_head",
        "scalar_hidden_dim": int(head_cfg.get("scalar_hidden_dim", 128)),
        "dropout": float(head_cfg.get("dropout", 0.5)),
        "lr": float(head_cfg.get("lr", 1.0e-4)),
        "weight_decay": float(head_cfg.get("weight_decay", 1.0e-2)),
        "batch_size": int(head_cfg.get("batch_size", 64)),
        "epochs": int(head_cfg.get("epochs", 80)),
        "loss": str(head_cfg.get("loss", "mae")),
        "selection_metric": str(head_cfg.get("selection_metric", "val_loss")),
    }


def _one_stage_cv_model_name(variant: VariantSpec, task: str) -> str:
    return f"one_stage_{variant.name}_{task}"


def _launcher_cv_model_config(variant: VariantSpec, task: str) -> dict[str, Any]:
    pipeline_suite = str(variant.payload.get("pipeline_suite", ""))
    if not pipeline_suite:
        raise ValueError(f"One-stage CV variant {variant.label!r} requires pipeline_suite")
    return {
        "name": _one_stage_cv_model_name(variant, task),
        "kind": "launcher",
        "mode": task,
        "launch": pipeline_suite,
        "overrides": [
            f"experiment.pretrain_model={variant.payload.get('pretrain_model', 'swin_L')}",
            f"++experiment.scratch_backbone={_hydra_scalar(bool(variant.payload.get('scratch_backbone', False)))}",
        ],
    }


def _hydra_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return str(value)
    return str(value)


def _hydra_list(values: list[Any] | tuple[Any, ...]) -> str:
    return "[" + ",".join(_hydra_scalar(value) for value in values) + "]"


def _hydra_dict(values: dict[str, Any]) -> str:
    return "{" + ",".join(f"{_hydra_scalar(key)}:{_hydra_scalar(value)}" for key, value in values.items()) + "}"


def _hydra_replace_dict_overrides(key: str, values: dict[str, Any]) -> list[str]:
    overrides = [f"~{key}"]
    if values:
        overrides.append(f"++{key}={_hydra_dict(values)}")
    return overrides


def _traffic_cv_embedding_overrides(variant: VariantSpec) -> list[str]:
    """Hydra overrides needed to rebuild the variant's pretrain model for export only."""
    pretrain_model = str(variant.payload.get("pretrain_model", "swin_L"))
    scratch_backbone = bool(variant.payload.get("scratch_backbone", False))
    overrides = [
        f"experiment.pretrain_model={pretrain_model}",
        f"++experiment.scratch_backbone={_hydra_scalar(scratch_backbone)}",
    ]

    if "mask_ratio" in variant.payload:
        overrides.append(f"reconstruct.mask_ratio={float(variant.payload['mask_ratio'])}")

    if "center_ratio" in variant.payload:
        center_ratio = float(variant.payload["center_ratio"])
        overrides.extend(
            [
                f"experiment.center_ratio={center_ratio}",
                f"pipeline.center_ratio={center_ratio}",
            ]
        )

    inputs = variant.payload.get("inputs")
    if isinstance(inputs, dict):
        sources = [str(source) for source in _as_list(inputs.get("sources"))]
        optional_sources = [str(source) for source in _as_list(inputs.get("optional_sources"))]
        include_city_mask = bool(inputs.get("include_city_mask_as_input", False))
        source_losses = _source_losses_for_inputs(sources, include_city_mask=include_city_mask)
        source_weights = _source_weights_for_inputs(sources)
        overrides.extend(
            [
                f"pipeline.inputs.sources={_hydra_list(sources)}",
                f"pipeline.include_city_mask_as_input={_hydra_scalar(include_city_mask)}",
                "pipeline.city_mask_source=city_mask",
                "pipeline.loss.aux_weight=0.0",
                "pipeline.targets.raster=[]",
                "++pipeline.targets.scalar=[]",
                "pipeline.targets.crop_scalar=[]",
                f"pipeline.targets.input_reconstruction.scalar_sources={_hydra_list(_scalar_sources_for_inputs(sources))}",
                *_hydra_replace_dict_overrides("pipeline.targets.input_reconstruction.source_losses", source_losses),
                *_hydra_replace_dict_overrides("pipeline.targets.input_reconstruction.source_weights", source_weights),
            ]
        )
        if optional_sources:
            overrides.append(f"++pipeline.inputs.optional_sources={_hydra_list(optional_sources)}")
    elif variant.payload.get("no_aux"):
        overrides.extend(
            [
                "pipeline.loss.aux_weight=0.0",
                "pipeline.targets.raster=[]",
                "pipeline.targets.crop_scalar=[]",
            ]
        )

    return overrides


def compose_and_write_traffic_cv_config(
    *,
    suite_cfg: dict[str, Any],
    variant: VariantSpec,
    overrides: list[str],
    run_dir: Path,
    smoke: bool,
) -> tuple[dict[str, Any], Path]:
    group = _variant_group(str(suite_cfg["experiment_group_prefix"]), variant)
    base_suite = str(suite_cfg["base_pipeline_suite"])
    pipeline_suite = _variant_pipeline_suite(suite_cfg, variant)
    traffic_cfg = suite_cfg.get("traffic_cv", {})
    smoke_cfg = suite_cfg.get("smoke", {})
    folds = int(traffic_cfg.get("folds", 5))
    embedding_batch_size = int(traffic_cfg.get("embedding_batch_size", 64))
    if smoke:
        folds = int(smoke_cfg.get("traffic_cv_folds", min(2, folds)))
        embedding_batch_size = int(smoke_cfg.get("traffic_cv_embedding_batch_size", embedding_batch_size))
    cv_payload: dict[str, Any] = {
        "folds": folds,
        "fold_seed": int(traffic_cfg.get("fold_seed", 42)),
        "experiment_group": group,
        "run_name": _traffic_cv_run_name(suite_cfg),
        "fold_manifest_dir": None,
        "embedding_cache_dir": None,
        "target_pipeline": f"{pipeline_suite}/traffic",
    }
    if _is_one_stage_variant(variant):
        cv_payload["models"] = [
            _launcher_cv_model_config(variant, TRAFFIC_CV_TASK),
        ]
    else:
        years = _as_list(traffic_cfg.get("embedding_years", [2017, 2020]))
        if len(years) != 2:
            raise ValueError("ablation_suite.traffic_cv.embedding_years must contain [START, END]")
        encoder_group = _pretrain_source_group(variant) if _has_pretrain_source(variant) and _model_file(_pretrain_source_group(variant), "pretrain", str(suite_cfg["encoder_checkpoint_file"])).exists() else group
        embedding_pipeline = str(variant.payload.get("pretrain_pipeline", f"{base_suite}/pretrain"))
        cv_payload["embedding"] = {
            "model_name": f"{encoder_group}/pretrain",
            "checkpoint_file": str(suite_cfg["encoder_checkpoint_file"]),
            "checkpoint_source": str(traffic_cfg.get("checkpoint_source", "auto")),
            "pipeline": embedding_pipeline,
            "years": [int(years[0]), int(years[1])],
            "batch_size": embedding_batch_size,
            "workers_per_gpu": int(traffic_cfg.get("workers_per_gpu", 4)),
            "torch_threads": int(traffic_cfg.get("torch_threads", 1)),
            "overwrite": bool(traffic_cfg.get("overwrite_embeddings", False)),
            "overrides": _traffic_cv_embedding_overrides(variant),
        }
        cv_payload["models"] = [
            _traffic_cv_model_config(suite_cfg, smoke=smoke),
        ]
    payload = {"traffic_cv": cv_payload}
    path = _traffic_cv_config_path(run_dir, variant)
    _write_yaml(path, payload)
    return payload, path


def _traffic_cv_command(
    config_path: Path,
    *,
    suite_cfg: dict[str, Any],
    variant: VariantSpec,
    overrides: list[str],
) -> tuple[list[str], dict[str, str]]:
    runner_cfg = suite_cfg.get("runner", {})
    gpu, nproc, _ = _gpu_env("train", runner_cfg)
    if _is_one_stage_variant(variant):
        model_name = _one_stage_cv_model_name(variant, TRAFFIC_CV_TASK)
    else:
        model_name = str(suite_cfg.get("traffic_cv", {}).get("model_name", "swin_embed_head_128_d05"))
    cmd = [
        _python_command(),
        "script/experiments/run_city_scalar_cv.py",
        "--config",
        str(config_path),
        "--models",
        model_name,
        *overrides,
    ]
    env = {
        "GPU": gpu,
        "CUDA_VISIBLE_DEVICES": gpu,
        "NPROC": str(nproc),
        "LOG_PROGRESS_BAR": os.environ.get("LOG_PROGRESS_BAR", "0"),
        "EUMP_TORCH_SHARING_STRATEGY": os.environ.get(
            "EUMP_TORCH_SHARING_STRATEGY",
            str(runner_cfg.get("torch_sharing_strategy", "file_system")),
        ),
    }
    return cmd, env


PM25_CV_RUN_NAME_DEFAULT = "pm25_cv_seed42"


def _pm25_cv_run_name(suite_cfg: dict[str, Any]) -> str:
    pm25_cfg = suite_cfg.get("pm25_cv", {})
    return str(pm25_cfg.get("run_name", PM25_CV_RUN_NAME_DEFAULT))


def _pm25_cv_results_path(group: str, suite_cfg: dict[str, Any]) -> Path:
    primary = REPO_ROOT / "log" / group / _pm25_cv_run_name(suite_cfg) / "pm25_cv_results.jsonl"
    if primary.exists():
        return primary
    alt_pattern = list((REPO_ROOT / "log" / group).glob("pm25_cv_*/pm25_cv_results.jsonl"))
    if alt_pattern:
        return sorted(alt_pattern)[-1]
    return primary


def _pm25_cv_config_path(run_dir: Path, variant: VariantSpec) -> Path:
    return _config_paths(run_dir, variant, "pm25_cv")


def _pm25_cv_model_config(suite_cfg: dict[str, Any], *, smoke: bool) -> dict[str, Any]:
    pm25_cfg = suite_cfg.get("pm25_cv", {})
    head_cfg = dict(pm25_cfg.get("head", {}))
    if smoke:
        smoke_cfg = suite_cfg.get("smoke", {})
        head_cfg["epochs"] = int(smoke_cfg.get("traffic_cv_epochs", 1))
        head_cfg["batch_size"] = int(smoke_cfg.get("traffic_cv_batch_size", head_cfg.get("batch_size", 64)))
    return {
        "name": str(pm25_cfg.get("model_name", "swin_embed_head_128_d05")),
        "kind": "swin_embedding_head",
        "scalar_hidden_dim": int(head_cfg.get("scalar_hidden_dim", 128)),
        "dropout": float(head_cfg.get("dropout", 0.5)),
        "lr": float(head_cfg.get("lr", 1.0e-4)),
        "weight_decay": float(head_cfg.get("weight_decay", 1.0e-2)),
        "batch_size": int(head_cfg.get("batch_size", 64)),
        "epochs": int(head_cfg.get("epochs", 80)),
        "loss": str(head_cfg.get("loss", "mae")),
        "selection_metric": str(head_cfg.get("selection_metric", "val_loss")),
    }


def compose_and_write_pm25_cv_config(
    *,
    suite_cfg: dict[str, Any],
    variant: VariantSpec,
    overrides: list[str],
    run_dir: Path,
    smoke: bool,
) -> tuple[dict[str, Any], Path]:
    group = _variant_group(str(suite_cfg["experiment_group_prefix"]), variant)
    pipeline_suite = _variant_pipeline_suite(suite_cfg, variant)
    pm25_cfg = suite_cfg.get("pm25_cv", {})
    smoke_cfg = suite_cfg.get("smoke", {})
    folds = int(pm25_cfg.get("folds", 5))
    embedding_batch_size = int(pm25_cfg.get("embedding_batch_size", 64))
    if smoke:
        folds = int(smoke_cfg.get("traffic_cv_folds", min(2, folds)))
        embedding_batch_size = int(smoke_cfg.get("traffic_cv_embedding_batch_size", embedding_batch_size))
    cv_payload: dict[str, Any] = {
        "target_name": str(pm25_cfg.get("target_name", "PM25_mean")),
        "output_prefix": "pm25_cv",
        "folds": folds,
        "fold_seed": int(pm25_cfg.get("fold_seed", 42)),
        "experiment_group": group,
        "run_name": _pm25_cv_run_name(suite_cfg),
        "fold_manifest_dir": None,
        "embedding_cache_dir": None,
        "target_pipeline": f"{pipeline_suite}/pm25",
    }
    if _is_one_stage_variant(variant):
        cv_payload["models"] = [
            _launcher_cv_model_config(variant, PM25_CV_TASK),
        ]
    else:
        years = _as_list(pm25_cfg.get("embedding_years", [2015, 2019]))
        if len(years) != 2:
            raise ValueError("ablation_suite.pm25_cv.embedding_years must contain [START, END]")
        encoder_group = _pretrain_source_group(variant) if _has_pretrain_source(variant) and _model_file(_pretrain_source_group(variant), "pretrain", str(suite_cfg["encoder_checkpoint_file"])).exists() else group
        cv_payload["embedding"] = {
            "model_name": f"{encoder_group}/pretrain",
            "checkpoint_file": str(suite_cfg["encoder_checkpoint_file"]),
            "checkpoint_source": str(pm25_cfg.get("checkpoint_source", "auto")),
            "pipeline": str(variant.payload.get("pretrain_pipeline", f"{suite_cfg['base_pipeline_suite']}/pretrain")),
            "years": [int(years[0]), int(years[1])],
            "batch_size": embedding_batch_size,
            "workers_per_gpu": int(pm25_cfg.get("workers_per_gpu", 4)),
            "torch_threads": int(pm25_cfg.get("torch_threads", 1)),
            "overwrite": False,
            "overrides": _traffic_cv_embedding_overrides(variant),
        }
        cv_payload["models"] = [
            _pm25_cv_model_config(suite_cfg, smoke=smoke),
        ]
    payload = {"city_scalar_cv": cv_payload}
    path = _pm25_cv_config_path(run_dir, variant)
    _write_yaml(path, payload)
    return payload, path


def _pm25_cv_command(
    config_path: Path,
    *,
    suite_cfg: dict[str, Any],
    variant: VariantSpec,
    overrides: list[str],
) -> tuple[list[str], dict[str, str]]:
    runner_cfg = suite_cfg.get("runner", {})
    gpu, nproc, _ = _gpu_env("train", runner_cfg)
    if _is_one_stage_variant(variant):
        model_name = _one_stage_cv_model_name(variant, PM25_CV_TASK)
    else:
        model_name = str(suite_cfg.get("pm25_cv", {}).get("model_name", "swin_embed_head_128_d05"))
    cmd = [
        _python_command(),
        "script/experiments/run_city_scalar_cv.py",
        "--config",
        str(config_path),
        "--models",
        model_name,
        *overrides,
    ]
    env = {
        "GPU": gpu,
        "CUDA_VISIBLE_DEVICES": gpu,
        "NPROC": str(nproc),
        "LOG_PROGRESS_BAR": os.environ.get("LOG_PROGRESS_BAR", "0"),
        "EUMP_TORCH_SHARING_STRATEGY": os.environ.get(
            "EUMP_TORCH_SHARING_STRATEGY",
            str(runner_cfg.get("torch_sharing_strategy", "file_system")),
        ),
    }
    return cmd, env


def run_command(cmd: list[str], *, env: dict[str, str], dry_run: bool, commands_log: Path) -> None:
    line = _printable_command(cmd, env=env)
    print(line)
    commands_log.parent.mkdir(parents=True, exist_ok=True)
    with commands_log.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    if dry_run:
        return
    merged_env = os.environ.copy()
    merged_env.update(env)
    subprocess.run(cmd, cwd=REPO_ROOT, env=merged_env, check=True)


def _model_file(group: str, name: str, file_name: str) -> Path:
    return REPO_ROOT / "model" / group / name / file_name


def _eval_metrics_file(group: str, task: str, split: str) -> Path:
    return REPO_ROOT / "model" / group / task / "evaluation" / f"{split}_metrics.json"


def _scene_has_compatible_row(
    reader: ProcessedSceneStoreReader,
    scene_id: str,
    pipeline_spec: PipelineSpec,
) -> bool:
    task_config = pipeline_spec.to_task_config()
    for row in reader.iter_scene_years(scene_ids=[scene_id], year_range=pipeline_spec.year_range):
        if reader.missing_sources_for_task(row["scene_id"], int(row["year"]), task_config):
            continue
        record = reader.load_scene_record(row["scene_id"], int(row["year"]), task_config)
        if _record_has_valid_required_targets(record, pipeline_spec):
            return True
    return False


def _record_has_valid_required_targets(record, pipeline_spec: PipelineSpec) -> bool:
    if pipeline_spec.kind != "eump_downstream":
        return True

    crop_box = sample_random_crop_box(record.spatial_shape, pipeline_spec.crop_size, center_ratio=1.0)
    targets = record.targets.crop(crop_box)
    for target in pipeline_spec.targets:
        if target.target_type == "input_reconstruction":
            continue
        if target.source is None:
            return False

        if target.target_type == "scalar":
            block = targets.aux_scalar.get(target.source) or targets.prediction_scalar.get(target.source)
        elif target.target_type == "raster":
            block = targets.aux_raster.get(target.source) or targets.prediction_raster.get(target.source)
        elif target.target_type == "crop_scalar":
            if target.source_kind == "prediction_raster":
                block = targets.prediction_raster.get(target.source)
            elif target.source_kind in {"aux_raster", None}:
                block = targets.aux_raster.get(target.source)
            elif target.source_kind == "reconstruct_raster":
                block = targets.reconstruct_raster
            else:
                block = None
        else:
            block = None

        if block is not None and bool(block.valid_mask.any()):
            return True
    return False


def _smoke_pipeline_specs(
    *,
    suite_cfg: dict[str, Any],
    variants: list[VariantSpec],
    selected_tasks: list[str],
    overrides: list[str],
) -> list[tuple[str, PipelineSpec]]:
    specs: list[tuple[str, PipelineSpec]] = []
    base_suite = str(suite_cfg["base_pipeline_suite"])
    for variant in variants:
        if _uses_pretrain(variant):
            pretrain_cfg = _compose_config("train", pipeline=f"{base_suite}/pretrain", overrides=overrides)
            pretrain_cfg = apply_variant_config(
                pretrain_cfg,
                suite_cfg=suite_cfg,
                variant=variant,
                task=None,
                is_pretrain=True,
                is_eval=False,
                smoke=False,
                smoke_split_path=None,
            )
            specs.append((f"{variant.label}/pretrain", PipelineSpec.from_mapping(pretrain_cfg["pipeline"])))

        for task in selected_tasks:
            task_pipeline = TASK_PIPELINES[task]
            pipeline_suite = _variant_pipeline_suite(suite_cfg, variant)
            downstream_cfg = _compose_config("train", pipeline=f"{pipeline_suite}/{task_pipeline}", overrides=overrides)
            downstream_cfg = apply_variant_config(
                downstream_cfg,
                suite_cfg=suite_cfg,
                variant=variant,
                task=task,
                is_pretrain=False,
                is_eval=False,
                smoke=False,
                smoke_split_path=None,
            )
            specs.append((f"{variant.label}/{task}", PipelineSpec.from_mapping(downstream_cfg["pipeline"])))
    return specs


def _select_smoke_scene_ids(
    *,
    reader: ProcessedSceneStoreReader,
    candidate_scene_ids: list[str],
    specs: list[tuple[str, PipelineSpec]],
    limit: int,
    split_name: str,
) -> tuple[list[str], dict[str, Any]]:
    selected: list[str] = []
    rejected: list[dict[str, Any]] = []
    for scene_id in candidate_scene_ids:
        incompatible = [
            label
            for label, pipeline_spec in specs
            if not _scene_has_compatible_row(reader, scene_id, pipeline_spec)
        ]
        if incompatible:
            if len(rejected) < 20:
                rejected.append({"scene_id": scene_id, "incompatible": incompatible})
            continue
        selected.append(scene_id)
        if len(selected) >= limit:
            break

    if len(selected) < limit:
        raise ValueError(
            "Unable to build smoke split with enough compatible scene IDs for "
            f"{split_name}: requested={limit} selected={len(selected)}. "
            f"First rejected examples: {rejected}"
        )

    return selected, {
        "candidate_count": len(candidate_scene_ids),
        "requested_count": limit,
        "selected_count": len(selected),
        "rejected_examples": rejected,
    }


def _generate_smoke_split(
    *,
    suite_cfg: dict[str, Any],
    variants: list[VariantSpec],
    selected_tasks: list[str],
    overrides: list[str],
    run_dir: Path,
    dry_run: bool,
) -> Path | None:
    base_cfg = _compose_config("train", pipeline=f"{suite_cfg['base_pipeline_suite']}/pretrain", overrides=overrides)
    data_cfg = base_cfg.get("data", {})
    split_manifest = data_cfg.get("split_manifest_path")
    if not split_manifest:
        return None
    source_path = _resolve_path(split_manifest)
    if not source_path.exists():
        if dry_run:
            return run_dir / "smoke_split_seed42.json"
        raise FileNotFoundError(f"Smoke split source manifest not found: {source_path}")
    eump_root = _resolve_path(data_cfg["eump_root"])
    if dry_run and not eump_root.exists():
        return run_dir / "smoke_split_seed42.json"

    original = load_split_manifest(source_path)
    reader = ProcessedSceneStoreReader(eump_root)
    specs = _smoke_pipeline_specs(
        suite_cfg=suite_cfg,
        variants=variants,
        selected_tasks=selected_tasks,
        overrides=overrides,
    )
    scene_counts = suite_cfg.get("smoke", {}).get("scene_counts", {})
    payload = deepcopy(original)
    smoke_report: dict[str, Any] = {
        "compatibility_specs": [label for label, _ in specs],
        "splits": {},
    }
    for split_name in ("train", "val", "test"):
        key = f"{split_name}_scene_ids"
        ids = [str(value) for value in original.get(key, ())]
        limit = int(scene_counts.get(split_name, len(ids)))
        limit = max(1, min(limit, len(ids)))
        payload[key], smoke_report["splits"][split_name] = _select_smoke_scene_ids(
            reader=reader,
            candidate_scene_ids=ids,
            specs=specs,
            limit=limit,
            split_name=split_name,
        )
    payload["pool_scene_ids"] = sorted(
        {scene_id for split_name in ("train", "val", "test") for scene_id in payload[f"{split_name}_scene_ids"]},
        key=str,
    )
    payload["generator"] = "eump_ablation_suite_smoke_v3"
    payload["source_split_manifest_path"] = str(source_path)
    payload["compatibility_report"] = smoke_report
    output_path = run_dir / "smoke_split_seed42.json"
    _write_json(output_path, payload)
    return output_path


def _config_paths(run_dir: Path, variant: VariantSpec, stem: str) -> Path:
    return run_dir / "configs" / variant.name / f"{stem}.yaml"


def prepare_variant_manifest(run_dir: Path, variants: list[VariantSpec], suite_cfg: dict[str, Any]) -> None:
    rows = []
    for variant in variants:
        rows.append(
            {
                "family": variant.family,
                "variant": variant.name,
                "experiment_group": _variant_group(str(suite_cfg["experiment_group_prefix"]), variant),
                "payload": variant.payload,
            }
        )
    _write_json(run_dir / "variant_manifest.json", rows)


def compose_and_write_pretrain_config(
    *,
    suite_cfg: dict[str, Any],
    variant: VariantSpec,
    overrides: list[str],
    run_dir: Path,
    smoke: bool,
    smoke_split_path: Path | None,
) -> tuple[dict[str, Any], Path]:
    cfg = _compose_config("train", pipeline=f"{suite_cfg['base_pipeline_suite']}/pretrain", overrides=overrides)
    resolved = apply_variant_config(
        cfg,
        suite_cfg=suite_cfg,
        variant=variant,
        task=None,
        is_pretrain=True,
        is_eval=False,
        smoke=smoke,
        smoke_split_path=smoke_split_path,
    )
    _validate_pretrain_aux_policy(
        suite_cfg=suite_cfg,
        variant=variant,
        base_cfg=cfg,
        resolved_cfg=resolved,
    )
    path = _config_paths(run_dir, variant, "train_pretrain")
    _write_yaml(path, resolved)
    return resolved, path


def compose_and_write_downstream_config(
    *,
    suite_cfg: dict[str, Any],
    variant: VariantSpec,
    task: str,
    overrides: list[str],
    run_dir: Path,
    smoke: bool,
    smoke_split_path: Path | None,
) -> tuple[dict[str, Any], Path]:
    pipeline_suite = _variant_pipeline_suite(suite_cfg, variant)
    cfg = _compose_config("train", pipeline=f"{pipeline_suite}/{TASK_PIPELINES[task]}", overrides=overrides)
    resolved = apply_variant_config(
        cfg,
        suite_cfg=suite_cfg,
        variant=variant,
        task=task,
        is_pretrain=False,
        is_eval=False,
        smoke=smoke,
        smoke_split_path=smoke_split_path,
    )
    path = _config_paths(run_dir, variant, f"train_{task}")
    _write_yaml(path, resolved)
    return resolved, path


def compose_and_write_eval_config(
    *,
    suite_cfg: dict[str, Any],
    variant: VariantSpec,
    task: str,
    split: str,
    overrides: list[str],
    run_dir: Path,
    smoke: bool,
    smoke_split_path: Path | None,
) -> tuple[dict[str, Any], Path]:
    pipeline_suite = _variant_pipeline_suite(suite_cfg, variant)
    cfg = _compose_config("evaluate", pipeline=f"{pipeline_suite}/{TASK_PIPELINES[task]}", overrides=overrides)
    resolved = apply_variant_config(
        cfg,
        suite_cfg=suite_cfg,
        variant=variant,
        task=task,
        is_pretrain=False,
        is_eval=True,
        smoke=smoke,
        smoke_split_path=smoke_split_path,
    )
    resolved["evaluation"]["split"] = split
    path = _config_paths(run_dir, variant, f"eval_{task}_{split}")
    _write_yaml(path, resolved)
    return resolved, path


def _metric_rows_from_summary(summary: dict[str, Any], *, variant: VariantSpec) -> list[dict[str, Any]]:
    experiment = summary.get("experiment", {})
    checkpoint = summary.get("checkpoint", {})
    rows: list[dict[str, Any]] = []
    for scope, metrics in summary.get("metrics", {}).items():
        row = {
            "family": variant.family,
            "variant": variant.name,
            "task": experiment.get("name"),
            "fold": "",
            "split": summary.get("split"),
            "target": summary.get("target"),
            "target_kind": summary.get("target_kind"),
            "scope": scope,
            "experiment_group": experiment.get("group"),
            "experiment_name": experiment.get("name"),
            "model_path": str(REPO_ROOT / "model" / str(experiment.get("model_name", ""))),
            "checkpoint_path": checkpoint.get("path"),
        }
        for key in ("rmse", "mae", "mse", "r2", "count"):
            row[key] = metrics.get(key)
        rows.append(row)
    return rows


def _metric_rows_from_traffic_cv(
    *,
    suite_cfg: dict[str, Any],
    variant: VariantSpec,
    eval_splits: list[str],
    group_override: str | None = None,
) -> list[dict[str, Any]]:
    group = group_override if group_override else _variant_group(str(suite_cfg["experiment_group_prefix"]), variant)
    path = _traffic_cv_results_path(group, suite_cfg)
    if not path.exists():
        return []
    allowed_splits = {str(split) for split in eval_splits}
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            split = str(payload.get("split", ""))
            if split not in allowed_splits:
                continue
            model = str(payload.get("model", "swin_embed_head"))
            target = str(payload.get("target", "Traffic"))
            target_kind = str(payload.get("target_kind") or ("one_stage_city_scalar_cv" if payload.get("model_kind") == "launcher" else "swin_embedding_city_scalar"))
            grouped.setdefault((model, split, target, target_kind), []).append(payload)

    rows: list[dict[str, Any]] = []
    for (model, split, target, target_kind), payloads in sorted(grouped.items()):
        folds = {
            str(payload.get("fold"))
            for payload in payloads
            if payload.get("fold") not in (None, "")
        }
        row = {
            "family": variant.family,
            "variant": variant.name,
            "task": TRAFFIC_CV_TASK,
            "fold": "kfold_mean",
            "fold_count": len(folds) if folds else len(payloads),
            "split": split,
            "target": target,
            "target_kind": target_kind,
            "scope": "global",
            "experiment_group": group,
            "experiment_name": f"{model}_kfold_mean",
            "model_path": "",
            "checkpoint_path": "",
        }
        for key in ("rmse", "mae", "mse", "r2", "count"):
            values = [float(payload[key]) for payload in payloads if payload.get(key) is not None]
            row[key] = _mean(values)
            row[f"{key}_std"] = _std(values)
        rows.append(row)
    return rows


def _metric_rows_from_pm25_cv(
    *,
    suite_cfg: dict[str, Any],
    variant: VariantSpec,
    eval_splits: list[str],
    group_override: str | None = None,
) -> list[dict[str, Any]]:
    group = group_override if group_override else _variant_group(str(suite_cfg["experiment_group_prefix"]), variant)
    path = _pm25_cv_results_path(group, suite_cfg)
    if not path.exists():
        return []
    allowed_splits = {str(split) for split in eval_splits}
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            split = str(payload.get("split", ""))
            if split not in allowed_splits:
                continue
            model = str(payload.get("model", "swin_embed_head"))
            target = str(payload.get("target", "PM25_mean"))
            target_kind = str(payload.get("target_kind") or ("one_stage_city_scalar_cv" if payload.get("model_kind") == "launcher" else "swin_embedding_city_scalar"))
            grouped.setdefault((model, split, target, target_kind), []).append(payload)

    rows: list[dict[str, Any]] = []
    for (model, split, target, target_kind), payloads in sorted(grouped.items()):
        folds = {
            str(payload.get("fold"))
            for payload in payloads
            if payload.get("fold") not in (None, "")
        }
        row = {
            "family": variant.family,
            "variant": variant.name,
            "task": PM25_CV_TASK,
            "fold": "kfold_mean",
            "fold_count": len(folds) if folds else len(payloads),
            "split": split,
            "target": target,
            "target_kind": target_kind,
            "scope": "global",
            "experiment_group": group,
            "experiment_name": f"{model}_kfold_mean",
            "model_path": "",
            "checkpoint_path": "",
        }
        for key in ("rmse", "mae", "mse", "r2", "count"):
            values = [float(payload[key]) for payload in payloads if payload.get(key) is not None]
            row[key] = _mean(values)
            row[f"{key}_std"] = _std(values)
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...] | list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def collect_results(
    *,
    suite_cfg: dict[str, Any],
    variants: list[VariantSpec],
    tasks: list[str],
    eval_splits: list[str],
    run_dir: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in variants:
        if _is_reference_variant(variant):
            group = _reference_group(variant)
        else:
            group = _variant_group(str(suite_cfg["experiment_group_prefix"]), variant)
        for task in tasks:
            if _uses_traffic_cv(variant, task):
                rows.extend(_metric_rows_from_traffic_cv(suite_cfg=suite_cfg, variant=variant, eval_splits=eval_splits, group_override=group if _is_reference_variant(variant) else None))
                continue
            if _uses_pm25_cv(variant, task):
                rows.extend(_metric_rows_from_pm25_cv(suite_cfg=suite_cfg, variant=variant, eval_splits=eval_splits, group_override=group if _is_reference_variant(variant) else None))
                continue
            for split in eval_splits:
                metrics_path = _eval_metrics_file(group, task, split)
                if not metrics_path.exists():
                    continue
                summary = json.loads(metrics_path.read_text(encoding="utf-8"))
                rows.extend(_metric_rows_from_summary(summary, variant=variant))

    results_path = run_dir / "results.jsonl"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    _write_csv(run_dir / "summary.csv", rows, METRIC_COLUMNS)
    _write_family_summary(run_dir / "summary_by_family.csv", rows)
    return rows


def _mean(values: list[float]) -> float | None:
    return None if not values else float(statistics.mean(values))


def _std(values: list[float]) -> float | None:
    return None if len(values) < 2 else float(statistics.stdev(values))


def _write_family_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["family"]), str(row["task"]), str(row["split"]), str(row["scope"]))
        grouped.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for (family, task, split, scope), group_rows in sorted(grouped.items()):
        rmse_values = [float(row["rmse"]) for row in group_rows if row.get("rmse") is not None]
        r2_values = [float(row["r2"]) for row in group_rows if row.get("r2") is not None]
        output.append(
            {
                "family": family,
                "task": task,
                "split": split,
                "scope": scope,
                "variant_count": len({row["variant"] for row in group_rows}),
                "row_count": len(group_rows),
                "rmse_mean": _mean(rmse_values),
                "rmse_std": _std(rmse_values),
                "r2_mean": _mean(r2_values),
                "r2_std": _std(r2_values),
            }
        )
    _write_csv(
        path,
        output,
        ["family", "task", "split", "scope", "variant_count", "row_count", "rmse_mean", "rmse_std", "r2_mean", "r2_std"],
    )


def worker_main(args: argparse.Namespace) -> int:
    if args.worker_config is None:
        raise ValueError("--worker-config is required in worker mode")
    payload = OmegaConf.to_container(OmegaConf.load(args.worker_config), resolve=True)
    if args.worker_stage == "train":
        run_training(payload, source="ablation_suite")
        return 0
    if args.worker_stage == "eval":
        run_evaluation(payload, source="ablation_suite")
        return 0
    raise ValueError(f"Unsupported worker stage: {args.worker_stage!r}")


def run_suite(args: argparse.Namespace) -> int:
    suite_cfg = _load_suite_config(args.config)
    selected_families = _selected_values(_as_list(suite_cfg.get("default_families")), args.families)
    selected_variants = _comma_list(args.variants)
    selected_tasks = _selected_tasks(suite_cfg, selected_families=selected_families, selected=args.tasks)
    eval_splits = _selected_values(_as_list(suite_cfg.get("default_eval_splits")), args.eval_splits)
    stages = set(_selected_values(["train", "eval", "collect"], args.stages))
    invalid_stages = sorted(stages - {"train", "eval", "collect"})
    if invalid_stages:
        raise ValueError(f"Unsupported stage(s): {', '.join(invalid_stages)}")
    invalid_tasks = sorted(set(selected_tasks) - set(TASK_PIPELINES))
    if invalid_tasks:
        raise ValueError(f"Unsupported downstream task(s): {', '.join(invalid_tasks)}")

    variants = build_variants(
        suite_cfg,
        selected_families=selected_families,
        selected_variants=selected_variants,
    )
    run_name = str(suite_cfg["run_name"])
    if args.smoke:
        run_name = f"{run_name}_{suite_cfg.get('smoke', {}).get('run_name_suffix', 'smoke')}"
    run_dir = EXPERIMENT_DIR / "outputs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    dry_run = bool(args.dry_run or _truthy(os.environ.get("DRY_RUN")))
    overrides = _merge_overrides(_as_list(suite_cfg.get("data_overrides")), list(args.overrides))
    smoke_split_path = _generate_smoke_split(
        suite_cfg=suite_cfg,
        variants=variants,
        selected_tasks=selected_tasks,
        overrides=overrides,
        run_dir=run_dir,
        dry_run=dry_run,
    ) if args.smoke else None

    if not args.continue_run:
        for path in (run_dir / "commands.log", run_dir / "results.jsonl", run_dir / "summary.csv", run_dir / "summary_by_family.csv"):
            path.write_text("", encoding="utf-8")

    runtime_config = {
        "config": str(args.config),
        "run_name": run_name,
        "families": selected_families,
        "variants": [variant.label for variant in variants],
        "tasks": selected_tasks,
        "eval_splits": eval_splits,
        "stages": sorted(stages),
        "smoke": bool(args.smoke),
        "overrides": overrides,
    }
    _write_yaml(run_dir / "runtime_config.yaml", runtime_config)
    prepare_variant_manifest(run_dir, variants, suite_cfg)

    commands_log = run_dir / "commands.log"
    for variant in variants:
        group = _variant_group(str(suite_cfg["experiment_group_prefix"]), variant)
        if _is_reference_variant(variant):
            ref = _reference_group(variant)
            if "train" in stages:
                print(f"Reference variant={variant.label} -> {ref}: skip training")
            if "eval" in stages:
                print(f"Reference variant={variant.label} -> {ref}: skip evaluation")
            continue
        if "train" in stages:
            if _uses_pretrain(variant) and not _has_pretrain_source(variant):
                _, pretrain_config_path = compose_and_write_pretrain_config(
                    suite_cfg=suite_cfg,
                    variant=variant,
                    overrides=overrides,
                    run_dir=run_dir,
                    smoke=args.smoke,
                    smoke_split_path=smoke_split_path,
                )
                pretrain_checkpoint = _model_file(group, "pretrain", str(suite_cfg["encoder_checkpoint_file"]))
                if args.skip_existing and pretrain_checkpoint.exists():
                    print(f"Skip existing pretrain variant={variant.label}: {pretrain_checkpoint}")
                else:
                    cmd, env = _worker_command("train", pretrain_config_path, suite_cfg=suite_cfg)
                    run_command(cmd, env=env, dry_run=dry_run, commands_log=commands_log)
            elif _has_pretrain_source(variant):
                source_group = _pretrain_source_group(variant)
                source_checkpoint = _model_file(source_group, "pretrain", str(suite_cfg["encoder_checkpoint_file"]))
                if source_checkpoint.exists() or dry_run:
                    print(f"Pretrain source variant={variant.label} -> {source_group}: reuse existing checkpoint")
                else:
                    print(f"Pretrain source variant={variant.label}: {source_group} checkpoint not found, training from scratch")
                    _, pretrain_config_path = compose_and_write_pretrain_config(
                        suite_cfg=suite_cfg,
                        variant=variant,
                        overrides=overrides,
                        run_dir=run_dir,
                        smoke=args.smoke,
                        smoke_split_path=smoke_split_path,
                    )
                    pretrain_checkpoint = _model_file(group, "pretrain", str(suite_cfg["encoder_checkpoint_file"]))
                    if args.skip_existing and pretrain_checkpoint.exists():
                        print(f"Skip existing pretrain variant={variant.label}: {pretrain_checkpoint}")
                    else:
                        cmd, env = _worker_command("train", pretrain_config_path, suite_cfg=suite_cfg)
                        run_command(cmd, env=env, dry_run=dry_run, commands_log=commands_log)

            for task in selected_tasks:
                if _uses_traffic_cv(variant, task):
                    _, traffic_cv_config_path = compose_and_write_traffic_cv_config(
                        suite_cfg=suite_cfg,
                        variant=variant,
                        overrides=overrides,
                        run_dir=run_dir,
                        smoke=args.smoke,
                    )
                    results_path = _traffic_cv_results_path(group, suite_cfg)
                    if args.skip_existing and results_path.exists():
                        print(f"Skip existing Traffic CV variant={variant.label}: {results_path}")
                        continue
                    cmd, env = _traffic_cv_command(
                        traffic_cv_config_path,
                        suite_cfg=suite_cfg,
                        variant=variant,
                        overrides=overrides,
                    )
                    run_command(cmd, env=env, dry_run=dry_run, commands_log=commands_log)
                    continue
                if _uses_pm25_cv(variant, task):
                    _, pm25_cv_config_path = compose_and_write_pm25_cv_config(
                        suite_cfg=suite_cfg,
                        variant=variant,
                        overrides=overrides,
                        run_dir=run_dir,
                        smoke=args.smoke,
                    )
                    results_path = _pm25_cv_results_path(group, suite_cfg)
                    if args.skip_existing and results_path.exists():
                        print(f"Skip existing PM2.5 CV variant={variant.label}: {results_path}")
                        continue
                    cmd, env = _pm25_cv_command(
                        pm25_cv_config_path,
                        suite_cfg=suite_cfg,
                        variant=variant,
                        overrides=overrides,
                    )
                    run_command(cmd, env=env, dry_run=dry_run, commands_log=commands_log)
                    continue
                _, downstream_config_path = compose_and_write_downstream_config(
                    suite_cfg=suite_cfg,
                    variant=variant,
                    task=task,
                    overrides=overrides,
                    run_dir=run_dir,
                    smoke=args.smoke,
                    smoke_split_path=smoke_split_path,
                )
                model_path = _model_file(group, task, "eump_downstream_model.pt")
                if args.skip_existing and model_path.exists():
                    print(f"Skip existing downstream variant={variant.label} task={task}: {model_path}")
                    continue
                cmd, env = _worker_command("train", downstream_config_path, suite_cfg=suite_cfg)
                run_command(cmd, env=env, dry_run=dry_run, commands_log=commands_log)

        if "eval" in stages:
            for task in selected_tasks:
                if _uses_traffic_cv(variant, task):
                    continue
                if _uses_pm25_cv(variant, task):
                    continue
                for split in eval_splits:
                    _, eval_config_path = compose_and_write_eval_config(
                        suite_cfg=suite_cfg,
                        variant=variant,
                        task=task,
                        split=split,
                        overrides=overrides,
                        run_dir=run_dir,
                        smoke=args.smoke,
                        smoke_split_path=smoke_split_path,
                    )
                    metrics_path = _eval_metrics_file(group, task, split)
                    if args.skip_existing and metrics_path.exists():
                        print(f"Skip existing eval variant={variant.label} task={task} split={split}: {metrics_path}")
                        continue
                    cmd, env = _worker_command("eval", eval_config_path, suite_cfg=suite_cfg)
                    run_command(cmd, env=env, dry_run=dry_run, commands_log=commands_log)

    if "collect" in stages:
        rows = collect_results(
            suite_cfg=suite_cfg,
            variants=variants,
            tasks=selected_tasks,
            eval_splits=eval_splits,
            run_dir=run_dir,
        )
        print(f"Collected {len(rows)} metric row(s): {run_dir / 'summary.csv'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker_stage is not None:
        return worker_main(args)
    return run_suite(args)


if __name__ == "__main__":
    raise SystemExit(main())
