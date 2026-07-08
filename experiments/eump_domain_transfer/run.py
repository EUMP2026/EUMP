#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CONFIG_DIR = REPO_ROOT / "configs"
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = EXPERIMENT_DIR / "config.yaml"
DEFAULT_RUN_NAME = "eump_domain_transfer_mae_v3_city_classes_multi_transfer"
DEFAULT_TASKS = ("lst", "precip", "pm25", "traffic")
SUPPORTED_MODELS = ("swin", "scalar_mlp", "aef_mlp")
REQUIRED_LABEL_COLUMNS = ("ID",)
RESULT_FIELDS = (
    "model",
    "scenario",
    "train_domain",
    "eval_domain",
    "task",
    "target",
    "scope",
    "rmse",
    "mae",
    "r2",
    "count",
    "city_count",
    "row_count",
    "year_range",
    "model_path",
    "status",
    "message",
)


@dataclass(frozen=True, slots=True)
class ScenarioSpec:
    name: str
    column: str
    source_values: tuple[str, ...]
    targets: tuple[str, ...]
    source_name: str | None = None

    @property
    def source(self) -> str:
        if self.source_name is not None and str(self.source_name).strip():
            return str(self.source_name).strip()
        return "+".join(self.source_values)


@dataclass(frozen=True, slots=True)
class DomainLoaders:
    loader: Any
    rows: tuple[dict[str, Any], ...]
    scene_ids: tuple[str, ...]

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def city_count(self) -> int:
        return len({str(row["scene_id"]) for row in self.rows})


@dataclass(frozen=True, slots=True)
class ScalarDomainLoaders:
    loader: Any
    scene_ids: tuple[str, ...]
    row_count_value: int

    @property
    def row_count(self) -> int:
        return int(self.row_count_value)

    @property
    def city_count(self) -> int:
        return len(set(self.scene_ids))


class RankShardSampler:
    def __init__(self, length: int, *, rank: int, world_size: int) -> None:
        self.length = int(length)
        self.rank = int(rank)
        self.world_size = max(1, int(world_size))

    def __iter__(self):
        return iter(range(self.rank, self.length, self.world_size))

    def __len__(self) -> int:
        if self.rank >= self.length:
            return 0
        return (self.length - 1 - self.rank) // self.world_size + 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EUMP domain-transfer experiments.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--models", default=None, help="Comma-separated model names: swin,scalar_mlp,aef_mlp.")
    parser.add_argument("--tasks", default=None, help="Comma-separated task names, e.g. lst,pm25,traffic.")
    parser.add_argument("--scenarios", default=None, help="Comma-separated scenario names.")
    parser.add_argument(
        "--skip-pretrain",
        action="store_true",
        help="When pretrain.enabled=true, reuse existing scenario pretrain checkpoints.",
    )
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate existing downstream final models.")
    parser.add_argument("--continue-run", action="store_true", help="Keep existing results.jsonl/summary.csv and upsert rows.")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--stage", choices=("pretrain", "downstream"), default="downstream", help=argparse.SUPPRESS)
    parser.add_argument("--model-kind", choices=SUPPORTED_MODELS, default="swin", help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--scenario", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--task", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--train-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("overrides", nargs="*", help="Hydra overrides forwarded to runnable pipeline configs.")
    return parser.parse_args(argv)


def normalize_scene_id(value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("Empty scene ID")
    try:
        numeric = float(text)
    except ValueError:
        return text
    if numeric.is_integer():
        return str(int(numeric))
    return text


def natural_scene_sort_key(value: Any) -> tuple[int, int | str]:
    text = normalize_scene_id(value)
    try:
        return 0, int(text)
    except ValueError:
        return 1, text


def sorted_scene_ids(values: Iterable[Any]) -> tuple[str, ...]:
    return tuple(sorted((normalize_scene_id(value) for value in values), key=natural_scene_sort_key))


def comma_filter(value: str | None) -> set[str] | None:
    if value is None or not str(value).strip():
        return None
    return {item.strip() for item in str(value).split(",") if item.strip()}


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    return [value]


def repo_path(value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (REPO_ROOT / path)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def required_label_columns_for_scenarios(scenarios: dict[str, ScenarioSpec]) -> tuple[str, ...]:
    columns = {"ID"}
    for scenario in scenarios.values():
        if not scenario.column:
            raise ValueError(f"Scenario {scenario.name!r} must define a non-empty label CSV column")
        columns.add(str(scenario.column))
    return tuple(sorted(columns))


def load_domain_labels(
    path: Path,
    *,
    required_columns: Iterable[str] = REQUIRED_LABEL_COLUMNS,
) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Domain label CSV has no header: {path}")
        required = tuple(str(column) for column in required_columns)
        missing = [column for column in required if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"Domain label CSV is missing required column(s): {', '.join(missing)}")
        rows = list(reader)

    labels: dict[str, dict[str, str]] = {}
    duplicates: list[str] = []
    for row in rows:
        scene_id = normalize_scene_id(row["ID"])
        if scene_id in labels:
            duplicates.append(scene_id)
        labels[scene_id] = {key: str(value).strip() for key, value in row.items()}

    if duplicates:
        preview = ", ".join(sorted(set(duplicates), key=natural_scene_sort_key)[:20])
        raise ValueError(f"Domain label CSV contains duplicate ID value(s): {preview}")
    return labels


def scenario_specs_from_config(config: dict[str, Any]) -> dict[str, ScenarioSpec]:
    raw = config.get("scenarios")
    if not isinstance(raw, dict) or not raw:
        raise ValueError("eump_domain_transfer.scenarios must be a non-empty mapping")
    specs: dict[str, ScenarioSpec] = {}
    for name, payload in raw.items():
        if not isinstance(payload, dict):
            raise TypeError(f"Scenario {name!r} must be a mapping")
        source_values = tuple(str(item).strip() for item in as_list(payload.get("source")) if str(item).strip())
        if not source_values:
            raise ValueError(f"Scenario {name!r} must define at least one source domain")
        targets = tuple(str(item).strip() for item in as_list(payload.get("targets")) if str(item).strip())
        if not targets:
            raise ValueError(f"Scenario {name!r} must define at least one target domain")
        column = str(payload.get("column", "")).strip()
        if not column:
            raise ValueError(f"Scenario {name!r} must define a non-empty label CSV column")
        specs[str(name)] = ScenarioSpec(
            name=str(name),
            column=column,
            source_values=source_values,
            targets=targets,
            source_name=None if payload.get("source_name") is None else str(payload.get("source_name")),
        )
    return specs


def build_domain_manifest(
    *,
    labels: dict[str, dict[str, str]],
    available_scene_ids: Iterable[Any],
    scenarios: dict[str, ScenarioSpec],
) -> tuple[dict[str, Any], dict[str, Any]]:
    available = set(sorted_scene_ids(available_scene_ids))
    labeled = set(labels)
    usable = available & labeled
    manifest: dict[str, Any] = {
        "label_csv_row_count": len(labels),
        "available_scene_count": len(available),
        "usable_scene_count": len(usable),
        "scenarios": {},
    }
    report: dict[str, Any] = {
        "label_scene_count": len(labeled),
        "available_scene_count": len(available),
        "usable_scene_count": len(usable),
        "label_not_in_processed": list(sorted_scene_ids(labeled - available)),
        "processed_not_in_label": list(sorted_scene_ids(available - labeled)),
        "columns": {},
    }

    for scenario in scenarios.values():
        source_values = set(scenario.source_values)
        source_ids = [
            scene_id
            for scene_id in usable
            if labels[scene_id].get(scenario.column, "") in source_values
        ]
        if not source_ids:
            raise ValueError(
                f"Scenario {scenario.name!r} source domain {scenario.column} in "
                f"{tuple(scenario.source_values)!r} "
                "has no usable processed scene IDs"
            )
        target_payload: dict[str, Any] = {}
        for target in scenario.targets:
            target_ids = [
                scene_id
                for scene_id in usable
                if labels[scene_id].get(scenario.column, "") == target
            ]
            if not target_ids:
                raise ValueError(
                    f"Scenario {scenario.name!r} target domain {scenario.column}={target!r} "
                    "has no usable processed scene IDs"
                )
            target_payload[target] = {
                "scene_ids": list(sorted_scene_ids(target_ids)),
                "scene_count": len(target_ids),
            }

        values: dict[str, int] = {}
        for scene_id in usable:
            value = labels[scene_id].get(scenario.column, "")
            values[value] = values.get(value, 0) + 1
        report["columns"][scenario.column] = dict(sorted(values.items()))
        manifest["scenarios"][scenario.name] = {
            "column": scenario.column,
            "source": scenario.source,
            "source_values": list(scenario.source_values),
            "source_name": scenario.source_name,
            "source_scene_ids": list(sorted_scene_ids(source_ids)),
            "source_scene_count": len(source_ids),
            "targets": target_payload,
        }
    return manifest, report


def split_overrides_for_config(overrides: list[str]) -> tuple[list[str], list[str]]:
    script_roots = {
        "run_name",
        "label_csv",
        "output_root",
        "pipeline_suite",
        "model_suites",
        "models",
        "tasks",
        "source_checkpoint",
        "metric_scale",
        "downstream_year_range",
        "ddp_timeout_seconds",
        "model_init_stagger_seconds",
        "loader",
        "baselines",
        "pretrain",
        "training",
        "scenarios",
    }
    script_overrides: list[str] = []
    hydra_overrides: list[str] = []
    for override in overrides:
        key = override.split("=", 1)[0].lstrip("+")
        root = key.split(".", 1)[0]
        if root == "eump_domain_transfer":
            script_overrides.append(override)
        elif root in script_roots:
            script_overrides.append(f"eump_domain_transfer.{override}")
        else:
            hydra_overrides.append(override)
    return script_overrides, hydra_overrides


def load_experiment_config(path: Path, overrides: list[str]) -> tuple[dict[str, Any], list[str]]:
    try:
        from omegaconf import OmegaConf
    except ModuleNotFoundError as exc:
        raise RuntimeError("omegaconf is required to run this experiment") from exc

    path = path if path.is_absolute() else (REPO_ROOT / path)
    script_overrides, hydra_overrides = split_overrides_for_config(overrides)
    loaded = OmegaConf.load(path)
    if "eump_domain_transfer" not in loaded:
        loaded = OmegaConf.create({"eump_domain_transfer": loaded})
    if script_overrides:
        loaded = OmegaConf.merge(loaded, OmegaConf.from_dotlist(script_overrides))
    config = OmegaConf.to_container(loaded["eump_domain_transfer"], resolve=True)
    if not isinstance(config, dict):
        raise TypeError(f"eump_domain_transfer config must be a mapping: {path}")
    return config, hydra_overrides


def compose_runtime_config(*, pipeline: str, overrides: list[str]) -> dict[str, Any]:
    try:
        from hydra import compose, initialize_config_dir
    except ModuleNotFoundError as exc:
        raise RuntimeError("hydra-core is required to compose EUMP runtime configs") from exc
    from urban_embedding.training.runner import normalize_runtime_config

    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="train", overrides=[f"pipeline={pipeline}", *overrides])
    return normalize_runtime_config(cfg)


def apply_loader_overrides(runtime_config: dict[str, Any], experiment_config: dict[str, Any]) -> None:
    loader_config = experiment_config.get("loader")
    if not isinstance(loader_config, dict):
        return
    pipeline_config = runtime_config.setdefault("pipeline", {})
    for key in ("num_workers", "pin_memory", "persistent_workers", "prefetch_factor"):
        if key in loader_config and loader_config[key] is not None:
            pipeline_config[key] = loader_config[key]


def apply_downstream_year_range(runtime_config: dict[str, Any], experiment_config: dict[str, Any]) -> None:
    value = experiment_config.get("downstream_year_range")
    if value is None:
        return
    year_range = tuple(int(item) for item in as_list(value))
    if len(year_range) != 2:
        raise ValueError("eump_domain_transfer.downstream_year_range must contain exactly two years")
    start_year, end_year = year_range
    if start_year > end_year:
        raise ValueError(
            "eump_domain_transfer.downstream_year_range start year must be <= end year: "
            f"{start_year}>{end_year}"
        )
    runtime_config.setdefault("pipeline", {})["year_range"] = [start_year, end_year]
    city_scalar_cache = runtime_config["pipeline"].get("city_scalar_cache")
    if isinstance(city_scalar_cache, dict):
        feature_source = city_scalar_cache.get("feature_source")
        if isinstance(feature_source, dict) and feature_source.get("kind") == "aef_csv":
            feature_source["year_range"] = [start_year, end_year]


def override_key(override: str) -> str:
    key = str(override).split("=", 1)[0].strip()
    return key.lstrip("+~")


def has_override_key(overrides: Iterable[str], key: str, *, script_key: bool = False) -> bool:
    candidates = {key}
    if script_key:
        candidates.add(f"eump_domain_transfer.{key}")
    for override in overrides:
        current = override_key(override)
        if current in candidates:
            return True
    return False


def has_override_prefix(overrides: Iterable[str], key: str, *, script_key: bool = False) -> bool:
    candidates = {key}
    if script_key:
        candidates.add(f"eump_domain_transfer.{key}")
    for override in overrides:
        current = override_key(override)
        for candidate in candidates:
            if current == candidate or current.startswith(f"{candidate}."):
                return True
    return False


def config_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def apply_transfer_training_overrides(
    runtime_config: dict[str, Any],
    experiment_config: dict[str, Any],
    *,
    model_kind: str,
    task: str,
    overrides: Iterable[str],
) -> dict[str, Any]:
    training_cfg = config_mapping(experiment_config.get("training", {}))
    task_epochs = config_mapping(training_cfg.get("task_epochs", {}))
    model_task_epochs = config_mapping(training_cfg.get("model_task_epochs", {}))
    model_epochs = config_mapping(model_task_epochs.get(model_kind, {}))
    task_dropout = config_mapping(training_cfg.get("task_dropout", {}))

    explicit_global_epochs = has_override_key(overrides, "training.epochs", script_key=True)
    explicit_task_epochs = has_override_prefix(overrides, "training.task_epochs", script_key=True)
    explicit_model_task_epochs = has_override_prefix(overrides, "training.model_task_epochs", script_key=True)
    if explicit_global_epochs and not explicit_task_epochs and not explicit_model_task_epochs:
        epochs = training_cfg.get("epochs", 100)
    elif explicit_task_epochs and not explicit_model_task_epochs and task in task_epochs:
        epochs = task_epochs[task]
    elif task in model_epochs:
        epochs = model_epochs[task]
    elif task in task_epochs:
        epochs = task_epochs[task]
    else:
        epochs = training_cfg.get("epochs", 100)

    downstream_cfg = runtime_config["downstream"]
    downstream_cfg["epochs"] = int(epochs)
    downstream_cfg["lr"] = float(training_cfg.get("lr", 1.0e-4))
    downstream_cfg["batch_size"] = int(training_cfg.get("batch_size", 64))
    downstream_cfg["weight_decay"] = float(training_cfg.get("weight_decay", 1.0e-2))
    downstream_cfg["accumulation_steps"] = int(training_cfg.get("accumulation_steps", 1))
    downstream_cfg["scheduler"] = dict(training_cfg.get("scheduler", downstream_cfg.get("scheduler", {})))

    explicit_model_dropout = has_override_prefix(overrides, "model.dropout")
    explicit_task_dropout = has_override_prefix(overrides, "training.task_dropout", script_key=True)
    if task in task_dropout and (explicit_task_dropout or not explicit_model_dropout):
        runtime_config.setdefault("model", {})["dropout"] = float(task_dropout[task])

    return training_cfg


def available_scene_ids_from_runtime_config(runtime_config: dict[str, Any]) -> tuple[str, ...]:
    from urban_embedding.data.processed.reader import ProcessedSceneStoreReader

    reader = ProcessedSceneStoreReader(Path(str(runtime_config["data"]["eump_root"])))
    return sorted_scene_ids(reader.scene_index.keys())


def selected_tasks(config: dict[str, Any], value: str | None) -> tuple[str, ...]:
    configured = tuple(str(item) for item in as_list(config.get("tasks", DEFAULT_TASKS)))
    unsupported_configured = sorted(set(configured) - set(DEFAULT_TASKS))
    if unsupported_configured:
        raise ValueError(
            "EUMP domain transfer currently supports only "
            f"{', '.join(DEFAULT_TASKS)}; unsupported configured task(s): {', '.join(unsupported_configured)}"
        )
    requested = comma_filter(value)
    if requested is None:
        return configured
    unknown = sorted(requested - set(configured))
    if unknown:
        raise ValueError(f"Unsupported task(s): {', '.join(unknown)}. Configured tasks: {', '.join(configured)}")
    return tuple(task for task in configured if task in requested)


def selected_models(config: dict[str, Any], value: str | None) -> tuple[str, ...]:
    configured = tuple(str(item) for item in as_list(config.get("models", ("swin",))))
    unsupported_configured = sorted(set(configured) - set(SUPPORTED_MODELS))
    if unsupported_configured:
        raise ValueError(
            "EUMP domain transfer supports model(s) "
            f"{', '.join(SUPPORTED_MODELS)}; unsupported configured model(s): "
            f"{', '.join(unsupported_configured)}"
        )
    requested = comma_filter(value)
    if requested is None:
        return configured
    unknown = sorted(requested - set(SUPPORTED_MODELS))
    if unknown:
        raise ValueError(f"Unsupported model(s): {', '.join(unknown)}. Supported models: {', '.join(SUPPORTED_MODELS)}")
    return tuple(model for model in SUPPORTED_MODELS if model in requested)


def selected_scenarios(scenarios: dict[str, ScenarioSpec], value: str | None) -> tuple[str, ...]:
    requested = comma_filter(value)
    names = tuple(scenarios)
    if requested is None:
        return names
    unknown = sorted(requested - set(names))
    if unknown:
        raise ValueError(f"Unsupported scenario(s): {', '.join(unknown)}. Configured scenarios: {', '.join(names)}")
    return tuple(name for name in names if name in requested)


def pipeline_suite_for_model(config: dict[str, Any], model_kind: str) -> str:
    if model_kind == "swin":
        return str(config.get("pipeline_suite", "eump_full/mae_v3")).strip().rstrip("/")
    suites = config.get("model_suites", {})
    if not isinstance(suites, dict):
        suites = {}
    default = "eump_full/scalar_mlp" if model_kind == "scalar_mlp" else "eump_full/aef_mlp"
    return str(suites.get(model_kind, default)).strip().rstrip("/")


def printable_command(cmd: list[str], env: dict[str, str] | None = None) -> str:
    prefix = [f"{key}={value}" for key, value in (env or {}).items()]
    return "+ " + " ".join(shlex.quote(part) for part in [*prefix, *cmd])


def run_command(cmd: list[str], *, env: dict[str, str], dry_run: bool, commands_log: Path) -> None:
    line = printable_command(cmd, env=env)
    print(line)
    commands_log.parent.mkdir(parents=True, exist_ok=True)
    with commands_log.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    if dry_run:
        return
    merged_env = os.environ.copy()
    merged_env.update(env)
    subprocess.run(cmd, cwd=REPO_ROOT, env=merged_env, check=True)


def configure_torch_multiprocessing(logger: Any | None = None, *, rank: int = 0) -> None:
    requested_strategy = os.environ.get("EUMP_TORCH_SHARING_STRATEGY")
    if not requested_strategy:
        return

    import torch.multiprocessing as torch_mp

    available_strategies = set(torch_mp.get_all_sharing_strategies())
    if requested_strategy not in available_strategies:
        available = ", ".join(sorted(available_strategies))
        raise ValueError(
            f"Unsupported EUMP_TORCH_SHARING_STRATEGY={requested_strategy!r}. "
            f"Available strategies: {available}"
        )

    current_strategy = torch_mp.get_sharing_strategy()
    if current_strategy != requested_strategy:
        torch_mp.set_sharing_strategy(requested_strategy)

    if int(rank) == 0 and logger is not None:
        logger.info(f"Use torch multiprocessing sharing strategy: {requested_strategy}")


def first_gpu_id(value: str | None) -> str | None:
    if value is None:
        return None
    for item in str(value).split(","):
        item = item.strip()
        if item:
            return item
    return None


def worker_command(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    scenario_name: str,
    stage: str,
    model_kind: str = "swin",
    task: str | None = None,
    resource_profile: str = "train",
    train_only: bool = False,
    eval_only: bool = False,
) -> tuple[list[str], dict[str, str]]:
    if resource_profile not in {"train", "eval"}:
        raise ValueError(f"Unsupported worker resource profile: {resource_profile!r}")
    is_eval_profile = resource_profile == "eval"
    nproc_key = "EVAL_NPROC" if is_eval_profile else "NPROC"
    default_nproc = "1" if is_eval_profile else "8"
    nproc = int(os.environ.get(nproc_key, default_nproc))
    script = str(Path(__file__).resolve())
    base_args = [
        script,
        "--worker",
        "--config",
        str(args.config),
        "--run-dir",
        str(run_dir),
        "--scenario",
        scenario_name,
        "--stage",
        stage,
        "--model-kind",
        model_kind,
    ]
    if task is not None:
        base_args.extend(["--task", task])
    if args.skip_pretrain:
        base_args.append("--skip-pretrain")
    if train_only:
        base_args.append("--train-only")
    if eval_only or args.eval_only:
        base_args.append("--eval-only")
    if args.continue_run:
        base_args.append("--continue-run")
    base_args.extend(args.overrides)
    env: dict[str, str] = {
        "EUMP_TORCH_SHARING_STRATEGY": os.environ.get("EUMP_TORCH_SHARING_STRATEGY", "file_system"),
        "EUMP_DDP_TIMEOUT_SECONDS": os.environ.get("EUMP_DDP_TIMEOUT_SECONDS", "3600"),
    }
    if is_eval_profile:
        gpu = os.environ.get("EVAL_GPU") or first_gpu_id(os.environ.get("GPU")) or "0"
    else:
        gpu = os.environ.get("GPU", "0,1,2,3,4,5,6,7")
    if gpu:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    if nproc > 1:
        master_port_key = "EVAL_MASTER_PORT" if is_eval_profile else "MASTER_PORT"
        default_master_port = "29501" if is_eval_profile else "29500"
        cmd = [
            "torchrun",
            "--nproc_per_node",
            str(nproc),
            "--nnodes",
            "1",
            "--node_rank",
            "0",
            "--master_addr",
            os.environ.get("MASTER_ADDR", "127.0.0.1"),
            "--master_port",
            os.environ.get(master_port_key, default_master_port),
            *base_args,
        ]
    else:
        cmd = [sys.executable, *base_args]
    return cmd, env


def pretrain_model_path(run_dir: Path, scenario_name: str) -> Path:
    return run_dir / "model" / scenario_name / "pretrain" / "eump_reconstruction_final_model.pt"


def pretrain_checkpoint_path(run_dir: Path, scenario_name: str) -> Path:
    return run_dir / "model" / scenario_name / "pretrain" / "eump_reconstruction_final_checkpoint.pt"


def downstream_model_dir(run_dir: Path, model_kind: str, scenario_name: str, task: str) -> Path:
    if model_kind == "swin":
        return run_dir / "model" / scenario_name / task
    return run_dir / "model" / model_kind / scenario_name / task


def downstream_model_file_name(model_kind: str) -> str:
    return "eump_downstream_final_model.pt" if model_kind == "swin" else "eump_scalar_mlp_final_model.pt"


def downstream_checkpoint_file_name(model_kind: str) -> str:
    return "eump_downstream_final_checkpoint.pt" if model_kind == "swin" else "eump_scalar_mlp_final_checkpoint.pt"


def stagger_model_initialization(config: dict[str, Any], local_rank: int, logger: Any | None = None) -> None:
    stagger_seconds = float(config.get("model_init_stagger_seconds", 0.0) or 0.0)
    delay = stagger_seconds * int(local_rank)
    if delay <= 0:
        return
    if local_rank == 0 and logger is not None:
        logger.info(f"Stagger model initialization by {stagger_seconds:.2f}s per local rank")
    time.sleep(delay)


def build_domain_loader(
    *,
    data_config: dict[str, Any],
    pipeline_spec: Any,
    scene_ids: Iterable[str],
    logger: Any,
    random_seed: int,
    batch_size: int,
    local_rank: int,
    world_size: int,
    train: bool,
    allow_empty: bool = False,
    preprocessing_spec: dict[str, Any] | None = None,
    batch_schema: Any | None = None,
):
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from urban_embedding.core.reproducibility import seed_worker, seeded_torch_generator
    from urban_embedding.data.processed.dataset import (
        ProcessedSceneDataset,
        _filter_rows_for_task,
        _select_rows_for_scene_ids,
        collate_processed_samples,
    )
    from urban_embedding.data.processed.pipeline_spec import (
        build_active_preprocessing_spec,
        build_eump_batch_schema,
    )
    from urban_embedding.data.processed.processing import (
        load_preprocessing_spec,
        validate_preprocessing_spec,
        validate_preprocessing_stats_scope,
    )
    from urban_embedding.data.processed.reader import ProcessedSceneStoreReader

    reader = ProcessedSceneStoreReader(Path(str(data_config["eump_root"])))
    task_config = pipeline_spec.to_task_config()
    reader.validate_task_config(task_config)

    if preprocessing_spec is None:
        preprocessing_path = data_config.get("preprocessing_path", data_config["eump_root"])
        preprocessing_spec = load_preprocessing_spec(preprocessing_path)
        validate_preprocessing_spec(preprocessing_spec)
        validate_preprocessing_stats_scope(
            preprocessing_spec,
            split_manifest_path=data_config.get("split_manifest_path"),
            expected_year_range=pipeline_spec.year_range,
        )
        preprocessing_spec = build_active_preprocessing_spec(pipeline_spec, reader.source_specs, preprocessing_spec)

    if batch_schema is None:
        batch_schema = build_eump_batch_schema(pipeline_spec, reader.source_specs, preprocessing_spec)

    rows = list(reader.iter_scene_years(year_range=pipeline_spec.year_range))
    selected = _select_rows_for_scene_ids(rows, set(sorted_scene_ids(scene_ids)))
    selected = _filter_rows_for_task(selected, reader=reader, task_config=task_config)
    if not selected:
        if allow_empty:
            if local_rank == 0:
                logger.warning(
                    "Domain eval rows=0 cities=0 after task compatibility filtering; "
                    "this domain will be recorded as no_compatible_rows."
                )
            return None, preprocessing_spec, batch_schema
        raise ValueError("Domain scene IDs produced no compatible rows after task compatibility filtering")

    dataset = ProcessedSceneDataset(
        selected,
        reader=reader,
        pipeline_spec=pipeline_spec,
        preprocessing_spec=preprocessing_spec,
        batch_schema=batch_schema,
        random_seed=random_seed,
        cache_scenes=bool(pipeline_spec.cache_scenes),
        preload_scenes=bool(pipeline_spec.preload_scenes),
    )

    sampler = None
    shuffle = bool(train)
    if train or world_size > 1:
        if train:
            sampler = DistributedSampler(
                dataset,
                shuffle=True,
                seed=random_seed,
                num_replicas=world_size,
                rank=local_rank,
            )
        else:
            sampler = RankShardSampler(len(dataset), rank=local_rank, world_size=world_size)
        shuffle = False

    num_workers = max(0, int(pipeline_spec.num_workers) // max(1, world_size))
    kwargs: dict[str, Any] = {
        "batch_size": int(batch_size),
        "pin_memory": bool(pipeline_spec.pin_memory),
        "num_workers": num_workers,
        "collate_fn": collate_processed_samples,
        "shuffle": shuffle,
        "sampler": sampler,
        "worker_init_fn": seed_worker,
        "generator": seeded_torch_generator(
            random_seed,
            offset=(1 if train else 2) + local_rank * 1_000_000,
        ),
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = int(pipeline_spec.prefetch_factor)
        kwargs["persistent_workers"] = bool(pipeline_spec.persistent_workers)
    loader = DataLoader(dataset=dataset, **kwargs)
    if local_rank == 0:
        split_name = "train" if train else "eval"
        logger.info(
            f"Domain {split_name} rows={len(selected)} cities={len({str(row['scene_id']) for row in selected})}"
        )
    return DomainLoaders(
        loader=loader,
        rows=tuple(selected),
        scene_ids=tuple(sorted_scene_ids({row["scene_id"] for row in selected})),
    ), preprocessing_spec, batch_schema


def _city_scalar_arrays_for_scene_ids(joined: Any, scene_ids: Iterable[str]) -> tuple[Any, Any, tuple[str, ...]]:
    import numpy as np

    requested = set(sorted_scene_ids(scene_ids))
    normalized = np.asarray([normalize_scene_id(value) for value in joined.scene_ids.tolist()], dtype=object)
    mask = np.asarray([scene_id in requested for scene_id in normalized], dtype=bool)
    return (
        joined.features[mask].astype(np.float32, copy=False),
        joined.targets[mask].astype(np.float32, copy=False),
        tuple(str(value) for value in normalized[mask].tolist()),
    )


def build_scalar_domain_loader(
    *,
    joined: Any,
    scene_ids: Iterable[str],
    batch_size: int,
    random_seed: int,
    local_rank: int,
    world_size: int,
    train: bool,
    pin_memory: bool,
    allow_empty: bool = False,
) -> ScalarDomainLoaders | None:
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from urban_embedding.core.reproducibility import seeded_torch_generator
    from urban_embedding.training.eump.city_scalar_mlp.cache import (
        CityScalarTensorDataset,
        collate_cached_city_scalar_samples,
    )

    features, targets, row_scene_ids = _city_scalar_arrays_for_scene_ids(joined, scene_ids)
    if features.shape[0] <= 0:
        if allow_empty:
            return None
        raise ValueError("City-scalar domain scene IDs produced no compatible rows after cache join")
    dataset = CityScalarTensorDataset(features=features, targets=targets)
    sampler = None
    shuffle = bool(train)
    if train or world_size > 1:
        if train:
            sampler = DistributedSampler(
                dataset,
                shuffle=True,
                seed=random_seed,
                num_replicas=world_size,
                rank=local_rank,
            )
        else:
            sampler = RankShardSampler(len(dataset), rank=local_rank, world_size=world_size)
        shuffle = False
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=0,
        pin_memory=bool(pin_memory),
        collate_fn=collate_cached_city_scalar_samples,
        generator=seeded_torch_generator(
            random_seed,
            offset=(11 if train else 12) + local_rank * 1_000_000,
        ),
    )
    return ScalarDomainLoaders(
        loader=loader,
        scene_ids=tuple(sorted_scene_ids(set(row_scene_ids))),
        row_count_value=int(features.shape[0]),
    )


def train_scalar_mlp_one_epoch(
    *,
    model: Any,
    loader: Any,
    optimizer: Any,
    scheduler: Any,
    device: Any,
    epoch: int,
    accumulation_steps: int,
    grad_clip_norm: float,
    target_schema: Any,
) -> float:
    from urban_embedding.training.losses import compute_weighted_loss

    sampler = getattr(loader, "sampler", None)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)
    model.train()
    optimizer.zero_grad()
    loss_sum = 0.0
    loss_count = 0
    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device)
        prediction = model(batch.features)
        loss = compute_weighted_loss(target_schema.spec.loss, prediction, batch.target) * float(target_schema.spec.weight)
        (loss / accumulation_steps).backward()
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(loader):
            if grad_clip_norm > 0:
                import torch

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        loss_sum += float(loss.detach().item())
        loss_count += 1
    return reduce_sum_count(loss_sum, loss_count, device)


def summarize_scalar_mlp_domain_metrics(
    *,
    model: Any,
    domain: ScalarDomainLoaders,
    device: Any,
    source_spec: dict[str, Any],
    metric_scale: str,
) -> dict[str, Any]:
    import torch
    import torch.distributed as dist
    from urban_embedding.training.regression_metrics import (
        accumulate_regression_stats,
        empty_regression_stats,
        inverse_processed_tensor,
        summarize_regression_stats,
    )

    def metric_tensor(values):
        if metric_scale == "physical":
            return inverse_processed_tensor(values, source_spec)
        return values

    stats = empty_regression_stats(device=device)
    model.eval()
    with torch.no_grad():
        for batch in domain.loader:
            batch = batch.to(device)
            prediction = metric_tensor(model(batch.features))
            target = metric_tensor(batch.target)
            accumulate_regression_stats(stats, prediction, target)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    metrics, _ = summarize_regression_stats(stats)
    return {"global": metrics}


def compute_loss_report(
    *,
    batch: Any,
    target_schema: Any,
    target_kind: str,
    loss_config: Any,
    raster_predictions: dict[str, Any],
    scalar_predictions: dict[str, Any],
) -> Any:
    from urban_embedding.training.eump_loss import aggregate_loss_report, compute_target_loss, coverage

    if target_kind == "raster":
        prediction = raster_predictions[target_schema.name]
        target_block = batch.raster_targets[target_schema.name]
        city_mask = batch.city_mask
    else:
        prediction = scalar_predictions[target_schema.name]
        target_block = batch.scalar_targets[target_schema.name]
        city_mask = None

    zero = prediction.sum() * 0.0
    raw_loss, target_stats, support_weight, target_debug_terms = compute_target_loss(
        target=target_schema,
        prediction=prediction,
        target_data=target_block.data,
        valid_mask=target_block.valid_mask,
        city_mask=city_mask,
        masked_patch=None,
        loss_config=loss_config,
    )
    effective_weight = raw_loss.new_tensor(float(target_schema.spec.weight)) * support_weight
    mask_stats = {"mask/city_mask_coverage": coverage(batch.city_mask > 0, like=zero)}
    mask_stats.update(target_stats)
    return aggregate_loss_report(
        loss_config=loss_config,
        target_raw={target_schema.name: raw_loss},
        target_weighted={target_schema.name: raw_loss * effective_weight},
        target_groups={target_schema.name: target_schema.spec.group},
        target_weights={target_schema.name: effective_weight},
        target_sources={target_schema.name: target_schema.spec.source or target_schema.name},
        target_debug_terms={target_schema.name: target_debug_terms},
        mask_stats=mask_stats,
        zero=zero,
    )


def reduce_sum_count(sum_value: Any, count_value: int, device: Any) -> float:
    import torch
    import torch.distributed as dist

    payload = torch.tensor([float(sum_value), float(count_value)], dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(payload, op=dist.ReduceOp.SUM)
    total_count = max(1.0, float(payload[1].item()))
    return float(payload[0].item() / total_count)


def train_one_epoch(
    *,
    model: Any,
    loader: Any,
    optimizer: Any,
    scheduler: Any,
    device: Any,
    epoch: int,
    accumulation_steps: int,
    grad_clip_norm: float,
    target_schema: Any,
    target_kind: str,
    loss_config: Any,
) -> float:
    dataset = getattr(loader, "dataset", None)
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(epoch)
    sampler = getattr(loader, "sampler", None)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)

    model.train()
    optimizer.zero_grad()
    loss_sum = 0.0
    loss_count = 0
    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device)
        raster_predictions, scalar_predictions = model(batch.raster_inputs, batch.scalar_inputs)
        report = compute_loss_report(
            batch=batch,
            target_schema=target_schema,
            target_kind=target_kind,
            loss_config=loss_config,
            raster_predictions=raster_predictions,
            scalar_predictions=scalar_predictions,
        )
        loss = report.total
        (loss / accumulation_steps).backward()
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(loader):
            if grad_clip_norm > 0:
                import torch

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        loss_sum += float(loss.detach().item())
        loss_count += 1
    return reduce_sum_count(loss_sum, loss_count, device)


def build_reconstruction_mask(*, model: Any, batch: Any, mask_ratio: float, device: Any) -> Any:
    import math as _math
    from urban_embedding.training.distributed import get_model
    from urban_embedding.training.masking import generate_mask

    base_model = get_model(model)
    image_size = int(batch.raster_inputs.shape[-1])
    patch_size = int(base_model.config.patch_size)
    num_patches = (image_size // patch_size) ** 2
    num_masked_patches = _math.ceil(num_patches * float(mask_ratio))
    return generate_mask(
        batch.raster_inputs.shape[0],
        num_patches,
        num_masked_patches,
        device=device,
    )


def masked_patch_tensor(*, model: Any, bool_masked_pos: Any, batch_size: int, height: int, width: int) -> Any:
    if bool_masked_pos is None:
        return None
    from urban_embedding.training.distributed import get_model

    base_model = get_model(model)
    patch_size = int(base_model.config.patch_size)
    patch_h = height // patch_size
    patch_w = width // patch_size
    mask = bool_masked_pos.reshape(batch_size, patch_h, patch_w)
    return (
        mask.repeat_interleave(patch_size, 1)
        .repeat_interleave(patch_size, 2)
        .unsqueeze(1)
        .contiguous()
    )


def split_reconstruction_raster_predictions(*, batch_schema: Any, raster_prediction: Any) -> dict[str, Any]:
    predictions: dict[str, Any] = {}
    start = 0
    for target in batch_schema.raster_targets:
        end = start + int(target.prediction_channels)
        predictions[target.name] = raster_prediction[:, start:end, ...]
        start = end
    if start != int(raster_prediction.shape[1]):
        raise ValueError(
            f"Raster prediction channels consumed {start}, but model returned {raster_prediction.shape[1]}"
        )
    return predictions


def compute_reconstruction_loss_report(
    *,
    model: Any,
    batch_schema: Any,
    loss_config: Any,
    batch: Any,
    raster_prediction: Any,
    scalar_predictions: dict[str, Any],
    bool_masked_pos: Any,
) -> Any:
    from urban_embedding.training.eump_loss import aggregate_loss_report, compute_target_loss, coverage

    batch_size, _, height, width = batch.raster_inputs.shape
    masked_patch = masked_patch_tensor(
        model=model,
        bool_masked_pos=bool_masked_pos,
        batch_size=batch_size,
        height=height,
        width=width,
    )
    raster_predictions = split_reconstruction_raster_predictions(
        batch_schema=batch_schema,
        raster_prediction=raster_prediction,
    )
    zero = raster_prediction.sum() * 0.0
    target_raw: dict[str, Any] = {}
    target_weighted: dict[str, Any] = {}
    target_groups: dict[str, str] = {}
    target_weights: dict[str, Any] = {}
    target_sources: dict[str, str] = {}
    target_debug_terms: dict[str, dict[str, Any]] = {}
    mask_stats: dict[str, Any] = {
        "mask/city_mask_coverage": coverage(batch.city_mask > 0, like=zero),
        "mask/masked_mask_coverage": coverage(masked_patch, like=zero),
    }

    for target in batch_schema.raster_targets:
        target_block = batch.raster_targets[target.name]
        raw_loss, stats, support_weight, debug_terms = compute_target_loss(
            target=target,
            prediction=raster_predictions[target.name],
            target_data=target_block.data,
            valid_mask=target_block.valid_mask,
            city_mask=batch.city_mask,
            masked_patch=masked_patch,
            loss_config=loss_config,
        )
        effective_weight = raw_loss.new_tensor(float(target.spec.weight)) * support_weight
        target_raw[target.name] = raw_loss
        target_weighted[target.name] = raw_loss * effective_weight
        target_groups[target.name] = target.spec.group
        target_weights[target.name] = effective_weight
        target_sources[target.name] = target.spec.source or target.name
        target_debug_terms[target.name] = debug_terms
        mask_stats.update(stats)

    for target in batch_schema.scalar_targets:
        if target.name not in scalar_predictions:
            raise KeyError(f"Missing scalar prediction for target {target.name!r}")
        target_block = batch.scalar_targets[target.name]
        raw_loss, stats, support_weight, debug_terms = compute_target_loss(
            target=target,
            prediction=scalar_predictions[target.name],
            target_data=target_block.data,
            valid_mask=target_block.valid_mask,
            city_mask=None,
            masked_patch=None,
            loss_config=loss_config,
        )
        effective_weight = raw_loss.new_tensor(float(target.spec.weight)) * support_weight
        target_raw[target.name] = raw_loss
        target_weighted[target.name] = raw_loss * effective_weight
        target_groups[target.name] = target.spec.group
        target_weights[target.name] = effective_weight
        target_sources[target.name] = target.spec.source or target.name
        target_debug_terms[target.name] = debug_terms
        mask_stats.update(stats)

    return aggregate_loss_report(
        loss_config=loss_config,
        target_raw=target_raw,
        target_weighted=target_weighted,
        target_groups=target_groups,
        target_weights=target_weights,
        target_sources=target_sources,
        target_debug_terms=target_debug_terms,
        mask_stats=mask_stats,
        zero=zero,
    )


def train_pretrain_one_epoch(
    *,
    model: Any,
    loader: Any,
    optimizer: Any,
    scheduler: Any,
    device: Any,
    epoch: int,
    accumulation_steps: int,
    grad_clip_norm: float,
    mask_ratio: float,
    batch_schema: Any,
    loss_config: Any,
) -> float:
    dataset = getattr(loader, "dataset", None)
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(epoch)
    sampler = getattr(loader, "sampler", None)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)

    model.train()
    optimizer.zero_grad()
    loss_sum = 0.0
    loss_count = 0
    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device)
        bool_masked_pos = build_reconstruction_mask(
            model=model,
            batch=batch,
            mask_ratio=mask_ratio,
            device=device,
        )
        raster_prediction, scalar_predictions = model(batch.raster_inputs, bool_masked_pos)
        report = compute_reconstruction_loss_report(
            model=model,
            batch_schema=batch_schema,
            loss_config=loss_config,
            batch=batch,
            raster_prediction=raster_prediction,
            scalar_predictions=scalar_predictions,
            bool_masked_pos=bool_masked_pos,
        )
        loss = report.total
        (loss / accumulation_steps).backward()
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(loader):
            if grad_clip_norm > 0:
                import torch

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        loss_sum += float(loss.detach().item())
        loss_count += 1
    return reduce_sum_count(loss_sum, loss_count, device)


def summarize_domain_metrics(
    *,
    model: Any,
    domain: DomainLoaders,
    device: Any,
    target_schema: Any,
    target_kind: str,
    source_spec: dict[str, Any],
    metric_scale: str,
) -> dict[str, Any]:
    import torch
    import torch.distributed as dist
    from urban_embedding.training.regression_metrics import (
        accumulate_regression_stats,
        empty_regression_stats,
        inverse_processed_tensor,
        raster_city_mean_metric_tensors,
        raster_global_metric_tensors,
        scalar_metric_tensors,
        summarize_regression_stats,
    )

    def metric_tensor(values):
        if metric_scale == "physical":
            return inverse_processed_tensor(values, source_spec)
        return values

    stats = {"global": empty_regression_stats(device=device)}
    if target_kind == "raster":
        stats["city_mean"] = empty_regression_stats(device=device)

    model.eval()
    with torch.no_grad():
        for batch in domain.loader:
            batch = batch.to(device)
            raster_predictions, scalar_predictions = model(batch.raster_inputs, batch.scalar_inputs)
            if target_kind == "scalar":
                prediction = metric_tensor(scalar_predictions[target_schema.name])
                target_block = batch.scalar_targets[target_schema.name]
                target = metric_tensor(target_block.data)
                pred_values, target_values = scalar_metric_tensors(
                    prediction=prediction,
                    target=target,
                    valid_mask=target_block.valid_mask,
                    mask_strategy=target_schema.spec.mask,
                )
                accumulate_regression_stats(stats["global"], pred_values, target_values)
            else:
                prediction = metric_tensor(raster_predictions[target_schema.name])
                target_block = batch.raster_targets[target_schema.name]
                target = metric_tensor(target_block.data)
                pred_values, target_values = raster_global_metric_tensors(
                    prediction=prediction,
                    target=target,
                    valid_mask=target_block.valid_mask,
                    city_mask=batch.city_mask,
                    mask_strategy=target_schema.spec.mask,
                )
                accumulate_regression_stats(stats["global"], pred_values, target_values)
                city_pred, city_target = raster_city_mean_metric_tensors(
                    prediction=prediction,
                    target=target,
                    valid_mask=target_block.valid_mask,
                    city_mask=batch.city_mask,
                    mask_strategy=target_schema.spec.mask,
                )
                accumulate_regression_stats(stats["city_mean"], city_pred, city_target)

    output: dict[str, Any] = {}
    for name, tensor in stats.items():
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        metrics, _ = summarize_regression_stats(tensor)
        output["global_pixels" if target_kind == "raster" and name == "global" else name] = metrics
    return output


def write_summary_csv(run_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    results_path = run_dir / "results.jsonl"
    if results_path.exists():
        with results_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    summary_path = run_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(RESULT_FIELDS))
        writer.writeheader()
        for row in rows:
            if "model" not in row:
                row["model"] = "swin"
            writer.writerow({field: row.get(field) for field in RESULT_FIELDS})


def append_result_rows(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    results_path = run_dir / "results.jsonl"
    existing_rows: list[dict[str, Any]] = []
    if results_path.exists():
        with results_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if "model" not in row:
                        row["model"] = "swin"
                    existing_rows.append(row)
    for row in rows:
        if "model" not in row:
            row["model"] = "swin"
    key_fields = ("model", "scenario", "eval_domain", "task", "target", "scope")
    new_keys = {tuple(row.get(field) for field in key_fields) for row in rows}
    merged_rows = [
        row
        for row in existing_rows
        if tuple(row.get(field) for field in key_fields) not in new_keys
    ]
    merged_rows.extend(rows)
    with results_path.open("w", encoding="utf-8") as handle:
        for row in merged_rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    write_summary_csv(run_dir)


def empty_eval_result_rows(
    *,
    model_kind: str = "swin",
    scenario_name: str,
    train_domain: str,
    eval_domain: str,
    task: str,
    target: str,
    target_kind: str,
    year_range: str,
    model_path: Path,
    message: str,
) -> list[dict[str, Any]]:
    scopes = ("global_pixels", "city_mean") if target_kind == "raster" else ("global",)
    return [
        {
            "model": model_kind,
            "scenario": scenario_name,
            "train_domain": train_domain,
            "eval_domain": eval_domain,
            "task": task,
            "target": target,
            "scope": scope,
            "rmse": None,
            "mae": None,
            "r2": None,
            "count": 0,
            "city_count": 0,
            "row_count": 0,
            "year_range": year_range,
            "model_path": str(model_path),
            "status": "no_compatible_rows",
            "message": message,
        }
        for scope in scopes
    ]


def pretrain_worker_main(args: argparse.Namespace) -> int:
    if args.run_dir is None or args.scenario is None:
        raise ValueError("--stage pretrain requires --run-dir and --scenario")

    import torch
    from urban_embedding.core.logging import save_runtime_config, setup_logger
    from urban_embedding.core.paths import MODEL_DIR
    from urban_embedding.core.reproducibility import set_reproducible
    from urban_embedding.data.processed.pipeline_spec import PipelineLossConfig, PipelineSpec
    from urban_embedding.models.eump.swin.reconstruction import EumpSwinReconstructionModel
    from urban_embedding.training.distributed import (
        destroy_distributed_process_group,
        get_model,
        get_rank,
        get_world_size,
        init_distributed_process_group,
        resolve_device,
        wrap_ddp,
    )
    from urban_embedding.training.schedulers import (
        build_warmup_cosine_scheduler,
        normalize_warmup_cosine_scheduler_config,
    )

    config, hydra_overrides = load_experiment_config(args.config, args.overrides)
    scenarios = scenario_specs_from_config(config)
    scenario = scenarios[str(args.scenario)]
    run_dir = args.run_dir if args.run_dir.is_absolute() else (REPO_ROOT / args.run_dir)
    manifest = json.loads((run_dir / "domain_manifest.json").read_text(encoding="utf-8"))
    scenario_manifest = manifest["scenarios"][scenario.name]
    source_scene_ids = tuple(str(item) for item in scenario_manifest["source_scene_ids"])

    init_distributed_process_group()
    try:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        world_size = get_world_size()
        rank = get_rank()
        device = resolve_device(local_rank)
        log_dir = run_dir / "log" / scenario.name / "pretrain"
        logger = setup_logger(log_dir=str(log_dir), rank=rank, log_level="INFO")
        if rank == 0:
            logger.info(f"Start domain transfer pretrain scenario={scenario.name}")
        configure_torch_multiprocessing(logger, rank=rank)

        pipeline_suite = pipeline_suite_for_model(config, "swin")
        runtime_config = compose_runtime_config(pipeline=f"{pipeline_suite}/pretrain", overrides=hydra_overrides)
        apply_loader_overrides(runtime_config, config)
        pretrain_cfg = dict(config.get("pretrain", {}))
        reconstruct_cfg = runtime_config["reconstruct"]
        reconstruct_cfg["epochs"] = int(pretrain_cfg.get("epochs", 100))
        reconstruct_cfg["lr"] = float(pretrain_cfg.get("lr", 1.0e-4))
        reconstruct_cfg["batch_size"] = int(pretrain_cfg.get("batch_size", 64))
        reconstruct_cfg["weight_decay"] = float(pretrain_cfg.get("weight_decay", 1.0e-2))
        reconstruct_cfg["accumulation_steps"] = int(pretrain_cfg.get("accumulation_steps", 1))
        reconstruct_cfg["mask_ratio"] = float(pretrain_cfg.get("mask_ratio", reconstruct_cfg.get("mask_ratio", 0.6)))
        reconstruct_cfg["scalar_head_mode"] = str(pretrain_cfg.get("scalar_head_mode", reconstruct_cfg.get("scalar_head_mode", "per_target")))
        reconstruct_cfg["scheduler"] = dict(pretrain_cfg.get("scheduler", reconstruct_cfg.get("scheduler", {})))
        runtime_config["experiment"]["group"] = str(
            config.get("run_name", DEFAULT_RUN_NAME)
        )
        runtime_config["experiment"]["name"] = f"{scenario.name}_pretrain"
        runtime_config["experiment"]["model_name"] = (
            f"{runtime_config['experiment']['group']}/{runtime_config['experiment']['name']}"
        )
        if rank == 0:
            save_runtime_config(str(log_dir), runtime_config)

        global_batch_size = int(reconstruct_cfg["batch_size"])
        if global_batch_size % max(1, world_size) != 0:
            raise ValueError(
                f"pretrain.batch_size={global_batch_size} must be divisible by world_size={world_size}"
            )
        local_batch_size = global_batch_size // max(1, world_size)
        set_reproducible(int(runtime_config["experiment"]["random_seed"]))

        pipeline_spec = PipelineSpec.from_mapping(runtime_config["pipeline"])
        train_domain, _, batch_schema = build_domain_loader(
            data_config=runtime_config["data"],
            pipeline_spec=pipeline_spec,
            scene_ids=source_scene_ids,
            logger=logger,
            random_seed=int(runtime_config["experiment"]["random_seed"]),
            batch_size=local_batch_size,
            local_rank=local_rank,
            world_size=world_size,
            train=True,
        )
        stagger_model_initialization(config, local_rank, logger)
        model = EumpSwinReconstructionModel(
            model_path=os.path.join(MODEL_DIR, runtime_config["experiment"]["pretrain_model"]),
            input_channels=batch_schema.raster_input_channels,
            batch_schema=batch_schema,
            load_pretrained=not bool(runtime_config["experiment"]["scratch_backbone"]),
            scalar_head_mode=str(reconstruct_cfg["scalar_head_mode"]),
        ).to(device)
        model = wrap_ddp(model, device, local_rank)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(reconstruct_cfg["lr"]),
            weight_decay=float(reconstruct_cfg["weight_decay"]),
        )
        scheduler_cfg = normalize_warmup_cosine_scheduler_config(reconstruct_cfg.get("scheduler"))
        scheduler, scheduler_metadata = build_warmup_cosine_scheduler(
            optimizer=optimizer,
            base_lr=float(reconstruct_cfg["lr"]),
            num_batches=len(train_domain.loader),
            accumulation_steps=int(reconstruct_cfg["accumulation_steps"]),
            epochs=int(reconstruct_cfg["epochs"]),
            warmup_fraction=scheduler_cfg.warmup_fraction,
            lr_start_factor=scheduler_cfg.lr_start_factor,
            lr_end_factor=scheduler_cfg.lr_end_factor,
        )
        loss_config = pipeline_spec.loss_config if hasattr(pipeline_spec, "loss_config") else PipelineLossConfig()

        epochs = int(reconstruct_cfg["epochs"])
        log_every = max(1, int(pretrain_cfg.get("log_every_n_epochs", 1)))
        start_time = time.time()
        for epoch in range(1, epochs + 1):
            avg_loss = train_pretrain_one_epoch(
                model=model,
                loader=train_domain.loader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                epoch=epoch,
                accumulation_steps=int(reconstruct_cfg["accumulation_steps"]),
                grad_clip_norm=float(pretrain_cfg.get("grad_clip_norm", 1.0)),
                mask_ratio=float(reconstruct_cfg["mask_ratio"]),
                batch_schema=batch_schema,
                loss_config=loss_config,
            )
            if rank == 0 and (epoch == 1 or epoch == epochs or epoch % log_every == 0):
                logger.info(
                    f"DOMAIN PRETRAIN scenario={scenario.name} epoch={epoch}/{epochs} "
                    f"train_loss={avg_loss:.6f} lr={optimizer.param_groups[0]['lr']:.8f}"
                )

        model_path = pretrain_model_path(run_dir, scenario.name)
        checkpoint_path = pretrain_checkpoint_path(run_dir, scenario.name)
        if rank == 0:
            model_path.parent.mkdir(parents=True, exist_ok=True)
            unwrapped = get_model(model)
            torch.save(unwrapped.state_dict(), model_path)
            torch.save(
                {
                    "epoch": epochs,
                    "model_state_dict": unwrapped.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scheduler_metadata": scheduler_metadata,
                    "runtime_config": runtime_config,
                    "scenario": scenario.name,
                    "stage": "pretrain",
                    "train_domain": scenario.source,
                },
                checkpoint_path,
            )
            logger.info(f"Saved domain pretrain final model to {model_path}")
            logger.info(f"Saved domain pretrain final checkpoint to {checkpoint_path}")
            logger.info(
                f"Finished domain pretrain in {(time.time() - start_time) / 60.0:.2f} min: "
                f"scenario={scenario.name}"
            )
    finally:
        destroy_distributed_process_group()
    return 0


def scalar_mlp_worker_main(args: argparse.Namespace) -> int:
    if args.run_dir is None or args.scenario is None or args.task is None:
        raise ValueError("city-scalar domain transfer worker requires --run-dir, --scenario, and --task")
    if args.model_kind not in {"scalar_mlp", "aef_mlp"}:
        raise ValueError(f"Unsupported city-scalar model kind: {args.model_kind!r}")
    if bool(args.train_only) and bool(args.eval_only):
        raise ValueError("--train-only and --eval-only are mutually exclusive")

    import torch
    from urban_embedding.core.logging import save_runtime_config, setup_logger
    from urban_embedding.core.reproducibility import set_reproducible
    from urban_embedding.data.processed.pipeline_spec import (
        PipelineSpec,
        validate_eump_scalar_mlp_downstream_spec,
    )
    from urban_embedding.models.eump.city_scalar_mlp.model import build_eump_scalar_mlp_model
    from urban_embedding.training.distributed import (
        destroy_distributed_process_group,
        distributed_barrier,
        get_model,
        get_rank,
        get_world_size,
        init_distributed_process_group,
        resolve_device,
        wrap_ddp,
    )
    from urban_embedding.training.eump.city_scalar_mlp.cache import (
        compute_cached_city_level_standardization_stats,
        ensure_city_scalar_cache_files,
        join_city_scalar_caches,
        normalize_city_scalar_feature_standardization,
    )
    from urban_embedding.training.schedulers import (
        build_warmup_cosine_scheduler,
        normalize_warmup_cosine_scheduler_config,
    )

    config, hydra_overrides = load_experiment_config(args.config, args.overrides)
    scenarios = scenario_specs_from_config(config)
    scenario = scenarios[str(args.scenario)]
    run_dir = args.run_dir if args.run_dir.is_absolute() else (REPO_ROOT / args.run_dir)
    manifest = json.loads((run_dir / "domain_manifest.json").read_text(encoding="utf-8"))
    scenario_manifest = manifest["scenarios"][scenario.name]
    source_scene_ids = tuple(str(item) for item in scenario_manifest["source_scene_ids"])

    init_distributed_process_group()
    try:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        world_size = get_world_size()
        rank = get_rank()
        device = resolve_device(local_rank)
        log_dir = run_dir / "log" / str(args.model_kind) / scenario.name / str(args.task)
        logger = setup_logger(log_dir=str(log_dir), rank=rank, log_level="INFO")
        if rank == 0:
            logger.info(
                f"Start domain transfer worker model={args.model_kind} "
                f"scenario={scenario.name} task={args.task}"
            )
        configure_torch_multiprocessing(logger, rank=rank)

        pipeline_suite = pipeline_suite_for_model(config, str(args.model_kind))
        runtime_config = compose_runtime_config(pipeline=f"{pipeline_suite}/{args.task}", overrides=hydra_overrides)
        apply_loader_overrides(runtime_config, config)
        apply_downstream_year_range(runtime_config, config)
        training_cfg = apply_transfer_training_overrides(
            runtime_config,
            config,
            model_kind=str(args.model_kind),
            task=str(args.task),
            overrides=args.overrides,
        )
        downstream_cfg = runtime_config["downstream"]
        baseline_cfg = config.get("baselines", {})
        if not isinstance(baseline_cfg, dict):
            baseline_cfg = {}
        feature_standardization = normalize_city_scalar_feature_standardization(
            baseline_cfg.get("feature_standardization", downstream_cfg.get("feature_standardization", "none"))
        )
        downstream_cfg["feature_standardization"] = feature_standardization
        runtime_config["experiment"]["group"] = str(
            config.get("run_name", DEFAULT_RUN_NAME)
        )
        runtime_config["experiment"]["name"] = f"{args.model_kind}_{scenario.name}_{args.task}"
        runtime_config["experiment"]["model_name"] = (
            f"{runtime_config['experiment']['group']}/{runtime_config['experiment']['name']}"
        )
        if rank == 0:
            save_runtime_config(str(log_dir), runtime_config)

        global_batch_size = int(downstream_cfg["batch_size"])
        if global_batch_size % max(1, world_size) != 0:
            raise ValueError(
                f"training.batch_size={global_batch_size} must be divisible by world_size={world_size}"
            )
        local_batch_size = global_batch_size // max(1, world_size)
        set_reproducible(int(runtime_config["experiment"]["random_seed"]))

        pipeline_spec = PipelineSpec.from_mapping(runtime_config["pipeline"])
        validate_eump_scalar_mlp_downstream_spec(pipeline_spec)
        (
            feature_cache,
            target_cache,
            _batch_schema,
            target_schema,
            feature_layout,
            preprocessing_spec,
        ) = ensure_city_scalar_cache_files(
            data_config=runtime_config["data"],
            pipeline_config=runtime_config["pipeline"],
            logger=logger,
            random_seed=int(runtime_config["experiment"]["random_seed"]),
            local_rank=local_rank,
            log_dir=str(log_dir),
        )
        joined = join_city_scalar_caches(feature_cache, target_cache)
        if rank == 0:
            logger.info(
                f"Prepared domain city-scalar cache model={args.model_kind} "
                f"target={target_schema.name} features={feature_layout.feature_dim} "
                f"joined_valid={joined.metadata.get('valid_joined_rows')}"
            )
        train_features, train_targets, _ = _city_scalar_arrays_for_scene_ids(joined, source_scene_ids)
        if train_features.shape[0] <= 0:
            raise ValueError(
                f"City-scalar domain source {scenario.source!r} produced no compatible rows "
                f"for model={args.model_kind} task={args.task}"
            )
        standardization = compute_cached_city_level_standardization_stats(
            features=train_features,
            targets=train_targets,
            device=device,
            feature_standardization=feature_standardization,
        )
        train_domain = build_scalar_domain_loader(
            joined=joined,
            scene_ids=source_scene_ids,
            batch_size=local_batch_size,
            random_seed=int(runtime_config["experiment"]["random_seed"]),
            local_rank=local_rank,
            world_size=world_size,
            train=True,
            pin_memory=bool(pipeline_spec.pin_memory),
        )
        if train_domain is None:
            raise ValueError("Unexpected empty source domain loader")
        if rank == 0:
            logger.info(f"Domain train rows={train_domain.row_count} cities={train_domain.city_count}")

        eval_domains: dict[str, ScalarDomainLoaders | None] = {}
        for target_domain in scenario.targets:
            target_scene_ids = tuple(str(item) for item in scenario_manifest["targets"][target_domain]["scene_ids"])
            domain = build_scalar_domain_loader(
                joined=joined,
                scene_ids=target_scene_ids,
                batch_size=local_batch_size,
                random_seed=int(runtime_config["experiment"]["random_seed"]),
                local_rank=local_rank,
                world_size=world_size,
                train=False,
                pin_memory=bool(pipeline_spec.pin_memory),
                allow_empty=True,
            )
            eval_domains[target_domain] = domain
        eval_domains[f"source:{scenario.source}"] = build_scalar_domain_loader(
            joined=joined,
            scene_ids=source_scene_ids,
            batch_size=local_batch_size,
            random_seed=int(runtime_config["experiment"]["random_seed"]),
            local_rank=local_rank,
            world_size=world_size,
            train=False,
            pin_memory=bool(pipeline_spec.pin_memory),
        )

        model = build_eump_scalar_mlp_model(
            model_config=runtime_config["model"],
            input_dim=feature_layout.feature_dim,
            output_dim=target_schema.channels,
        ).to(device)
        model.set_standardization(
            feature_mean=standardization.feature_mean.to(device),
            feature_std=standardization.feature_std.to(device),
            target_mean=standardization.target_mean.to(device),
            target_std=standardization.target_std.to(device),
        )
        model = wrap_ddp(model, device, local_rank)
        optimizer = torch.optim.AdamW(
            [param for param in model.parameters() if param.requires_grad],
            lr=float(downstream_cfg["lr"]),
            weight_decay=float(downstream_cfg["weight_decay"]),
        )
        scheduler_cfg = normalize_warmup_cosine_scheduler_config(downstream_cfg.get("scheduler"))
        scheduler, scheduler_metadata = build_warmup_cosine_scheduler(
            optimizer=optimizer,
            base_lr=float(downstream_cfg["lr"]),
            num_batches=len(train_domain.loader),
            accumulation_steps=int(downstream_cfg["accumulation_steps"]),
            epochs=int(downstream_cfg["epochs"]),
            warmup_fraction=scheduler_cfg.warmup_fraction,
            lr_start_factor=scheduler_cfg.lr_start_factor,
            lr_end_factor=scheduler_cfg.lr_end_factor,
        )

        epochs = int(downstream_cfg["epochs"])
        log_every = max(1, int(training_cfg.get("log_every_n_epochs", 1)))
        start_time = time.time()
        model_dir = downstream_model_dir(run_dir, str(args.model_kind), scenario.name, str(args.task))
        model_path = model_dir / downstream_model_file_name(str(args.model_kind))
        checkpoint_path = model_dir / downstream_checkpoint_file_name(str(args.model_kind))

        if bool(args.eval_only):
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Eval-only city-scalar checkpoint not found: {checkpoint_path}")
            if rank == 0:
                logger.info(f"Eval-only mode: load final training checkpoint from {checkpoint_path}")
        else:
            for epoch in range(1, epochs + 1):
                avg_loss = train_scalar_mlp_one_epoch(
                    model=model,
                    loader=train_domain.loader,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    device=device,
                    epoch=epoch,
                    accumulation_steps=int(downstream_cfg["accumulation_steps"]),
                    grad_clip_norm=float(training_cfg.get("grad_clip_norm", 1.0)),
                    target_schema=target_schema,
                )
                if rank == 0 and (epoch == 1 or epoch == epochs or epoch % log_every == 0):
                    logger.info(
                        f"DOMAIN TRAIN model={args.model_kind} scenario={scenario.name} task={args.task} "
                        f"epoch={epoch}/{epochs} target={target_schema.name} train_loss={avg_loss:.6f} "
                        f"lr={optimizer.param_groups[0]['lr']:.8f}"
                    )

            if rank == 0:
                model_dir.mkdir(parents=True, exist_ok=True)
                unwrapped = get_model(model)
                torch.save(unwrapped.state_dict(), model_path)
                torch.save(
                    {
                        "epoch": epochs,
                        "model_state_dict": unwrapped.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "scheduler_metadata": scheduler_metadata,
                        "runtime_config": runtime_config,
                        "scenario": scenario.name,
                        "task": args.task,
                        "model_kind": args.model_kind,
                        "train_domain": scenario.source,
                        "cache_metadata": joined.metadata,
                        "feature_standardization": feature_standardization,
                    },
                    checkpoint_path,
                )
                logger.info(f"Saved final model to {model_path}")
                logger.info(f"Saved final checkpoint to {checkpoint_path}")

        distributed_barrier(local_rank)
        if bool(args.train_only):
            if rank == 0:
                logger.info("Train-only mode: skip city-scalar domain evaluation")
            return 0
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise KeyError(f"Final training checkpoint has no model_state_dict: {checkpoint_path}")
        get_model(model).load_state_dict(checkpoint["model_state_dict"])

        source_name = target_schema.spec.source
        if source_name is None:
            raise ValueError(f"Target {target_schema.name!r} requires a source for metric inversion")
        source_spec = preprocessing_spec["sources"][source_name]
        result_rows: list[dict[str, Any]] = []
        metric_scale = str(config.get("metric_scale", "physical")).lower()
        if joined.years.shape[0] > 0:
            year_range = f"{int(joined.years.min())}-{int(joined.years.max())}"
        else:
            year_range = "-".join(str(item) for item in pipeline_spec.year_range)
        for eval_domain_name, domain in eval_domains.items():
            if domain is None:
                if rank == 0:
                    message = (
                        "No compatible city-scalar cache rows remained after filtering this domain "
                        f"for model={args.model_kind} task={args.task} target={target_schema.name}."
                    )
                    result_rows.extend(
                        empty_eval_result_rows(
                            model_kind=str(args.model_kind),
                            scenario_name=scenario.name,
                            train_domain=scenario.source,
                            eval_domain=eval_domain_name,
                            task=str(args.task),
                            target=target_schema.name,
                            target_kind="scalar",
                            year_range=year_range,
                            model_path=model_path,
                            message=message,
                        )
                    )
                    logger.warning(
                        f"DOMAIN EVAL skipped model={args.model_kind} scenario={scenario.name} "
                        f"task={args.task} train={scenario.source} eval={eval_domain_name}: {message}"
                    )
                continue
            metrics_by_scope = summarize_scalar_mlp_domain_metrics(
                model=model,
                domain=domain,
                device=device,
                source_spec=source_spec,
                metric_scale=metric_scale,
            )
            if rank == 0:
                for scope, metrics in metrics_by_scope.items():
                    row = {
                        "model": args.model_kind,
                        "scenario": scenario.name,
                        "train_domain": scenario.source,
                        "eval_domain": eval_domain_name,
                        "task": args.task,
                        "target": target_schema.name,
                        "scope": scope,
                        "rmse": metrics.get("rmse"),
                        "mae": metrics.get("mae"),
                        "r2": metrics.get("r2"),
                        "count": metrics.get("count"),
                        "city_count": domain.city_count,
                        "row_count": domain.row_count,
                        "year_range": year_range,
                        "model_path": str(model_path),
                        "status": "ok",
                        "message": "",
                    }
                    result_rows.append(row)
                    logger.info(
                        "DOMAIN EVAL model={model} scenario={scenario} task={task} train={train} eval={eval} "
                        "target={target} scope={scope} rmse={rmse} r2={r2} count={count}".format(
                            model=args.model_kind,
                            scenario=scenario.name,
                            task=args.task,
                            train=scenario.source,
                            eval=eval_domain_name,
                            target=target_schema.name,
                            scope=scope,
                            rmse=("null" if row["rmse"] is None else f"{float(row['rmse']):.6f}"),
                            r2=("null" if row["r2"] is None else f"{float(row['r2']):.6f}"),
                            count=row["count"],
                        )
                    )
        if rank == 0:
            append_result_rows(run_dir, result_rows)
            logger.info(
                f"Finished domain transfer worker in {(time.time() - start_time) / 60.0:.2f} min: "
                f"model={args.model_kind} scenario={scenario.name} task={args.task}"
            )
    finally:
        destroy_distributed_process_group()
    return 0


def worker_main(args: argparse.Namespace) -> int:
    if args.stage == "pretrain":
        return pretrain_worker_main(args)
    if args.run_dir is None or args.scenario is None or args.task is None:
        raise ValueError("--stage downstream requires --run-dir, --scenario, and --task")
    if bool(args.train_only) and bool(args.eval_only):
        raise ValueError("--train-only and --eval-only are mutually exclusive")
    if args.model_kind in {"scalar_mlp", "aef_mlp"}:
        return scalar_mlp_worker_main(args)

    import torch
    from urban_embedding.core.logging import save_runtime_config, setup_logger
    from urban_embedding.core.paths import MODEL_DIR, resolve_model_artifact_path
    from urban_embedding.core.reproducibility import set_reproducible
    from urban_embedding.data.processed.pipeline_spec import (
        PipelineLossConfig,
        PipelineSpec,
        get_single_downstream_target_schema,
        validate_eump_downstream_spec,
    )
    from urban_embedding.models.eump.swin.downstream import build_eump_downstream_model
    from urban_embedding.training.distributed import (
        destroy_distributed_process_group,
        distributed_barrier,
        get_model,
        get_rank,
        get_world_size,
        init_distributed_process_group,
        resolve_device,
        wrap_ddp,
    )
    from urban_embedding.training.schedulers import (
        build_warmup_cosine_scheduler,
        normalize_warmup_cosine_scheduler_config,
    )

    config, hydra_overrides = load_experiment_config(args.config, args.overrides)
    scenarios = scenario_specs_from_config(config)
    scenario = scenarios[str(args.scenario)]
    run_dir = args.run_dir if args.run_dir.is_absolute() else (REPO_ROOT / args.run_dir)
    manifest = json.loads((run_dir / "domain_manifest.json").read_text(encoding="utf-8"))
    scenario_manifest = manifest["scenarios"][scenario.name]
    source_scene_ids = tuple(str(item) for item in scenario_manifest["source_scene_ids"])

    init_distributed_process_group()
    try:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        world_size = get_world_size()
        rank = get_rank()
        device = resolve_device(local_rank)
        log_dir = run_dir / "log" / scenario.name / str(args.task)
        logger = setup_logger(log_dir=str(log_dir), rank=rank, log_level="INFO")
        if rank == 0:
            logger.info(f"Start domain transfer worker scenario={scenario.name} task={args.task}")
        configure_torch_multiprocessing(logger, rank=rank)

        pipeline_suite = pipeline_suite_for_model(config, "swin")
        pipeline_name = f"{pipeline_suite}/{args.task}"
        runtime_config = compose_runtime_config(pipeline=pipeline_name, overrides=hydra_overrides)
        apply_loader_overrides(runtime_config, config)
        apply_downstream_year_range(runtime_config, config)
        training_cfg = apply_transfer_training_overrides(
            runtime_config,
            config,
            model_kind="swin",
            task=str(args.task),
            overrides=args.overrides,
        )
        downstream_cfg = runtime_config["downstream"]
        reuse_scenario_pretrain = bool(config.get("pretrain", {}).get("enabled", True))
        if reuse_scenario_pretrain:
            scenario_pretrain_checkpoint = pretrain_checkpoint_path(run_dir, scenario.name)
            if not scenario_pretrain_checkpoint.exists():
                raise FileNotFoundError(
                    f"Scenario pretrain checkpoint not found: {scenario_pretrain_checkpoint}. "
                    "Run the pretrain stage before downstream training, or set pretrain.enabled=false "
                    "to use source_checkpoint."
                )
            downstream_cfg["encoder_checkpoint_file"] = str(scenario_pretrain_checkpoint)
        else:
            downstream_cfg["encoder_checkpoint_file"] = str(
                config.get("source_checkpoint", "eump_reconstruction_best_checkpoint.pt")
            )
        runtime_config["experiment"]["group"] = str(
            config.get("run_name", DEFAULT_RUN_NAME)
        )
        runtime_config["experiment"]["name"] = f"{scenario.name}_{args.task}"
        runtime_config["experiment"]["model_name"] = f"{runtime_config['experiment']['group']}/{runtime_config['experiment']['name']}"

        if rank == 0:
            save_runtime_config(str(log_dir), runtime_config)

        global_batch_size = int(downstream_cfg["batch_size"])
        if global_batch_size % max(1, world_size) != 0:
            raise ValueError(
                f"training.batch_size={global_batch_size} must be divisible by world_size={world_size}"
            )
        local_batch_size = global_batch_size // max(1, world_size)
        set_reproducible(int(runtime_config["experiment"]["random_seed"]))

        pipeline_spec = PipelineSpec.from_mapping(runtime_config["pipeline"])
        validate_eump_downstream_spec(pipeline_spec)
        train_domain, preprocessing_spec, batch_schema = build_domain_loader(
            data_config=runtime_config["data"],
            pipeline_spec=pipeline_spec,
            scene_ids=source_scene_ids,
            logger=logger,
            random_seed=int(runtime_config["experiment"]["random_seed"]),
            batch_size=local_batch_size,
            local_rank=local_rank,
            world_size=world_size,
            train=True,
        )
        target_schema = get_single_downstream_target_schema(batch_schema)
        target_kind = (
            "raster"
            if any(target.name == target_schema.name for target in batch_schema.raster_targets)
            else "scalar"
        )
        source_name = target_schema.spec.source
        if source_name is None:
            raise ValueError(f"Target {target_schema.name!r} requires a source for metric inversion")
        source_spec = preprocessing_spec["sources"][source_name]

        eval_domains: dict[str, DomainLoaders | None] = {}
        for target_domain in scenario.targets:
            target_scene_ids = tuple(str(item) for item in scenario_manifest["targets"][target_domain]["scene_ids"])
            domain, _, _ = build_domain_loader(
                data_config=runtime_config["data"],
                pipeline_spec=pipeline_spec,
                scene_ids=target_scene_ids,
                logger=logger,
                random_seed=int(runtime_config["experiment"]["random_seed"]),
                batch_size=local_batch_size,
                local_rank=local_rank,
                world_size=world_size,
                train=False,
                allow_empty=True,
                preprocessing_spec=preprocessing_spec,
                batch_schema=batch_schema,
            )
            eval_domains[target_domain] = domain
        eval_domains[f"source:{scenario.source}"] = build_domain_loader(
            data_config=runtime_config["data"],
            pipeline_spec=pipeline_spec,
            scene_ids=source_scene_ids,
            logger=logger,
            random_seed=int(runtime_config["experiment"]["random_seed"]),
            batch_size=local_batch_size,
            local_rank=local_rank,
            world_size=world_size,
            train=False,
            preprocessing_spec=preprocessing_spec,
            batch_schema=batch_schema,
        )[0]

        encoder_checkpoint_path = resolve_model_artifact_path(
            str(downstream_cfg["encoder_model_name"]),
            str(downstream_cfg["encoder_checkpoint_file"]),
        )
        stagger_model_initialization(config, local_rank, logger)
        model = build_eump_downstream_model(
            model_config=runtime_config["model"],
            batch_schema=batch_schema,
            target_schema=target_schema,
            base_swin_path=os.path.join(MODEL_DIR, runtime_config["experiment"]["pretrain_model"]),
            encoder_source=str(downstream_cfg["encoder_source"]),
            encoder_checkpoint_path=encoder_checkpoint_path,
            load_pretrained=not bool(runtime_config["experiment"]["scratch_backbone"]),
            freeze_encoder=bool(downstream_cfg["freeze_encoder"]),
            map_location=device,
        ).to(device)
        model = wrap_ddp(model, device, local_rank)
        trainable_params = [param for param in model.parameters() if param.requires_grad]
        if not trainable_params:
            raise ValueError("Domain transfer downstream model has no trainable parameters")
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=float(downstream_cfg["lr"]),
            weight_decay=float(downstream_cfg["weight_decay"]),
        )
        scheduler_cfg = normalize_warmup_cosine_scheduler_config(downstream_cfg.get("scheduler"))
        scheduler, scheduler_metadata = build_warmup_cosine_scheduler(
            optimizer=optimizer,
            base_lr=float(downstream_cfg["lr"]),
            num_batches=len(train_domain.loader),
            accumulation_steps=int(downstream_cfg["accumulation_steps"]),
            epochs=int(downstream_cfg["epochs"]),
            warmup_fraction=scheduler_cfg.warmup_fraction,
            lr_start_factor=scheduler_cfg.lr_start_factor,
            lr_end_factor=scheduler_cfg.lr_end_factor,
        )
        loss_config = pipeline_spec.loss_config if hasattr(pipeline_spec, "loss_config") else PipelineLossConfig()

        epochs = int(downstream_cfg["epochs"])
        log_every = max(1, int(training_cfg.get("log_every_n_epochs", 1)))
        start_time = time.time()
        model_dir = downstream_model_dir(run_dir, "swin", scenario.name, str(args.task))
        model_path = model_dir / downstream_model_file_name("swin")
        checkpoint_path = model_dir / downstream_checkpoint_file_name("swin")

        if bool(args.eval_only):
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Eval-only downstream checkpoint not found: {checkpoint_path}")
            if rank == 0:
                logger.info(f"Eval-only mode: load final training checkpoint from {checkpoint_path}")
        else:
            for epoch in range(1, epochs + 1):
                avg_loss = train_one_epoch(
                    model=model,
                    loader=train_domain.loader,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    device=device,
                    epoch=epoch,
                    accumulation_steps=int(downstream_cfg["accumulation_steps"]),
                    grad_clip_norm=float(training_cfg.get("grad_clip_norm", 1.0)),
                    target_schema=target_schema,
                    target_kind=target_kind,
                    loss_config=loss_config,
                )
                if rank == 0 and (epoch == 1 or epoch == epochs or epoch % log_every == 0):
                    logger.info(
                        f"DOMAIN TRAIN scenario={scenario.name} task={args.task} epoch={epoch}/{epochs} "
                        f"target={target_schema.name} train_loss={avg_loss:.6f} lr={optimizer.param_groups[0]['lr']:.8f}"
                    )

            if rank == 0:
                model_dir.mkdir(parents=True, exist_ok=True)
                unwrapped = get_model(model)
                torch.save(unwrapped.state_dict(), model_path)
                torch.save(
                    {
                        "epoch": epochs,
                        "model_state_dict": unwrapped.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "scheduler_metadata": scheduler_metadata,
                        "runtime_config": runtime_config,
                        "scenario": scenario.name,
                        "task": args.task,
                        "train_domain": scenario.source,
                    },
                    checkpoint_path,
                )
                logger.info(f"Saved final model to {model_path}")
                logger.info(f"Saved final checkpoint to {checkpoint_path}")

        distributed_barrier(local_rank)
        if bool(args.train_only):
            if rank == 0:
                logger.info("Train-only mode: skip Swin domain evaluation")
            return 0
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise KeyError(f"Final training checkpoint has no model_state_dict: {checkpoint_path}")
        get_model(model).load_state_dict(checkpoint["model_state_dict"])

        result_rows: list[dict[str, Any]] = []
        metric_scale = str(config.get("metric_scale", "physical")).lower()
        year_range = "-".join(str(item) for item in pipeline_spec.year_range)
        for eval_domain_name, domain in eval_domains.items():
            if domain is None:
                if rank == 0:
                    message = (
                        "No compatible processed rows remained after filtering this domain "
                        f"for task={args.task} target={target_schema.name}."
                    )
                    result_rows.extend(
                        empty_eval_result_rows(
                            scenario_name=scenario.name,
                            train_domain=scenario.source,
                            eval_domain=eval_domain_name,
                            task=str(args.task),
                            target=target_schema.name,
                            target_kind=target_kind,
                            year_range=year_range,
                            model_path=model_path,
                            message=message,
                        )
                    )
                    logger.warning(
                        f"DOMAIN EVAL skipped scenario={scenario.name} task={args.task} "
                        f"train={scenario.source} eval={eval_domain_name}: {message}"
                    )
                continue
            metrics_by_scope = summarize_domain_metrics(
                model=model,
                domain=domain,
                device=device,
                target_schema=target_schema,
                target_kind=target_kind,
                source_spec=source_spec,
                metric_scale=metric_scale,
            )
            if rank == 0:
                for scope, metrics in metrics_by_scope.items():
                    row = {
                        "model": "swin",
                        "scenario": scenario.name,
                        "train_domain": scenario.source,
                        "eval_domain": eval_domain_name,
                        "task": args.task,
                        "target": target_schema.name,
                        "scope": scope,
                        "rmse": metrics.get("rmse"),
                        "mae": metrics.get("mae"),
                        "r2": metrics.get("r2"),
                        "count": metrics.get("count"),
                        "city_count": domain.city_count,
                        "row_count": domain.row_count,
                        "year_range": year_range,
                        "model_path": str(model_path),
                        "status": "ok",
                        "message": "",
                    }
                    result_rows.append(row)
                    logger.info(
                        "DOMAIN EVAL scenario={scenario} task={task} train={train} eval={eval} "
                        "target={target} scope={scope} rmse={rmse} r2={r2} count={count}".format(
                            scenario=scenario.name,
                            task=args.task,
                            train=scenario.source,
                            eval=eval_domain_name,
                            target=target_schema.name,
                            scope=scope,
                            rmse=("null" if row["rmse"] is None else f"{float(row['rmse']):.6f}"),
                            r2=("null" if row["r2"] is None else f"{float(row['r2']):.6f}"),
                            count=row["count"],
                        )
                    )
        if rank == 0:
            append_result_rows(run_dir, result_rows)
            logger.info(
                f"Finished domain transfer worker in {(time.time() - start_time) / 60.0:.2f} min: "
                f"scenario={scenario.name} task={args.task}"
            )
    finally:
        destroy_distributed_process_group()
    return 0


def orchestrator_main(args: argparse.Namespace) -> int:
    config, hydra_overrides = load_experiment_config(args.config, args.overrides)
    scenarios = scenario_specs_from_config(config)
    model_names = selected_models(config, args.models)
    task_names = selected_tasks(config, args.tasks)
    scenario_names = selected_scenarios(scenarios, args.scenarios)
    if not model_names:
        raise ValueError("No models selected")
    if not task_names:
        raise ValueError("No tasks selected")
    if not scenario_names:
        raise ValueError("No scenarios selected")

    run_name = str(config.get("run_name", DEFAULT_RUN_NAME))
    output_root = repo_path(config.get("output_root", EXPERIMENT_DIR / "outputs"))
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "model").mkdir(parents=True, exist_ok=True)
    commands_log = run_dir / "commands.log"
    continue_run = bool(args.continue_run or args.skip_pretrain or args.eval_only or args.models is not None)
    if not continue_run:
        commands_log.write_text("", encoding="utf-8")
        (run_dir / "results.jsonl").write_text("", encoding="utf-8")
    else:
        commands_log.parent.mkdir(parents=True, exist_ok=True)
        commands_log.touch(exist_ok=True)
        (run_dir / "results.jsonl").touch(exist_ok=True)

    pipeline_suite = pipeline_suite_for_model(config, "swin")
    runtime_config = compose_runtime_config(pipeline=f"{pipeline_suite}/{task_names[0]}", overrides=hydra_overrides)
    apply_loader_overrides(runtime_config, config)
    apply_downstream_year_range(runtime_config, config)
    available_scene_ids = available_scene_ids_from_runtime_config(runtime_config)
    selected_scenario_specs = {name: scenarios[name] for name in scenario_names}
    labels = load_domain_labels(
        repo_path(config["label_csv"]),
        required_columns=required_label_columns_for_scenarios(scenarios),
    )
    domain_manifest, join_report = build_domain_manifest(
        labels=labels,
        available_scene_ids=available_scene_ids,
        scenarios=selected_scenario_specs,
    )
    runtime_payload = {
        "config": config,
        "hydra_overrides": hydra_overrides,
        "selected_models": list(model_names),
        "selected_tasks": list(task_names),
        "selected_scenarios": list(scenario_names),
        "skip_pretrain": bool(args.skip_pretrain),
        "eval_only": bool(args.eval_only),
        "continue_run": continue_run,
        "dry_run": bool(os.environ.get("DRY_RUN")),
    }
    save_json(run_dir / "runtime_config.yaml", runtime_payload)
    save_json(run_dir / "domain_manifest.json", domain_manifest)
    save_json(run_dir / "domain_join_report.json", join_report)
    if not continue_run:
        write_summary_csv(run_dir)

    print(f"Domain transfer run_dir: {run_dir}")
    print(f"Selected models: {', '.join(model_names)}")
    print(f"Selected scenarios: {', '.join(scenario_names)}")
    print(f"Selected tasks: {', '.join(task_names)}")
    for scenario_name in scenario_names:
        payload = domain_manifest["scenarios"][scenario_name]
        target_counts = ", ".join(
            f"{name}={target['scene_count']}" for name, target in payload["targets"].items()
        )
        print(
            f"Scenario {scenario_name}: train {payload['source']}={payload['source_scene_count']} "
            f"test {target_counts}"
        )

    dry_run = bool(os.environ.get("DRY_RUN"))
    for scenario_name in scenario_names:
        if (
            "swin" in model_names
            and bool(config.get("pretrain", {}).get("enabled", True))
            and not bool(args.skip_pretrain)
            and not bool(args.eval_only)
        ):
            cmd, env = worker_command(
                args=args,
                run_dir=run_dir,
                scenario_name=scenario_name,
                stage="pretrain",
                model_kind="swin",
                resource_profile="train",
            )
            run_command(cmd, env=env, dry_run=dry_run, commands_log=commands_log)
        for model_kind in model_names:
            for task in task_names:
                if bool(args.eval_only):
                    cmd, env = worker_command(
                        args=args,
                        run_dir=run_dir,
                        scenario_name=scenario_name,
                        stage="downstream",
                        model_kind=model_kind,
                        task=task,
                        resource_profile="eval",
                        eval_only=True,
                    )
                    run_command(cmd, env=env, dry_run=dry_run, commands_log=commands_log)
                    continue

                train_cmd, train_env = worker_command(
                    args=args,
                    run_dir=run_dir,
                    scenario_name=scenario_name,
                    stage="downstream",
                    model_kind=model_kind,
                    task=task,
                    resource_profile="train",
                    train_only=True,
                )
                run_command(train_cmd, env=train_env, dry_run=dry_run, commands_log=commands_log)
                eval_cmd, eval_env = worker_command(
                    args=args,
                    run_dir=run_dir,
                    scenario_name=scenario_name,
                    stage="downstream",
                    model_kind=model_kind,
                    task=task,
                    resource_profile="eval",
                    eval_only=True,
                )
                run_command(eval_cmd, env=eval_env, dry_run=dry_run, commands_log=commands_log)
    if dry_run:
        print("DRY_RUN=1: commands were written but not executed.")
    else:
        print(f"Saved domain transfer summary: {run_dir / 'summary.csv'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker:
        return worker_main(args)
    return orchestrator_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
