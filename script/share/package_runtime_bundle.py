#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any

from hydra import compose, initialize_config_dir


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from urban_embedding.core.paths import resolve_model_artifact_path
from urban_embedding.eump_launch import EumpLaunchJob, resolve_launch_jobs
from urban_embedding.training.evaluation import normalize_evaluation_config


CONFIG_DIR = str(PROJECT_ROOT / "configs")
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "dist" / "research_bundles"
DEFAULT_CONTAINER_WORKDIR = "/workspace/urban_embedding_v2"
DEFAULT_EXTERNAL_EUMP_ROOT = "/mnt/eump_2020"
SUPPORTED_PIPELINE_KIND = "eump_downstream"
SUPPORTED_MODEL_KIND = "swin_downstream_light"
SUPPORTED_LAUNCH_JOB_KIND = "swin_downstream"
SUPPORTED_CHECKPOINT_SUFFIXES = {".pt", ".pth", ".ckpt", ".bin", ".safetensors"}


@dataclass(frozen=True, slots=True)
class BundleFile:
    repo_relative_path: str
    source_path: Path
    kind: str


@dataclass(frozen=True, slots=True)
class BundleTask:
    task_name: str
    pipeline_name: str
    model_name: str
    checkpoint_relative_path: str
    checkpoint_source: str
    encoder_checkpoint_relative_path: str | None
    encoder_source: str | None
    encoder_model_name: str | None
    encoder_checkpoint_file: str | None
    pretrain_backbone: str
    cfg: dict[str, Any]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Package a Docker-based source research bundle for one downstream EUMP "
            "model or for multiple downstream tasks resolved from a launch config."
        )
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help=(
            "Single-task mode only. Checkpoint file to share, relative to the "
            "repository root or absolute. Expected form: model/<group>/<name>/<file>."
        ),
    )
    parser.add_argument(
        "--pipeline",
        default=None,
        help="Single-task mode only. Runnable downstream pipeline name, for example eump_full/mae_v2/lst.",
    )
    parser.add_argument(
        "--launch",
        default=None,
        help="Multi-task mode. Launch config name, for example eump_full/mae_v2.",
    )
    parser.add_argument(
        "--jobs",
        nargs="+",
        default=None,
        help=(
            "Multi-task mode. Explicit downstream job names from the launch config, "
            "for example lst precip traffic."
        ),
    )
    parser.add_argument(
        "--launch-mode",
        default="all",
        help=(
            "Multi-task mode. Launch eval mode used when --jobs is omitted. "
            "Default: all."
        ),
    )
    parser.add_argument(
        "--experiment-group",
        default=None,
        help=(
            "Multi-task mode only. Shared experiment.group that contains the "
            "downstream task directories, for example eump_full_mae_v2."
        ),
    )
    parser.add_argument(
        "--pretrain-backbone",
        required=True,
        help="Backbone directory under model/, for example swin_T or swin_L.",
    )
    parser.add_argument(
        "--bundle-name",
        default=None,
        help="Output bundle directory name. Defaults to a name derived from the selected tasks.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Bundle output root. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--checkpoint-source",
        choices=("best_model", "training_checkpoint"),
        default=None,
        help=(
            "Checkpoint source for bundled downstream tasks. In multi-task mode "
            "the default is best_model. In single-task mode the default is inferred "
            "from the selected checkpoint file name."
        ),
    )
    parser.add_argument(
        "--docker-image-tag",
        default=None,
        help="Docker image tag to record in generated instructions. Defaults to a sanitized bundle name.",
    )
    parser.add_argument(
        "--include-path",
        action="append",
        default=[],
        help=(
            "Extra file or directory to include, relative to the repository root or absolute. "
            "Useful for untracked notes or extra experiment scripts."
        ),
    )
    parser.add_argument(
        "--container-workdir",
        default=DEFAULT_CONTAINER_WORKDIR,
        help=f"Container workdir used in generated commands. Default: {DEFAULT_CONTAINER_WORKDIR}",
    )
    parser.add_argument(
        "--external-eump-root",
        default=DEFAULT_EXTERNAL_EUMP_ROOT,
        help=f"Mounted processed EUMP root used in generated commands. Default: {DEFAULT_EXTERNAL_EUMP_ROOT}",
    )
    parser.add_argument(
        "--config-name",
        default="evaluate",
        help="Hydra config name used to resolve downstream defaults. Default: evaluate.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output bundle directory.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help=(
            "Optional Hydra overrides applied while resolving downstream defaults, "
            "for example downstream.encoder_model_name=custom_group/pretrain."
        ),
    )
    return parser.parse_args(argv)


def _repo_abspath(path: Path | str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    try:
        candidate.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(f"Path must stay inside the repository: {path}") from exc
    return candidate


def _resolve_repo_entry(path: Path | str) -> tuple[Path, Path]:
    repo_path = _repo_abspath(path)
    try:
        resolved_path = repo_path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Path does not exist: {repo_path}") from exc
    return repo_path, resolved_path


def _repo_relative(path: Path) -> str:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    try:
        return candidate.relative_to(PROJECT_ROOT).as_posix()
    except ValueError as exc:
        raise ValueError(f"Path is not located under the repository root: {path}") from exc


def _symlink_resolves_inside_repo(path: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(PROJECT_ROOT)
    except ValueError:
        return False
    return True


def _format_skipped_symlink(path: Path) -> str:
    raw_target = os.readlink(path)
    resolved_target = path.resolve(strict=False)
    return f"{_repo_relative(path)} -> {raw_target} (resolved: {resolved_target})"


def _skipped_entry_repo_path(entry: str) -> str:
    return entry.split(" -> ", maxsplit=1)[0]


def _sanitize_bundle_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    if not cleaned:
        raise ValueError(f"Unable to derive a bundle name from {value!r}")
    return cleaned


def _sanitize_docker_tag(value: str) -> str:
    lowered = value.lower().replace("_", "-")
    lowered = re.sub(r"[^a-z0-9./-]+", "-", lowered)
    lowered = re.sub(r"-{2,}", "-", lowered).strip("-")
    if not lowered:
        raise ValueError(f"Unable to derive a docker image tag from {value!r}")
    return lowered


def _infer_checkpoint_source(checkpoint_name: str, explicit: str | None) -> str:
    if explicit:
        return str(explicit)
    stem = Path(checkpoint_name).stem.lower()
    return "training_checkpoint" if "checkpoint" in stem else "best_model"


def _default_multi_checkpoint_source(explicit: str | None) -> str:
    return str(explicit) if explicit else "best_model"


def _git_bundle_sources() -> tuple[list[Path], list[str]]:
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--modified",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
    )
    paths: list[Path] = []
    skipped_external_symlinks: list[str] = []
    seen: set[str] = set()
    for raw_path in result.stdout.decode("utf-8").split("\x00"):
        if not raw_path or raw_path in seen:
            continue
        seen.add(raw_path)
        repo_path = PROJECT_ROOT / raw_path
        if repo_path.is_symlink() and not _symlink_resolves_inside_repo(repo_path):
            skipped_external_symlinks.append(_format_skipped_symlink(repo_path))
            continue
        paths.append(repo_path)
    return paths, skipped_external_symlinks


def _bundle_files_for_repo_entry(path: Path | str, *, kind: str) -> list[BundleFile]:
    repo_path, resolved_path = _resolve_repo_entry(path)
    repo_relative_path = _repo_relative(repo_path)
    if resolved_path.is_file():
        return [BundleFile(repo_relative_path, resolved_path, kind)]
    if resolved_path.is_dir():
        bundle_files: list[BundleFile] = []
        for source_path in sorted(candidate for candidate in resolved_path.rglob("*") if candidate.is_file()):
            suffix = source_path.relative_to(resolved_path).as_posix()
            destination_relative = f"{repo_relative_path}/{suffix}"
            bundle_files.append(BundleFile(destination_relative, source_path, kind))
        return bundle_files
    raise FileNotFoundError(f"Path does not exist: {repo_path}")


def _filter_skipped_repo_entries(
    skipped_repo_entries: list[str],
    bundle_files: "OrderedDict[str, BundleFile]",
) -> list[str]:
    included_paths = set(bundle_files)
    filtered: list[str] = []
    for entry in skipped_repo_entries:
        repo_path = _skipped_entry_repo_path(entry)
        if repo_path in included_paths:
            continue
        if any(path.startswith(f"{repo_path}/") for path in included_paths):
            continue
        filtered.append(entry)
    return filtered


def _compose_downstream_config(
    *,
    config_name: str,
    pipeline_name: str,
    experiment_group: str,
    experiment_name: str,
    pretrain_backbone: str,
    overrides: list[str],
) -> dict[str, Any]:
    resolved_overrides = [
        f"pipeline={pipeline_name}",
        f"experiment.group={experiment_group}",
        f"experiment.name={experiment_name}",
        f"experiment.pretrain_model={pretrain_backbone}",
        *overrides,
    ]
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name=config_name, overrides=resolved_overrides)
    normalized = normalize_evaluation_config(cfg)
    pipeline_kind = str(normalized["pipeline"]["kind"])
    model_kind = str(normalized["model"]["kind"])
    if pipeline_kind != SUPPORTED_PIPELINE_KIND:
        raise ValueError(
            "package_runtime_bundle.py currently supports only downstream Swin bundles. "
            f"Resolved pipeline.kind={pipeline_kind!r}"
        )
    if model_kind != SUPPORTED_MODEL_KIND:
        raise ValueError(
            "package_runtime_bundle.py currently supports only downstream Swin bundles. "
            f"Resolved model.kind={model_kind!r}"
        )
    return normalized


def _validate_checkpoint_candidate(relative_path: str) -> None:
    parts = Path(relative_path).parts
    if len(parts) != 4 or parts[0] != "model":
        raise ValueError(
            "checkpoint-path must look like model/<group>/<name>/<file>, "
            f"got {relative_path!r}"
        )
    suffix = Path(relative_path).suffix.lower()
    if suffix not in SUPPORTED_CHECKPOINT_SUFFIXES:
        raise ValueError(
            "checkpoint-path must point to a model file with one of "
            f"{sorted(SUPPORTED_CHECKPOINT_SUFFIXES)}, got {relative_path!r}"
        )


def _single_task_model_name(relative_path: str) -> str:
    _validate_checkpoint_candidate(relative_path)
    parts = Path(relative_path).parts
    return f"{parts[1]}/{parts[2]}"


def _resolved_checkpoint_relative_path(*, model_name: str, checkpoint_file: str) -> str:
    resolved = resolve_model_artifact_path(model_name, checkpoint_file)
    if resolved is None:
        raise ValueError(f"Unable to resolve checkpoint path for model={model_name!r} file={checkpoint_file!r}")
    repo_path = _repo_abspath(Path(resolved))
    if not repo_path.is_file():
        raise FileNotFoundError(f"Checkpoint file does not exist: {repo_path}")
    relative = _repo_relative(repo_path)
    _validate_checkpoint_candidate(relative)
    return relative


def _resolve_encoder_checkpoint(cfg: dict[str, Any]) -> tuple[str | None, str | None, str | None, str | None]:
    encoder_source = str(cfg["downstream"]["encoder_source"])
    if encoder_source != "reconstruction_checkpoint":
        return None, encoder_source, None, None
    encoder_model_name = str(cfg["downstream"]["encoder_model_name"])
    encoder_checkpoint_file = str(cfg["downstream"]["encoder_checkpoint_file"])
    relative = _resolved_checkpoint_relative_path(
        model_name=encoder_model_name,
        checkpoint_file=encoder_checkpoint_file,
    )
    return relative, encoder_source, encoder_model_name, encoder_checkpoint_file


def _single_task_bundle(args: argparse.Namespace) -> list[BundleTask]:
    checkpoint_repo_path = _repo_abspath(args.checkpoint_path)
    if not checkpoint_repo_path.is_file():
        raise FileNotFoundError(f"checkpoint-path does not exist: {checkpoint_repo_path}")
    checkpoint_relative_path = _repo_relative(checkpoint_repo_path)
    model_name = _single_task_model_name(checkpoint_relative_path)
    experiment_group, experiment_name = model_name.split("/", maxsplit=1)
    cfg = _compose_downstream_config(
        config_name=args.config_name,
        pipeline_name=str(args.pipeline),
        experiment_group=experiment_group,
        experiment_name=experiment_name,
        pretrain_backbone=str(args.pretrain_backbone),
        overrides=list(args.overrides),
    )
    checkpoint_source = _infer_checkpoint_source(checkpoint_repo_path.name, args.checkpoint_source)
    encoder_checkpoint_relative_path, encoder_source, encoder_model_name, encoder_checkpoint_file = _resolve_encoder_checkpoint(cfg)
    return [
        BundleTask(
            task_name=experiment_name,
            pipeline_name=str(args.pipeline),
            model_name=model_name,
            checkpoint_relative_path=checkpoint_relative_path,
            checkpoint_source=checkpoint_source,
            encoder_checkpoint_relative_path=encoder_checkpoint_relative_path,
            encoder_source=encoder_source,
            encoder_model_name=encoder_model_name,
            encoder_checkpoint_file=encoder_checkpoint_file,
            pretrain_backbone=str(cfg["experiment"]["pretrain_model"]),
            cfg=cfg,
        )
    ]


def _resolve_launch_selection(args: argparse.Namespace) -> tuple[str, list[EumpLaunchJob]]:
    available_jobs = resolve_launch_jobs(str(args.launch), mode="all", phase="eval")
    downstream_jobs = [job for job in available_jobs if job.kind == SUPPORTED_LAUNCH_JOB_KIND]
    jobs_by_name = {job.name: job for job in downstream_jobs}
    if args.jobs:
        selected_names: list[str] = []
        for name in args.jobs:
            if name not in jobs_by_name:
                supported = ", ".join(sorted(jobs_by_name))
                raise ValueError(
                    f"Unknown downstream launch job {name!r} for launch={args.launch!r}. "
                    f"Supported downstream jobs: {supported}"
                )
            if name not in selected_names:
                selected_names.append(name)
        return "explicit_jobs", [jobs_by_name[name] for name in selected_names]

    resolved_jobs = resolve_launch_jobs(str(args.launch), mode=str(args.launch_mode), phase="eval")
    selected_jobs = [job for job in resolved_jobs if job.kind == SUPPORTED_LAUNCH_JOB_KIND]
    if not selected_jobs:
        raise ValueError(
            f"launch={args.launch!r} mode={args.launch_mode!r} does not resolve any "
            f"{SUPPORTED_LAUNCH_JOB_KIND!r} jobs"
        )
    return str(args.launch_mode), selected_jobs


def _multi_task_bundle(args: argparse.Namespace) -> tuple[list[BundleTask], str]:
    if not args.experiment_group:
        raise ValueError("--experiment-group is required when --launch is used")
    selection_label, selected_jobs = _resolve_launch_selection(args)
    checkpoint_source = _default_multi_checkpoint_source(args.checkpoint_source)
    tasks: list[BundleTask] = []
    for job in selected_jobs:
        cfg = _compose_downstream_config(
            config_name=args.config_name,
            pipeline_name=job.pipeline,
            experiment_group=str(args.experiment_group),
            experiment_name=job.name,
            pretrain_backbone=str(args.pretrain_backbone),
            overrides=[*job.extra_overrides, *args.overrides],
        )
        checkpoint_file = (
            str(cfg["downstream"]["save_file"])
            if checkpoint_source == "best_model"
            else str(cfg["downstream"]["checkpoint_file"])
        )
        checkpoint_relative_path = _resolved_checkpoint_relative_path(
            model_name=str(cfg["experiment"]["model_name"]),
            checkpoint_file=checkpoint_file,
        )
        encoder_checkpoint_relative_path, encoder_source, encoder_model_name, encoder_checkpoint_file = _resolve_encoder_checkpoint(cfg)
        tasks.append(
            BundleTask(
                task_name=job.name,
                pipeline_name=job.pipeline,
                model_name=str(cfg["experiment"]["model_name"]),
                checkpoint_relative_path=checkpoint_relative_path,
                checkpoint_source=checkpoint_source,
                encoder_checkpoint_relative_path=encoder_checkpoint_relative_path,
                encoder_source=encoder_source,
                encoder_model_name=encoder_model_name,
                encoder_checkpoint_file=encoder_checkpoint_file,
                pretrain_backbone=str(cfg["experiment"]["pretrain_model"]),
                cfg=cfg,
            )
        )
    return tasks, selection_label


def _resolve_tasks(args: argparse.Namespace) -> tuple[list[BundleTask], dict[str, Any]]:
    single_mode = args.checkpoint_path is not None or args.pipeline is not None
    launch_mode = args.launch is not None
    if single_mode and launch_mode:
        raise ValueError("Use either single-task mode (--checkpoint-path/--pipeline) or multi-task mode (--launch), not both")
    if not single_mode and not launch_mode:
        raise ValueError(
            "Select one mode: either pass --checkpoint-path and --pipeline, or pass --launch"
        )
    if launch_mode:
        if args.checkpoint_path is not None or args.pipeline is not None:
            raise ValueError("--checkpoint-path and --pipeline are not used with --launch")
        tasks, selection_label = _multi_task_bundle(args)
        return tasks, {
            "bundle_mode": "launch_bundle",
            "launch_name": str(args.launch),
            "selection": selection_label,
        }
    if args.checkpoint_path is None or args.pipeline is None:
        raise ValueError("Single-task mode requires both --checkpoint-path and --pipeline")
    tasks = _single_task_bundle(args)
    return tasks, {
        "bundle_mode": "single_task",
        "launch_name": None,
        "selection": None,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_file(bundle_file: BundleFile, destination_root: Path) -> None:
    destination = destination_root / bundle_file.repo_relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if bundle_file.source_path.is_symlink():
        destination.symlink_to(os.readlink(bundle_file.source_path))
        return
    shutil.copy2(bundle_file.source_path, destination)


def _docker_shell_command(*, docker_image_tag: str, container_workdir: str, external_eump_root: str) -> str:
    return " ".join(
        [
            "docker run --rm -it --gpus all",
            f"-v \"$PWD\":{container_workdir}",
            f"-v /absolute/path/to/eump_2020:{external_eump_root}:ro",
            f"-w {container_workdir}",
            docker_image_tag,
            "bash",
        ]
    )


def _evaluation_command(task: BundleTask, *, external_eump_root: str) -> str:
    cfg = task.cfg
    return " ".join(
        [
            "CUDA_VISIBLE_DEVICES=0",
            "torchrun --nproc_per_node=1 --nnodes=1",
            "main/evaluate_eump.py",
            f"pipeline={task.pipeline_name}",
            f"experiment.group={cfg['experiment']['group']}",
            f"experiment.name={cfg['experiment']['name']}",
            f"experiment.pretrain_model={cfg['experiment']['pretrain_model']}",
            f"evaluation.checkpoint_file={Path(task.checkpoint_relative_path).name}",
            f"evaluation.checkpoint_source={task.checkpoint_source}",
            f"data.eump_root={external_eump_root}",
            f"data.preprocessing_path={external_eump_root}/manifest/stats/preprocessing_v1.json",
            f"data.split_manifest_path={external_eump_root}/manifest/splits/legacy_compatible_seed42.json",
            "downstream.batch_size=32",
            "pipeline.num_workers=4",
        ]
    )


def _counterfactual_command(task: BundleTask, *, external_eump_root: str) -> str:
    cfg = task.cfg
    return " ".join(
        [
            "python script/experiments/run_eump_input_counterfactual.py",
            "--source NDVI",
            "--delta 0.05",
            "--mask-scope city",
            "--min-value -1.0",
            "--max-value 1.0",
            "--limit 8",
            "--batch-size 2",
            f"--output-dir outputs/{task.task_name}_ndvi_counterfactual",
            f"pipeline={task.pipeline_name}",
            f"experiment.group={cfg['experiment']['group']}",
            f"experiment.name={cfg['experiment']['name']}",
            f"experiment.pretrain_model={cfg['experiment']['pretrain_model']}",
            f"evaluation.checkpoint_file={Path(task.checkpoint_relative_path).name}",
            f"evaluation.checkpoint_source={task.checkpoint_source}",
            f"data.eump_root={external_eump_root}",
            f"data.preprocessing_path={external_eump_root}/manifest/stats/preprocessing_v1.json",
            f"data.split_manifest_path={external_eump_root}/manifest/splits/legacy_compatible_seed42.json",
            "pipeline.num_workers=0",
            "'pipeline.year_range=[2020,2020]'",
        ]
    )


def _build_manifest(
    *,
    bundle_name: str,
    docker_image_tag: str,
    bundle_meta: dict[str, Any],
    tasks: list[BundleTask],
    pretrain_backbone: str,
    external_eump_root: str,
    container_workdir: str,
    bundle_root: Path,
    commands: dict[str, Any],
    kinds_by_path: dict[str, str],
    skipped_repo_entries: list[str],
) -> dict[str, Any]:
    file_entries: list[dict[str, Any]] = []
    for file_path in sorted(candidate for candidate in bundle_root.rglob("*") if candidate.is_file()):
        relative_path = file_path.relative_to(bundle_root).as_posix()
        file_entries.append(
            {
                "path": relative_path,
                "kind": kinds_by_path.get(relative_path, "source"),
                "size_bytes": file_path.stat().st_size,
                "sha256": _sha256(file_path),
            }
        )

    return {
        "bundle_name": bundle_name,
        "bundle_mode": bundle_meta["bundle_mode"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "launch": {
            "name": bundle_meta["launch_name"],
            "selection": bundle_meta["selection"],
        },
        "docker": {
            "dockerfile": "Dockerfile",
            "image_tag": docker_image_tag,
            "container_workdir": container_workdir,
        },
        "pretrain_backbone": pretrain_backbone,
        "external_mounts": {
            "eump_root": external_eump_root,
            "preprocessing_path": f"{external_eump_root}/manifest/stats/preprocessing_v1.json",
            "split_manifest_path": f"{external_eump_root}/manifest/splits/legacy_compatible_seed42.json",
        },
        "tasks": [
            {
                "task_name": task.task_name,
                "pipeline": task.pipeline_name,
                "model_name": task.model_name,
                "checkpoint": {
                    "source": task.checkpoint_source,
                    "path": task.checkpoint_relative_path,
                },
                "encoder_checkpoint": None if task.encoder_checkpoint_relative_path is None else {
                    "source": task.encoder_source,
                    "path": task.encoder_checkpoint_relative_path,
                    "model_name": task.encoder_model_name,
                    "file": task.encoder_checkpoint_file,
                },
            }
            for task in tasks
        ],
        "commands": commands,
        "skipped_repo_entries": skipped_repo_entries,
        "files": file_entries,
    }


def _bundle_readme(
    *,
    docker_image_tag: str,
    tasks: list[BundleTask],
    bundle_meta: dict[str, Any],
    external_eump_root: str,
    container_workdir: str,
) -> str:
    task_lines = [
        f"- `{task.task_name}`: pipeline=`{task.pipeline_name}`, checkpoint=`{task.checkpoint_relative_path}`"
        for task in tasks
    ]
    encoder_lines = sorted(
        {
            f"- `{task.encoder_checkpoint_relative_path}`"
            for task in tasks
            if task.encoder_checkpoint_relative_path is not None
        }
    )
    evaluation_sections: list[str] = []
    for task in tasks:
        evaluation_sections.extend(
            [
                f"### {task.task_name}",
                "",
                "```bash",
                _evaluation_command(task, external_eump_root=external_eump_root),
                "```",
                "",
            ]
        )

    example_task = tasks[0]
    scope_lines = [
        "Supported:",
        "",
        "- downstream evaluation and input-perturbation studies on compatible EUMP data",
        "- adding new scripts that reuse the current source tree and configs",
        "- exporting embeddings or writing custom analyses on mounted processed data",
        "",
        "Not supported:",
        "",
        "- arbitrary raw external data as a plug-and-play inference input",
        "- a stable public SDK with a generic `predict()` function",
        "- a non-technical one-click runtime",
    ]

    launch_lines: list[str] = []
    if bundle_meta["bundle_mode"] == "launch_bundle":
        launch_lines = [
            "## Bundle Selection",
            "",
            f"- launch config: `{bundle_meta['launch_name']}`",
            f"- selection: `{bundle_meta['selection']}`",
            "",
        ]

    content = [
        "# Research Bundle",
        "",
        "This bundle shares the project source, the selected downstream checkpoints,",
        "the matching encoder assets, and a Docker runtime for research-oriented",
        "experiments on processed EUMP data.",
        "",
        *launch_lines,
        "## Included Downstream Tasks",
        "",
        *task_lines,
        "",
        "## Included Shared Assets",
        "",
        f"- Base Swin directory: `model/{tasks[0].pretrain_backbone}`",
        "- Reference perturbation script: `script/experiments/run_eump_input_counterfactual.py`",
    ]
    if encoder_lines:
        content.extend(["- Matching reconstruction encoder checkpoint(s):", *encoder_lines])
    content.extend(
        [
            "",
            "## Prerequisites",
            "",
            "- Docker",
            "- NVIDIA Container Toolkit",
            f"- A processed EUMP data root mounted to `{external_eump_root}`",
            "",
            "The mounted processed EUMP data root should contain at least:",
            "",
            f"- `{external_eump_root}/manifest/stats/preprocessing_v1.json`",
            f"- `{external_eump_root}/manifest/splits/legacy_compatible_seed42.json`",
            "",
            "## Build The Runtime Image",
            "",
            "Run from the bundle root:",
            "",
            "```bash",
            f"docker build -t {docker_image_tag} .",
            "```",
            "",
            "## Start A Research Shell",
            "",
            "The recommended workflow is to bind-mount this bundle directory so the",
            "recipient can edit or add scripts without rebuilding the image.",
            "",
            "```bash",
            _docker_shell_command(
                docker_image_tag=docker_image_tag,
                container_workdir=container_workdir,
                external_eump_root=external_eump_root,
            ),
            "```",
            "",
            "Inside the container, the project root is already the current working",
            f"directory: `{container_workdir}`.",
            "",
            "## Run The Existing Downstream Evaluation",
            "",
            *evaluation_sections,
            "## Run The NDVI Counterfactual Template",
            "",
            "The example below uses the first bundled task. For other tasks, replace",
            "`pipeline=...` and `experiment.name=...` with the corresponding task entry",
            "listed above.",
            "",
            "```bash",
            _counterfactual_command(example_task, external_eump_root=external_eump_root),
            "```",
            "",
            "## Write New Research Scripts",
            "",
            "You can add your own scripts directly inside this bundle, for example under",
            "`script/experiments/`, and reuse:",
            "",
            "- `urban_embedding.data.processed.dataset.build_eump_dataloaders`",
            "- `urban_embedding.models.eump.swin.downstream.build_eump_downstream_model`",
            "- `urban_embedding.training.checkpointing.load_model_state_for_evaluation`",
            "- `urban_embedding.data.processed.processing.restore_processed_raster_block`",
            "",
            *scope_lines,
        ]
    )
    return "\n".join(content).replace("\n\n\n", "\n\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    tasks, bundle_meta = _resolve_tasks(args)

    if bundle_meta["bundle_mode"] == "single_task":
        default_bundle_name = _sanitize_bundle_name(
            f"{tasks[0].model_name.replace('/', '__')}__{Path(tasks[0].checkpoint_relative_path).stem}"
        )
    else:
        launch_stub = str(bundle_meta["launch_name"]).replace("/", "__")
        task_stub = "__".join(task.task_name for task in tasks)
        default_bundle_name = _sanitize_bundle_name(f"{launch_stub}__{task_stub}")

    bundle_name = args.bundle_name or default_bundle_name
    docker_image_tag = args.docker_image_tag or _sanitize_docker_tag(bundle_name)

    source_files, skipped_repo_entries = _git_bundle_sources()
    bundle_files: "OrderedDict[str, BundleFile]" = OrderedDict()
    for source_path in source_files:
        relative = _repo_relative(source_path)
        bundle_files.setdefault(relative, BundleFile(relative, source_path, "source"))

    included_backbones: set[str] = set()
    for task in tasks:
        for bundle_file in _bundle_files_for_repo_entry(task.checkpoint_relative_path, kind="primary_checkpoint"):
            bundle_files[bundle_file.repo_relative_path] = bundle_file
        if task.encoder_checkpoint_relative_path is not None:
            for bundle_file in _bundle_files_for_repo_entry(
                task.encoder_checkpoint_relative_path,
                kind="encoder_checkpoint",
            ):
                bundle_files[bundle_file.repo_relative_path] = bundle_file
        if task.pretrain_backbone not in included_backbones:
            included_backbones.add(task.pretrain_backbone)
            for bundle_file in _bundle_files_for_repo_entry(
                Path("model") / task.pretrain_backbone,
                kind="backbone",
            ):
                bundle_files[bundle_file.repo_relative_path] = bundle_file

    for include_path in args.include_path:
        for bundle_file in _bundle_files_for_repo_entry(include_path, kind="user_extra"):
            bundle_files[bundle_file.repo_relative_path] = bundle_file

    skipped_repo_entries = _filter_skipped_repo_entries(skipped_repo_entries, bundle_files)

    output_root = Path(args.output_root).expanduser()
    if not output_root.is_absolute():
        output_root = (PROJECT_ROOT / output_root).resolve()
    bundle_root = output_root / bundle_name

    if bundle_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Bundle output already exists: {bundle_root}. Use --overwrite to replace it."
            )
        shutil.rmtree(bundle_root)

    bundle_root.mkdir(parents=True, exist_ok=False)
    for bundle_file in bundle_files.values():
        _copy_file(bundle_file, bundle_root)

    readme_path = bundle_root / "BUNDLE_README.md"
    readme_path.write_text(
        _bundle_readme(
            docker_image_tag=docker_image_tag,
            tasks=tasks,
            bundle_meta=bundle_meta,
            external_eump_root=args.external_eump_root,
            container_workdir=args.container_workdir,
        )
        + "\n",
        encoding="utf-8",
    )

    commands = {
        "docker_build": f"docker build -t {docker_image_tag} .",
        "docker_shell": _docker_shell_command(
            docker_image_tag=docker_image_tag,
            container_workdir=args.container_workdir,
            external_eump_root=args.external_eump_root,
        ),
        "downstream_evaluation": {
            task.task_name: _evaluation_command(task, external_eump_root=args.external_eump_root)
            for task in tasks
        },
        "counterfactual_template": {
            task.task_name: _counterfactual_command(task, external_eump_root=args.external_eump_root)
            for task in tasks
        },
    }

    kinds_by_path = {path: bundle_file.kind for path, bundle_file in bundle_files.items()}
    kinds_by_path["BUNDLE_README.md"] = "generated"

    manifest = _build_manifest(
        bundle_name=bundle_name,
        docker_image_tag=docker_image_tag,
        bundle_meta=bundle_meta,
        tasks=tasks,
        pretrain_backbone=tasks[0].pretrain_backbone,
        external_eump_root=args.external_eump_root,
        container_workdir=args.container_workdir,
        bundle_root=bundle_root,
        commands=commands,
        kinds_by_path=kinds_by_path,
        skipped_repo_entries=skipped_repo_entries,
    )
    manifest_path = bundle_root / "bundle_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    summary = {
        "bundle_root": str(bundle_root),
        "bundle_mode": bundle_meta["bundle_mode"],
        "launch_name": bundle_meta["launch_name"],
        "tasks": [task.task_name for task in tasks],
        "docker_image_tag": docker_image_tag,
        "file_count": len(manifest["files"]),
        "skipped_repo_entries": skipped_repo_entries,
    }
    if skipped_repo_entries:
        print("Skipped repository symlink entries that resolve outside the repository:", file=sys.stderr)
        for entry in skipped_repo_entries:
            print(f"  - {entry}", file=sys.stderr)
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
