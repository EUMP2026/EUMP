from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


SUPPORTED_JOB_KINDS = {
    "swin_pretrain",
    "swin_downstream",
    "city_scalar_mlp",
    "city_scalar_cache",
    "city_scalar_cv",
}


@dataclass(frozen=True, slots=True)
class EumpLaunchJob:
    name: str
    kind: str
    pipeline: str
    extra_overrides: tuple[str, ...] = ()


def default_config_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "configs"


def launch_config_path(launch_name: str, *, config_dir: Path | None = None) -> Path:
    root = default_config_dir() if config_dir is None else Path(config_dir)
    return root / "launch" / f"{launch_name}.yaml"


def _parse_scalar(value: str) -> Any:
    if value == "null":
        return None
    if value == "true":
        return True
    if value == "false":
        return False
    return value


def _load_simple_yaml_mapping(path: Path) -> dict[str, Any]:
    lines: list[tuple[int, str]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        content = raw_line.split("#", 1)[0].rstrip()
        if not content.strip():
            continue
        indent = len(content) - len(content.lstrip(" "))
        lines.append((indent, content.strip()))

    def parse_block(index: int, indent: int) -> tuple[Any, int]:
        if index >= len(lines) or lines[index][0] < indent:
            return {}, index
        is_list = lines[index][1].startswith("- ")
        if is_list:
            values: list[Any] = []
            while index < len(lines) and lines[index][0] == indent and lines[index][1].startswith("- "):
                item = lines[index][1][2:].strip()
                if item:
                    values.append(_parse_scalar(item))
                    index += 1
                else:
                    child_indent = lines[index + 1][0] if index + 1 < len(lines) else indent + 2
                    child, index = parse_block(index + 1, child_indent)
                    values.append(child)
            return values, index

        values: dict[str, Any] = {}
        while index < len(lines) and lines[index][0] == indent and not lines[index][1].startswith("- "):
            key, separator, value = lines[index][1].partition(":")
            if not separator:
                raise ValueError(f"Unsupported YAML line in {path}: {lines[index][1]!r}")
            key = key.strip()
            value = value.strip()
            if value:
                values[key] = _parse_scalar(value)
                index += 1
                continue
            if index + 1 < len(lines) and lines[index + 1][0] > indent:
                child, index = parse_block(index + 1, lines[index + 1][0])
            else:
                child = {}
                index += 1
            values[key] = child
        return values, index

    parsed, final_index = parse_block(0, lines[0][0] if lines else 0)
    if final_index != len(lines) or not isinstance(parsed, dict):
        raise ValueError(f"Unable to parse launch YAML file {path}")
    return parsed


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        from omegaconf import OmegaConf
    except ModuleNotFoundError:
        return _load_simple_yaml_mapping(path)
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
    if not isinstance(payload, dict):
        raise TypeError(f"YAML file {path} must be a mapping")
    return payload


def _load_launch_config(launch_name: str, *, config_dir: Path | None = None) -> dict[str, Any]:
    path = launch_config_path(launch_name, config_dir=config_dir)
    if not path.exists():
        raise ValueError(f"Unknown launch config {launch_name!r}: {path} does not exist")
    payload = _load_yaml_mapping(path)
    launch = payload.get("launch")
    if not isinstance(launch, dict):
        raise ValueError(
            f"launch={launch_name} is not a valid launch config. "
            "Pass a launch config such as launch=eump_full/mae_v3 to the launcher."
        )
    return payload


def _mode_table(launch: dict[str, Any], *, phase: str) -> tuple[str, dict[str, Any]]:
    if phase == "train" or phase == "verify":
        key = "train_modes"
    elif phase == "eval":
        key = "eval_modes"
    else:
        raise ValueError(f"phase must be one of train, eval, verify; got {phase!r}")
    modes = launch.get(key)
    if not isinstance(modes, dict):
        raise ValueError(f"launch.{key} must be a mapping")
    return key, modes


def _mode_jobs(launch: dict[str, Any], *, mode: str | None, phase: str) -> tuple[str, list[str]]:
    mode_key, modes = _mode_table(launch, phase=phase)
    selected_mode = mode or str(launch.get("default_mode", "all"))
    if phase == "verify" and selected_mode == "verify-data":
        selected_mode = str(launch.get("default_mode", "all"))
    if selected_mode not in modes:
        supported = ", ".join(sorted(modes))
        raise ValueError(
            f"Unsupported {phase} mode {selected_mode!r} for this suite. "
            f"Supported modes from launch.{mode_key}: {supported}"
        )
    names = modes[selected_mode]
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise TypeError(f"launch.{mode_key}.{selected_mode} must be a list of job names")
    return selected_mode, list(names)


def _jobs_mapping(launch: dict[str, Any]) -> dict[str, Any]:
    jobs = launch.get("jobs")
    if not isinstance(jobs, dict):
        raise ValueError("launch.jobs must be a mapping")
    return jobs


def _expand_job(job_name: str, job: Any) -> list[EumpLaunchJob]:
    if not isinstance(job, dict):
        raise TypeError(f"launch.jobs.{job_name} must be a mapping")
    kind = str(job.get("kind", "")).strip()
    if kind not in SUPPORTED_JOB_KINDS:
        supported = ", ".join(sorted(SUPPORTED_JOB_KINDS))
        raise ValueError(f"launch.jobs.{job_name}.kind must be one of {supported}, got {kind!r}")
    if kind == "city_scalar_cache":
        pipelines = job.get("pipelines")
        if pipelines is None:
            pipeline = job.get("pipeline")
            pipelines = [] if pipeline is None else [pipeline]
        if not isinstance(pipelines, list) or not all(isinstance(pipeline, str) for pipeline in pipelines):
            raise TypeError(f"launch.jobs.{job_name}.pipelines must be a list of pipeline names")
        if not pipelines:
            raise ValueError(f"launch.jobs.{job_name}.pipelines must not be empty")
        expanded = []
        for index, pipeline in enumerate(pipelines):
            leaf_name = pipeline.rsplit("/", 1)[-1]
            extra_overrides: tuple[str, ...] = ()
            if index > 0:
                extra_overrides = ("pipeline.city_scalar_cache.rebuild_features=false",)
            expanded.append(
                EumpLaunchJob(
                    name=f"{job_name}:{leaf_name}",
                    kind=kind,
                    pipeline=pipeline,
                    extra_overrides=extra_overrides,
                )
            )
        return expanded

    pipeline = job.get("pipeline")
    if not isinstance(pipeline, str) or not pipeline:
        raise ValueError(f"launch.jobs.{job_name}.pipeline must be a non-empty string")
    return [EumpLaunchJob(name=job_name, kind=kind, pipeline=pipeline)]


def resolve_launch_jobs(
    launch_name: str,
    *,
    mode: str | None = None,
    phase: str,
    config_dir: Path | None = None,
) -> tuple[EumpLaunchJob, ...]:
    launch_config = _load_launch_config(launch_name, config_dir=config_dir)
    launch = launch_config["launch"]
    _, job_names = _mode_jobs(launch, mode=mode, phase=phase)
    jobs = _jobs_mapping(launch)
    resolved: list[EumpLaunchJob] = []
    seen_verify_pipelines: set[str] = set()
    for job_name in job_names:
        if job_name not in jobs:
            raise ValueError(f"launch mode references unknown job {job_name!r}")
        for job in _expand_job(job_name, jobs[job_name]):
            if phase == "verify":
                if job.kind == "city_scalar_cache" or job.pipeline in seen_verify_pipelines:
                    continue
                seen_verify_pipelines.add(job.pipeline)
            resolved.append(job)
    return tuple(resolved)


__all__ = [
    "SUPPORTED_JOB_KINDS",
    "EumpLaunchJob",
    "default_config_dir",
    "launch_config_path",
    "resolve_launch_jobs",
]
