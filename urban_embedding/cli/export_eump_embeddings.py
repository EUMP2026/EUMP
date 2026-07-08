from __future__ import annotations

import argparse
import json
import os
import shutil
import zipfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from urban_embedding.core.paths import MODEL_DIR, resolve_model_artifact_path
from urban_embedding.core.reproducibility import seed_worker, seeded_torch_generator
from urban_embedding.data.processed.dataset import (
    ProcessedSceneDataset,
    collate_processed_samples,
)
from urban_embedding.data.processed.pipeline_spec import (
    EUMP_CONTRASTIVE_KIND,
    EUMP_RECONSTRUCTION_KIND,
    PipelineSpec,
    build_active_preprocessing_spec,
    build_eump_batch_schema,
)
from urban_embedding.data.processed.processing import (
    load_preprocessing_spec,
    validate_preprocessing_spec,
    validate_preprocessing_stats_scope,
)
from urban_embedding.data.processed.reader import ProcessedSceneStoreReader
from urban_embedding.models.reconstruct import EumpSwinReconstructionModel
from urban_embedding.models.eump.convnext_unet.reconstruction import EumpConvNextUnetReconstructionModel
from urban_embedding.training.distributed import distributed_barrier as torch_distributed_barrier
from urban_embedding.training.runner import normalize_runtime_config


CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs")
CHECKPOINT_SOURCES = {"auto", "state_dict", "training_checkpoint"}
DEFAULT_WORKERS_PER_GPU = 4
DEFAULT_EXPORT_BATCH_SIZE = 32
DEFAULT_TORCH_THREADS = 1
MAX_MERGE_COPY_ELEMENTS = 4_000_000


@dataclass(frozen=True, slots=True)
class ExportParallelRuntime:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @property
    def distributed(self) -> bool:
        return self.world_size > 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export center-crop EUMP embeddings from a trained reconstruction model. "
            "Hydra overrides may be appended after the options, for example "
            "data=eump_full/server_v1 pipeline=eump_full/mae_v2/pretrain."
        )
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Single scene year to export. Default: 2020 when no year options are provided.",
    )
    parser.add_argument(
        "--years",
        default=None,
        help=(
            "Comma-separated years or ranges to export into the same .npz file, "
            "for example 2018,2020 or 2003-2020. Space-separated years are also "
            "accepted before Hydra overrides."
        ),
    )
    parser.add_argument(
        "--year-range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        default=None,
        help="Inclusive scene year range to export into the same .npz file.",
    )
    parser.add_argument(
        "--experiment-group",
        default=None,
        help="Experiment group under model/. Defaults to the composed config experiment.group.",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Experiment name under model/<group>/. Defaults to the composed config experiment.name.",
    )
    parser.add_argument(
        "--checkpoint-name",
        default=None,
        help=(
            "Alias for the experiment name. If it contains '/', it is treated as a full "
            "model artifact name such as eump_full_mae_v2/pretrain."
        ),
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Full model artifact name under model/, for example eump_full_mae_v2/pretrain.",
    )
    parser.add_argument(
        "--checkpoint-file",
        default=None,
        help="Checkpoint file inside model/<model-name>/, or an absolute/relative checkpoint path.",
    )
    parser.add_argument(
        "--checkpoint-source",
        choices=sorted(CHECKPOINT_SOURCES),
        default="auto",
        help=(
            "How to read checkpoint-file. Use state_dict for eump_reconstruction_model.pt, "
            "training_checkpoint for eump_reconstruction_checkpoint.pt, or auto. Default: auto."
        ),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output .npz path. Default: next to the checkpoint.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Per-process export batch size. Default: 32. This is intentionally "
            "lower than reconstruction training batch size because export batches "
            "materialize full raster crops and masks before pooling embeddings."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help=(
            "DataLoader workers for this export process. Overrides --workers-per-gpu. "
            "Default: 4 workers for the active export GPU/process."
        ),
    )
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=DEFAULT_WORKERS_PER_GPU,
        help=(
            "Default DataLoader workers per active export GPU/process when --num-workers "
            "is not set. Default: 4."
        ),
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=DEFAULT_TORCH_THREADS,
        help="Torch CPU threads per export process. Default: 1 to avoid CPU oversubscription.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device: auto, cpu, cuda, cuda:0, etc. Default: auto.",
    )
    parser.add_argument(
        "--scene-ids",
        nargs="*",
        default=None,
        help="Optional scene IDs to export. Default: all compatible scenes for the selected years.",
    )
    parser.add_argument(
        "--keep-spatial",
        action="store_true",
        help="Keep embedding shape [N, C, H, W] instead of average-pooling the last two dimensions.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output-path if it already exists.",
    )
    parser.add_argument(
        "--keep-shards",
        action="store_true",
        help="Keep per-rank temporary shard directories after export merge.",
    )
    parser.add_argument(
        "--compress-output",
        action="store_true",
        help=(
            "Deflate-compress the final .npz. This can be much slower and more CPU-heavy; "
            "the default is an uncompressed .npz for large exports."
        ),
    )
    parser.add_argument(
        "--config-name",
        default="train",
        help="Hydra config name to compose. Default: train.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional Hydra overrides, e.g. data=eump_full/raw_sample pipeline=eump_full/mae_v2/pretrain.",
    )
    args = parser.parse_args(argv)
    if args.years is not None:
        extra_year_tokens = []
        while args.overrides and _looks_like_year_token(args.overrides[0]):
            extra_year_tokens.append(args.overrides.pop(0))
        if extra_year_tokens:
            args.years = ",".join([str(args.years), *extra_year_tokens])
    return args


def _looks_like_year_token(value: str) -> bool:
    text = str(value).strip()
    if not text:
        return False
    return all(piece.strip().isdigit() for piece in text.split("-", maxsplit=1))


def _parse_year_list(value: str) -> list[int]:
    years: set[int] = set()
    for raw_token in str(value).split(","):
        token = raw_token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = (part.strip() for part in token.split("-", maxsplit=1))
            if not start_text.isdigit() or not end_text.isdigit():
                raise ValueError(f"Invalid year range token {token!r}")
            start, end = int(start_text), int(end_text)
            if start > end:
                raise ValueError(f"Year range START must be <= END, got {token!r}")
            years.update(range(start, end + 1))
        else:
            if not token.isdigit():
                raise ValueError(f"Invalid year token {token!r}")
            years.add(int(token))
    if not years:
        raise ValueError("--years must contain at least one year")
    return sorted(years)


def resolve_export_years(args: argparse.Namespace) -> list[int]:
    selectors = [
        args.year is not None,
        args.years is not None,
        args.year_range is not None,
    ]
    if sum(selectors) > 1:
        raise ValueError("Use only one of --year, --years, or --year-range")
    if args.year_range is not None:
        start, end = (int(args.year_range[0]), int(args.year_range[1]))
        if start > end:
            raise ValueError(f"--year-range START must be <= END, got {start} {end}")
        return list(range(start, end + 1))
    if args.years is not None:
        return _parse_year_list(args.years)
    return [2020 if args.year is None else int(args.year)]


def compose_runtime_config(config_name: str, overrides: list[str]) -> dict[str, Any]:
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)
    return normalize_runtime_config(cfg)


def resolve_export_model_name(cfg: dict[str, Any], args: argparse.Namespace) -> str:
    if args.model_name:
        return _clean_model_name(args.model_name)

    if args.checkpoint_name and _looks_like_model_name(args.checkpoint_name):
        return _clean_model_name(args.checkpoint_name)

    group = args.experiment_group or cfg["experiment"]["group"]
    name = args.experiment_name or args.checkpoint_name or cfg["experiment"]["name"]
    return _clean_model_name(f"{group}/{name}")


def _looks_like_model_name(value: str) -> bool:
    return "/" in value or "\\" in value


def _clean_model_name(value: str) -> str:
    model_name = str(value).strip().strip("/\\")
    if not model_name:
        raise ValueError("Model name is empty")
    return model_name.replace("\\", "/")


def resolve_checkpoint_path(
    *,
    cfg: dict[str, Any],
    model_name: str,
    checkpoint_file: str | None,
) -> Path:
    file_name = checkpoint_file or cfg["reconstruct"].get("save_file", "eump_reconstruction_model.pt")
    path = Path(str(file_name)).expanduser()
    if path.is_absolute() or len(path.parts) > 1:
        return path.resolve()
    resolved = resolve_model_artifact_path(model_name, str(file_name))
    if resolved is None:
        raise ValueError("checkpoint_file resolved to None")
    return Path(resolved).resolve()


def _years_label(years: list[int]) -> str:
    if not years:
        raise ValueError("years must not be empty")
    ordered = sorted({int(year) for year in years})
    if len(ordered) == 1:
        return str(ordered[0])
    if ordered == list(range(ordered[0], ordered[-1] + 1)):
        return f"{ordered[0]}-{ordered[-1]}"
    return "_".join(str(year) for year in ordered)


def default_output_path(*, checkpoint_path: Path, years: list[int], pooled: bool) -> Path:
    suffix = "pooled" if pooled else "spatial"
    return checkpoint_path.with_name(f"{checkpoint_path.stem}_embeddings_{_years_label(years)}_{suffix}.npz")


def _env_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None or str(raw_value).strip() == "":
        return int(default)
    return int(raw_value)


def resolve_parallel_runtime() -> ExportParallelRuntime:
    return ExportParallelRuntime(
        rank=_env_int("RANK", 0),
        local_rank=_env_int("LOCAL_RANK", 0),
        world_size=max(1, _env_int("WORLD_SIZE", 1)),
    )


def initialize_distributed_if_needed(runtime: ExportParallelRuntime) -> bool:
    if not runtime.distributed:
        return False
    if not dist.is_available():
        raise RuntimeError("Distributed export requested but torch.distributed is unavailable")
    if dist.is_initialized():
        return False
    dist.init_process_group(backend="gloo")
    return True


def finalize_distributed_if_needed(initialized: bool) -> None:
    if initialized and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def distributed_barrier(runtime: ExportParallelRuntime) -> None:
    if runtime.distributed:
        torch_distributed_barrier(runtime.local_rank)


def resolve_device(device: str, *, runtime: ExportParallelRuntime | None = None) -> torch.device:
    runtime = ExportParallelRuntime() if runtime is None else runtime
    if device == "auto":
        if torch.cuda.is_available():
            cuda_index = int(runtime.local_rank) if runtime.distributed else 0
            torch.cuda.set_device(cuda_index)
            return torch.device("cuda", cuda_index)
        return torch.device("cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device {device!r}, but CUDA is not available")
    if resolved.type == "cuda":
        if resolved.index is None and runtime.distributed:
            resolved = torch.device("cuda", int(runtime.local_rank))
        torch.cuda.set_device(resolved)
    return resolved


def resolve_export_num_workers(args: argparse.Namespace) -> int:
    if args.num_workers is not None:
        num_workers = int(args.num_workers)
        if num_workers < 0:
            raise ValueError(f"--num-workers must be >= 0, got {num_workers}")
        return num_workers

    workers_per_gpu = int(getattr(args, "workers_per_gpu", DEFAULT_WORKERS_PER_GPU))
    if workers_per_gpu < 0:
        raise ValueError(f"--workers-per-gpu must be >= 0, got {workers_per_gpu}")
    return workers_per_gpu


def resolve_export_batch_size(args: argparse.Namespace) -> int:
    batch_size = DEFAULT_EXPORT_BATCH_SIZE if args.batch_size is None else int(args.batch_size)
    if batch_size <= 0:
        raise ValueError(f"--batch-size must be positive, got {batch_size}")
    return batch_size


def configure_torch_threads(args: argparse.Namespace) -> int:
    thread_count = int(getattr(args, "torch_threads", DEFAULT_TORCH_THREADS))
    if thread_count <= 0:
        raise ValueError(f"--torch-threads must be > 0, got {thread_count}")
    torch.set_num_threads(thread_count)
    try:
        torch.set_num_interop_threads(thread_count)
    except RuntimeError:
        pass
    return thread_count


def shard_rows_for_runtime(
    rows: list[dict[str, Any]],
    *,
    runtime: ExportParallelRuntime,
) -> list[dict[str, Any]]:
    world_size = max(1, int(runtime.world_size))
    rank = int(runtime.rank)
    if rank < 0 or rank >= world_size:
        raise ValueError(f"Invalid distributed rank={rank} for world_size={world_size}")
    indexed_rows = [
        {
            **row,
            "export_row_index": row_index,
            "export_world_size": world_size,
        }
        for row_index, row in enumerate(rows)
    ]
    return indexed_rows[rank::world_size]


class ExportProcessedSceneDataset(ProcessedSceneDataset):
    def __getitem__(self, index: int):
        sample = super().__getitem__(index)
        row = self.rows[index]
        sample.meta["export_row_index"] = int(row.get("export_row_index", index))
        sample.meta["export_world_size"] = int(row.get("export_world_size", 1))
        return sample


def state_dict_from_checkpoint_payload(payload: Any, checkpoint_source: str) -> dict[str, torch.Tensor]:
    checkpoint_source = str(checkpoint_source)
    if checkpoint_source not in CHECKPOINT_SOURCES:
        raise ValueError(
            f"Unsupported checkpoint_source={checkpoint_source!r}; expected one of {sorted(CHECKPOINT_SOURCES)}"
        )
    if checkpoint_source == "training_checkpoint":
        if not isinstance(payload, dict) or "model_state_dict" not in payload:
            raise KeyError("training_checkpoint requires a checkpoint payload with 'model_state_dict'")
        state_dict = payload["model_state_dict"]
    elif checkpoint_source == "state_dict":
        state_dict = payload
    else:
        state_dict = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload

    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected checkpoint state dict mapping, got {type(state_dict)!r}")
    return state_dict


def export_pipeline_spec(cfg: dict[str, Any], *, years: list[int], num_workers: int | None) -> PipelineSpec:
    if not years:
        raise ValueError("At least one export year is required")
    pipeline_cfg = deepcopy(cfg["pipeline"])
    pipeline_cfg["year_range"] = [int(min(years)), int(max(years))]
    pipeline_cfg["center_ratio"] = 1.0
    pipeline_cfg["cache_scenes"] = False
    pipeline_cfg["preload_scenes"] = False
    pipeline_cfg["pin_memory"] = False
    pipeline_cfg["prefetch_factor"] = 1
    pipeline_cfg["persistent_workers"] = False
    if num_workers is not None:
        pipeline_cfg["num_workers"] = int(num_workers)
    pipeline_spec = PipelineSpec.from_mapping(pipeline_cfg)
    if pipeline_spec.kind not in (EUMP_RECONSTRUCTION_KIND, EUMP_CONTRASTIVE_KIND):
        raise ValueError(
            "Embedding export expects a reconstruction or contrastive pipeline; "
            f"got pipeline.kind={pipeline_spec.kind!r}"
        )
    return pipeline_spec


def build_export_dataloader(
    *,
    data_config: dict[str, Any],
    pipeline_spec: PipelineSpec,
    random_seed: int,
    batch_size: int,
    scene_ids: list[str] | None,
    years: set[int] | None = None,
    runtime: ExportParallelRuntime | None = None,
) -> tuple[DataLoader, Any, int]:
    runtime = ExportParallelRuntime() if runtime is None else runtime
    reader = ProcessedSceneStoreReader(Path(data_config["eump_root"]))
    task_config = pipeline_spec.to_task_config()
    reader.validate_task_config(task_config)

    preprocessing_path = data_config.get("preprocessing_path", data_config["eump_root"])
    preprocessing_spec = load_preprocessing_spec(preprocessing_path)
    validate_preprocessing_spec(preprocessing_spec)
    # Embedding export often selects a downstream-only year subset while reusing
    # preprocessing fitted on the full pretraining year range.
    validate_preprocessing_stats_scope(
        preprocessing_spec,
        split_manifest_path=data_config.get("split_manifest_path"),
        expected_year_range=None,
    )
    preprocessing_spec = build_active_preprocessing_spec(pipeline_spec, reader.source_specs, preprocessing_spec)

    rows = reader.iter_scene_years(scene_ids=scene_ids, year_range=pipeline_spec.year_range)
    if years is not None:
        rows = [row for row in rows if int(row["year"]) in years]
    compatible_rows = [
        row
        for row in rows
        if not reader.missing_sources_for_task(row["scene_id"], int(row["year"]), task_config)
    ]
    compatible_rows = sorted(
        compatible_rows,
        key=lambda row: (_natural_scene_sort_key(row["scene_id"]), int(row["year"])),
    )
    if not compatible_rows:
        year_range = pipeline_spec.year_range
        year_text = sorted(years) if years is not None else year_range
        raise ValueError(f"No compatible EUMP scene-year rows found for years={year_text}")

    total_rows = len(compatible_rows)
    compatible_rows = shard_rows_for_runtime(compatible_rows, runtime=runtime)
    batch_schema = build_eump_batch_schema(pipeline_spec, reader.source_specs, preprocessing_spec)
    dataset = ExportProcessedSceneDataset(
        compatible_rows,
        reader=reader,
        pipeline_spec=pipeline_spec,
        preprocessing_spec=preprocessing_spec,
        batch_schema=batch_schema,
        random_seed=random_seed,
        cache_scenes=pipeline_spec.cache_scenes,
        preload_scenes=pipeline_spec.preload_scenes,
    )
    num_workers = int(pipeline_spec.num_workers)
    kwargs: dict[str, Any] = {
        "batch_size": int(batch_size),
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": bool(pipeline_spec.pin_memory),
        "collate_fn": collate_processed_samples,
        "worker_init_fn": seed_worker,
        "generator": seeded_torch_generator(
            random_seed,
            offset=runtime.local_rank * 1_000_000,
        ),
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = int(pipeline_spec.prefetch_factor)
        kwargs["persistent_workers"] = bool(pipeline_spec.persistent_workers)
    return DataLoader(dataset=dataset, **kwargs), batch_schema, total_rows


def _natural_scene_sort_key(value: Any) -> tuple[int, int | str]:
    text = str(value)
    try:
        return 0, int(text)
    except ValueError:
        return 1, text


def build_reconstruction_model(
    *,
    cfg: dict[str, Any],
    batch_schema,
    checkpoint_path: Path,
    checkpoint_source: str,
    device: torch.device,
) -> EumpSwinReconstructionModel:
    model_kind = cfg.get("model", {}).get("kind", "swin_reconstruction")
    pretrain_model = cfg["experiment"]["pretrain_model"]

    if model_kind == "convnext_unet_reconstruction":
        model_cfg = cfg.get("model", {})
        model = EumpConvNextUnetReconstructionModel(
            input_channels=batch_schema.raster_input_channels,
            batch_schema=batch_schema,
            depths=tuple(model_cfg.get("depths", [3, 3, 27, 3])),
            dims=tuple(model_cfg.get("dims", [192, 384, 768, 1536])),
            drop_path_rate=float(model_cfg.get("drop_path_rate", 0.0)),
        )
    else:
        model = EumpSwinReconstructionModel(
            model_path=str(Path(MODEL_DIR) / pretrain_model),
            input_channels=batch_schema.raster_input_channels,
            batch_schema=batch_schema,
            load_pretrained=False,
            scalar_head_mode=cfg.get("reconstruct", {}).get("scalar_head_mode", "per_target"),
        )

    payload = torch.load(checkpoint_path, map_location="cpu")
    state_dict = state_dict_from_checkpoint_payload(payload, checkpoint_source)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        encoder_missing = [k for k in missing if not k.startswith("raster_") and not k.startswith("scalar_")]
        if encoder_missing:
            raise RuntimeError(
                f"Encoder keys missing from checkpoint: {encoder_missing[:10]}"
            )
    return model.to(device).eval()


def distributed_shard_path(output_path: Path, *, rank: int, world_size: int) -> Path:
    return output_path.with_name(
        f"{output_path.stem}.rank{int(rank):05d}-of-{int(world_size):05d}.shard"
    )


def distributed_shard_paths(output_path: Path, *, world_size: int) -> list[Path]:
    return [
        distributed_shard_path(output_path, rank=rank, world_size=world_size)
        for rank in range(int(world_size))
    ]


def merged_stage_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}.merge.shard")


def _array_path(stage_path: Path, name: str) -> Path:
    return stage_path / f"{name}.npy"


def _manifest_path(stage_path: Path) -> Path:
    return stage_path / "manifest.json"


def _stage_array_paths(stage_path: Path) -> dict[str, Path]:
    names = (
        "embeddings",
        "scene_ids",
        "sample_ids",
        "years",
        "crop_boxes",
        "row_indices",
        "raster_input_channel_names",
        "scalar_input_names",
        "metadata_json",
    )
    return {name: _array_path(stage_path, name) for name in names}


def prepare_stage_path(stage_path: Path, *, overwrite: bool) -> None:
    if stage_path.exists():
        if not overwrite:
            raise FileExistsError(f"Export temporary shard already exists, pass --overwrite to replace it: {stage_path}")
        if stage_path.is_dir():
            shutil.rmtree(stage_path)
        else:
            stage_path.unlink()
    stage_path.mkdir(parents=True, exist_ok=False)


def _save_small_stage_array(stage_path: Path, name: str, value: np.ndarray) -> None:
    np.save(_array_path(stage_path, name), value)


def write_stage_manifest(
    stage_path: Path,
    *,
    sample_count: int,
    embedding_shape: tuple[int, ...] | None,
) -> None:
    _manifest_path(stage_path).write_text(
        json.dumps(
            {
                "sample_count": int(sample_count),
                "embedding_shape": None if embedding_shape is None else [int(value) for value in embedding_shape],
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )


def read_stage_manifest(stage_path: Path) -> dict[str, Any]:
    return json.loads(_manifest_path(stage_path).read_text(encoding="utf-8"))


def _zip_npy_arrays(
    *,
    stage_path: Path,
    output_path: Path,
    compress: bool,
) -> None:
    compression = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, mode="w", compression=compression, allowZip64=True) as archive:
        for name, path in _stage_array_paths(stage_path).items():
            if path.exists():
                archive.write(path, arcname=f"{name}.npy")


def package_stage_as_npz(
    *,
    stage_path: Path,
    output_path: Path,
    overwrite: bool,
    compress: bool,
) -> None:
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists, pass --overwrite to replace it: {output_path}")
        output_path.unlink()
    _zip_npy_arrays(stage_path=stage_path, output_path=output_path, compress=compress)


def _metadata_from_stage(stage_path: Path) -> dict[str, Any]:
    metadata_array = np.load(_array_path(stage_path, "metadata_json"), allow_pickle=False)
    if getattr(metadata_array, "shape", ()) == ():
        metadata_text = str(metadata_array.item())
    else:
        metadata_text = str(metadata_array.tolist())
    return json.loads(metadata_text)


class StreamingEmbeddingShardWriter:
    def __init__(
        self,
        *,
        stage_path: Path,
        sample_count: int,
        batch_schema,
        metadata: dict[str, Any],
        overwrite: bool,
    ) -> None:
        self.stage_path = stage_path
        self.sample_count = int(sample_count)
        self.batch_schema = batch_schema
        self.metadata = dict(metadata)
        self.offset = 0
        self.embedding_shape: tuple[int, ...] | None = None
        self.embedding_memmap: np.memmap | None = None
        self.scene_ids: list[str] = []
        self.sample_ids: list[str] = []
        self.years: list[int] = []
        self.crop_boxes: list[tuple[int, int, int, int]] = []
        self.row_indices: list[int] = []
        prepare_stage_path(stage_path, overwrite=overwrite)

    def _ensure_embedding_memmap(self, embedding_batch: np.ndarray) -> None:
        if self.embedding_memmap is not None:
            return
        self.embedding_shape = (self.sample_count, *tuple(int(value) for value in embedding_batch.shape[1:]))
        self.embedding_memmap = np.lib.format.open_memmap(
            _array_path(self.stage_path, "embeddings"),
            mode="w+",
            dtype=np.float32,
            shape=self.embedding_shape,
        )

    def write_batch(self, embedding_batch: np.ndarray, metas: list[dict[str, Any]], default_year: int) -> None:
        embedding_batch = embedding_batch.astype(np.float32, copy=False)
        self._ensure_embedding_memmap(embedding_batch)
        if self.embedding_memmap is None:
            raise RuntimeError("Embedding memmap was not initialized")
        batch_size = int(embedding_batch.shape[0])
        end = self.offset + batch_size
        if end > self.sample_count:
            raise ValueError(f"Shard writer received {end} rows but expected {self.sample_count}")
        self.embedding_memmap[self.offset:end] = embedding_batch
        self.offset = end

        for meta in metas:
            self.scene_ids.append(str(meta.get("scene_id", "")))
            self.sample_ids.append("" if meta.get("sample_id") is None else str(meta.get("sample_id")))
            self.years.append(int(meta.get("year", default_year)))
            self.row_indices.append(int(meta.get("export_row_index", len(self.row_indices))))
            crop_box = meta.get("crop_box")
            if crop_box is None:
                self.crop_boxes.append((-1, -1, -1, -1))
            else:
                self.crop_boxes.append(tuple(int(value) for value in crop_box))

    def finalize(self) -> None:
        if self.offset != self.sample_count:
            raise ValueError(f"Shard writer wrote {self.offset} rows but expected {self.sample_count}")
        if self.embedding_memmap is not None:
            self.embedding_memmap.flush()
            del self.embedding_memmap
            self.embedding_memmap = None

        _save_small_stage_array(self.stage_path, "scene_ids", np.asarray(self.scene_ids, dtype=str))
        _save_small_stage_array(self.stage_path, "sample_ids", np.asarray(self.sample_ids, dtype=str))
        _save_small_stage_array(self.stage_path, "years", np.asarray(self.years, dtype=np.int32))
        _save_small_stage_array(self.stage_path, "crop_boxes", np.asarray(self.crop_boxes, dtype=np.int32))
        _save_small_stage_array(self.stage_path, "row_indices", np.asarray(self.row_indices, dtype=np.int64))
        _save_small_stage_array(
            self.stage_path,
            "raster_input_channel_names",
            np.asarray(self.batch_schema.raster_input_channel_names, dtype=str),
        )
        _save_small_stage_array(
            self.stage_path,
            "scalar_input_names",
            np.asarray(self.batch_schema.scalar_input_names, dtype=str),
        )
        _save_small_stage_array(
            self.stage_path,
            "metadata_json",
            np.asarray(json.dumps(self.metadata, ensure_ascii=True)),
        )
        write_stage_manifest(
            self.stage_path,
            sample_count=self.sample_count,
            embedding_shape=self.embedding_shape,
        )


def merge_embedding_shards(
    *,
    shard_paths: list[Path],
    output_path: Path,
    overwrite: bool,
    keep_shards: bool,
    compress: bool,
) -> dict[str, Any]:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists, pass --overwrite to replace it: {output_path}")

    scene_id_parts: list[np.ndarray] = []
    sample_id_parts: list[np.ndarray] = []
    year_parts: list[np.ndarray] = []
    crop_box_parts: list[np.ndarray] = []
    row_index_parts: list[np.ndarray] = []
    source_shard_parts: list[np.ndarray] = []
    local_index_parts: list[np.ndarray] = []
    metadata: dict[str, Any] | None = None
    embedding_shape: tuple[int, ...] | None = None
    total_samples = 0

    for shard_idx, shard_path in enumerate(shard_paths):
        if not shard_path.exists():
            raise FileNotFoundError(f"Expected distributed embedding shard is missing: {shard_path}")
        manifest = read_stage_manifest(shard_path)
        sample_count = int(manifest["sample_count"])
        if metadata is None:
            metadata = _metadata_from_stage(shard_path)
        if sample_count <= 0:
            continue
        shard_embedding_shape = tuple(int(value) for value in manifest["embedding_shape"])
        if embedding_shape is None:
            embedding_shape = shard_embedding_shape
        elif embedding_shape[1:] != shard_embedding_shape[1:]:
            raise ValueError(
                f"Shard embedding shape mismatch: expected {embedding_shape}, got {shard_embedding_shape}"
            )

        shard_row_indices = np.load(_array_path(shard_path, "row_indices"), allow_pickle=False)
        row_index_parts.append(np.asarray(shard_row_indices, dtype=np.int64))
        scene_id_parts.append(np.load(_array_path(shard_path, "scene_ids"), allow_pickle=False))
        sample_id_parts.append(np.load(_array_path(shard_path, "sample_ids"), allow_pickle=False))
        year_parts.append(np.load(_array_path(shard_path, "years"), allow_pickle=False))
        crop_box_parts.append(np.load(_array_path(shard_path, "crop_boxes"), allow_pickle=False))
        source_shard_parts.append(np.full((sample_count,), shard_idx, dtype=np.int64))
        local_index_parts.append(np.arange(sample_count, dtype=np.int64))
        total_samples += sample_count

    if metadata is None or embedding_shape is None:
        raise ValueError("No shard metadata was available for distributed embedding merge")
    if not row_index_parts:
        raise ValueError("Distributed embedding shards contain no samples")

    row_indices = np.concatenate(row_index_parts, axis=0)
    order = np.argsort(row_indices, kind="stable")
    scene_ids = np.concatenate(scene_id_parts, axis=0)[order]
    sample_ids = np.concatenate(sample_id_parts, axis=0)[order]
    years = np.concatenate(year_parts, axis=0)[order]
    crop_boxes = np.concatenate(crop_box_parts, axis=0)[order]
    source_shards = np.concatenate(source_shard_parts, axis=0)[order]
    local_indices = np.concatenate(local_index_parts, axis=0)[order]
    row_indices = row_indices[order]

    metadata = dict(metadata)
    metadata["distributed"] = True
    metadata["world_size"] = len(shard_paths)
    metadata["shard_paths"] = [str(path) for path in shard_paths]
    metadata["compressed"] = bool(compress)

    stage_path = merged_stage_path(output_path)
    prepare_stage_path(stage_path, overwrite=True)
    final_embedding_shape = (int(total_samples), *embedding_shape[1:])
    final_embeddings = np.lib.format.open_memmap(
        _array_path(stage_path, "embeddings"),
        mode="w+",
        dtype=np.float32,
        shape=final_embedding_shape,
    )
    per_sample_elements = max(1, int(np.prod(final_embedding_shape[1:], dtype=np.int64)))
    merge_chunk_size = max(1, int(MAX_MERGE_COPY_ELEMENTS // per_sample_elements))
    for shard_idx, shard_path in enumerate(shard_paths):
        final_positions = np.flatnonzero(source_shards == shard_idx)
        if final_positions.size == 0:
            continue
        shard_embeddings = np.load(_array_path(shard_path, "embeddings"), mmap_mode="r", allow_pickle=False)
        shard_local_indices = local_indices[final_positions]
        for start in range(0, int(final_positions.size), merge_chunk_size):
            end = min(start + merge_chunk_size, int(final_positions.size))
            positions = final_positions[start:end]
            selected = shard_local_indices[start:end]
            final_embeddings[positions] = shard_embeddings[selected]
        del shard_embeddings
    final_embeddings.flush()
    del final_embeddings

    first_nonempty_shard = shard_paths[int(source_shards[0])]
    _save_small_stage_array(stage_path, "scene_ids", scene_ids)
    _save_small_stage_array(stage_path, "sample_ids", sample_ids)
    _save_small_stage_array(stage_path, "years", years.astype(np.int32, copy=False))
    _save_small_stage_array(stage_path, "crop_boxes", crop_boxes.astype(np.int32, copy=False))
    _save_small_stage_array(stage_path, "row_indices", row_indices.astype(np.int64, copy=False))
    _save_small_stage_array(
        stage_path,
        "raster_input_channel_names",
        np.load(_array_path(first_nonempty_shard, "raster_input_channel_names"), allow_pickle=False),
    )
    _save_small_stage_array(
        stage_path,
        "scalar_input_names",
        np.load(_array_path(first_nonempty_shard, "scalar_input_names"), allow_pickle=False),
    )
    _save_small_stage_array(stage_path, "metadata_json", np.asarray(json.dumps(metadata, ensure_ascii=True)))
    write_stage_manifest(
        stage_path,
        sample_count=int(total_samples),
        embedding_shape=final_embedding_shape,
    )
    package_stage_as_npz(
        stage_path=stage_path,
        output_path=output_path,
        overwrite=True,
        compress=compress,
    )

    if not keep_shards:
        for shard_path in shard_paths:
            if shard_path.exists():
                shutil.rmtree(shard_path)
        shutil.rmtree(stage_path, ignore_errors=True)

    return {
        **metadata,
        "output_path": str(output_path),
        "samples": int(total_samples),
        "embedding_shape": list(final_embedding_shape),
    }


def pool_embedding(embedding: torch.Tensor, *, keep_spatial: bool) -> torch.Tensor:
    if keep_spatial:
        return embedding
    return F.adaptive_avg_pool2d(embedding, (1, 1)).flatten(1)


def export_embeddings(
    *,
    cfg: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    torch_threads = configure_torch_threads(args)
    runtime = resolve_parallel_runtime()
    initialized_distributed = initialize_distributed_if_needed(runtime)
    export_years = resolve_export_years(args)
    model_name = resolve_export_model_name(cfg, args)
    checkpoint_path = resolve_checkpoint_path(
        cfg=cfg,
        model_name=model_name,
        checkpoint_file=args.checkpoint_file,
    )
    try:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Reconstruction checkpoint not found: {checkpoint_path}")

        pooled = not bool(args.keep_spatial)
        output_path = (
            default_output_path(checkpoint_path=checkpoint_path, years=export_years, pooled=pooled)
            if args.output_path is None
            else args.output_path.expanduser().resolve()
        )
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"Output already exists, pass --overwrite to replace it: {output_path}")

        shard_paths = distributed_shard_paths(output_path, world_size=runtime.world_size)
        active_stage_paths = shard_paths if runtime.distributed else [distributed_shard_path(output_path, rank=0, world_size=1)]
        if not args.overwrite:
            existing_shards = [str(path) for path in active_stage_paths if path.exists()]
            existing_merge = merged_stage_path(output_path)
            if existing_merge.exists():
                existing_shards.append(str(existing_merge))
            if existing_shards:
                raise FileExistsError(
                    "Export temporary shard already exists, pass --overwrite to replace stale shards: "
                    f"{existing_shards}"
        )

        num_workers = resolve_export_num_workers(args)
        pipeline_spec = export_pipeline_spec(cfg, years=export_years, num_workers=num_workers)
        batch_size = resolve_export_batch_size(args)

        dataloader, batch_schema, total_rows = build_export_dataloader(
            data_config=cfg["data"],
            pipeline_spec=pipeline_spec,
            random_seed=int(cfg["experiment"]["random_seed"]),
            batch_size=batch_size,
            scene_ids=None if args.scene_ids is None else [str(scene_id) for scene_id in args.scene_ids],
            years=set(export_years),
            runtime=runtime,
        )
        device = resolve_device(args.device, runtime=runtime)
        model = build_reconstruction_model(
            cfg=cfg,
            batch_schema=batch_schema,
            checkpoint_path=checkpoint_path,
            checkpoint_source=args.checkpoint_source,
            device=device,
        )

        metadata = {
            "year": int(export_years[0]) if len(export_years) == 1 else None,
            "years": list(export_years),
            "year_range": [int(min(export_years)), int(max(export_years))],
            "pooled": pooled,
            "model_name": model_name,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_source": args.checkpoint_source,
            "pipeline_kind": pipeline_spec.kind,
            "crop_size": pipeline_spec.crop_size,
            "center_ratio": pipeline_spec.center_ratio,
            "input_channel_names": list(batch_schema.raster_input_channel_names),
            "scalar_input_names": list(batch_schema.scalar_input_names),
            "num_workers": int(num_workers),
            "batch_size": int(batch_size),
            "torch_threads": int(torch_threads),
            "distributed": runtime.distributed,
            "rank": int(runtime.rank),
            "local_rank": int(runtime.local_rank),
            "world_size": int(runtime.world_size),
            "total_rows": int(total_rows),
            "compressed": bool(args.compress_output),
            "streaming": True,
        }
        write_path = (
            distributed_shard_path(output_path, rank=runtime.rank, world_size=runtime.world_size)
            if runtime.distributed
            else distributed_shard_path(output_path, rank=0, world_size=1)
        )
        writer = StreamingEmbeddingShardWriter(
            stage_path=write_path,
            sample_count=len(dataloader.dataset),
            batch_schema=batch_schema,
            metadata=metadata,
            overwrite=True,
        )

        progress_disabled = runtime.distributed and runtime.rank != 0
        with torch.inference_mode():
            for batch in tqdm(dataloader, desc=f"export embeddings rank {runtime.rank}", unit="batch", disable=progress_disabled):
                raster_inputs = batch.raster_inputs.to(device, non_blocking=True)
                embedding = model.backbone.encode(raster_inputs, bool_masked_pos=None)
                embedding = pool_embedding(embedding, keep_spatial=args.keep_spatial)
                writer.write_batch(
                    embedding.detach().cpu().numpy(),
                    batch.meta,
                    default_year=export_years[0],
                )
        writer.finalize()

        summary = {
            **metadata,
            "output_path": str(write_path),
            "final_output_path": str(output_path),
            "samples": int(writer.sample_count),
            "embedding_shape": [] if writer.embedding_shape is None else list(writer.embedding_shape),
        }

        if not runtime.distributed:
            package_stage_as_npz(
                stage_path=write_path,
                output_path=output_path,
                overwrite=True,
                compress=bool(args.compress_output),
            )
            summary["output_path"] = str(output_path)
            summary["final_output_path"] = str(output_path)
            if not args.keep_shards:
                shutil.rmtree(write_path, ignore_errors=True)
            return summary

        distributed_barrier(runtime)
        if runtime.rank == 0:
            summary = merge_embedding_shards(
                shard_paths=shard_paths,
                output_path=output_path,
                overwrite=True,
                keep_shards=bool(args.keep_shards),
                compress=bool(args.compress_output),
            )
            summary["rank"] = 0
        distributed_barrier(runtime)
        return summary
    finally:
        finalize_distributed_if_needed(initialized_distributed)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = compose_runtime_config(args.config_name, args.overrides)
    summary = export_embeddings(cfg=cfg, args=args)
    if int(summary.get("rank", 0)) == 0:
        print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
