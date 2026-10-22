from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
from pathlib import Path
from typing import Any
import zlib

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from urban_embedding.core.reproducibility import seed_worker, seeded_torch_generator
from urban_embedding.data.crop import materialize_crop_sample, sample_random_crop_box, stack_model_raster_inputs
from .batch import ProcessedModelBatch, TensorBlock
from .split_manifest import load_split_manifest, split_scene_ids_from_manifest
from .pipeline_spec import (
    EUMP_CITY_SCALAR_DOWNSTREAM_KIND,
    PipelineSpec,
    ProcessedBatchSchema,
    TargetSchema,
    build_active_preprocessing_spec,
    build_eump_batch_schema,
    normalize_eump_pipeline_kind,
)
from .processing import (
    load_preprocessing_spec,
    process_scene_record,
    validate_preprocessing_spec,
    validate_preprocessing_stats_scope,
)
from .reader import ProcessedSceneStoreReader


@dataclass(slots=True)
class ProcessedSample:
    raster_inputs: np.ndarray
    raster_valid_mask: np.ndarray
    scalar_inputs: np.ndarray
    scalar_valid_mask: np.ndarray
    city_mask: np.ndarray
    raster_targets: dict[str, tuple[np.ndarray, np.ndarray, tuple[str, ...]]]
    scalar_targets: dict[str, tuple[np.ndarray, np.ndarray, tuple[str, ...]]]
    meta: dict[str, Any]


def _empty_vector(length: int, dtype=np.float32) -> np.ndarray:
    return np.zeros((int(length),), dtype=dtype)


def _missing_raster_target(target: TargetSchema, height: int, width: int) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    shape = (target.channels, int(height), int(width))
    return (
        np.full(shape, np.nan, dtype=np.float32),
        np.zeros(shape, dtype=bool),
        target.channel_names,
    )


def _missing_scalar_target(target: TargetSchema) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    shape = (target.channels,)
    return (
        np.full(shape, np.nan, dtype=np.float32),
        np.zeros(shape, dtype=bool),
        target.channel_names,
    )


def _find_raster_target(crop_sample, target: TargetSchema):
    source = target.spec.source
    if target.spec.target_type == "input_reconstruction":
        raise ValueError("input_reconstruction is materialized from model inputs, not target blocks")
    if source is None:
        raise ValueError(f"Raster target {target.name!r} requires a source")
    if crop_sample.targets.reconstruct_raster is not None:
        names = crop_sample.targets.reconstruct_raster.channel_names or ()
        if source in names or names == (source,):
            return crop_sample.targets.reconstruct_raster
    if source in crop_sample.targets.aux_raster:
        return crop_sample.targets.aux_raster[source]
    if source in crop_sample.targets.prediction_raster:
        return crop_sample.targets.prediction_raster[source]
    raise KeyError(f"Raster target source {source!r} is unavailable in crop sample")


def _find_scalar_target(crop_sample, target: TargetSchema):
    if target.spec.target_type == "crop_scalar":
        if target.name in crop_sample.targets.aux_scalar:
            return crop_sample.targets.aux_scalar[target.name]
        raise KeyError(f"Crop scalar target {target.name!r} is unavailable in crop sample")

    source = target.spec.source
    if source is None:
        raise ValueError(f"Scalar target {target.name!r} requires a source")
    if source in crop_sample.targets.aux_scalar:
        return crop_sample.targets.aux_scalar[source]
    if source in crop_sample.targets.prediction_scalar:
        return crop_sample.targets.prediction_scalar[source]
    raise KeyError(f"Scalar target source {source!r} is unavailable in crop sample")


def _split_rows_by_scene(
    rows: list[dict[str, Any]],
    *,
    random_seed: int,
    test_size: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not rows:
        raise ValueError("No compatible processed scene-year rows are available")

    scene_ids = sorted({str(row["scene_id"]) for row in rows})
    if len(scene_ids) == 1:
        return rows, rows, rows

    rng = np.random.default_rng(random_seed)
    shuffled = list(scene_ids)
    rng.shuffle(shuffled)

    if len(shuffled) == 2:
        train_ids = {shuffled[0]}
        val_ids = {shuffled[1]}
        test_ids = {shuffled[1]}
    else:
        tmp_count = max(2, int(round(len(shuffled) * test_size)))
        tmp_count = min(tmp_count, len(shuffled) - 1)
        train_ids = set(shuffled[:-tmp_count])
        tmp_ids = shuffled[-tmp_count:]
        val_cut = max(1, len(tmp_ids) // 2)
        val_ids = set(tmp_ids[:val_cut])
        test_ids = set(tmp_ids[val_cut:] or tmp_ids[:val_cut])

    def select(allowed: set[str]) -> list[dict[str, Any]]:
        selected = [row for row in rows if str(row["scene_id"]) in allowed]
        if not selected:
            raise ValueError(f"Split with scene ids {sorted(allowed)} produced no rows")
        return selected

    return select(train_ids), select(val_ids), select(test_ids)


def _filter_rows_for_task(
    rows: list[dict[str, Any]],
    *,
    reader: ProcessedSceneStoreReader,
    task_config,
) -> list[dict[str, Any]]:
    return [
        row for row in rows
        if not reader.missing_sources_for_task(row["scene_id"], int(row["year"]), task_config)
    ]


def _select_rows_for_scene_ids(rows: list[dict[str, Any]], allowed_scene_ids: set[str]) -> list[dict[str, Any]]:
    return [row for row in rows if str(row["scene_id"]) in allowed_scene_ids]


class ProcessedSceneDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        reader: ProcessedSceneStoreReader,
        pipeline_spec: PipelineSpec,
        preprocessing_spec: dict[str, Any],
        batch_schema: ProcessedBatchSchema,
        random_seed: int = 42,
        cache_scenes: bool = False,
        preload_scenes: bool = False,
    ) -> None:
        self.rows = list(rows)
        self.reader = reader
        self.pipeline_spec = pipeline_spec
        self.task_config = pipeline_spec.to_task_config()
        self.preprocessing_spec = preprocessing_spec
        self.batch_schema = batch_schema
        self.random_seed = int(random_seed)
        self.cache_scenes = bool(cache_scenes)
        self._scene_cache: dict[tuple[str, int], Any] | None = {} if cache_scenes else None
        self._current_epoch = mp.Value("q", 0)

        self._channel_drop = pipeline_spec.input_channel_drop
        self._droppable_channel_indices: tuple[int, ...] = ()
        if self._channel_drop.enabled:
            self._droppable_channel_indices = self._channel_drop.resolve_droppable_indices(
                batch_schema.raster_input_channel_names
            )

        if preload_scenes:
            if self._scene_cache is None:
                self._scene_cache = {}
            for row in self.rows:
                key = (str(row["scene_id"]), int(row["year"]))
                self._scene_cache[key] = self._load_processed_scene(row)

    def __len__(self) -> int:
        return len(self.rows)

    def set_epoch(self, epoch: int) -> None:
        self._current_epoch.value = int(epoch)

    def _crop_seed(self, row: dict[str, Any], index: int) -> int:
        scene_key = f"{row['scene_id']}:{int(row['year'])}"
        scene_hash = zlib.crc32(scene_key.encode("utf-8")) & 0xFFFFFFFF
        seed_sequence = np.random.SeedSequence(
            [
                int(self.random_seed) & 0xFFFFFFFF,
                int(self._current_epoch.value) & 0xFFFFFFFF,
                int(index) & 0xFFFFFFFF,
                int(scene_hash),
            ]
        )
        return int(seed_sequence.generate_state(1, dtype=np.uint32)[0])

    def _apply_channel_drop(
        self,
        raster_inputs: np.ndarray,
        raster_valid_mask: np.ndarray,
        row: dict[str, Any],
        crop_box: tuple[int, int, int, int],
    ) -> tuple[str, ...]:
        """Zero-fill k random encoder input channels for this sample.

        Returns the dropped channel names. Deterministic per
        ``(seed, scene_id, year, crop_box)`` and a no-op when disabled.
        """
        drop = self._channel_drop
        if not drop.enabled or not self._droppable_channel_indices:
            return ()
        crop_key = (int(crop_box.top), int(crop_box.left), int(crop_box.height), int(crop_box.width))
        scene_key = f"{drop.seed}:{row['scene_id']}:{int(row['year'])}:{crop_key}"
        scene_hash = zlib.crc32(scene_key.encode("utf-8")) & 0xFFFFFFFF
        drop_rng = np.random.default_rng(np.random.SeedSequence([int(drop.seed) & 0xFFFFFFFF, int(scene_hash)]))
        max_drop = min(int(drop.max_drop), len(self._droppable_channel_indices))
        k = int(drop_rng.integers(int(drop.min_drop), max_drop + 1))
        chosen = drop_rng.choice(np.asarray(self._droppable_channel_indices, dtype=np.int64), size=k, replace=False)
        chosen = sorted(int(idx) for idx in chosen)
        raster_inputs[chosen, ...] = 0.0
        raster_valid_mask[chosen, ...] = False
        channel_names = self.batch_schema.raster_input_channel_names
        return tuple(channel_names[idx] for idx in chosen)

    def _load_processed_scene(self, row: dict[str, Any]):
        scene = self.reader.load_scene_record(row["scene_id"], int(row["year"]), self.task_config)
        processed = process_scene_record(scene, self.task_config, self.preprocessing_spec)
        processed.targets.crop_scalar_specs = self.pipeline_spec.crop_scalar_specs
        return processed

    def _get_processed_scene(self, row: dict[str, Any]):
        key = (str(row["scene_id"]), int(row["year"]))
        if self._scene_cache is not None and key in self._scene_cache:
            return self._scene_cache[key]
        scene = self._load_processed_scene(row)
        if self._scene_cache is not None:
            self._scene_cache[key] = scene
        return scene

    def __getitem__(self, index: int) -> ProcessedSample:
        row = self.rows[index]
        scene = self._get_processed_scene(row)
        seed = self._crop_seed(row, index)
        rng = np.random.default_rng(seed)
        crop_box = sample_random_crop_box(
            scene.spatial_shape,
            self.pipeline_spec.crop_size,
            center_ratio=self.pipeline_spec.center_ratio,
            rng=rng,
        )
        crop_sample = materialize_crop_sample(scene, crop_box, preprocessing_spec=self.preprocessing_spec)
        if self.pipeline_spec.include_city_mask_as_input and crop_sample.inputs.city_mask is None:
            raise ValueError(
                "pipeline.include_city_mask_as_input=true requires city_mask data, "
                "but the sampled scene has no city_mask"
            )
        raster_inputs, raster_valid_mask = stack_model_raster_inputs(
            crop_sample.inputs,
            include_city_mask=self.pipeline_spec.include_city_mask_as_input,
            broadcast_scalar=self.pipeline_spec.broadcast_scalar,
        )
        raster_inputs = raster_inputs.astype(np.float32, copy=False)
        raster_valid_mask = raster_valid_mask.astype(bool, copy=False)
        dropped_channels = self._apply_channel_drop(raster_inputs, raster_valid_mask, row, crop_box)

        if crop_sample.inputs.scalar is None:
            scalar_inputs = _empty_vector(self.batch_schema.scalar_input_dim)
            scalar_valid_mask = _empty_vector(self.batch_schema.scalar_input_dim, dtype=bool)
        else:
            scalar_inputs = crop_sample.inputs.scalar.data.astype(np.float32, copy=False)
            scalar_valid_mask = crop_sample.inputs.scalar.valid_mask.astype(bool, copy=False)

        city_mask = crop_sample.inputs.city_mask
        if city_mask is None:
            height, width = raster_inputs.shape[-2:]
            city_mask = np.ones((1, height, width), dtype=np.float32)
        else:
            city_mask = city_mask.astype(np.float32, copy=False)

        raster_targets: dict[str, tuple[np.ndarray, np.ndarray, tuple[str, ...]]] = {}
        height, width = raster_inputs.shape[-2:]
        for target in self.batch_schema.raster_targets:
            if target.spec.target_type == "input_reconstruction":
                raster_targets[target.name] = (
                    raster_inputs,
                    raster_valid_mask,
                    target.channel_names,
                )
                continue
            try:
                block = _find_raster_target(crop_sample, target)
            except KeyError:
                if target.spec.optional:
                    raster_targets[target.name] = _missing_raster_target(target, height, width)
                    continue
                raise
            raster_targets[target.name] = (
                block.data.astype(np.float32, copy=False),
                block.valid_mask.astype(bool, copy=False),
                block.channel_names or target.channel_names,
            )

        scalar_targets: dict[str, tuple[np.ndarray, np.ndarray, tuple[str, ...]]] = {}
        for target in self.batch_schema.scalar_targets:
            try:
                block = _find_scalar_target(crop_sample, target)
            except KeyError:
                if target.spec.optional:
                    scalar_targets[target.name] = _missing_scalar_target(target)
                    continue
                raise
            scalar_targets[target.name] = (
                block.data.astype(np.float32, copy=False),
                block.valid_mask.astype(bool, copy=False),
                block.names or target.channel_names,
            )

        meta = dict(crop_sample.meta)
        meta["raster_input_channel_names"] = self.batch_schema.raster_input_channel_names
        meta["scalar_input_names"] = self.batch_schema.scalar_input_names
        if self._channel_drop.enabled:
            meta["dropped_channels"] = dropped_channels
        return ProcessedSample(
            raster_inputs=raster_inputs,
            raster_valid_mask=raster_valid_mask,
            scalar_inputs=scalar_inputs,
            scalar_valid_mask=scalar_valid_mask,
            city_mask=city_mask,
            raster_targets=raster_targets,
            scalar_targets=scalar_targets,
            meta=meta,
        )


def collate_processed_samples(samples: list[ProcessedSample]) -> ProcessedModelBatch:
    first = samples[0]

    def stack_array(name: str, arrays: list[np.ndarray], dtype=None) -> torch.Tensor:
        stacked = np.stack(arrays, axis=0)
        if dtype is not None:
            stacked = stacked.astype(dtype, copy=False)
        return torch.from_numpy(stacked)

    raster_inputs = stack_array("raster_inputs", [sample.raster_inputs for sample in samples], np.float32)
    raster_valid_mask = stack_array("raster_valid_mask", [sample.raster_valid_mask for sample in samples], bool)

    def target_aliases_raster_input(target_name: str) -> bool:
        for sample in samples:
            target_data, target_valid, _ = sample.raster_targets[target_name]
            if target_data.shape != sample.raster_inputs.shape or target_valid.shape != sample.raster_valid_mask.shape:
                return False
            if not np.shares_memory(target_data, sample.raster_inputs):
                return False
            if not np.shares_memory(target_valid, sample.raster_valid_mask):
                return False
        return True

    raster_targets: dict[str, TensorBlock] = {}
    for target_name in first.raster_targets:
        if target_aliases_raster_input(target_name):
            data = raster_inputs
            valid = raster_valid_mask
        else:
            data = stack_array(target_name, [sample.raster_targets[target_name][0] for sample in samples], np.float32)
            valid = stack_array(target_name, [sample.raster_targets[target_name][1] for sample in samples], bool)
        raster_targets[target_name] = TensorBlock(
            data=data,
            valid_mask=valid,
            names=first.raster_targets[target_name][2],
        )

    scalar_targets: dict[str, TensorBlock] = {}
    for target_name in first.scalar_targets:
        data = stack_array(target_name, [sample.scalar_targets[target_name][0] for sample in samples], np.float32)
        valid = stack_array(target_name, [sample.scalar_targets[target_name][1] for sample in samples], bool)
        scalar_targets[target_name] = TensorBlock(
            data=data,
            valid_mask=valid,
            names=first.scalar_targets[target_name][2],
        )

    return ProcessedModelBatch(
        raster_inputs=raster_inputs,
        raster_valid_mask=raster_valid_mask,
        scalar_inputs=stack_array("scalar_inputs", [sample.scalar_inputs for sample in samples], np.float32),
        scalar_valid_mask=stack_array("scalar_valid_mask", [sample.scalar_valid_mask for sample in samples], bool),
        city_mask=stack_array("city_mask", [sample.city_mask for sample in samples], np.float32),
        raster_targets=raster_targets,
        scalar_targets=scalar_targets,
        meta=[sample.meta for sample in samples],
    )


def build_eump_dataloaders(
    *,
    data_config: dict[str, Any],
    pipeline_spec: PipelineSpec,
    logger,
    random_seed: int,
    batch_size: int,
    local_rank: int,
    world_size: int,
    is_distributed: bool = True,
    shuffle_train: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader, ProcessedBatchSchema]:
    eump_root = Path(data_config["eump_root"])
    reader = ProcessedSceneStoreReader(eump_root)
    task_config = pipeline_spec.to_task_config()
    reader.validate_task_config(task_config)

    preprocessing_path = data_config.get("preprocessing_path", eump_root)
    preprocessing_spec = load_preprocessing_spec(preprocessing_path)
    validate_preprocessing_spec(preprocessing_spec)
    validate_preprocessing_stats_scope(
        preprocessing_spec,
        split_manifest_path=data_config.get("split_manifest_path"),
        expected_year_range=pipeline_spec.year_range,
    )
    preprocessing_spec = build_active_preprocessing_spec(pipeline_spec, reader.source_specs, preprocessing_spec)

    rows = reader.iter_scene_years(year_range=pipeline_spec.year_range)
    batch_schema = build_eump_batch_schema(pipeline_spec, reader.source_specs, preprocessing_spec)
    preprocessing_split_manifest_path = data_config.get("split_manifest_path")
    runtime_split_manifest_path = data_config.get("runtime_split_manifest_path") or preprocessing_split_manifest_path
    if runtime_split_manifest_path:
        split_manifest = load_split_manifest(
            runtime_split_manifest_path,
            available_scene_ids=reader.scene_index.keys(),
            expected_year_range=pipeline_spec.year_range,
        )
        train_scene_ids, val_scene_ids, test_scene_ids = split_scene_ids_from_manifest(split_manifest)
        split_rows = {
            "train": _select_rows_for_scene_ids(rows, train_scene_ids),
            "val": _select_rows_for_scene_ids(rows, val_scene_ids),
            "test": _select_rows_for_scene_ids(rows, test_scene_ids),
        }
        train_rows = _filter_rows_for_task(split_rows["train"], reader=reader, task_config=task_config)
        val_rows = _filter_rows_for_task(split_rows["val"], reader=reader, task_config=task_config)
        test_rows = _filter_rows_for_task(split_rows["test"], reader=reader, task_config=task_config)
        for split_name, split_selected_rows in (("train", train_rows), ("val", val_rows), ("test", test_rows)):
            if not split_selected_rows:
                raise ValueError(
                    "Configured split manifest produced an empty "
                    f"{split_name} split after task compatibility filtering for pipeline.kind={pipeline_spec.kind!r}"
                )
    else:
        compatible_rows = _filter_rows_for_task(rows, reader=reader, task_config=task_config)
        if not compatible_rows:
            raise ValueError("No processed scene-year rows satisfy the configured pipeline sources and targets")
        train_rows, val_rows, test_rows = _split_rows_by_scene(
            compatible_rows,
            random_seed=random_seed,
            test_size=pipeline_spec.test_size,
        )

    compatible_rows = train_rows + val_rows + test_rows

    datasets = [
        ProcessedSceneDataset(
            split_rows,
            reader=reader,
            pipeline_spec=pipeline_spec,
            preprocessing_spec=preprocessing_spec,
            batch_schema=batch_schema,
            random_seed=random_seed + split_idx * 10_000,
            cache_scenes=pipeline_spec.cache_scenes,
            preload_scenes=pipeline_spec.preload_scenes,
        )
        for split_idx, split_rows in enumerate((train_rows, val_rows, test_rows))
    ]

    samplers = []
    if is_distributed:
        for dataset, shuffle in zip(datasets, (True, False, False)):
            samplers.append(
                DistributedSampler(
                    dataset,
                    shuffle=shuffle,
                    seed=random_seed,
                    num_replicas=world_size,
                    rank=local_rank,
                )
            )
    else:
        samplers = [None, None, None]

    num_workers = max(0, pipeline_spec.num_workers // max(1, world_size))
    common_kwargs = {
        "batch_size": batch_size,
        "pin_memory": pipeline_spec.pin_memory,
        "num_workers": num_workers,
        "collate_fn": collate_processed_samples,
    }
    if num_workers > 0:
        common_kwargs["prefetch_factor"] = pipeline_spec.prefetch_factor
        common_kwargs["persistent_workers"] = pipeline_spec.persistent_workers

    split_shuffle = (bool(shuffle_train), False, False)
    loaders = [
        DataLoader(
            dataset=dataset,
            shuffle=(sampler is None and shuffle),
            sampler=sampler,
            worker_init_fn=seed_worker,
            generator=seeded_torch_generator(
                random_seed,
                offset=split_idx * 10_000 + local_rank * 1_000_000,
            ),
            **common_kwargs,
        )
        for split_idx, (dataset, sampler, shuffle) in enumerate(zip(datasets, samplers, split_shuffle))
    ]

    if local_rank == 0:
        if runtime_split_manifest_path and runtime_split_manifest_path != preprocessing_split_manifest_path:
            logger.info(f"Use runtime split manifest: {runtime_split_manifest_path}")
        logger.info(
            "Processed samples: "
            f"total={len(compatible_rows)} train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}"
        )
        if normalize_eump_pipeline_kind(pipeline_spec.kind) == EUMP_CITY_SCALAR_DOWNSTREAM_KIND:
            scalar_feature_dim = batch_schema.raster_input_channels + batch_schema.scalar_input_dim
            logger.info(
                "Processed city-scalar schema: "
                f"processed_source_features={scalar_feature_dim} "
                f"city_mask_source={pipeline_spec.city_mask_source or 'none'} "
                f"scalar_targets={batch_schema.scalar_output_dims}"
            )
        else:
            logger.info(
                "Processed batch schema: "
                f"raster_input_channels={batch_schema.raster_input_channels} "
                f"raster_output_channels={batch_schema.raster_output_channels} "
                f"scalar_targets={batch_schema.scalar_output_dims}"
            )

    return loaders[0], loaders[1], loaders[2], batch_schema


__all__ = [
    "ProcessedSample",
    "ProcessedSceneDataset",
    "build_eump_dataloaders",
    "collate_processed_samples",
]
