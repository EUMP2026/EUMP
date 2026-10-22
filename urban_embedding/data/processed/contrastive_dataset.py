from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from urban_embedding.core.reproducibility import seed_worker, seeded_torch_generator
from urban_embedding.data.crop import (
    materialize_crop_sample,
    sample_random_crop_box,
    stack_model_raster_inputs,
)

from .contrastive_batch import ContrastiveBatch, ContrastivePair
from .dataset import (
    _filter_rows_for_task,
    _select_rows_for_scene_ids,
    _split_rows_by_scene,
)
from .pipeline_spec import (
    PipelineSpec,
    ProcessedBatchSchema,
    build_active_preprocessing_spec,
    build_eump_batch_schema,
)
from .processing import (
    load_preprocessing_spec,
    process_scene_record,
    validate_preprocessing_spec,
    validate_preprocessing_stats_scope,
)
from .reader import ProcessedSceneStoreReader
from .split_manifest import load_split_manifest, split_scene_ids_from_manifest


def _apply_view_corruption(
    raster: np.ndarray,
    valid_mask: np.ndarray,
    *,
    rng: np.random.Generator,
    channel_dropout_p: float,
    noise_std: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return an independent view of ``raster`` with channel dropout and noise.

    Only pixels with ``valid_mask==True`` get noise added; dropped channels
    are zeroed and their ``valid_mask`` rows are set to False so the encoder
    treats them as missing rather than as a meaningful zero.
    """
    raster = raster.copy()
    valid_mask = valid_mask.copy()
    num_channels = raster.shape[0]
    if channel_dropout_p > 0.0 and num_channels > 0:
        drop = rng.random(num_channels) < channel_dropout_p
        if drop.any():
            raster[drop] = 0.0
            valid_mask[drop] = False
    if noise_std > 0.0:
        noise = rng.standard_normal(raster.shape).astype(np.float32) * float(noise_std)
        noise = noise * valid_mask.astype(np.float32)
        raster = raster + noise
    return raster, valid_mask


class ProcessedContrastivePairDataset(Dataset):
    """Materialize one crop per ``(scene_id, year)`` row and return two views.

    The crop box is shared between the two views (``same_crop_two_stochastic_views``);
    corruption diverges via independent channel dropout plus Gaussian noise.
    No targets are constructed because contrastive pretraining does not use
    reconstruction supervision.
    """

    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        reader: ProcessedSceneStoreReader,
        pipeline_spec: PipelineSpec,
        preprocessing_spec: dict[str, Any],
        batch_schema: ProcessedBatchSchema,
        channel_dropout_p: float = 0.1,
        noise_std: float = 0.01,
        random_seed: int = 42,
    ) -> None:
        self.rows = list(rows)
        self.reader = reader
        self.pipeline_spec = pipeline_spec
        self.task_config = pipeline_spec.to_task_config()
        self.preprocessing_spec = preprocessing_spec
        self.batch_schema = batch_schema
        self.channel_dropout_p = float(channel_dropout_p)
        self.noise_std = float(noise_std)
        self.random_seed = int(random_seed)
        self._current_epoch = 0

    def __len__(self) -> int:
        return len(self.rows)

    def set_epoch(self, epoch: int) -> None:
        self._current_epoch = int(epoch)

    def _seed_for(self, row: dict[str, Any], index: int, view: int) -> int:
        import zlib

        key = f"{row['scene_id']}:{int(row['year'])}:{view}"
        h = zlib.crc32(key.encode("utf-8")) & 0xFFFFFFFF
        seed_seq = np.random.SeedSequence(
            [
                int(self.random_seed) & 0xFFFFFFFF,
                int(self._current_epoch) & 0xFFFFFFFF,
                int(index) & 0xFFFFFFFF,
                int(h),
            ]
        )
        return int(seed_seq.generate_state(1, dtype=np.uint32)[0])

    def _materialize_crop(self, row: dict[str, Any], index: int) -> tuple[np.ndarray, np.ndarray]:
        scene = self.reader.load_scene_record(row["scene_id"], int(row["year"]), self.task_config)
        processed = process_scene_record(scene, self.task_config, self.preprocessing_spec)
        crop_seed = self._seed_for(row, index, view=0)
        crop_box = sample_random_crop_box(
            processed.spatial_shape,
            self.pipeline_spec.crop_size,
            center_ratio=self.pipeline_spec.center_ratio,
            rng=np.random.default_rng(crop_seed),
        )
        crop_sample = materialize_crop_sample(processed, crop_box, preprocessing_spec=self.preprocessing_spec)
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
        return raster_inputs.astype(np.float32, copy=False), raster_valid_mask.astype(bool, copy=False)

    def __getitem__(self, index: int) -> ContrastivePair:
        row = self.rows[index]
        raster, valid_mask = self._materialize_crop(row, index)
        view_a, valid_a = _apply_view_corruption(
            raster,
            valid_mask,
            rng=np.random.default_rng(self._seed_for(row, index, view=1)),
            channel_dropout_p=self.channel_dropout_p,
            noise_std=self.noise_std,
        )
        view_b, valid_b = _apply_view_corruption(
            raster,
            valid_mask,
            rng=np.random.default_rng(self._seed_for(row, index, view=2)),
            channel_dropout_p=self.channel_dropout_p,
            noise_std=self.noise_std,
        )
        meta = {
            "scene_id": str(row["scene_id"]),
            "year": int(row["year"]),
        }
        return ContrastivePair(
            view_a=torch.from_numpy(view_a),
            view_b=torch.from_numpy(view_b),
            valid_mask_a=torch.from_numpy(valid_a),
            valid_mask_b=torch.from_numpy(valid_b),
            meta=meta,
        )


def collate_contrastive_pairs(pairs: list[ContrastivePair]) -> ContrastiveBatch:
    return ContrastiveBatch(
        view_a=torch.stack([pair.view_a for pair in pairs], dim=0),
        view_b=torch.stack([pair.view_b for pair in pairs], dim=0),
        valid_mask_a=torch.stack([pair.valid_mask_a for pair in pairs], dim=0),
        valid_mask_b=torch.stack([pair.valid_mask_b for pair in pairs], dim=0),
        meta=[pair.meta for pair in pairs],
    )


def build_eump_contrastive_dataloaders(
    *,
    data_config: dict[str, Any],
    pipeline_spec: PipelineSpec,
    logger,
    random_seed: int,
    batch_size: int,
    local_rank: int,
    world_size: int,
    channel_dropout_p: float = 0.1,
    noise_std: float = 0.01,
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

    runtime_split_manifest_path = data_config.get("runtime_split_manifest_path") or data_config.get(
        "split_manifest_path"
    )
    if runtime_split_manifest_path:
        split_manifest = load_split_manifest(
            runtime_split_manifest_path,
            available_scene_ids=reader.scene_index.keys(),
            expected_year_range=pipeline_spec.year_range,
        )
        train_scene_ids, val_scene_ids, test_scene_ids = split_scene_ids_from_manifest(split_manifest)
        train_rows = _filter_rows_for_task(
            _select_rows_for_scene_ids(rows, train_scene_ids), reader=reader, task_config=task_config
        )
        val_rows = _filter_rows_for_task(
            _select_rows_for_scene_ids(rows, val_scene_ids), reader=reader, task_config=task_config
        )
        test_rows = _filter_rows_for_task(
            _select_rows_for_scene_ids(rows, test_scene_ids), reader=reader, task_config=task_config
        )
    else:
        compatible_rows = _filter_rows_for_task(rows, reader=reader, task_config=task_config)
        train_rows, val_rows, test_rows = _split_rows_by_scene(
            compatible_rows,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            random_seed=random_seed,
        )

    if local_rank == 0:
        logger.info(
            f"EUMP contrastive dataset rows: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}"
        )

    def _make_loader(split_rows: list[dict[str, Any]], *, shuffle: bool) -> DataLoader:
        dataset = ProcessedContrastivePairDataset(
            split_rows,
            reader=reader,
            pipeline_spec=pipeline_spec,
            preprocessing_spec=preprocessing_spec,
            batch_schema=batch_schema,
            channel_dropout_p=channel_dropout_p,
            noise_std=noise_std,
            random_seed=random_seed,
        )
        sampler = (
            DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=shuffle)
            if is_distributed and world_size > 1
            else None
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=(sampler is None and shuffle),
            num_workers=int(data_config.get("num_workers", 4)),
            pin_memory=bool(data_config.get("pin_memory", True)),
            persistent_workers=bool(data_config.get("persistent_workers", False)),
            collate_fn=collate_contrastive_pairs,
            worker_init_fn=seed_worker,
            generator=seeded_torch_generator(random_seed),
        )

    return (
        _make_loader(train_rows, shuffle=shuffle_train),
        _make_loader(val_rows, shuffle=False),
        _make_loader(test_rows, shuffle=False),
        batch_schema,
    )


__all__ = [
    "ProcessedContrastivePairDataset",
    "build_eump_contrastive_dataloaders",
    "collate_contrastive_pairs",
]



