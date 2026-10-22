from __future__ import annotations

from collections.abc import Mapping
import math
import os
from typing import Any

import torch

from urban_embedding.core.paths import MODEL_DIR
from urban_embedding.data.processed.contrastive_batch import ContrastiveBatch
from urban_embedding.data.processed.contrastive_dataset import build_eump_contrastive_dataloaders
from urban_embedding.data.processed.pipeline_spec import PipelineSpec, ProcessedBatchSchema
from urban_embedding.models.eump.contrastive.moco_model import EumpSwinMoCoModel
from urban_embedding.training.distributed import get_model, wrap_ddp
from urban_embedding.training.masking import generate_mask
from urban_embedding.training.schedulers import (
    build_warmup_cosine_scheduler,
    normalize_warmup_cosine_scheduler_config,
    warmup_cosine_metadata,
)
from urban_embedding.training.tensorboard import TensorBoardScalarWriter

from .loss import nt_xent_loss


class EumpSwinContrastiveTrainer:
    """Momentum-encoder contrastive pretrainer for the EUMP Swin encoder.

    Each minibatch is a ``ContrastiveBatch`` of two stochastic views of the
    same crop. Two independent patch masks zero-fill masked positions before
    the encoder forward. The encoder produces L2-normalized projections
    via the projection head; the NT-Xent loss draws negatives from the
    DDP-gathered global batch.

    ``scheduler_epochs`` optionally defines a longer learning-rate horizon
    than the executed training budget, for a fixed truncation of the schedule.

    Checkpoint payload exposes encoder weights with the same key prefixes
    as the Swin MAE trainer (``module.backbone.swin.swinv2.*``,
    ``module.backbone.conv_layer.*``), so the existing downstream loader
    consumes it without modification. The projection head lives under
    ``module.projection.*`` and is skipped by the downstream loader.
    """

    def __init__(
        self,
        *,
        logger,
        device: torch.device,
        model_name: str,
        pretrain_model: str,
        data_config: dict,
        pipeline_config: dict,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.05,
        batch_size: int = 32,
        n_epochs: int = 200,
        scheduler_epochs: int | None = None,
        random_seed: int = 42,
        local_rank: int = 0,
        world_size: int = 1,
        accumulation_steps: int = 1,
        scratch_backbone: bool = False,
        save_file_name: str = "eump_contrastive_model.pt",
        checkpoint_file_name: str = "eump_contrastive_checkpoint.pt",
        resume: bool = False,
        resume_checkpoint: str | None = None,
        log_dir: str | None = None,
        save_best_model: bool = False,
        save_best_checkpoint: bool = True,
        save_final_model: bool = True,
        best_checkpoint_file_name: str = "eump_contrastive_best_checkpoint.pt",
        scheduler_config: Mapping[str, object] | None = None,
        mask_ratio: float = 0.6,
        temperature: float = 0.1,
        projection_hidden_dim: int = 2048,
        projection_dim: int = 128,
        channel_dropout_p: float = 0.1,
        noise_std: float = 0.01,
        momentum: float = 0.996,
    ) -> None:
        self.logger = logger
        self.device = device
        self.model_name = model_name
        self.pretrain_model = pretrain_model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.scheduler_epochs = int(scheduler_epochs) if scheduler_epochs is not None else None
        self.random_seed = random_seed
        self.local_rank = local_rank
        self.world_size = world_size
        self.accumulation_steps = max(1, int(accumulation_steps))
        self.scratch_backbone = scratch_backbone
        self.save_file_name = save_file_name
        self.checkpoint_file_name = checkpoint_file_name
        self.resume = resume
        self.resume_checkpoint = resume_checkpoint
        self.save_best_model_enabled = bool(save_best_model)
        self.save_best_checkpoint_enabled = bool(save_best_checkpoint)
        self.save_final_model_enabled = bool(save_final_model)
        self.best_checkpoint_file_name = best_checkpoint_file_name
        self.scheduler_config = normalize_warmup_cosine_scheduler_config(scheduler_config)
        self.mask_ratio = float(mask_ratio)
        self.temperature = float(temperature)
        self.projection_hidden_dim = int(projection_hidden_dim)
        self.projection_dim = int(projection_dim)
        self.channel_dropout_p = float(channel_dropout_p)
        self.noise_std = float(noise_std)
        self.momentum = float(momentum)
        self.start_epoch = 1
        self.best_val_loss = float("inf")
        self.global_step = 0
        if self.n_epochs < 1:
            raise ValueError(f"n_epochs must be >= 1, got {self.n_epochs}")
        if self.scheduler_epochs is not None and self.scheduler_epochs < self.n_epochs:
            raise ValueError(
                "scheduler_epochs must be >= n_epochs when set; "
                f"got scheduler_epochs={self.scheduler_epochs}, n_epochs={self.n_epochs}"
            )

        self.pipeline_spec = PipelineSpec.from_mapping(pipeline_config)
        self.tensorboard = TensorBoardScalarWriter(log_dir=log_dir, enabled=True, rank=self.local_rank)
        self.train_dl, self.val_dl, self.test_dl, self.batch_schema = build_eump_contrastive_dataloaders(
            data_config=data_config,
            pipeline_spec=self.pipeline_spec,
            logger=self.logger,
            random_seed=self.random_seed,
            batch_size=self.batch_size,
            local_rank=self.local_rank,
            world_size=self.world_size,
            channel_dropout_p=self.channel_dropout_p,
            noise_std=self.noise_std,
        )
        self.model = self.initialize_model(self.batch_schema)
        self.model = wrap_ddp(self.model, self.device, self.local_rank, find_unused_parameters=True)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self.scheduler = self.build_scheduler(self.n_epochs)

    def initialize_model(self, batch_schema: ProcessedBatchSchema) -> EumpSwinMoCoModel:
        if batch_schema.raster_input_channels <= 0:
            raise ValueError("EUMP Swin contrastive requires at least one raster input channel")
        model_path = os.path.join(MODEL_DIR, self.pretrain_model)
        model = EumpSwinMoCoModel(
            model_path=model_path,
            input_channels=batch_schema.raster_input_channels,
            load_pretrained=not self.scratch_backbone,
            projection_hidden_dim=self.projection_hidden_dim,
            projection_dim=self.projection_dim,
            momentum=self.momentum,
        )
        origin = "scratch" if self.scratch_backbone else "pretrained"
        if self.local_rank == 0:
            self.logger.info(f"Initialize EUMP Swin MoCo model ({origin}, m={self.momentum}) from {model_path}")
        return model.to(self.device)

    def build_scheduler(self, n_epochs=None):
        schedule_epochs = self.n_epochs if n_epochs is None else n_epochs
        if self.scheduler_epochs is not None:
            schedule_epochs = self.scheduler_epochs
        scheduler, metadata = build_warmup_cosine_scheduler(
            optimizer=self.optimizer,
            base_lr=self.learning_rate,
            num_batches=len(self.train_dl),
            accumulation_steps=self.accumulation_steps,
            epochs=schedule_epochs,
            warmup_fraction=self.scheduler_config.warmup_fraction,
            lr_start_factor=self.scheduler_config.lr_start_factor,
            lr_end_factor=self.scheduler_config.lr_end_factor,
        )
        self.scheduler_metadata = metadata
        return scheduler

    def current_scheduler_metadata(self):
        return getattr(self, "scheduler_metadata", None)

    def scheduler_metadata_for_epochs(self, n_epochs):
        return warmup_cosine_metadata(
            base_lr=self.learning_rate,
            num_batches=len(self.train_dl),
            accumulation_steps=self.accumulation_steps,
            epochs=self.scheduler_epochs if self.scheduler_epochs is not None else n_epochs,
            warmup_fraction=self.scheduler_config.warmup_fraction,
            lr_start_factor=self.scheduler_config.lr_start_factor,
            lr_end_factor=self.scheduler_config.lr_end_factor,
        )

    def reset_scheduler(self, n_epochs):
        self.scheduler = self.build_scheduler(n_epochs)

    def _apply_patch_mask(
        self,
        raster: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.mask_ratio <= 0.0:
            return raster
        base = get_model(self.model)
        patch_size = int(base.config.patch_size)
        image_size = int(raster.shape[-1])
        num_patches = (image_size // patch_size) ** 2
        num_masked = math.ceil(num_patches * self.mask_ratio)
        mask = generate_mask(raster.shape[0], num_patches, num_masked, device=raster.device)
        if mask is None:
            return raster
        patch_h = image_size // patch_size
        patch_w = int(raster.shape[-2]) // patch_size
        spatial_mask = (
            mask.reshape(raster.shape[0], patch_h, patch_w)
            .repeat_interleave(patch_size, 1)
            .repeat_interleave(patch_size, 2)
            .unsqueeze(1)
            .contiguous()
        )
        keep = (~spatial_mask).to(dtype=raster.dtype)
        return raster * keep

    def _current_learning_rate(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _write_tensorboard(
        self,
        *,
        phase: str,
        level: str,
        metrics: Mapping[str, torch.Tensor | float | int],
        step: int,
    ) -> None:
        if self.tensorboard is None:
            return
        scalars = {}
        for name, value in metrics.items():
            if torch.is_tensor(value):
                scalars[name] = float(value.detach().cpu().item())
            else:
                scalars[name] = float(value)
        self.tensorboard.add_scalars(phase=phase, level=level, scalars=scalars, step=step)

    def _run_epoch(self, dataloader, *, train: bool, epoch: int, phase: str) -> dict[str, torch.Tensor]:
        dataset = getattr(dataloader, "dataset", None)
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)
        sampler = getattr(dataloader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        self.model.train(mode=train)
        metric_sums: dict[str, torch.Tensor] = {}
        metric_counts: dict[str, int] = {}

        if train:
            self.optimizer.zero_grad()

        context = torch.enable_grad() if train else torch.no_grad()
        num_batches = len(dataloader)
        with context:
            for batch_idx, batch in enumerate(dataloader):
                batch = batch.to(self.device)
                view_a = self._apply_patch_mask(batch.view_a, batch.valid_mask_a)
                view_b = self._apply_patch_mask(batch.view_b, batch.valid_mask_b)
                base_model = get_model(self.model)
                has_momentum = hasattr(base_model, "forward_momentum")
                if has_momentum:
                    z_a = self.model(view_a)
                    with torch.no_grad():
                        z_b = base_model.forward_momentum(view_b)
                else:
                    z_both = self.model(torch.cat([view_a, view_b], dim=0))
                    z_a, z_b = z_both.chunk(2, dim=0)
                loss, stats = nt_xent_loss(z_a, z_b, temperature=self.temperature)

                if train:
                    scaled_loss = loss / self.accumulation_steps
                    scaled_loss.backward()
                    is_step = (batch_idx + 1) % self.accumulation_steps == 0 or (batch_idx + 1) == num_batches
                    if is_step:
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                        self.global_step += 1
                        if has_momentum:
                            base_model.update_momentum()
                        stats = dict(stats)
                        stats["state/grad_norm"] = grad_norm.detach()

                stats = dict(stats)
                stats["loss/total"] = loss.detach()
                for name, value in stats.items():
                    if name not in metric_sums:
                        metric_sums[name] = value.detach().clone()
                        metric_counts[name] = 1
                    else:
                        metric_sums[name] = metric_sums[name] + value.detach()
                        metric_counts[name] += 1

        epoch_metrics = {
            name: metric_sums[name] / max(1, metric_counts[name]) for name in metric_sums
        }
        tensorboard_metrics: dict[str, torch.Tensor | float | int] = dict(epoch_metrics)
        tensorboard_metrics.update(
            {
                "state/learning_rate": self._current_learning_rate(),
                "state/epoch": epoch,
                "state/global_step": self.global_step,
            }
        )
        self._write_tensorboard(phase=phase, level="epoch", metrics=tensorboard_metrics, step=epoch)
        return epoch_metrics

    def format_epoch_log(self, phase: str, metrics: Mapping[str, float]) -> str:
        loss = float(metrics.get("loss/total", 0.0))
        top1 = float(metrics.get("contrastive/top1", 0.0))
        pos = float(metrics.get("contrastive/pos_sim", 0.0))
        return f"{phase} nt_xent={loss:.6f} top1={top1:.4f} pos_sim={pos:.4f}"

    def format_training_summary(self, duration_minutes: float) -> str:
        return f"Contrastive pretrain finished, Total Training Time: {duration_minutes:.2f} mins"

    def close(self) -> None:
        writer = getattr(self, "tensorboard", None)
        if writer is not None:
            writer.close()

    def train_epoch(self, epoch):
        return self._run_epoch(self.train_dl, train=True, epoch=epoch, phase="train")

    def validate_epoch(self, epoch):
        return self._run_epoch(self.val_dl, train=False, epoch=epoch, phase="val")

    def test_epoch(self, epoch):
        return self._run_epoch(self.test_dl, train=False, epoch=epoch, phase="test")


__all__ = ["EumpSwinContrastiveTrainer"]



