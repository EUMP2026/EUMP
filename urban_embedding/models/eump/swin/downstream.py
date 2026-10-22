from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from urban_embedding.data.processed.pipeline_spec import ProcessedBatchSchema, TargetSchema

from urban_embedding.models.eump.convnext_unet.encoder import ConvNeXtEncoder

from .reconstruction import SwinWithConv


RASTER_HEAD_TYPES = {"simple", "progressive_upsample"}
RASTER_UPSAMPLE_MODES = {"bilinear", "nearest"}


class SegFormerStyleUpsampleHead(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )
        self.out = nn.Conv2d(hidden_dim, out_channels, kernel_size=1)

    def forward(self, embedding: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        features = self.proj(embedding)
        features = F.interpolate(features, size=output_size, mode="bilinear", align_corners=False)
        return self.out(features)


def _normalize_raster_head_type(value: object | None) -> str:
    head_type = "simple" if value is None else str(value).lower()
    if head_type not in RASTER_HEAD_TYPES:
        raise ValueError(f"Unsupported model.raster_head_type={head_type!r}; expected one of {sorted(RASTER_HEAD_TYPES)}")
    return head_type


def _normalize_raster_upsample_mode(value: object | None) -> str:
    mode = "bilinear" if value is None else str(value).lower()
    if mode not in RASTER_UPSAMPLE_MODES:
        raise ValueError(
            f"Unsupported model.raster_upsample_mode={mode!r}; expected one of {sorted(RASTER_UPSAMPLE_MODES)}"
        )
    return mode


def _group_norm_groups(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class _ProgressiveConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_group_norm_groups(out_channels), out_channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_group_norm_groups(out_channels), out_channels),
            nn.GELU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class ProgressiveUpsampleRasterHead(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        stages: int = 5,
        min_channels: int = 48,
        upsample_mode: str = "bilinear",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"ProgressiveUpsampleRasterHead in_channels must be positive, got {in_channels}")
        if out_channels <= 0:
            raise ValueError(f"ProgressiveUpsampleRasterHead out_channels must be positive, got {out_channels}")
        if int(stages) <= 0:
            raise ValueError(f"model.raster_upsample_stages must be > 0, got {stages}")
        if int(min_channels) <= 0:
            raise ValueError(f"model.raster_min_channels must be > 0, got {min_channels}")
        self.upsample_mode = _normalize_raster_upsample_mode(upsample_mode)
        self.stages = int(stages)
        min_channels = int(min_channels)

        stem_channels = max(min_channels, int(in_channels) // 2)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, kernel_size=1),
            nn.GroupNorm(_group_norm_groups(stem_channels), stem_channels),
            nn.GELU(),
        )

        blocks: list[nn.Module] = []
        current_channels = stem_channels
        for stage_idx in range(self.stages):
            next_channels = max(min_channels, current_channels if stage_idx == 0 else current_channels // 2)
            blocks.append(_ProgressiveConvBlock(current_channels, next_channels, dropout=dropout))
            current_channels = next_channels
        self.blocks = nn.ModuleList(blocks)
        self.out = nn.Conv2d(current_channels, out_channels, kernel_size=1)

    def _upsample(self, features: torch.Tensor, *, size: tuple[int, int] | None = None) -> torch.Tensor:
        kwargs: dict[str, object] = {"mode": self.upsample_mode}
        if self.upsample_mode == "bilinear":
            kwargs["align_corners"] = False
        if size is None:
            return F.interpolate(features, scale_factor=2, **kwargs)
        return F.interpolate(features, size=size, **kwargs)

    def forward(self, embedding: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        features = self.stem(embedding)
        for block in self.blocks:
            features = self._upsample(features)
            features = block(features)
        if tuple(features.shape[-2:]) != tuple(output_size):
            features = self._upsample(features, size=output_size)
        return self.out(features)


class PooledScalarHead(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.net(embedding)


class EumpLightDualHeadModel(nn.Module):
    def __init__(
        self,
        *,
        encoder: nn.Module,
        embedding_channels: int,
        target_schema: TargetSchema,
        target_kind: str,
        freeze_encoder: bool = True,
        decoder_hidden_dim: int = 256,
        scalar_hidden_dim: int = 256,
        dropout: float = 0.1,
        raster_head_type: str = "simple",
        raster_upsample_mode: str = "bilinear",
        raster_upsample_stages: int = 5,
        raster_min_channels: int = 48,
    ) -> None:
        super().__init__()
        if target_kind not in {"raster", "scalar"}:
            raise ValueError(f"Unsupported downstream target kind {target_kind!r}")
        self.encoder = encoder
        self.freeze_encoder = bool(freeze_encoder)
        self.target_name = target_schema.name
        self.target_kind = target_kind
        self.raster_heads = nn.ModuleDict()
        self.scalar_heads = nn.ModuleDict()

        if target_kind == "raster":
            raster_head_type = _normalize_raster_head_type(raster_head_type)
            if raster_head_type == "progressive_upsample":
                self.raster_heads[self.target_name] = ProgressiveUpsampleRasterHead(
                    in_channels=embedding_channels,
                    out_channels=target_schema.channels,
                    stages=raster_upsample_stages,
                    min_channels=raster_min_channels,
                    upsample_mode=raster_upsample_mode,
                    dropout=dropout,
                )
            else:
                self.raster_heads[self.target_name] = SegFormerStyleUpsampleHead(
                    in_channels=embedding_channels,
                    out_channels=target_schema.channels,
                    hidden_dim=decoder_hidden_dim,
                    dropout=dropout,
                )
        else:
            self.scalar_heads[self.target_name] = PooledScalarHead(
                in_channels=embedding_channels,
                out_dim=target_schema.channels,
                hidden_dim=scalar_hidden_dim,
                dropout=dropout,
            )

        if self.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()
        else:
            self._freeze_downstream_unused_encoder_parameters()

    def _freeze_downstream_unused_encoder_parameters(self) -> None:
        """Freeze Swin MIM-only parameters that downstream forward never touches."""
        swin = getattr(self.encoder, "swin", None)
        decoder = getattr(swin, "decoder", None)
        if decoder is not None:
            for param in decoder.parameters():
                param.requires_grad = False

        for name, param in self.encoder.named_parameters():
            if name.endswith("mask_token") or ".mask_token" in name:
                param.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def _encode(self, raster_inputs: torch.Tensor) -> torch.Tensor:
        if self.freeze_encoder:
            with torch.no_grad():
                return self.encoder.encode(raster_inputs, bool_masked_pos=None)
        return self.encoder.encode(raster_inputs, bool_masked_pos=None)

    def forward(
        self,
        raster_inputs: torch.Tensor,
        scalar_inputs: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        del scalar_inputs
        embedding = self._encode(raster_inputs)
        output_size = tuple(int(value) for value in raster_inputs.shape[-2:])
        raster_predictions = {
            name: head(embedding, output_size)
            for name, head in self.raster_heads.items()
        }
        scalar_predictions = {
            name: head(embedding)
            for name, head in self.scalar_heads.items()
        }
        return raster_predictions, scalar_predictions


@dataclass(frozen=True, slots=True)
class EncoderLoadReport:
    loaded_keys: tuple[str, ...]
    skipped_keys: tuple[str, ...]


def _checkpoint_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict) and "model_state_dict" in payload:
        payload = payload["model_state_dict"]
    if not isinstance(payload, dict):
        raise TypeError(f"Expected checkpoint state dict mapping, got {type(payload)!r}")
    return payload


def _encoder_key(checkpoint_key: str) -> str | None:
    key = checkpoint_key.removeprefix("module.")
    if key.startswith("backbone."):
        key = key.removeprefix("backbone.")
    elif key.startswith("encoder."):
        key = key.removeprefix("encoder.")

    if key.startswith("swin.decoder") or key.startswith("scalar_heads"):
        return None
    if key.startswith("projection.") or key.startswith("pool."):
        # Projection head from contrastive pretrain; downstream does not load it.
        return None
    if key.startswith("swin.swinv2") or key.startswith("conv_layer"):
        return key
    # ConvNeXt reconstruction checkpoints store the encoder below
    # ``backbone.*``.  The prefix is removed above, leaving these keys.
    if key.startswith(("patch_embed.", "stages.", "downsamples.")):
        return key
    return None


def load_eump_reconstruction_encoder_weights(
    encoder: nn.Module,
    checkpoint_path: str | Path,
    *,
    map_location: torch.device | str = "cpu",
    require_input_projection: bool = False,
) -> EncoderLoadReport:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"EUMP reconstruction checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    checkpoint_state = _checkpoint_state_dict(checkpoint)
    encoder_state = encoder.state_dict()
    loadable_state: dict[str, torch.Tensor] = {}
    skipped: list[str] = []

    for checkpoint_key, value in checkpoint_state.items():
        encoder_key = _encoder_key(str(checkpoint_key))
        if encoder_key is None:
            skipped.append(str(checkpoint_key))
            continue
        if encoder_key not in encoder_state or tuple(encoder_state[encoder_key].shape) != tuple(value.shape):
            skipped.append(str(checkpoint_key))
            continue
        loadable_state[encoder_key] = value

    if require_input_projection:
        required = {"conv_layer.weight", "conv_layer.bias"}
        missing = tuple(sorted(required.difference(loadable_state)))
        if missing:
            raise ValueError(
                "Frozen EUMP downstream encoder requires reconstruction checkpoint "
                "input projection weights to load. Missing or shape-mismatched keys: "
                f"{missing}. Use a reconstruction checkpoint trained with the same "
                "broadcast_scalar setting and raster input channels."
            )

    if not loadable_state:
        raise ValueError(f"No encoder weights could be loaded from {checkpoint_path}")

    encoder_state.update(loadable_state)
    encoder.load_state_dict(encoder_state)
    return EncoderLoadReport(
        loaded_keys=tuple(sorted(loadable_state)),
        skipped_keys=tuple(skipped),
    )


def _target_kind(batch_schema: ProcessedBatchSchema, target_schema: TargetSchema) -> str:
    if any(target.name == target_schema.name for target in batch_schema.raster_targets):
        return "raster"
    if any(target.name == target_schema.name for target in batch_schema.scalar_targets):
        return "scalar"
    raise ValueError(f"Target {target_schema.name!r} is not present in batch schema")


def build_eump_downstream_model(
    *,
    model_config: dict[str, Any],
    batch_schema: ProcessedBatchSchema,
    target_schema: TargetSchema,
    base_swin_path: str,
    encoder_source: str,
    encoder_checkpoint_path: str | None = None,
    load_pretrained: bool = True,
    freeze_encoder: bool = True,
    map_location: torch.device | str = "cpu",
) -> EumpLightDualHeadModel:
    arch = str(model_config.get("downstream_arch", "light_dual_head"))
    if arch != "light_dual_head":
        raise ValueError("Unsupported model.downstream_arch={!r}; expected 'light_dual_head'".format(arch))
    backbone_kind = str(model_config.get("backbone_kind", "swin")).lower()
    if backbone_kind not in {"swin", "convnext"}:
        raise ValueError(
            "model.backbone_kind must be 'swin' or 'convnext', "
            f"got {backbone_kind!r}"
        )

    valid_sources = {"reconstruction_checkpoint", "swin_pretrained"}
    if backbone_kind == "convnext":
        valid_sources = {
            "convnext_reconstruction_checkpoint",
            "convnext_pretrained",
            "convnext_scratch",
        }
    if encoder_source not in valid_sources:
        raise ValueError(
            f"downstream.encoder_source for backbone_kind={backbone_kind!r} must be one of "
            f"{sorted(valid_sources)}, "
            f"got {encoder_source!r}"
        )

    if backbone_kind == "convnext":
        depths = tuple(int(value) for value in model_config.get("depths", (3, 3, 27, 3)))
        dims = tuple(int(value) for value in model_config.get("dims", (192, 384, 768, 1536)))
        if len(depths) != 4 or len(dims) != 4:
            raise ValueError("ConvNeXt downstream depths and dims must each contain four stages")
        encoder = ConvNeXtEncoder(
            input_channels=batch_schema.raster_input_channels,
            depths=depths,
            dims=dims,
            drop_path_rate=float(
                model_config.get("drop_path_rate", model_config.get("backbone_drop_path_rate", 0.0))
            ),
        )
    else:
        encoder = SwinWithConv(
            model_path=base_swin_path,
            input_channels=batch_schema.raster_input_channels,
            output_channels=max(1, batch_schema.raster_output_channels),
            load_pretrained=load_pretrained,
            backbone_hidden_dropout_prob=model_config.get("backbone_hidden_dropout_prob"),
            backbone_attention_probs_dropout_prob=model_config.get("backbone_attention_probs_dropout_prob"),
        )

    if encoder_source in {"reconstruction_checkpoint", "convnext_reconstruction_checkpoint", "convnext_pretrained"}:
        if encoder_checkpoint_path is None:
            raise ValueError("downstream.encoder_checkpoint_file is required for reconstruction_checkpoint source")
        load_report = load_eump_reconstruction_encoder_weights(
            encoder,
            encoder_checkpoint_path,
            map_location=map_location,
            require_input_projection=freeze_encoder,
        )
        expected_prefix = "swin.swinv2" if backbone_kind == "swin" else "stages"
        if not any(key.startswith(expected_prefix) for key in load_report.loaded_keys):
            raise ValueError(
                f"Checkpoint {encoder_checkpoint_path} did not contain {backbone_kind} encoder weights; "
                f"loaded keys did not include {expected_prefix!r}."
            )

    if backbone_kind == "convnext":
        embedding_channels = int(encoder.num_features)
    else:
        embedding_channels = encoder.config.embed_dim * (2 ** (encoder.config.num_layers - 1))
    return EumpLightDualHeadModel(
        encoder=encoder,
        embedding_channels=int(embedding_channels),
        target_schema=target_schema,
        target_kind=_target_kind(batch_schema, target_schema),
        freeze_encoder=freeze_encoder,
        decoder_hidden_dim=int(model_config.get("decoder_hidden_dim", 256)),
        scalar_hidden_dim=int(model_config.get("scalar_hidden_dim", 256)),
        dropout=float(model_config.get("dropout", 0.1)),
        raster_head_type=str(model_config.get("raster_head_type", "simple")),
        raster_upsample_mode=str(model_config.get("raster_upsample_mode", "bilinear")),
        raster_upsample_stages=int(model_config.get("raster_upsample_stages", 5)),
        raster_min_channels=int(model_config.get("raster_min_channels", 48)),
    )


__all__ = [
    "EncoderLoadReport",
    "PooledScalarHead",
    "EumpLightDualHeadModel",
    "ProgressiveUpsampleRasterHead",
    "SegFormerStyleUpsampleHead",
    "build_eump_downstream_model",
    "load_eump_reconstruction_encoder_weights",
]
