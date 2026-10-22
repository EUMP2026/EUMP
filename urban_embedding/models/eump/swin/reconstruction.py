from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from urban_embedding.data.processed.pipeline_spec import ProcessedBatchSchema, is_hurdle_loss

from urban_embedding.models.backbones.swinv2_masked import SwinWithConv


REGRESSION_LOSSES = {"mse", "mae", "huber"}
CLASSIFICATION_LOSSES = {"bce", "ce", "kl"}
SCALAR_HEAD_MODES = {"per_target", "merged"}


@dataclass(frozen=True, slots=True)
class EumpReconstructionOutputRouting:
    raster_output_channels: int
    raster_regression_indices: tuple[int, ...]
    raster_classification_indices: tuple[int, ...]
    scalar_regression_dims: dict[str, int]
    scalar_classification_dims: dict[str, int]

    @property
    def raster_regression_channels(self) -> int:
        return len(self.raster_regression_indices)

    @property
    def raster_classification_channels(self) -> int:
        return len(self.raster_classification_indices)

    @property
    def scalar_regression_channels(self) -> int:
        return sum(self.scalar_regression_dims.values())

    @property
    def scalar_classification_channels(self) -> int:
        return sum(self.scalar_classification_dims.values())


def _head_kind(loss: str) -> str:
    loss = str(loss).lower()
    if loss in REGRESSION_LOSSES:
        return "regression"
    if loss in CLASSIFICATION_LOSSES:
        return "classification"
    raise ValueError(
        f"Unsupported reconstruction output loss {loss!r}; "
        f"expected one of {sorted(REGRESSION_LOSSES | CLASSIFICATION_LOSSES)}"
    )


def _validate_classification_channels(*, name: str, loss: str, channels: int) -> None:
    loss = str(loss).lower()
    if loss in {"ce", "kl"} and channels < 2:
        raise ValueError(f"{loss.upper()} output {name!r} requires at least 2 channels, got {channels}")


def _append_raster_route(
    *,
    loss: str,
    name: str,
    final_indices: tuple[int, ...],
    regression_indices: list[int],
    classification_indices: list[int],
) -> None:
    kind = _head_kind(loss)
    if kind == "classification":
        _validate_classification_channels(name=name, loss=loss, channels=len(final_indices))
        classification_indices.extend(final_indices)
    else:
        regression_indices.extend(final_indices)


def build_eump_reconstruction_output_routing(
    batch_schema: ProcessedBatchSchema,
) -> EumpReconstructionOutputRouting:
    raster_regression_indices: list[int] = []
    raster_classification_indices: list[int] = []
    offset = 0

    for target in batch_schema.raster_targets:
        if target.loss_slices:
            for loss_slice in target.loss_slices:
                prediction_indices = loss_slice.prediction_channel_indices or loss_slice.channel_indices
                if is_hurdle_loss(loss_slice.loss):
                    raster_regression_indices.extend(offset + idx for idx in prediction_indices)
                    raster_classification_indices.extend(offset + idx for idx in loss_slice.hurdle_logit_channel_indices)
                else:
                    final_indices = tuple(offset + idx for idx in prediction_indices)
                    _append_raster_route(
                        loss=loss_slice.loss,
                        name=f"{target.name}/{loss_slice.name}",
                        final_indices=final_indices,
                        regression_indices=raster_regression_indices,
                        classification_indices=raster_classification_indices,
                    )
        elif is_hurdle_loss(target.spec.loss):
            value_indices = tuple(range(offset, offset + target.channels))
            logit_indices = tuple(range(offset + target.channels, offset + target.prediction_channels))
            raster_regression_indices.extend(value_indices)
            raster_classification_indices.extend(logit_indices)
        else:
            final_indices = tuple(range(offset, offset + target.prediction_channels))
            _append_raster_route(
                loss=target.spec.loss,
                name=target.name,
                final_indices=final_indices,
                regression_indices=raster_regression_indices,
                classification_indices=raster_classification_indices,
            )
        offset += target.prediction_channels

    scalar_regression_dims: dict[str, int] = {}
    scalar_classification_dims: dict[str, int] = {}
    for target in batch_schema.scalar_targets:
        kind = _head_kind(target.spec.loss)
        if kind == "classification":
            _validate_classification_channels(name=target.name, loss=target.spec.loss, channels=target.channels)
            scalar_classification_dims[target.name] = target.channels
        else:
            scalar_regression_dims[target.name] = target.channels

    return EumpReconstructionOutputRouting(
        raster_output_channels=batch_schema.raster_output_channels,
        raster_regression_indices=tuple(raster_regression_indices),
        raster_classification_indices=tuple(raster_classification_indices),
        scalar_regression_dims=scalar_regression_dims,
        scalar_classification_dims=scalar_classification_dims,
    )


class RasterPixelShuffleHead(nn.Module):
    def __init__(self, *, in_channels: int, out_channels: int, encoder_stride: int) -> None:
        super().__init__()
        if out_channels <= 0:
            raise ValueError(f"RasterPixelShuffleHead out_channels must be positive, got {out_channels}")
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=int(encoder_stride) ** 2 * int(out_channels),
                kernel_size=1,
            ),
            nn.PixelShuffle(int(encoder_stride)),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.net(embedding)


class RasterRegressionHead(RasterPixelShuffleHead):
    pass


class RasterClassificationHead(RasterPixelShuffleHead):
    pass


class ScalarPooledHead(nn.Module):
    def __init__(self, *, in_channels: int, out_dim: int) -> None:
        super().__init__()
        if out_dim <= 0:
            raise ValueError(f"ScalarPooledHead out_dim must be positive, got {out_dim}")
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels, out_dim),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.net(embedding)


class EumpSwinReconstructionModel(nn.Module):
    def __init__(
        self,
        model_path: str,
        input_channels: int,
        *,
        batch_schema: ProcessedBatchSchema,
        load_pretrained: bool = True,
        scalar_head_mode: str = "per_target",
    ) -> None:
        super().__init__()
        scalar_head_mode = str(scalar_head_mode).lower()
        if scalar_head_mode not in SCALAR_HEAD_MODES:
            raise ValueError(
                f"Unsupported reconstruct.scalar_head_mode={scalar_head_mode!r}; "
                f"expected one of {sorted(SCALAR_HEAD_MODES)}"
            )
        output_routing = build_eump_reconstruction_output_routing(batch_schema)

        self.scalar_head_mode = scalar_head_mode
        self.output_routing = output_routing
        self.raster_output_channels = output_routing.raster_output_channels
        self.raster_regression_channel_indices = output_routing.raster_regression_indices
        self.raster_classification_channel_indices = output_routing.raster_classification_indices
        self.backbone = SwinWithConv(
            model_path=model_path,
            input_channels=input_channels,
            output_channels=0,
            load_pretrained=load_pretrained,
        )
        self.config = self.backbone.config
        num_features = self.config.embed_dim * (2 ** (self.config.num_layers - 1))
        encoder_stride = int(self.config.encoder_stride)

        self.raster_regression_head = (
            RasterRegressionHead(
                in_channels=num_features,
                out_channels=output_routing.raster_regression_channels,
                encoder_stride=encoder_stride,
            )
            if output_routing.raster_regression_channels > 0
            else None
        )
        self.raster_classification_head = (
            RasterClassificationHead(
                in_channels=num_features,
                out_channels=output_routing.raster_classification_channels,
                encoder_stride=encoder_stride,
            )
            if output_routing.raster_classification_channels > 0
            else None
        )
        self.scalar_regression_heads = nn.ModuleDict()
        self.scalar_classification_heads = nn.ModuleDict()
        self.scalar_pool = None
        self.scalar_regression_head = None
        self.scalar_classification_head = None
        if scalar_head_mode == "merged":
            if output_routing.scalar_regression_channels > 0 or output_routing.scalar_classification_channels > 0:
                self.scalar_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
            self.scalar_regression_head = (
                nn.Linear(num_features, output_routing.scalar_regression_channels)
                if output_routing.scalar_regression_channels > 0
                else None
            )
            self.scalar_classification_head = (
                nn.Linear(num_features, output_routing.scalar_classification_channels)
                if output_routing.scalar_classification_channels > 0
                else None
            )
        else:
            self.scalar_regression_heads = nn.ModuleDict(
                {
                    name: ScalarPooledHead(in_channels=num_features, out_dim=output_dim)
                    for name, output_dim in output_routing.scalar_regression_dims.items()
                }
            )
            self.scalar_classification_heads = nn.ModuleDict(
                {
                    name: ScalarPooledHead(in_channels=num_features, out_dim=output_dim)
                    for name, output_dim in output_routing.scalar_classification_dims.items()
                }
            )
        self.register_buffer(
            "_raster_regression_indices",
            torch.tensor(output_routing.raster_regression_indices, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_raster_classification_indices",
            torch.tensor(output_routing.raster_classification_indices, dtype=torch.long),
            persistent=False,
        )

    def _stitch_raster_prediction(
        self,
        *,
        embedding: torch.Tensor,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        batch_size = int(embedding.shape[0])
        raster_prediction = embedding.new_zeros(
            (batch_size, self.raster_output_channels, int(output_size[0]), int(output_size[1]))
        )
        if self.raster_regression_head is not None:
            regression_prediction = self.raster_regression_head(embedding)
            raster_prediction = raster_prediction.index_copy(
                1,
                self._raster_regression_indices,
                regression_prediction,
            )
        if self.raster_classification_head is not None:
            classification_prediction = self.raster_classification_head(embedding)
            raster_prediction = raster_prediction.index_copy(
                1,
                self._raster_classification_indices,
                classification_prediction,
            )
        return raster_prediction

    def _split_scalar_prediction(
        self,
        prediction: torch.Tensor,
        dims: dict[str, int],
    ) -> dict[str, torch.Tensor]:
        predictions: dict[str, torch.Tensor] = {}
        start = 0
        for name, dim in dims.items():
            end = start + int(dim)
            predictions[name] = prediction[:, start:end]
            start = end
        if start != prediction.shape[1]:
            raise ValueError(
                f"Scalar prediction dims consumed {start}, but head returned {prediction.shape[1]}"
            )
        return predictions

    def _predict_scalars(self, embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        if getattr(self, "scalar_head_mode", "per_target") == "merged":
            scalar_predictions: dict[str, torch.Tensor] = {}
            scalar_pool = getattr(self, "scalar_pool", None)
            if scalar_pool is None:
                return scalar_predictions
            pooled = scalar_pool(embedding)
            scalar_regression_head = getattr(self, "scalar_regression_head", None)
            if scalar_regression_head is not None:
                scalar_predictions.update(
                    self._split_scalar_prediction(
                        scalar_regression_head(pooled),
                        self.output_routing.scalar_regression_dims,
                    )
                )
            scalar_classification_head = getattr(self, "scalar_classification_head", None)
            if scalar_classification_head is not None:
                scalar_predictions.update(
                    self._split_scalar_prediction(
                        scalar_classification_head(pooled),
                        self.output_routing.scalar_classification_dims,
                    )
                )
            return scalar_predictions

        scalar_predictions = {
            name: head(embedding)
            for name, head in self.scalar_regression_heads.items()
        }
        scalar_predictions.update(
            {
                name: head(embedding)
                for name, head in self.scalar_classification_heads.items()
            }
        )
        return scalar_predictions

    def forward(
        self,
        input_tensor: torch.Tensor,
        bool_masked_pos: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        embedding = self.backbone.encode(input_tensor, bool_masked_pos=bool_masked_pos)
        raster_prediction = self._stitch_raster_prediction(
            embedding=embedding,
            output_size=tuple(input_tensor.shape[-2:]),
        )
        return raster_prediction, self._predict_scalars(embedding)


__all__ = [
    "CLASSIFICATION_LOSSES",
    "REGRESSION_LOSSES",
    "SCALAR_HEAD_MODES",
    "EumpReconstructionOutputRouting",
    "EumpSwinReconstructionModel",
    "RasterClassificationHead",
    "RasterPixelShuffleHead",
    "RasterRegressionHead",
    "ScalarPooledHead",
    "SwinWithConv",
    "build_eump_reconstruction_output_routing",
]
