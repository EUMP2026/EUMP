from __future__ import annotations

import math

import torch
import torch.nn as nn
from transformers import Swinv2Config, Swinv2ForMaskedImageModeling


class SwinWithConv(nn.Module):
    def __init__(
        self,
        model_path: str,
        input_channels: int,
        output_channels: int,
        load_pretrained: bool = True,
        backbone_hidden_dropout_prob: float | None = None,
        backbone_attention_probs_dropout_prob: float | None = None,
    ) -> None:
        super().__init__()
        config = Swinv2Config.from_pretrained(model_path)
        _apply_optional_dropout(
            config,
            "hidden_dropout_prob",
            backbone_hidden_dropout_prob,
            "model.backbone_hidden_dropout_prob",
        )
        _apply_optional_dropout(
            config,
            "attention_probs_dropout_prob",
            backbone_attention_probs_dropout_prob,
            "model.backbone_attention_probs_dropout_prob",
        )
        if load_pretrained:
            self.swin = Swinv2ForMaskedImageModeling.from_pretrained(model_path, config=config)
        else:
            self.swin = Swinv2ForMaskedImageModeling(config)
        self.config = self.swin.config
        if output_channels < 0:
            raise ValueError(f"output_channels must be non-negative, got {output_channels}")
        self.config.channels = output_channels
        self.conv_layer = nn.Conv2d(
            in_channels=input_channels,
            out_channels=3,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        num_features = self.config.embed_dim * (2 ** (self.config.num_layers - 1))
        encoder_stride = self.config.encoder_stride

        if output_channels > 0:
            self.swin.decoder = nn.Sequential(
                nn.Conv2d(
                    in_channels=num_features,
                    out_channels=encoder_stride ** 2 * output_channels,
                    kernel_size=1,
                ),
                nn.PixelShuffle(encoder_stride),
            )
        else:
            self.swin.decoder = None

    def encode(self, input_tensor: torch.Tensor, bool_masked_pos: torch.Tensor | None = None) -> torch.Tensor:
        x = self.conv_layer(input_tensor) if self.conv_layer is not None else input_tensor
        outputs = self.swin.swinv2(x, bool_masked_pos=bool_masked_pos)

        sequence_output = outputs[0].transpose(1, 2).contiguous()
        batch_size, num_channels, sequence_length = sequence_output.shape
        height = width = math.floor(sequence_length ** 0.5)
        sequence_output = sequence_output.reshape(batch_size, num_channels, height, width).contiguous()
        return sequence_output

    def forward(
        self,
        input_tensor: torch.Tensor,
        bool_masked_pos: torch.Tensor | None,
        criterion,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.swin.decoder is None:
            raise RuntimeError("SwinWithConv.forward requires output_channels > 0")
        sequence_output = self.encode(input_tensor, bool_masked_pos)
        reconstructed_pixel_values = self.swin.decoder(sequence_output.contiguous())
        recon_loss = criterion(input_tensor, reconstructed_pixel_values)

        if bool_masked_pos is not None:
            size = self.config.image_size // self.config.patch_size
            bool_masked_pos = bool_masked_pos.reshape(-1, size, size)
            mask = (
                bool_masked_pos.repeat_interleave(self.config.patch_size, 1)
                .repeat_interleave(self.config.patch_size, 2)
                .unsqueeze(1)
                .contiguous()
            )
            channels = input_tensor.size(1)
            mask = mask.expand(-1, channels, -1, -1).contiguous()
            loss = recon_loss * mask
        else:
            loss = recon_loss
            mask = torch.ones_like(input_tensor)
        return loss, reconstructed_pixel_values, sequence_output, mask

    def embd_forward(self, embedding: torch.Tensor) -> torch.Tensor:
        if self.swin.decoder is None:
            raise RuntimeError("SwinWithConv.embd_forward requires output_channels > 0")
        return self.swin.decoder(embedding)


def _apply_optional_dropout(
    config: Swinv2Config,
    attribute: str,
    value: float | None,
    config_key: str,
) -> None:
    if value is None:
        return
    dropout = float(value)
    if dropout < 0.0 or dropout >= 1.0:
        raise ValueError(f"{config_key} must be in [0, 1), got {value}")
    setattr(config, attribute, dropout)


__all__ = ["SwinWithConv"]
