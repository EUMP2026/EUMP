from __future__ import annotations

import torch


def generate_mask(
    batch_size: int,
    num_patches: int,
    num_masked_patches: int,
    device: str | torch.device = "cpu",
):
    if num_masked_patches == 0:
        return None
    num_masked_patches = min(num_masked_patches, num_patches)

    mask = torch.zeros((batch_size, num_patches), device=device, dtype=torch.bool)
    for batch_idx in range(batch_size):
        indices = torch.randperm(num_patches, device=device)
        mask[batch_idx, indices[:num_masked_patches]] = True
    return mask


__all__ = ["generate_mask"]
