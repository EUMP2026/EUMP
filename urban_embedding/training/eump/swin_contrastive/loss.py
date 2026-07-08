from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F


def _all_gather_with_grad(tensor: torch.Tensor) -> torch.Tensor:
    """All-gather across DDP ranks preserving grad for the local slice."""
    if not dist.is_available() or not dist.is_initialized():
        return tensor
    world_size = dist.get_world_size()
    if world_size <= 1:
        return tensor
    tensor = tensor.contiguous()
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    gathered[dist.get_rank()] = tensor
    return torch.cat(gathered, dim=0)


def nt_xent_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    *,
    temperature: float = 0.1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """SimCLR NT-Xent loss with DDP all-gather for negatives.

    Inputs are L2-normalized projections of two stochastic views of the same
    crop. Negatives are drawn from all other samples in the global batch
    (gathered across ranks). Loss is computed only on the local rank's anchors
    to avoid gradient duplication.
    """
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if z_a.shape != z_b.shape or z_a.dim() != 2:
        raise ValueError(f"Expected matching 2D tensors; got {z_a.shape} and {z_b.shape}")

    local_batch = z_a.shape[0]
    z_a_full = _all_gather_with_grad(z_a)
    z_b_full = _all_gather_with_grad(z_b)
    global_batch = z_a_full.shape[0]
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    local_offset = rank * local_batch

    all_views = torch.cat([z_a_full, z_b_full], dim=0)
    anchors = torch.cat([z_a, z_b], dim=0)
    logits = anchors @ all_views.t() / temperature

    rows = torch.arange(2 * local_batch, device=z_a.device)
    self_cols = torch.cat([
        torch.arange(local_offset, local_offset + local_batch, device=z_a.device),
        torch.arange(
            global_batch + local_offset,
            global_batch + local_offset + local_batch,
            device=z_a.device,
        ),
    ])
    logits[rows, self_cols] = float("-inf")

    targets = torch.cat([
        torch.arange(
            global_batch + local_offset,
            global_batch + local_offset + local_batch,
            device=z_a.device,
        ),
        torch.arange(local_offset, local_offset + local_batch, device=z_a.device),
    ])
    loss = F.cross_entropy(logits, targets)

    with torch.no_grad():
        top1 = (logits.argmax(dim=1) == targets).float().mean()
        pos_sim = F.cosine_similarity(anchors, all_views[targets], dim=1).mean()
    stats = {
        "contrastive/loss": loss.detach(),
        "contrastive/top1": top1,
        "contrastive/pos_sim": pos_sim,
        "contrastive/temperature": loss.new_tensor(float(temperature)),
    }
    return loss, stats


__all__ = ["nt_xent_loss"]
