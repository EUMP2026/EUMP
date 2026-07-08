from __future__ import annotations

import os
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def get_model(model):
    return model.module if isinstance(model, DDP) else model


def get_rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def resolve_device(local_rank: int) -> torch.device:
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def get_device(logger=None) -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if logger is not None:
            gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
            logger.info(f"GPU Name: {gpu_name}")
    else:
        device = torch.device("cpu")
    if logger is not None:
        logger.info(f"device: {device}")
    return device


def init_distributed_process_group() -> str:
    if dist.is_available() and dist.is_initialized():
        return dist.get_backend()

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    timeout_seconds = os.environ.get("EUMP_DDP_TIMEOUT_SECONDS")
    timeout_kwargs = {}
    if timeout_seconds:
        timeout_kwargs["timeout"] = timedelta(seconds=int(timeout_seconds))
    if backend == "nccl":
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        try:
            dist.init_process_group(
                backend=backend,
                init_method="env://",
                device_id=torch.device("cuda", local_rank),
                **timeout_kwargs,
            )
        except TypeError:
            dist.init_process_group(backend=backend, init_method="env://", **timeout_kwargs)
    else:
        dist.init_process_group(backend=backend, init_method="env://", **timeout_kwargs)
    return backend


def destroy_distributed_process_group() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def distributed_barrier(local_rank: int | None = None) -> None:
    if not dist.is_available() or not dist.is_initialized() or int(dist.get_world_size()) <= 1:
        return
    backend = str(dist.get_backend()).lower()
    if backend == "nccl" and torch.cuda.is_available():
        if local_rank is None:
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        dist.barrier(device_ids=[int(local_rank)])
        return
    dist.barrier()


def wrap_ddp(
    model,
    device: torch.device,
    local_rank: int,
    *,
    find_unused_parameters: bool = False,
):
    if device.type == "cuda":
        return DDP(
            model,
            device_ids=[local_rank],
            find_unused_parameters=find_unused_parameters,
        )
    return DDP(model, find_unused_parameters=find_unused_parameters)


__all__ = [
    "destroy_distributed_process_group",
    "distributed_barrier",
    "get_device",
    "get_model",
    "get_rank",
    "get_world_size",
    "init_distributed_process_group",
    "resolve_device",
    "wrap_ddp",
]
