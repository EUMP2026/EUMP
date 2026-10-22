from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch


def set_reproducible(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Seed Python and NumPy RNGs inside a PyTorch DataLoader worker."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def seeded_torch_generator(seed: int, *, offset: int = 0) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed((int(seed) + int(offset)) % (2**63 - 1))
    return generator


def capture_rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": numpy_state[1].tolist(),
            "pos": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Any) -> None:
    if not isinstance(state, dict):
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        numpy_state = state["numpy"]
        if isinstance(numpy_state, dict):
            np.random.set_state(
                (
                    str(numpy_state["bit_generator"]),
                    np.asarray(numpy_state["state"], dtype=np.uint32),
                    int(numpy_state["pos"]),
                    int(numpy_state["has_gauss"]),
                    float(numpy_state["cached_gaussian"]),
                )
            )
        else:
            np.random.set_state(numpy_state)
    torch_state = state.get("torch")
    if torch_state is not None:
        torch.set_rng_state(torch_state.cpu() if torch.is_tensor(torch_state) else torch_state)
    cuda_states = state.get("torch_cuda")
    if cuda_states is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [item.cpu() if torch.is_tensor(item) else item for item in cuda_states]
        )


__all__ = [
    "capture_rng_state",
    "restore_rng_state",
    "seed_worker",
    "seeded_torch_generator",
    "set_reproducible",
]
