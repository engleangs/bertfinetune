import random

import numpy as np
import torch


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve a requested training device and validate explicit choices."""
    requested = device.lower()

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")

        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")

        return torch.device("cpu")

    resolved = torch.device(requested)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but this PyTorch installation has no CUDA support. "
            "Use '--device cpu' (or '--device auto'), or install a CUDA-enabled "
            "PyTorch build."
        )

    if resolved.type == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError(
                "MPS was requested, but it is not available. "
                "Use '--device cpu' or '--device auto'."
            )

    return resolved


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
