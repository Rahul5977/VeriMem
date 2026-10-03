"""Device selection: cuda -> mps -> cpu. Never hard-code a device."""

from __future__ import annotations

import functools


@functools.lru_cache(maxsize=1)
def get_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@functools.lru_cache(maxsize=1)
def supports_bf16() -> bool:
    """Ada GPUs (4070/4050) have bf16; a Kaggle T4 does not -> callers fall back to fp16."""
    if get_device() != "cuda":
        return False
    import torch

    return torch.cuda.is_bf16_supported()


def dtype_for_training():
    """bf16 where available, else fp16 on CUDA, else fp32."""
    import torch

    if supports_bf16():
        return torch.bfloat16
    if get_device() == "cuda":
        return torch.float16
    return torch.float32
