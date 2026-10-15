"""Explicit per-process execution settings for isolated benchmark workers.

Import this module before numerical libraries in a worker to set native thread
limits early. Ordinary callers with cpu_threads=None retain their existing limits.
CUDA_VISIBLE_DEVICES is owned by the scheduler; CUDA always uses visible device 0.
"""
from __future__ import annotations

from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from torch import nn

_threadpool_limiter = None


def validate_execution_options(execution_device: str, cpu_threads: int | None) -> None:
    if execution_device not in {"cpu", "cuda"}:
        raise ValueError("execution_device must be cpu or cuda")
    if cpu_threads is not None and (type(cpu_threads) is not int or cpu_threads < 1):
        raise ValueError("cpu_threads must be a positive integer or None")


def initialize_execution_runtime(cpu_threads: int | None = None) -> None:
    """Apply an explicit worker CPU allocation; None has no process-wide effect.

    The scheduler must use an isolated process for an explicit allocation. This
    also limits libraries already loaded before this call, via threadpoolctl.
    """
    validate_execution_options("cpu", cpu_threads)
    if cpu_threads is None:
        return
    import os

    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(cpu_threads)
    import torch
    from threadpoolctl import threadpool_limits

    torch.set_num_threads(cpu_threads)
    if torch.get_num_interop_threads() != cpu_threads:
        torch.set_num_interop_threads(cpu_threads)
    global _threadpool_limiter
    _threadpool_limiter = threadpool_limits(limits=cpu_threads)


def resolve_execution_device(execution_device: str = "cpu") -> torch.device:
    """Resolve the requested device without silently changing CUDA to CPU."""
    validate_execution_options(execution_device, None)
    import torch

    if execution_device == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available to this worker")
    # Both matrix multiplication and cuDNN convolutions use FP32, including on
    # Ampere hardware where cuDNN otherwise permits TF32 by default.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch.device("cuda:0")


def model_device(model: nn.Module) -> torch.device:
    """Read the device of model parameters or buffers (including PCA buffers)."""
    import torch

    tensor = next(chain(model.parameters(), model.buffers()), None)
    return tensor.device if tensor is not None else torch.device("cpu")


def synchronize_device(device: str | torch.device) -> None:
    """Finish this process's CUDA work before entering/leaving a timing span."""
    import torch

    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_deep_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    execution_device: str = "cpu",
) -> nn.Module:
    """Restore FP32 weights and registered buffers on the requested device."""
    import torch

    device = resolve_execution_device(execution_device)
    model.to(device=device, dtype=torch.float32)
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model
