"""Reference operators and correctness helpers."""

from __future__ import annotations

from dataclasses import dataclass


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for ForgeNPU reference operators. "
            "Install the correct PyTorch build for your CPU/CUDA environment."
        ) from exc
    return torch


@dataclass(frozen=True)
class ErrorStats:
    max_abs_error: float
    max_rel_error: float


def torch_matmul(a, b):
    """Trusted PyTorch matmul reference."""
    torch = _require_torch()
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise TypeError("torch_matmul expects torch.Tensor inputs")
    return torch.matmul(a, b)


def max_error(actual, expected, *, eps: float = 1e-12) -> ErrorStats:
    """Return max absolute and relative error between two tensors."""
    torch = _require_torch()
    if actual.shape != expected.shape:
        raise ValueError(
            f"shape mismatch: actual={tuple(actual.shape)} expected={tuple(expected.shape)}"
        )

    diff = (actual - expected).abs().to(torch.float32)
    denom = torch.maximum(
        expected.abs().to(torch.float32),
        torch.tensor(eps, device=expected.device, dtype=torch.float32),
    )
    rel = diff / denom
    return ErrorStats(max_abs_error=float(diff.max().item()), max_rel_error=float(rel.max().item()))
