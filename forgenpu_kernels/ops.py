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

    if actual.numel() == 0:
        return ErrorStats(max_abs_error=0.0, max_rel_error=0.0)

    comparison_dtype = error_comparison_dtype(torch, actual, expected)
    actual_for_error = actual.to(comparison_dtype)
    expected_for_error = expected.to(comparison_dtype)
    diff = (actual_for_error - expected_for_error).abs()
    denom = torch.maximum(
        expected_for_error.abs(),
        torch.tensor(eps, device=expected.device, dtype=diff.dtype),
    )
    rel = diff / denom
    return ErrorStats(max_abs_error=float(diff.max().item()), max_rel_error=float(rel.max().item()))


def error_comparison_dtype(torch, actual, expected):
    """Choose a dtype that avoids half-precision overflow/underflow in error math."""
    promoted = torch.promote_types(actual.dtype, expected.dtype)
    if promoted in {torch.float16, torch.bfloat16}:
        return torch.float32
    if torch.is_floating_point(actual) or torch.is_floating_point(expected):
        return promoted
    return torch.float64
