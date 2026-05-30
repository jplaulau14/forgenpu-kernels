"""Triton FP32 matmul implementation."""

from __future__ import annotations

from typing import Any

try:
    import triton
    import triton.language as tl
except ImportError as exc:
    triton = None
    tl = None
    _TRITON_IMPORT_ERROR = exc
else:
    _TRITON_IMPORT_ERROR = None

BLOCK_M = 16
BLOCK_N = 16
BLOCK_K = 32


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for Triton matmul. Install the correct PyTorch build first."
        ) from exc
    return torch


def _require_triton() -> tuple[Any, Any]:
    if triton is None or tl is None:
        raise RuntimeError(
            "Triton matmul requires the optional Triton dependency. "
            "Install it with `uv sync --extra triton --extra dev` on Linux."
        ) from _TRITON_IMPORT_ERROR
    return triton, tl


def triton_matmul_unavailable_reason() -> str | None:
    """Return why Triton matmul cannot run in this environment, or None."""
    try:
        torch = _require_torch()
    except RuntimeError as exc:
        return str(exc)

    try:
        _require_triton()
    except RuntimeError as exc:
        return str(exc)

    if not torch.cuda.is_available():
        return "PyTorch CUDA is not available"
    return None


def has_triton_matmul() -> bool:
    """Return whether this environment appears able to run the Triton matmul."""
    return triton_matmul_unavailable_reason() is None


def _validate_inputs(a, b) -> None:
    torch = _require_torch()
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise TypeError("triton_matmul expects torch.Tensor inputs")
    if a.ndim != 2 or b.ndim != 2:
        raise RuntimeError(f"triton_matmul expects 2D inputs; got {a.ndim}D and {b.ndim}D")
    if a.shape[1] != b.shape[0]:
        raise RuntimeError(f"shape mismatch: A is {tuple(a.shape)}, B is {tuple(b.shape)}")
    if a.dtype != torch.float32 or b.dtype != torch.float32:
        raise RuntimeError("triton_matmul only supports float32 inputs")
    if not a.is_cuda or not b.is_cuda:
        raise RuntimeError("triton_matmul requires CUDA tensors")
    if a.device != b.device:
        raise RuntimeError(
            f"input tensors must be on the same device; got {a.device} and {b.device}"
        )


def triton_matmul(a, b):
    """Run the M3 blocked FP32 Triton matmul kernel."""
    triton, _ = _require_triton()
    _validate_inputs(a, b)

    m = a.shape[0]
    k = a.shape[1]
    n = b.shape[1]
    c = a.new_empty((m, n))
    if m == 0 or n == 0:
        return c

    grid = (triton.cdiv(m, BLOCK_M), triton.cdiv(n, BLOCK_N))
    if _matmul_kernel is None:
        raise RuntimeError("Triton matmul kernel was not initialized")

    _matmul_kernel[grid](
        a,
        b,
        c,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return c


if triton is not None and tl is not None:

    @triton.jit
    def _matmul_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        m: tl.constexpr,
        n: tl.constexpr,
        k: tl.constexpr,
        stride_am: tl.constexpr,
        stride_ak: tl.constexpr,
        stride_bk: tl.constexpr,
        stride_bn: tl.constexpr,
        stride_cm: tl.constexpr,
        stride_cn: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        program_m = tl.program_id(0)
        program_n = tl.program_id(1)

        offsets_m = program_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offsets_n = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offsets_k = tl.arange(0, BLOCK_K)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

        for k_start in range(0, k, BLOCK_K):
            k_offsets = k_start + offsets_k
            a_offsets = offsets_m[:, None] * stride_am + k_offsets[None, :] * stride_ak
            b_offsets = k_offsets[:, None] * stride_bk + offsets_n[None, :] * stride_bn
            a_mask = (offsets_m[:, None] < m) & (k_offsets[None, :] < k)
            b_mask = (k_offsets[:, None] < k) & (offsets_n[None, :] < n)
            a_tile = tl.load(a_ptr + a_offsets, mask=a_mask, other=0.0)
            b_tile = tl.load(b_ptr + b_offsets, mask=b_mask, other=0.0)
            accumulator += tl.dot(a_tile, b_tile, input_precision="ieee")

        c_offsets = offsets_m[:, None] * stride_cm + offsets_n[None, :] * stride_cn
        c_mask = (offsets_m[:, None] < m) & (offsets_n[None, :] < n)
        tl.store(c_ptr + c_offsets, accumulator, mask=c_mask)
else:
    _matmul_kernel = None
