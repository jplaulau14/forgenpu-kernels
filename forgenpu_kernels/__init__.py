"""Public Python surface for ForgeNPU Kernels."""

from .bindings import (
    cuda_matmul_naive,
    cuda_matmul_tiled,
    has_cuda_matmul_naive,
    has_cuda_matmul_tiled,
)
from .ops import max_error, torch_matmul
from .triton import has_triton_matmul, triton_matmul, triton_matmul_unavailable_reason

__all__ = [
    "cuda_matmul_naive",
    "cuda_matmul_tiled",
    "has_cuda_matmul_naive",
    "has_cuda_matmul_tiled",
    "has_triton_matmul",
    "max_error",
    "torch_matmul",
    "triton_matmul",
    "triton_matmul_unavailable_reason",
]
