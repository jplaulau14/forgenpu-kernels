"""Python bridge for native CUDA kernels."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for native ForgeNPU bindings. "
            "Run `uv sync --extra dev` before using CUDA kernels."
        ) from exc
    return torch


def _matmul_naive_source() -> Path:
    return Path(__file__).resolve().parents[1] / "kernels" / "cuda" / "matmul" / "matmul_naive.cu"


def has_cuda_matmul_naive() -> bool:
    """Return whether the current environment can build and run the M1 CUDA matmul."""
    try:
        torch = _require_torch()
        from torch.utils.cpp_extension import CUDA_HOME
    except RuntimeError:
        return False
    except ImportError:
        return False

    return bool(torch.cuda.is_available() and CUDA_HOME and _matmul_naive_source().exists())


@lru_cache(maxsize=1)
def _load_matmul_extension() -> Any:
    torch = _require_torch()
    if not torch.cuda.is_available():
        raise RuntimeError("Naive CUDA matmul requires a CUDA-capable PyTorch environment.")

    from torch.utils.cpp_extension import CUDA_HOME, load

    if CUDA_HOME is None:
        raise RuntimeError("Naive CUDA matmul requires a CUDA toolkit with nvcc available.")

    source = _matmul_naive_source()
    if not source.exists():
        raise RuntimeError(f"Naive CUDA matmul source not found: {source}")

    verbose = os.environ.get("FORGENPU_EXT_VERBOSE", "0") == "1"
    return load(
        name="forgenpu_matmul_naive",
        sources=[str(source)],
        extra_cflags=["-O2"],
        extra_cuda_cflags=["-O2"],
        verbose=verbose,
    )


def cuda_matmul_naive(a, b):
    """Run the M1 naive FP32 CUDA matmul kernel."""
    return _load_matmul_extension().matmul_naive(a, b)
