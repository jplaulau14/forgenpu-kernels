"""Python bridge for native CUDA kernels."""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
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


@dataclass(frozen=True)
class CudaExtensionSpec:
    name: str
    source_filename: str
    function_name: str
    label: str


MATMUL_NAIVE = CudaExtensionSpec(
    name="forgenpu_matmul_naive",
    source_filename="matmul_naive.cu",
    function_name="matmul_naive",
    label="Naive matmul",
)
MATMUL_TILED = CudaExtensionSpec(
    name="forgenpu_matmul_tiled",
    source_filename="matmul_tiled.cu",
    function_name="matmul_tiled",
    label="Tiled matmul",
)


def _kernel_source(spec: CudaExtensionSpec) -> Path:
    return Path(__file__).resolve().parents[1] / "kernels" / "cuda" / "matmul" / spec.source_filename


def _missing_source_message(kernel: str, source: Path) -> str:
    return (
        f"{kernel} CUDA source not found: {source}. "
        "The current extension loader expects a source-tree checkout with kernels/cuda present; "
        "use the repository directly or an editable install for CUDA benchmark runs."
    )


def _ninja_available() -> bool:
    return shutil.which("ninja") is not None


def _host_compiler_available() -> bool:
    if sys.platform == "win32":
        return shutil.which("cl") is not None
    return True


def _missing_ninja_message(kernel: str) -> str:
    return (
        f"{kernel} requires ninja to build PyTorch C++/CUDA extensions. "
        "Run `uv sync --extra dev` from the repository root and make sure the virtualenv "
        "bin directory is on PATH."
    )


def _matmul_naive_source() -> Path:
    return _kernel_source(MATMUL_NAIVE)


def _matmul_tiled_source() -> Path:
    return _kernel_source(MATMUL_TILED)


def _has_cuda_extension(spec: CudaExtensionSpec) -> bool:
    try:
        torch = _require_torch()
        from torch.utils.cpp_extension import CUDA_HOME
    except RuntimeError:
        return False
    except ImportError:
        return False

    return bool(
        torch.cuda.is_available()
        and CUDA_HOME
        and _kernel_source(spec).exists()
        and _ninja_available()
        and _host_compiler_available()
    )


def has_cuda_matmul_naive() -> bool:
    """Return whether the current environment can build and run the M1 CUDA matmul."""
    return _has_cuda_extension(MATMUL_NAIVE)


def has_cuda_matmul_tiled() -> bool:
    """Return whether the current environment can build and run the M2 CUDA matmul."""
    return _has_cuda_extension(MATMUL_TILED)


@lru_cache(maxsize=None)
def _load_cuda_extension(spec: CudaExtensionSpec) -> Any:
    torch = _require_torch()
    if not torch.cuda.is_available():
        raise RuntimeError(f"{spec.label} requires a CUDA-capable PyTorch environment.")

    from torch.utils.cpp_extension import CUDA_HOME, load

    if CUDA_HOME is None:
        raise RuntimeError(f"{spec.label} requires a CUDA toolkit with nvcc available.")

    if not _ninja_available():
        raise RuntimeError(_missing_ninja_message(spec.label))

    if not _host_compiler_available():
        raise RuntimeError(
            f"{spec.label} requires the MSVC C++ compiler `cl` on Windows. "
            "Install Visual Studio Build Tools with the C++ workload, then run from a "
            "Developer PowerShell or Developer Command Prompt."
        )

    source = _kernel_source(spec)
    if not source.exists():
        raise RuntimeError(_missing_source_message(spec.label, source))

    verbose = os.environ.get("FORGENPU_EXT_VERBOSE", "0") == "1"
    return load(
        name=spec.name,
        sources=[str(source)],
        extra_cflags=["-O2"],
        extra_cuda_cflags=["-O2"],
        verbose=verbose,
    )


def _run_cuda_extension(spec: CudaExtensionSpec, a, b):
    return getattr(_load_cuda_extension(spec), spec.function_name)(a, b)


def cuda_matmul_naive(a, b):
    """Run the M1 naive FP32 CUDA matmul kernel."""
    return _run_cuda_extension(MATMUL_NAIVE, a, b)


def cuda_matmul_tiled(a, b):
    """Run the M2 tiled shared-memory FP32 CUDA matmul kernel."""
    return _run_cuda_extension(MATMUL_TILED, a, b)
