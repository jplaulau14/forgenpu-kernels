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
    source_tree_path = (
        Path(__file__).resolve().parents[1] / "kernels" / "cuda" / "matmul" / spec.source_filename
    )
    if source_tree_path.exists():
        return source_tree_path
    return Path(__file__).resolve().parent / "cuda" / "matmul" / spec.source_filename


def _missing_source_message(kernel: str, source: Path) -> str:
    return (
        f"{kernel} CUDA source not found: {source}. "
        "Expected CUDA sources either in the repository kernels/cuda tree or in the installed "
        "forgenpu_kernels package data."
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
    return cuda_extension_unavailable_reason(spec) is None


def cuda_extension_unavailable_reason(spec: CudaExtensionSpec) -> str | None:
    try:
        torch = _require_torch()
        from torch.utils.cpp_extension import CUDA_HOME
    except RuntimeError as exc:
        return str(exc)
    except ImportError as exc:
        return f"{spec.label} requires PyTorch CUDA extension support: {exc}"

    if not torch.cuda.is_available():
        return f"{spec.label} requires a CUDA-capable PyTorch environment."
    if CUDA_HOME is None:
        return f"{spec.label} requires a CUDA toolkit with nvcc available."
    source = _kernel_source(spec)
    if not source.exists():
        return _missing_source_message(spec.label, source)
    if not _ninja_available():
        return _missing_ninja_message(spec.label)
    if not _host_compiler_available():
        return (
            f"{spec.label} requires the MSVC C++ compiler `cl` on Windows. "
            "Install Visual Studio Build Tools with the C++ workload, then run from a "
            "Developer PowerShell or Developer Command Prompt."
        )
    return None


def has_cuda_matmul_naive() -> bool:
    """Return whether the current environment can build and run the M1 CUDA matmul."""
    return _has_cuda_extension(MATMUL_NAIVE)


def has_cuda_matmul_tiled() -> bool:
    """Return whether the current environment can build and run the M2 CUDA matmul."""
    return _has_cuda_extension(MATMUL_TILED)


def cuda_matmul_naive_unavailable_reason() -> str | None:
    """Return why the M1 CUDA matmul is unavailable, or None when it is available."""
    return cuda_extension_unavailable_reason(MATMUL_NAIVE)


def cuda_matmul_tiled_unavailable_reason() -> str | None:
    """Return why the M2 CUDA matmul is unavailable, or None when it is available."""
    return cuda_extension_unavailable_reason(MATMUL_TILED)


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
