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
MATMUL_WMMA = CudaExtensionSpec(
    name="forgenpu_matmul_wmma",
    source_filename="matmul_wmma.cu",
    function_name="matmul_wmma",
    label="WMMA Tensor Core matmul",
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


def _unsupported_device_message(spec: CudaExtensionSpec, torch, device=None) -> str | None:
    if spec != MATMUL_WMMA:
        return None

    major, minor = torch.cuda.get_device_capability(device)
    if major < 7:
        return (
            f"{spec.label} requires NVIDIA WMMA support on compute capability 7.0 or newer; "
            f"visible CUDA device is sm_{major}{minor}."
        )
    return None


def _matmul_naive_source() -> Path:
    return _kernel_source(MATMUL_NAIVE)


def _matmul_tiled_source() -> Path:
    return _kernel_source(MATMUL_TILED)


def _matmul_wmma_source() -> Path:
    return _kernel_source(MATMUL_WMMA)


def _has_cuda_extension(spec: CudaExtensionSpec, *, device=None) -> bool:
    return cuda_extension_unavailable_reason(spec, device=device) is None


def cuda_extension_unavailable_reason(spec: CudaExtensionSpec, *, device=None) -> str | None:
    try:
        torch = _require_torch()
        from torch.utils.cpp_extension import CUDA_HOME
    except RuntimeError as exc:
        return str(exc)
    except ImportError as exc:
        return f"{spec.label} requires PyTorch CUDA extension support: {exc}"

    if not torch.cuda.is_available():
        return f"{spec.label} requires a CUDA-capable PyTorch environment."
    unsupported_device = _unsupported_device_message(spec, torch, device)
    if unsupported_device is not None:
        return unsupported_device
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


def has_cuda_matmul_naive(*, device=None) -> bool:
    """Return whether the current environment can build and run the M1 CUDA matmul."""
    return _has_cuda_extension(MATMUL_NAIVE, device=device)


def has_cuda_matmul_tiled(*, device=None) -> bool:
    """Return whether the current environment can build and run the M2 CUDA matmul."""
    return _has_cuda_extension(MATMUL_TILED, device=device)


def has_cuda_matmul_wmma(*, device=None) -> bool:
    """Return whether the current environment can build and run the M4 WMMA matmul."""
    return _has_cuda_extension(MATMUL_WMMA, device=device)


def cuda_matmul_naive_unavailable_reason(*, device=None) -> str | None:
    """Return why the M1 CUDA matmul is unavailable, or None when it is available."""
    return cuda_extension_unavailable_reason(MATMUL_NAIVE, device=device)


def cuda_matmul_tiled_unavailable_reason(*, device=None) -> str | None:
    """Return why the M2 CUDA matmul is unavailable, or None when it is available."""
    return cuda_extension_unavailable_reason(MATMUL_TILED, device=device)


def cuda_matmul_wmma_unavailable_reason(*, device=None) -> str | None:
    """Return why the M4 WMMA matmul is unavailable, or None when it is available."""
    return cuda_extension_unavailable_reason(MATMUL_WMMA, device=device)


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
    module = _load_cuda_extension(spec)
    if spec == MATMUL_WMMA:
        torch = _require_torch()
        if isinstance(a, torch.Tensor):
            unsupported_device = _unsupported_device_message(spec, torch, a.device)
            if unsupported_device is not None:
                raise RuntimeError(unsupported_device)
    return getattr(module, spec.function_name)(a, b)


def cuda_matmul_naive(a, b):
    """Run the M1 naive FP32 CUDA matmul kernel."""
    return _run_cuda_extension(MATMUL_NAIVE, a, b)


def cuda_matmul_tiled(a, b):
    """Run the M2 tiled shared-memory FP32 CUDA matmul kernel."""
    return _run_cuda_extension(MATMUL_TILED, a, b)


def cuda_matmul_wmma(a, b):
    """Run the M4 WMMA FP16-input FP32-output CUDA matmul kernel."""
    return _run_cuda_extension(MATMUL_WMMA, a, b)
