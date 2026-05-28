"""Matmul benchmark orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Literal

from forgenpu_kernels.benchmarks import benchmark_torch_callable, machine_info_dict
from forgenpu_kernels.bindings import (
    cuda_matmul_naive,
    cuda_matmul_tiled,
    has_cuda_matmul_naive,
    has_cuda_matmul_tiled,
)
from forgenpu_kernels.ops import max_error, torch_matmul

MatmulImplementation = Literal["torch", "cuda_naive", "cuda_tiled"]
MatmulImplementationSelection = Literal["torch", "cuda_naive", "cuda_tiled", "all"]

MATMUL_IMPLEMENTATIONS: tuple[MatmulImplementation, ...] = (
    "torch",
    "cuda_naive",
    "cuda_tiled",
)
TILE_SIZE = 16


@dataclass(frozen=True)
class MatmulBenchmarkConfig:
    shape: tuple[int, int, int]
    warmup: int
    iterations: int
    device: str
    dtype: str
    implementation: MatmulImplementationSelection


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for this benchmark. Install the correct PyTorch build first."
        ) from exc
    return torch


def resolve_device(requested: str) -> str:
    torch = require_torch()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def resolve_dtype(torch, dtype_name: str):
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_name]


def validate_dtype_for_device(*, device: str, dtype, dtype_name: str) -> None:
    torch = require_torch()
    if device == "cpu" and dtype in {torch.float16, torch.bfloat16}:
        raise RuntimeError(f"{dtype_name} CPU timing is not used for the matmul benchmark harness")


def make_inputs(torch, *, m: int, n: int, k: int, device: str, dtype):
    generator = torch.Generator(device=device)
    generator.manual_seed(0)
    a = torch.randn((m, k), device=device, dtype=dtype, generator=generator)
    b = torch.randn((k, n), device=device, dtype=dtype, generator=generator)
    return a, b


def matmul_workload_metrics(*, implementation: str, shape: tuple[int, int, int], p50_ms: float) -> dict:
    m, n, k = shape
    flops = 2 * m * n * k
    compulsory_io_bytes = 4 * ((m * k) + (k * n) + (m * n))
    tile_m = ceil_div(m, TILE_SIZE)
    tile_n = ceil_div(n, TILE_SIZE)

    if implementation == "cuda_naive":
        estimated_global_memory_bytes = 4 * ((2 * m * n * k) + (m * n))
    elif implementation == "cuda_tiled":
        estimated_global_memory_bytes = 4 * (((tile_n * m * k) + (tile_m * k * n)) + (m * n))
    else:
        estimated_global_memory_bytes = None

    seconds = p50_ms / 1_000
    achieved_tflops = (flops / seconds) / 1_000_000_000_000 if seconds > 0 else None
    arithmetic_intensity = flops / compulsory_io_bytes if compulsory_io_bytes > 0 else None

    return {
        "estimated_flops": flops,
        "compulsory_io_bytes": compulsory_io_bytes,
        "estimated_global_memory_bytes": estimated_global_memory_bytes,
        "arithmetic_intensity_flop_per_byte": arithmetic_intensity,
        "achieved_tflops": achieved_tflops,
    }


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def selected_implementations(
    selection: MatmulImplementationSelection,
) -> tuple[MatmulImplementation, ...]:
    if selection == "all":
        return MATMUL_IMPLEMENTATIONS
    return (selection,)


def validate_implementation(*, implementation: MatmulImplementation, dtype_name: str, device: str) -> None:
    if implementation == "torch":
        return

    if dtype_name != "float32":
        raise RuntimeError(f"{implementation} only supports float32")
    if device != "cuda":
        raise RuntimeError(f"{implementation} requires --device cuda or CUDA-capable --device auto")

    if implementation == "cuda_naive" and not has_cuda_matmul_naive():
        raise RuntimeError("cuda_naive requires CUDA, PyTorch CUDA, and nvcc")
    if implementation == "cuda_tiled" and not has_cuda_matmul_tiled():
        raise RuntimeError("cuda_tiled requires CUDA, PyTorch CUDA, and nvcc")


def run_implementation(implementation: MatmulImplementation, a, b):
    if implementation == "torch":
        return torch_matmul(a, b)
    if implementation == "cuda_naive":
        return cuda_matmul_naive(a, b)
    if implementation == "cuda_tiled":
        return cuda_matmul_tiled(a, b)
    raise ValueError(f"unknown implementation: {implementation}")


def benchmark_one(
    *,
    implementation: MatmulImplementation,
    a,
    b,
    expected,
    dtype_name: str,
    shape: tuple[int, int, int],
    warmup: int,
    iterations: int,
    device: str,
) -> dict[str, Any]:
    validate_implementation(implementation=implementation, dtype_name=dtype_name, device=device)

    benchmark_target = partial(run_implementation, implementation, a, b)
    actual = benchmark_target()
    errors = max_error(actual, expected)
    timing = benchmark_torch_callable(
        benchmark_target,
        warmup=warmup,
        iterations=iterations,
        device=device,
    )
    machine = machine_info_dict(device=device)
    m, n, k = shape

    result = {
        "operator": "matmul",
        "implementation": implementation,
        "dtype": dtype_name,
        "shape": {"m": m, "n": n, "k": k},
        "warmup": warmup,
        "iterations": iterations,
        "baseline": "torch.matmul",
        "max_abs_error": errors.max_abs_error,
        "max_rel_error": errors.max_rel_error,
        **timing,
        "device": machine["device"],
        "machine": machine,
    }
    result.update(
        matmul_workload_metrics(
            implementation=implementation,
            shape=shape,
            p50_ms=result["p50_ms"],
        )
    )
    return result


def add_speedups(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    torch_p50 = p50_for_implementation(results, "torch")
    naive_p50 = p50_for_implementation(results, "cuda_naive")

    for result in results:
        result["speedup_vs_baseline"] = speedup(torch_p50, result["p50_ms"])
        result["speedup_vs_naive"] = speedup_vs_naive(result, naive_p50)
    return results


def p50_for_implementation(results: list[dict[str, Any]], implementation: str) -> float | None:
    return next(
        (result["p50_ms"] for result in results if result["implementation"] == implementation),
        None,
    )


def speedup(baseline_p50_ms: float | None, p50_ms: float) -> float | None:
    if baseline_p50_ms is None:
        return None
    return baseline_p50_ms / p50_ms


def speedup_vs_naive(result: dict[str, Any], naive_p50_ms: float | None) -> float | None:
    if result["implementation"] == "cuda_naive":
        return 1.0
    return speedup(naive_p50_ms, result["p50_ms"])


def result_payload(results: list[dict[str, Any]]) -> dict[str, Any] | list[dict[str, Any]]:
    return results[0] if len(results) == 1 else results


def run_matmul_benchmark(
    config: MatmulBenchmarkConfig,
    *,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    torch = require_torch()
    device = resolve_device(config.device)
    dtype = resolve_dtype(torch, config.dtype)
    validate_dtype_for_device(device=device, dtype=dtype, dtype_name=config.dtype)

    m, n, k = config.shape
    emit(progress, f"device={device} dtype={config.dtype} shape={m}x{n}x{k} implementation={config.implementation}")
    emit(progress, "creating deterministic inputs")
    a, b = make_inputs(torch, m=m, n=n, k=k, device=device, dtype=dtype)
    emit(progress, "computing torch.matmul correctness oracle")
    expected = torch.matmul(a, b)

    results: list[dict[str, Any]] = []
    for implementation in selected_implementations(config.implementation):
        emit(progress, f"running {implementation} warmup={config.warmup} iterations={config.iterations}")
        results.append(
            benchmark_one(
                implementation=implementation,
                a=a,
                b=b,
                expected=expected,
                dtype_name=config.dtype,
                shape=config.shape,
                warmup=config.warmup,
                iterations=config.iterations,
                device=device,
            )
        )

    return add_speedups(results)


def emit(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
