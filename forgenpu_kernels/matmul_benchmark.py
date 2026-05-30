"""Matmul benchmark orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Literal, TypedDict

from forgenpu_kernels.benchmarks import benchmark_torch_callable, machine_info_dict
from forgenpu_kernels.bindings import (
    cuda_matmul_naive,
    cuda_matmul_naive_unavailable_reason,
    cuda_matmul_tiled,
    cuda_matmul_tiled_unavailable_reason,
    cuda_matmul_wmma,
    cuda_matmul_wmma_unavailable_reason,
    has_cuda_matmul_naive,
    has_cuda_matmul_tiled,
    has_cuda_matmul_wmma,
)
from forgenpu_kernels.ops import max_error, torch_matmul
from forgenpu_kernels.triton import (
    has_triton_matmul,
    triton_matmul,
    triton_matmul_unavailable_reason,
)

MatmulImplementation = Literal["torch", "cuda_naive", "cuda_tiled", "cuda_wmma", "triton"]
MatmulImplementationSelection = Literal[
    "torch", "cuda_naive", "cuda_tiled", "cuda_wmma", "triton", "all"
]

MATMUL_CUDA_IMPLEMENTATIONS: tuple[MatmulImplementation, ...] = (
    "cuda_naive",
    "cuda_tiled",
)
MATMUL_CUDA_TENSOR_CORE_IMPLEMENTATIONS: tuple[MatmulImplementation, ...] = ("cuda_wmma",)
MATMUL_TRITON_IMPLEMENTATIONS: tuple[MatmulImplementation, ...] = ("triton",)
TILE_SIZE = 16


@dataclass(frozen=True)
class MatmulBenchmarkConfig:
    shape: tuple[int, int, int]
    warmup: int
    iterations: int
    device: str
    dtype: str
    implementation: MatmulImplementationSelection


class ShapeRecord(TypedDict):
    m: int
    n: int
    k: int


class MatmulWorkloadMetrics(TypedDict):
    estimated_flops: int
    compulsory_io_bytes: int
    estimated_global_memory_bytes: int | None
    arithmetic_intensity_flop_per_byte: float | None
    achieved_tflops: float | None


class MatmulBenchmarkResult(MatmulWorkloadMetrics):
    operator: str
    implementation: MatmulImplementation
    dtype: str
    input_dtype: str
    accumulation_dtype: str
    output_dtype: str
    shape: ShapeRecord
    warmup: int
    iterations: int
    baseline: str
    max_abs_error: float
    max_rel_error: float
    p50_ms: float
    p95_ms: float
    mean_ms: float
    device: str
    machine: dict[str, Any]
    speedup_vs_baseline: float | None
    speedup_vs_naive: float | None


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
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--device cuda was requested, but PyTorch CUDA is not available. "
                "Run `make env` to inspect the installed PyTorch build and visible GPU."
            )
        device = torch.device(requested)
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"--device {requested} was requested, but only "
                f"{torch.cuda.device_count()} CUDA device(s) are visible."
            )
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


def validate_config(config: MatmulBenchmarkConfig) -> None:
    m, n, k = config.shape
    if min(m, n, k) <= 0:
        raise RuntimeError(f"--shape values must be positive integers; got {m} {n} {k}")
    if config.warmup < 0:
        raise RuntimeError(f"--warmup must be non-negative; got {config.warmup}")
    if config.iterations <= 0:
        raise RuntimeError(f"--iterations must be positive; got {config.iterations}")


def make_inputs(torch, *, m: int, n: int, k: int, device: str, dtype):
    generator = torch.Generator(device=device)
    generator.manual_seed(0)
    a = torch.randn((m, k), device=device, dtype=dtype, generator=generator)
    b = torch.randn((k, n), device=device, dtype=dtype, generator=generator)
    return a, b


def torch_matmul_oracle(a, b, *, dtype_name: str):
    """Return the correctness oracle for the configured dtype."""
    if dtype_name in {"float16", "bfloat16"}:
        return torch_matmul(a.float(), b.float())
    return torch_matmul(a, b)


def torch_dtype_name(torch, dtype) -> str:
    if dtype == torch.float32:
        return "float32"
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.bfloat16:
        return "bfloat16"
    return str(dtype).replace("torch.", "")


def accumulation_dtype_name(implementation: MatmulImplementation, dtype_name: str) -> str:
    if implementation == "cuda_wmma":
        return "float32"
    if dtype_name == "float32":
        return "float32"
    return "implementation_defined"


def matmul_workload_metrics(
    *,
    implementation: str,
    shape: tuple[int, int, int],
    p50_ms: float,
    dtype_name: str = "float32",
    output_dtype_name: str | None = None,
) -> MatmulWorkloadMetrics:
    m, n, k = shape
    element_bytes = dtype_element_size(dtype_name)
    output_element_bytes = dtype_element_size(output_dtype_name or dtype_name)
    flops = 2 * m * n * k
    compulsory_io_bytes = element_bytes * ((m * k) + (k * n)) + output_element_bytes * (m * n)
    tile_m = ceil_div(m, TILE_SIZE)
    tile_n = ceil_div(n, TILE_SIZE)
    tile_k = ceil_div(k, TILE_SIZE)

    if implementation == "cuda_naive":
        estimated_global_memory_bytes = element_bytes * (2 * m * n * k) + output_element_bytes * (
            m * n
        )
    elif implementation in {"cuda_tiled", "triton"}:
        estimated_global_memory_bytes = element_bytes * (
            (tile_n * m * k) + (tile_m * k * n)
        ) + output_element_bytes * (m * n)
    elif implementation == "cuda_wmma":
        m_padded = tile_m * TILE_SIZE
        n_padded = tile_n * TILE_SIZE
        k_padded = tile_k * TILE_SIZE
        estimated_global_memory_bytes = element_bytes * (
            (tile_n * m_padded * k_padded) + (tile_m * k_padded * n_padded)
        ) + output_element_bytes * (m_padded * n_padded)
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


def dtype_element_size(dtype_name: str) -> int:
    sizes = {
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
    }
    return sizes[dtype_name]


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def selected_implementations(
    selection: MatmulImplementationSelection,
    *,
    device: str | None = None,
    dtype_name: str = "float32",
) -> tuple[MatmulImplementation, ...]:
    if selection == "all":
        implementations: list[MatmulImplementation] = ["torch"]
        if device is not None and not device.startswith("cuda"):
            return tuple(implementations)
        if dtype_name == "float32":
            for implementation in MATMUL_CUDA_IMPLEMENTATIONS:
                if implementation == "cuda_naive" and has_cuda_matmul_naive(device=device):
                    implementations.append(implementation)
                if implementation == "cuda_tiled" and has_cuda_matmul_tiled(device=device):
                    implementations.append(implementation)
            for implementation in MATMUL_TRITON_IMPLEMENTATIONS:
                if implementation == "triton" and has_triton_matmul():
                    implementations.append(implementation)
        elif dtype_name == "float16":
            for implementation in MATMUL_CUDA_TENSOR_CORE_IMPLEMENTATIONS:
                if implementation == "cuda_wmma" and has_cuda_matmul_wmma(device=device):
                    implementations.append(implementation)
        return tuple(implementations)
    return (selection,)


def validate_implementation(
    *, implementation: MatmulImplementation, dtype_name: str, device: str
) -> None:
    if implementation == "torch":
        return

    if not device.startswith("cuda"):
        raise RuntimeError(f"{implementation} requires --device cuda or CUDA-capable --device auto")

    if implementation in {"cuda_naive", "cuda_tiled", "triton"} and dtype_name != "float32":
        raise RuntimeError(f"{implementation} only supports float32")
    if implementation == "cuda_wmma" and dtype_name != "float16":
        raise RuntimeError("cuda_wmma only supports float16 inputs")

    if implementation == "cuda_naive" and not has_cuda_matmul_naive(device=device):
        reason = cuda_matmul_naive_unavailable_reason(device=device)
        raise RuntimeError(f"cuda_naive is unavailable: {reason}")
    if implementation == "cuda_tiled" and not has_cuda_matmul_tiled(device=device):
        reason = cuda_matmul_tiled_unavailable_reason(device=device)
        raise RuntimeError(f"cuda_tiled is unavailable: {reason}")
    if implementation == "cuda_wmma" and not has_cuda_matmul_wmma(device=device):
        reason = cuda_matmul_wmma_unavailable_reason(device=device)
        raise RuntimeError(f"cuda_wmma is unavailable: {reason}")
    if implementation == "triton" and not has_triton_matmul():
        reason = triton_matmul_unavailable_reason()
        raise RuntimeError(f"triton is unavailable: {reason}")


def run_implementation(implementation: MatmulImplementation, a, b):
    if implementation == "torch":
        return torch_matmul(a, b)
    if implementation == "cuda_naive":
        return cuda_matmul_naive(a, b)
    if implementation == "cuda_tiled":
        return cuda_matmul_tiled(a, b)
    if implementation == "cuda_wmma":
        return cuda_matmul_wmma(a, b)
    if implementation == "triton":
        return triton_matmul(a, b)
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
) -> MatmulBenchmarkResult:
    validate_implementation(implementation=implementation, dtype_name=dtype_name, device=device)

    benchmark_target = partial(run_implementation, implementation, a, b)
    actual = benchmark_target()
    errors = max_error(actual, expected)
    torch = require_torch()
    output_dtype_name = torch_dtype_name(torch, actual.dtype)
    timing = benchmark_torch_callable(
        benchmark_target,
        warmup=warmup,
        iterations=iterations,
        device=device,
    )
    machine = machine_info_dict(device=device)
    m, n, k = shape

    result: MatmulBenchmarkResult = {
        "operator": "matmul",
        "implementation": implementation,
        "dtype": dtype_name,
        "input_dtype": dtype_name,
        "accumulation_dtype": accumulation_dtype_name(implementation, dtype_name),
        "output_dtype": output_dtype_name,
        "shape": {"m": m, "n": n, "k": k},
        "warmup": warmup,
        "iterations": iterations,
        "baseline": "torch.matmul",
        "max_abs_error": errors.max_abs_error,
        "max_rel_error": errors.max_rel_error,
        **timing,
        "device": machine["device"],
        "machine": machine,
        "speedup_vs_baseline": None,
        "speedup_vs_naive": None,
        **matmul_workload_metrics(
            implementation=implementation,
            shape=shape,
            p50_ms=timing["p50_ms"],
            dtype_name=dtype_name,
            output_dtype_name=output_dtype_name,
        ),
    }
    return result


def add_speedups(results: list[MatmulBenchmarkResult]) -> list[MatmulBenchmarkResult]:
    torch_p50 = p50_for_implementation(results, "torch")
    naive_p50 = p50_for_implementation(results, "cuda_naive")

    for result in results:
        result["speedup_vs_baseline"] = speedup(torch_p50, result["p50_ms"])
        result["speedup_vs_naive"] = speedup_vs_naive(result, naive_p50)
    return results


def p50_for_implementation(
    results: list[MatmulBenchmarkResult], implementation: str
) -> float | None:
    return next(
        (result["p50_ms"] for result in results if result["implementation"] == implementation),
        None,
    )


def speedup(baseline_p50_ms: float | None, p50_ms: float) -> float | None:
    if baseline_p50_ms is None:
        return None
    return baseline_p50_ms / p50_ms


def speedup_vs_naive(result: MatmulBenchmarkResult, naive_p50_ms: float | None) -> float | None:
    if result["implementation"] == "cuda_naive":
        return 1.0
    return speedup(naive_p50_ms, result["p50_ms"])


def result_payload(
    results: list[MatmulBenchmarkResult],
) -> MatmulBenchmarkResult | list[MatmulBenchmarkResult]:
    return results[0] if len(results) == 1 else results


def run_matmul_benchmark(
    config: MatmulBenchmarkConfig,
    *,
    progress: Callable[[str], None] | None = None,
) -> list[MatmulBenchmarkResult]:
    validate_config(config)
    torch = require_torch()
    device = resolve_device(config.device)
    dtype = resolve_dtype(torch, config.dtype)
    validate_dtype_for_device(device=device, dtype=dtype, dtype_name=config.dtype)

    m, n, k = config.shape
    emit(
        progress,
        f"device={device} dtype={config.dtype} shape={m}x{n}x{k} implementation={config.implementation}",
    )
    emit(progress, "creating deterministic inputs")
    a, b = make_inputs(torch, m=m, n=n, k=k, device=device, dtype=dtype)
    emit(progress, "computing torch.matmul correctness oracle")
    expected = torch_matmul_oracle(a, b, dtype_name=config.dtype)

    results: list[MatmulBenchmarkResult] = []
    for implementation in selected_implementations(
        config.implementation, device=device, dtype_name=config.dtype
    ):
        emit(
            progress,
            f"running {implementation} warmup={config.warmup} iterations={config.iterations}",
        )
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
