#!/usr/bin/env python
"""Benchmark PyTorch and ForgeNPU matmul implementations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forgenpu_kernels.benchmarks import benchmark_torch_callable, machine_info_dict, write_json_result  # noqa: E402
from forgenpu_kernels.bindings import cuda_matmul_naive, has_cuda_matmul_naive  # noqa: E402
from forgenpu_kernels.ops import max_error, torch_matmul  # noqa: E402


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is required for this benchmark. Install the correct PyTorch build first."
        ) from exc
    return torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark matmul implementations.")
    parser.add_argument("--shape", nargs=3, type=int, metavar=("M", "N", "K"), default=[512, 512, 512])
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument(
        "--implementation",
        default="torch",
        choices=["torch", "cuda_naive", "all"],
        help="Implementation to benchmark. M1 CUDA matmul supports float32 CUDA tensors only.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    torch = _require_torch()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def make_inputs(torch, *, m: int, n: int, k: int, device: str, dtype):
    generator = torch.Generator(device=device)
    generator.manual_seed(0)
    a = torch.randn((m, k), device=device, dtype=dtype, generator=generator)
    b = torch.randn((k, n), device=device, dtype=dtype, generator=generator)
    return a, b


def benchmark_one(
    *,
    implementation: str,
    a,
    b,
    expected,
    dtype_name: str,
    shape: tuple[int, int, int],
    warmup: int,
    iterations: int,
    device: str,
) -> dict:
    if implementation == "torch":
        def fn():
            return torch_matmul(a, b)

        baseline = "torch.matmul"
    elif implementation == "cuda_naive":
        if dtype_name != "float32":
            raise SystemExit("cuda_naive only supports float32 in M1")
        if device != "cuda":
            raise SystemExit("cuda_naive requires --device cuda or CUDA-capable --device auto")
        if not has_cuda_matmul_naive():
            raise SystemExit("cuda_naive requires CUDA, PyTorch CUDA, and nvcc")

        def fn():
            return cuda_matmul_naive(a, b)

        baseline = "torch.matmul"
    else:
        raise ValueError(f"unknown implementation: {implementation}")

    actual = fn()
    errors = max_error(actual, expected)
    timing = benchmark_torch_callable(fn, warmup=warmup, iterations=iterations, device=device)
    machine = machine_info_dict(device=device)
    m, n, k = shape

    return {
        "operator": "matmul",
        "implementation": implementation,
        "dtype": dtype_name,
        "shape": {"m": m, "n": n, "k": k},
        "warmup": warmup,
        "iterations": iterations,
        "baseline": baseline,
        "max_abs_error": errors.max_abs_error,
        "max_rel_error": errors.max_rel_error,
        **timing,
        "device": machine["device"],
        "machine": machine,
    }


def main() -> None:
    args = parse_args()
    torch = _require_torch()

    device = resolve_device(args.device)
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]

    if device == "cpu" and dtype in {torch.float16, torch.bfloat16}:
        raise SystemExit(f"{args.dtype} CPU timing is not used for the matmul benchmark harness")

    m, n, k = args.shape
    a, b = make_inputs(torch, m=m, n=n, k=k, device=device, dtype=dtype)
    expected = torch.matmul(a, b)

    implementations = ["torch", "cuda_naive"] if args.implementation == "all" else [args.implementation]
    results = [
        benchmark_one(
            implementation=implementation,
            a=a,
            b=b,
            expected=expected,
            dtype_name=args.dtype,
            shape=(m, n, k),
            warmup=args.warmup,
            iterations=args.iterations,
            device=device,
        )
        for implementation in implementations
    ]

    torch_p50 = next((result["p50_ms"] for result in results if result["implementation"] == "torch"), None)
    for result in results:
        if torch_p50 is not None:
            result["speedup_vs_baseline"] = torch_p50 / result["p50_ms"]
        elif result["implementation"] == "torch":
            result["speedup_vs_baseline"] = 1.0
        else:
            result["speedup_vs_baseline"] = None

    result = results[0] if len(results) == 1 else results
    write_json_result(result, args.output)


if __name__ == "__main__":
    main()
