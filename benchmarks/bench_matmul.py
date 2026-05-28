#!/usr/bin/env python
"""Initial PyTorch matmul benchmark command."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forgenpu_kernels.benchmarks import benchmark_torch_callable, machine_info_dict, write_json_result  # noqa: E402
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
    parser = argparse.ArgumentParser(description="Benchmark the PyTorch matmul baseline.")
    parser.add_argument("--shape", nargs=3, type=int, metavar=("M", "N", "K"), default=[512, 512, 512])
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch = _require_torch()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]

    if device == "cpu" and dtype in {torch.float16, torch.bfloat16}:
        raise SystemExit(f"{args.dtype} CPU timing is not used for the M0 benchmark harness")

    m, n, k = args.shape
    generator = torch.Generator(device=device)
    generator.manual_seed(0)
    a = torch.randn((m, k), device=device, dtype=dtype, generator=generator)
    b = torch.randn((k, n), device=device, dtype=dtype, generator=generator)

    actual = torch_matmul(a, b)
    expected = torch.matmul(a, b)
    errors = max_error(actual, expected)

    timing = benchmark_torch_callable(
        lambda: torch_matmul(a, b),
        warmup=args.warmup,
        iterations=args.iterations,
        device=device,
    )
    machine = machine_info_dict(device=device)

    result = {
        "operator": "matmul",
        "implementation": "torch.matmul",
        "dtype": args.dtype,
        "shape": {"m": m, "n": n, "k": k},
        "warmup": args.warmup,
        "iterations": args.iterations,
        "baseline": "torch.matmul",
        "speedup_vs_baseline": 1.0,
        "max_abs_error": errors.max_abs_error,
        "max_rel_error": errors.max_rel_error,
        **timing,
        "device": machine["device"],
        "machine": machine,
    }
    write_json_result(result, args.output)


if __name__ == "__main__":
    main()
