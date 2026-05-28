#!/usr/bin/env python
"""CLI for benchmarking PyTorch and ForgeNPU matmul implementations."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forgenpu_kernels.matmul_benchmark import (  # noqa: E402
    MatmulBenchmarkConfig,
    result_payload,
    run_matmul_benchmark,
)
from forgenpu_kernels.reporting import write_benchmark_result  # noqa: E402


@dataclass(frozen=True)
class ProgressLogger:
    quiet: bool

    def __call__(self, message: str) -> None:
        if not self.quiet:
            print(f"[bench] {message}", file=sys.stderr, flush=True)


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
        choices=["torch", "cuda_naive", "cuda_tiled", "all"],
        help="Implementation to benchmark. CUDA matmul implementations support float32 CUDA tensors only.",
    )
    parser.add_argument(
        "--format",
        default="json",
        choices=["json", "table"],
        help="Output format. JSON is the stable format for scripts; table is easier to read interactively.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress logs on stderr.")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> MatmulBenchmarkConfig:
    return MatmulBenchmarkConfig(
        shape=tuple(args.shape),
        warmup=args.warmup,
        iterations=args.iterations,
        device=args.device,
        dtype=args.dtype,
        implementation=args.implementation,
    )


def main() -> None:
    args = parse_args()
    logger = ProgressLogger(quiet=args.quiet)

    try:
        results = run_matmul_benchmark(config_from_args(args), progress=logger)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    write_benchmark_result(result_payload(results), args.output, args.format)
    logger("done")


if __name__ == "__main__":
    main()
