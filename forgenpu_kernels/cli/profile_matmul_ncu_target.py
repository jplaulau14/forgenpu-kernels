"""Small CUDA target for Nsight Compute matmul profiling."""

from __future__ import annotations

import argparse
import sys

from forgenpu_kernels.bindings import cuda_matmul_tiled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run only the tiled CUDA matmul for ncu.")
    parser.add_argument(
        "--shape", nargs=3, type=int, metavar=("M", "N", "K"), default=[1024, 1024, 1024]
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--cuda-profiler-api",
        action="store_true",
        help="Call cudaProfilerStart/Stop around the measured kernel launches.",
    )
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[profile-matmul-ncu-target] {message}", file=sys.stderr)


def cuda_profiler_start(torch) -> None:
    try:
        torch.cuda.cudart().cudaProfilerStart()
    except Exception as exc:  # pragma: no cover - requires CUDA runtime behavior
        raise SystemExit(f"cudaProfilerStart failed: {exc}") from exc


def cuda_profiler_stop(torch) -> None:
    try:
        torch.cuda.cudart().cudaProfilerStop()
    except Exception as exc:  # pragma: no cover - requires CUDA runtime behavior
        raise SystemExit(f"cudaProfilerStop failed: {exc}") from exc


def main() -> None:
    args = parse_args()

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("Nsight Compute target requires CUDA availability")
    if args.warmup < 0:
        raise SystemExit("--warmup must be non-negative")
    if args.iterations <= 0:
        raise SystemExit("--iterations must be positive")

    m, n, k = args.shape
    if min(m, n, k) <= 0:
        raise SystemExit("--shape values must be positive")

    shape_label = f"{m}x{n}x{k}"
    log(
        f"shape={shape_label} warmup={args.warmup} "
        f"iterations={args.iterations} cuda_profiler_api={args.cuda_profiler_api}"
    )

    generator = torch.Generator(device="cuda")
    generator.manual_seed(0)
    a = torch.randn((m, k), device="cuda", dtype=torch.float32, generator=generator)
    b = torch.randn((k, n), device="cuda", dtype=torch.float32, generator=generator)

    log("running warmup")
    for _ in range(args.warmup):
        cuda_matmul_tiled(a, b)
    torch.cuda.synchronize()

    log("running profiled tiled matmul launches")
    if args.cuda_profiler_api:
        log("calling cudaProfilerStart")
        cuda_profiler_start(torch)
    try:
        for _ in range(args.iterations):
            cuda_matmul_tiled(a, b)
        torch.cuda.synchronize()
    finally:
        if args.cuda_profiler_api:
            log("calling cudaProfilerStop")
            cuda_profiler_stop(torch)
    log("done")


if __name__ == "__main__":
    main()
