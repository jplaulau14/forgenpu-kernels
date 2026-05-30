"""Capture a PyTorch profiler trace for the M2 tiled matmul."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from forgenpu_kernels.bindings import cuda_matmul_tiled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile the tiled CUDA matmul with PyTorch profiler."
    )
    parser.add_argument(
        "--shape", nargs=3, type=int, metavar=("M", "N", "K"), default=[1024, 1024, 1024]
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("results/profiles"))
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[profile-matmul-torch] {message}", file=sys.stderr)


def main() -> None:
    args = parse_args()

    import torch
    from torch.profiler import ProfilerActivity, profile

    if not torch.cuda.is_available():
        raise SystemExit("PyTorch CUDA profiler requires CUDA availability")

    m, n, k = args.shape
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shape_label = f"{m}x{n}x{k}"
    trace_path = args.output_dir / f"matmul_tiled_{shape_label}_torch_trace.json"
    summary_path = args.output_dir / f"matmul_tiled_{shape_label}_torch_profile.txt"

    log(f"shape={shape_label} warmup={args.warmup} iterations={args.iterations}")
    log(f"trace={trace_path}")
    log(f"summary={summary_path}")

    generator = torch.Generator(device="cuda")
    generator.manual_seed(0)
    a = torch.randn((m, k), device="cuda", dtype=torch.float32, generator=generator)
    b = torch.randn((k, n), device="cuda", dtype=torch.float32, generator=generator)

    log("running warmup")
    for _ in range(args.warmup):
        cuda_matmul_tiled(a, b)
    torch.cuda.synchronize()

    log("capturing PyTorch profiler trace")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True
    ) as prof:
        for _ in range(args.iterations):
            cuda_matmul_tiled(a, b)
        torch.cuda.synchronize()

    prof.export_chrome_trace(str(trace_path))
    summary = prof.key_averages().table(sort_by="cuda_time_total", row_limit=12)
    summary_path.write_text(summary + "\n", encoding="utf-8")

    log(f"PyTorch profiler trace: {trace_path}")
    log(f"PyTorch profiler summary: {summary_path}")


if __name__ == "__main__":
    main()
