# Design

ForgeNPU Kernels follows this flow:

```text
PyTorch reference
  -> correctness tests
  -> custom CUDA/C++ and Triton operator
  -> benchmark harness
  -> profiler tooling
  -> decoder-block integration
  -> docs and writeups
```

M0 established the reference, test, benchmark, and build surfaces. M1 added the first custom CUDA kernel with naive FP32 matmul. M2 keeps the naive kernel available and adds a tiled shared-memory FP32 matmul so the repo can compare direct global-memory access against block-level data reuse. M3 adds a Triton FP32 matmul so the repo can compare CUDA C++ with a higher-level GPU kernel language.

## Boundaries

- Python owns correctness orchestration and benchmark reporting.
- CUDA/C++ owns the low-level kernel implementations.
- Triton owns rapid experiments and comparison implementations.
- Docs own methodology, limitations, and profiler interpretation.

## Current M3 Modules

- `forgenpu_kernels/cli/bench_matmul.py` owns the user-facing Typer command.
- `forgenpu_kernels/matmul_benchmark.py` owns matmul benchmark orchestration and workload estimates.
- `forgenpu_kernels/reporting.py` owns JSON and table output.
- `forgenpu_kernels/bindings.py` owns lazy PyTorch CUDA extension loading.
- `kernels/cuda/matmul/matmul_naive.cu` and `kernels/cuda/matmul/matmul_tiled.cu` own the M1/M2 CUDA kernels.
- `kernels/triton/matmul.py` and `forgenpu_kernels/triton/matmul.py` own the M3 Triton kernel.
- `docs/matmul.md`, `docs/benchmark-methodology.md`, `docs/roofline.md`, and `docs/profiler-notes.md` explain the current learning path and evidence boundaries.

The CUDA extension loader prefers the source-tree `kernels/cuda` files during development and falls back to packaged CUDA source files in normal installs. That keeps the development layout readable while making wheel installs able to compile the runtime PyTorch CUDA extensions.

The Triton implementation is regular Python source mirrored in both the public kernel tree and importable package tree. The source tree keeps the kernel discoverable, while the package tree gives benchmark code a stable import path.
