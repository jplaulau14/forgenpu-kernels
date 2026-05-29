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

M0 established the reference, test, benchmark, and build surfaces. M1 added the first custom CUDA kernel with naive FP32 matmul. M2 keeps the naive kernel available and adds a tiled shared-memory FP32 matmul so the repo can compare direct global-memory access against block-level data reuse.

## Boundaries

- Python owns correctness orchestration and benchmark reporting.
- CUDA/C++ owns the low-level kernel implementations.
- Triton owns rapid experiments and comparison implementations.
- Docs own methodology, limitations, and profiler interpretation.

## Current M2 Modules

- `forgenpu_kernels/cli/bench_matmul.py` owns the user-facing Typer command.
- `forgenpu_kernels/matmul_benchmark.py` owns matmul benchmark orchestration and workload estimates.
- `forgenpu_kernels/reporting.py` owns JSON and table output.
- `forgenpu_kernels/bindings.py` owns lazy PyTorch CUDA extension loading.
- `kernels/cuda/matmul/matmul_naive.cu` and `kernels/cuda/matmul/matmul_tiled.cu` own the M1/M2 CUDA kernels.
- `docs/matmul.md`, `docs/benchmark-methodology.md`, `docs/roofline.md`, and `docs/profiler-notes.md` explain the current learning path and evidence boundaries.

The current CUDA extension loader expects a source-tree checkout with `kernels/cuda` present. Wheel-style packaging of CUDA source files is intentionally not treated as solved in M2.
