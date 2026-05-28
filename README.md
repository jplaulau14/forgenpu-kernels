# ForgeNPU Kernels

ForgeNPU Kernels is a CUDA/C++ and Triton transformer inference kernel systems project. The long-term target is a set of custom kernels, correctness tests, reproducible benchmarks, profiler-backed performance notes, and a minimal decoder-block execution path.

This repository is intentionally growing milestone by milestone. M0 established the project foundation, M1 added the first real custom CUDA kernel, and M2 adds a tiled shared-memory FP32 matmul for comparing global-memory reuse against the naive baseline.

## Why This Exists

Framework kernels are already strong. The point of this project is not to pretend that a first custom kernel beats PyTorch. The point is to build and explain transformer inference operators below PyTorch:

- how data moves through GPU memory,
- where launch overhead and memory bandwidth dominate,
- how shared-memory tiling and Tensor Cores change matmul behavior,
- why attention and KV cache layout are inference systems problems,
- and how isolated operators affect an integrated decoder block.

Every performance claim in this repo should include a baseline, shape, machine context, timing method, and limitations.

## Current Milestone: M2

M2 includes:

- project structure for CUDA, Triton, Python, tests, benchmarks, scripts, and docs,
- an environment check script,
- a thin CMake/C++ bridge,
- PyTorch reference operators,
- a naive FP32 CUDA matmul kernel,
- a tiled shared-memory FP32 CUDA matmul kernel,
- Python binding through a PyTorch CUDA extension,
- matmul correctness tests across square, rectangular, and projection-like shapes,
- benchmark selection for PyTorch, CUDA naive, CUDA tiled, or all implementations,
- profiler capture script for matmul,
- first roofline-style note explaining arithmetic intensity and memory traffic.

The tiled CUDA kernel is still intentionally simple. It is expected to improve over the naive kernel on meaningful GPU shapes, but it is not expected to beat PyTorch. Its purpose is to prove shared-memory tiling, benchmark discipline, and profiler-backed interpretation before Tensor Core work.

## Repository Layout

```text
forgenpu-kernels/
  CMakeLists.txt
  Makefile
  pyproject.toml
  docs/
  kernels/
    cuda/
    triton/
  cpp/
    include/
    src/
  forgenpu_kernels/
  tests/
  benchmarks/
  scripts/
  results/
```

## Setup

Use Python 3.10 or newer. Dependencies are managed with `uv`:

```bash
uv sync --extra dev
```

PyTorch is a project dependency. On a CUDA Linux machine, verify that the resolved PyTorch build matches the target CUDA runtime before trusting GPU benchmark results.

## Quick Start

Run the local CPU validation path:

```bash
uv sync --extra dev
make quickstart
```

Inspect the benchmark CLI:

```bash
uv run forgenpu-bench-matmul --help
```

Run a readable CPU smoke benchmark:

```bash
uv run forgenpu-bench-matmul \
  --implementation torch \
  --device auto \
  --shape 512 512 512 \
  --warmup 5 \
  --iterations 20 \
  --format table
```

Run the M2 GPU comparison on a CUDA machine:

```bash
uv run forgenpu-bench-matmul \
  --implementation all \
  --device cuda \
  --shape 1024 1024 1024 \
  --warmup 25 \
  --iterations 100 \
  --format table
```

Save a JSON result for scripts or later analysis:

```bash
uv run forgenpu-bench-matmul \
  --implementation all \
  --device cuda \
  --shape 1024 1024 1024 \
  --warmup 25 \
  --iterations 100 \
  --format json \
  --quiet \
  --output results/matmul_1024_runpod.json
```

The historical script path still works:

```bash
uv run python benchmarks/bench_matmul.py --help
```

## Development Commands

Check the environment:

```bash
uv run python scripts/env_check.py
```

Run the initial correctness test:

```bash
uv run --extra dev pytest
```

Run the PyTorch matmul benchmark:

```bash
uv run forgenpu-bench-matmul --shape 512 512 512 --warmup 5 --iterations 20
```

Run PyTorch, naive CUDA, and tiled CUDA matmul on a CUDA machine:

```bash
uv run forgenpu-bench-matmul \
  --implementation all \
  --device cuda \
  --shape 512 512 512 \
  --warmup 5 \
  --iterations 20
```

Use the table format for interactive runs:

```bash
uv run forgenpu-bench-matmul \
  --implementation all \
  --device cuda \
  --shape 1024 1024 1024 \
  --warmup 25 \
  --iterations 100 \
  --format table
```

Configure and build the C++ bridge:

```bash
cmake -S . -B build
cmake --build build
```

The same commands are available through `make`:

```bash
make help
make quickstart
make env
make lint
make test
make bench-matmul
make bench-matmul-table
make bench-matmul-gpu
make profile-matmul
make build-cpp
```

For a GPU-backed reproducibility run, see [docs/reproducibility.md](docs/reproducibility.md).

## Benchmark Method

The matmul benchmark uses:

- warmup iterations before timing,
- `torch.cuda.Event` timing on CUDA tensors,
- `time.perf_counter` timing on CPU tensors,
- synchronization around measured regions,
- p50, p95, and mean latency,
- machine and software metadata in the JSON output.

M2 supports:

- `--implementation torch`
- `--implementation cuda_naive`
- `--implementation cuda_tiled`
- `--implementation all`
- `--format json` for scripts and saved artifacts
- `--format table` for interactive reading

The CUDA implementations only support FP32 CUDA tensors. Benchmark records include estimated FLOPs, achieved TFLOP/s, compulsory IO bytes, and an estimated global-memory byte count for custom kernels.

## Roadmap

- M3: Triton matmul and CUDA-vs-Triton ergonomics note.
- M4: Tensor Core matmul path with dtype and layout notes.
- M5-M10: normalization, softmax, RoPE, KV cache, attention, FlashAttention-style attention, and decoder-block integration.

## Known Limits

- The custom CUDA matmul implementations are FP32-only.
- The CUDA matmul implementations require a CUDA-capable PyTorch environment and nvcc.
- The tiled kernel uses shared-memory tiling, but no register blocking, Tensor Cores, vectorized loads, or layout transforms.
- CPU benchmark output is useful for harness validation, not GPU-kernel conclusions.
