# ForgeNPU Kernels

ForgeNPU Kernels is a CUDA/C++ and Triton transformer inference kernel systems project. The long-term target is a set of custom kernels, correctness tests, reproducible benchmarks, profiler-backed performance notes, and a minimal decoder-block execution path.

This repository is intentionally starting from a small foundation. M0 establishes the project shape, environment checks, Python benchmark harness, CMake bridge, and the first PyTorch reference test. Custom CUDA kernels begin in M1.

## Why This Exists

Framework kernels are already strong. The point of this project is not to pretend that a first custom kernel beats PyTorch. The point is to build and explain transformer inference operators below PyTorch:

- how data moves through GPU memory,
- where launch overhead and memory bandwidth dominate,
- how shared-memory tiling and Tensor Cores change matmul behavior,
- why attention and KV cache layout are inference systems problems,
- and how isolated operators affect an integrated decoder block.

Every performance claim in this repo should include a baseline, shape, machine context, timing method, and limitations.

## Current Milestone: M0

M0 includes:

- project structure for CUDA, Triton, Python, tests, benchmarks, scripts, and docs,
- an environment check script,
- a thin CMake/C++ bridge,
- PyTorch reference operators,
- one matmul correctness test,
- one initial matmul benchmark command.

M0 does not include a custom CUDA matmul yet.

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

## First Commands

Check the environment:

```bash
uv run python scripts/env_check.py
```

Run the initial correctness test:

```bash
uv run --extra dev pytest tests/test_matmul.py
```

Run the initial PyTorch matmul benchmark:

```bash
uv run python benchmarks/bench_matmul.py --shape 512 512 512 --warmup 5 --iterations 20
```

Configure and build the C++ bridge:

```bash
cmake -S . -B build
cmake --build build
```

The same commands are available through `make`:

```bash
make env
make test
make bench-matmul
make build-cpp
```

## Benchmark Method

The initial benchmark uses:

- warmup iterations before timing,
- `torch.cuda.Event` timing on CUDA tensors,
- `time.perf_counter` timing on CPU tensors,
- synchronization around measured regions,
- p50, p95, and mean latency,
- machine and software metadata in the JSON output.

This is only the baseline harness. Later milestones add custom CUDA, tiled CUDA, Tensor Core, and Triton implementations behind the same benchmark surface.

## Roadmap

- M1: PyTorch matmul benchmark, naive CUDA matmul, correctness across at least three shapes.
- M2: tiled shared-memory CUDA matmul, benchmark against naive and PyTorch, first profiler artifact.
- M3: Triton matmul and CUDA-vs-Triton ergonomics note.
- M4: Tensor Core matmul path with dtype and layout notes.
- M5-M10: normalization, softmax, RoPE, KV cache, attention, FlashAttention-style attention, and decoder-block integration.

## Known Limits

- The current benchmark is a PyTorch baseline only.
- No custom CUDA kernel is exposed yet.
- GPU profiling artifacts are not present in M0.
- CPU benchmark output is useful for harness validation, not GPU-kernel conclusions.
