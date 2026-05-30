# ForgeNPU Kernels

ForgeNPU Kernels is a CUDA/C++ and Triton transformer inference kernel systems project. The long-term target is a set of custom kernels, correctness tests, reproducible benchmarks, profiler-backed performance notes, and a minimal decoder-block execution path.

This repository is intentionally growing milestone by milestone. M0 established the project foundation, M1 added the first real custom CUDA kernel, M2 added a tiled shared-memory FP32 matmul, and M3 adds a Triton FP32 matmul for comparing CUDA C++ against a higher-level GPU kernel language.

## Why This Exists

Framework kernels are already strong. The point of this project is not to pretend that a first custom kernel beats PyTorch. The point is to build and explain transformer inference operators below PyTorch:

- how data moves through GPU memory,
- where launch overhead and memory bandwidth dominate,
- how shared-memory tiling and Tensor Cores change matmul behavior,
- why attention and KV cache layout are inference systems problems,
- and how isolated operators affect an integrated decoder block.

Every performance claim in this repo should include a baseline, shape, machine context, timing method, and limitations.

## Current Milestone: M3

M3 includes:

- project structure for CUDA, Triton, Python, tests, benchmarks, scripts, and docs,
- an environment check script,
- a thin CMake/C++ bridge,
- PyTorch reference operators,
- a naive FP32 CUDA matmul kernel,
- a tiled shared-memory FP32 CUDA matmul kernel,
- a blocked FP32 Triton matmul kernel,
- Python binding through a PyTorch CUDA extension,
- Triton correctness tests against `torch.matmul` when Triton and CUDA are available,
- matmul correctness tests across square, rectangular, projection-like, and non-tile-multiple shapes,
- benchmark selection for PyTorch, CUDA naive, CUDA tiled, Triton, or all implementations runnable in the current environment,
- profiler capture script for matmul,
- first roofline-style note explaining arithmetic intensity and memory traffic.

The custom CUDA and Triton kernels are still intentionally simple. The CUDA kernels expose low-level thread/block and memory-control mechanics. Triton expresses the same blocked matmul idea with less launch and indexing boilerplate. PyTorch remains the production-grade baseline.

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

For Triton validation on a Linux CUDA machine, include the optional Triton extra:

```bash
uv sync --extra dev --extra triton
```

PyTorch is a project dependency. On a CUDA Linux machine, verify that the resolved PyTorch build matches the target CUDA runtime before trusting GPU benchmark results.

GPU profiling tools are system dependencies, not Python packages. `ncu` usually comes from the CUDA toolkit. `nsys` comes from Nsight Systems and may be missing from rented GPU images. Check both with:

```bash
make env
make profile-check
```

If `nsys` is missing on an Ubuntu-based GPU container and you have root or sudo access, install the Nsight Systems CLI with:

```bash
make install-nsight-systems
```

This helper installs `nsight-systems-cli` when available, falling back to NVIDIA's devtools apt repository. If the provider blocks package installation, choose an image with Nsight Systems preinstalled or rely on the PyTorch profiler fallback.

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

Run the M3 GPU comparison on a CUDA machine:

```bash
uv run forgenpu-bench-matmul \
  --implementation all \
  --device cuda \
  --shape 1024 1024 1024 \
  --warmup 25 \
  --iterations 100 \
  --format table
```

Run only the Triton path:

```bash
uv run forgenpu-bench-matmul \
  --implementation triton \
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

Run PyTorch, naive CUDA, tiled CUDA, and Triton matmul on a CUDA machine when all are available:

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
make install-nsight-systems
make profile-check
make profile-matmul
make profile-matmul-nsys
make build-cpp
```

`make profile-matmul` profiles the CUDA tiled path. It is still useful CUDA evidence during M3, but it is not a Triton profiler target.

For a GPU-backed reproducibility run, see [docs/reproducibility.md](docs/reproducibility.md).

## Benchmark Method

The matmul benchmark uses:

- warmup iterations before timing,
- `torch.cuda.Event` timing on CUDA tensors,
- `time.perf_counter` timing on CPU tensors,
- synchronization around measured regions,
- p50, p95, and mean latency,
- machine and software metadata in the JSON output.

M3 supports:

- `--implementation torch`
- `--implementation cuda_naive`
- `--implementation cuda_tiled`
- `--implementation triton`
- `--implementation all`
- `--format json` for scripts and saved artifacts
- `--format table` for interactive reading

The custom CUDA and Triton implementations only support FP32 CUDA tensors. `--implementation all` includes custom CUDA extensions and Triton only when those paths appear runnable in the current environment. Benchmark records include estimated FLOPs, achieved TFLOP/s, compulsory IO bytes, and an estimated global-memory byte count for custom kernels.

## Documentation Format

This nested repository is public code documentation, so Markdown files intentionally use normal GitHub Markdown instead of Obsidian YAML frontmatter. Private learning notes, concept traces, and journal entries live outside this package under `_private-specs/forgenpu-kernels/`.

## Roadmap

- M4: Tensor Core matmul path with dtype and layout notes.
- M5-M10: normalization, softmax, RoPE, KV cache, attention, FlashAttention-style attention, and decoder-block integration.

## Known Limits

- The custom CUDA matmul implementations are FP32-only.
- The Triton matmul implementation is FP32-only.
- The CUDA matmul implementations require a CUDA-capable PyTorch environment and nvcc.
- The Triton matmul implementation requires a CUDA-capable PyTorch environment and the optional Triton dependency.
- CUDA benchmark runs compile `.cu` files at runtime through PyTorch extension loading. Source checkouts use `kernels/cuda`; packaged installs use CUDA source files included as package data.
- The tiled kernel uses shared-memory tiling, but no register blocking, Tensor Cores, vectorized loads, or layout transforms.
- CPU benchmark output is useful for harness validation, not GPU-kernel conclusions.
