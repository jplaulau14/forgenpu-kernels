# ForgeNPU Kernels

ForgeNPU Kernels is a CUDA/C++ and Triton transformer inference kernel systems project. The long-term target is a set of custom kernels, correctness tests, reproducible benchmarks, profiler-backed performance notes, and a minimal decoder-block execution path.

This repository is intentionally growing milestone by milestone. M0 established the project foundation, M1 added the first real custom CUDA kernel, M2 added a tiled shared-memory FP32 matmul, M3 added a Triton FP32 matmul, and M4 adds a WMMA Tensor Core matmul path for FP16 inputs with FP32 accumulation/output.

## Why This Exists

Framework kernels are already strong. The point of this project is not to pretend that a first custom kernel beats PyTorch. The point is to build and explain transformer inference operators below PyTorch:

- how data moves through GPU memory,
- where launch overhead and memory bandwidth dominate,
- how shared-memory tiling and Tensor Cores change matmul behavior,
- why attention and KV cache layout are inference systems problems,
- and how isolated operators affect an integrated decoder block.

Every performance claim in this repo should include a baseline, shape, machine context, timing method, and limitations.

## Current Milestone: M4

M4 includes:

- project structure for CUDA, Triton, Python, tests, benchmarks, scripts, and docs,
- an environment check script,
- a thin CMake/C++ bridge,
- PyTorch reference operators,
- a naive FP32 CUDA matmul kernel,
- a tiled shared-memory FP32 CUDA matmul kernel,
- a blocked FP32 Triton matmul kernel,
- a WMMA Tensor Core CUDA matmul kernel for FP16 inputs and FP32 output,
- Python binding through a PyTorch CUDA extension,
- Triton correctness tests against `torch.matmul` when Triton and CUDA are available,
- WMMA correctness tests against a FP32 PyTorch oracle when CUDA is available,
- matmul correctness tests across square, rectangular, projection-like, and non-tile-multiple shapes,
- dtype-aware benchmark selection for PyTorch, CUDA naive, CUDA tiled, CUDA WMMA, Triton, or all implementations runnable in the current environment,
- profiler capture script for matmul,
- first roofline-style note explaining arithmetic intensity and memory traffic.

The custom CUDA and Triton kernels are still intentionally simple. The CUDA kernels expose low-level thread/block, shared-memory, and WMMA mechanics. Triton expresses a blocked matmul idea with less launch and indexing boilerplate. PyTorch remains the production-grade baseline.

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

Run the FP32 GPU comparison on a CUDA machine:

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
  --output results/matmul_1024_fp32.json
```

Run the M4 FP16 WMMA comparison on a CUDA machine with Tensor Core support:

```bash
uv run forgenpu-bench-matmul \
  --implementation all \
  --device cuda \
  --dtype float16 \
  --shape 1024 1024 1024 \
  --warmup 25 \
  --iterations 100 \
  --format table
```

Save the M4 FP16 result:

```bash
uv run forgenpu-bench-matmul \
  --implementation all \
  --device cuda \
  --dtype float16 \
  --shape 1024 1024 1024 \
  --warmup 25 \
  --iterations 100 \
  --format json \
  --quiet \
  --output results/matmul_1024_m4_fp16.json
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

Run the FP32 PyTorch, naive CUDA, tiled CUDA, and Triton comparison on a CUDA machine when all are available:

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

Run the FP16-input PyTorch and WMMA comparison on a WMMA-capable CUDA machine:

```bash
uv run forgenpu-bench-matmul \
  --implementation all \
  --device cuda \
  --dtype float16 \
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
make bench-matmul-gpu-fp16
make install-nsight-systems
make profile-check
make profile-matmul
make profile-matmul-nsys
make build-cpp
```

`make profile-matmul` profiles the CUDA tiled FP32 path. Profiler artifacts from a GTX 1660 Ti are useful for validating profiler workflow and FP32-kernel behavior in this project, but they should not be used as H100 Tensor Core performance evidence.

The FP16-input benchmark is not an identical output-dtype contract across rows. The `torch` row reports PyTorch's actual output dtype for the requested input dtype, while `cuda_wmma` intentionally returns FP32 after FP32 accumulation. Use the input, accumulation, and output dtype columns when reading speedups.

For a GPU-backed reproducibility run, see [docs/reproducibility.md](docs/reproducibility.md).

The first committed M3 H100 benchmark summary is [results/profiles/m3-h100-triton.md](results/profiles/m3-h100-triton.md). It is timing evidence, not profiler-counter evidence. When doing profiling work, pair the profiler capture with a benchmark run from the same device, shape, dtype, commit, and implementation.

## Benchmark Method

The matmul benchmark uses:

- warmup iterations before timing,
- `torch.cuda.Event` timing on CUDA tensors,
- `time.perf_counter_ns` timing on CPU tensors,
- synchronization around measured regions,
- p50, p95, and mean latency,
- machine and software metadata in the JSON output.

M4 supports:

- `--implementation torch`
- `--implementation cuda_naive`
- `--implementation cuda_tiled`
- `--implementation cuda_wmma`
- `--implementation triton`
- `--implementation all`
- `--format json` for scripts and saved artifacts
- `--format table` for interactive reading

The FP32 custom CUDA and Triton implementations support CUDA `float32` tensors. The M4 WMMA implementation supports CUDA `float16` inputs and returns `float32` output. `--implementation all` is dtype-aware: FP32 runs include the FP32 CUDA/Triton paths when available, while FP16 runs include `torch` and `cuda_wmma` when available. Benchmark records include input, accumulation, and output dtype fields, estimated FLOPs, achieved TFLOP/s, compulsory IO bytes, and an estimated global-memory byte count for custom kernels.

## Documentation Format

This nested repository is public code documentation, so Markdown files intentionally use normal GitHub Markdown instead of Obsidian YAML frontmatter. Private learning notes, concept traces, and journal entries live outside this package under `_private-specs/forgenpu-kernels/`.

## Roadmap

- M5-M10: normalization, softmax, RoPE, KV cache, attention, FlashAttention-style attention, and decoder-block integration.

## Known Limits

- `cuda_naive` and `cuda_tiled` are FP32-only.
- `cuda_wmma` is FP16-input and FP32-output only.
- The Triton matmul implementation is FP32-only.
- The CUDA matmul implementations require a CUDA-capable PyTorch environment and nvcc.
- The WMMA path requires NVIDIA WMMA support on compute capability 7.0 or newer; performance claims should be collected on the target benchmark GPU.
- The Triton matmul implementation requires a CUDA-capable PyTorch environment and the optional Triton dependency.
- CUDA benchmark runs compile `.cu` files at runtime through PyTorch extension loading. Source checkouts use `kernels/cuda`; packaged installs use CUDA source files included as package data.
- The tiled kernel uses shared-memory tiling, but no register blocking, Tensor Cores, vectorized loads, or layout transforms.
- The WMMA kernel is intentionally pedagogical: it uses one warp per output tile and explicit host-side padding rather than a production GEMM schedule.
- CPU benchmark output is useful for harness validation, not GPU-kernel conclusions.
