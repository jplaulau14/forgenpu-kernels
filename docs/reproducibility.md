# Reproducibility

This document describes how to reproduce the project validation path locally or on a GPU instance.

M1 includes a naive FP32 CUDA matmul. The GPU run proves that the project environment, PyTorch CUDA visibility, correctness tests, benchmark harness, C++ bridge, and CUDA extension build work on a CUDA-capable machine.

## Local CPU Validation

From the repository root:

```bash
uv sync --extra dev
make env
make test
make bench-matmul
make build-cpp
```

Expected result:

- `make env` reports PyTorch installed.
- CUDA may be unavailable on CPU-only machines.
- `make test` passes.
- `make bench-matmul` prints benchmark JSON.
- `make build-cpp` builds the native bridge.

CPU benchmark output is useful for harness validation, not GPU performance claims.

## RunPod GPU Validation

Connect to the GPU instance with a sanitized SSH command:

```bash
ssh -i ~/.ssh/runpod_ed25519 <runpod-user>@ssh.runpod.io
```

Do not commit the concrete RunPod user, pod ID, host, or private key path if it contains account-specific information.

On the instance:

```bash
nvidia-smi
nvcc --version
git clone <repo-url>
cd forgenpu-kernels
uv sync --extra dev
make env
make test
make bench-matmul
make build-cpp
```

The GPU run should show:

- an NVIDIA GPU in `nvidia-smi`,
- `cuda_available: true` in `make env`,
- a non-null PyTorch CUDA version compatible with the installed NVIDIA driver,
- benchmark JSON with `"device"` set to the GPU name,
- passing correctness tests,
- successful C++ bridge build.

For M1, `make test` should run the CUDA matmul tests instead of skipping them.

## Benchmark Capture

For a small smoke benchmark:

```bash
uv run python benchmarks/bench_matmul.py --shape 512 512 512 --warmup 5 --iterations 20
```

For a larger M1-ready baseline:

```bash
uv run python benchmarks/bench_matmul.py --shape 1024 1024 1024 --warmup 25 --iterations 100
```

For PyTorch vs naive CUDA:

```bash
uv run python benchmarks/bench_matmul.py \
  --implementation all \
  --device cuda \
  --shape 512 512 512 \
  --warmup 25 \
  --iterations 100
```

Save generated benchmark outputs under `results/` when needed:

```bash
uv run python benchmarks/bench_matmul.py \
  --shape 1024 1024 1024 \
  --warmup 25 \
  --iterations 100 \
  --output results/matmul_torch_1024_runpod.json
```

Raw generated result files are ignored by default. Curated numbers should be copied into docs only when they are ready to explain.

## What Counts As Reproduced

M1 is reproduced on GPU when the following command sequence succeeds:

```bash
uv sync --extra dev
make env
make test
make bench-matmul
make build-cpp
```

The environment output must show CUDA availability. The test output must include the CUDA matmul correctness tests, and the benchmark output must include a `cuda_naive` record when `--implementation all --device cuda` is used. If CUDA is unavailable on the GPU instance, the issue is environment setup, not the M1 harness.

## PyTorch And Driver Compatibility

RunPod images can expose a recent GPU with a driver that supports CUDA 12.x but not CUDA 13.x. The project pins Linux PyTorch to a CUDA-12-compatible release so `uv sync --extra dev` works on the current GPU validation environment.

If `make env` reports a PyTorch CUDA version but `cuda_available: false`, check the warning text. A message about an insufficient driver means the resolved PyTorch CUDA runtime is newer than the installed NVIDIA driver supports.
