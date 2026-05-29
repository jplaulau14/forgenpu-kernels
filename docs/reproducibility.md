# Reproducibility

This document describes how to reproduce the project validation path locally or on a GPU instance.

M2 includes naive and tiled FP32 CUDA matmul implementations. The GPU run proves that the project environment, PyTorch CUDA visibility, correctness tests, benchmark harness, profiler entry point, C++ bridge, and CUDA extension builds work on a CUDA-capable machine.

## Local CPU Validation

From the repository root:

```bash
uv sync --extra dev
make quickstart
make build-cpp
```

Expected result:

- `make env` reports PyTorch installed.
- CUDA may be unavailable on CPU-only machines.
- `make test` passes.
- `make bench-matmul-table` prints a readable CPU benchmark table.
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
make bench-matmul-gpu
make profile-check
make profile-matmul
make build-cpp
```

Run CUDA benchmarks from a repository checkout or editable install. M2 compiles `.cu` files from `kernels/cuda` through PyTorch extension loading; wheel-style packaging of those CUDA sources is not part of the current milestone.

The GPU run should show:

- an NVIDIA GPU in `nvidia-smi`,
- `cuda_available: true` in `make env`,
- a non-null PyTorch CUDA version compatible with the installed NVIDIA driver,
- a benchmark table with `torch`, `cuda_naive`, and `cuda_tiled`,
- passing correctness tests,
- profiler output or an explicit profiler fallback note,
- successful C++ bridge build.

Use strict Nsight Compute mode when the goal is to prove `ncu` works, not just to get any profiler artifact:

```bash
make profile-matmul-ncu
```

This target sets `REQUIRE_NCU=1`. It fails if the `ncu` executable is missing, times out, or does not produce a `.ncu-rep` report.

If the output says `No kernels were profiled`, `ncu` was installed but did not capture the target kernel. The profile script now treats that as a failed Nsight Compute run and writes the full `ncu` log under `results/profiles/`.

The Nsight Compute path first runs a target smoke test without `ncu` and writes the log to `results/profiles/*_target_smoke.log`. If that smoke test fails, debug the CUDA extension or Python environment before debugging Nsight. If the smoke test passes but `ncu` still reports no kernels, the failure is in profiler attachment, CUDA profiler API handling, or the rented GPU container permissions.

`make profile-check` should report a `ninja` executable. PyTorch uses Ninja to build the CUDA extension at runtime. If `ninja` is missing after pulling new code, run `uv sync --extra dev` again before profiling.

The Makefile does not install Nsight tools. Nsight Compute and Nsight Systems are NVIDIA system tools that depend on the GPU image, CUDA/toolkit installation, container permissions, and host driver configuration. The repo checks and uses them; the rented GPU image should provide them, or they should be installed through the provider's supported image/package flow.

For M2, `make test` should run the CUDA matmul tests instead of skipping them.

## Benchmark Capture

For a small smoke benchmark:

```bash
uv run forgenpu-bench-matmul --shape 512 512 512 --warmup 5 --iterations 20
```

For a larger M1-ready baseline:

```bash
uv run forgenpu-bench-matmul --shape 1024 1024 1024 --warmup 25 --iterations 100
```

For PyTorch vs custom CUDA:

```bash
uv run forgenpu-bench-matmul \
  --implementation all \
  --device cuda \
  --shape 512 512 512 \
  --warmup 25 \
  --iterations 100
```

For a human-readable RunPod check:

```bash
uv run forgenpu-bench-matmul \
  --implementation all \
  --device cuda \
  --shape 1024 1024 1024 \
  --warmup 25 \
  --iterations 100 \
  --format table
```

Save generated benchmark outputs under `results/` when needed:

```bash
uv run forgenpu-bench-matmul \
  --shape 1024 1024 1024 \
  --warmup 25 \
  --iterations 100 \
  --output results/matmul_torch_1024_runpod.json
```

Raw generated result files are ignored by default. Curated numbers should be copied into docs only when they are ready to explain.

## Profiler Capture

On a CUDA machine:

```bash
scripts/profile_matmul.sh 1024 1024 1024
```

The script writes generated artifacts under `results/profiles/`. Raw `.ncu-rep`, `.nsys-rep`, logs, traces, and benchmark JSON files are ignored by default. Curated summaries can be committed as Markdown when they support a documented claim.

Before profiling, inspect the environment:

```bash
make profile-check
```

The profiler workflow logs each major step with a `[profile-matmul]` prefix. Look for one of these result markers:

- `PROFILE_RESULT=nsight_compute`
- `PROFILE_RESULT=nsight_systems`
- `PROFILE_RESULT=no_profiler`

For hardware-counter profiling, prefer `PROFILE_RESULT=nsight_compute`.

## What Counts As Reproduced

M2 is reproduced on GPU when the following command sequence succeeds:

```bash
uv sync --extra dev
make env
make test
uv run forgenpu-bench-matmul --implementation all --device cuda --shape 1024 1024 1024 --warmup 25 --iterations 100
scripts/profile_matmul.sh 1024 1024 1024
make build-cpp
```

The environment output must show CUDA availability. The test output must include the CUDA matmul correctness tests, and the benchmark output must include `torch`, `cuda_naive`, and `cuda_tiled` records when `--implementation all --device cuda` is used. If CUDA is unavailable on the GPU instance, the issue is environment setup, not the M2 harness.

## PyTorch And Driver Compatibility

RunPod images can expose a recent GPU with a driver that supports CUDA 12.x but not CUDA 13.x. The project pins Linux PyTorch to a CUDA-12-compatible release so `uv sync --extra dev` works on the current GPU validation environment.

If `make env` reports a PyTorch CUDA version but `cuda_available: false`, check the warning text. A message about an insufficient driver means the resolved PyTorch CUDA runtime is newer than the installed NVIDIA driver supports.
