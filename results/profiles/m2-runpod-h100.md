# M2 RunPod H100 Profile Summary

## Setup

- Device: `NVIDIA H100 80GB HBM3`
- Driver: `570.124.06`
- CUDA toolkit: `12.8.93`
- PyTorch: `2.7.1+cu126`
- PyTorch CUDA runtime: `12.6`
- Python: `3.12.3`
- Shape: `1024 x 1024 x 1024`
- Dtype: `float32`

## Benchmark

Command:

```bash
uv run python benchmarks/bench_matmul.py \
  --implementation all \
  --device cuda \
  --shape 1024 1024 1024 \
  --warmup 25 \
  --iterations 100
```

| Implementation | p50 ms | TFLOP/s | Speedup vs PyTorch | Speedup vs naive | Max abs error |
|---|---:|---:|---:|---:|---:|
| `torch` | 0.0682 | 31.50 | 1.0000 | 6.3600 | 0.0 |
| `cuda_naive` | 0.4336 | 4.95 | 0.1572 | 1.0000 | 0.0 |
| `cuda_tiled` | 0.2810 | 7.64 | 0.2427 | 1.5433 | 0.0 |

Profiler-script benchmark smoke run:

| Implementation | p50 ms | TFLOP/s | Speedup vs PyTorch | Speedup vs naive | Max abs error |
|---|---:|---:|---:|---:|---:|
| `torch` | 0.0691 | 31.10 | 1.0000 | 6.2845 | 0.0 |
| `cuda_naive` | 0.4340 | 4.95 | 0.1591 | 1.0000 | 0.0 |
| `cuda_tiled` | 0.2842 | 7.56 | 0.2430 | 1.5269 | 0.0 |

## Profiler Artifact

Command:

```bash
NCU_TIMEOUT_SECONDS=5 PROFILE_ITERATIONS=1 PROFILE_WARMUP=1 WARMUP=3 ITERATIONS=5 \
  scripts/profile_matmul.sh 1024 1024 1024
```

Nsight Compute was present, but this container run did not produce a kernel profile before the timeout:

```text
==WARNING== No kernels were profiled.
Nsight Compute did not complete; falling back to PyTorch profiler.
```

The script generated:

- `results/profiles/matmul_m2_1024x1024x1024_benchmark.json`
- `results/profiles/matmul_tiled_1024x1024x1024_torch_trace.json`
- `results/profiles/matmul_tiled_1024x1024x1024_torch_profile.txt`

Raw generated JSON and traces are ignored. This Markdown file is the committed profiler summary.

Evidence boundary:

- This artifact proves the tiled CUDA kernel built, launched, and appeared in the PyTorch CUDA timeline.
- This artifact does not include Nsight Compute hardware counters, so it should not be used to claim measured occupancy, memory bandwidth, register pressure, or stall reasons.
- The memory-traffic numbers in the docs remain explanatory estimates until a successful hardware-counter profile is captured.

PyTorch profiler summary for one `cuda_tiled` launch:

```text
(anonymous namespace)::matmul_tiled_kernel(...) Self CUDA: 273.852us, CUDA total: 273.852us, calls: 1
cudaLaunchKernel Self CPU: 890.250us, calls: 1
cudaDeviceSynchronize Self CPU total: 194.350us, calls: 2
Self CUDA time total: 273.852us
```

## Interpretation

`cuda_tiled` improves over `cuda_naive` because each block reuses `16 x 16` tiles of `A` and `B` from shared memory instead of rereading all operands from global memory for each output element.

`cuda_tiled` remains slower than PyTorch because it still lacks register blocking, vectorized memory access, Tensor Cores, asynchronous copy, and architecture-specific tuning.
