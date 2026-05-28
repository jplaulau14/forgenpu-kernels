# Matmul Roofline Notes

This is the first roofline-style note for the project. It is intentionally scoped to M2 FP32 matmul on CUDA.

## What Roofline Analysis Is For

Roofline analysis relates arithmetic work to memory movement. It helps answer whether a kernel is limited by compute throughput, memory bandwidth, launch overhead, or implementation quality.

For matmul:

```text
C[M, N] = A[M, K] @ B[K, N]
```

Estimated work:

```text
FLOPs = 2 * M * N * K
```

Compulsory FP32 IO lower bound:

```text
bytes = 4 * (M*K + K*N + M*N)
```

Arithmetic intensity:

```text
FLOPs / compulsory bytes
```

This lower bound is not the same as what the naive kernel actually does. It assumes each input is read once and the output is written once.

## Naive Global-Memory Behavior

`cuda_naive` maps one thread to one output element. Each thread loops over the full `K` dimension:

```text
sum += A[row, inner] * B[inner, col]
```

That means neighboring output elements reread overlapping input values from global memory. A rough traffic estimate is:

```text
4 * (2*M*N*K + M*N)
```

The key problem is not correctness. The key problem is reuse: the kernel does not give nearby threads a way to share the `A` and `B` values they all need.

## Tiled Shared-Memory Behavior

`cuda_tiled` keeps the same one-thread-per-output-element mapping, but each block cooperatively loads:

- one `16 x 16` tile from `A`,
- one `16 x 16` tile from `B`.

Threads then reuse those values from shared memory before the block advances to the next `K` tile.

For tile size 16, a rough global-memory traffic estimate is:

```text
tile_m = ceil(M / 16)
tile_n = ceil(N / 16)
bytes = 4 * (tile_n*M*K + tile_m*K*N + M*N)
```

This is still an estimate. It does not replace profiler counters, cache behavior, or instruction-level analysis. It is useful because it explains why shared-memory tiling should reduce redundant global-memory reads.

## How To Read M2 Results

Expected M2 story:

- `cuda_tiled` should beat `cuda_naive` on large enough shapes.
- `cuda_tiled` should still lose to `torch.matmul`.
- If `cuda_tiled` does not beat `cuda_naive`, inspect tile size, occupancy, synchronization overhead, memory coalescing, and whether the shape is too small.

Do not claim production optimization from M2. A serious optimized matmul still needs techniques not present here:

- register blocking,
- vectorized loads,
- Tensor Cores,
- warp-level matrix multiply instructions,
- asynchronous copy,
- architecture-specific schedule tuning,
- better launch/configuration search.

## M2 Evidence To Record

For a RunPod H100 run, record:

- device,
- driver,
- CUDA runtime,
- PyTorch version,
- shape,
- warmup and iterations,
- p50 latency for `torch`, `cuda_naive`, and `cuda_tiled`,
- speedup of `cuda_tiled` vs `cuda_naive`,
- speedup of `cuda_tiled` vs `torch`,
- achieved TFLOP/s,
- profiler artifact path or fallback note.

## M2 RunPod H100 Result

Curated artifact:

- `results/profiles/m2-runpod-h100.md`

For `1024 x 1024 x 1024` FP32 matmul on the RunPod H100:

| Implementation | p50 ms | TFLOP/s | Speedup vs PyTorch | Speedup vs naive |
|---|---:|---:|---:|---:|
| `torch` | 0.0682 | 31.50 | 1.0000 | 6.3600 |
| `cuda_naive` | 0.4336 | 4.95 | 0.1572 | 1.0000 |
| `cuda_tiled` | 0.2810 | 7.64 | 0.2427 | 1.5433 |

The tiled kernel is about `1.54x` faster than the naive kernel for this shape. That supports the expected memory-reuse story. It is still about `4.12x` slower than PyTorch by p50 latency, which is expected because this kernel does not use Tensor Cores, register blocking, vectorized loads, asynchronous copy, or vendor-tuned schedules.

The PyTorch profiler fallback captured one `matmul_tiled_kernel` launch at about `273.852us` CUDA time for this shape. Nsight Compute was present in the container, but the scripted attempt did not produce a kernel profile before timeout, so the committed profiler artifact uses the PyTorch CUDA timeline summary.
