# Matmul

M1 added the first custom CUDA operator: naive FP32 matrix multiplication. M2 adds a tiled shared-memory FP32 implementation so the repo can compare direct global-memory access against block-level data reuse.

## Implementations

### `torch`

`torch` uses `torch.matmul` as both the correctness oracle and the baseline implementation.

### `cuda_naive`

`cuda_naive` is a deliberately simple CUDA kernel:

- input `A`: shape `[M, K]`
- input `B`: shape `[K, N]`
- output `C`: shape `[M, N]`
- dtype: `float32`
- one CUDA thread computes one output element `C[row, col]`
- each thread loops over `K` and accumulates in FP32
- block shape: `16 x 16`
- grid shape: `ceil(N / 16) x ceil(M / 16)`

Indexing:

```text
C[row, col] = sum(A[row, inner] * B[inner, col] for inner in 0..K-1)
```

The kernel uses boundary checks so non-multiple-of-16 shapes work correctly.

### `cuda_tiled`

`cuda_tiled` computes the same operation but stages input tiles in shared memory:

- input `A`: shape `[M, K]`
- input `B`: shape `[K, N]`
- output `C`: shape `[M, N]`
- dtype: `float32`
- one CUDA thread still computes one output element `C[row, col]`
- block shape: `16 x 16`
- each block loads a `16 x 16` tile of `A` and a `16 x 16` tile of `B`
- threads synchronize after loading each tile
- each thread accumulates across the tile before the block advances to the next `K` tile

Tile loop:

```text
for tile_start in range(0, K, 16):
    shared_A[thread_y, thread_x] = A[row, tile_start + thread_x]
    shared_B[thread_y, thread_x] = B[tile_start + thread_y, col]
    synchronize block
    accumulate 16 products from shared memory
    synchronize block
```

Out-of-range tile loads are filled with zero, so non-multiple-of-16 shapes work correctly.

## Why This Is Naive

The M1 `cuda_naive` kernel reads directly from global memory for every multiply-add. It does not use:

- shared-memory tiling,
- vectorized loads,
- Tensor Cores,
- warp-level scheduling tricks,
- register tiling,
- fusion,
- layout transforms.

That is intentional. It proves the custom CUDA integration path and establishes the slow baseline.

## Why Tiling Helps

Neighboring output elements reuse overlapping values from `A` and `B`. The naive kernel rereads those values from global memory. The tiled kernel lets a block cooperatively load a small tile once, then reuse it from shared memory for multiple multiply-adds.

For a `1024 x 1024 x 1024` FP32 matmul with tile size 16, the benchmark reports:

- naive estimated global-memory bytes: roughly every output element reads `K` values from `A` and `K` values from `B`,
- tiled estimated global-memory bytes: each output block reuses staged tiles across 256 output elements,
- compulsory IO bytes: the lower-bound read/write volume for the two inputs and one output.

This is still not a production matmul. M2 does not include:

- register blocking,
- vectorized loads,
- Tensor Cores,
- warp-level MMA,
- layout transforms,
- asynchronous copy pipelines.

## Expected Performance

`cuda_naive` is expected to lose to `torch.matmul` on realistic GPU shapes. `cuda_tiled` should usually improve over `cuda_naive` on sufficiently large shapes, but it is still expected to lose to PyTorch. PyTorch routes matmul to highly optimized vendor kernels, while these kernels demonstrate:

- CUDA launch integration,
- row/column indexing,
- shape boundary handling,
- correctness against PyTorch,
- benchmark comparability,
- the impact of shared-memory tiling.

Do not describe either custom CUDA implementation as production optimized.

## How To Run

PyTorch baseline:

```bash
uv run python benchmarks/bench_matmul.py --implementation torch
```

Naive CUDA:

```bash
uv run python benchmarks/bench_matmul.py --implementation cuda_naive --device cuda
```

Tiled CUDA:

```bash
uv run python benchmarks/bench_matmul.py --implementation cuda_tiled --device cuda
```

All implementations:

```bash
uv run python benchmarks/bench_matmul.py --implementation all --device cuda
```

Profiler capture:

```bash
scripts/profile_matmul.sh 1024 1024 1024
```

## Known Limits

- FP32 only.
- CUDA only.
- Requires PyTorch CUDA and nvcc.
- `cuda_naive` has no shared memory.
- `cuda_tiled` has no register blocking or Tensor Core usage.
- No Tensor Core usage.
- No batching.
- No transpose flags.
