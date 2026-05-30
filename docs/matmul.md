# Matmul

M1 added the first custom CUDA operator: naive FP32 matrix multiplication. M2 added a tiled shared-memory FP32 implementation so the repo can compare direct global-memory access against block-level data reuse. M3 adds a Triton FP32 matmul so the repo can compare CUDA C++ with a higher-level GPU kernel language.

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

### `triton`

`triton` computes the same FP32 matmul through a Python-embedded Triton kernel:

- input `A`: shape `[M, K]`
- input `B`: shape `[K, N]`
- output `C`: shape `[M, N]`
- dtype: `float32`
- device: CUDA
- program tile: `16 x 16` output elements
- reduction tile: `32` elements along `K`
- one Triton program owns one output tile
- `tl.load` masks handle non-multiple tile boundaries
- `tl.dot` expresses the tile multiply-accumulate

Triton still launches GPU work, but the programmer writes tile-level tensor operations instead of manually mapping every CUDA thread to one output element. The source-tree implementation mirror is `kernels/triton/matmul.py`; the importable implementation lives in `forgenpu_kernels/triton/matmul.py`.

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

This is still not a production matmul. The current custom matmul paths do not include:

- register blocking,
- vectorized loads,
- Tensor Cores,
- warp-level MMA,
- layout transforms,
- asynchronous copy pipelines.

## Why Triton Exists Beside CUDA

CUDA C++ is the low-level path. It makes thread/block geometry, shared memory, synchronization, and launch integration explicit. That is useful when the learning target is the GPU execution model itself.

Triton is the higher-level kernel path. It keeps the code close to Python, lets the implementation describe blocks of tensor work directly, and removes much of the host-side extension and per-thread indexing boilerplate. That is useful for rapid iteration on transformer operators such as matmul, normalization, softmax, and attention.

The repo keeps both because they answer different questions:

- CUDA shows what the hardware-facing mechanics look like.
- Triton shows how much of the same blocked-kernel idea can be expressed with a smaller kernel surface.
- PyTorch shows the production-grade baseline that custom kernels must be compared against.

## Expected Performance

`cuda_naive` is expected to lose to `torch.matmul` on realistic GPU shapes. `cuda_tiled` should usually improve over `cuda_naive` on sufficiently large shapes. `triton` may improve readability, performance, or both relative to the simple CUDA kernels, but it should still be treated as a baseline Triton implementation rather than a production matmul. PyTorch routes matmul to highly optimized vendor kernels, while these kernels demonstrate:

- CUDA launch integration,
- row/column indexing,
- shape boundary handling,
- correctness against PyTorch,
- benchmark comparability,
- the impact of shared-memory tiling.
- CUDA-vs-Triton implementation ergonomics.

Do not describe either custom CUDA implementation as production optimized.

## How To Run

PyTorch baseline:

```bash
uv run forgenpu-bench-matmul --implementation torch
```

Naive CUDA:

```bash
uv run forgenpu-bench-matmul --implementation cuda_naive --device cuda
```

Tiled CUDA:

```bash
uv run forgenpu-bench-matmul --implementation cuda_tiled --device cuda
```

Triton:

```bash
uv run forgenpu-bench-matmul --implementation triton --device cuda
```

All implementations:

```bash
uv run forgenpu-bench-matmul --implementation all --device cuda
```

Profiler capture:

```bash
scripts/profile_matmul.sh 1024 1024 1024
```

## Known Limits

- FP32 only.
- CUDA only for custom implementations.
- CUDA implementations require PyTorch CUDA and nvcc.
- Triton implementation requires PyTorch CUDA and Triton.
- `cuda_naive` has no shared memory.
- `cuda_tiled` has no register blocking or Tensor Core usage.
- `triton` is a baseline blocked matmul, not an autotuned production schedule.
- No Tensor Core usage.
- No batching.
- No transpose flags.
