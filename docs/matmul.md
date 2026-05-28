# Matmul

M1 adds the first custom CUDA operator: naive FP32 matrix multiplication.

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

## Why This Is Naive

This kernel reads directly from global memory for every multiply-add. It does not use:

- shared-memory tiling,
- vectorized loads,
- Tensor Cores,
- warp-level scheduling tricks,
- register tiling,
- fusion,
- layout transforms.

That is intentional. M1 proves the custom CUDA integration path and establishes the slow baseline. M2 should improve the memory behavior with shared-memory tiling.

## Expected Performance

`cuda_naive` is expected to lose to `torch.matmul` on realistic GPU shapes. PyTorch routes matmul to highly optimized vendor kernels, while this M1 kernel mostly demonstrates:

- CUDA launch integration,
- row/column indexing,
- shape boundary handling,
- correctness against PyTorch,
- benchmark comparability.

Do not describe `cuda_naive` as optimized.

## How To Run

PyTorch baseline:

```bash
uv run python benchmarks/bench_matmul.py --implementation torch
```

Naive CUDA:

```bash
uv run python benchmarks/bench_matmul.py --implementation cuda_naive --device cuda
```

Both:

```bash
uv run python benchmarks/bench_matmul.py --implementation all --device cuda
```

## Known Limits

- FP32 only.
- CUDA only.
- Requires PyTorch CUDA and nvcc.
- No shared memory.
- No Tensor Core usage.
- No batching.
- No transpose flags.
