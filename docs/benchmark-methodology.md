# Benchmark Methodology

## Matmul Method

The matmul benchmark compares `torch.matmul` against available ForgeNPU matmul implementations.

Current behavior:

- warm up before timing,
- synchronize CUDA before and during event timing when CUDA is available,
- use `torch.cuda.Event` on CUDA,
- use `time.perf_counter_ns` on CPU,
- report p50, p95, mean latency, correctness error, shape, dtype, and machine metadata.

M1 implementations:

- `torch`: PyTorch baseline.
- `cuda_naive`: one CUDA thread computes one output element.
- `all`: run both implementations and report an array of benchmark records.

CPU results are for harness validation only. GPU performance conclusions require a CUDA-capable PyTorch environment.

## M1 Correctness

The naive CUDA matmul is checked against `torch.matmul`.

Shape coverage includes:

- square: `16 x 16 x 16`
- rectangular: `31 x 47 x 19`
- projection-like: `8 x 128 x 64`

M1 is FP32-only. The test tolerance is intentionally modest because accumulation order can differ:

```text
rtol = 1e-4
atol = 1e-4
```

## M1 Benchmark Commands

PyTorch baseline:

```bash
uv run python benchmarks/bench_matmul.py \
  --implementation torch \
  --shape 512 512 512 \
  --warmup 5 \
  --iterations 20
```

CUDA naive on a GPU machine:

```bash
uv run python benchmarks/bench_matmul.py \
  --implementation cuda_naive \
  --device cuda \
  --shape 512 512 512 \
  --warmup 5 \
  --iterations 20
```

Both implementations:

```bash
uv run python benchmarks/bench_matmul.py \
  --implementation all \
  --device cuda \
  --shape 512 512 512 \
  --warmup 5 \
  --iterations 20
```

## Reporting Standard

Each benchmark result should include:

- operator and implementation,
- shape and dtype,
- warmup and timed iteration count,
- p50, p95, and mean latency,
- baseline,
- error tolerance or measured error,
- device, CUDA, PyTorch, Triton, driver, and commit metadata.
