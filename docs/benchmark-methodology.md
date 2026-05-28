# Benchmark Methodology

## Matmul Method

The matmul benchmark compares `torch.matmul` against available ForgeNPU matmul implementations.

Current behavior:

- warm up before timing,
- synchronize CUDA before and during event timing when CUDA is available,
- use `torch.cuda.Event` on CUDA,
- use `time.perf_counter_ns` on CPU,
- report p50, p95, mean latency, correctness error, shape, dtype, and machine metadata.
- write progress logs to stderr so long CUDA extension builds and benchmark runs do not look stalled.

M2 implementations:

- `torch`: PyTorch baseline.
- `cuda_naive`: one CUDA thread computes one output element.
- `cuda_tiled`: one CUDA thread computes one output element while each block stages `16 x 16` input tiles in shared memory.
- `all`: run every implementation and report an array of benchmark records.

CPU results are for harness validation only. GPU performance conclusions require a CUDA-capable PyTorch environment.

## Correctness

The CUDA matmul implementations are checked against `torch.matmul`.

Shape coverage includes:

- square: `16 x 16 x 16`
- rectangular: `31 x 47 x 19`
- projection-like: `8 x 128 x 64`
- non-tile-multiple M2 shape: `65 x 33 x 129`

The CUDA implementations are FP32-only. The test tolerance is intentionally modest because accumulation order can differ:

```text
rtol = 1e-4
atol = 1e-4
```

## Benchmark Metrics

Matmul benchmark records include:

- `estimated_flops`: `2 * M * N * K`
- `compulsory_io_bytes`: lower-bound FP32 bytes for reading `A`, reading `B`, and writing `C`
- `estimated_global_memory_bytes`: rough custom-kernel global-memory traffic estimate
- `arithmetic_intensity_flop_per_byte`: estimated FLOPs divided by compulsory bytes
- `achieved_tflops`: estimated FLOPs divided by p50 runtime
- `speedup_vs_baseline`: p50 speedup relative to `torch.matmul`
- `speedup_vs_naive`: p50 speedup relative to `cuda_naive` when available

The memory estimates are explanatory, not hardware-counter measurements. Use profiler output when making bottleneck claims.

## M2 Benchmark Commands

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

CUDA tiled on a GPU machine:

```bash
uv run python benchmarks/bench_matmul.py \
  --implementation cuda_tiled \
  --device cuda \
  --shape 512 512 512 \
  --warmup 5 \
  --iterations 20
```

All implementations:

```bash
uv run python benchmarks/bench_matmul.py \
  --implementation all \
  --device cuda \
  --shape 512 512 512 \
  --warmup 5 \
  --iterations 20
```

Interactive table output:

```bash
uv run python benchmarks/bench_matmul.py \
  --implementation all \
  --device cuda \
  --shape 1024 1024 1024 \
  --warmup 25 \
  --iterations 100 \
  --format table
```

Use `--quiet` when a script should suppress progress logs. JSON remains the default output format because it is the stable artifact format for saved benchmark results and profiler workflows.

Profiler-backed matmul run:

```bash
scripts/profile_matmul.sh 1024 1024 1024
```

## Reporting Standard

Each benchmark result should include:

- operator and implementation,
- shape and dtype,
- warmup and timed iteration count,
- p50, p95, and mean latency,
- baseline,
- error tolerance or measured error,
- estimated FLOPs and achieved TFLOP/s,
- memory traffic estimates,
- device, CUDA, PyTorch, Triton, driver, and commit metadata.
