# Benchmark Methodology

## Matmul Method

The matmul benchmark compares `torch.matmul` against available ForgeNPU matmul implementations.

Current behavior:

- warm up before timing,
- synchronize CUDA before and during event timing when CUDA is available,
- use `torch.cuda.Event` on CUDA,
- use `time.perf_counter_ns` on CPU,
- report p50, p95, mean latency, correctness error, shape, dtype, and machine metadata.
- expose a Typer CLI at `uv run forgenpu-bench-matmul`,
- render interactive tables with Rich,
- write progress logs to stderr so long CUDA extension builds and benchmark runs do not look stalled.

M3 implementations:

- `torch`: PyTorch baseline.
- `cuda_naive`: one CUDA thread computes one output element.
- `cuda_tiled`: one CUDA thread computes one output element while each block stages `16 x 16` input tiles in shared memory.
- `triton`: one Triton program computes a `16 x 16` output tile with a blocked `tl.dot` loop over `K`.
- `all`: run every implementation that appears runnable in the current environment. CPU runs include only `torch`; CUDA runs add the custom CUDA extensions and Triton only when their dependencies are available.

CPU results are for harness validation only. GPU performance conclusions require a CUDA-capable PyTorch environment.

## Correctness

The custom CUDA and Triton matmul implementations are checked against `torch.matmul`.

Shape coverage includes:

- square: `16 x 16 x 16`
- rectangular: `31 x 47 x 19`
- projection-like: `8 x 128 x 64`
- non-tile-multiple M2 shape: `65 x 33 x 129`

The custom CUDA and Triton implementations are FP32-only. The test tolerance is intentionally modest because accumulation order can differ:

```text
rtol = 1e-4
atol = 1e-4
```

## Benchmark Metrics

Matmul benchmark records include:

- `estimated_flops`: `2 * M * N * K`
- `compulsory_io_bytes`: lower-bound dtype-aware bytes for reading `A`, reading `B`, and writing `C`
- `estimated_global_memory_bytes`: rough dtype-aware custom-kernel global-memory traffic estimate
- `arithmetic_intensity_flop_per_byte`: estimated FLOPs divided by compulsory bytes
- `achieved_tflops`: estimated FLOPs divided by p50 runtime
- `speedup_vs_baseline`: p50 speedup relative to `torch.matmul` when the same run includes a `torch` row
- `speedup_vs_naive`: p50 speedup relative to `cuda_naive` when available

For `cuda_tiled` and `triton`, the estimated global-memory byte model uses the same tiled-read formula because both implementations operate on blocked input tiles. The memory estimates are explanatory, not hardware-counter measurements. Use profiler output when making bottleneck claims.

## M3 Benchmark Commands

PyTorch baseline:

```bash
uv run forgenpu-bench-matmul \
  --implementation torch \
  --shape 512 512 512 \
  --warmup 5 \
  --iterations 20
```

CUDA naive on a GPU machine:

```bash
uv run forgenpu-bench-matmul \
  --implementation cuda_naive \
  --device cuda \
  --shape 512 512 512 \
  --warmup 5 \
  --iterations 20
```

CUDA tiled on a GPU machine:

```bash
uv run forgenpu-bench-matmul \
  --implementation cuda_tiled \
  --device cuda \
  --shape 512 512 512 \
  --warmup 5 \
  --iterations 20
```

Triton on a Linux CUDA machine:

```bash
uv sync --extra dev --extra triton
uv run forgenpu-bench-matmul \
  --implementation triton \
  --device cuda \
  --shape 512 512 512 \
  --warmup 5 \
  --iterations 20
```

Available GPU comparison:

```bash
uv run forgenpu-bench-matmul \
  --implementation all \
  --device cuda \
  --shape 512 512 512 \
  --warmup 5 \
  --iterations 20
```

Interactive table output:

```bash
uv run forgenpu-bench-matmul \
  --implementation all \
  --device cuda \
  --shape 1024 1024 1024 \
  --warmup 25 \
  --iterations 100 \
  --format table
```

Use `--quiet` when a script should suppress progress logs. JSON remains the default output format because it is the stable artifact format for saved benchmark results and profiler workflows.

## 1660 Ti Validation Target

The first M3 validation target is an NVIDIA GTX 1660 Ti accessed over SSH. Treat its numbers as accessible development evidence, not as H100/A100-class portfolio claims.

Suggested command sequence:

```bash
uv sync --extra dev --extra triton
uv run --extra dev --extra triton pytest tests/test_matmul.py -k "triton or torch"
uv run forgenpu-bench-matmul \
  --implementation all \
  --device cuda \
  --shape 1024 1024 1024 \
  --warmup 25 \
  --iterations 100 \
  --format json \
  --quiet \
  --output results/matmul_1024_gtx1660ti_m3.json
uv run forgenpu-bench-matmul \
  --implementation all \
  --device cuda \
  --shape 1024 1024 1024 \
  --warmup 25 \
  --iterations 100 \
  --format table
```

When recording results, include the GPU name, CUDA version, PyTorch version, Triton version, warmup count, iteration count, shape, dtype, and which implementations were included. If `cuda_naive`, `cuda_tiled`, or `triton` are absent from `--implementation all`, run that implementation explicitly to capture the unavailable reason.

CUDA tiled profiler run:

```bash
scripts/profile_matmul.sh 1024 1024 1024
```

On the GTX 1660 Ti, use the SM75 architecture explicitly if you profile the CUDA extension path:

```bash
TORCH_CUDA_ARCH_LIST=7.5 scripts/profile_matmul.sh 1024 1024 1024
```

The profiler script currently profiles `cuda_tiled`, not the M3 Triton kernel. Use it as CUDA tiled evidence and use the benchmark commands above for Triton-vs-CUDA timing until a Triton-specific profiler target exists.

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
