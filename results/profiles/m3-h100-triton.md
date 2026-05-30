# M3 H100 Triton Benchmark

## Run Context

- Device: `NVIDIA H100 80GB HBM3`
- Commit: `50f3889`
- Shape: `1024 x 1024 x 1024`
- Dtype: `float32`
- Warmup iterations: `25`
- Timed iterations: `100`
- Python: `3.12.3`
- PyTorch: `2.7.1+cu126`
- CUDA runtime: `12.6`
- NVIDIA driver: `570.124.06`
- Triton: `3.3.1`
- Platform: `Linux-6.8.0-56-generic-x86_64-with-glibc2.39`

## Command

```bash
uv run forgenpu-bench-matmul \
  --implementation all \
  --device cuda \
  --shape 1024 1024 1024 \
  --warmup 25 \
  --iterations 100 \
  --format json \
  --quiet \
  --output results/matmul_1024_m3.json
```

## Results

| implementation | p50 ms | p95 ms | mean ms | TFLOP/s | vs torch | vs naive | max abs err | est global GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| torch | 0.068 | 0.072 | 0.069 | 31.462 | 1.000 | 6.322 | 0.00 | - |
| cuda_naive | 0.432 | 0.434 | 0.432 | 4.977 | 0.158 | 1.000 | 0.00 | 8.00 |
| cuda_tiled | 0.279 | 0.281 | 0.279 | 7.707 | 0.245 | 1.549 | 0.00 | 0.50 |
| triton | 0.187 | 0.192 | 0.188 | 11.487 | 0.365 | 2.308 | 0.00 | 0.50 |

## Interpretation

- All implementations matched `torch.matmul` for this run with `max_abs_error = 0.0`.
- Triton was `2.31x` faster than `cuda_naive` and `1.49x` faster than `cuda_tiled` by p50 latency.
- Triton still trailed `torch.matmul`; its p50 latency was `0.365x` of the PyTorch baseline speed.
- The tiled CUDA and Triton global-memory byte counts are explanatory estimates, not profiler counter measurements.
- This artifact is benchmark evidence, not hardware-counter profiler evidence.

## Evidence Boundary

This result supports M3 timing claims for this H100 environment, shape, dtype, commit, and software stack. It should not be used as profiler evidence for bottleneck causes. Any future profiler note should include a same-device benchmark run before drawing kernel-level conclusions.
