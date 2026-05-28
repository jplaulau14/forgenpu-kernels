# Benchmark Methodology

## M0 Method

The M0 benchmark command validates the benchmark harness using `torch.matmul`.

Current behavior:

- warm up before timing,
- synchronize CUDA before and during event timing when CUDA is available,
- use `torch.cuda.Event` on CUDA,
- use `time.perf_counter_ns` on CPU,
- report p50, p95, mean latency, correctness error, shape, dtype, and machine metadata.

CPU results are for harness validation only. GPU performance conclusions require a CUDA-capable PyTorch environment and, later, custom kernels.

## Reporting Standard

Each benchmark result should include:

- operator and implementation,
- shape and dtype,
- warmup and timed iteration count,
- p50, p95, and mean latency,
- baseline,
- error tolerance or measured error,
- device, CUDA, PyTorch, Triton, driver, and commit metadata.
