# Design

ForgeNPU Kernels follows this flow:

```text
PyTorch reference
  -> correctness tests
  -> custom CUDA/C++ and Triton operator
  -> benchmark harness
  -> profiler tooling
  -> decoder-block integration
  -> docs and writeups
```

M0 established the reference, test, benchmark, and build surfaces. M1 adds the first custom CUDA kernel.

## Boundaries

- Python owns correctness orchestration and benchmark reporting.
- CUDA/C++ owns the low-level kernel implementations.
- Triton owns rapid experiments and comparison implementations.
- Docs own methodology, limitations, and profiler interpretation.
