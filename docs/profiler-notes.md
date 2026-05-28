# Profiler Notes

No profiler artifact exists in M0.

The first profiler-backed note belongs to M2, after the tiled shared-memory CUDA matmul exists. That note should explain:

- what bottleneck was expected,
- what Nsight Systems or Nsight Compute showed,
- how the tiled implementation changed memory behavior,
- where the custom kernel still falls short.
