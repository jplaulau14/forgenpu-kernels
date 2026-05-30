# Tensor Core Notes

M4 adds a WMMA matmul path exposed as `--implementation cuda_wmma`.

The implementation lives in `kernels/cuda/matmul/matmul_wmma.cu`. It accepts contiguous CUDA `float16` inputs, pads non-multiple-of-16 shapes with zeros, launches one warp per `16 x 16` output tile, uses `nvcuda::wmma` fragments for `16 x 16 x 16` matrix multiply-accumulate, and returns a `float32` output tensor.

## Constraints

- input dtype: `float16`
- accumulation dtype: `float32`
- output dtype: `float32`
- input layout: row-major contiguous tensors
- tile shape: `16 x 16 x 16`
- device support: NVIDIA WMMA support on compute capability 7.0 or newer

The wrapper handles arbitrary positive `M`, `N`, and `K` by padding to WMMA tile multiples. That keeps the kernel readable and makes non-multiple shapes correct, but it adds allocation and copy overhead that a production GEMM would avoid or hide.

## Evidence Boundary

`cuda_wmma` is a learning-quality Tensor Core path, not a production GEMM. It is useful for showing how dtype, layout, tile size, and accumulation type constrain Tensor Core programming.

Performance claims should come from a target GPU benchmark run, preferably the planned H100 run. GTX 1660 Ti profiler runs are useful for profiler workflow and FP32 kernel analysis in this project, but they should not be used as H100 Tensor Core evidence.
