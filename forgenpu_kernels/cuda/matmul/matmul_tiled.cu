#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cstdint>

namespace {

constexpr int kTileSize = 16;

__global__ void matmul_tiled_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int64_t m,
    int64_t n,
    int64_t k) {
  __shared__ float a_tile[kTileSize][kTileSize];
  __shared__ float b_tile[kTileSize][kTileSize];

  const int64_t row = static_cast<int64_t>(blockIdx.y) * kTileSize + threadIdx.y;
  const int64_t col = static_cast<int64_t>(blockIdx.x) * kTileSize + threadIdx.x;

  float sum = 0.0f;
  for (int64_t tile_start = 0; tile_start < k; tile_start += kTileSize) {
    const int64_t a_col = tile_start + threadIdx.x;
    const int64_t b_row = tile_start + threadIdx.y;

    a_tile[threadIdx.y][threadIdx.x] = (row < m && a_col < k) ? a[row * k + a_col] : 0.0f;
    b_tile[threadIdx.y][threadIdx.x] = (b_row < k && col < n) ? b[b_row * n + col] : 0.0f;

    __syncthreads();

#pragma unroll
    for (int inner = 0; inner < kTileSize; ++inner) {
      sum += a_tile[threadIdx.y][inner] * b_tile[inner][threadIdx.x];
    }

    __syncthreads();
  }

  if (row < m && col < n) {
    c[row * n + col] = sum;
  }
}

void check_matmul_inputs(const torch::Tensor& a, const torch::Tensor& b) {
  TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
  TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
  TORCH_CHECK(a.device() == b.device(), "a and b must be on the same CUDA device");
  TORCH_CHECK(a.scalar_type() == torch::kFloat32, "a must be float32");
  TORCH_CHECK(b.scalar_type() == torch::kFloat32, "b must be float32");
  TORCH_CHECK(a.dim() == 2, "a must be rank 2");
  TORCH_CHECK(b.dim() == 2, "b must be rank 2");
  TORCH_CHECK(a.size(1) == b.size(0), "shape mismatch: a.shape[1] must equal b.shape[0]");
}

}  // namespace

torch::Tensor matmul_tiled(torch::Tensor a, torch::Tensor b) {
  check_matmul_inputs(a, b);

  const c10::cuda::CUDAGuard device_guard(a.device());
  a = a.contiguous();
  b = b.contiguous();

  const int64_t m = a.size(0);
  const int64_t k = a.size(1);
  const int64_t n = b.size(1);

  auto c = torch::empty({m, n}, a.options());

  const dim3 block(kTileSize, kTileSize);
  const dim3 grid(
      static_cast<unsigned int>((n + kTileSize - 1) / kTileSize),
      static_cast<unsigned int>((m + kTileSize - 1) / kTileSize));

  const auto stream = at::cuda::getCurrentCUDAStream();
  matmul_tiled_kernel<<<grid, block, 0, stream>>>(
      a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), m, n, k);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matmul_tiled", &matmul_tiled, "Tiled shared-memory FP32 CUDA matmul");
}
