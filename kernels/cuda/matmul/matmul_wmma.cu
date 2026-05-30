#include <ATen/Indexing.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda_fp16.h>
#include <mma.h>

#include <cstdint>

namespace {

constexpr int kWmmaTile = 16;
constexpr int kWarpSize = 32;

int64_t round_up_to_wmma_tile(int64_t value) {
  return ((value + kWmmaTile - 1) / kWmmaTile) * kWmmaTile;
}

__global__ void matmul_wmma_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    float* __restrict__ c,
    int64_t m_padded,
    int64_t n_padded,
    int64_t k_padded) {
  namespace wmma = nvcuda::wmma;

  const int64_t tile_m = static_cast<int64_t>(blockIdx.y);
  const int64_t tile_n = static_cast<int64_t>(blockIdx.x);

  wmma::fragment<wmma::matrix_a, kWmmaTile, kWmmaTile, kWmmaTile, half, wmma::row_major>
      a_fragment;
  wmma::fragment<wmma::matrix_b, kWmmaTile, kWmmaTile, kWmmaTile, half, wmma::row_major>
      b_fragment;
  wmma::fragment<wmma::accumulator, kWmmaTile, kWmmaTile, kWmmaTile, float> accumulator;

  wmma::fill_fragment(accumulator, 0.0f);

  for (int64_t tile_k = 0; tile_k < k_padded; tile_k += kWmmaTile) {
    const half* a_tile = a + (tile_m * kWmmaTile * k_padded) + tile_k;
    const half* b_tile = b + (tile_k * n_padded) + (tile_n * kWmmaTile);

    wmma::load_matrix_sync(a_fragment, a_tile, k_padded);
    wmma::load_matrix_sync(b_fragment, b_tile, n_padded);
    wmma::mma_sync(accumulator, a_fragment, b_fragment, accumulator);
  }

  float* c_tile = c + (tile_m * kWmmaTile * n_padded) + (tile_n * kWmmaTile);
  wmma::store_matrix_sync(c_tile, accumulator, n_padded, wmma::mem_row_major);
}

void check_matmul_wmma_inputs(const torch::Tensor& a, const torch::Tensor& b) {
  TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
  TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
  TORCH_CHECK(a.device() == b.device(), "a and b must be on the same CUDA device");
  TORCH_CHECK(a.scalar_type() == torch::kFloat16, "a must be float16");
  TORCH_CHECK(b.scalar_type() == torch::kFloat16, "b must be float16");
  TORCH_CHECK(a.dim() == 2, "a must be rank 2");
  TORCH_CHECK(b.dim() == 2, "b must be rank 2");
  TORCH_CHECK(a.size(1) == b.size(0), "shape mismatch: a.shape[1] must equal b.shape[0]");
}

}  // namespace

torch::Tensor matmul_wmma(torch::Tensor a, torch::Tensor b) {
  check_matmul_wmma_inputs(a, b);

  const c10::cuda::CUDAGuard device_guard(a.device());
  a = a.contiguous();
  b = b.contiguous();

  const int64_t m = a.size(0);
  const int64_t k = a.size(1);
  const int64_t n = b.size(1);

  auto c_options = a.options().dtype(torch::kFloat32);
  if (m == 0 || n == 0) {
    return torch::empty({m, n}, c_options);
  }

  const int64_t m_padded = round_up_to_wmma_tile(m);
  const int64_t n_padded = round_up_to_wmma_tile(n);
  const int64_t k_padded = round_up_to_wmma_tile(k);

  auto a_padded = torch::zeros({m_padded, k_padded}, a.options());
  auto b_padded = torch::zeros({k_padded, n_padded}, b.options());

  using torch::indexing::Slice;
  if (m > 0 && k > 0) {
    a_padded.index({Slice(0, m), Slice(0, k)}).copy_(a);
  }
  if (k > 0 && n > 0) {
    b_padded.index({Slice(0, k), Slice(0, n)}).copy_(b);
  }

  auto c_padded = torch::empty({m_padded, n_padded}, c_options);
  const dim3 block(kWarpSize);
  const dim3 grid(
      static_cast<unsigned int>(n_padded / kWmmaTile),
      static_cast<unsigned int>(m_padded / kWmmaTile));

  const auto stream = at::cuda::getCurrentCUDAStream();
  matmul_wmma_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const half*>(a_padded.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(b_padded.data_ptr<at::Half>()),
      c_padded.data_ptr<float>(),
      m_padded,
      n_padded,
      k_padded);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return c_padded.index({Slice(0, m), Slice(0, n)}).contiguous();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matmul_wmma", &matmul_wmma, "WMMA Tensor Core FP16-input FP32-output CUDA matmul");
}
