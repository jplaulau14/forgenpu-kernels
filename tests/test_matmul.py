from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from forgenpu_kernels.bindings import (  # noqa: E402
    cuda_matmul_naive,
    cuda_matmul_tiled,
    has_cuda_matmul_naive,
    has_cuda_matmul_tiled,
)
from forgenpu_kernels.ops import max_error, torch_matmul  # noqa: E402
from forgenpu_kernels.triton import has_triton_matmul, triton_matmul  # noqa: E402


@pytest.mark.parametrize(
    ("m", "n", "k"),
    [
        (16, 16, 16),
        (32, 64, 16),
        (8, 128, 64),
    ],
)
def test_torch_matmul_reference_matches_pytorch(m: int, n: int, k: int) -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    a = torch.randn((m, k), generator=generator)
    b = torch.randn((k, n), generator=generator)

    actual = torch_matmul(a, b)
    expected = torch.matmul(a, b)
    errors = max_error(actual, expected)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert errors.max_abs_error == 0.0
    assert errors.max_rel_error == 0.0


def test_max_error_handles_identical_zero_float16_tensors() -> None:
    actual = torch.zeros((4, 4), dtype=torch.float16)
    expected = torch.zeros((4, 4), dtype=torch.float16)

    errors = max_error(actual, expected)

    assert errors.max_abs_error == 0.0
    assert errors.max_rel_error == 0.0


def test_max_error_casts_float16_before_subtracting() -> None:
    actual = torch.tensor([65504.0], dtype=torch.float16)
    expected = torch.tensor([-65504.0], dtype=torch.float16)

    errors = max_error(actual, expected)

    assert errors.max_abs_error == 131_008.0
    assert errors.max_rel_error == 2.0


def test_max_error_handles_matching_empty_tensors() -> None:
    actual = torch.empty((0, 4), dtype=torch.float32)
    expected = torch.empty((0, 4), dtype=torch.float32)

    errors = max_error(actual, expected)

    assert errors.max_abs_error == 0.0
    assert errors.max_rel_error == 0.0


@pytest.mark.skipif(not has_cuda_matmul_naive(), reason="CUDA extension toolchain is unavailable")
@pytest.mark.parametrize(
    ("m", "n", "k"),
    [
        (16, 16, 16),
        (31, 47, 19),
        (8, 128, 64),
    ],
)
def test_cuda_matmul_naive_matches_pytorch(m: int, n: int, k: int) -> None:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(0)
    a = torch.randn((m, k), device="cuda", dtype=torch.float32, generator=generator)
    b = torch.randn((k, n), device="cuda", dtype=torch.float32, generator=generator)

    actual = cuda_matmul_naive(a, b)
    expected = torch.matmul(a, b)
    errors = max_error(actual, expected)

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    assert errors.max_abs_error < 1e-3


@pytest.mark.skipif(not has_cuda_matmul_naive(), reason="CUDA extension toolchain is unavailable")
@pytest.mark.parametrize(("m", "n", "k"), [(0, 16, 8), (16, 0, 8)])
def test_cuda_matmul_naive_matches_empty_pytorch_output(m: int, n: int, k: int) -> None:
    a = torch.empty((m, k), device="cuda", dtype=torch.float32)
    b = torch.empty((k, n), device="cuda", dtype=torch.float32)

    actual = cuda_matmul_naive(a, b)
    expected = torch.matmul(a, b)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert tuple(actual.shape) == (m, n)


@pytest.mark.skipif(not has_cuda_matmul_tiled(), reason="CUDA extension toolchain is unavailable")
@pytest.mark.parametrize(
    ("m", "n", "k"),
    [
        (16, 16, 16),
        (31, 47, 19),
        (8, 128, 64),
        (65, 33, 129),
    ],
)
def test_cuda_matmul_tiled_matches_pytorch(m: int, n: int, k: int) -> None:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(0)
    a = torch.randn((m, k), device="cuda", dtype=torch.float32, generator=generator)
    b = torch.randn((k, n), device="cuda", dtype=torch.float32, generator=generator)

    actual = cuda_matmul_tiled(a, b)
    expected = torch.matmul(a, b)
    errors = max_error(actual, expected)

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    assert errors.max_abs_error < 1e-3


@pytest.mark.skipif(not has_cuda_matmul_tiled(), reason="CUDA extension toolchain is unavailable")
@pytest.mark.parametrize(("m", "n", "k"), [(0, 16, 8), (16, 0, 8)])
def test_cuda_matmul_tiled_matches_empty_pytorch_output(m: int, n: int, k: int) -> None:
    a = torch.empty((m, k), device="cuda", dtype=torch.float32)
    b = torch.empty((k, n), device="cuda", dtype=torch.float32)

    actual = cuda_matmul_tiled(a, b)
    expected = torch.matmul(a, b)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert tuple(actual.shape) == (m, n)


@pytest.mark.skipif(not has_triton_matmul(), reason="Triton CUDA environment is unavailable")
@pytest.mark.parametrize(
    ("m", "n", "k"),
    [
        (16, 16, 16),
        (31, 47, 19),
        (8, 128, 64),
        (65, 33, 129),
    ],
)
def test_triton_matmul_matches_pytorch(m: int, n: int, k: int) -> None:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(0)
    a = torch.randn((m, k), device="cuda", dtype=torch.float32, generator=generator)
    b = torch.randn((k, n), device="cuda", dtype=torch.float32, generator=generator)

    actual = triton_matmul(a, b)
    expected = torch.matmul(a, b)
    errors = max_error(actual, expected)

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    assert errors.max_abs_error < 1e-3


@pytest.mark.skipif(not has_triton_matmul(), reason="Triton CUDA environment is unavailable")
@pytest.mark.parametrize(("m", "n", "k"), [(0, 16, 8), (16, 0, 8)])
def test_triton_matmul_matches_empty_pytorch_output(m: int, n: int, k: int) -> None:
    a = torch.empty((m, k), device="cuda", dtype=torch.float32)
    b = torch.empty((k, n), device="cuda", dtype=torch.float32)

    actual = triton_matmul(a, b)
    expected = torch.matmul(a, b)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert tuple(actual.shape) == (m, n)
