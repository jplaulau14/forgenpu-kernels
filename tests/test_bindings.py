from __future__ import annotations

import pytest

from forgenpu_kernels.bindings import (
    cuda_matmul_naive,
    cuda_matmul_tiled,
    has_cuda_matmul_naive,
    has_cuda_matmul_tiled,
)


def test_cuda_matmul_bridge_reports_boolean_availability() -> None:
    assert isinstance(has_cuda_matmul_naive(), bool)
    assert isinstance(has_cuda_matmul_tiled(), bool)


def test_cuda_matmul_naive_explains_missing_cuda_environment() -> None:
    if has_cuda_matmul_naive():
        pytest.skip("CUDA matmul is available in this environment")

    with pytest.raises(RuntimeError, match="CUDA"):
        cuda_matmul_naive(None, None)


def test_cuda_matmul_tiled_explains_missing_cuda_environment() -> None:
    if has_cuda_matmul_tiled():
        pytest.skip("CUDA matmul is available in this environment")

    with pytest.raises(RuntimeError, match="CUDA"):
        cuda_matmul_tiled(None, None)
