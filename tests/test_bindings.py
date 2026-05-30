from __future__ import annotations

from pathlib import Path

import pytest

import forgenpu_kernels
from forgenpu_kernels.bindings import (
    MATMUL_NAIVE,
    MATMUL_TILED,
    cuda_matmul_naive_unavailable_reason,
    cuda_matmul_tiled_unavailable_reason,
    cuda_matmul_naive,
    cuda_matmul_tiled,
    has_cuda_matmul_naive,
    has_cuda_matmul_tiled,
    _kernel_source,
)


def test_cuda_matmul_bridge_reports_boolean_availability() -> None:
    assert isinstance(has_cuda_matmul_naive(), bool)
    assert isinstance(has_cuda_matmul_tiled(), bool)


def test_packaged_cuda_sources_stay_in_sync_with_source_tree() -> None:
    repo_root = Path(forgenpu_kernels.__file__).resolve().parents[1]

    for spec in (MATMUL_NAIVE, MATMUL_TILED):
        source_tree_file = repo_root / "kernels" / "cuda" / "matmul" / spec.source_filename
        packaged_file = repo_root / "forgenpu_kernels" / "cuda" / "matmul" / spec.source_filename

        assert source_tree_file.read_text(encoding="utf-8") == packaged_file.read_text(
            encoding="utf-8"
        )
        assert _kernel_source(spec) == source_tree_file


def test_cuda_matmul_naive_explains_missing_cuda_environment() -> None:
    if has_cuda_matmul_naive():
        pytest.skip("CUDA matmul is available in this environment")

    reason = cuda_matmul_naive_unavailable_reason()
    assert reason

    with pytest.raises(RuntimeError, match="requires"):
        cuda_matmul_naive(None, None)


def test_cuda_matmul_tiled_explains_missing_cuda_environment() -> None:
    if has_cuda_matmul_tiled():
        pytest.skip("CUDA matmul is available in this environment")

    reason = cuda_matmul_tiled_unavailable_reason()
    assert reason

    with pytest.raises(RuntimeError, match="requires"):
        cuda_matmul_tiled(None, None)
