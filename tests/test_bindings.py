from __future__ import annotations

import pytest

from forgenpu_kernels.bindings import cuda_matmul_naive, has_cuda_matmul_naive


def test_cuda_matmul_bridge_is_explicitly_unavailable_in_m0() -> None:
    assert has_cuda_matmul_naive() is False
    with pytest.raises(NotImplementedError, match="starts in M1"):
        cuda_matmul_naive()
