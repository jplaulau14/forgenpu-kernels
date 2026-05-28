from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from forgenpu_kernels.ops import max_error, torch_matmul  # noqa: E402


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
