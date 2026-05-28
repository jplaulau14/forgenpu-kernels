"""Python bridge for native kernels.

M0 exposes the bridge boundary before custom CUDA kernels exist. M1 should replace
the placeholder with a real extension-backed naive matmul implementation.
"""


def has_cuda_matmul_naive() -> bool:
    return False


def cuda_matmul_naive(*_args, **_kwargs):
    raise NotImplementedError("Naive CUDA matmul starts in M1; M0 only exposes the bridge boundary.")
