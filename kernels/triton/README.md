# Triton Kernels

Triton implementations start in M3.

- `matmul.py` is the source-tree implementation mirror for the M3 FP32 Triton matmul.
- The importable package implementation lives at `forgenpu_kernels/triton/matmul.py`.
- The benchmark exposes it with `uv run forgenpu-bench-matmul --implementation triton --device cuda`.

Validation on a Linux CUDA machine:

```bash
uv sync --extra dev --extra triton
uv run --extra dev --extra triton pytest tests/test_matmul.py -k "triton or torch"
uv run forgenpu-bench-matmul --implementation triton --device cuda
```
