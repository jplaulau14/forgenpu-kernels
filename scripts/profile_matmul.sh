#!/usr/bin/env bash
set -euo pipefail

if command -v nsys >/dev/null 2>&1; then
  nsys profile -o results/matmul_torch_baseline uv run python benchmarks/bench_matmul.py "$@"
else
  echo "nsys not found; running benchmark without Nsight Systems" >&2
  uv run python benchmarks/bench_matmul.py "$@"
fi
