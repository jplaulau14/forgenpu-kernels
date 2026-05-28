#!/usr/bin/env bash
set -euo pipefail

M="${1:-1024}"
N="${2:-1024}"
K="${3:-1024}"
WARMUP="${WARMUP:-25}"
ITERATIONS="${ITERATIONS:-100}"
PROFILE_WARMUP="${PROFILE_WARMUP:-5}"
PROFILE_ITERATIONS="${PROFILE_ITERATIONS:-10}"
OUT_DIR="${OUT_DIR:-results/profiles}"

mkdir -p "${OUT_DIR}"

shape_label="${M}x${N}x${K}"
benchmark_output="${OUT_DIR}/matmul_m2_${shape_label}_benchmark.json"

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"

uv run python benchmarks/bench_matmul.py \
  --implementation all \
  --device cuda \
  --shape "${M}" "${N}" "${K}" \
  --warmup "${WARMUP}" \
  --iterations "${ITERATIONS}" \
  --output "${benchmark_output}"

if command -v ncu >/dev/null 2>&1; then
  ncu_command=(
    ncu
    --target-processes all \
    --kernel-name regex:matmul_tiled_kernel \
    --launch-count 1 \
    --section LaunchStats \
    --section Occupancy \
    --force-overwrite \
    --export "${OUT_DIR}/matmul_tiled_${shape_label}_ncu" \
    uv run python benchmarks/bench_matmul.py \
      --implementation cuda_tiled \
      --device cuda \
      --shape "${M}" "${N}" "${K}" \
      --warmup "${PROFILE_WARMUP}" \
      --iterations "${PROFILE_ITERATIONS}"
  )

  if command -v timeout >/dev/null 2>&1; then
    if timeout "${NCU_TIMEOUT_SECONDS:-120}" "${ncu_command[@]}"; then
      echo "Nsight Compute report: ${OUT_DIR}/matmul_tiled_${shape_label}_ncu.ncu-rep"
    else
      echo "Nsight Compute did not complete; falling back to PyTorch profiler."
      uv run python scripts/profile_matmul_torch.py \
        --shape "${M}" "${N}" "${K}" \
        --warmup "${PROFILE_WARMUP}" \
        --iterations "${PROFILE_ITERATIONS}" \
        --output-dir "${OUT_DIR}"
    fi
  elif "${ncu_command[@]}"; then
    echo "Nsight Compute report: ${OUT_DIR}/matmul_tiled_${shape_label}_ncu.ncu-rep"
  else
    echo "Nsight Compute did not complete; falling back to PyTorch profiler."
    uv run python scripts/profile_matmul_torch.py \
      --shape "${M}" "${N}" "${K}" \
      --warmup "${PROFILE_WARMUP}" \
      --iterations "${PROFILE_ITERATIONS}" \
      --output-dir "${OUT_DIR}"
  fi
elif command -v nsys >/dev/null 2>&1; then
  nsys profile \
    --force-overwrite true \
    --output "${OUT_DIR}/matmul_tiled_${shape_label}_nsys" \
    uv run python benchmarks/bench_matmul.py \
      --implementation cuda_tiled \
      --device cuda \
      --shape "${M}" "${N}" "${K}" \
      --warmup "${PROFILE_WARMUP}" \
      --iterations "${PROFILE_ITERATIONS}"
  echo "Nsight Systems report: ${OUT_DIR}/matmul_tiled_${shape_label}_nsys.nsys-rep"
else
  fallback_output="${OUT_DIR}/matmul_tiled_${shape_label}_profile_fallback.txt"
  {
    echo "No ncu or nsys executable was found."
    echo "Benchmark evidence was still captured at: ${benchmark_output}"
    echo "Install Nsight Compute or Nsight Systems for hardware-counter profiling."
  } > "${fallback_output}"
  echo "Profiler fallback note: ${fallback_output}"
fi
