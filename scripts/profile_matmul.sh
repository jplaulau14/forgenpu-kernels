#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--check" ]]; then
  CHECK_ONLY=1
  shift
else
  CHECK_ONLY=0
fi

M="${1:-1024}"
N="${2:-1024}"
K="${3:-1024}"
WARMUP="${WARMUP:-25}"
ITERATIONS="${ITERATIONS:-100}"
PROFILE_WARMUP="${PROFILE_WARMUP:-5}"
PROFILE_ITERATIONS="${PROFILE_ITERATIONS:-10}"
OUT_DIR="${OUT_DIR:-results/profiles}"
NCU_TIMEOUT_SECONDS="${NCU_TIMEOUT_SECONDS:-120}"
REQUIRE_NCU="${REQUIRE_NCU:-0}"
PROFILE_PYTHON="${PROFILE_PYTHON:-}"

mkdir -p "${OUT_DIR}"

shape_label="${M}x${N}x${K}"
benchmark_output="${OUT_DIR}/matmul_m2_${shape_label}_benchmark.json"
target_smoke_log="${OUT_DIR}/matmul_tiled_${shape_label}_target_smoke.log"
ncu_report_base="${OUT_DIR}/matmul_tiled_${shape_label}_ncu"
ncu_report="${ncu_report_base}.ncu-rep"
ncu_log="${OUT_DIR}/matmul_tiled_${shape_label}_ncu.log"
nsys_report_base="${OUT_DIR}/matmul_tiled_${shape_label}_nsys"
nsys_report="${nsys_report_base}.nsys-rep"
nsys_log="${OUT_DIR}/matmul_tiled_${shape_label}_nsys.log"
torch_profile_log="${OUT_DIR}/matmul_tiled_${shape_label}_torch_profile_run.log"

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"

log() {
  printf '[profile-matmul] %s %s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$*" >&2
}

die() {
  log "ERROR: $*"
  exit 1
}

have_command() {
  command -v "$1" >/dev/null 2>&1
}

quoted_command() {
  printf '%q ' "$@"
}

run_logged() {
  local output_log="$1"
  shift
  log "Running: $(quoted_command "$@")"
  "$@" > >(tee "${output_log}") 2>&1
}

run_logged_with_optional_timeout() {
  local timeout_seconds="$1"
  local output_log="$2"
  shift 2

  if have_command timeout; then
    run_logged "${output_log}" timeout "${timeout_seconds}" "$@"
  else
    log "timeout command not found; running without timeout"
    run_logged "${output_log}" "$@"
  fi
}

resolve_profile_python() {
  if [[ -n "${PROFILE_PYTHON}" ]]; then
    printf '%s\n' "${PROFILE_PYTHON}"
    return
  fi

  uv run python -c 'import sys; print(sys.executable)'
}

ensure_profile_python_bin_on_path() {
  local python_path="$1"
  local python_bin_dir
  python_bin_dir="$(dirname "${python_path}")"

  case ":${PATH}:" in
    *":${python_bin_dir}:"*) ;;
    *)
      export PATH="${python_bin_dir}:${PATH}"
      log "Prepended profile Python bin directory to PATH: ${python_bin_dir}"
      ;;
  esac
}

nsight_compute_profile_succeeded() {
  [[ -s "${ncu_report}" ]] && ! grep -qi "No kernels were profiled" "${ncu_log}"
}

print_tool_version() {
  local tool="$1"
  shift

  if have_command "${tool}"; then
    log "${tool}: $(command -v "${tool}")"
    "$@" 2>&1 | sed "s/^/[profile-matmul] ${tool} version: /" >&2 || true
  else
    log "${tool}: not found"
  fi
}

check_environment() {
  log "Checking profiler environment"
  log "shape=${shape_label} out_dir=${OUT_DIR}"
  log "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
  log "REQUIRE_NCU=${REQUIRE_NCU} NCU_TIMEOUT_SECONDS=${NCU_TIMEOUT_SECONDS}"

  print_tool_version nvidia-smi nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
  print_tool_version nvcc nvcc --version
  print_tool_version ncu ncu --version
  print_tool_version nsys nsys --version

  if have_command timeout; then
    log "timeout: $(command -v timeout)"
  else
    log "timeout: not found"
  fi

  uv run python -c 'import torch; print(f"[profile-matmul] torch={torch.__version__} cuda_available={torch.cuda.is_available()} cuda={torch.version.cuda}")' >&2
  profile_python="$(resolve_profile_python)"
  ensure_profile_python_bin_on_path "${profile_python}"
  log "profile_python=${profile_python}"
  print_tool_version ninja ninja --version

  if ! have_command ncu; then
    log "Nsight Compute CLI was not found. Strict NCU profiling will fail."
  fi
}

run_pytorch_profiler_fallback() {
  if [[ "${REQUIRE_NCU}" == "1" ]]; then
    die "REQUIRE_NCU=1, so fallback profiling is disabled."
  fi

  log "Falling back to PyTorch profiler timeline"
  run_logged "${torch_profile_log}" \
    uv run python -m forgenpu_kernels.cli.profile_matmul_torch \
      --shape "${M}" "${N}" "${K}" \
      --warmup "${PROFILE_WARMUP}" \
      --iterations "${PROFILE_ITERATIONS}" \
      --output-dir "${OUT_DIR}"
  log "PyTorch profiler run log: ${torch_profile_log}"
}

run_nsight_systems_profile() {
  if [[ "${REQUIRE_NCU}" == "1" ]]; then
    die "REQUIRE_NCU=1, so Nsight Systems fallback is disabled."
  fi

  if ! have_command nsys; then
    log "nsys not found; skipping Nsight Systems fallback"
    return 1
  fi

  log "Attempting Nsight Systems (nsys)"
  if run_logged "${nsys_log}" \
    nsys profile \
    --force-overwrite true \
    --output "${nsys_report_base}" \
    uv run forgenpu-bench-matmul \
      --implementation cuda_tiled \
      --device cuda \
      --shape "${M}" "${N}" "${K}" \
      --warmup "${PROFILE_WARMUP}" \
      --iterations "${PROFILE_ITERATIONS}" \
      --quiet; then
    log "PROFILE_RESULT=nsight_systems"
    log "Nsight Systems report: ${nsys_report}"
    log "Nsight Systems log: ${nsys_log}"
    return 0
  fi

  log "Nsight Systems failed. Log: ${nsys_log}"
  return 1
}

run_fallback_profile_chain() {
  if run_nsight_systems_profile; then
    return 0
  fi

  run_pytorch_profiler_fallback
}

if [[ "${CHECK_ONLY}" == "1" ]]; then
  check_environment
  exit 0
fi

check_environment

log "Capturing benchmark JSON before profiling"
uv run forgenpu-bench-matmul \
  --implementation all \
  --device cuda \
  --shape "${M}" "${N}" "${K}" \
  --warmup "${WARMUP}" \
  --iterations "${ITERATIONS}" \
  --format json \
  --quiet \
  --output "${benchmark_output}"
log "Benchmark JSON: ${benchmark_output}"

if command -v ncu >/dev/null 2>&1; then
  profile_python="$(resolve_profile_python)"
  ensure_profile_python_bin_on_path "${profile_python}"
  log "Smoke-testing dedicated Nsight target before ncu"
  run_logged "${target_smoke_log}" \
    "${profile_python}" -m forgenpu_kernels.cli.profile_matmul_ncu_target \
      --shape "${M}" "${N}" "${K}" \
      --warmup 1 \
      --iterations 1
  log "Nsight target smoke log: ${target_smoke_log}"

  ncu_command=(
    ncu
    --profile-from-start off \
    --launch-count 1 \
    --section LaunchStats \
    --section Occupancy \
    --force-overwrite \
    --export "${ncu_report_base}" \
    "${profile_python}" -m forgenpu_kernels.cli.profile_matmul_ncu_target \
      --shape "${M}" "${N}" "${K}" \
      --warmup "${PROFILE_WARMUP}" \
      --iterations "${PROFILE_ITERATIONS}" \
      --cuda-profiler-api
  )

  log "Attempting Nsight Compute (ncu)"
  if run_logged_with_optional_timeout "${NCU_TIMEOUT_SECONDS}" "${ncu_log}" "${ncu_command[@]}"; then
    if nsight_compute_profile_succeeded; then
      log "PROFILE_RESULT=nsight_compute"
      log "Nsight Compute report: ${ncu_report}"
      log "Nsight Compute log: ${ncu_log}"
    else
      log "ncu exited successfully, but no usable kernel report was produced."
      log "Expected report: ${ncu_report}"
      if grep -qi "No kernels were profiled" "${ncu_log}"; then
        log "ncu reported: No kernels were profiled."
      fi
      run_fallback_profile_chain
    fi
  else
    log "Nsight Compute failed or timed out. Log: ${ncu_log}"
    if grep -Eqi 'ERR_NVGPUCTRPERM|permission|No kernels were profiled' "${ncu_log}"; then
      log "ncu log suggests a profiler permission/counter/kernels issue; inspect ${ncu_log}"
    fi
    run_fallback_profile_chain
  fi
elif command -v nsys >/dev/null 2>&1; then
  if [[ "${REQUIRE_NCU}" == "1" ]]; then
    die "ncu was not found and REQUIRE_NCU=1."
  fi

  run_nsight_systems_profile || run_pytorch_profiler_fallback
else
  if [[ "${REQUIRE_NCU}" == "1" ]]; then
    die "ncu was not found and REQUIRE_NCU=1."
  fi

  log "No ncu or nsys executable found"
  fallback_output="${OUT_DIR}/matmul_tiled_${shape_label}_profile_fallback.txt"
  {
    echo "No ncu or nsys executable was found."
    echo "Benchmark evidence was still captured at: ${benchmark_output}"
    echo "Install Nsight Compute or Nsight Systems for hardware-counter profiling."
  } > "${fallback_output}"
  log "PROFILE_RESULT=no_profiler"
  log "Profiler fallback note: ${fallback_output}"
fi
