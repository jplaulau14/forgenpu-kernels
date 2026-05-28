"""Benchmark utilities shared by operator benchmark scripts."""

from __future__ import annotations

import json
import platform
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for benchmarks. Install the correct PyTorch build "
            "for your CPU/CUDA environment."
        ) from exc
    return torch


@dataclass(frozen=True)
class MachineInfo:
    python: str
    platform: str
    processor: str
    torch: str | None
    cuda_available: bool
    cuda_version: str | None
    device: str
    driver: str | None
    triton: str | None
    commit: str | None


def package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def current_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def nvidia_driver_version() -> str | None:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip().splitlines()[0]


def collect_machine_info(device: str | None = None) -> MachineInfo:
    torch = _require_torch()
    cuda_available = bool(torch.cuda.is_available())
    selected_device = device or ("cuda" if cuda_available else "cpu")

    if selected_device.startswith("cuda") and cuda_available:
        device_name = torch.cuda.get_device_name(torch.device(selected_device))
    else:
        device_name = "cpu"

    return MachineInfo(
        python=platform.python_version(),
        platform=platform.platform(),
        processor=platform.processor(),
        torch=torch.__version__,
        cuda_available=cuda_available,
        cuda_version=torch.version.cuda,
        device=device_name,
        driver=nvidia_driver_version(),
        triton=package_version("triton"),
        commit=current_commit(),
    )


def percentile(values: list[float], pct: float) -> float:
    if not values:
        raise ValueError("cannot compute percentile for empty values")
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def benchmark_torch_callable(
    fn: Callable[[], Any],
    *,
    warmup: int,
    iterations: int,
    device: str,
) -> dict[str, float]:
    torch = _require_torch()
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    uses_cuda = device.startswith("cuda") and torch.cuda.is_available()

    for _ in range(warmup):
        fn()
    if uses_cuda:
        torch.cuda.synchronize()

    timings_ms: list[float] = []
    for _ in range(iterations):
        if uses_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            timings_ms.append(float(start.elapsed_time(end)))
        else:
            start_ns = time.perf_counter_ns()
            fn()
            end_ns = time.perf_counter_ns()
            timings_ms.append((end_ns - start_ns) / 1_000_000)

    return {
        "p50_ms": percentile(timings_ms, 0.50),
        "p95_ms": percentile(timings_ms, 0.95),
        "mean_ms": statistics.fmean(timings_ms),
    }


def write_json_result(result: dict[str, Any], output: Path | None) -> None:
    payload = json.dumps(result, indent=2, sort_keys=True)
    if output is None:
        print(payload)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload + "\n", encoding="utf-8")


def machine_info_dict(device: str | None = None) -> dict[str, Any]:
    return asdict(collect_machine_info(device=device))
