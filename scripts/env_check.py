#!/usr/bin/env python
"""Print environment facts needed for benchmark reports."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version


def package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def command_output(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def torch_info() -> dict[str, object]:
    try:
        import torch
    except ImportError:
        return {
            "installed": False,
            "version": None,
            "cuda_available": False,
            "cuda_version": None,
            "gpu": None,
            "gpu_memory_bytes": None,
        }

    cuda_available = bool(torch.cuda.is_available())
    gpu = None
    gpu_memory_bytes = None
    if cuda_available:
        device = torch.device("cuda")
        gpu = torch.cuda.get_device_name(device)
        gpu_memory_bytes = torch.cuda.get_device_properties(device).total_memory

    return {
        "installed": True,
        "version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "gpu": gpu,
        "gpu_memory_bytes": gpu_memory_bytes,
    }


def main() -> None:
    nvidia_smi = shutil.which("nvidia-smi")
    driver = None
    if nvidia_smi:
        driver = command_output([nvidia_smi, "--query-gpu=driver_version", "--format=csv,noheader"])
        if driver:
            driver = driver.splitlines()[0]

    cmake = None
    if shutil.which("cmake"):
        cmake_output = command_output(["cmake", "--version"])
        cmake = cmake_output.splitlines()[0] if cmake_output else None

    info = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch_info(),
        "triton": package_version("triton"),
        "nvidia_smi": nvidia_smi,
        "driver": driver,
        "cmake": cmake,
    }
    print(json.dumps(info, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
