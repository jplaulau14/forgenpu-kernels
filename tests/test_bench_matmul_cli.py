from __future__ import annotations

import shutil
import subprocess

from forgenpu_kernels.reporting import format_benchmark_table


def test_format_table_includes_key_benchmark_columns() -> None:
    rows = [
        {
            "implementation": "torch",
            "p50_ms": 0.0682239979505539,
            "p95_ms": 0.07234879955649376,
            "mean_ms": 0.0688441601395607,
            "achieved_tflops": 31.476954041251183,
            "speedup_vs_baseline": 1.0,
            "speedup_vs_naive": 6.330441078665527,
            "max_abs_error": 0.0,
            "estimated_global_memory_bytes": None,
        },
        {
            "implementation": "cuda_tiled",
            "p50_ms": 0.2807680070400238,
            "p95_ms": 0.2825647920370102,
            "mean_ms": 0.2809456020593643,
            "achieved_tflops": 7.648605233337265,
            "speedup_vs_baseline": 0.24299064081339042,
            "speedup_vs_naive": 1.5382379343363468,
            "max_abs_error": 0.0,
            "estimated_global_memory_bytes": 541065216,
        },
    ]

    table = format_benchmark_table(rows)

    assert "implementation" in table
    assert "TFLOP/s" in table
    assert "vs torch" in table
    assert "vs naive" in table
    assert "est global GiB" in table
    assert "cuda_tiled" in table
    assert "1.538" in table
    assert "0.504" in table


def test_typer_cli_help_is_direct_command() -> None:
    command = shutil.which("forgenpu-bench-matmul") or "forgenpu-bench-matmul"

    completed = subprocess.run(
        [command, "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Usage: forgenpu-bench-matmul [OPTIONS]" in completed.stdout
    assert "COMMAND [ARGS]" not in completed.stdout
    assert "--implementation" in completed.stdout
    assert "--format" in completed.stdout


def test_typer_cli_table_smoke() -> None:
    command = shutil.which("forgenpu-bench-matmul") or "forgenpu-bench-matmul"

    completed = subprocess.run(
        [
            command,
            "--implementation",
            "torch",
            "--device",
            "cpu",
            "--shape",
            "8",
            "8",
            "8",
            "--warmup",
            "0",
            "--iterations",
            "1",
            "--format",
            "table",
            "--quiet",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Matmul benchmark" in completed.stdout
    assert "implementation" in completed.stdout
    assert "torch" in completed.stdout
    assert completed.stderr == ""
