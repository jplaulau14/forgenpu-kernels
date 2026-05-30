"""Human-readable and machine-readable benchmark reporting helpers."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

from rich import box
from rich.console import Console
from rich.table import Table

from forgenpu_kernels.matmul_benchmark import MatmulBenchmarkResult


def format_float(value: object, *, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def format_bytes_as_gib(value: object) -> str:
    if value is None:
        return "-"
    return f"{int(value) / (1024**3):.3f}"


def build_benchmark_table(rows: list[MatmulBenchmarkResult]) -> Table:
    table = Table(
        title="Matmul benchmark",
        box=box.ASCII,
        show_lines=False,
        header_style="bold",
        title_style="bold cyan",
    )
    table.add_column("implementation")
    table.add_column("input")
    table.add_column("accum")
    table.add_column("output")
    table.add_column("p50 ms", justify="right")
    table.add_column("p95 ms", justify="right")
    table.add_column("mean ms", justify="right")
    table.add_column("TFLOP/s", justify="right")
    table.add_column("vs torch", justify="right")
    table.add_column("vs naive", justify="right")
    table.add_column("max abs err", justify="right")
    table.add_column("est global GiB", justify="right")

    for row in rows:
        table.add_row(
            str(row["implementation"]),
            str(row.get("input_dtype", row.get("dtype", "-"))),
            str(row.get("accumulation_dtype", "-")),
            str(row.get("output_dtype", row.get("dtype", "-"))),
            format_float(row["p50_ms"]),
            format_float(row["p95_ms"]),
            format_float(row["mean_ms"]),
            format_float(row["achieved_tflops"]),
            format_float(row["speedup_vs_baseline"], digits=3),
            format_float(row["speedup_vs_naive"], digits=3),
            format_float(row["max_abs_error"], digits=2),
            format_bytes_as_gib(row["estimated_global_memory_bytes"]),
        )
    return table


def format_benchmark_table(rows: list[MatmulBenchmarkResult]) -> str:
    buffer = StringIO()
    console = Console(
        color_system=None,
        file=buffer,
        force_terminal=False,
        highlight=False,
        width=140,
    )
    console.print(build_benchmark_table(rows))
    return buffer.getvalue().rstrip()


def normalize_result_rows(
    result: MatmulBenchmarkResult | list[MatmulBenchmarkResult],
) -> list[MatmulBenchmarkResult]:
    return result if isinstance(result, list) else [result]


def write_benchmark_result(
    result: MatmulBenchmarkResult | list[MatmulBenchmarkResult],
    output: Path | None,
    output_format: str,
) -> None:
    if output_format == "json":
        payload = json.dumps(result, indent=2, sort_keys=True)
    elif output_format == "table":
        payload = format_benchmark_table(normalize_result_rows(result))
    else:
        raise ValueError(f"unknown output format: {output_format}")

    if output is None:
        print(payload, flush=True)
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload + "\n", encoding="utf-8")
