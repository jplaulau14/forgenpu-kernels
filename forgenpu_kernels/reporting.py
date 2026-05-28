"""Human-readable and machine-readable benchmark reporting helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


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


def format_table_row(cells: list[str], widths: list[int]) -> str:
    return "  ".join(cell.ljust(width) for cell, width in zip(cells, widths, strict=True))


def format_benchmark_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "implementation",
        "p50 ms",
        "p95 ms",
        "mean ms",
        "TFLOP/s",
        "vs torch",
        "vs naive",
        "max abs err",
        "est global GiB",
    ]
    table_rows = [
        [
            row["implementation"],
            format_float(row["p50_ms"]),
            format_float(row["p95_ms"]),
            format_float(row["mean_ms"]),
            format_float(row["achieved_tflops"]),
            format_float(row["speedup_vs_baseline"], digits=3),
            format_float(row["speedup_vs_naive"], digits=3),
            format_float(row["max_abs_error"], digits=2),
            format_bytes_as_gib(row["estimated_global_memory_bytes"]),
        ]
        for row in rows
    ]
    widths = [
        max(len(str(cell)) for cell in column)
        for column in zip(headers, *table_rows, strict=True)
    ]

    separator = "  ".join("-" * width for width in widths)
    lines = [format_table_row(headers, widths), separator]
    lines.extend(format_table_row(row, widths) for row in table_rows)
    return "\n".join(lines)


def normalize_result_rows(result: dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    return result if isinstance(result, list) else [result]


def write_benchmark_result(
    result: dict[str, Any] | list[dict[str, Any]],
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
        print(payload)
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload + "\n", encoding="utf-8")
