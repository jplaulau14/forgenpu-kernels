"""Typer CLI for matmul benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.text import Text

from forgenpu_kernels.matmul_benchmark import (
    MatmulBenchmarkConfig,
    result_payload,
    run_matmul_benchmark,
)
from forgenpu_kernels.reporting import write_benchmark_result


class Device(str, Enum):
    auto = "auto"
    cpu = "cpu"
    cuda = "cuda"


class DType(str, Enum):
    float32 = "float32"
    float16 = "float16"
    bfloat16 = "bfloat16"


class Implementation(str, Enum):
    torch = "torch"
    cuda_naive = "cuda_naive"
    cuda_tiled = "cuda_tiled"
    all = "all"


class OutputFormat(str, Enum):
    json = "json"
    table = "table"


@dataclass(frozen=True)
class ProgressLogger:
    quiet: bool
    console: Console

    def __call__(self, message: str) -> None:
        if not self.quiet:
            self.console.print(Text.assemble(("[bench]", "bold cyan"), " ", message))


def benchmark(
    shape: Annotated[
        tuple[int, int, int],
        typer.Option(
            "--shape",
            metavar="M N K",
            help="Matmul shape: A is MxK, B is KxN, output is MxN.",
        ),
    ] = (512, 512, 512),
    warmup: Annotated[int, typer.Option(help="Warmup iterations before timing.")] = 25,
    iterations: Annotated[int, typer.Option(help="Timed benchmark iterations.")] = 100,
    device: Annotated[Device, typer.Option(help="Execution device.")] = Device.auto,
    dtype: Annotated[DType, typer.Option(help="Tensor dtype.")] = DType.float32,
    implementation: Annotated[
        Implementation,
        typer.Option(help="Implementation to benchmark."),
    ] = Implementation.torch,
    output_format: Annotated[
        OutputFormat,
        typer.Option("--format", help="Output format. Use table for humans and JSON for scripts."),
    ] = OutputFormat.json,
    quiet: Annotated[bool, typer.Option(help="Suppress progress logs on stderr.")] = False,
    output: Annotated[Path | None, typer.Option(help="Optional output file.")] = None,
) -> None:
    """Run a matmul benchmark and print JSON or a readable table."""
    config = MatmulBenchmarkConfig(
        shape=shape,
        warmup=warmup,
        iterations=iterations,
        device=device.value,
        dtype=dtype.value,
        implementation=implementation.value,
    )
    logger = ProgressLogger(quiet=quiet, console=Console(stderr=True, highlight=False))

    try:
        results = run_matmul_benchmark(config, progress=logger)
    except (RuntimeError, ValueError) as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(1) from exc

    write_benchmark_result(result_payload(results), output, output_format.value)
    logger("done")


def main() -> None:
    typer.run(benchmark)
