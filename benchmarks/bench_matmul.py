#!/usr/bin/env python
"""Compatibility wrapper for the packaged Typer matmul benchmark CLI."""

from __future__ import annotations

from forgenpu_kernels.cli.bench_matmul import main


if __name__ == "__main__":
    main()
