#!/usr/bin/env python
"""Compatibility wrapper for the packaged Typer matmul benchmark CLI."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forgenpu_kernels.cli.bench_matmul import main  # noqa: E402


if __name__ == "__main__":
    main()
