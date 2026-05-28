#!/usr/bin/env python
"""Emit benchmark machine metadata as JSON."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forgenpu_kernels.benchmarks import machine_info_dict  # noqa: E402


def main() -> None:
    print(json.dumps(machine_info_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
