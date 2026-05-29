#!/usr/bin/env python
"""Emit benchmark machine metadata as JSON."""

from __future__ import annotations

import json

from forgenpu_kernels.benchmarks import machine_info_dict


def main() -> None:
    print(json.dumps(machine_info_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
