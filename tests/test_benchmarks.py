from __future__ import annotations

import subprocess
from pathlib import Path

import forgenpu_kernels.benchmarks as benchmarks


def test_current_commit_is_resolved_from_package_repo_root(tmp_path, monkeypatch) -> None:
    repo_root = Path(benchmarks.__file__).resolve().parents[1]
    expected = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    monkeypatch.chdir(tmp_path)

    assert benchmarks.current_commit() == expected
