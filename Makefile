.PHONY: env test bench-matmul configure-cpp build-cpp clean

env:
	uv run python scripts/env_check.py

test:
	uv run --extra dev pytest

bench-matmul:
	uv run python benchmarks/bench_matmul.py --shape 512 512 512 --warmup 5 --iterations 20

configure-cpp:
	cmake -S . -B build

build-cpp: configure-cpp
	cmake --build build

clean:
	rm -rf build .pytest_cache .ruff_cache
