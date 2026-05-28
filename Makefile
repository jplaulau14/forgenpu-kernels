.PHONY: help quickstart env test lint bench-matmul bench-matmul-table bench-matmul-gpu profile-matmul configure-cpp build-cpp clean

help:
	@echo "ForgeNPU quick commands"
	@echo "  make quickstart        - run env, lint, tests, and a CPU table benchmark"
	@echo "  make env               - print Python/PyTorch/CUDA environment facts"
	@echo "  make lint              - run ruff"
	@echo "  make test              - run pytest"
	@echo "  make bench-matmul      - run a JSON CPU/PyTorch matmul benchmark"
	@echo "  make bench-matmul-table - run a readable CPU/PyTorch matmul benchmark"
	@echo "  make bench-matmul-gpu  - run M2 CUDA comparison on a GPU machine"
	@echo "  make profile-matmul    - run M2 profiler workflow on a GPU machine"

quickstart: env lint test bench-matmul-table

env:
	uv run python scripts/env_check.py

lint:
	uv run --extra dev ruff check .

test:
	uv run --extra dev pytest

bench-matmul:
	uv run forgenpu-bench-matmul --shape 512 512 512 --warmup 5 --iterations 20 --quiet

bench-matmul-table:
	uv run forgenpu-bench-matmul --implementation torch --device auto --shape 512 512 512 --warmup 5 --iterations 20 --format table

bench-matmul-gpu:
	uv run forgenpu-bench-matmul --implementation all --device cuda --shape 1024 1024 1024 --warmup 25 --iterations 100 --format table

profile-matmul:
	scripts/profile_matmul.sh 1024 1024 1024

configure-cpp:
	cmake -S . -B build

build-cpp: configure-cpp
	cmake --build build

clean:
	rm -rf build .pytest_cache .ruff_cache
