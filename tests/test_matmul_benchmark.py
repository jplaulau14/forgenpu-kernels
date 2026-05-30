from __future__ import annotations

import pytest

import forgenpu_kernels.matmul_benchmark as matmul_benchmark
from forgenpu_kernels.matmul_benchmark import (
    MatmulBenchmarkConfig,
    accumulation_dtype_name,
    dtype_element_size,
    matmul_workload_metrics,
    result_payload,
    resolve_device,
    selected_implementations,
    validate_config,
)


def test_selected_implementations_expands_all_with_triton_when_available(monkeypatch) -> None:
    monkeypatch.setattr(matmul_benchmark, "has_cuda_matmul_naive", lambda **_: True)
    monkeypatch.setattr(matmul_benchmark, "has_cuda_matmul_tiled", lambda **_: True)
    monkeypatch.setattr(matmul_benchmark, "has_cuda_matmul_wmma", lambda **_: True)
    monkeypatch.setattr(matmul_benchmark, "has_triton_matmul", lambda: True)

    assert selected_implementations("all", device="cuda") == (
        "torch",
        "cuda_naive",
        "cuda_tiled",
        "triton",
    )
    assert selected_implementations("all", device="cuda", dtype_name="float16") == (
        "torch",
        "cuda_wmma",
    )
    assert selected_implementations("all", device="cuda", dtype_name="bfloat16") == ("torch",)
    assert selected_implementations("cuda_tiled") == ("cuda_tiled",)


def test_selected_implementations_skips_triton_when_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(matmul_benchmark, "has_cuda_matmul_naive", lambda **_: True)
    monkeypatch.setattr(matmul_benchmark, "has_cuda_matmul_tiled", lambda **_: True)
    monkeypatch.setattr(matmul_benchmark, "has_cuda_matmul_wmma", lambda **_: True)
    monkeypatch.setattr(matmul_benchmark, "has_triton_matmul", lambda: False)

    assert selected_implementations("all", device="cuda") == ("torch", "cuda_naive", "cuda_tiled")


def test_selected_implementations_skips_unavailable_cuda_extensions(monkeypatch) -> None:
    monkeypatch.setattr(matmul_benchmark, "has_cuda_matmul_naive", lambda **_: False)
    monkeypatch.setattr(matmul_benchmark, "has_cuda_matmul_tiled", lambda **_: False)
    monkeypatch.setattr(matmul_benchmark, "has_cuda_matmul_wmma", lambda **_: False)
    monkeypatch.setattr(matmul_benchmark, "has_triton_matmul", lambda: True)

    assert selected_implementations("all", device="cuda") == ("torch", "triton")
    assert selected_implementations("all", device="cuda", dtype_name="float16") == ("torch",)


def test_selected_implementations_cpu_all_is_torch_only(monkeypatch) -> None:
    monkeypatch.setattr(matmul_benchmark, "has_cuda_matmul_naive", lambda **_: True)
    monkeypatch.setattr(matmul_benchmark, "has_cuda_matmul_tiled", lambda **_: True)
    monkeypatch.setattr(matmul_benchmark, "has_cuda_matmul_wmma", lambda **_: True)
    monkeypatch.setattr(matmul_benchmark, "has_triton_matmul", lambda: True)

    assert selected_implementations("all", device="cpu") == ("torch",)


def test_matmul_workload_metrics_estimates_naive_and_tiled_global_bytes() -> None:
    shape = (1024, 1024, 1024)

    naive = matmul_workload_metrics(implementation="cuda_naive", shape=shape, p50_ms=1.0)
    tiled = matmul_workload_metrics(implementation="cuda_tiled", shape=shape, p50_ms=1.0)
    triton = matmul_workload_metrics(implementation="triton", shape=shape, p50_ms=1.0)
    wmma = matmul_workload_metrics(
        implementation="cuda_wmma",
        shape=shape,
        p50_ms=1.0,
        dtype_name="float16",
        output_dtype_name="float32",
    )
    torch = matmul_workload_metrics(implementation="torch", shape=shape, p50_ms=1.0)

    assert naive["estimated_global_memory_bytes"] == 8_594_128_896
    assert tiled["estimated_global_memory_bytes"] == 541_065_216
    assert triton["estimated_global_memory_bytes"] == tiled["estimated_global_memory_bytes"]
    assert wmma["estimated_global_memory_bytes"] < tiled["estimated_global_memory_bytes"]
    assert torch["estimated_global_memory_bytes"] is None
    assert tiled["estimated_global_memory_bytes"] < naive["estimated_global_memory_bytes"]


def test_matmul_workload_metrics_uses_dtype_element_size() -> None:
    shape = (1024, 1024, 1024)

    fp32 = matmul_workload_metrics(
        implementation="torch", shape=shape, p50_ms=1.0, dtype_name="float32"
    )
    fp16 = matmul_workload_metrics(
        implementation="torch", shape=shape, p50_ms=1.0, dtype_name="float16"
    )
    bf16 = matmul_workload_metrics(
        implementation="torch", shape=shape, p50_ms=1.0, dtype_name="bfloat16"
    )

    assert dtype_element_size("float32") == 4
    assert dtype_element_size("float16") == 2
    assert dtype_element_size("bfloat16") == 2
    assert fp16["compulsory_io_bytes"] == fp32["compulsory_io_bytes"] // 2
    assert bf16["compulsory_io_bytes"] == fp16["compulsory_io_bytes"]
    assert fp16["arithmetic_intensity_flop_per_byte"] == (
        2 * fp32["arithmetic_intensity_flop_per_byte"]
    )


def test_wmma_workload_metrics_use_padded_kernel_dimensions_for_odd_shapes() -> None:
    shape = (31, 47, 19)

    tiled = matmul_workload_metrics(
        implementation="cuda_tiled", shape=shape, p50_ms=1.0, dtype_name="float16"
    )
    wmma = matmul_workload_metrics(
        implementation="cuda_wmma",
        shape=shape,
        p50_ms=1.0,
        dtype_name="float16",
        output_dtype_name="float32",
    )

    assert wmma["estimated_global_memory_bytes"] > tiled["estimated_global_memory_bytes"]


def test_accumulation_dtype_names_are_explicit() -> None:
    assert accumulation_dtype_name("cuda_naive", "float32") == "float32"
    assert accumulation_dtype_name("cuda_wmma", "float16") == "float32"
    assert accumulation_dtype_name("torch", "float16") == "implementation_defined"


def test_result_payload_unwraps_single_result_only() -> None:
    one = [{"implementation": "torch"}]
    many = [{"implementation": "torch"}, {"implementation": "cuda_naive"}]

    assert result_payload(one) == {"implementation": "torch"}
    assert result_payload(many) == many


def test_validate_config_rejects_non_positive_shape_and_iterations() -> None:
    valid = MatmulBenchmarkConfig(
        shape=(8, 8, 8),
        warmup=0,
        iterations=1,
        device="cpu",
        dtype="float32",
        implementation="torch",
    )
    validate_config(valid)

    invalid_shape = MatmulBenchmarkConfig(
        shape=(8, 0, 8),
        warmup=0,
        iterations=1,
        device="cpu",
        dtype="float32",
        implementation="torch",
    )
    invalid_iterations = MatmulBenchmarkConfig(
        shape=(8, 8, 8),
        warmup=0,
        iterations=0,
        device="cpu",
        dtype="float32",
        implementation="torch",
    )

    with pytest.raises(RuntimeError, match="shape values must be positive"):
        validate_config(invalid_shape)
    with pytest.raises(RuntimeError, match="iterations must be positive"):
        validate_config(invalid_iterations)


def test_resolve_device_rejects_explicit_cuda_when_pytorch_cuda_is_unavailable(monkeypatch) -> None:
    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class FakeTorch:
        cuda = FakeCuda()

    monkeypatch.setattr(matmul_benchmark, "require_torch", lambda: FakeTorch)

    with pytest.raises(RuntimeError, match="PyTorch CUDA is not available"):
        resolve_device("cuda")
