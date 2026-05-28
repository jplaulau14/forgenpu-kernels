from __future__ import annotations

from forgenpu_kernels.matmul_benchmark import (
    matmul_workload_metrics,
    result_payload,
    selected_implementations,
)


def test_selected_implementations_expands_all_in_stable_order() -> None:
    assert selected_implementations("all") == ("torch", "cuda_naive", "cuda_tiled")
    assert selected_implementations("cuda_tiled") == ("cuda_tiled",)


def test_matmul_workload_metrics_estimates_naive_and_tiled_global_bytes() -> None:
    shape = (1024, 1024, 1024)

    naive = matmul_workload_metrics(implementation="cuda_naive", shape=shape, p50_ms=1.0)
    tiled = matmul_workload_metrics(implementation="cuda_tiled", shape=shape, p50_ms=1.0)
    torch = matmul_workload_metrics(implementation="torch", shape=shape, p50_ms=1.0)

    assert naive["estimated_global_memory_bytes"] == 8_594_128_896
    assert tiled["estimated_global_memory_bytes"] == 541_065_216
    assert torch["estimated_global_memory_bytes"] is None
    assert tiled["estimated_global_memory_bytes"] < naive["estimated_global_memory_bytes"]


def test_result_payload_unwraps_single_result_only() -> None:
    one = [{"implementation": "torch"}]
    many = [{"implementation": "torch"}, {"implementation": "cuda_naive"}]

    assert result_payload(one) == {"implementation": "torch"}
    assert result_payload(many) == many
