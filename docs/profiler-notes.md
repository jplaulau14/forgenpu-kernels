# Profiler Notes

M2 adds the first profiler entry point:

```bash
scripts/profile_matmul.sh 1024 1024 1024
```

The script captures a benchmark JSON file and then tries profiler tools in this order:

1. Nsight Compute (`ncu`) with launch statistics and occupancy sections, time-boxed by `NCU_TIMEOUT_SECONDS`.
2. Nsight Systems (`nsys`) timeline capture.
3. PyTorch profiler trace if Nsight Compute does not complete.
4. A fallback text note if no profiler executable is installed.

Raw `.ncu-rep`, `.nsys-rep`, and generated benchmark JSON files are ignored by git because they are machine-specific. Curated observations belong in this file, `docs/roofline.md`, or a committed summary under `results/profiles/`.

The script keeps the default profiler pass intentionally light. Full roofline counter collection can be run manually with Nsight Compute after the basic profiler path is stable. In containers where `ncu` hangs or cannot collect counters, the PyTorch profiler fallback still records a CUDA timeline trace.

## M2 Expected Bottleneck

The M1 naive matmul rereads `A` and `B` values from global memory for every output element. The M2 tiled implementation should reduce global-memory traffic by staging `16 x 16` tiles in shared memory.

The expected result is:

- `cuda_tiled` improves over `cuda_naive` on large enough FP32 shapes.
- `cuda_tiled` still loses to `torch.matmul`.
- PyTorch remains faster because it uses deeper optimizations such as Tensor Cores, register blocking, vectorized memory access, tuned schedules, and architecture-specific kernels.

## M2 Profiler Questions

The first profiler artifact should answer:

- Did the tiled kernel run as the measured kernel?
- What was the kernel duration for the selected shape?
- How much shared memory did the kernel allocate per block?
- Did the profiler identify memory throughput, occupancy, or compute utilization as the limiting factor?
- Does the profiler evidence match the benchmark story?

## M2 RunPod Artifact

The first committed profiler summary is:

- `results/profiles/m2-runpod-h100.md`

The RunPod container exposed `ncu`, but the scripted Nsight Compute pass did not profile a kernel before timeout. The profiling workflow then fell back to PyTorch profiler and generated a CUDA timeline trace plus text summary.

### What This Artifact Proves

- The benchmark path ran on a CUDA-capable H100 environment.
- The `cuda_tiled` extension built and launched.
- The profiler observed `matmul_tiled_kernel` in the CUDA timeline.
- The measured `cuda_tiled` latency improved over `cuda_naive` for the selected `1024 x 1024 x 1024` shape.

### What This Artifact Does Not Prove Yet

- It does not include Nsight Compute hardware-counter evidence for achieved memory bandwidth.
- It does not prove occupancy, register pressure, or warp stall reasons.
- It does not prove that global-memory traffic matched the explanatory estimates in `docs/roofline.md`.

Those claims require a successful Nsight Compute counter run. Until then, M2 should describe memory traffic numbers as estimates and profiler evidence as CUDA timeline evidence.

For one `1024 x 1024 x 1024` `cuda_tiled` launch, PyTorch profiler reported:

```text
matmul_tiled_kernel CUDA total: 273.852us
cudaLaunchKernel CPU total: 890.250us
```

The benchmark evidence showed `cuda_tiled` improving over `cuda_naive`, while still trailing `torch.matmul`. That matches the M2 expectation: shared-memory tiling helps, but the implementation is still far below vendor matmul kernels.
