# Profiler Notes

M2 adds the first profiler entry point:

```bash
scripts/profile_matmul.sh 1024 1024 1024
```

Before running the full profile, check tool availability:

```bash
make profile-check
```

If `profile-check` reports `ncu` but `nsys: not found`, the CUDA toolkit includes Nsight Compute but the image is missing Nsight Systems. On Ubuntu-based GPU containers with root or sudo access, install the Nsight Systems CLI with:

```bash
make install-nsight-systems
which nsys
nsys --version
```

If package installation is blocked by the provider, use a GPU image with Nsight Systems preinstalled or let the script fall back to the PyTorch profiler.

To test Nsight Systems directly:

```bash
make profile-matmul-nsys
```

This sets `PROFILE_TOOL=nsys` and requires an `.nsys-rep` capture. For manual runs, the accepted values are:

```bash
PROFILE_TOOL=auto scripts/profile_matmul.sh 1024 1024 1024
PROFILE_TOOL=ncu scripts/profile_matmul.sh 1024 1024 1024
PROFILE_TOOL=nsys scripts/profile_matmul.sh 1024 1024 1024
PROFILE_TOOL=torch scripts/profile_matmul.sh 1024 1024 1024
```

If `nsys` prints an importer error like:

```text
Importer error status: The importer binary and its dependencies were not found.
Unable to retrieve the importer version: skipping importation of the QDSTRM file.
Generated:
    results/profiles/matmul_tiled_1024x1024x1024_nsys.qdstrm
```

then Nsight Systems captured a raw `.qdstrm` stream but did not produce the final `.nsys-rep` report. The script records this as:

```text
PROFILE_RESULT=nsight_systems_raw_stream
```

This is partial evidence: the profiler launched and captured a timeline stream, but the local container lacks the importer needed to turn it into a normal report. Install the full Nsight Systems package/importer dependencies, or copy the `.qdstrm` to a machine with Nsight Systems and import/open it there.

When Nsight Compute evidence is required, use strict mode:

```bash
make profile-matmul-ncu
```

This fails instead of falling back if `ncu` is missing, times out, or does not produce a `.ncu-rep` report.

The script captures a benchmark JSON file, smoke-tests the dedicated profiling target, and then tries profiler tools in this order:

1. Nsight Compute (`ncu`) with launch statistics and occupancy sections, time-boxed by `NCU_TIMEOUT_SECONDS`.
2. Nsight Systems (`nsys`) timeline capture if `ncu` is unavailable or does not produce a usable kernel report.
3. PyTorch profiler trace if the Nsight tools do not produce a usable report.
4. A fallback text note if no profiler executable is installed.

The Nsight Compute path profiles a small dedicated target:

```bash
uv run python -m forgenpu_kernels.cli.profile_matmul_ncu_target
```

This avoids profiling the full benchmark harness. The shell script resolves the virtualenv Python executable first and runs `ncu` directly against Python instead of profiling the `uv` launcher. The target warms up the tiled kernel, calls `cudaProfilerStart`, runs the measured tiled launches, then calls `cudaProfilerStop`. The `ncu` command uses `--profile-from-start off` so setup kernels from PyTorch tensor allocation and extension loading are not collected.

Raw `.ncu-rep`, `.nsys-rep`, and generated benchmark JSON files are ignored by git because they are machine-specific. Curated observations belong in this file, `docs/roofline.md`, or a committed summary under `results/profiles/`.

The script keeps the default profiler pass intentionally light. Full roofline counter collection can be run manually with Nsight Compute after the basic profiler path is stable. In containers where `ncu` hangs or cannot collect counters, the PyTorch profiler fallback still records a CUDA timeline trace.

## Benchmark Requirement For Profiling

Every profiler artifact should be paired with a benchmark run on the same device, shape, dtype, commit, and implementation. The benchmark gives the timing story; the profiler explains the kernel behavior behind that timing.

Do not use hardware-counter evidence from one GPU to make definitive claims about timings collected on another GPU. Cross-device comparisons are useful for learning, but they should be labeled as hypotheses unless benchmark and profiler evidence were collected on the same hardware.

During M3, `scripts/profile_matmul.sh` still profiles `cuda_tiled`, not the Triton kernel. Triton performance claims should therefore cite benchmark artifacts such as `results/profiles/m3-h100-triton.md` until a Triton-specific profiler target exists.

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
