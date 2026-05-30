param(
  [int]$M = 512,
  [int]$N = 512,
  [int]$K = 512,
  [int]$Warmup = 2,
  [int]$Iterations = 5,
  [string]$Implementation = "cuda_tiled",
  [string]$KernelName = "regex:matmul_tiled_kernel",
  [string]$OutputDir = "results\profiles",
  [string]$ArchList = "7.5"
)

$ErrorActionPreference = "Stop"

function Log-Step {
  param([string]$Message)
  $timestamp = Get-Date -Format "yyyy-MM-ddTHH:mm:ssK"
  Write-Host "[profile-matmul-ncu] $timestamp $Message"
}

function Require-Command {
  param([string]$Name)
  if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
    throw "Required command not found on PATH: $Name"
  }
}

function Resolve-NcuPath {
  $command = Get-Command "ncu" -ErrorAction SilentlyContinue
  if ($command) {
    return $command.Source
  }

  $searchRoots = @(
    "C:\Program Files\NVIDIA Corporation",
    "C:\Program Files\NVIDIA GPU Computing Toolkit"
  )

  foreach ($root in $searchRoots) {
    if (Test-Path $root) {
      $candidate = Get-ChildItem $root -Recurse -Filter "ncu.exe" -ErrorAction SilentlyContinue |
        Select-Object -First 1
      if ($candidate) {
        return $candidate.FullName
      }
    }
  }

  throw "Nsight Compute CLI not found. Install Nsight Compute or add the directory containing ncu.exe to PATH."
}

Require-Command "uv"
$ncuPath = Resolve-NcuPath

if (-not $env:TORCH_CUDA_ARCH_LIST) {
  $env:TORCH_CUDA_ARCH_LIST = $ArchList
}

New-Item -ItemType Directory -Force $OutputDir | Out-Null

$shapeLabel = "${M}x${N}x${K}"
$outputBase = Join-Path $OutputDir "matmul_${Implementation}_${shapeLabel}_ncu"

Log-Step "device=cuda shape=$shapeLabel implementation=$Implementation"
Log-Step "TORCH_CUDA_ARCH_LIST=$env:TORCH_CUDA_ARCH_LIST"
Log-Step "ncu=$ncuPath"
Log-Step "checking PyTorch CUDA"
uv run python -c "import torch; print(f'torch={torch.__version__} torch_cuda={torch.version.cuda} cuda_available={torch.cuda.is_available()} device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

Log-Step "running target smoke test"
uv run forgenpu-bench-matmul `
  --implementation $Implementation `
  --device cuda `
  --shape $M $N $K `
  --warmup 1 `
  --iterations 1 `
  --quiet

Log-Step "running Nsight Compute"
& $ncuPath `
  --target-processes all `
  --kernel-name $KernelName `
  --launch-count 1 `
  --set full `
  --force-overwrite `
  --export $outputBase `
  uv run forgenpu-bench-matmul `
    --implementation $Implementation `
    --device cuda `
    --shape $M $N $K `
    --warmup $Warmup `
    --iterations $Iterations `
    --quiet

Log-Step "PROFILE_RESULT=nsight_compute"
Log-Step "Nsight Compute report: $outputBase.ncu-rep"
