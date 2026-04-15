#!/usr/bin/env bash
# Shared CUDA environment setup for GPU CI and bench workflows.
# Finds the CUDA toolkit, detects the GPU's compute capability, and
# exports env vars that candle-kernels/build.rs needs to compile WMMA
# kernels to matching SASS.
#
# Exports (via $GITHUB_ENV / $GITHUB_PATH):
#   PATH            — prepends <cuda>/bin
#   CUDA_PATH       — root of the CUDA installation
#   LIBRARY_PATH    — <cuda>/lib64
#   LD_LIBRARY_PATH — <cuda>/lib64
#   INFERRS_WMMA_ARCH  — numeric compute cap (e.g. 120)
#   CUDA_COMPUTE_CAP   — same value, used by bindgen_cuda
set -euo pipefail

# ---------- Find CUDA installation ----------
CUDA_FOUND=""
for d in /usr/local/cuda /opt/cuda /usr; do
  if [ -x "$d/bin/nvcc" ]; then
    echo "Found CUDA at $d"
    echo "$d/bin" >> "$GITHUB_PATH"
    echo "CUDA_PATH=$d" >> "$GITHUB_ENV"
    echo "LIBRARY_PATH=$d/lib64:${LIBRARY_PATH:-}" >> "$GITHUB_ENV"
    echo "LD_LIBRARY_PATH=$d/lib64:${LD_LIBRARY_PATH:-}" >> "$GITHUB_ENV"
    CUDA_FOUND="$d"
    break
  fi
done
if [ -z "$CUDA_FOUND" ]; then
  echo "::error::No CUDA installation found (checked /usr/local/cuda, /opt/cuda, /usr)"
  exit 1
fi
"$CUDA_FOUND/bin/nvcc" --version

# ---------- Detect GPU compute capability ----------
RAW=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
echo "Detected GPU compute capability: sm_${RAW}"
echo "INFERRS_WMMA_ARCH=${RAW}" >> "$GITHUB_ENV"
echo "CUDA_COMPUTE_CAP=${RAW}" >> "$GITHUB_ENV"

# ---------- GPU info ----------
nvidia-smi
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv

# ---------- Blackwell nvcc version check ----------
# sm_120+ requires CUDA toolkit >= 12.8.
NVCC_VER=$("$CUDA_FOUND/bin/nvcc" --version | grep -o 'release [0-9]*\.[0-9]*' | awk '{print $2}')
echo "nvcc version: ${NVCC_VER}"
if [ "${RAW}" -ge 120 ]; then
  MAJOR=$(echo "$NVCC_VER" | cut -d. -f1)
  MINOR=$(echo "$NVCC_VER" | cut -d. -f2)
  if [ "$MAJOR" -lt 12 ] || { [ "$MAJOR" -eq 12 ] && [ "$MINOR" -lt 8 ]; }; then
    echo "::error::sm_${RAW} (Blackwell) requires CUDA >= 12.8 but found ${NVCC_VER}"
    exit 1
  fi
fi
