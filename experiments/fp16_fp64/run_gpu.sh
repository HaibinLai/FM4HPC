#!/usr/bin/env bash
# PREPARED ONLY: not run/compiled in this environment.
# Requires an isolated, complete OzBLAS checkout at the pinned revision.
# Run on an allocated compute node with CUDA supporting sm_70.
set -euo pipefail
oz_repo=${1:?Usage: bash run_gpu.sh FULL_OZBLAS_CHECKOUT INPUT.bin OUTPUT_PREFIX}
fixture=${2:?Missing input fixture}
out_prefix=${3:?Missing output prefix}
script_dir=$(cd -- "$(dirname -- "$0")" && pwd)
oz_repo=$(cd -- "$oz_repo" && pwd)
expected=a6f6198bc7af6cf288607ef867693f991cf48862
test "$(git -C "$oz_repo" rev-parse HEAD)" = "$expected" || { echo 'Wrong source commit' >&2; exit 1; }
test -z "$(git -C "$oz_repo" status --porcelain --untracked-files=no)" || { echo 'Checkout has tracked modifications; use an isolated clean checkout' >&2; exit 1; }
test -z "${LD_PRELOAD:-}" || { echo 'LD_PRELOAD must be empty to keep the cuBLAS baseline independent' >&2; exit 1; }
nvcc_path=$(command -v nvcc)
cuda_root=$(cd -- "$(dirname -- "$nvcc_path")/.." && pwd)
"$nvcc_path" --list-gpu-code | grep -q sm_70 || { echo 'This toolkit cannot compile sm_70; select a compatible CUDA toolkit' >&2; exit 1; }
mkdir -p -- "$(dirname -- "$out_prefix")"
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv > "$out_prefix.environment.txt"
"$nvcc_path" --version >> "$out_prefix.environment.txt"
git -C "$oz_repo" rev-parse HEAD >> "$out_prefix.environment.txt"
# Force rebuilding upstream tracked .o files; never reuse architecture-specific binaries.
make -B -C "$oz_repo/src/cuda" -j2 CUDA_PATH="$cuda_root" \
  AXPY=no DOT=no NRM2=no GEMV=no CSRMV=no CG=no \
  NVCCFLAGS='-O2 -std=c++17 -arch=sm_70 --expt-relaxed-constexpr -include cuda_fp16.h' \
  FLAGS="-O2 -std=c++17 -I$cuda_root/include -include cuda_fp16.h" \
  > "$out_prefix.build.log" 2>&1
"$nvcc_path" -O2 -std=c++17 -arch=sm_70 -I"$oz_repo/include" \
  "$script_dir/gpu_compare.cu" "$oz_repo/lib/libcuozblas.a" \
  -lcublas -lcusparse -o "$out_prefix.exe" >> "$out_prefix.build.log" 2>&1
sha256sum "$out_prefix.exe" "$fixture" >> "$out_prefix.environment.txt"
"$out_prefix.exe" "$fixture" "$out_prefix.bin" > "$out_prefix.run.log" 2>&1
python3 "$script_dir/analyze_gpu.py" "$fixture" "$out_prefix.bin" > "$out_prefix.metrics.json"
