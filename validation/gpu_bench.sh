#!/bin/bash
# Time per step of the Julia model on one backend for a range of grid sizes (waves experiment).
#
# Usage: validation/gpu_bench.sh <backend> [nsteps=10] [julia_threads=1]
#   backend: cuda | amdgpu | cpu | opencl ...    Results are appended to validation/runs/gpu_bench_<backend>.txt
# Sizes (NI NJ NK ngrid, cells): edit SIZES below. The native waves grid is 16 768 192 (2.4M cells).
# Memory: roughly 1 kB per grid cell (host and device each), e.g. ~5 GB at 4.7M cells.
set -e
here="$(cd "$(dirname "$0")" && pwd)"
export PATH="$HOME/.juliaup/bin:$PATH"
be="${1:-cuda}"; n="${2:-10}"; th="${3:-1}"
SIZES=("16 192 96 4" "32 384 96 5" "16 768 192 5" "64 768 96 5" "128 768 192 6")
out="$here/runs/gpu_bench_$be.txt"; mkdir -p "$here/runs"
ulimit -s unlimited 2>/dev/null || ulimit -s 65520 2>/dev/null || true
for sz in "${SIZES[@]}"; do
    echo "== $sz" | tee -a "$out"
    PSOM_SIZE="$sz" PSOM_BACKEND="$be" julia -t"$th" --project="$here" "$here/bench.jl" waves "$n" 2>&1 \
        | grep -A1 "^julia" | tee -a "$out" || echo "  FAILED (see above)" | tee -a "$out"
done
