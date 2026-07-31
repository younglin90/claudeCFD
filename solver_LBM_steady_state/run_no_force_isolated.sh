#!/usr/bin/env bash
set -euo pipefail

# Launch no-force benchmark with pinned cores and explicit threading.
# Usage:
#   ./run_no_force_isolated.sh METHOD [CORESET] [THREADS] [--extra python flags]
# Example:
#   ./run_no_force_isolated.sh proposed 0-23 24 --no-vtk --no-cache

METHOD="${1:-proposed}"
CORESET="${2:-0-23}"
THREADS="${3:-24}"
if [ "$#" -ge 1 ]; then shift; else shift 0; fi
if [ "$#" -ge 1 ]; then shift; else shift 0; fi
if [ "$#" -ge 1 ]; then shift; else shift 0; fi

# Forward remaining args (e.g., --no-cache, --methods)
EXTRA_ARGS=("$@")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR"

if ! command -v taskset >/dev/null 2>&1; then
  echo "taskset not found; cannot bind cores." >&2
  exit 1
fi

export NUMBA_NUM_THREADS="${THREADS}"
export OMP_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "[run] method=${METHOD} cores=${CORESET} threads=${THREADS}"
taskset -c "${CORESET}" python3 paper_60case_benchmark_no_force.py --no-vtk --methods "${METHOD}" "${EXTRA_ARGS[@]}"
