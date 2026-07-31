#!/usr/bin/env bash
set -euo pipefail

# Run Mach3 autoresearch probe on reserved core set (default 24-47).
# Usage:
#   ./run_mach3_isolated.sh <method-name>

METHOD="${1:-iter838}"
CORESET="${2:-24-47}"
THREADS="${3:-24}"
if [ "$#" -ge 1 ]; then shift; else shift 0; fi
if [ "$#" -ge 1 ]; then shift; else shift 0; fi
if [ "$#" -ge 1 ]; then shift; else shift 0; fi
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
taskset -c "${CORESET}" python3 tools/autoresearch/run_mach3_grid_probe.py --nx 240 --ny 80 --out autoresearch-results/current_mach3_step --recon-key tmlpu_mach3_step_on --method-name "${METHOD}" "${EXTRA_ARGS[@]}"
