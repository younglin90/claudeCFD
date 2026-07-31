#!/bin/bash
# Run the ACID solver (DENNER_ACID=1) on the given case IDs and report metrics.
# Usage: bash scripts/acid_cases.sh 13 24 25
cd /home/younglin90/work/claude_code/claudeCFD/solver_denner || exit 1
B=./build-cpp/cpp/denner_1d/denner1d_dump
for cs in "$@"; do
  DENNER_ACID=1 timeout 150 "$B" "$cs" 2>/dev/null > "/tmp/ac_${cs}.csv"
  python3 scripts/acid_check.py "/tmp/ac_${cs}.csv" "case${cs} ACID"
done
