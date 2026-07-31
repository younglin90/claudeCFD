#!/bin/bash
# Test the ACID solver on the single-phase acoustic case04 (iso + full energy).
cd /home/younglin90/work/claude_code/claudeCFD/solver_denner || exit 1
cmake --build build-cpp -j 8 2>&1 | grep -E "error:|Error" | head
B=./build-cpp/cpp/denner_1d/denner1d_dump

DENNER_ACID=1 ACID_ISOTHERMAL=1 timeout 90 "$B" 04 2>/dev/null >/tmp/acid04_iso.csv
python3 scripts/acid_check.py /tmp/acid04_iso.csv "case04 ISOTHERMAL"

DENNER_ACID=1 timeout 90 "$B" 04 2>/dev/null >/tmp/acid04_full.csv
python3 scripts/acid_check.py /tmp/acid04_full.csv "case04 FULL"
