#!/bin/bash
cd /home/younglin90/work/claude_code/claudeCFD/solver_denner
cmake --build build-cpp -j8 2>&1 | grep -E "error:|Error" | head
V=./build-cpp/cpp/denner_1d/denner1d_validate
for c in 0.45 0.35 0.25 0.52; do
  echo "=== CFL07=$c ==="
  ACID_CFL07="$c" python3 ana07.py 2>/dev/null | grep -E "slope-reversals"
  DENNER_ACID=1 ACID_CFL07="$c" timeout 90 $V --only 07 2>/dev/null | grep -oE '"(corr_p|amp_ratio_p|l2_p|hf_p|pass)":[^,}]*' | tr '\n' ' '; echo
done
echo DONE
