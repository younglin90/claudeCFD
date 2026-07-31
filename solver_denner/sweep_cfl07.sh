#!/bin/bash
cd /home/younglin90/work/claude_code/claudeCFD/solver_denner
cmake --build build-cpp -j8 2>&1 | grep -E "error:|Error" | head
M() { DENNER_ACID=1 ACID_CFL07="$1" timeout 90 ./build-cpp/cpp/denner_1d/denner1d_validate --only 07 2>/dev/null | grep -oE '"(hf_p|hf_u|corr_p|l2_p|amp_ratio_p|pass)":[^,}]*' | tr '\n' ' '; echo; }
for v in 0.15 0.25 0.45 0.65 0.9; do
  echo "=== ACID_CFL07=$v ==="
  M "$v"
done
echo DONE
