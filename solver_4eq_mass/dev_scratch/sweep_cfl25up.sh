#!/bin/bash
cd /home/younglin90/work/claude_code/claudeCFD/solver_denner
M() { DENNER_ACID=1 ACID_CFL25="$1" timeout 90 ./build-cpp/cpp/denner_1d/denner1d_validate --only 25 2>/dev/null | grep -oE '"(hf_p|corr_p|l2_p|amp_ratio_p|corr_u|pass)":[^,}]*' | tr '\n' ' '; echo; }
for v in 0.45 0.7 1.0 1.5 2.5 4.0; do
  echo "=== ACID_CFL25=$v ==="
  M "$v"
done
echo DONE
