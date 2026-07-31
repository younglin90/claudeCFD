#!/bin/bash
cd /home/younglin90/work/claude_code/claudeCFD/solver_denner
cmake --build build-cpp -j8 2>&1 | grep -E "error:|Error" | head
echo "=== BASELINE cfl=0.45 ==="
DENNER_ACID=1 timeout 60 ./build-cpp/cpp/denner_1d/denner1d_validate --only 25 2>/dev/null | grep -oE '"(hf_p|corr_p|l2_p|amp_ratio_p|pass)":[^,}]*' | tr '\n' ' '; echo
for v in 0.25 0.12 0.06; do
  echo "=== ACID_CFL25=$v ==="
  DENNER_ACID=1 ACID_CFL25="$v" timeout 120 ./build-cpp/cpp/denner_1d/denner1d_validate --only 25 2>/dev/null | grep -oE '"(hf_p|corr_p|l2_p|amp_ratio_p|pass)":[^,}]*' | tr '\n' ' '; echo
done
echo DONE
