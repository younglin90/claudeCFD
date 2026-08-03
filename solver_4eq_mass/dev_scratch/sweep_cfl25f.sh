#!/bin/bash
cd /home/younglin90/work/claude_code/claudeCFD/solver_denner
M() { DENNER_ACID=1 ACID_CFL25="$1" timeout 90 ./build-cpp/cpp/denner_1d/denner1d_validate --only 25 2>/dev/null | grep -oE '"(hf_p|hf_u|corr_p|corr_u|l2_p|amp_ratio_p|pass)":[^,}]*' | tr '\n' ' '; echo; }
for v in 0.60 0.638 0.65; do
  echo "=== ACID_CFL25=$v (Co_us ~ $(python3 -c "print(round(0.784*$v,3))")) ==="
  M "$v"
done
# first-step dt -> exact Co_us at cfl=0.638
echo "--- first-step dt @ cfl=0.638 ---"
DENNER_ACID=1 ACID_CFL25=0.638 ACID_DBG=1 timeout 90 ./build-cpp/cpp/denner_1d/denner1d_validate --only 25 2>&1 | grep -iE "ACID step 0 |step 0 t=" | head -1
echo DONE
