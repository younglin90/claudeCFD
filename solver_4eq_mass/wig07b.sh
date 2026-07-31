#!/bin/bash
cd /home/younglin90/work/claude_code/claudeCFD/solver_denner
cmake --build build-cpp -j8 2>&1 | grep -E "error:|Error" | head
V=./build-cpp/cpp/denner_1d/denner1d_validate
for th in 1.0 0.9 0.8 0.7 0.6; do
  echo "=== BDF_THETA=$th ==="
  ACID_BDF_THETA="$th" python3 ana07.py 2>/dev/null | grep -E "slope-reversals"
  DENNER_ACID=1 ACID_BDF_THETA="$th" $V --only 07 2>/dev/null | grep -oE '"(corr_p|hf_p|amp_ratio_p|l2_p|pass)":[^,}]*' | tr '\n' ' '; echo
done
echo DONE
