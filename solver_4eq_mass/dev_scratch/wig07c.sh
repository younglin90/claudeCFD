#!/bin/bash
cd /home/younglin90/work/claude_code/claudeCFD/solver_denner
cmake --build build-cpp -j8 2>&1 | grep -E "error:|Error" | head
V=./build-cpp/cpp/denner_1d/denner1d_validate
for e in 0.0 0.005 0.01 0.02 -0.005 -0.01 -0.02; do
  echo "=== JST4=$e ==="
  ACID_JST4="$e" python3 ana07.py 2>/dev/null | grep -E "slope-reversals"
  DENNER_ACID=1 ACID_JST4="$e" timeout 60 $V --only 07 2>/dev/null | grep -oE '"(corr_p|amp_ratio_p|l2_p|pass)":[^,}]*' | tr '\n' ' '; echo
done
echo DONE
