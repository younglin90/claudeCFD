#!/bin/bash
cd /home/younglin90/work/claude_code/claudeCFD/solver_denner
cmake --build build-cpp -j8 2>&1 | grep -E "error:|Error" | head
V=./build-cpp/cpp/denner_1d/denner1d_validate
W() {  # $1 label, $2 env
  echo "=== $1 ==="
  eval "$2 python3 ana07.py" 2>/dev/null | grep -E "slope-reversals"
  eval "DENNER_ACID=1 $2 $V --only 07" 2>/dev/null | grep -oE '"(corr_p|hf_p|amp_ratio_p|l2_p|pass)":[^,}]*' | tr '\n' ' '; echo
}
W "baseline (BDF2)" ""
W "NOBDF2 (BE)" "ACID_NOBDF2=1"
echo DONE
