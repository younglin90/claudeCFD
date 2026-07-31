#!/bin/bash
cd /home/younglin90/work/claude_code/claudeCFD/solver_denner
cmake --build build-cpp -j8 2>&1 | grep -E "error:|Error" | head
echo "=== BASELINE (current) ==="
python3 verify.py 25 07
echo "=== ACID_APADV=1 (Denner e_P advection dhat) ==="
ACID_APADV=1 python3 verify.py 25 07
echo "=== ACID_GPW=1 (Denner density-weighted gpbar) ==="
ACID_GPW=1 python3 verify.py 25 07
echo "=== BOTH ==="
ACID_APADV=1 ACID_GPW=1 python3 verify.py 25 07
echo DONE
