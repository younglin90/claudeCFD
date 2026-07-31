#!/bin/bash
cd /home/younglin90/work/claude_code/claudeCFD/solver_denner
cmake --build build-cpp -j8 2>&1 | grep -E "error:|Error" | head
for v in 1.0 2.0 4.0 100.0; do
  echo "=== ACID_MWICLAMP=$v ==="
  ACID_MWICLAMP="$v" python3 verify.py 25
done
echo DONE
