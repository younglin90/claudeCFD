#!/bin/bash
cd /home/younglin90/work/claude_code/claudeCFD/solver_denner
cmake --build build-cpp -j8 2>&1 | grep -E "error:|Error" | head
for v in 1.0 1.5 2.0 3.0; do
  echo "=== ACID_DHK=$v ==="
  ACID_DHK="$v" python3 verify.py 25 07
done
echo DONE
