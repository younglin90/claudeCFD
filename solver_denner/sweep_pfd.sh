#!/bin/bash
cd /home/younglin90/work/claude_code/claudeCFD/solver_denner
for v in 0.1 0.2 0.3 0.5; do
  echo "=== ACID_PFD=$v ==="
  ACID_PFD="$v" python3 verify.py 25 13 14
done
