#!/bin/bash
cd /home/younglin90/work/claude_code/claudeCFD/solver_denner
for c in 01 02 04 05 07 13 14 24 25 26 27 28 30 31 33 34 35 36; do
  f="results_cpp/figs/case${c}.png"
  if [ -f "$f" ]; then echo "case${c} OK $(stat -c%s "$f")"; else echo "case${c} MISSING"; fi
done
