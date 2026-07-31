#!/bin/bash
cd /home/younglin90/work/claude_code/claudeCFD/solver_denner
for c in 01 02 04 05 07 13 24 25; do
  DENNER_ACID=1 timeout 450 ./build-cpp/cpp/denner_1d/denner1d_dump "$c" 2>/dev/null > "/tmp/case_$c.csv"
  echo "dumped $c ($(wc -l < /tmp/case_$c.csv) lines)"
done
python3 plot_results.py
