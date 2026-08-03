#!/bin/bash
cd /home/younglin90/work/claude_code/claudeCFD/solver_denner
for c in 14 15; do
  DENNER_ACID=1 timeout 300 ./build-cpp/cpp/denner_1d/denner1d_dump "$c" 2>/dev/null > "/tmp/case_$c.csv"
  echo "dumped $c ($(wc -l < /tmp/case_$c.csv) lines)"
done
python3 plot_results.py
cp results_cpp/figs/case14.png results_cpp/figs/case15.png /mnt/c/Users/user/AppData/Local/Temp/denner_figs/
echo "copied 14,15 to windows temp"
# upload 14,15 to tmpfiles
cd results_cpp/figs
for c in 14 15; do
  resp=$(curl -s --max-time 60 -F "file=@case$c.png" https://tmpfiles.org/api/v1/upload)
  url=$(echo "$resp" | python3 -c "import sys,json; print(json.load(sys.stdin)['data']['url'])" 2>/dev/null)
  echo "case$c -> ${url:-FAILED}"
done
