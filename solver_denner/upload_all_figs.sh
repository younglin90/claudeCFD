#!/bin/bash
cd /home/younglin90/work/claude_code/claudeCFD/solver_denner
python3 plot_all_sp.py
echo "--- copy to Windows Downloads ---"
mkdir -p /mnt/c/Users/user/Downloads/denner_figs
cp results_cpp/figs/case*.png /mnt/c/Users/user/Downloads/denner_figs/ && echo "copied to Downloads/denner_figs"
echo "--- upload to tmpfiles.org ---"
for c in 01 02 04 05 07 13 14 15 24 25 26 27 28 30 31 33 34 35 36; do
  f="results_cpp/figs/case$c.png"
  [ -f "$f" ] || { echo "case$c MISSING"; continue; }
  resp=$(curl -s --max-time 60 -F "file=@$f" https://tmpfiles.org/api/v1/upload)
  url=$(echo "$resp" | python3 -c "import sys,json; print(json.load(sys.stdin)['data']['url'])" 2>/dev/null)
  echo "case$c -> ${url:-UPLOAD_FAILED}"
done
echo DONE
