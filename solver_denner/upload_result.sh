#!/bin/bash
cd /home/younglin90/work/claude_code/claudeCFD/solver_denner
python3 plot_results_sp.py
cp /tmp/fix_status.png results_cpp/figs/before_after.png
echo "--- upload to tmpfiles.org ---"
for f in results_cpp/figs/before_after.png results_cpp/figs/case25.png results_cpp/figs/case07.png; do
  resp=$(curl -s --max-time 60 -F "file=@$f" https://tmpfiles.org/api/v1/upload)
  url=$(echo "$resp" | python3 -c "import sys,json; print(json.load(sys.stdin)['data']['url'])" 2>/dev/null)
  echo "$(basename $f) -> ${url:-UPLOAD_FAILED ($resp)}"
done
