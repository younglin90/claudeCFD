#!/bin/bash
cd /home/younglin90/work/claude_code/claudeCFD/solver_denner/results_cpp/figs
for f in case01 case02 case04 case05 case07 case13 case24 case25; do
  resp=$(curl -s --max-time 60 -F "file=@$f.png" https://tmpfiles.org/api/v1/upload)
  url=$(echo "$resp" | python3 -c "import sys,json; print(json.load(sys.stdin)['data']['url'])" 2>/dev/null)
  echo "$f -> ${url:-UPLOAD_FAILED ($resp)}"
done
