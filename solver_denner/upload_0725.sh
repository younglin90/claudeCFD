#!/bin/bash
cd /home/younglin90/work/claude_code/claudeCFD/solver_denner
# fresh dump (case25 reflects cfl=0.638; case07 cfl=0.45)
DENNER_ACID=1 ./build-cpp/cpp/denner_1d/denner1d_dump 07 2>/dev/null > /tmp/case_07.csv
DENNER_ACID=1 ./build-cpp/cpp/denner_1d/denner1d_dump 25 2>/dev/null > /tmp/case_25.csv
echo "dumped: 07=$(wc -l < /tmp/case_07.csv) lines, 25=$(wc -l < /tmp/case_25.csv) lines"
python3 plot_results.py 2>&1 | grep -E "case0?7|case25|saved.*case07|saved.*case25"
echo "--- upload to tmpfiles.org ---"
cd results_cpp/figs
for f in case07 case25; do
  resp=$(curl -s --max-time 60 -F "file=@$f.png" https://tmpfiles.org/api/v1/upload)
  url=$(echo "$resp" | python3 -c "import sys,json; print(json.load(sys.stdin)['data']['url'])" 2>/dev/null)
  echo "$f.png -> ${url:-UPLOAD_FAILED ($resp)}"
done
