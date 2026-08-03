cd /home/younglin90/work/claude_code/claudeCFD/solver_denner
D=./build-cpp/cpp/denner_1d/denner1d_dump
echo "=== dump case15 FD + AJAC ==="
DENNER_ACID=1 $D 15 > /tmp/case_15_fd.csv 2>/dev/null && echo "fd rows: $(wc -l < /tmp/case_15_fd.csv)"
ACID_AJAC=1 DENNER_ACID=1 $D 15 > /tmp/case_15_ajac.csv 2>/dev/null && echo "ajac rows: $(wc -l < /tmp/case_15_ajac.csv)"
echo "=== plot ==="
python3 plot15.py
echo "=== upload tmpfiles.org ==="
resp=$(curl -s --max-time 60 -F "file=@results_cpp/figs/case15_ajac_vs_fd.png" https://tmpfiles.org/api/v1/upload)
echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); print('URL:', d['data']['url'])" 2>/dev/null || echo "RAW: $resp"
echo DONE
