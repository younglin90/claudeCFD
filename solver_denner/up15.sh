cd /home/younglin90/work/claude_code/claudeCFD/solver_denner
F=results_cpp/figs/case15_ajac_vs_fd.png
ls -la "$F"
for try in 1 2 3; do
  resp=$(curl -s --max-time 60 -F "file=@$F" https://tmpfiles.org/api/v1/upload)
  url=$(echo "$resp" | python3 -c "import sys,json; print(json.load(sys.stdin)['data']['url'])" 2>/dev/null)
  if [ -n "$url" ]; then
    echo "VIEW: $url"
    echo "DIRECT: $(echo "$url" | sed 's#tmpfiles.org/#tmpfiles.org/dl/#')"
    break
  fi
  echo "try $try failed: $resp"
  sleep 2
done
