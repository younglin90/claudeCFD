#!/usr/bin/env bash
# Emit one line per progress step and a terminal line when the production
# shock-tube run finishes or diverges. Used by the Monitor tool.
LOG=/tmp/mbq/shktube_prod.log
prev=""
for i in $(seq 1 290); do
  if grep -qaE "ShockTube3D done|DIVERGED|FINISHED rc=" "$LOG" 2>/dev/null; then
    grep -aE "done|DIVERGED|mass|slice written|TFINAL" "$LOG"
    echo "__PROD_TERMINAL__"
    exit 0
  fi
  cur=$(grep -aE "\[step " "$LOG" 2>/dev/null | tail -1)
  if [ -n "$cur" ] && [ "$cur" != "$prev" ]; then echo "$cur"; prev="$cur"; fi
  sleep 2
done
echo "__PROD_TIMEOUT_WATCH__"; tail -2 "$LOG" 2>/dev/null
