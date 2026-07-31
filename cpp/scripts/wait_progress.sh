#!/usr/bin/env bash
# Poll the shock-tube run log until a terminal marker or step milestone appears.
LOG=/tmp/mbq/shktube_run.log
TARGET="${1:-step 300}"
for i in $(seq 1 60); do
  if grep -qaE "ShockTube3D done|DIVERGED|FINISHED" "$LOG" 2>/dev/null; then
    echo "=== TERMINAL ==="
    grep -aE "done|DIVERGED|FINISHED|TFINAL|mass|slice written" "$LOG"
    exit 0
  fi
  if grep -qaF "$TARGET" "$LOG" 2>/dev/null; then break; fi
  sleep 3
done
echo "=== SNAPSHOT (target=$TARGET) ==="
grep -aE "\[step " "$LOG" | tail -6
echo "--- load ---"; cat /proc/loadavg
