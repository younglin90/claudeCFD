#!/usr/bin/env bash
# Build baseline + capture reference dumps BEFORE any Y-transport edits.
set -u
W=/home/younglin90/work/claude_code/claudeCFD/solver_4eq_mass
cd "$W" || exit 1
echo "=== BUILD ==="
cmake --build build-cpp -j8 2>&1 | tail -15
echo "=== VALIDATE (alpha baseline) ==="
DENNER_ACID=1 ./build-cpp/cpp/denner_1d/denner1d_validate 2>&1 | tail -40
echo "=== DUMPS ==="
mkdir -p /tmp/yadv_base
for c in 01 02 14 25; do
  ./build-cpp/cpp/denner_1d/denner1d_dump "$c" > "/tmp/yadv_base/case${c}.txt" 2>&1
  echo "case$c bytes=$(wc -c < /tmp/yadv_base/case${c}.txt) md5=$(md5sum < /tmp/yadv_base/case${c}.txt)"
done
