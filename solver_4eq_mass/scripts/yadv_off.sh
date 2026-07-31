#!/usr/bin/env bash
# Build + verify switch-OFF byte identity vs the pre-change baseline, and run the unit test.
set -u
W=/home/younglin90/work/claude_code/claudeCFD/solver_4eq_mass
cd "$W" || exit 1
echo "=== BUILD ==="
cmake --build build-cpp -j8 2>&1 | tail -25
rc=${PIPESTATUS[0]}
echo "build_rc=$rc"
[ "$rc" -ne 0 ] && exit 1
echo "=== UNIT ==="
./build-cpp/cpp/denner_1d/denner1d_unit 2>&1 | tail -20
echo "unit_rc=$?"
echo "=== VALIDATE (ACID_YADV unset = alpha path) ==="
DENNER_ACID=1 ./build-cpp/cpp/denner_1d/denner1d_validate 2>&1 | tail -3
echo "=== DUMP DIFF vs baseline ==="
mkdir -p /tmp/yadv_off
for c in 01 02 14 25; do
  ./build-cpp/cpp/denner_1d/denner1d_dump "$c" > "/tmp/yadv_off/case${c}.txt" 2>&1
  if cmp -s "/tmp/yadv_base/case${c}.txt" "/tmp/yadv_off/case${c}.txt"; then
    echo "case$c IDENTICAL"
  else
    echo "case$c DIFFERS: $(diff /tmp/yadv_base/case${c}.txt /tmp/yadv_off/case${c}.txt | head -6)"
  fi
done
