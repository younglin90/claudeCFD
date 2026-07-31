#!/usr/bin/env bash
# Full A/B: rebuild, unit test, OFF byte-identity, then the ACID_YADV=1 sweep.
set -u
W=/home/younglin90/work/claude_code/claudeCFD/solver_4eq_mass
cd "$W" || exit 1
echo "=== BUILD ==="
cmake --build build-cpp -j8 > /tmp/yadv_build.log 2>&1
echo "build_rc=$?"
grep -E "error|Error" /tmp/yadv_build.log | head -20
echo "=== UNIT ==="
./build-cpp/cpp/denner_1d/denner1d_unit 2>&1 | tail -10
echo "=== OFF: validate + dump identity ==="
DENNER_ACID=1 ./build-cpp/cpp/denner_1d/denner1d_validate > /tmp/yadv_off_val.txt 2>&1
tail -1 /tmp/yadv_off_val.txt
mkdir -p /tmp/yadv_off
for c in 01 02 14 25; do
  ./build-cpp/cpp/denner_1d/denner1d_dump "$c" > "/tmp/yadv_off/case${c}.txt" 2>&1
  if cmp -s "/tmp/yadv_base/case${c}.txt" "/tmp/yadv_off/case${c}.txt"; then
    echo "OFF case$c IDENTICAL"
  else
    echo "OFF case$c DIFFERS"
  fi
done
echo "=== ON: ACID_YADV=1 validate ==="
DENNER_ACID=1 ACID_YADV=1 ./build-cpp/cpp/denner_1d/denner1d_validate > /tmp/yadv_on_val.txt 2>&1
echo "on_rc=$?"
cat /tmp/yadv_on_val.txt
