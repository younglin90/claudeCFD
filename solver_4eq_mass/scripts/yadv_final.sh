#!/usr/bin/env bash
set -u
W=/home/younglin90/work/claude_code/claudeCFD/solver_4eq_mass
cd "$W" || exit 1
echo "=== reconfigure + rebuild ==="
cmake -S . -B build-cpp -DCMAKE_BUILD_TYPE=Release > /tmp/yadv_cfg.log 2>&1
echo "cfg_rc=$?"
cmake --build build-cpp -j8 > /tmp/yadv_final_build.log 2>&1
echo "build_rc=$?"
grep -c ' error' /tmp/yadv_final_build.log
echo "=== unit ==="
./build-cpp/cpp/denner_1d/denner1d_unit
echo "unit_rc=$?"
echo "=== OFF validate ==="
DENNER_ACID=1 ./build-cpp/cpp/denner_1d/denner1d_validate 2>&1 | tail -1
echo "=== ON validate ==="
DENNER_ACID=1 ACID_YADV=1 ./build-cpp/cpp/denner_1d/denner1d_validate 2>&1 | tail -1
echo "=== cases.cpp / validation.cpp vs the reference workspace (must be identical) ==="
R=/home/younglin90/work/claude_code/claudeCFD/solver_denner/cpp/denner_1d/src
for f in cases.cpp validation.cpp; do
  if cmp -s "$R/$f" "cpp/denner_1d/src/$f"; then echo "$f UNCHANGED vs solver_denner"; else echo "$f CHANGED"; fi
done
echo "=== files I modified vs the reference workspace ==="
diff -q "$R/acid.cpp" cpp/denner_1d/src/acid.cpp
diff -q /home/younglin90/work/claude_code/claudeCFD/solver_denner/cpp/denner_1d/include/denner1d/eos.hpp cpp/denner_1d/include/denner1d/eos.hpp
diff -q /home/younglin90/work/claude_code/claudeCFD/solver_denner/cpp/denner_1d/tests/denner1d_unit.cpp cpp/denner_1d/tests/denner1d_unit.cpp
