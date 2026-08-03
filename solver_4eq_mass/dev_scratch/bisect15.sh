#!/bin/bash
cd /home/younglin90/work/claude_code/claudeCFD/solver_denner
V=./build-cpp/cpp/denner_1d/denner1d_validate
M15() { DENNER_ACID=1 timeout 300 $V --only 15 2>/dev/null | grep -oE '"(pass|hf_u|corr_u|l2_u)":[^,}]*' | tr '\n' ' '; echo; }
B() { cmake --build build-cpp -j8 2>&1 | grep -E "error:" | head -3; }
mkdir -p /tmp/pa
cp cpp/denner_1d/src/acid.cpp /tmp/pa/acid.cpp
cp cpp/denner_1d/src/cases.cpp /tmp/pa/cases.cpp
cp cpp/denner_1d/include/denner1d/types.hpp /tmp/pa/types.hpp
echo "=== 0) HEAD baseline (all reverted) ==="
git checkout HEAD -- cpp/denner_1d/src/acid.cpp cpp/denner_1d/src/cases.cpp cpp/denner_1d/include/denner1d/types.hpp
B; M15
echo "=== 1) + acid.cpp edits only (auto_material + single dhat formula) ==="
cp /tmp/pa/acid.cpp cpp/denner_1d/src/acid.cpp
B; M15
echo "=== 2) + cases.cpp edits (per-case removals, cfl uniform) ==="
cp /tmp/pa/cases.cpp cpp/denner_1d/src/cases.cpp
B; M15
echo "=== 3) + types.hpp edits (fields removed, max_steps) == full Phase A ==="
cp /tmp/pa/types.hpp cpp/denner_1d/include/denner1d/types.hpp
B; M15
echo BISECT_DONE
