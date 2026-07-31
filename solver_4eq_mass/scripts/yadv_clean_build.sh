#!/usr/bin/env bash
# Full clean reconfigure + rebuild of this workspace (stale objects have misled a previous
# session -- see docs/YADV_RESEARCH.md section 3). Never -march=native.
set -eu
W=/home/younglin90/work/claude_code/claudeCFD/solver_4eq_mass
cd "$W"
rm -rf build-cpp
cmake -S . -B build-cpp -DCMAKE_BUILD_TYPE=Release > /tmp/yadv_cmake.log 2>&1
cmake --build build-cpp -j8 > /tmp/yadv_make.log 2>&1
echo "BUILD OK"
./build-cpp/cpp/denner_1d/denner1d_unit
