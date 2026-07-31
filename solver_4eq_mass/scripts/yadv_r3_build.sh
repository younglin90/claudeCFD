#!/usr/bin/env bash
# round 3: clean configure + build + unit test
set -e
W=/home/younglin90/work/claude_code/claudeCFD/solver_4eq_mass
cd "$W"
rm -rf build-cpp
cmake -S . -B build-cpp -DCMAKE_BUILD_TYPE=Release > /tmp/yadv_r3_cmake.log 2>&1
cmake --build build-cpp -j8 > /tmp/yadv_r3_build.log 2>&1
echo "BUILD OK"
./build-cpp/cpp/denner_1d/denner1d_unit
