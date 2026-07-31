#!/usr/bin/env bash
set -eu
W=/home/younglin90/work/claude_code/claudeCFD/solver_4eq_mass
cd "$W"
g++ -O2 -std=c++17 -Icpp/denner_1d/include scripts/yadv_hugoniot.cpp \
    -o /tmp/yadv_hugoniot build-cpp/cpp/denner_1d/libdenner1d.a -fopenmp
/tmp/yadv_hugoniot
