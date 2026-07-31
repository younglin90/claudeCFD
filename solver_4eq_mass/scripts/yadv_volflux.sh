#!/usr/bin/env bash
# Diagnostic probe run: alpha-space THINC + UNCONVERTED (volume-average) face value.
set -u
W=/home/younglin90/work/claude_code/claudeCFD/solver_4eq_mass
cd "$W" || exit 1
DENNER_ACID=1 ACID_YADV=1 ACID_YADV_VOLFLUX=1 ./build-cpp/cpp/denner_1d/denner1d_validate \
    > /tmp/yadv_v3_on.txt 2>&1
echo "PROBE: $(tail -1 /tmp/yadv_v3_on.txt)"
grep -E '"case":"(02|13|14|25|30)"' /tmp/yadv_v3_on.txt
