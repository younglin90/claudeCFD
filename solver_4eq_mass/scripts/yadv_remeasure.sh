#!/usr/bin/env bash
# Re-run every A/B measurement on the FULLY RECOMPILED build (the copied build-cpp shipped
# with stale object files; the fresh build matches solver_denner's published binary exactly).
set -u
W=/home/younglin90/work/claude_code/claudeCFD/solver_4eq_mass
cd "$W" || exit 1
DENNER_ACID=1 ./build-cpp/cpp/denner_1d/denner1d_validate > /tmp/yadv_off_val.txt 2>&1
echo "OFF: $(tail -1 /tmp/yadv_off_val.txt)"
DENNER_ACID=1 ACID_YADV=1 ./build-cpp/cpp/denner_1d/denner1d_validate > /tmp/yadv_on_val.txt 2>&1
echo "ON : $(tail -1 /tmp/yadv_on_val.txt)"
echo "=== case01 / 13 / 15 ON rows ==="
grep -E '"case":"(01|13|15)"' /tmp/yadv_on_val.txt
echo "=== case13 OFF row ==="
grep -E '"case":"13"' /tmp/yadv_off_val.txt
