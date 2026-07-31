#!/usr/bin/env bash
# Run the OFF (alpha) and ON (Y) validate sweeps into a named tag directory.
#   usage: yadv_run_ab.sh <tag>      -> /tmp/yadv_<tag>_off.txt, /tmp/yadv_<tag>_on.txt
set -u
TAG=${1:?tag required}
W=/home/younglin90/work/claude_code/claudeCFD/solver_4eq_mass
cd "$W" || exit 1
env -u ACID_YADV DENNER_ACID=1 ./build-cpp/cpp/denner_1d/denner1d_validate > "/tmp/yadv_${TAG}_off.txt" 2>&1
echo "OFF: $(tail -1 "/tmp/yadv_${TAG}_off.txt")"
DENNER_ACID=1 ACID_YADV=1 ./build-cpp/cpp/denner_1d/denner1d_validate > "/tmp/yadv_${TAG}_on.txt" 2>&1
echo "ON : $(tail -1 "/tmp/yadv_${TAG}_on.txt")"
