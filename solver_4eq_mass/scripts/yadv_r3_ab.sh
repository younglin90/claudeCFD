#!/usr/bin/env bash
# round 3: OFF + ON (+ ACID_YADV_RHOOLD probe) validate sweeps
W=/home/younglin90/work/claude_code/claudeCFD/solver_4eq_mass
cd "$W" || exit 1
V=./build-cpp/cpp/denner_1d/denner1d_validate
DENNER_ACID=1 $V > /tmp/yadv_v3_off.txt 2>/tmp/yadv_v3_off.err
echo "OFF: $(grep DENNER1D_CPP_METRIC /tmp/yadv_v3_off.txt | tail -1)"
DENNER_ACID=1 ACID_YADV=1 $V > /tmp/yadv_v3_on.txt 2>/tmp/yadv_v3_on.err
echo "ON : $(grep DENNER1D_CPP_METRIC /tmp/yadv_v3_on.txt | tail -1)"
if [ "$1" = "probe" ]; then
  DENNER_ACID=1 ACID_YADV=1 ACID_YADV_RHOOLD=1 $V > /tmp/yadv_v3_rhoold.txt 2>/dev/null
  echo "RHOOLD: $(grep DENNER1D_CPP_METRIC /tmp/yadv_v3_rhoold.txt | tail -1)"
fi
cp /tmp/yadv_v3_off.txt /tmp/yadv_off_val.txt
cp /tmp/yadv_v3_on.txt  /tmp/yadv_on_val.txt
