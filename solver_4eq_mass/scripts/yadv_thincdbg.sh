#!/usr/bin/env bash
# THINC activation counts, alpha path vs Y path (alpha-space reconstruction).
set -u
W=/home/younglin90/work/claude_code/claudeCFD/solver_4eq_mass
cd "$W" || exit 1
for c in 02 13 14 25 30; do
  a=$(env -u ACID_YADV DENNER_ACID=1 ACID_THINC_DBG=1 ./build-cpp/cpp/denner_1d/denner1d_dump "$c" 2>&1 >/dev/null | grep -i thinc)
  y=$(DENNER_ACID=1 ACID_YADV=1 ACID_THINC_DBG=1 ./build-cpp/cpp/denner_1d/denner1d_dump "$c" 2>&1 >/dev/null | grep -i thinc)
  echo "case$c  ALPHA: $a"
  echo "case$c  YADV : $y"
done
