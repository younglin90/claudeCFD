#!/bin/bash
cd /home/younglin90/work/claude_code/claudeCFD/solver_denner
cmake --build build-cpp -j8 2>&1 | grep -E "error:|Error" | head
M() { DENNER_ACID=1 $1 timeout 90 ./build-cpp/cpp/denner_1d/denner1d_validate --only 07 2>/dev/null | grep -oE '"(hf_p|hf_u|corr_p|l2_p|amp_ratio_p|pass)":[^,}]*' | tr '\n' ' '; echo; }
echo "=== baseline (lowdiss+minmod+bdf2) ==="; M ""
echo "=== NOLOWDISS (2nd-order central) ===";   M "ACID_NOLOWDISS=1"
echo "=== NOMINMOD (1st-order upwind) ===";     M "ACID_NOMINMOD=1"
echo "=== NOBDF2 (BE time) ===";                M "ACID_NOBDF2=1"
echo "=== NOLOWDISS+NOMINMOD ===";              M "ACID_NOLOWDISS=1 ACID_NOMINMOD=1"
echo DONE
