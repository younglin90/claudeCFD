#!/bin/bash
cd /home/younglin90/work/claude_code/claudeCFD/solver_denner
cmake --build build-cpp -j8 2>&1 | grep -E "error:|Error" | head
V=./build-cpp/cpp/denner_1d/denner1d_validate
M() { DENNER_ACID=1 ACID_N="$2" timeout 200 $V --only "$1" 2>/dev/null | grep -oE '"(hf_p|hf_u|corr_p|l2_p|amp_ratio_p|pass)":[^,}]*' | tr '\n' ' '; echo; }
echo "### case25 grid convergence (N=400 base)"
for N in 400 800 1600; do echo -n "N=$N  "; M 25 "$N"; done
echo "### case07 grid convergence (N=750 base)"
for N in 750 1500 3000; do echo -n "N=$N  "; M 07 "$N"; done
echo DONE
