#!/bin/bash
cd /home/younglin90/work/claude_code/claudeCFD/solver_denner
V=./build-cpp/cpp/denner_1d/denner1d_validate
M() { DENNER_ACID=1 ACID_DHK="$2" timeout 120 $V --only "$1" 2>/dev/null | grep -oE '"(hf_p|hf_u|corr_p|corr_u|l2_p|amp_ratio_p|pass)":[^,}]*' | tr '\n' ' '; echo; }
echo "### case25 (per-case dhk; want amp->~1.05, corr kept)"
for k in 1 3 5 8 12; do echo -n "dhk=$k  "; M 25 "$k"; done
echo "### case07 (want hf_p down, corr/amp kept)"
for k in 1 3 5 8 12; do echo -n "dhk=$k  "; M 07 "$k"; done
echo DONE
