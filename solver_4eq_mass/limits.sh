#!/bin/bash
cd /home/younglin90/work/claude_code/claudeCFD/solver_denner
V=./build-cpp/cpp/denner_1d/denner1d_validate
M(){ eval "DENNER_ACID=1 $2 timeout 300 $V --only $1" 2>/dev/null | grep -oE '"(pass|corr_p|corr_rho|l2_p|amp_ratio_p|hf_p|hf_u)":[^,}]*' | tr '\n' ' '; echo; }
echo "### case24: does the case25 MWI-dissipation prescription transfer? (limit test)"
echo -n "base           "; M 24 ""
echo -n "APADV          "; M 24 "ACID_APADV=1"
echo -n "APADV+DHK4     "; M 24 "ACID_APADV=1 ACID_DHK=4"
echo -n "APADV+DHK8     "; M 24 "ACID_APADV=1 ACID_DHK=8"
echo -n "DHK4           "; M 24 "ACID_DHK=4"
echo "### case13: dissipation headroom"
echo -n "base           "; M 13 ""
echo -n "DHK2           "; M 13 "ACID_DHK=2"
echo -n "DHK4           "; M 13 "ACID_DHK=4"
echo "### case04: amplitude loss -- resolution-limited or scheme-limited?"
echo -n "N500(base)     "; M 04 ""
echo -n "N1000          "; M 04 "ACID_N=1000"
echo -n "N2000          "; M 04 "ACID_N=2000"
echo "### case05: same"
echo -n "N400(base)     "; M 05 ""
echo -n "N800           "; M 05 "ACID_N=800"
echo "### case02: contact smearing -- grid convergence"
echo -n "N500(base)     "; M 02 ""
echo -n "N1000          "; M 02 "ACID_N=1000"
echo -n "N2000          "; M 02 "ACID_N=2000"
echo LIMITS_DONE
