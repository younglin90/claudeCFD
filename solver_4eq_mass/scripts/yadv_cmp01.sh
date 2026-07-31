#!/usr/bin/env bash
set -u
cd /home/younglin90/work/claude_code/claudeCFD/solver_4eq_mass || exit 1
for c in 01 02 14 25; do
  if cmp -s "/tmp/yadv_base/case${c}.txt" "/tmp/yadv_on_case${c}.txt"; then
    echo "case${c}: ON == OFF  BYTE-IDENTICAL"
  else
    nd=$(diff "/tmp/yadv_base/case${c}.txt" "/tmp/yadv_on_case${c}.txt" | grep -c '^<')
    echo "case${c}: ON differs, ${nd} rows changed"
  fi
done
echo "--- case01 ON, first/last data rows ---"
sed -n '2p;$p' /tmp/yadv_on_case01.txt
echo "--- dump column precision (source) ---"
grep -n 'alpha' cpp/denner_1d/src/*.cpp | grep -n 'printf' | head -5
grep -rn 'x,alpha,p,u,rho' cpp/denner_1d/src/ | head -3
