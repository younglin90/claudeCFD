cd /home/younglin90/work/claude_code/claudeCFD/solver_denner
D=./build-cpp/cpp/denner_1d/denner1d_dump
V=./build-cpp/cpp/denner_1d/denner1d_validate
ACID_CFL24=0.30 DENNER_ACID=1 $D 24 >/tmp/case_24_cfl030.csv 2>/dev/null
ACID_CFL24=0.60 DENNER_ACID=1 $D 24 >/tmp/case_24_cfl060.csv 2>/dev/null
echo "=== case24 hf vs cfl ==="
for cf in 0.30 0.45 0.60; do
  ACID_CFL24=$cf DENNER_ACID=1 $V --only 24 >/tmp/m24_$cf.txt 2>/dev/null
  hp=$(grep -ao '"hf_p":[0-9.e-]*' /tmp/m24_$cf.txt)
  hu=$(grep -ao '"hf_u":[0-9.e-]*' /tmp/m24_$cf.txt)
  lp=$(grep -ao '"linf_p":[0-9.e-]*' /tmp/m24_$cf.txt)
  cp=$(grep -ao '"corr_p":[0-9.e-]*' /tmp/m24_$cf.txt)
  echo "  cfl=$cf : $hp $hu $lp $cp"
done
echo "=== case07 hf ==="
grep -ao '"hf_p":[0-9.e-]*\|"hf_u":[0-9.e-]*' /tmp/m_07.txt 2>/dev/null || DENNER_ACID=1 $V --only 07 2>/dev/null | grep -ao '"hf_p":[0-9.e-]*'
python3 diag.py
echo DONE
