set -e
cd /home/younglin90/work/claude_code/claudeCFD/solver_denner
echo "=== build (MP bound) ==="
cmake --build build-cpp -j8 2>&1 | grep -E "error:|Error" && { echo BUILD_FAIL; exit 1; } || true
echo BUILD_OK
V=./build-cpp/cpp/denner_1d/denner1d_validate
echo "=== full 10/10 ==="
DENNER_ACID=1 $V --only 01,02,04,05,07,13,14,15,24,25 2>/dev/null | grep -ao "pass_count=[0-9]* total=[0-9]*"
echo "=== case07/04/05: MP-on vs MP-off (hf_p, corr_p, amp_ratio_p) ==="
for c in 04 05 07; do
  DENNER_ACID=1 $V --only $c >/tmp/on_$c.txt 2>/dev/null
  ACID_NO_MPB=1 DENNER_ACID=1 $V --only $c >/tmp/off_$c.txt 2>/dev/null
  echo "case$c MP-ON : $(grep -ao "\"hf_p\":[0-9.e-]*\|\"corr_p\":[0-9.e-]*\|\"amp_ratio_p\":[0-9.e-]*" /tmp/on_$c.txt | tr "\n" " ")"
  echo "case$c MP-OFF: $(grep -ao "\"hf_p\":[0-9.e-]*\|\"corr_p\":[0-9.e-]*\|\"amp_ratio_p\":[0-9.e-]*" /tmp/off_$c.txt | tr "\n" " ")"
done
echo "=== regen case07 plot ==="
./build-cpp/cpp/denner_1d/denner1d_dump 07 >/tmp/case_07.csv 2>/dev/null
DENNER_ACID=1 ./build-cpp/cpp/denner_1d/denner1d_dump 07 >/tmp/case_07.csv 2>/dev/null
python3 -c "
import csv,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
r=list(csv.DictReader(open('/tmp/case_07.csv')))
x=[float(a['x']) for a in r]; p=[float(a['p']) for a in r]; pr=[float(a['p_ref']) for a in r]
fig,ax=plt.subplots(figsize=(13,4))
ax.plot(x,pr,'k--',lw=2,label='reference'); ax.plot(x,p,'r-',lw=1,label='ACID (MP bound)')
ax.set_title('case07 pressure -- MP-bounded 4th order'); ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig('/tmp/diag07_mp.png',dpi=120)
print('plotted')
"
echo DONE
