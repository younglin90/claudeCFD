cd /home/younglin90/work/claude_code/claudeCFD/solver_denner
D=./build-cpp/cpp/denner_1d/denner1d_dump
V=./build-cpp/cpp/denner_1d/denner1d_validate
echo "=== dump CSVs (DENNER_ACID default = AJAC) ==="
for c in 01 02 04 05 07 13 14 15 24 25; do
  DENNER_ACID=1 $D $c > /tmp/case_$c.csv 2>/dev/null
  echo "  case$c rows=$(wc -l < /tmp/case_$c.csv)"
done
echo "=== metrics JSON per case ==="
: > /tmp/metrics.txt
for c in 01 02 04 05 07 13 14 15 24 25; do
  DENNER_ACID=1 $V --only $c >/tmp/m_$c.txt 2>/dev/null
  line=$(grep -a "\"case\"" /tmp/m_$c.txt | head -1)
  echo "$line" >> /tmp/metrics.txt
done
cat /tmp/metrics.txt
echo "=== plots ==="
python3 plot_report.py
echo DONE
