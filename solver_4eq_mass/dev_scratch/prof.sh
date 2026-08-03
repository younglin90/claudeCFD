cd /home/younglin90/work/claude_code/claudeCFD/solver_denner
echo "=== build flags (OpenMP? -O3?) ==="
grep -rniE "openmp|O3|O2|march|CMAKE_CXX_FLAGS|CMAKE_BUILD_TYPE|fopenmp" CMakeLists.txt cpp/CMakeLists.txt cpp/denner_1d/CMakeLists.txt 2>/dev/null | head -20
echo "=== nproc ==="; nproc
echo "=== per-case wall time (FD default) ==="
V=./build-cpp/cpp/denner_1d/denner1d_validate
tot=0
for c in 01 02 04 05 07 13 14 15 24 25; do
  t0=$(date +%s.%N); DENNER_ACID=1 $V --only $c >/dev/null 2>&1; t1=$(date +%s.%N)
  dt=$(echo "$t1-$t0"|bc); tot=$(echo "$tot+$dt"|bc); printf "case %s : %6.2fs\n" $c $dt
done
echo "TOTAL: ${tot}s"
echo DONE
