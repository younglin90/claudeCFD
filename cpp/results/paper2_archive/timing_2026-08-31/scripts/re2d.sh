#!/bin/bash
# Re-run the 2D cases whose forced coverage was well below 100%: LeVeque 60.4%, cfg3 49.5%,
# double Mach 35.3%. With the no-gradient cells now also reconstructed, every case reaches 100%,
# so the fraction of cells reaching the kernel stops being a variable between cases.
# shock/mixing (99.4%), shock-vortex (97.3%) and Mach 3 (94.7%) were already close and are reused.
set -u
CPP=/home/younglin90/work/claude_code/claudeCFD/cpp
MSH=$CPP/results/paper_final/meshes
OUT=/tmp/mbq/stack; cd "$CPP/build" || exit 1
LOG=$OUT/runner2d_full.log; : > "$LOG"
B2="BVD_CHENG3=1 MLP_U2=0.001 HLLC_HLLBLEND=0 HLLC_PVRS=1"
run(){ local name=$1 sc=$2 cenv=$3; shift 3
  local K; [ $sc = s1 ] && K="THINCQQ_TANH=1" || K="THINCQQ_GAUSS=1"
  echo "[$(date +%H:%M:%S)] START $name $sc" | tee -a "$LOG"; local t0=$(date +%s)
  { echo "### threads=1 case=$name scheme=$sc steps=500 mode=forced100"; } > "$OUT/${name}_${sc}.log"
  taskset -c 32 env OMP_NUM_THREADS=1 CFD_MAXSTEP=500 CHENG3_PROF=1 THINCQQ_FORCE_ALL=1 \
    $cenv $B2 $K "$@" >> "$OUT/${name}_${sc}.log" 2>&1
  echo "[$(date +%H:%M:%S)] END   $name $sc ($(( $(date +%s)-t0 ))s)" | tee -a "$LOG"
}
for name in leveque config3 doublemach; do
  for sc in s1 s2; do
    case $name in
      leveque)    run leveque    $sc "" ./leveque_bench 200 ;;
      config3)    run config3    $sc "C3_FLUX=llf C3_BC=dirichlet C3_SKIP_MLP=1" ./config3_bench 280 0.8 ;;
      doublemach) run doublemach $sc "DM_MESH2D=$MSH/dmr_wc_paper.mesh2d DM_SINGLE=1" ./double_mach_bench 240 60 0.2 ;;
    esac
  done
done
echo "[$(date +%H:%M:%S)] 2D-FULL DONE" | tee -a "$LOG"
