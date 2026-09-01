#!/bin/bash
# 2D forced mode again, this time on ONE core, to separate the kernel effect from the thread effect.
#
# The 24-thread 2D run showed a smaller reconstruction speed-up (shock/mixing 1.23x) than the
# single-thread gated run did (2.12x), even though forcing raised the interface fraction to 99%.
# That is the memory system, not the scheme: at 16-24 threads the code is bandwidth-bound and the
# arithmetic the closure removes waits behind memory anyway. One core measures the kernel.
#
# Runs on core 48, which the 3D sweep (cores 0-46 even) is not using. Its 24 threads still saturate
# memory bandwidth, so ABSOLUTE times here are inflated -- but tanh and GAUSS are interleaved per
# case, so the contention lands on both equally and the RATIO, which is the point, holds.
set -u
CPP=/home/younglin90/work/claude_code/claudeCFD/cpp
MSH=$CPP/results/paper_final/meshes
OUT=/tmp/mbq/chart2D1T
mkdir -p "$OUT"; cd "$CPP/build" || exit 1
LOG=$OUT/runner.log; : > "$LOG"
B2="BVD_CHENG3=1 MLP_U2=0.001 HLLC_HLLBLEND=0 HLLC_PVRS=1"
run(){ local name=$1 sc=$2 cenv=$3; shift 3
  local K; [ $sc = s1 ] && K="THINCQQ_TANH=1" || K="THINCQQ_GAUSS=1"
  echo "[$(date +%H:%M:%S)] START $name $sc" | tee -a "$LOG"
  local t0=$(date +%s)
  { echo "### threads=1 case=$name scheme=$sc steps=500 mode=forced core=48"; } > "$OUT/${name}_${sc}.log"
  taskset -c 48 env OMP_NUM_THREADS=1 CFD_MAXSTEP=500 CHENG3_PROF=1 THINCQQ_FORCE_ALL=1 \
    $cenv $B2 $K "$@" >> "$OUT/${name}_${sc}.log" 2>&1
  echo "[$(date +%H:%M:%S)] END   $name $sc ($(( $(date +%s)-t0 ))s)" | tee -a "$LOG"
}
# interleaved per case so contention from the 3D sweep hits both kernels the same way
for name in leveque config3 shockvortex shockmixing mach3 doublemach; do
  for sc in s1 s2; do
    case $name in
      leveque)     run leveque     $sc "" ./leveque_bench 200 ;;
      config3)     run config3     $sc "C3_FLUX=llf C3_BC=dirichlet C3_SKIP_MLP=1" ./config3_bench 280 0.8 ;;
      shockvortex) run shockvortex $sc "VS_ONLY=16 VS_NSCALE=4" ./validation_smoke ;;
      shockmixing) run shockmixing $sc "" ./shock_mixing_bench 400 80 ;;
      mach3)       run mach3       $sc "M3_MESH=uniform M3_BVD_ONLY=1 M3_FLUX=hll M3_CFL=0.3" ./mach3_bench 480 160 4.0 ;;
      doublemach)  run doublemach  $sc "DM_MESH2D=$MSH/dmr_wc_paper.mesh2d DM_SINGLE=1" ./double_mach_bench 240 60 0.2 ;;
    esac
  done
done
echo "[$(date +%H:%M:%S)] 2D-1T DONE" | tee -a "$LOG"
