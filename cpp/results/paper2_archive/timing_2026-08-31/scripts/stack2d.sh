#!/bin/bash
# Remaining 2D cases for the stacked breakdown, core 32. leveque and config3 already have logs.
set -u
CPP=/home/younglin90/work/claude_code/claudeCFD/cpp
MSH=$CPP/results/paper_final/meshes
OUT=/tmp/mbq/stack; cd "$CPP/build" || exit 1
LOG=$OUT/runner2d.log; : > "$LOG"
B2="BVD_CHENG3=1 MLP_U2=0.001 HLLC_HLLBLEND=0 HLLC_PVRS=1"
run(){ local name=$1 sc=$2 cenv=$3; shift 3
  local K; [ $sc = s1 ] && K="THINCQQ_TANH=1" || K="THINCQQ_GAUSS=1"
  echo "[$(date +%H:%M:%S)] START $name $sc" | tee -a "$LOG"; local t0=$(date +%s)
  { echo "### threads=1 case=$name scheme=$sc steps=500 mode=forced"; } > "$OUT/${name}_${sc}.log"
  taskset -c 32 env OMP_NUM_THREADS=1 CFD_MAXSTEP=500 CHENG3_PROF=1 THINCQQ_FORCE_ALL=1 \
    $cenv $B2 $K "$@" >> "$OUT/${name}_${sc}.log" 2>&1
  echo "[$(date +%H:%M:%S)] END   $name $sc ($(( $(date +%s)-t0 ))s)" | tee -a "$LOG"
}
for name in shockvortex shockmixing mach3 doublemach; do
  for sc in s1 s2; do
    case $name in
      shockvortex) run shockvortex $sc "VS_ONLY=16 VS_NSCALE=4" ./validation_smoke ;;
      shockmixing) run shockmixing $sc "" ./shock_mixing_bench 400 80 ;;
      mach3)       run mach3       $sc "M3_MESH=uniform M3_BVD_ONLY=1 M3_FLUX=hll M3_CFL=0.3" ./mach3_bench 480 160 4.0 ;;
      doublemach)  run doublemach  $sc "DM_MESH2D=$MSH/dmr_wc_paper.mesh2d DM_SINGLE=1" ./double_mach_bench 240 60 0.2 ;;
    esac
  done
done
echo "[$(date +%H:%M:%S)] 2D DONE" | tee -a "$LOG"
