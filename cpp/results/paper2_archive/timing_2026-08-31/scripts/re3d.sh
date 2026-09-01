#!/bin/bash
# All three 3D cases: their forced coverage was 10.4-62.6%, so all of them change. Core 40, so it
# runs beside the 2D job on core 32; both are single-threaded and neither saturates memory.
set -u
CPP=/home/younglin90/work/claude_code/claudeCFD/cpp
MSH=$CPP/results/paper_final/meshes
OUT=/tmp/mbq/stack; cd "$CPP/build" || exit 1
LOG=$OUT/runner3d_full.log; : > "$LOG"
B3="MLP_U2=0.001 BVD_BETA_L=1.4 BVD_BETA_S=0.8"
run(){ local name=$1 sc=$2 cenv=$3; shift 3
  local K; [ $sc = s1 ] && K="THINCQQ_TANH=1" || K="THINCQQ_GAUSS=1"
  echo "[$(date +%H:%M:%S)] START $name $sc" | tee -a "$LOG"; local t0=$(date +%s)
  { echo "### threads=1 case=$name scheme=$sc steps=20 mode=forced100 core=40"; } > "$OUT/${name}_${sc}.log"
  taskset -c 40 env OMP_NUM_THREADS=1 CFD_MAXSTEP=20 E3D_PROF=1 THINCQQ_FORCE_ALL=1 \
    $cenv $B3 $K "$@" >> "$OUT/${name}_${sc}.log" 2>&1
  echo "[$(date +%H:%M:%S)] END   $name $sc ($(( $(date +%s)-t0 ))s)" | tee -a "$LOG"
}
for name in deform3d twocyl3d sphere3d; do
  for sc in s1 s2; do
    case $name in
      deform3d) run deform3d $sc "DEF_RECON=bvd ADV_NOMOOD=1 DEF_CFL=0.25 DEF_T=3.0 N3_MESH=$MSH/def_nx76.umsh" ./deform3d_unstr_bench ;;
      twocyl3d) run twocyl3d $sc "MOOD_OFF=1 LANG_CASE=vortex BVD_PPFLOOR=0.1 N3_FLUX=1 N3_T=0.7 N3_MESH=$MSH/lang_vortex_nx112.umsh" ./langseth3d_unstr_bench ;;
      sphere3d) run sphere3d $sc "MOOD_OFF=1 BVD_PPFLOOR=0.1 N3_FLUX=1 N3_T=0.8 N3_MESH=$MSH/lang_nx100.umsh" ./langseth3d_unstr_bench ;;
    esac
  done
done
echo "[$(date +%H:%M:%S)] 3D-FULL DONE" | tee -a "$LOG"
