#!/bin/bash
# Scenario B: 2D 500 steps, 3D 200 steps, one pinned core.
#
# Why longer: at 20 steps the 3D cases had covered 0.3-1.0% of their physical time -- the vortex
# and shock structure had not formed, so the interface fraction (and with it everything the GAUSS
# kernel touches) was measured on a nearly-undeveloped flow. mach3 showed the size of that effect
# in 2D: 0.08% interface at 5 steps against 0.90% at 200. 3D at 200 steps is 10x the development.
#
# Why still one thread: at 24 threads the run-to-run spread measured 23% on LeVeque, the 2D meshes
# are too small to keep 24 threads busy (sync cost competes with the work), and the code becomes
# bandwidth-bound there, which hides exactly the arithmetic the GAUSS kernel removes -- D2 was
# worth -43% at 24 threads and -4% at one. A pinned single core measures the kernel, reproducibly.
#
# 2D first: it finishes in ~1.4h, so partial results exist early if anything goes wrong.
set -u
CPP=/home/younglin90/work/claude_code/claudeCFD/cpp
MSH=$CPP/results/paper_final/meshes
OUT=/tmp/mbq/tblB
mkdir -p "$OUT"; cd "$CPP/build" || exit 1
make -j16 > "$OUT/build.log" 2>&1 || { echo BUILD_FAIL; tail -20 "$OUT/build.log"; exit 1; }
LOG=$OUT/runner.log; : > "$LOG"
echo "2D md5 $(md5sum leveque_bench | cut -d' ' -f1)  3D md5 $(md5sum langseth3d_unstr_bench | cut -d' ' -f1)" | tee -a "$LOG"
S1_2D="BVD_CHENG3=1 MLP_U2=0.001 HLLC_HLLBLEND=0 HLLC_PVRS=1 THINCQQ_TANH=1"
S2_2D="BVD_CHENG3=1 MLP_U2=0.001 HLLC_HLLBLEND=0 HLLC_PVRS=1 THINCQQ_GAUSS=1"
S1_3D="MLP_U2=0.001 BVD_BETA_L=1.4 BVD_BETA_S=0.8 THINCQQ_TANH=1"
S2_3D="MLP_U2=0.001 BVD_BETA_L=1.4 BVD_BETA_S=0.8 THINCQQ_GAUSS=1"
run(){ local name=$1 sc=$2 dim=$3 steps=$4 cenv=$5; shift 5
  local K
  if [ "$dim" = 2 ]; then [ $sc = s1 ] && K="$S1_2D" || K="$S2_2D"
  else                    [ $sc = s1 ] && K="$S1_3D" || K="$S2_3D"; fi
  echo "[$(date +%H:%M:%S)] START $name $sc (${steps} steps)" | tee -a "$LOG"
  local t0=$(date +%s)
  taskset -c 32 env OMP_NUM_THREADS=1 CFD_MAXSTEP=$steps CHENG3_PROF=1 E3D_PROF=1 $cenv $K "$@" \
    > "$OUT/${name}_${sc}.log" 2>&1
  echo "[$(date +%H:%M:%S)] END   $name $sc  ($(( $(date +%s)-t0 ))s)" | tee -a "$LOG"
}
for sc in s1 s2; do
  run leveque     $sc 2 500 "" ./leveque_bench 200
  run shockmixing $sc 2 500 "" ./shock_mixing_bench 400 80
  run shockvortex $sc 2 500 "VS_ONLY=16 VS_NSCALE=4" ./validation_smoke
  run mach3       $sc 2 500 "M3_MESH=uniform M3_BVD_ONLY=1 M3_FLUX=hll M3_CFL=0.3" ./mach3_bench 480 160 4.0
  run config3     $sc 2 500 "C3_FLUX=llf C3_BC=dirichlet C3_SKIP_MLP=1" ./config3_bench 280 0.8
  run doublemach  $sc 2 500 "DM_MESH2D=$MSH/dmr_wc_paper.mesh2d DM_SINGLE=1" ./double_mach_bench 240 60 0.2
done
echo "[$(date +%H:%M:%S)] 2D DONE" | tee -a "$LOG"
for name in deform3d twocyl3d sphere3d; do
  for sc in s1 s2; do
    case $name in
      deform3d) run deform3d $sc 3 200 "DEF_RECON=bvd ADV_NOMOOD=1 DEF_CFL=0.25 DEF_T=3.0 N3_MESH=$MSH/def_nx76.umsh" ./deform3d_unstr_bench ;;
      twocyl3d) run twocyl3d $sc 3 200 "MOOD_OFF=1 LANG_CASE=vortex BVD_PPFLOOR=0.1 N3_FLUX=1 N3_T=0.7 N3_MESH=$MSH/lang_vortex_nx112.umsh" ./langseth3d_unstr_bench ;;
      sphere3d) run sphere3d $sc 3 200 "MOOD_OFF=1 BVD_PPFLOOR=0.1 N3_FLUX=1 N3_T=0.8 N3_MESH=$MSH/lang_nx100.umsh" ./langseth3d_unstr_bench ;;
    esac
  done
done
echo "[$(date +%H:%M:%S)] ALL DONE" | tee -a "$LOG"
