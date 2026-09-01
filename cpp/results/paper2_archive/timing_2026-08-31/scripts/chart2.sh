#!/bin/bash
# Timing decomposition for the bar chart: 500 iterations, FORCED mode only, both kernels.
#
# Thread count is per case, taken from a measured strong-scaling probe rather than assumed. The 2D
# meshes span 32k-370k cells, so one setting cannot suit them: leveque and mach3 are SLOWER at 24
# threads than at 16 (the barrier starts costing more than the work it splits), while shockmixing
# and shock-vortex still gain out to 24. The 3D meshes (1.8-4.0M cells) take 24.
#
# Two clocks, recorded separately because they are not the same quantity:
#   phase buckets (chrono) = WALL time of a stage
#   D-solve / face-value   = CPU cycles SUMMED across threads (RDTSC under a reduction)
# They coincide only at one thread. The thread count is written into each log so the chart can
# keep them on separate axes.
#
# THINCQQ_FORCE_ALL builds a THINC profile on cells where the constraint has no finite solution.
# It stays numerically finite -- a flat neighbourhood collapses the profile back to the cell
# average -- but it moves the answer (twocyl enstrophy 1.8042 -> 1.8035) and is not the published
# scheme. These runs isolate the cost of the two kernels; they are not a physics result.
set -u
CPP=/home/younglin90/work/claude_code/claudeCFD/cpp
MSH=$CPP/results/paper_final/meshes
OUT=/tmp/mbq/chartF
mkdir -p "$OUT"; cd "$CPP/build" || exit 1
make -j16 > "$OUT/build.log" 2>&1 || { echo BUILD_FAIL; tail -20 "$OUT/build.log"; exit 1; }
LOG=$OUT/runner.log; : > "$LOG"
B2="BVD_CHENG3=1 MLP_U2=0.001 HLLC_HLLBLEND=0 HLLC_PVRS=1"
B3="MLP_U2=0.001 BVD_BETA_L=1.4 BVD_BETA_S=0.8"
run(){ local name=$1 sc=$2 dim=$3 T=$4 cenv=$5; shift 5
  local K B; [ "$dim" = 2 ] && B="$B2" || B="$B3"
  [ $sc = s1 ] && K="THINCQQ_TANH=1" || K="THINCQQ_GAUSS=1"
  local cpus=$(seq 0 2 $((2*T-2)) | paste -sd,)
  echo "[$(date +%H:%M:%S)] START $name $sc (T=$T)" | tee -a "$LOG"
  local t0=$(date +%s)
  { echo "### threads=$T case=$name scheme=$sc steps=500 mode=forced"; } > "$OUT/${name}_${sc}.log"
  taskset -c $cpus env OMP_NUM_THREADS=$T OMP_PROC_BIND=close OMP_PLACES=cores \
    CFD_MAXSTEP=500 CHENG3_PROF=1 E3D_PROF=1 THINCQQ_FORCE_ALL=1 $cenv $B $K "$@" \
    >> "$OUT/${name}_${sc}.log" 2>&1
  echo "[$(date +%H:%M:%S)] END   $name $sc ($(( $(date +%s)-t0 ))s)" | tee -a "$LOG"
}
for sc in s1 s2; do
  run leveque     $sc 2 16 "" ./leveque_bench 200
  run shockmixing $sc 2 24 "" ./shock_mixing_bench 400 80
  run shockvortex $sc 2 24 "VS_ONLY=16 VS_NSCALE=4" ./validation_smoke
  run mach3       $sc 2 16 "M3_MESH=uniform M3_BVD_ONLY=1 M3_FLUX=hll M3_CFL=0.3" ./mach3_bench 480 160 4.0
  run config3     $sc 2 16 "C3_FLUX=llf C3_BC=dirichlet C3_SKIP_MLP=1" ./config3_bench 280 0.8
  run doublemach  $sc 2 24 "DM_MESH2D=$MSH/dmr_wc_paper.mesh2d DM_SINGLE=1" ./double_mach_bench 240 60 0.2
done
echo "[$(date +%H:%M:%S)] 2D DONE" | tee -a "$LOG"
for sc in s1 s2; do
  run deform3d $sc 3 24 "DEF_RECON=bvd ADV_NOMOOD=1 DEF_CFL=0.25 DEF_T=3.0 N3_MESH=$MSH/def_nx76.umsh" ./deform3d_unstr_bench
  run twocyl3d $sc 3 24 "MOOD_OFF=1 LANG_CASE=vortex BVD_PPFLOOR=0.1 N3_FLUX=1 N3_T=0.7 N3_MESH=$MSH/lang_vortex_nx112.umsh" ./langseth3d_unstr_bench
  run sphere3d $sc 3 24 "MOOD_OFF=1 BVD_PPFLOOR=0.1 N3_FLUX=1 N3_T=0.8 N3_MESH=$MSH/lang_nx100.umsh" ./langseth3d_unstr_bench
done
echo "[$(date +%H:%M:%S)] ALL DONE" | tee -a "$LOG"
