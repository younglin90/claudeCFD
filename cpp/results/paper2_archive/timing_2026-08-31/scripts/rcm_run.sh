#!/bin/bash
# Does RCM restore stencil locality, and does the reconstruction speed-up follow?
# Locality first (1 step), then the full 500-step forced pair with and without it.
set -u
CPP=/home/younglin90/work/claude_code/claudeCFD/cpp
MSH=$CPP/results/paper_final/meshes
OUT=/tmp/mbq/rcm; mkdir -p "$OUT"; cd "$CPP/build" || exit 1
B="BVD_CHENG3=1 MLP_U2=0.001 HLLC_HLLBLEND=0 HLLC_PVRS=1"
DM="DM_MESH2D=$MSH/dmr_wc_paper.mesh2d DM_SINGLE=1"
echo "=== stencil locality"
for r in 0 1; do
  E=""; [ $r = 1 ] && E="MESH_RCM=1"
  printf "  RCM=%s  " "$r"
  taskset -c 32 env OMP_NUM_THREADS=1 CFD_MAXSTEP=1 MESH_LOCALITY=1 $E $DM $B THINCQQ_TANH=1 \
    ./double_mach_bench 240 60 0.2 2>&1 | grep -oP '\[MESH_LOCALITY\] \K.*' | head -1
done
echo "=== 500-step forced pair, RCM on"
for sc in s1 s2; do
  [ $sc = s1 ] && K="THINCQQ_TANH=1" || K="THINCQQ_GAUSS=1"
  echo "[$(date +%H:%M:%S)] START $sc"
  { echo "### threads=1 case=doublemach scheme=$sc steps=500 mode=forced100 rcm=1"; } \
    > "$OUT/doublemach_${sc}.log"
  taskset -c 32 env OMP_NUM_THREADS=1 CFD_MAXSTEP=500 CHENG3_PROF=1 THINCQQ_FORCE_ALL=1 \
    MESH_RCM=1 $DM $B $K ./double_mach_bench 240 60 0.2 >> "$OUT/doublemach_${sc}.log" 2>&1
  echo "[$(date +%H:%M:%S)] END   $sc"
done
echo "=== result"
for sc in s1 s2; do
  grep -hoE "MUSCL=[0-9.]+s|geom=[0-9.]+s|face=[0-9.]+s|BVD_sel=[0-9.]+s|D-solve [0-9.]+ms|face-value [0-9.]+ms|TOTAL wall=[0-9.]+s|posOK=[01]|rho\[[0-9.,]+\]" \
    "$OUT/doublemach_${sc}.log" | tr '\n' ' '; echo " <- $sc"
done
