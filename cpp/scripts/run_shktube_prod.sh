#!/usr/bin/env bash
# Definitive Daru-Tenaud viscous shock-tube production run (uninterrupted).
cd /home/younglin90/work/claude_code/claudeCFD/cpp/build || exit 1
export OMP_NUM_THREADS=24
export SOLVE3D_PROGRESS=100
export SHKTUBE_CFL=0.2
export SHKTUBE_NX=120
export SHKTUBE_NY=60
export SHKTUBE_NZ=4
export SHKTUBE_T=1.0
./shock_tube3d_bench > /tmp/mbq/shktube_final.log 2>&1
echo "FINISHED rc=$?" >> /tmp/mbq/shktube_final.log
