#!/bin/bash
# One step per case, just to build the context and print its stencil locality.
set -u
CPP=/home/younglin90/work/claude_code/claudeCFD/cpp
MSH=$CPP/results/paper_final/meshes
cd "$CPP/build" || exit 1
make -j16 2>&1 | grep -cE "error"; echo "build errors above"
B="BVD_CHENG3=1 MLP_U2=0.001 HLLC_HLLBLEND=0 HLLC_PVRS=1 THINCQQ_TANH=1"
r(){ printf "%-13s " "$1"; shift
  taskset -c 32 env OMP_NUM_THREADS=1 CFD_MAXSTEP=1 MESH_LOCALITY=1 "$@" 2>&1 \
    | grep -oP '\[MESH_LOCALITY\] \K.*' | head -1; }
r leveque     $B ./leveque_bench 200
r config3     C3_FLUX=llf C3_BC=dirichlet C3_SKIP_MLP=1 $B ./config3_bench 280 0.8
r shockmixing $B ./shock_mixing_bench 400 80
r mach3       M3_MESH=uniform M3_BVD_ONLY=1 M3_FLUX=hll M3_CFL=0.3 $B ./mach3_bench 480 160 4.0
r doublemach  DM_MESH2D=$MSH/dmr_wc_paper.mesh2d DM_SINGLE=1 $B ./double_mach_bench 240 60 0.2
