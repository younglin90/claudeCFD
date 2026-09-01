#!/bin/bash
# Where does each 2D case stop gaining from more threads? Mesh sizes span 32k to 370k cells, so the
# knee is not the same for all of them -- picking one thread count for the group would either leave
# the big cases slow or push the small ones past the point where the barrier costs more than the
# work. Forced mode, since that is what the production runs will use (it raises the work per cell,
# which shifts the knee upward).
set -u
CPP=/home/younglin90/work/claude_code/claudeCFD/cpp
MSH=$CPP/results/paper_final/meshes
cd "$CPP/build" || exit 1
B2="BVD_CHENG3=1 MLP_U2=0.001 HLLC_HLLBLEND=0 HLLC_PVRS=1 THINCQQ_TANH=1 THINCQQ_FORCE_ALL=1"
probe(){ local name=$1 cenv=$2; shift 2
  printf "%-12s" "$name"
  for T in 1 2 4 8 16 24; do
    local cpus=$(seq 0 2 $((2*T-2)) | paste -sd,)
    local t0=$(date +%s.%N)
    taskset -c $cpus env OMP_NUM_THREADS=$T OMP_PROC_BIND=close OMP_PLACES=cores \
      CFD_MAXSTEP=40 $cenv $B2 "$@" > /dev/null 2>&1
    local d=$(echo "$(date +%s.%N) - $t0" | bc)
    printf "  T%-2d=%6.1fs" "$T" "$d"
  done
  echo
}
echo "2D strong scaling, 40 steps, forced tanh (wall seconds)"
probe leveque     "" ./leveque_bench 200
probe shockmixing "" ./shock_mixing_bench 400 80
probe shockvortex "VS_ONLY=16 VS_NSCALE=4" ./validation_smoke
probe mach3       "M3_MESH=uniform M3_BVD_ONLY=1 M3_FLUX=hll M3_CFL=0.3" ./mach3_bench 480 160 4.0
probe config3     "C3_FLUX=llf C3_BC=dirichlet C3_SKIP_MLP=1" ./config3_bench 280 0.8
probe doublemach  "DM_MESH2D=$MSH/dmr_wc_paper.mesh2d DM_SINGLE=1" ./double_mach_bench 240 60 0.2
