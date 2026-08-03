#!/bin/bash
cd /home/younglin90/work/claude_code/claudeCFD/solver_denner
run_one() {
  N=$1
  sed -i "s/auto c14 = base_config([0-9]*, 2.29e-4/auto c14 = base_config($N, 2.29e-4/" cpp/denner_1d/src/cases.cpp
  cmake --build build-cpp -j8 2>&1 | grep -E "error:" | head -2
  echo "=== N=$N ==="
  DENNER_ACID=1 timeout 300 ./build-cpp/cpp/denner_1d/denner1d_validate --only 14 2>/dev/null | grep -oE '"(pass|corr_u|l2_u|l2_rho|corr_rho|hf_p)":[^,}]*' | tr '\n' ' '; echo
  python3 - <<EOF
import csv,io,os,subprocess
env=dict(os.environ,DENNER_ACID="1")
out=subprocess.run(["./build-cpp/cpp/denner_1d/denner1d_dump","14"],capture_output=True,text=True,env=env).stdout
r=list(csv.DictReader(io.StringIO(out)))
x=[float(a["x"]) for a in r]; rho=[float(a["rho"]) for a in r]; rr=[float(a["rho_ref"]) for a in r]
jump=max(rr)-min(rr)
# band = cells where |rho-rho_ref| > 2% of jump, within x in [0.5,0.95] (contact region)
band=[i for i in range(len(x)) if 0.5<=x[i]<=0.95 and abs(rho[i]-rr[i])>0.02*jump]
w_cells=(band[-1]-band[0]+1) if band else 0
w_phys=(x[band[-1]]-x[band[0]]) if band else 0.0
mx=max((abs(rho[i]-rr[i]) for i in band), default=0.0)
print("band(>2%%jump): %d cells, %.4f m physical, max_dev %.1f%% of jump"%(w_cells,w_phys,100*mx/jump))
EOF
}
run_one 400
run_one 800
run_one 1600
sed -i "s/auto c14 = base_config([0-9]*, 2.29e-4/auto c14 = base_config(400, 2.29e-4/" cpp/denner_1d/src/cases.cpp
cmake --build build-cpp -j8 2>&1 | grep -cE "error:"
git diff --stat | head -2
echo SWEEP_DONE
