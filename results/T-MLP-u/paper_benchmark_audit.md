# T-MLP-u Paper Benchmark Audit

Generated from the current high-resolution artifacts in `results/T-MLP-u`.

## Concrete Objective Checklist

| Requirement | Evidence | Status |
|---|---|---|
| LeVeque rotation uses N=100 triangle mesh | `paper_benchmark_leveque_N100_current.json`: `mesh=criss_cross_triangles`, `mesh_cells=40000`, `logical_nx=100` | PASS |
| LeVeque OFF is pure downwind with `mlp_bound=False` | harness comparison definition and `T-MLP-u OFF` row: NaN at `t=0.2455989657` | PASS |
| LeVeque ON is pure downwind wrapped by T-MLP-u with `mlp_bound=True`, `vertex_mlp=True`, `tvb_M=0`, `extremum_relax=False` | harness comparison definition; ON row `ok=true`, `l1=0.0165254330`, `wiggle=4.69e-12` | PASS |
| Double Mach uses unstructured triangle mesh at paper-scale resolution | `paper_benchmark_double_mach_tri_paper.json`: `480x120`, `115200` triangles | PASS |
| Mach 3 step uses unstructured triangle mesh at paper-scale resolution | `paper_benchmark_mach3_step_tri_paper.json`: `240x80`, `32256` triangles after step cutout | PASS |
| Euler shock cases use primitive-variable reconstruction and HLLC-family flux | comparison definition: `primitive variables W=(rho,u,v,p)`, flux `hllc_adc` | PASS |
| Shock OFF is pure SUPERBEE with `mlp_bound=False` | comparison definition; Double Mach OFF NaN at `t=0.0001722543`; Mach 3 step OFF NaN at `t=0.0041686960` | PASS |
| Shock ON is SUPERBEE wrapped by T-MLP-u with `mlp_bound=True`, `vertex_mlp=True`, `tvb_M=0`, `extremum_relax=False` | comparison definition; Double Mach ON and Mach 3 step ON both `ok=true` | PASS |
| Mach 3 step reaches `t=4` robustly with no OFF-style NaN | ON row `ok=true`, `steps=10907`; OFF row `ok=false` | PASS |
| Double Mach reaches `t=0.2` robustly with no OFF-style NaN | ON row `ok=true`, `steps=4050`; OFF row `ok=false` | PASS |
| T-MLP-u beats all listed unstructured limiters on Double Mach vortex sharpness | ON `vortex_proxy=0.25536`; MLP-u1 `0.25330`, MLP-u2 `0.24939` | PASS |
| T-MLP-u beats all listed unstructured limiters on Mach 3 flag metric | ON `flag_proxy=0.10811`; MLP-u1 `0.09959`, MLP-u2 `0.09713` | PASS |
| Overall manifest says complete paper evidence is ready | `paper_benchmark_summary.json`: `evidence_ready=1`, all case-ready flags are `1` | PASS |

## Current Numerical Summary

### LeVeque N=100

| Method | Status | L1 | Wiggle |
|---|---:|---:|---:|
| MLP-u1 | ok | 0.01934245 | 9.48e-30 |
| MLP-u2 | ok | 0.02674952 | 2.44e-3 |
| T-MLP-u OFF | NaN | - | - |
| T-MLP-u ON | ok | 0.01652543 | 4.69e-12 |

### Double Mach, 480x120 logical, 115200 triangles

| Method | Status | Vortex Proxy | Checker |
|---|---:|---:|---:|
| MLP-u1 | ok | 0.25330 | 0.003620 |
| MLP-u2 | ok | 0.24939 | 0.003944 |
| T-MLP-u OFF | NaN | 0.0 | - |
| T-MLP-u ON | ok | 0.25536 | 0.005017 |

### Mach 3 Step, 240x80 logical, 32256 triangles

| Method | Status | Flag Proxy | Carbuncle Proxy |
|---|---:|---:|---:|
| MLP-u1 | ok | 0.09959 | 0.015984 |
| MLP-u2 | ok | 0.09713 | 0.017572 |
| T-MLP-u OFF | NaN | 0.0 | - |
| T-MLP-u ON | ok | 0.10811 | 0.018765 |

## Audit Conclusion

The current artifacts satisfy the benchmark evidence gate.  The unwrapped
pure downwind/SUPERBEE variants diverge, while the T-MLP-u wrapper remains
finite on the paper-scale triangle meshes.  T-MLP-u ON now has the lowest
LeVeque L1 error, the strongest Double Mach vortex proxy, and the strongest
Mach 3 step flag-waving proxy among the compared schemes.  The Mach 3
carbuncle proxy is finite and small but not the minimum; the paper should
frame this case as stronger flag-wave resolution with stable, non-divergent
shock handling rather than as the globally lowest pressure-checker metric.
