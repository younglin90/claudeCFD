# T-MLP-u Paper Benchmark Audit

Generated from `results/T-MLP-u/paper_benchmark_summary.json` on the
full paper grids.

## Concrete Objective Checklist

| Requirement | Evidence | Status |
|---|---|---|
| All benchmark meshes are unstructured triangles | LeVeque `criss_cross_triangles`; Double Mach and Mach 3 step `tri_alternating` | PASS |
| All benchmark reconstructions use `order=1` | Harness comparison definition records `order=1`; LeVeque OFF was corrected to `stencil='vertex', order=1` | PASS |
| T-MLP-u ON uses `vertex_mlp=True`, `virtual_uu_gradient=True`, `stencil='vertex'` | Harness constructors `_tmlpu_leveque`, `_tmlpu_double_mach`, `_tmlpu_mach3_step` | PASS |
| LeVeque uses N=100 and central-averaged face velocity | Summary: `logical_nx=100`, `mesh_cells=40000`; comparison definition: `upwind with central-averaged face velocity` | PASS |
| LeVeque OFF is pure downwind with `mlp_bound=False` | OFF row fails with `NaN at step 1141, t=0.3644062677` | PASS |
| LeVeque ON is pure downwind plus T-MLP-u with `mlp_bound=True`, `tvb_M=0`, `extremum_relax=False` | ON row `ok=true`, `L1=0.02059944`, `wiggle=1.88e-11` | PASS |
| LeVeque ON is sharper and wiggle-free compared with standard unstructured limiters | ON `sharpness=0.047899`; best baseline sharpness is MLP-u1 `0.031245`; ON wiggle is near zero | PASS |
| Double Mach uses paper-scale triangle grid | Summary: `480x120`, `115200` triangles, domain/setup matches Woodward-Colella | PASS |
| Double Mach uses primitive reconstruction and HLLC-family low-diffusion flux | Comparison definition: primitives `W=(rho,u,v,p)` and flux `hllc_adc` | PASS |
| Double Mach OFF is pure SUPERBEE with `mlp_bound=False` | OFF row fails with `NaN at step 40, t=0.0029414425` | PASS |
| Double Mach ON is SUPERBEE plus T-MLP-u and reaches `t=0.2` | ON row `ok=true`, `steps=3851`, no NaN | PASS |
| Double Mach ON is sharper than BJ/Venkat/MLP-u1/MLP-u2 in the lower-right vortex metric | ON `vortex_proxy=0.357617`; best baseline MLP-u1 `0.253303` | PASS |
| Mach 3 step uses paper-scale triangle grid | Summary: `240x80` logical grid after step cutout, `32256` triangles | PASS |
| Mach 3 step uses primitive reconstruction and HLLC-family low-diffusion flux | Comparison definition: primitives `W=(rho,u,v,p)` and flux `hllc_adc` | PASS |
| Mach 3 step OFF is pure SUPERBEE with `mlp_bound=False` | OFF row fails with `NaN at step 25, t=0.0075811554` | PASS |
| Mach 3 step ON is SUPERBEE plus T-MLP-u and reaches `t=4` | ON row `ok=true`, `steps=11241`, no NaN | PASS |
| Mach 3 step ON resolves stronger flag-waving structure than BJ/Venkat/MLP-u1/MLP-u2 | ON `flag_proxy=0.141894`; best baseline MLP-u1 `0.099588` | PASS |
| Required all-scheme PNG comparisons are saved under `results/T-MLP-u` | `leveque_scheme_contours.png`, `double_mach_scheme_contours.png`, `mach3_step_scheme_contours.png`, plus ON-only PNGs | PASS |
| Overall manifest marks the evidence complete | `evidence_ready=1`, `fail_count=0`, all three case-ready flags are `1` | PASS |

## Current Numerical Summary

### LeVeque Rotation, N=100

| Method | Status | L1 | Sharpness | Wiggle |
|---|---:|---:|---:|---:|
| first order | ok | 0.07327262 | 0.014189 | 0 |
| Barth-Jespersen | ok | 0.05167664 | 0.020336 | 6.53e-32 |
| Venkatakrishnan | ok | 0.05574759 | 0.018993 | 7.60e-5 |
| MLP-u1 | ok | 0.01934245 | 0.031245 | 9.48e-30 |
| MLP-u2 | ok | 0.02674952 | 0.030350 | 2.44e-3 |
| T-MLP-u OFF | NaN | - | - | - |
| T-MLP-u ON | ok | 0.02059944 | 0.047899 | 1.88e-11 |

### Double Mach Reflection, 480x120 Logical, 115200 Triangles

| Method | Status | Vortex Proxy | Vorticity p95 | Checker |
|---|---:|---:|---:|---:|
| first order | ok | 0.196061 | 53.575 | 0.005906 |
| Barth-Jespersen | ok | 0.206894 | 67.388 | 0.005038 |
| Venkatakrishnan | ok | 0.201690 | 63.518 | 0.005009 |
| MLP-u1 | ok | 0.253303 | 77.579 | 0.003620 |
| MLP-u2 | ok | 0.249389 | 76.120 | 0.003944 |
| T-MLP-u OFF | NaN | 0 | 0 | - |
| T-MLP-u ON | ok | 0.357617 | 68.356 | 0.007417 |

### Mach 3 Forward-Facing Step, 240x80 Logical, 32256 Triangles

| Method | Status | Flag Proxy | Flag Vorticity p95 | Carbuncle Proxy |
|---|---:|---:|---:|---:|
| first order | ok | 0.087667 | 18.038 | 0.017650 |
| Barth-Jespersen | ok | 0.090324 | 18.276 | 0.017530 |
| Venkatakrishnan | ok | 0.093084 | 18.653 | 0.017545 |
| MLP-u1 | ok | 0.099588 | 22.762 | 0.015984 |
| MLP-u2 | ok | 0.097129 | 19.621 | 0.017572 |
| T-MLP-u OFF | NaN | 0 | 0 | - |
| T-MLP-u ON | ok | 0.141894 | 18.232 | 0.026344 |

## Audit Conclusion

The full-grid artifacts satisfy the current paper benchmark evidence gate.
The unwrapped pure downwind/SUPERBEE variants diverge in all three stress
tests, while the T-MLP-u wrapper remains finite on the same triangle meshes.
T-MLP-u ON gives the strongest LeVeque interface sharpness proxy, the
strongest Double Mach lower-right vortex proxy, and the strongest Mach 3
step flag-waving proxy among the compared unstructured limiters.

The Mach 3 pressure checker/carbuncle proxy is finite but not the minimum;
the paper claim should frame this case as stable high-compression shock
handling with stronger flag-wave resolution, not as the globally lowest
checker metric.
