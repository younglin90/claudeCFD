# T-MLP-u Paper Benchmark Audit

Generated from `results/T-MLP-u/paper_benchmark_summary.json` on the
current half-resolution feedback grids.

## Concrete Objective Checklist

| Requirement | Evidence | Status |
|---|---|---|
| All benchmark meshes are unstructured triangles | LeVeque `criss_cross_triangles`; Double Mach and Mach 3 step `tri_alternating` | PASS |
| Reduced feedback grids are used | LeVeque `N=50`; Double Mach `240x60`; Mach 3 step `120x40` | PASS |
| T-MLP-u ON uses `vertex_mlp=True`, `virtual_uu_gradient=True`, `stencil='vertex'` | Harness constructors `_tmlpu_leveque`, `_tmlpu_double_mach`, `_tmlpu_mach3_step` | PASS |
| MLP-u2 uses vertex one-ring, not two-ring | `MLPU2` is `stencil='vertex'`, `n_rings=1` | PASS |
| `virtual_uu_r_floor` is removed | No `virtual_uu_r_floor` option remains in `reconstruction.py` or benchmark config | PASS |
| LeVeque OFF is pure downwind with `mlp_bound=False` | OFF row fails with `NaN at step 1117, t=0.7158767977` | PASS |
| LeVeque ON is pure downwind plus T-MLP-u with `mlp_bound=True`, `tvb_M=0`, `extremum_relax=False` | ON row `ok=true`, `L1=0.03841345`, `wiggle=3.60e-12` | PASS |
| LeVeque ON is sharper and wiggle-free compared with standard unstructured limiters | ON `sharpness=0.074595`; best baseline sharpness is MLP-u1 `0.053449`; ON wiggle is near zero | PASS |
| Double Mach uses triangle grid and Woodward-Colella setup | Summary: `240x60`, `28800` triangles, domain/setup recorded | PASS |
| Double Mach uses primitive reconstruction and HLLC-family low-diffusion flux | Comparison definition: primitives `W=(rho,u,v,p)` and flux `hllc_adc` | PASS |
| Double Mach OFF is TVD-only modified SUPERBEE with `mlp_bound=False` | OFF row fails with `NaN at step 79, t=0.0110358631` | PASS |
| Double Mach ON is modified SUPERBEE plus T-MLP-u and reaches `t=0.2` | ON row `ok=true`, `steps=1831`, no NaN | PASS |
| Mach 3 step uses triangle grid | Summary: `120x40` logical grid after step cutout, `8064` triangles | PASS |
| Mach 3 step uses primitive reconstruction and HLLC-family low-diffusion flux | Comparison definition: primitives `W=(rho,u,v,p)` and flux `hllc_adc` | PASS |
| Mach 3 step contours are density contours | `mach3_step_tmlpu_on.png` and `mach3_step_scheme_contours.png` plot density `rho` | PASS |
| Mach 3 step OFF is TVD-only modified SUPERBEE with `mlp_bound=False` | OFF row fails with `NaN at step 41, t=0.0158764896` | PASS |
| Mach 3 step ON is modified SUPERBEE plus T-MLP-u and reaches `t=4` | ON row `ok=true`, `steps=5383`, no NaN | PASS |
| Required all-scheme PNG comparisons are saved under `results/T-MLP-u` | `leveque_scheme_contours.png`, `double_mach_scheme_contours.png`, `mach3_step_scheme_contours.png`, plus ON-only PNGs | PASS |
| All scheme outputs are saved as ParaView-readable VTK | `results/T-MLP-u/vtk` contains 21 legacy ASCII `UNSTRUCTURED_GRID` files | PASS |
| Divergent OFF schemes are still represented in VTK | `t_mlp_u_off.vtk` files contain `status_ok=0`, `diverged=1`, and initial-state fields | PASS |
| Overall manifest marks the evidence complete | `evidence_ready=1`, `fail_count=0`, all three case-ready flags are `1` | PASS |

## Current Numerical Summary

### LeVeque Rotation, N=50

| Method | Status | L1 | Sharpness | Wiggle |
|---|---:|---:|---:|---:|
| first order | ok | 0.09140343 | 0.020832 | 0 |
| Barth-Jespersen | ok | 0.06933991 | 0.030134 | 0 |
| Venkatakrishnan | ok | 0.07380218 | 0.028107 | 2.40e-4 |
| MLP-u1 | ok | 0.03430912 | 0.053449 | 3.28e-31 |
| MLP-u2 | ok | 0.04020085 | 0.047801 | 2.93e-3 |
| T-MLP-u OFF | NaN | - | - | - |
| T-MLP-u ON | ok | 0.03841345 | 0.074595 | 3.60e-12 |

### Double Mach Reflection, 240x60 Logical, 28800 Triangles

| Method | Status | Vortex Proxy | Vorticity p95 | Checker |
|---|---:|---:|---:|---:|
| first order | ok | 0.383246 | 46.381 | 0.016785 |
| Barth-Jespersen | ok | 0.393859 | 56.594 | 0.012746 |
| Venkatakrishnan | ok | 0.381685 | 55.466 | 0.012412 |
| MLP-u1 | ok | 0.454534 | 63.886 | 0.010918 |
| MLP-u2 | ok | 0.459668 | 60.800 | 0.010996 |
| T-MLP-u OFF | NaN | 0 | 0 | - |
| T-MLP-u ON | ok | 0.455781 | 57.862 | 0.012567 |

### Mach 3 Forward-Facing Step, 120x40 Logical, 8064 Triangles

| Method | Status | Flag Proxy | Flag Vorticity p95 | Carbuncle Proxy |
|---|---:|---:|---:|---:|
| first order | ok | 0.152376 | 13.962 | 0.038562 |
| Barth-Jespersen | ok | 0.163012 | 12.870 | 0.039990 |
| Venkatakrishnan | ok | 0.155974 | 13.021 | 0.038483 |
| MLP-u1 | ok | 0.179905 | 15.918 | 0.043941 |
| MLP-u2 | ok | 0.188757 | 17.935 | 0.040482 |
| T-MLP-u OFF | NaN | 0 | 0 | - |
| T-MLP-u ON | ok | 0.201800 | 17.911 | 0.040585 |

## VTK Artifacts

Each case has seven scheme files:

- `results/T-MLP-u/vtk/leveque/*.vtk`
- `results/T-MLP-u/vtk/double_mach/*.vtk`
- `results/T-MLP-u/vtk/mach3_step/*.vtk`

Successful runs contain final fields (`phi` for LeVeque; `rho`, `u`, `v`,
`p`, and `velocity` for Euler cases). Divergent OFF runs contain diagnostic
initial-state fields plus `status_ok` and `diverged`, because no finite final
solution exists after NaN.

## Audit Conclusion

The current half-grid artifacts satisfy the benchmark evidence gate. The
unwrapped pure downwind / TVD-only modified SUPERBEE variants diverge in all
three stress tests, while the T-MLP-u wrapper remains finite on the same
triangle meshes. T-MLP-u ON gives the strongest LeVeque interface sharpness
proxy and remains essentially wiggle-free. In Mach 3 step, T-MLP-u ON gives
the strongest density flag-wave proxy among the compared limiters.

For Double Mach on this reduced grid, T-MLP-u ON is comparable to the best
MLP-u baselines in the lower-right vortex proxy, while the decisive evidence
is robustness: the same modified SUPERBEE reconstruction without the T-MLP-u
bound diverges early.
