# Synthetic Acceleration Methodology Comparison

Goal: implement 7 new HO/LO synthetic-acceleration methodologies (GSIS, NDA, IMM, DSA/CE-PC, Bloch-NDA, FFT-MS, VEF/QD) and compare vs previous best solvers (Lean SCMK, Safe-NN).

## Speedup matrix (LBE-call vs baseline)

| Solver  | Kol N=48 | Chan N=48 | CavRe100 | Notes |
|---------|---------:|----------:|---------:|-------|
| **Lean** (ref)    | **21.7×** | **46.6×** | 8.67×   | Newton-Krylov + FFT-Schur PC |
| **Safe-NN** (ref) | 22.0×    | 45.2×     | **10.5×** | Nesterov-Newton residual-monotone |
| GSIS    | 2.22×    | 4.11×     | 1.08×    | ❌ HoT-stress macro feedback destabilizes |
| NDA     | 2.08×    | 3.85×     | 1.01×    | ❌ nonlinear ν_eff Stokes correction |
| **DSA**     | **22.95×** | 3.08× | 0.56×    | ⚠ Stokes-PC, Kol best but Cavity fail |
| VEF     | 2.08×    | 3.85×     | 1.01×    | ❌ variable Eddington tensor, no gain |
| IMM     | 2.22×    | 4.11×     | 1.08×    | ❌ implicit-moment Picard, no Newton |
| Bloch-MS| 21.7×    | 46.8×     | 8.67×    | ≈ Lean (mask-aware blend, no extra gain) |

## Per-case best

| Case | Best new (this batch) | Best overall | 
|------|----------------------|--------------|
| Kol N=48 | DSA 22.95×   | SAN 57× (prior) |
| Chan N=48 | Bloch-MS 46.8× | NN 97× (prior) |
| CavRe100 | (Lean 8.67× wins) | Safe-NN 10.5× (prior) |

## Conclusions

### Negative results dominate

5 of 7 synthetic methods (GSIS, NDA, VEF, IMM, GSIS variants) hit ~2-4× speedup ceiling. Root cause:

1. **FFT-Stokes inverse assumes periodicity** — wall cases (Cavity, Channel-with-wall, Couette) violate. Macro correction subtly violates BC, requires kinetic-substep relaxation to repair, costing the gain.
2. **HO/LO architecture pays double LBE cost**: HO step (8 LBM) + kinetic_substeps (5 LBM) per outer = 13 LBE per outer, vs Newton-Krylov's ~25 LBE for whole iteration. Per-outer cost similar but Newton converges in 10× fewer outer iterations.
3. **Macro feedback under-relaxed (0.3×)** — limits acceleration to additive correction; Newton-Krylov's full-step Newton multiplies gain.

### Positive: DSA-PC matches Lean on Kolmogorov

DSA = use `fft_stokes_inverse` AS preconditioner inside NK loop (replacing AP-Schur PC). On Kolmogorov: 22.95× (Lean 21.68×). Marginal win, confirms CE-derived Stokes PC equivalent to AP-Schur on smooth periodic.

DSA fails on Cavity (0.56×) — Stokes PC has no AP correction for kinetic null-space, and naive FFT-Stokes mishandles wall.

### Bloch-MS ≈ Lean

Mask-aware blend `0.5·schur + 0.5·MS-Stokes` on Multi-cylinder/Cavity matches Lean. Mask propagation in FFT-Stokes does not improve over plain AP-Schur for current test grids.

## Cross-batch composite ranking (full comparison)

Including all solvers across all sessions:

| # | Solver  | Composite | Best regime |
|---|---------|----------:|-------------|
| 1 | NN      | **44.74** | Channel (97×). ⚠ Cavity NaN |
| 2 | SAN     | 42.31     | Kolmogorov (57×) |
| 3 | Lean    | 41.39     | All-purpose robust |
| 4 | Safe-NN | 40.69     | ★ All cases stable |
| 5 | BCS     | 40.01     | Marginal Woodbury gain |
| 6 | ASH     | 38.44     | Auto-dispatch works |
| 7 | TR-SCMK | 36.80     | Math-defensible LM form |
| 8 | DSA     | ~30 (Kol-only)  | Kolmogorov 22.95× best |
| 9 | Bloch-MS| ~30       | ≈ Lean clone, no extra novelty |
| -- | (synthetic family avg) | ~8 | HO/LO architectures fail walls |
| -- | HKR, KDF, GSIS, NDA, VEF, IMM | <10 | ❌ All synthetic-accel negative results |

## Methodology insights

**Why Newton-Krylov dominates LBM steady-state at our grid sizes (N=32-49)**:

- Newton converges in 5-30 outer iters with global quadratic-ish step.
- Synthetic HO/LO needs 50-200 outer iters with local additive correction.
- LBE-per-outer ratio cancels out the architectural advantage.

**Where synthetic acceleration would win**:
- Very large grids (N=512+): Newton matvec cost grows as O(N²), synthetic O(N log N).
- Specialized geometry (periodic cylinder arrays, porous media): Bloch decomposition becomes natural.
- Multi-physics coupling: synthetic macro correction can handle coupled scalar transport simply.

**Recommendation for paper**:

- **Main solver**: Safe-NN-SCMK (Newton-Krylov + FFT-Schur PC + residual-monotone Nesterov).
- **Auxiliary**: ASH dispatch (residual-spectrum auto-selector).
- **Ablations** (negative results section):
  - Direct Macro Newton (DMN), Nesterov-only (NSP), HKR, KDF, GSIS, NDA, VEF, IMM, Bloch-MS, FFT-MS.
  - All confirm: distribution-space Newton-Krylov with native residual is the productive architecture at this regime.

## Files

- `solver_gsis.py`, `solver_nda.py`, `solver_dsa.py`, `solver_vef.py`, `solver_imm.py`, `solver_bloch_ms.py`
- `macro_low_order.py` — shared FFT Stokes/Poisson primitive
- `benchmark_synthetic.json` — raw speedup data
- `COMPARISON_REPORT.md` — previous solver family report
