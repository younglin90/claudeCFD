# T-MLP-u Paper Verification Plan — LeVeque Rotation Benchmark

## Goal

Generate paper-publication-ready verification of the T-MLP-u (T-MLP-u, Park-Yoon-Kim 2010 + extension) reconstruction scheme on the LeVeque (1996) solid-body-rotation 2D advection benchmark.

**Critical claim**: T-MLP-u **on/off** must produce a stark contrast:
- **OFF (no T-MLP-u LMP wrapper)**: numerical instability / divergence / oscillation
- **ON (T-MLP-u LMP wrapper active)**: stable + best L1 error among references

## Scope (modifiable)

```
solver/solve_T-MLP-u/
├── reconstruction.py          # TMLPU class (3-tier adaptive limiter, vertex2 LSQ k=3)
├── limiters.py                # cicsam_co38, downwind, pure_downwind, van_leer, minmod
├── flux.py                    # upwind / central
├── solver.py                  # solve() FVM driver
├── tests/test_2d_leveque_rotation.py  # 4-case driver (parallel)
├── AUTORESEARCH_PLAN.md       # iter 0-28 history + final config
└── autoresearch_log_N100.tsv  # quantitative log (read-only by agent)
```

수정 금지: `solver/He2024/`, `solver/denner_1d/`, `validation/`, `archive/`, `백업_*/`.

## Current Best (paper baseline, iter 26)

T-MLP-u FINAL config:
- 3-tier adaptive limiter:
  - sharp (LSQ residual ≥ 0.10): cicsam_co38 (Hyper-C @ Co=0.38)
  - moderate (0.05 ≤ residual < 0.10): van_leer
  - very-smooth (residual < 0.05): minmod
- LMP wrapper: vertex 1-ring bound + TVB tolerance M=64
- LSQ: cubic k=3, vertex2 stencil, IDW p=6, virtual UU fallback
- SSP-RK3, CFL=0.4, 2-pt face Gauss

**Current N=100 result**: TOTAL L1 = 0.02008, range [-0.005, 1.002], drift 1.2e-8.

## Tasks (paper figure / table generators)

### Task 1: T-MLP-u on/off contrast (CORE PAPER FIGURE)

Run 4-case comparison `tests/test_2d_leveque_rotation.py` at N=100 with:
- A: T-MLP-u **ON** (3-tier adaptive + LMP) — paper headline
- B: T-MLP-u **OFF**, with same limiter (cicsam_co38) — should DIVERGE
- C: T-MLP-u OFF, with safer limiter (Roe ultrabee/downwind) — stable but worse
- D: pure CICSAM no LMP — divergent oscillation (range > 2)

Expected: A << C < B/D in TOTAL L1.  B/D ranges show overshoot or NaN.

Deliverable: `solver/solve_T-MLP-u/tests/output/2d_leveque_rotation.png` (overwrite each round).

### Task 2: T-MLP-u + pure_downwind verification

User hypothesis: pure_downwind (ψ≡2 always) + T-MLP-u LMP could be MORE compressive than CICSAM + LMP (because LMP clips overshoots; ψ≡2 means max compression even at low r).

Action: add Case B variant `tvd='pure_downwind'` with full T-MLP-u 3-tier wrapper, run alongside Case A (cicsam_co38 baseline).  Compare TOTAL L1 + per-shape (slot/cone/hump) + range monotonicity.

If pure_downwind + T-MLP-u beats cicsam_co38 + T-MLP-u → adopt for paper.
Else → keep cicsam_co38 as paper baseline, document the test.

### Task 3: N convergence study (paper-essential)

Run T-MLP-u FINAL at N ∈ {25, 50, 100, 200} (same config), produce log-log convergence plot.

For each N, capture:
- TOTAL L1
- per-shape L1 (slot, cone, hump)
- Convergence rate (slope of log L1 vs log N)

Expected order: ~1 (slot, sharp interface limited by limiter), ~2-3 (cone/hump, smooth-region cubic LSQ).

Deliverable: `results/leveque_convergence.png` log-log plot, table of L1 per N.

### Task 4: Documentation polish

Update `solver/solve_T-MLP-u/AUTORESEARCH_PLAN.md` with:
- Final paper-ready 4-case table from Task 1
- pure_downwind result from Task 2
- Convergence study result from Task 3
- Cross-references to solver/limiter implementation files

## Constraints

- 4-case parallel runs already working (~38 min wall on 4 worker × N=100)
- Code changes minimal — most work is running tests + analyzing.
- Each iter commit: `experiment: ...` prefix.
- Plot saved at fixed path `solver/solve_T-MLP-u/tests/output/2d_leveque_rotation.png` (overwrite).
- Convergence study new path `results/leveque_convergence.png`.

## Success Criteria

1. **Task 1**: Paper-ready figure showing T-MLP-u ON wins (sharp + monotone) and OFF diverges/oscillates.
2. **Task 2**: pure_downwind comparison documented — best limiter for sharp arm identified.
3. **Task 3**: Convergence plot shows expected order (~1 slot, ~2-3 smooth).
4. **Task 4**: AUTORESEARCH_PLAN.md updated to reflect all paper data.

## Stop Condition

All 4 tasks deliverables produced. Or: agent identifies no further marginal improvement (plateau).
