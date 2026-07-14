# Eliminating the tuned dissipation coefficient D — research conclusion

Date: 2026-07-13. Companion document: `research_D_free_pressure_coupling.md`
(literature study, families A–F). Experiment scripts: `~/amp_dfree_experiment.py`,
`~/amp_dfree_phase2.py`, `~/amp_dfree_phase2c.py` (WSL home).

## Question

Can the solver suppress the collocated pressure odd–even (checkerboard) mode
without the user-tuned biharmonic coefficient `imp_dissipation D = 0.02`?

## Answer: the production method already does. D only inhabits the analysis path.

### Finding 1 — the validated production path never uses D (code-verified)

`main.py` dispatch: `time_integrator='imex_ssp3'` with
`FIVE_EQ_IMEX_SSP3_FORM=shu_osher` (the default, used by the entire evidence
package) calls `imex_ad_ssp3_step` **without threading `imp_dissipation` at
all** (`main.py:483-493`). The same holds for `imex_ad` and `imex_ad_ssp2`.
Checkerboard control in this family comes from two parameter-free mechanisms:

- Z-weighted acoustic-Riemann face closure (`imex_ad.py:503-505`):
  `p̂_f = (Z_R p_L + Z_L p_R − Z_L Z_R Δu)/(Z_L+Z_R)`, `Z = ρc` from the EOS.
- SLAU2 velocity–pressure coupling (`imex_ad.py:634-643`):
  `û_f = ū_f − (1−M̂)² Δp/(ρ̄c̄)`, all quantities derived, no free constant
  (Shima–Kitamura 2011, "parameter-free" by design).

`D = 0.02` enters only the Newton/Helmholtz family (`be1`, `ars222`,
`imex_ssp3` in `stage_residual` form) through
`residual.py::implicit_face_pu:153-169`.

Consequence for the paper: the sentence "with D = 0.02 in all computations"
(Section 3.6) is accurate only for the amplification-analysis integrator, not
for the reported production runs. This needs correction; it also means the
reference method carries **no tuned dissipation coefficient**.

### Finding 2 — empirically, D is neither sufficient nor necessary (2026-07-13)

Spectral measurement (N=8 α-jump PE state, dt=3.7e-5, harness of
`tests/test_amplification_matrix.py`):

| closure | ρ(A) |
|---|---|
| be1, D=0.02 biharmonic (current reference) | 1.000919 |
| be1, D=0 bare central | 1.001204 |
| **be1, D=0, `imp_dissipation_form='acoustic_riemann'`** | **1.000038** |
| be1, D=0, `rhie_chow=True` | 1.001409 |
| imex_ad / imex_ad_ssp3 (production) | 1.001782 |

The parameter-free acoustic-Riemann closure beats the tuned biharmonic on the
checkerboard spectral radius. (dt-sweep rows at Ca ≫ 1 are indicative only:
the FD linearization of a limiter-bearing step is noisy there; one
acoustic_riemann outlier at dt=3.7e-3 (ρ≈4.7) is suspected Newton
non-convergence within max_iter=6 and needs a converged-solve recheck.)

Full-case test (exact mirror of the `verify_02_A` gate configuration,
N=100, dt=1e-2, one transit):

| run | p_rel_linf | verdict |
|---|---|---|
| be1, D=0.02 | 5.0e+01 | diverges |
| be1, D=0 central | 9.6e+01 | diverges |
| be1, D=0 acoustic_riemann | 3.3e+298 | diverges |
| imex_ad (D-free control) | **2.765e-15** | machine precision, reproduces the gate |

The be1 α-jump long-run divergence is the known open nonlinear instability
(`CLAUDE.md` guardrails; independent of D). D=0.02 does not rescue it, so D is
not load-bearing even where it exists. A be1-family acoustic-amplitude
comparison was not obtainable for the same reason.

### Finding 3 — literature: coefficient-free replacements are standard practice

From `research_D_free_pressure_coupling.md` (88 local paper summaries + web):

1. **(A) Acoustic-Riemann implicit assembly** (Peluchon–Gallice–Mieussens
   2017; Tallois–Peluchon–Villedieu 2022): assemble the implicit acoustic
   matrix from the Z-weighted flux itself; the Δp term of `û_f` becomes a
   compact pressure Laplacian in the p-equation, removing the checkerboard
   kernel automatically. Our `acoustic_riemann` face form is this flux; full
   parity requires promoting it from a face substitution into the operator
   assembly and deleting the optional `smooth=w` knob.
2. **(F1) Compact 3-point Helmholtz stencil** (Boscheri–Pareschi 2021 Eq. 53;
   Battisti–Boscheri 2025 Eq. 31): build the Schur pressure operator as
   `(p_{i+1}−p_i)/ρ_{i+1/2} − (p_i−p_{i−1})/ρ_{i−1/2}` instead of the
   wide 2Δx product stencil: "staggered-equivalent on a collocated grid."
   Structural, coefficient-free; PE-safe (vanishes identically on uniform p).
   Our `helmholtz.py` Schur block is prepared for this but was never completed
   (roadmap `five_eq_all_mach_plan.md:356`).
3. **(B) dt-consistent momentum-weighted interpolation** (Denner ACID
   Eq. 20-21; Cubero–Fueyo; Bartholomew 2018): the same 4-point third
   difference as our biharmonic but with the coefficient **derived** as
   `Δt/(4 ρ_f Δx)`-scaled instead of a chosen 0.02. Shows the "right D" is a
   derived, CFL-dependent quantity, i.e. a fixed 0.02 is dimensionally naive.
4. **(C) SLAU2 p̃ inside the implicit block: disqualified.** Its third term
   `√(ū²)(P₅⁺+P₅⁻−1)ρ̄c̄` violates discrete pressure equilibrium at moving
   material interfaces (c jumps → M_L ≠ M_R); Deng et al. 2025 gate it off at
   interfaces for the same reason. SLAU2 stays in the explicit material flux.
5. **(D) Staggered semi-implicit** (Dumbser–Casulli 2016): removes the kernel
   structurally, but requires face-momentum storage and a re-proof of the PE
   property for the 5-equation system — fallback option only.

Caveats recorded by the literature pass: Crank–Nicolson is only A-stable, so
checkerboard damping |g|→1 as the acoustic Courant number grows (no
amplification, but no decay either; backward Euler damps strongly); and the
low-Mach θ fix must multiply only the Δu term of `p̂_f`, never the Δp term of
`û_f`, or the 2Δx null space returns.

## Recommendation

1. **Paper (required correction):** rewrite the D sentences. Accurate claim:
   the production configuration uses the parameter-free Z-weighted
   acoustic-Riemann face closure with SLAU2 coupling and contains no tuned
   dissipation coefficient; the biharmonic term (25) with D=0.02 belongs to
   the backward-Euler analysis integrator used for the amplification study.
   This converts the "why 0.02?" reviewer exposure into a strength.
2. **Code (optional, analysis path):** make `imp_dissipation_form=
   'acoustic_riemann'` with `imp_dissipation=0.0` the be1 default (spectrally
   better than D=0.02 and coefficient-free); re-verify the be1 regression
   gates. The dt=3.7e-3 spectral outlier must be rechecked with a converged
   Newton solve first.
3. **Code (structural, future):** complete the compact 3-point Helmholtz
   stencil (F1) in the Schur block — the literature-standard structural cure,
   and the most promising attack on the separate be1 α-jump nonlinear
   instability.
4. The be1 α-jump divergence is orthogonal to D and remains the open issue it
   was; no dissipation constant fixes it.
