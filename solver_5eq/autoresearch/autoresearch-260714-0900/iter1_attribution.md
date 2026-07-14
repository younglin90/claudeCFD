# Iter1 — term-level attribution of the AW pressure packet inside the acoustic solve

Scope: term-level attribution only (not a fresh root-cause). Everything about the
packet's identity (pressure-only, 8-10 cell, born ~step 1080 at interface cell 133,
final p = raw acoustic-solve output) is taken as established in `docs/diag_aw_packet.md`
and `docs/research_aw_ringing_campaign.md`. This iteration answers: **which residual
term carries it**, and proposes the most targeted parameter-free fix.

## Method

Temporary env-gated instrumentation `FIVE_EQ_DIAG_TERMS` in `_solve_acoustic_ad`
(`solver/five_eq_IMEX/imex_ad.py`, after the linear solve, AW-only gate
`bc_l=='reflective' and n==400`). For steps 1000-1200, cells 110-175, it dumps the
three p/u-row terms recomputed at the **converged** state plus pre-solve faces,
coefficients, masks and `p_new-p0`. Post-solve faces are rebuilt with the production
reconstruction helper `_acoustic_faces_muscl_np` (default `component`+`superbee`+
`centered_interface`, identical to the torch residual). Offline
(`analyze_terms.py`) each term is compared to the exact d'Alembert term built from
the verifier `_exact_07`.

The residual (`imex_ad.py:3029`, torch path):
```
r_p = p_c - p0 + dt*( u0*dp_dx  +  beta*div_u )      beta = rho*c^2
r_u = rho*u_c - m_adv + dt*div_p
div_u = theta*(u_fr-u_fl)/dx + (1-theta)*du_old ,  faces = Z-Riemann on MUSCL states
```

Artifacts (fixed overwrite paths):
- `results/1D/07_B/diag_terms.npz` — 201 steps x window arrays
- `results/1D/07_B/diag_terms.png` — 6-panel (birth curves, term vs exact, p-increment
  decomposition, beta notch, alpha vs tolerances, p vs exact)

Faithfulness cross-check: at step 1150 `dp_num == -dt*comp` overlies to plotting
accuracy (panel 3); `max|r_p_check|/max|dp_num|` over all steps = 0.19, the residual of
the single Newton linearization concentrated at the sharpest birth transient (step
~1090), negligible elsewhere.

## Answers

### Q1 — which term deviates from exact linear-acoustics first / carries the packet
**The compression term `beta*div_u`, overwhelmingly.** Water window (cells 133-162):

| step | \|adv\|max | \|comp\|max | \|div_p\|max | HF \|comp\| | HF \|adv\| | HF dp_num |
|-----:|-----------:|------------:|-------------:|------------:|-----------:|----------:|
| 1050 | 4.2e-10 | 5.5e+02 | 2.9e-01 | 3.95e+02 | 5.1e-07 | 3.5e-04 |
| 1090 | 3.5e-09 | 6.6e+02 | 4.3e-01 | 8.24e+02 | 8.0e-05 | 1.0e-03 |
| 1100 | 2.5e-09 | 3.9e+02 | 2.5e-01 | 2.33e+04 | 1.1e-02 | 2.2e-02 |
| 1125 | 4.6e-04 | 5.6e+05 | 2.9e+02 | 4.13e+05 | 7.4e-01 | 5.6e-01 |
| 1150 | 1.1e-03 | 3.2e+05 | 2.0e+02 | 1.59e+05 | 3.1e-01 | 1.6e-01 |

- `comp = beta*div_u` is ~**10^9** larger than `adv = u0*dp_dx` and ~**10^3** larger
  than the momentum-row `div_p`. The p-increment is entirely the compression term:
  `p_new-p0 == -dt*comp` (panel 3), `-dt*adv` is a flat zero.
- The near-Nyquist (d2) HF content of `comp` tracks the HF of `p_new` step-for-step;
  the HF of `adv` is 5-6 orders smaller. **The packet is the HF of `beta*div_u`.**
- **The advection/upwind `dp_dx` stencil (Q4) is definitively NOT the seed** — it is
  dynamically dead in the stiff water bulk.

### Q2 — is the packet in the faces pre-solve, or created by the solve?
**Already present pre-solve, self-sustaining; the CN solve does not create it (mostly
damps it).** HF of the pre-solve face divergence `div_u_old` vs HF of the post-solve
p-increment `dp_num`:

| step | HF div_u_old (pre) | HF dp_num (post) | post/pre |
|-----:|-------------------:|-----------------:|---------:|
| 1100 | 1.24e-01 | 2.23e-02 | 0.18 |
| 1125 | 1.86e+00 | 5.63e-01 | 0.30 |
| 1150 | 1.86e+00 | 1.57e-01 | 0.08 |

The packet is carried in `div_u_old` (built from the previous step's converged faces) at
every step; `post/pre < 1` shows the theta=0.5 solve is weakly dissipative on it, not the
source. The single exception is the birth-transient step 1090 (post/pre=18.7) where the
arriving wave first excites the reconstruction. Mechanistically this is a phase-coherent
error that re-enters each step through `(1-theta)*du_old` and the reconstructed
`u_fr-u_fl`, matching the campaign-doc "accumulates over ~1600 steps" picture.

### Q3 — alpha, and a beta/Z single-cell glitch at the transition cell
There is **exactly one transition cell (133)**; cell 132 is pure air (alpha=1),
cells >=134 are pure water (air-fraction pinned at 1e-8). alpha[133] drifts
**monotonically** 1.00e-8 -> 1.70e-7 (s1125) -> 4.94e-7 (s1150) -> 8.68e-7 (s1200) —
no per-step flicker (confirms H3-was-not-a-flicker).

**But a real single-cell beta/Z notch exists at cell 133**, `beta/beta_water`:

| step | c132(air) | **c133** | c134 | c135 | alpha[133] |
|-----:|----------:|---------:|-----:|-----:|-----------:|
| 1090 | 1e-4 | **0.9998** | 0.9998 | 0.9998 | 1.00e-8 |
| 1125 | 1e-4 | **0.9970** | 1.0000 | 1.0000 | 1.70e-7 |
| 1150 | 1e-4 | **0.9914** | 1.0000 | 1.0000 | 4.94e-7 |
| 1200 | 1e-4 | **0.9850** | 1.0000 | 1.0000 | 8.68e-7 |

Root: a **tolerance mismatch**. `_phase_acoustic` (`explicit.py:32-38`) switches to the
pure-phase EOS with the **raw** `alpha_pure_tol=1e-8`, but the acoustic face masks
`_same_pure_material_face_mask` / `_pure_bulk_muscl_face_mask` call a cell pure with
`max(alpha_pure_tol, eps**0.25) = 1.22e-4`. When alpha[133] drifts into
`(1e-8, 1.22e-4)` the cell is **reconstructed as pure** (high_face[133]=1) yet given a
**Wood-mixture beta/Z that dips up to 1.5%** — a persistent single-cell impedance notch
anchored exactly where the packet is anchored, on the non-dissipative CN branch. This
was not in the 9-candidate ledger. Note it *lags* the packet explosion (0.02% at s1090
vs 1.5% at s1200), so it is a growing **amplifier**, not the initial excitation.

### Q4 — the `dp_dx` (advection) stencil
`imex_ad.py:3021-3025`:
```
p_l_eff = where(lb, p_fl_b, p_l);  p_r_eff = where(rb, p_fr_b, p_r)
dp_back = (p_c - p_l_eff)/dx ;  dp_forw = (p_r_eff - p_c)/dx
dp_dx   = where(up_i>0.5, dp_back, dp_forw)      # up_i = sign(u0), frozen at anchor
```
A frozen-sign one-sided (upwind) first difference; centered only through the boundary-
adjusted effective neighbours (irrelevant for interior cell 133). Its whole contribution
`u0*dp_dx` is ~1e-9..1e-3 in water — **no correlation with packet birth** (green curve,
panel 1, floor ~1e-9). Ruled out as seed.

### Q5 — theta / wave gating / masks / dp_old-du_old
- `theta_cell = 0.5` (Crank-Nicolson) on **all** near-interface cells 130-137, every
  step (`wave_cell=1` everywhere in the wave) — the packet band is on the
  non-dissipative CN branch, as assumed.
- masks at the interface (faces 131-136): `same_face=[1 1 0 1 1 1]` (only the true
  air/water face 133 is not same-pure) but `high_face=[1 1 1 1 1 1]` — **the impedance-
  jump face 133 still receives high-order superbee reconstruction** (pure_bulk mask's
  loose tol counts cell 133 as pure water). This is the reconstruction that excites the
  packet, consistent with the H4 limiter sweep.
- `du_old`/`dp_old` carry the packet each step (Q2); nothing anomalous beyond that.

## Attribution verdict

The packet is the near-Nyquist content of the **compression / pressure-work term
`beta*div_u`** in the implicit pressure row — *not* the advection term `u0*dp_dx`
(9 orders down) and *not* the momentum-row `div_p` (3 orders down). Within `beta*div_u`
there are two coupled contributors, both localized at the interface:
1. **(excitation, primary)** the superbee-MUSCL + Z-Riemann reconstruction of the
   velocity divergence across the 3800:1 impedance jump at the high-order interface face
   133; the resulting near-Nyquist mode is not damped by the CN (theta=0.5) solve and
   re-enters via `du_old` each step (Q1/Q2/Q5). This is the mechanism the campaign's H4
   / limiter sweep already implicated.
2. **(amplifier, new + parameter-free-fixable)** a single-cell `beta/Z` notch at the
   interface cell 133 from the `_phase_acoustic` vs face-mask pure-tolerance mismatch
   (Q3), growing to 1.5% and anchored exactly at the packet root.

## Tested candidate — H11a: acoustic pure-tolerance consistency

The most targeted *parameter-free* fix pointed to by the attribution that is **not** in
the failed ledger: make `_phase_acoustic`'s pure-branch use the **same** derived tol
`max(alpha_pure_tol, eps**0.25)` the acoustic face masks already use, so the interface
water cell keeps pure-water `beta/Z` (removes the notch). Env-gated
`FIVE_EQ_IMEX_ACOUSTIC_PURE_TOL_CONSISTENT` (default off = bit-identical); inert whenever
`alpha_pure_tol >= eps**0.25` (so 02_A alpha_floor=1e-3 is untouched by construction).
`eps**0.25` is a constant already in the codebase — no new tuned parameter. ~20 lines,
`imex_ad.py` after the `beta` assignment.

AW N=400 result (`FIVE_EQ_CASE07_ONLY=Air-Water FIVE_EQ_IMEX_ACOUSTIC_PURE_TOL_CONSISTENT=1`):

| metric | baseline | H11a | limit |
|--------|---------:|-----:|------:|
| p_smooth_local_tv_excess | 0.53549 | **0.53447** | <= 0.30 |
| L2p | 0.09004 | 0.08996 | < 0.216 |
| amp p / u | 1.00 / 0.97 | 1.00 / 0.97 | [0.85,1.10] |
| peak / symmetry | ok / 0.138 | ok / 0.138 | ok / <=0.38 |
| aw_wiggle | FAIL | **FAIL** | pass |

**Verdict: rejected (ineffective).** Removing the notch moves `tv_excess` by only
~0.2% (0.53549 -> 0.53447). This is a direct, quantitative confirmation that the
`beta/Z` notch is a **negligible amplifier, not the seed**: the packet is carried by
the *reconstructed* `div_u` across the impedance jump (contributor 1), not the
coefficient. The fix is still a legitimate latent consistency bug and is left in as a
default-off env option; it does not fix the guard.

02_A golden with candidate active: `p_rel_linf = 2.764863893389702e-15` (**unchanged**,
inert by construction since `alpha_floor=1e-3 >= eps**0.25`).

## Next candidate (specified, not yet run) — H11b: Rhie-Chow retention in `div_u`

The attribution isolates the mechanism precisely: the Z-Riemann velocity face
`u_f = (p_l - p_r + Z_l u_l + Z_r u_r)/den` contains a `(p_l - p_r)/den`
Rhie-Chow pressure-coupling term that damps odd-even `p`. On **same-pure high-order
faces** (water faces 134+, `same_face=1 & high_face=1`) the componentwise MUSCL
half-states make `p_lh - p_ch ~ O(dx^2)`, cancelling that coupling on smooth data ->
the Nyquist `p` mode goes undamped and feeds `beta*div_u`. This is exactly the
campaign-doc §5 "equivalent framing", now confirmed at the term level:
**blend back a wavenumber-selective fraction of the first-order Z-Riemann pressure
coupling into the `u`-face used by `div_u` on same-pure faces** (equivalently a
`+dt*eps_bh*dx^3*d4p/dx4` biharmonic term in `r_p` on `_same_pure_material_face_mask`
cells). This is `k^4`-selective (near-invisible to the resolved wave, strong on the
8-10 cell band). Parameter-free realization to pin down next: derive `eps_bh` from the
scheme's own dispersion (CFL/stencil) rather than a tuned value; disabling high-order
`u`-reconstruction on the single interface face alone is insufficient (that is
ledger-P1, already rejected), so the term must act on the same-pure water faces.

## Files
- instrumentation + candidate: `solver/five_eq_IMEX/imex_ad.py` (both env-gated;
  diag block reverted after this iter — see revert note in the run log)
- analysis: `analyze_terms.py` (scratchpad), plots `results/1D/07_B/diag_terms.png`
