---
description: Known pitfalls and guardrails for the Denner C++ solver and WSL toolchain
paths:
  - "cpp/denner_1d/**"
  - "*.sh"
  - "*.py"
---

# Denner Solver — Lessons Learned

Guardrails from the development history of this C++ Denner solver. These record
failed approaches and toolchain pitfalls so they are not repeated. All items are
verified facts — do not delete and do not retry the documented dead-ends blindly.

## Toolchain (WSL from Windows)

- `denner1d_validate`/`denner1d_run`/`denner1d_dump` require `DENNER_ACID=1` in the environment to
  select the ACID solver path at all. Without it, the binary silently runs a different, non-ACID
  default path and reports a plausible-looking but WRONG `pass_count` (11/19, stable and
  deterministic — not a crash, not NaN) regardless of any `ACID_YADV`/other env var. This produced a
  false "OFF path regression from 19/19 to 11/19" scare in round 20 (docs/YADV_RESEARCH.md sect.30.2)
  before being traced to the missing var, not a code bug. `scripts/yadv_r9_sweep.py`'s `base_env()`
  and `.claude/skills/yadv-round/SKILL.md` Step 6 both set it — always use one of those, or set it
  explicitly, never invoke the validate/run/dump binaries bare.
- The WSL login shell prints noise `your NNNN x1 screen size is bogus. expect trouble`
  on init; it pollutes `grep`/`cat`/`tr` output. Do NOT parse WSL command output with
  inline shell pipes for anything fragile — write a `.sh` or `.py` script file and run
  it, or use `2>/dev/null` and match only the exact expected line.
- Inline `for` loops inside `wsl.exe -d ubuntu bash -lc '...'` frequently expand loop
  variables to EMPTY (e.g. `for v in 1 2 3; do echo $v` prints blanks). Put the loop in
  a `.sh` script file instead.
- Nested single/double quotes in `bash -lc '...'` break (especially Python `-c` with
  JSON). Write a script file.
- Shell output redirection `> file` from `wsl.exe` is INTERMITTENTLY unreliable
  (produces 0-byte files). For capturing solver/dump output into Python, use
  `subprocess.run([...], capture_output=True, text=True)` inside a Python script,
  NOT shell redirection.
- Windows-side paths passed to `wsl.exe bash /home/...` get mangled by Git-Bash path
  translation. Wrap the WSL path in single quotes:
  `wsl.exe -d ubuntu bash -lc 'bash /home/.../script.sh'`.

## Build / numerics

- Do NOT add `-march=native` (or FMA-enabling flags): fused multiply-add breaks case01
  machine-exactness (`linf_p` must stay 0). The only measured gain was FMA, which is not
  worth losing bit-exact reproducibility.
- The residual `compute_R` is the single source of truth (defect-correction Newton): an
  approximate/finite-difference Jacobian changes only iteration COUNT, never the
  converged solution. So Jacobian changes are safe to A/B by convergence speed while
  keeping 10/10 byte-identical.
- Per-case knobs live in `SolverConfig` (`types.hpp`): `cfl`, `coupled`, `bdf2`,
  `minmod`, `lowdiss`, `ap_advection`, `dhat_scale`. Prefer per-case flags over global
  changes so other cases stay byte-unchanged.

## Physics / scheme (documented dead-ends — do not retry blindly)

- MWI pressure dissipation scales with dt (`dhat ~ dt`, transient-dominated aP), so SMALL
  time steps UNDER-damp the pressure-velocity coupling at strong shocks (case25
  reflected-shock overshoot). Raising Courant toward Denner's own value, or the
  `ap_advection` lever (Denner Eq.21's own e_P definition — physical), restores damping.
  This is the collocated small-dt checkerboard, consistent with Bartholomew et al.
  (JCP 375, 2018). `dhat_scale` (a bare tuned multiplier) also damps but is a NON-PHYSICAL
  fudge factor — REMOVED from all case defaults by user rule (physical coefficients only);
  it remains only as a research env knob (`ACID_DHK`), never for validation runs.
- Upwinding the face PRESSURE (pface) or the advecting velocity (ubar) at a shock is NOT
  valid — pressure is not advected; it breaks shock speed/position or diverges. Keep the
  conservative central pface.
- case07 residual wake wiggle (~1 Pa) is intrinsic BDF2 time-dispersion, shared by
  Denner's own scheme (he uses BE/BDF2 only, judged by amplitude-only gate). Removing it
  needs an L-stable integrator (TR-BDF2), which our `compute_R` "trans + full-flux"
  structure does not cleanly accept — do not attempt without restructuring the energy
  pressure-work source.
- case15 (double rarefaction) reference is a grid self-consistency test, not exact
  validation; the 4-eq model has no phase change, so the expansion-core pressure hits the
  EOS floor, not a physical vapour pressure. Do not present it as cavitation validation.
  **EXCLUDED from the registered suite since round 34** (user decision, applying the same
  sub-floor-state criterion `cases.cpp:599-602` already applies to case32): the exact
  double-rarefaction star pressure is `p*=9.05e-14 Pa`, 13 orders of magnitude below the
  1.0 Pa floor -- no grid-converged solution exists at any resolution (`YADV_RESEARCH.md`
  §42.3, §44). Config/IC/reference/gate code remains intact, unreachable, same status as
  cases 29/32.
- cases 24/33/34 (Denner 7.4.1 Fig.18 homogeneous Ms=10 mixture shocks, psi_water 0.5/0.25/0.75)
  **cannot pass under `ACID_YADV=1` for ANY numerical improvement** -- proved twice, in closed
  form (`YADV_RESEARCH.md` §36: an exact two-shock Riemann solution plus a full reachable-shock-
  family scan) and mechanistically (§41: the reference is the frozen-`T` limit and the OFF path
  the instant-`T`-equilibrium limit of the same thermal-relaxation continuum; the two exact
  closures differ by O(1), ~2x in rho/p, not by discretization error). The only correct fix is a
  4-8 round single-p/two-T Allaire/Kapila 5-equation model-class change. **EXCLUDED from the
  registered suite since round 35** by explicit user decision (the whole suite must validate
  under ONE solver and ONE technique; both the 5-eq rewrite and a `face_shock`-gated hybrid
  two-EOS approach were explicitly rejected). Note this is a DIFFERENT criterion from case15/32
  (representability below the 1.0 Pa floor) -- do not conflate them. Config/IC/reference/gate code
  all remain intact and unreachable, same status as 15/29/32. **Do not retry Jacobian/
  globalization/reconstruction/relaxation work on these three** -- rounds 4-8, 11, 23-28 all did,
  and §36 explains why none of it could ever have worked.
- THINC interface sharpening (DEFAULT ON, `ACID_NO_THINC` opts out) tanh-reconstructs the VOF
  FACE alpha in the colour-function transport (`acid.cpp` alpha loop) — only the face alpha,
  never the cell alpha the mass/mom/energy fluxes use, so it cannot break pressure
  equilibrium (case01 stays `linf_p=0`). INDICATOR (case-blind, applies only in a genuine
  material-interface cell): straddle `min(a_{i-1},a_{i+1})<0.5<max`, steep `|a_{i+1}-a_{i-1}|>0.5`,
  monotone, `1e-6<a_i<1-1e-6`; else plain upwind. `beta=3.5` is the ONE global scheme constant
  (literature-standard, like a limiter constant — permitted). FLUX FORM MATTERS (measured):
  the POINT downwind face value under-transports the sub-cell interface at finite CFL — the
  sharp case02 front lagged ~30 cells (corr_rho 0.98→0.85). RESOLVED by the CONSERVATIVE
  SEMI-LAGRANGIAN flux: the face alpha is the tanh profile AVERAGED over the dt departure
  region (`[1-c,1]` of the upwind cell for theta>=0, `[0,c]` for theta<0, `c=|theta_f|*dt/dx`),
  closed form via `D=(B-e^-beta)/(e^beta-B)` (verified vs quadrature to 4e-15). With it:
  suite 19/19, case02 front 36→1 cell, case30 rho contact 15→1, case31 12→3, case13 14→9;
  activates only on 02/13/14/25/30/31/35/36, never on 01/04/05/07/15/24/26/27/28/33/34 (those
  stay byte-identical). Do NOT tune beta per case.
  RHO-MONOTONICITY BVD GUARD (DEFAULT since 2026-07-14, Advisor-approved): at each THINC face
  the implied mixture density `af*rho_a(p_up,T_up)+(1-af)*rho_b(p_up,T_up)` must lie within the
  two adjacent CELL densities, else plain upwind (parameter-free; bounds = neighbour values).
  Measured: eliminates the case14 contact-band OSCILLATION at all N (TV-excess 44.6%→0.69% of
  the jump; N=800: 0.37%; band 42→23 = the OFF width; l2_u 0.113→0.103, corr_u 0.966→0.972)
  and cleans case25's interface ~100x (band ip 0.0121→0.0001, wave positions 8/1/9→0/0/1
  cells). ACCEPTED COSTS: case02 corr_rho 0.9999→0.9971 with a 1-cell front offset (82
  endpoint-class rejects: at a uniform-(p,T) contact the blend-vs-cell-rho test mismatches at
  ~1 ulp) — still far above the 0.90 gate and the 0.980 OFF baseline; case14 l2_rho
  0.031→0.038 = the HONEST monotone value (the former 0.031 was flattered by oscillation
  aliasing). The case14 signal (~50% blend mismatch) and the case02 noise (~1 ulp) live on the
  SAME endpoint-clamped instance class → no constant-free split exists (endpoint-exempt and
  rho-clamp forms restore case02 but bring the oscillation back — measured, do not retry).
  Full tables: docs/THINC_RHO_GUARD_RESEARCH.md.
- case14 THINC band spread — DIAGNOSED, indicator-flicker hypothesis DISPROVED (measured):
  per-step activation transitions equal exactly the moving-interface signature (case02: 700
  flips ≈ 2× the ~350 cells the front traverses — one activate + one deactivate per cell;
  case14: 106, same pattern), and a case-blind HYSTERESIS on the indicator (activate at
  `|a_{i+1}-a_{i-1}|>0.5`, hold active down to `>0.3`, previous-accepted-step memory like
  `theta_o`) changed case14's corr_u only in the 6th digit (band 42→42, l2_u 0.113→0.113) —
  a functional no-op, NOT committed. REAL mechanism (band dump): THINC keeps alpha sharp
  (2 cells) while the convected p/T stay 1st-order-smeared, so just past the interface the
  mixture rho is evaluated at the over-hot smeared T → rho undershoots the post-shock air
  plateau (158.7 vs 291.6) and recovers over ~15 cells. Any fix must sharpen the INTERFACE
  THERMODYNAMICS consistently with alpha (e.g. phase-wise T/energy reconstruction at
  interface cells), not the indicator. Do not retry hysteresis.
  UPDATE (measured, docs/THINC_CONSISTENCY_RESEARCH.md): the FACE-LEVEL version of that fix
  (two-sided tanh face T at THINC faces, all guards/variants) is REJECTED — it hits the V1
  ceiling short of acceptance, breaks case25's interface (iu 5.25, ACID Eqs.40-44
  telescoping), and mass/energy T-splitting collapses shock tubes. The real fix needs
  phase-consistent ENERGY TRANSPORT (model/scheme extension), not a face reconstruction.
