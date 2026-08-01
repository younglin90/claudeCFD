# YADV Round 16 Plan — the physical origin of case33's `dh`

Planner output (Agent, subagent_type="Plan", model=opus), round 16. Diagnostic round, one new
~20-line stderr-only print block.

Advisor spot-check: confirmed `T_from_hstat` (acid.cpp:334-362) clamps `T` to `[1e-6, 1e6]` at
every Newton sub-iteration and RETURNS TRUE even when saturated (`return isfinite(T) && T > 1e-6`
-- a saturated T=1e6 satisfies this unconditionally, regardless of whether `hmix(T)=hstat` actually
holds). Confirmed RMISM print ends ~line 1030-1033, HREINIT block (which overwrites s.h) starts at
line ~1042-1045 -- exactly the gap the plan specifies for the new RCELL block.

## 0. Executive summary (from the Planner's own diagnostic runs during planning)

Round 15 left open: why is `dh` at cell 79-81 six orders of magnitude beyond anything in a healthy
run? The Planner (with Bash access) ran diagnostics during planning and found:

1. Cell 79|80 is the IC discontinuity (x=0.1 for case33), not the moving shock front -- the true
   front at the stall time should be ~10.5 cells further right (cell ~90). **The shock never moves
   under `+ALPHA_IMPLICIT`.**
2. The stalled field (recovered from the pre-round-14-NaN-fill stale build, which still returns the
   raw partial state) shows a **3-cell vacuum blister frozen at the IC jump**: cell 80's density is
   `1.86e-6 kg/m^3` -- a literal vacuum, ~1.3e8x below the correct `250.368` -- with alpha driven to
   `0.99999432` (essentially pure air).
3. `s.h` at that cell is provably the runaway side of `dh`: both case33 phases have `b=0,eta=0` so
   `h_k = cp_k*T` exactly, bounding `Htot_o <~ 1.97e9` unconditionally given the `T<=1e6` ceiling --
   but round 15 measured `dh=3.7277e12`, implying `s.h` itself is ~3.7e12 (an implied T ~3700x
   above the ceiling).
4. Once `hstat` exceeds ~1.94e9, `T_from_hstat` saturates at `T=1e6` and (per the spot-check above)
   still returns `true` -- so `dT/dh=0`, `drho/dh=0` there: Newton can move its own energy unknown
   and the thermodynamic state does not respond. This produces exactly round 15's symptoms
   (`r_init` doubles per dt-halving, `fene->1.0`, dt-independent `dh`) with no reference to the
   alpha/Y REMAP channel (which round 15 measured clean).
5. Quantitatively closed: pre-shock `rho*H = 250.368*582247 = 1.4578e8`. If `rho` collapses to the
   measured `1.86e-6` while `rho*H` is approximately conserved, `H ~ 7.9e13` -- same order as the
   measured `3.7e12`. The enthalpy runaway is the arithmetic consequence of the density collapse,
   not an independent defect.
6. Why the density collapses: case33 is alpha=0.75 on BOTH sides (homogeneous mixture in
   alpha-space) but Y jumps 270x across the shock (`Y_pre=0.003466 -> Y_post=0.934388`). Case33's
   `Y_post` is the CLOSEST of the three cases (24/33/34) to the `alpha->1` singularity of the
   `Y->alpha` map. A cell receiving post-shock Y while still at pre-shock `(p,T)` maps to nearly
   pure air at 1 bar: `rho=1.157` instead of `250.37` -- a 216x collapse in one recovery step --
   then the `dalpha/dp|_Y < 0` feedback (falling p raises alpha) drives it the rest of the way to
   vacuum.
7. `+ALPHA_IMPLICIT`-specific: OFF shows nothing at cells 79-81 (alpha constant, nothing to
   transport, trivially exact). Plain `ACID_YADV=1` shows a DIFFERENT, global pathology (drifts to
   `alpha~0.0007` domain-wide by `t_end`, not a localized blister) -- consistent with round
   12/19.2's "case33 plain is the worst-fit case in the suite, `l2_p=1.573`". Only `+ALPHA_IMPLICIT`
   produces the localized runaway, because it closes the Y->alpha amplification into the Newton
   iteration itself.

**Conclusion going in**: nothing "physical" (in the sense of a genuine EOS/material limitation) is
happening -- the Y-form colour function is maximally ill-conditioned for exactly this case family
(homogeneous-in-alpha mixture, large Y jump, `Y_post` near 1), and `+ALPHA_IMPLICIT` turns a
one-step recovery error into a Newton-internal runaway. Fix likelihood: **unlikely to be free**
(every candidate touches either a common-path clamp or the residual's alpha map) -- see sect.5.

## 1. Practical finding -- stale build

The pre-round main-checkout build predates round 11 entirely (zero RMISM/RINIT/STALLED symbols).
A fresh rebuild on unmodified HEAD is mandatory before any edit, and its `denner1d_validate`
outputs are this round's pre-edit baselines (not any prior round's cached numbers).

## 2. The instrument -- one new print, one new env var (`ACID_RCELL`)

Inserted between the RMISM print (~line 1030) and the HREINIT block (~line 1045) -- after the
Y-transport/alpha-recovery and Eqs.43-44 rebuild (so `rho_o`/`hstat_o`/`Htot_o` are live), before
HREINIT (which overwrites `s.h`), so `s.h`/`s.rho`/`s.alpha` are still the natural `it==0` values.

New env var `ACID_RCELL="lo:hi"` (e.g. `"76:92"`; unset/malformed/`hi<lo` => the block never
executes), parsed once outside the time loop next to `rinit_dbg`. Gated on
`rcell_lo >= 0 && yadv`, independent of `rinit_dbg` -- deliberately a SEPARATE flag from
`ACID_RINIT` (not folded in) so that "ACID_RINIT=1 alone reproduces round 15's numbers exactly" stays
a checkable, unperturbed gate (G5), and so the ~17-line-per-retry window output doesn't multiply
round 13/15's established reproduce blocks. Reuses the EXISTING `ACID_BLK_STEP` (already shared by
RHIST/AJAC_BLK/DENSE/RMISM) for step selection -- no second step selector added.

Print format, one line per cell in the window per retry, tag `RCELL`:
`case=%s step=%d retry=%d dt=%.6e i=%d x=%.6f Y0=%.6e Y=%.6e al0=%.6f al=%.6f p_o=%.6e T_o=%.6e
u_o=%.6e h=%.6e Htot_o=%.6e rho=%.6e rho_o=%.6e` -- all read-only values already in scope
(`s0`, `Yv0`, `p_o/T_o/u_o`, `rho_o/hstat_o/Htot_o`, `Yv`, `s.*`), no new computation. `T_o` is the
decisive field (does it read exactly `1.000000e+06`, confirming ceiling saturation).

## 3. Measurement protocol (executed)

M1: `+ALPHA_IMPLICIT`, `ACID_RCELL=76:92`, steps 0/1/2/5 -- is the pathology born at step 0?
M2: `+ALPHA_IMPLICIT`, `ACID_RCELL=74:94 ACID_BLK_STEP=100 ACID_RINIT=1` -- the stall step, confirm
T-ceiling saturation and front position (P2/P3). M3: `ACID_TEND_SCALE` (existing, round 11) at
`scale=0.0186763` (matches the stall time `t~2.396e-6`) on OFF and plain, reading ONLY the solver
columns (alpha,p,u,rho) -- `*_ref` and validate metrics are meaningless under a scale, NEVER use in
a gate. M4: plain-path window at the same matched steps. M5 (cheap, optional): case24/34 at
`+ALPHA_IMPLICIT`, same window, steps 0-2 -- do they show the same but smaller `al[80]` excursion,
ordered by `Y_post` proximity to 1 (33 > 24 > 34)?

## 4. Decision rule on a fix

Default: no fix this round, stated up front regardless of findings. Three candidates named for a
FUTURE round (F1: upper-bound `s.h`, touches a common-path clamp, OFF-identity risk; F2: make
`T_from_hstat` return false on saturation, correct but changes OFF behavior on already-published
cases 13/14/25/28/29 that legitimately hit the ceiling transiently -- not a drive-by; F3: break the
p->alpha feedback inside the +ALPHA_IMPLICIT residual, targets the actual amplifier but is a
research-flag change needing its own round + full gates). Pre-registered order F3 > F1 > F2. A fix
is justified THIS round only if the measurement shows a single, isolated, off-by-construction bug
(e.g. alpha recovery using a wrong-time-level `(p_o,T_o)`, or `alpha_from_mass_fraction` receiving
unclamped/garbage `Y`) -- neither is expected (`Yv` is already clamped, the `(p_o,T_o)` usage
matches the Eqs.43-44 comment).

## 5. Gates (full battery -- a source change is made)

G0 build+unit. G1 OFF 19/19 byte-identical stdout, zero STALLED/DIVERGED across all 19 OFF dumps
(empirical). G2 plain-ON 15/19, per-case JSON byte-identical. G3 `+ALPHA_IMPLICIT` 14/19, per-case
JSON byte-identical (case24/34 must not move). G4 FD-invariance byte-identical. G5 `ACID_RCELL`
unset => zero RCELL lines in any gate run; AND `ACID_RINIT=1` with `ACID_RCELL` unset reproduces
round 15 sect.25.2's table exactly (dh=3.7277e12, dal_remap=2.2204e-16, r doubling from retry 6) --
doubles as the instrument-validity self-check. G6 `ACID_RCELL` set on an OFF run => zero RCELL
lines (yadv-gated), OFF validate stdout still byte-identical. G7 malformed `ACID_RCELL` (empty,
non-numeric, `hi<lo`, out-of-range) => no crash, no output. G8 `git status --short -- cpp/` shows
only `acid.cpp` modified, diff is exactly the RCELL block + one getenv.

## 6. Non-goals

No fix unless sect.4's narrow trigger fires (judged unlikely). No touching `ACID_STALL_ACCEPT`,
`ACID_YADV_HREINIT`, RINIT/RMISM's existing prints, or Stage 3c/diverged. No `cases.cpp`/
`validation.cpp` edits. No case24/34 work beyond M5's three read-only step-0 runs. No new numeric
constants/tuning/per-case branches. No changes to the Newton loop, line search, Jacobian, clamps,
advection stencil, or alpha recovery itself -- read-only print. No script edits. NEVER run
`ACID_TEND_SCALE` in a gate run.

## 7. Literature

No search. Both phenomena (start-up/vacuum generation at an initial discontinuity; ill-conditioning
of a mass-fraction-to-volume-fraction inversion near a pure-phase limit) are standard. If M5
confirms the alpha-form/Y-form conditioning contrast, a short citation check is worth doing when
this becomes a written claim in a future round, not now.
