# 07-B Air-Water pressure-packet root-cause diagnosis

**Case**: 07-B Air-Water acoustic pulse, N=400, dx=0.00375, interface x=0.5 (cell 133),
production config (`imex_ad`, `adaptive_bvd`, `tmlpu`, `regime_auto` ->
`pressure_work_consistent`, `kapila_closure=True`, `pure_branch=True`,
`alpha_pure_tol=1e-8`, CFL=0.4, 1620 steps).

**Metric under investigation**: `p_smooth_local_tv_excess = 0.5355` (limit 0.30);
`u` clean (`u_smooth_local_tv_excess=1.5e-4`), `rho` clean (`1.5e-4`). Pressure-only
8-10 cell packet in the water-side smooth region, at the base of the transmitted
Gaussian, growing with refinement (0.0 @N100 -> 0.481 @N200 -> 0.537 @N400).

Diagnosis method: env-gated, fully-removable instrumentation in `imex_ad.py`
(`FIVE_EQ_DIAG_PACKET`), one instrumented run + cheap causal probes. `imex_ad.py`
was restored byte-identical afterwards (verified: 02_A reproduces
`p_rel_linf = 2.764863893389702e-15`; AW reproduces `tv_excess = 0.5354865051035996`).

Evidence artifacts (fixed overwrite paths):
- `results/1D/07_B/diag_packet.npz` (full per-step scalars + windowed snapshots, cells 120-300)
- `results/1D/07_B/diag_packet.png` (6-panel: birth curve, spatial packet, m_adv glitch,
  H2 falsification, H1 correlate, H3 rejection)

---

## 1. Final-p assembly (the decisive question)

Traced the tail of `imex_ad_step` (`imex_ad.py:3123+`). For AW, `regime_auto`
resolves to `pressure_work_consistent` on **every** step (measured: `aux_missing=0`
for all 1620 steps; closure never flipped to `compressive_recovery`).

The step returns `W_new = (alpha_new, T1_new, T2_new, u_new, p_new)`. The final
primitive **p is `p_new` straight from the acoustic solve `_solve_acoustic_ad`**.
The pressure-work energy update
`rhoE_new = rhoE_adv - dt*div(p_f*u_f)` (`imex_ad.py:3254-3262`) is computed but
`rhoE_new` is **not** part of the returned primitive vector. It feeds back into `p`
only through `_recover_pressure_from_total_energy` inside the
`FIVE_EQ_IMEX_PW_PURE_SHOCK_RECOVERY` block, whose mask is
`_compressive_pressure_mask(W_n) & ~_pure_material_cell_mask(...)`.

**Measured, decisive:** in the packet window (water cells 150-285) the recovery mask
selects **0 cells on every step**, and `max|p_final - p_acoustic_raw| = 0.0 exactly`
on every step. The EOS/rhoE pressure-recovery path never touches the packet region.

=> Final p in the packet region is the raw `_solve_acoustic_ad` output. The
`div(p_f*u_f) -> rhoE -> recovered p` pathway is causally disconnected from the packet.

---

## 2. Packet birth: step and location

Instrumented water-window HF proxy `max|d2(p)|` over cells [150,285] per step:

| step | p_hf | u_hf | rho_hf | p_hf/u_hf |
|-----:|-----:|-----:|-------:|----------:|
| 800  | 4.4e-11 | 3.7e-17 | 0 | (roundoff) |
| 1000 | 2.7e-09 | 1.8e-15 | 0 | 1.5e6 |
| 1080 | (birth) | | | |
| 1100 | 2.7e-04 | 1.7e-10 | 2.1e-10 | 1.6e6 |
| 1200 | 2.8e-01 | 1.8e-07 | 3.1e-07 | 1.56e6 |
| 1400 | 4.6e-01 | 3.0e-07 | 3.4e-07 | 1.55e6 |

- **Birth step ~1080-1090.** The air pulse (right-moving simple wave, initial center
  x=0.1, sigma=0.014, air c=347.8) reaches the interface at x=0.5 at t=0.4/347.8
  ~= 1.15e-3 s = step ~1200; its leading edge (3 sigma ahead) arrives ~step 1075.
  Packet birth coincides exactly with the pulse first touching the interface.
- **Spatial birthplace = the interface cell 133.** At every birth step the leftmost
  water cell with `|p-ripple| > 1e-3` is exactly cell 133; the packet is anchored at
  the interface and fills the water rightward (transmitted wave direction). It is
  **not** born inside the smooth water bulk.
- **Pressure-only signature is extreme:** `p_hf/u_hf ~= 1.5e6`, `rho_hf` ~ 3e-7. The
  packet lives essentially entirely in `p`.

---

## 3. Seeding term and mechanism

The packet is generated **inside `_solve_acoustic_ad`**, in the high-order MUSCL
reconstruction of the transmitted acoustic wave across the air/water impedance jump
(Z_water/Z_air ~= 1.5e6 / 402 ~= 3800:1).

Acoustic residual (`imex_ad.py:2760-2761`):
```
r_u = rho*u_c - m_adv + dt*div_p
r_p = p_c - p0 + dt*(u0*dp_dx + beta*div_u)     beta = rho*c^2
```
`div_u`/`div_p` use MUSCL-reconstructed Z-Riemann faces with the **superbee** TVD
limiter by default (`FIVE_EQ_IMEX_ACOUSTIC_TVD`, falls through to superbee). Superbee
is the most compressive TVD member; on the smooth transmitted Gaussian it artificially
steepens the flanks and, combined with per-cell Crank-Nicolson (theta=0.5,
non-dissipative on "wave" cells), leaves the near-Nyquist collocated pressure mode
under-damped -> a dispersive 8-10 cell pressure packet at the Gaussian base.

Why pressure-only: water `beta = rho*c^2 ~= 998*1567^2 ~= 2.45e9`. Any velocity
divergence at the packet wavelength costs enormous pressure work, so the coupled
stiff acoustic operator forces the reconstruction error to be expressed in `p` with a
velocity response ~1e6 times smaller. This is the classic collocated-grid odd-even /
pressure-null-space decoupling, excited by the impedance-jump transmission.

Why it grows with refinement: superbee's anti-diffusion is more active as more cells
resolve the wave; the under-damped Nyquist band is wider/stronger at higher N.

---

## 4. Hypothesis verdicts

### H1 (split-velocity: SLAU2 material u_face vs acoustic u_f) - FALSIFIED as seed
- Correlate present: `|u_slau2_face - u_acoustic_face|` peaks (~3.4e-3) at interface
  faces 120-128 exactly at the crossing time, then **decays** (1.5e-3 -> 5e-5 -> 2e-5)
  while the packet persists. `corr(cumulative face-mismatch, p_hf) = 0.52`,
  `corr(instant mismatch, d p_hf/dstep) = 0.02`.
- The SLAU2-advected momentum `m_adv` does carry an interface glitch
  (`max|d2(m_adv)|` 4.7e-6 -> 1.07e-2 at crossing, propagating into water to 2.4e-4).
- **Causal probe (parameter-free, interface-gated):** disabled the SLAU2 face-velocity
  override at immiscible-interface faces (restored the pure acoustic Riemann face there).
  Result: `tv_excess 0.5355 -> 0.5369` (**unchanged**). The task's H1 mechanism (the
  split, via the pressure-work path) is **not** the seed. The face split is a symptom
  of two schemes disagreeing at the interface, not the cause of the p packet.

### H2 (pressure-work product dispersion via rhoE -> recovered p) - FALSIFIED
- Final p is not EOS-recovered in the packet region (Section 1): recovery mask = 0 cells,
  `|p_final - p_acoustic_raw| = 0.0` on every step. The `div(p_f*u_f)` dispersion cannot
  reach the final p there. (Consistent with the prior observation that `implicit_energy`
  and `pressure_work_consistent` give all-digits identical results.)

### H3 (CICSAM alpha micro-motion at the interface) - REJECTED
- Interface alpha (cell 133) drifts **monotonically** 1.0e-8 -> 8.7e-7 during crossing
  and settles at ~7.3e-7; `max|step-delta| = 1.4e-8`. It stays at pure-water level, so
  the frozen impedance Z at the interface barely changes. No per-step flicker -> no
  Z-flicker -> not the packet source.

### H4 (acoustic-solve high-order reconstruction / limiter) - CONFIRMED as seed
- Acoustic TVD limiter sweep (uniform, parameter-free), AW `tv_excess` / transmitted
  `p_peak_amp_ratio` (peak gate needs >= 0.85, wiggle gate needs <= 0.30):

  | acoustic limiter | tv_excess | p_peak_amp | wiggle gate | peak gate |
  |------------------|----------:|-----------:|:-----------:|:---------:|
  | superbee (default) | 0.5355 | 0.999 | FAIL | pass |
  | van Leer           | 0.4017 | 0.779 | FAIL | FAIL |
  | minmod             | 0.2602 | 0.632 | pass | FAIL |

  Monotonic dispersion/dissipation tradeoff: making the limiter more dissipative
  shrinks the packet **and** the physical transmitted peak in lockstep. This confirms
  the packet is dispersive reconstruction error of the acoustic solve, and that a pure
  limiter swap cannot pass both gates. (Matches the earlier finding that BE kills the
  packet but overdamps amplitude to 0.74; minmod overdamps to 0.63.)

- WAF probe (`FIVE_EQ_IMEX_ACOUSTIC_WAF=1`, sigma=nu, CFL-derived): `tv_excess -> 0.5637`
  (slightly worse), peak preserved 0.998. WAF adds weighted **anti-diffusion**
  (sharpening), the wrong sign for a dispersive packet -> ruled out.

---

## 5. Minimal parameter-free fix recommendation

**Root**: under-damped near-Nyquist collocated pressure mode in `_solve_acoustic_ad`,
excited by high-order (superbee) reconstruction of the transmitted wave across the
impedance jump, on the non-dissipative CN branch.

The fix must be **wavenumber-selective**: damp only the 8-10 cell (near-Nyquist) band
while leaving the resolved transmitted wave (support ~30-100 cells) untouched. Every
global knob tested (limiter, BE, WAF) instead hits the dispersion/dissipation tradeoff
because it acts on the whole wave.

**Recommended candidate (parameter-free, case-uniform): a 4th-order (biharmonic)
pressure filter in the acoustic pressure residual on same-pure-material smooth
stencils.** Add to `r_p` a term `+ dt * eps_bh * dx^3 * d4(p)/dx4` restricted to
`_same_pure_material_face_mask` cells, with `eps_bh` a stencil/CFL-derived constant
(no tuned value): the leading truncation is O(dx^3), so it is high-order-consistent and
near-invisible to the resolved wave, but its response scales as `k^4` and strongly
damps the Nyquist band. Precedent exists in-repo: the BE path already carries a
biharmonic `imp_dissipation`; this is the natural port of that mechanism into the
`imex_ad` acoustic residual.

Equivalent framing (same effect, arguably cleaner): **retain the low-order Z-Riemann
pressure-diffusion term in `div_u` even on high-order same-material faces.** The upwind
Z-Riemann face `u_f = (p_l - p_r + Z_l u_l + Z_r u_r)/den` already contains the
Rhie-Chow-style `(p_l - p_r)/den` pressure coupling that damps odd-even p; the MUSCL
high-order reconstruction cancels it on smooth cells, which is exactly what removes the
Nyquist damping. Blending back a k-selective fraction of that low-order coupling is the
mechanism-precise cure and aligns with roadmap Phase 4 (generalized Rhie-Chow).

**Expected risk to other cases**: low but must be validated.
- A biharmonic/Rhie-Chow term restricted to same-pure-material smooth stencils does not
  touch mixture cells or interface faces, so 02_A (single homogeneous NASG advection)
  and pure/mixture regression gates should be unaffected to roundoff; still, re-run the
  full 26-case gate and the amplification_matrix / uniform_flow / transport_eigenmode
  regressions.
- Main risk is slight extra dissipation of genuinely-resolved acoustic peaks in the
  other 07 subcases (Helium-Air, Argon-Air). Because the damping is O(dx^3) and
  k^4-weighted, the resolved-wave impact should be far smaller than the limiter swaps
  above (which cost 20-37% peak amplitude); target: keep all 07 `p_peak_amp_ratio >= 0.85`
  while bringing AW `tv_excess <= 0.30`.
- Not yet tested (out of run budget): `FIVE_EQ_IMEX_ACOUSTIC_WAF_SIGMA=pressure_sensor`
  is a *sharpening* sensor and will not help (Section 4). The biharmonic/Rhie-Chow
  dissipation is the correct next implementation-and-validate step.

## 6. Probe metrics summary (AW, N=400)

| configuration | tv_excess (limit 0.30) | p_peak_amp (limit 0.85) | verdict |
|---------------|-----------------------:|------------------------:|---------|
| baseline (superbee) | 0.5355 | 0.999 | packet present |
| SLAU2 iface -> acoustic ustar | 0.5369 | ~0.998 | H1 not causal |
| acoustic minmod | 0.2602 | 0.632 | H4 confirmed, overdamps peak |
| acoustic van Leer | 0.4017 | 0.779 | tradeoff |
| acoustic WAF (sigma=nu) | 0.5637 | 0.998 | anti-diffusion, wrong tool |

02_A regression across every run: `p_rel_linf = 2.764863893389702e-15` (unchanged).
