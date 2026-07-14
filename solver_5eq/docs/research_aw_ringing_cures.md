# Air–water interface acoustic ringing — literature study and parameter-free cures

Scope: low-amplitude pressure ringing at the base of the transmitted peak in the
air–water (Z ratio ~3600:1) acoustic transmission case; smoothed-pressure local TV
excess 0.537 vs guard 0.30 at N=400, ~5 turning points adjacent to the interface.
Constraint: any fix must be parameter-free (derived constants only) and identical for
all cases. Must not regress 02_A machine-precision PE preservation or shock tubes 13/14.

Current discrete closure under study (`solver/five_eq_IMEX/residual.py`):

```
Z_i   = rho_mix,i * c_wood,i                      (Kapila/Wood mixture, alpha-floored)
p*_f  = (Z_R p_L + Z_L p_R + Z_L Z_R (u_L - u_R)) / (Z_L + Z_R)
u*_f  = (p_L - p_R + Z_L u_L + Z_R u_R) / (Z_L + Z_R)
```

applied inside a Crank–Nicolson-weighted implicit acoustic block, with SLAU2
(1−M̂)² velocity coupling in the explicit advective flux and compressive
(CICSAM/BVD) alpha transport.

---

## 0. Mechanism analysis — why the wiggle exists

Three compounding effects, all documented in the literature:

**(M1) The smeared layer is a slow-sound-speed slab (Wood dip), i.e. a resonant
cavity — not an impedance "wall".**
For air (γ=1.4, ρ=1.157, c=347.8) / water (SG γ=4.1, Π=4.4e8, ρ=998, c=1344.6),
the Kapila/Wood speed at α=0.5 is

```
1/(rho c²)_wood = Σ_k α_k /(ρ_k c_k²)  →  c_wood(0.5) ≈ 24 m/s
```

— a factor ~15 below air and ~56 below water. The mixture *impedance*
Z_wood(0.5) = ρ c ≈ 1.2e4 actually lies between Z_air ≈ 402 and Z_water ≈ 1.34e6
(near the geometric mean), so in *steady* linear theory the layer is an
anti-reflection ladder, not a barrier. The transient problem is different: the
incident pulse enters a 1–3 cell slab whose internal propagation speed is ~24 m/s.
The pulse reverberates between the two property jumps (air→mix, mix→water) with a
round-trip time τ ≈ 2 L_layer / c_layer that is 1–2 orders longer than the
cell-crossing time in either bulk phase. Each internal round trip leaks a delayed,
sign-alternating micro-pulse into the water side — exactly a train of ~5 turning
points riding the *base* of the transmitted peak. Saurel–Petitpas–Berry (2009)
identify the non-monotonic Wood speed as the cause of "inaccuracies in wave
transmission across [diffuse] interfaces" and build their 6-equation
pressure-non-equilibrium model specifically to avoid propagating through the Wood
dip. Ballout et al. (2025) show for air–water 1D transmission that the diffuse
interface width is the dominant transmission-error source, with error → 0 in the
sharp limit.
[Cites: web S0021999108005895 (Saurel 2009); local `papers/49_ballout_2025_acoustic_diffuse_interface_summary.md`;
local `papers/md/26_ten_eikelder_2017_acoustic_convective_kapila.md` (Wood formula validity note).]

**(M2) CN weighting leaves the reverberation undamped.**
Crank–Nicolson is A-stable but not L-stable: for the stiffest resolved modes the
amplification factor → −1, so grid-scale components generated at the property jump
are preserved with alternating sign instead of decaying. This is the classic CN
oscillation pathology (Britz–Østerby–Strutwolf 2003, "Damping of Crank–Nicolson
error oscillations"). The layer reverberation of (M1) is exactly such near-Nyquist
content, so CN converts a one-pass transmission error into persistent ringing.
Both semi-implicit two-phase references in the local pool made the opposite choice
for the acoustic sub-step: Peluchon et al. 2017 use fully implicit BE (IM1),
selected as "the most robust approach and the simplest one"; Denner et al. 2018
(ACID) use the *second-order backward* (BDF2) scheme, which is L-stable AND 2nd
order — they report no ringing in the identical air–water benchmark (see §B).
Tallois et al. 2022 do offer a modified CN (their Eq. 3.14, with ϑⁿ frozen) but
only as a cost optimization on top of a robust BE base, and they keep the
transport-step velocity at u = (ūⁿ+ū†)/2 for conservation (their Eq. 3.17).
[Cites: local `papers/md/25_peluchon_2017_imex_acoustic_transport.md` (Algorithm 1 / IM1, "most robust");
local `papers/md/29_tallois_2022_2nd_order_imex_twophase.md`;
local `papers/md/[적용해볼것] ACID 2.md` §7.3 ("second-order backward Euler scheme is applied");
web S009784850200075X (Britz–Østerby–Strutwolf 2003).]

**(M3) The alpha floor makes the well permanent, and Z is evaluated from floored
mixture cells.** With alpha-flooring, the cells adjacent to the interface never
become pure, so at least one face on each side always carries a Wood-dipped cell
state into the Z-weighted closure and into the implicit block coefficient
(ρc²·Δt/Δx). Compressive alpha transport cannot remove this — it can only shrink
the slab to the floor width.

Summary: **wiggle = internal reverberation of a numerically created slow slab
(Wood dip), sustained by a non-L-stable time weighting.** Cures that raise the
layer's effective acoustic speed (A, B, D) attack the source; cures that damp
near-Nyquist content (C) attack the sustain. The two are independent and stack.

---

## A. Mixture-layer impedance trap — cures

### A-i. Sharpen alpha so the layer is 1–2 cells

- Formula/algorithm: already implemented (CICSAM/BVD). The literature limit is
  MMACM-Ex (He/Zhao-line 2025): FCT-style downwind volume-fraction flux keeps the
  interface within **2 cells** without a steepness parameter (replaces the tuned
  β=2.9 harmonic limiter of MMACM 2024).
- Parameter-free verdict: MMACM-Ex yes (FCT downwind construction); MMACM 2024 no
  (β=2.9).
- Expected effect: shrinks the slab but cannot eliminate the floored cell, so it
  reduces reverberation period, not its existence. Diminishing returns — Ballout
  2025 shows width is the dominant error, but our layer is already near-minimal.
- Other 12 cases: sharpening changes alpha transport globally; risk to shock tubes
  13/14 (interface-shock interaction) if the sharpening flux is not
  consistency-coupled (see §E). Do not pursue as the primary cure.
- Citations: local `papers/md/MMACM_2025.md`; local `papers/18_he_tan_2024_mmacm_summary.md`;
  local `papers/49_ballout_2025_acoustic_diffuse_interface_summary.md`.

### A-ii. Evaluate face impedance/coefficients WITHOUT the Wood dip

Three concrete, parameter-free constructions found:

**A-ii-1. Frozen / isobaric-closure mixture sound speed for Z** (Allaire-consistent).
For stiffened gas the isobaric-closure (Allaire) mixture speed is

```
1/(γ−1)      = Σ_k α_k/(γ_k−1)
γΠ/(γ−1)     = Σ_k α_k γ_k Π_k/(γ_k−1)
c_iso²       = γ (p + Π) / ρ_mix          →  c_iso(α=0.5, air–water) ≈ 643 m/s
Z_i          = ρ_mix,i · c_iso,i
```

This is exactly ACID's Eq. (57)–(58) (Denner 2018 calls it "a uniquely defined
average … a condition that Allaire et al. associated with well-posedness") and
exactly MMACM's Eqs. (10)–(11). Saurel–Petitpas–Berry 2009 institutionalized the
same idea as the "frozen" sound speed of the 6-equation model: hyperbolic step
with a monotone mixture speed, pressure relaxation afterwards. NASG analog is
available through the He2024 EOS mixture parameters (same 1/(γ−1) volume-average
structure). No coefficients; identical formula in all cells; in pure cells it
reduces exactly to the phase sound speed, so every face not touching a mixture
cell is bit-identical to today.
- Parameter-free verdict: **yes** (derived from EOS constants only).
- Citations: local `papers/md/[적용해볼것] ACID 2.md` Eq. 57–58; local
  `papers/md/MMACM_2025.md` Eq. 7–11; web S0021999108005895 (Saurel 2009);
  local `papers/49_ballout_2025_acoustic_diffuse_interface_summary.md`
  (linear phase-wise c interpolation transmits correctly in their DG framework —
  independent confirmation that a non-Wood layer speed is what transmits).

**A-ii-2. Single-valued max face impedance (ten Eikelder–Daude–Koren 2017).**
Their Lagrangian acoustic HLLC uses

```
a_{j+1/2} = max(ρ_j c_j, ρ_{j+1} c_{j+1})
u* = (u_j+u_{j+1})/2 + (p_j−p_{j+1})/(2a),   p* = (p_j+p_{j+1})/2 + a(u_j−u_{j+1})/2
```

(their Eqs. 32, 34a–b). At an air|mixture face the max picks the non-dipped side,
so the dipped cell's Wood impedance never enters the face closure. Slightly more
dissipative on the low-Z side (the 1/(2a) pressure-difference term shrinks).
- Parameter-free verdict: **yes**.
- Note: this collapses the two-sided Z-weighting to one value — it changes gas–gas
  faces too (Z_L ≠ Z_R even in single phase after reconstruction), so 07-A
  regression must be watched. A localized variant (max only where |Δα|>tol, i.e.
  the existing `_apply_interface_acoustic_riemann` mask) reuses the already
  present topology detection.
- Citation: local `papers/md/26_ten_eikelder_2017_acoustic_convective_kapila.md` §3.3.2.

**A-ii-3. ACID stencil-frozen (side-dominant) property evaluation** — see §B; the
row-asymmetric variant is already prototyped in the frozen He2024 tree
(`solver/He2024/explicit_mmacm_ex.py` `acid_interface` flag, ~line 3874: "all face
impedances seen from cell i are evaluated with cell i's own (ρ,c)").

### A-iii. Two-material Riemann solver at detected interface faces

- Algorithm: at faces flagged by the existing alpha-topology detection
  (`alpha_jump_tol` mask / pure-band logic), take Z_L from the nearest pure-α cell
  on the left and Z_R from the nearest pure-α cell on the right (i.e. extrapolate
  *bulk* impedances across the floored layer), then apply the same linear
  two-shock closure. This makes the discrete interface a single contact with the
  correct physical R/T coefficients (T = 2Z_R/(Z_L+Z_R)), eliminating the ladder.
- Parameter-free verdict: **yes if** the detection reuses the existing topology
  machinery (no new thresholds). The pure-band search is integer topology, not a
  coefficient.
- Risk: at true mixture regions (cavitating/mixture cases, e.g. cases with real
  Kapila mixtures) bulk extrapolation is wrong physics; the flag must key on
  "interface between pure bands", which the pure_branch/alpha_pure_tol logic
  already encodes.
- Precedent: GFM-family and ACID both assert one-fluid thermodynamics at the
  interface stencil; ACID §5.4 explicitly frames itself as the flux-level
  equivalent of GFM without ghost states.

---

## B. ACID mechanism in detail (Denner–Xiao–van Wachem 2018, JCP 367:192)

What they actually do (all from local full text `papers/md/[적용해볼것] ACID 2.md`):

1. **Stencil-frozen colour function** (§5, Fig. 4): when discretising cell P, the
   colour function of *every* cell in P's stencil is set to ψ_P. The stencil then
   contains a single fictitious fluid → the density/enthalpy jumps at the interface
   never enter any one cell's discretised equations → no unphysical pressure
   gradient (their Eqs. 35–36 illustrate the momentum inconsistency otherwise).
2. **Face density** (Eqs. 40–42): ρ̃_f interpolated between ρ*_U, ρ*_D where both
   are the *partial-density blends at ψ_P* — i.e. the neighbour's (p,T) but the
   local cell's ψ. Fluxes become asymmetric across a face (each adjacent cell sees
   its own version), which is accepted and stated (§5.4).
3. **Face enthalpy** (Eqs. 45–53): total enthalpy H=ρh treated the same way, via a
   deferred correction δh_f = ĥ_f − h̃_f added to the energy advection term; in the
   bulk δh_f = 0 identically.
4. **Previous time levels re-evaluated at current ψ_P** (Eqs. 43–44, 54–56) — the
   transient term must see the same fictitious fluid, otherwise the time
   derivative itself manufactures a jump.
5. **Interface-region sound speed** = isobaric-closure mixture value (Eq. 57–58) —
   the monotone average of §A-ii-1, NOT Wood. Verified to <0.33% against theory
   throughout 0≤ψ≤1 (their Fig. 11, §7.3.1).
6. **Time scheme**: second-order backward Euler (BDF2) — L-stable (their §7.3
   setup paragraph).
7. **TVD trap** (§5.4): with air–water the nonlinear limiter on ρ and h oscillated
   the outer iterations; the cure is to *enforce first-order upwind for ρ and h*
   at faces where |ψ_P − ψ_Q| > η (η = existing solver tolerance — no new
   parameter). Velocity keeps its high-order scheme.

**Benchmark (identical to ours)**: 1D, Δx = 2×10⁻³ m, Co = 0.48, inlet velocity
pulse Δu₀ = 0.02·u₀, f = 5000 s⁻¹, air–water (SG water γ=4.1, Π=4.4e8).
Result (their Fig. 14, §7.3.2): transmitted/reflected pressure-amplitude ratio
1.995 vs theory 2.001 (**0.3% error**), no base-of-peak ringing visible or
discussed — with monotone Minmod interfacial-flow spatial scheme and BDF2 time.
Their impedance-matched control case (§7.3.3) quantifies the residual interface
artifact: spurious reflection amplitude ratio 8.77×10⁻³, traced *quantitatively*
to the 2.1% impedance error of the single mixture cell (ψ=0.83 → Z=489.46 vs 500)
— i.e. in ACID the only remaining acoustic artifact is the (monotone) property
blend of one cell, not a reverberating slab.

Transferable conclusions for a 5-eq FVM: (a) interface-region acoustic properties
must be a *monotone* one-fluid blend (they use isobaric closure); (b) the time
scheme over the acoustic operator should be L-stable; (c) first-order property
upwinding at interface faces is acceptable and even necessary (their §5.4), and it
is parameter-free.

---

## C. CN vs BE/BDF2 weighting of the implicit acoustic solve

- Physics: reflected near-Nyquist content at the property jump sits at the CN
  amplification-factor limit g → −1 (undamped, sign-alternating). BE gives |g| → 0
  (over-damped, 1st order); BDF2 gives L-stable damping at 2nd order.
- Literature practice:
  - Peluchon 2017: acoustic sub-step is BE (IM1); IM2/Newton variants exist but
    IM1 chosen for large density/pressure ratios ("most robust", §5).
  - Tallois 2022: modified CN (Eq. 3.14; ϑⁿ frozen so formally not 2nd order)
    introduced purely to avoid running the implicit solve twice per Heun step;
    conservative coupling to transport via u_f = (ūⁿ + ū†)/2 (Eq. 3.17). Their
    compressive limiter is explicitly NOT applied in the acoustic step —
    (u,p) reconstructed with monotone β=1, compressive β=2 only on z in the
    transport step (§3.1, "Acoustic step: (u,p) reconstructed with β=1").
  - Denner 2018 ACID: BDF2 (2nd order, L-stable) for all acoustic tests.
- Is 2nd-order-in-time worth keeping? Yes, and it does not conflict with damping:
  **BDF2 provides both**. If BDF2 is too invasive (two-step, needs one stored
  level + BE startup step), plain BE is the documented robust fallback; the
  accuracy loss appears as slight amplitude smearing of the transmitted pulse
  (first-order dissipation ~ O(Δt) on resolved wavelengths), which at acoustic
  CFL ~0.5 and ~70 cells/wavelength is small; the gas–gas amplitude items
  (Liu metric) must be re-measured.
- Parameter-free verdict: **yes** for both BE and BDF2 (scheme choice, no
  coefficient). A tuned θ∈(0.5,1) theta-scheme would NOT be parameter-free unless
  θ is derived; no paper found that derives a θ from local data for this purpose —
  avoid.
- Effect on other cases: 02_A PE preservation — uniform (p,u,T) states are exact
  fixed points of BE/BDF2/CN alike (the residual is identically zero; the face
  closure returns p*=p, u*=u for any Z>0), so machine-precision preservation is
  unaffected. Shock tubes 13/14 — BE/BDF2 add damping only near Nyquist; shock
  positions are flux-controlled; ACID's shock suite (their §7.4, incl. air–water
  shock tube Fig. 28) runs entirely on BDF2 without spurious interface
  oscillations. Amplification-matrix gate: recompute ρ(A); BE1 already the
  project standard for the coupled step, so aligning the acoustic sub-weighting
  is consistent with the existing "be1 only" guardrail.
- Citations: local `25_peluchon_2017_imex_acoustic_transport.md`,
  `29_tallois_2022_2nd_order_imex_twophase.md`, `[적용해볼것] ACID 2.md`;
  Britz–Østerby–Strutwolf 2003 (sciencedirect S009784850200075X).

---

## D. Kapila vs Allaire closure and the interface acoustic response

- The D₁ compaction term is what turns the model's mixture sound speed from the
  Allaire *frozen/isobaric* value into the Kapila *Wood* value. Physically, Wood
  is correct for true dispersed mixtures at moderate frequency (ten Eikelder §1
  cites experimental agreement). But the smeared interface layer is **not a
  physical mixture** — it is a numerical artifact — so making the scheme propagate
  through it at the Wood speed imposes a spurious physical model on non-physical
  cells.
- Literature verdicts:
  - Saurel–Petitpas–Berry 2009: the non-monotonic Wood speed causes wave
    transmission inaccuracies; their cure (6-eq, two pressures, stiff relaxation)
    makes the *hyperbolic step* see the monotone frozen speed and recovers the
    Kapila solution via relaxation. Parameter-free (stiff limit), widely adopted.
  - Denner 2018 (ACID) and Allaire's own model use the isobaric-closure speed at
    interfaces; ACID demonstrates clean air–water acoustic transmission with it.
  - No paper found that demonstrates *better* linear-acoustic interface
    transmission with the Wood speed active inside the diffuse layer.
- Practical translation to this solver: keep the Kapila D₁ term in the α-equation
  (it matters for genuine mixture dynamics and the project's validated cases) but
  evaluate the *numerical acoustic properties* (face Z; optionally the implicit
  block's ρc² coefficient in flagged interface cells) with the frozen/isobaric
  speed — i.e. Saurel's philosophy collapsed into the linearized face closure.
  Changing only Z is provably PE-safe (see §C note); changing the cell ρc²
  coefficient additionally alters the transient response of mixture cells and
  should be gated by the alpha-topology mask (still parameter-free) and verified
  against the Kapila mixture cases.
- Citations: web S0021999108005895; local `26_ten_eikelder_2017_...md`,
  `[적용해볼것] ACID 2.md`, `MMACM_2025.md` (Eq. 7–11 isobaric-closure speed).

---

## E. He/Tan (MMACM) consistency chain — pressure–alpha coupled correction

- The exact content (local `papers/18_he_tan_2024_mmacm_summary.md`): any
  interface-sharpening flux Ĝ^α on the α-equation must be propagated to the
  conservative equations as

```
Ĝ^{α_k ρ_k} = ρ̃_k Ĝ^{α_k},   Ĝ^{ρu} = Σ_k (ρ_k u)~ Ĝ^{α_k},   Ĝ^{ρE} = Σ_k (ρ_k E_k)~ Ĝ^{α_k}
```

  (upwinded phasic states), derived from the immiscibility-preservation conditions
  A_k = M_k/ρ_k, Θ = u·P + Σ_k e_k M_k (their Eq. 46). This chain is
  **coefficient-free given Ĝ^α**; the residual tuning lives in Ĝ^α itself (β=2.9
  harmonic limiter in MMACM 2024; replaced by an FCT-downwind construction in
  MMACM-Ex 2025).
- Relevance to the ringing: if alpha is sharpened (CICSAM/BVD) while mass/energy
  are not co-corrected, the layer's state drifts off the isobaric-closure
  manifold; the EOS then produces p/c values inconsistent with α — an additional
  source of partial reflections. The consistency chain removes that error family
  without any new constant.
- Parameter-free verdict: chain yes; Ĝ^α construction — FCT-downwind (MMACM-Ex)
  yes, harmonic-β no.
- Effect on other cases: mass conservation Δm ~ 1e-15 and PE at machine precision
  are demonstrated in their Tables 2–3; consistent with 02_A requirements.
- Citations: local `papers/18_he_tan_2024_mmacm_summary.md`,
  `papers/md/MMACM_2024.md`, `papers/md/MMACM_2025.md`.

---

## F. Other parameter-free, case-uniform findings

1. **Deng–Xie–Matar–Boivin 2025 (arXiv 2502.02570)** gate the SLAU2
   pressure-diffusion term in the mass flux off at material interfaces:
   ṁ = … − θ·(χ/c̄)(p_R − p_L), with θ=0 iff 0.01<α₁<0.99 (and surface tension
   present). The mechanism (pressure-difference term acting across a
   property-jump face creates spurious mass flux/velocity) is real, but the
   0.01/0.99 band is a tuned threshold → **not parameter-free as published**. If
   ever needed, the existing pure-band topology mask can substitute for the band.
   No acoustic transmission test is reported in that paper.
   [web arxiv.org/html/2502.02570; local `papers/40_hybrid_allmach_summary.md`.]
2. **Saade–Lohse–Fuster 2023 (JCP 476, 111865)** solve pressure–temperature
   coupled implicitly on the Fuster–Popinet all-Mach base; the Basilisk
   implementation evaluates ρc² cell-wise from the EOS with VOF-weighted phase
   densities and uses face 1/ρ from the arithmetic-mean density; no special
   interface acoustic fix is present in the public source — i.e. this line of
   work does not contribute a cure beyond property-consistency.
   [web basilisk.fr/src/compressible/two-phase.h; search result JCP 476.]
3. **Ballout–Marino–Ntoukas–Rubio–Ferrer 2025** (JCP accepted, arXiv 2504.01727):
   linear phase-wise interpolation of c_s across the diffuse layer (NOT Wood)
   transmits air–water acoustics with spectral accuracy; transmitted-amplitude
   error is controlled by interface width. Corroborates A-ii-1/D.
   [local `papers/49_ballout_2025_acoustic_diffuse_interface_summary.md`.]
4. **Existing biharmonic face dissipation** (`imp_dissipation=0.02`) is itself a
   tuned coefficient. If shortlist items land and the ringing guard passes, this
   constant becomes a candidate for reduction/removal, moving the solver *toward*
   the parameter-free requirement rather than away from it.
5. **Tallois's split of reconstruction roles** — compressive limiter confined to
   the transport step, monotone reconstruction in the acoustic step — is already
   the shape of this solver (compressive alpha transport, linear acoustic
   closure); no action, but it confirms that further compressive treatment of
   (u,p) in the acoustic block would be the wrong direction.

---

## Ranked implementation shortlist (concrete discrete formulas)

Ranking weighs: mechanism match, parameter-freeness, blast radius on the other
12 cases, implementation cost.

**1. L-stable time weighting of the implicit acoustic block (mechanism M2; family C).**
Replace the CN-type weighting with BDF2 (BE for the startup step):

```
(3 q^{n+1} − 4 qⁿ + q^{n−1}) / (2Δt) + L_ac(q^{n+1}) = 0        (BDF2)
step 1 fallback:  (q^{n+1} − qⁿ)/Δt + L_ac(q^{n+1}) = 0          (BE)
```

where L_ac is the existing Z-weighted acoustic operator. Parameter-free; identical
for all cases; kills the sustain of the reverberation (near-Nyquist |g| < 1
strictly). PE 02_A: uniform states are exact fixed points — untouched to machine
precision. Shock tubes: precedent ACID runs its full shock suite on BDF2.
Cheapest to try: pure BE first (one-line weight change) to confirm the ringing is
CN-sustained; then BDF2 to recover 2nd order if the gas-gas amplitude metrics
(07-A Liu) degrade under BE.

**2. Frozen/isobaric-closure impedance in the face Riemann closure (mechanisms M1/M3; families A-ii-1, D).**
In `_mixture_impedance`, replace the Wood mixture speed by the isobaric-closure
(Allaire) speed for the *closure impedance only*:

```
1/(γ_i −1)     = α_i/(γ₁−1) + (1−α_i)/(γ₂−1)
γΠ_i/(γ_i −1) = α_i γ₁Π₁/(γ₁−1) + (1−α_i) γ₂Π₂/(γ₂−1)
c_iso,i        = sqrt( γ_i (p_i + Π_i) / ρ_i )                 (SG; NASG analog via He2024 EOS)
Z_i            = ρ_i c_iso,i
```

then the unchanged two-sided closure p*, u*. In pure cells c_iso ≡ c_phase, so
all faces away from mixture cells are bit-identical; PE preservation holds for any
Z>0 (p*=p, u*=u at uniform states). Raises the layer's closure speed 24→~640 m/s:
reverberation period shrinks ~27×, echoes merge into the transmitted peak.
Physical model (Kapila D₁ in the α-equation, EOS energy closure) untouched.
Watch: cases that rely on genuine Wood-mixture acoustics (physical mixture zones,
not interfaces) — for those the closure Z is a numerical dissipation weight, not
the propagation speed (that lives in the ρc² block coefficient), so first-order
impact only; verify with the mixture-zone regression cases.

**3. ACID-style stencil-frozen property evaluation for the interface faces (family A-ii-3/B).**
For the flagged faces (existing |Δα| topology mask), evaluate the *pair* (Z_L, Z_R)
of the closure asymmetrically per adjacent cell row: cell i's row sees both face
impedances at its own α_i (ACID §5; already prototyped as `acid_interface` in
`solver/He2024/explicit_mmacm_ex.py` line ~3874 — port, don't re-derive). Combined
with item 2 this reproduces ACID's full property treatment; ACID's measured
residual artifact on the same benchmark is a 8.8e-3 spurious reflection ratio
(impedance-matched control), i.e. more than an order below the current guard.
Cost: asymmetric implicit matrix entries (block-tridiagonal stays); needs care in
the Schur path.

**4. Two-material face Riemann at interface faces (family A-iii).** Extrapolate
bulk-side impedances across the floored layer using the existing pure-band
topology (no new threshold): Z_L ← nearest pure-cell impedance left, Z_R ← nearest
pure-cell impedance right, same closure. Strongest possible ladder removal
(single contact with exact linear R/T), but wrong for genuine mixture regions —
strictly gated to pure-band/pure-band interfaces. Rank below 2/3 because gating
semantics carry model risk while 2/3 are unconditional.

**5. MMACM consistency co-sharpening (family E).** If/when alpha sharpening is
strengthened, propagate Ĝ^α to (α_kρ_k, ρu, ρE) via the He–Tan chain (formulas in
§E). Not a direct ringing cure at current layer width; prevents a secondary
reflection source. Adopt opportunistically.

Not shortlisted: Deng-2025 θ-gating (tuned band, and the ringing lives in the
implicit block, not the SLAU2 mass flux); DG/spectral approaches (architecture
mismatch, Ballout); tuned θ-schemes (no derivation for θ exists in the pool).

Recommended order of experiments: 1 (BE, then BDF2) → 2 → re-measure guard →
3 only if excess persists. Items 1+2 are both one-function changes, independent,
and each is individually reversible for A/B attribution.

---

## Citation index

- Denner, Xiao, van Wachem 2018, JCP 367:192 — local `papers/md/[적용해볼것] ACID 2.md`
  (full text; §5 ACID, §7.3.2 air–water benchmark, Eq. 57 sound speed, §5.4 TVD trap).
- Peluchon, Gallice, Mieussens 2017 — local `papers/md/25_peluchon_2017_imex_acoustic_transport.md`
  (IM1 BE acoustic step; two-sided ā± acoustic Riemann closure).
- Tallois, Peluchon, Gallice 2022 — local `papers/md/29_tallois_2022_2nd_order_imex_twophase.md`
  (compressive limiter confined to transport step; modified CN Eq. 3.14; conservative u_f Eq. 3.17).
- ten Eikelder, Daude, Koren, Tijsseling 2017 — local
  `papers/md/26_ten_eikelder_2017_acoustic_convective_kapila.md`
  (acoustic/convective split; a_face = max(ρc); Wood-speed validity note).
- He, Tan 2024 (immiscibility conditions / MMACM) — local `papers/18_he_tan_2024_mmacm_summary.md`.
- MMACM-Ex 2025 — local `papers/md/MMACM_2025.md` (isobaric-closure c, Eq. 7–11; 2-cell sharpening).
- Ballout, Marino, Ntoukas, Rubio, Ferrer 2025 — local
  `papers/49_ballout_2025_acoustic_diffuse_interface_summary.md` (arXiv 2504.01727).
- Saurel, Petitpas, Berry 2009, JCP 228:1678 —
  https://www.sciencedirect.com/science/article/abs/pii/S0021999108005895
  (non-monotonic Wood speed → transmission inaccuracy; 6-eq frozen-speed cure).
- Britz, Østerby, Strutwolf 2003, "Damping of Crank–Nicolson error oscillations" —
  https://www.sciencedirect.com/science/article/abs/pii/S009784850200075X.
- Deng, Xie, Matar, Boivin 2025 — https://arxiv.org/abs/2502.02570 (θ-gated SLAU2
  pressure-diffusion; thresholds 0.01/0.99 → not parameter-free).
- Saade, Lohse, Fuster 2023, JCP 476:111865 — pressure–temperature coupled
  all-Mach; implementation base https://basilisk.fr/src/compressible/two-phase.h.
- Solver ground truth: `solver_5eq/solver/five_eq_IMEX/residual.py`
  (`_mixture_impedance`, `_apply_interface_acoustic_riemann`, `implicit_face_pu`);
  prior in-repo ACID/harmonic-Z prototypes: `solver_5eq/solver/He2024/explicit_mmacm_ex.py`
  (lines ~3874 `acid_interface`, ~4128 narrow-band harmonic-Z, ~5290 Lagrangian HLLC).
