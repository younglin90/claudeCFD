# Detailed Mathematical and Numerical Methods — 1D Euler Five-Equation IMEX Solver

> Companion methods document for the 1D Euler manuscript (`solver/five_eq_IMEX/`).
> All equations and pseudocode mirror the production implementation under `FIVE_EQ_IMEX_TIME_INTEGRATOR=imex_ssp3`, `FIVE_EQ_IMEX_ALPHA_SCHEME=adaptive_bvd`, `FIVE_EQ_IMEX_PRIMITIVE_SCHEME=tmlpu`, `FIVE_EQ_IMEX_TMLPU_TVD=superbee`, `FIVE_EQ_IMEX_MATERIAL_FLUX=slau2`.
> Section numbers match the main manuscript (`1d_euler_imex_method_revised.md`).

---

## 1. Governing equations

### 1.1 Five-equation diffuse-interface model

Let \(\alpha_1(x,t)\in[0,1]\) be the volume fraction of phase 1 and \(\alpha_2 = 1-\alpha_1\).  Each phase carries its own density \(\rho_k(x,t)\), specific internal energy \(e_k(x,t)\), and temperature \(T_k(x,t)\); the two phases share a single velocity \(u(x,t)\) and a single pressure \(p(x,t)\).  The conservative state vector is

$$
\mathbf{U}(x,t) = \bigl(q_1,\ q_2,\ q_3,\ q_4,\ q_5\bigr)^{\!\top}
= \bigl(\alpha_1\rho_1,\ \alpha_2\rho_2,\ \rho u,\ \rho E,\ \alpha_1\bigr)^{\!\top},
\tag{2.1}
$$

with mixture density \(\rho = \alpha_1\rho_1 + \alpha_2\rho_2\) and total specific energy \(\rho E = \alpha_1\rho_1 e_1 + \alpha_2\rho_2 e_2 + \tfrac12\rho u^2\).  The continuous five-equation system in 1-D is

$$
\partial_t(\alpha_k\rho_k) + \partial_x(\alpha_k\rho_k u) = 0, \qquad k=1,2,
\tag{2.2}
$$

$$
\partial_t(\rho u) + \partial_x(\rho u^2 + p) = 0,
\tag{2.3}
$$

$$
\partial_t(\rho E) + \partial_x\bigl((\rho E + p)u\bigr) = 0,
\tag{2.4}
$$

$$
\partial_t \alpha_1 + u\,\partial_x \alpha_1 \;=\; (\alpha_1 + D_1)\,\partial_x u .
\tag{2.5}
$$

The *non-conservative* coefficient \(D_1\) selects the model variant:

* **Allaire model** (composition advection only): \(D_1 \equiv 0\).
* **Kapila / pressure-equilibrium model** (instantaneous pressure relaxation, weak compaction):
$$
D_1 = \frac{\alpha_1\alpha_2(\rho_2 c_2^{2} - \rho_1 c_1^{2})}
            {\alpha_2\rho_1 c_1^{2} + \alpha_1\rho_2 c_2^{2}},
\tag{2.6}
$$

with phase sound speeds \(c_k\) defined in §1.3.  Murrone & Guillard's reduced five-equation form coincides with the Kapila closure in the same regime.

### 1.2 Choice of primitive variables

Three primitive sets are common: \((\alpha_1,\rho_1,\rho_2,u,p)\), \((\alpha_1,p,T_1,T_2,u)\), and \((\alpha_1,\rho,u,p,Y_1)\).  We use the **temperature-based** primitive

$$
\mathbf{W} = \bigl(\alpha_1,\ T_1,\ T_2,\ u,\ p\bigr)^{\!\top},
\tag{2.7}
$$

for three reasons.  (i) \(p\) is a primitive unknown, which removes the dominant source of pressure round-off in the conservative-to-primitive inversion.  (ii) The implicit pressure step in §3 solves for \(p\) and \(u\) simultaneously, so \((u,p)\) belonging to \(\mathbf{W}\) yields a 2×2 acoustic block in the Newton system.  (iii) Per-phase temperatures \(T_k\) are the natural arguments for the Noble-Abel stiffened-gas (NASG) equation of state, where \(\rho_k = \rho_k(p,T_k)\) and \(e_k = e_k(p,T_k)\) are explicit closed-form expressions.

### 1.3 Equations of state

Three EOS families are admissible per phase: ideal gas (IG), stiffened gas (SG), and Noble-Abel stiffened gas (NASG).  The user assigns each phase independently, so air-water and helium-air mixtures coexist in the validation suite without code branching.

**Ideal gas.**  With ratio of specific heats \(\gamma_k\) and gas constant \(R_k\):

$$
\rho_k(p,T_k) = \frac{p}{R_k T_k}, \qquad
e_k(\rho_k,p) = \frac{p}{(\gamma_k-1)\rho_k}, \qquad
c_k^{2} = \gamma_k\frac{p}{\rho_k}.
\tag{2.8}
$$

**Stiffened gas** (parameter \(p_{\infty,k}\)):

$$
\rho_k(p,T_k) = \frac{p + p_{\infty,k}}{(\gamma_k-1) c_{v,k} T_k}, \qquad
e_k(\rho_k,p) = \frac{p + \gamma_k p_{\infty,k}}{(\gamma_k-1)\rho_k}, \qquad
c_k^{2} = \gamma_k\frac{p + p_{\infty,k}}{\rho_k}.
\tag{2.9}
$$

**Noble-Abel stiffened gas** (covolume \(b_k\), \(p_{\infty,k}\)):

$$
\rho_k(p,T_k) = \frac{1}{b_k + (\gamma_k-1) c_{v,k} T_k / (p+p_{\infty,k})},
\tag{2.10a}
$$

$$
e_k(\rho_k,p) = \frac{p + \gamma_k p_{\infty,k}}{(\gamma_k-1)\rho_k}\,\bigl(1 - b_k \rho_k\bigr) + q_k,
\tag{2.10b}
$$

$$
c_k^{2} = \gamma_k\frac{p + p_{\infty,k}}{\rho_k(1 - b_k\rho_k)} .
\tag{2.10c}
$$

NASG is required for water in strong gas-liquid interaction tests because a plain SG distorts liquid-density levels at moderate \(p\); the implementation of `eos_facade.py` instantiates per-phase EOS objects so that, e.g., phase 1 is `IdealEOS(\gamma_1=1.4)` (air) and phase 2 is `NASGEOS(\gamma_2,p_{\infty,2},b_2,c_{v,2},q_2)` (water).

### 1.4 Analytic primitive-conservative Jacobian

The implicit pressure block in §3.1 requires the closed-form Jacobian \(d\mathbf{U}/d\mathbf{W}\).  Using \(\mathbf{W}=(\alpha_1,T_1,T_2,u,p)^{\!\top}\) and abbreviating \(\rho_{k,p}\equiv\partial\rho_k/\partial p\big|_{T_k}\), \(\rho_{k,T}\equiv\partial\rho_k/\partial T_k\big|_{p}\), \(e_{k,p}\equiv\partial e_k/\partial p\big|_{T_k}\), \(e_{k,T}\equiv\partial e_k/\partial T_k\big|_{p}\):

$$
\frac{d\mathbf{U}}{d\mathbf{W}} =
\begin{pmatrix}
\rho_1 & \alpha_1\rho_{1,T} & 0 & 0 & \alpha_1\rho_{1,p}\\
-\rho_2 & 0 & \alpha_2\rho_{2,T} & 0 & \alpha_2\rho_{2,p}\\
(\rho_1-\rho_2)u & \alpha_1\rho_{1,T}u & \alpha_2\rho_{2,T}u & \rho & \rho_p u\\
J_{4,1} & J_{4,2} & J_{4,3} & \rho u & J_{4,5}\\
1 & 0 & 0 & 0 & 0
\end{pmatrix},
\tag{2.11}
$$

with

$$
\rho_p = \alpha_1\rho_{1,p} + \alpha_2\rho_{2,p},\qquad
J_{4,1} = \rho_1 e_1 - \rho_2 e_2 + \tfrac12 (\rho_1-\rho_2) u^2,
$$

$$
J_{4,2} = \alpha_1\bigl(\rho_{1,T} e_1 + \rho_1 e_{1,T}\bigr),\qquad
J_{4,3} = \alpha_2\bigl(\rho_{2,T} e_2 + \rho_2 e_{2,T}\bigr),
$$

$$
J_{4,5} = \alpha_1\bigl(\rho_{1,p}e_1 + \rho_1 e_{1,p}\bigr) + \alpha_2\bigl(\rho_{2,p}e_2 + \rho_2 e_{2,p}\bigr) + \tfrac12 \rho_p u^2 .
$$

For ideal gas the four EOS derivatives are explicit closed-form; for SG and NASG they are obtained by symbolic differentiation of (2.9)–(2.10) and verified by central finite differences in the unit-test suite (`tests/test_eos_derivatives.py`).  All entries of (2.11) are evaluated analytically — no autograd or finite-difference fallback — and the solver disables Rusanov fallback in the production evidence to ensure that any Jacobian inconsistency surfaces immediately as a Newton-divergence failure.

### 1.5 Sound speed and acoustic impedance

Per-phase sound speed is \(c_k\) from (2.8)/(2.9)/(2.10c).  The frozen mixture sound speed used by SLAU2 is the Wood-style harmonic average,

$$
\frac{1}{\rho c_{\text{mix}}^{2}} = \frac{\alpha_1}{\rho_1 c_1^{2}} + \frac{\alpha_2}{\rho_2 c_2^{2}}.
\tag{2.12}
$$

Acoustic impedance \(Z_k = \rho_k c_k\) governs reflection and transmission across material interfaces; the 07_B Air-Water case is intentionally severe with \(Z_2/Z_1 \approx 3.34\times 10^{3}\).

### 1.6 Continuous pressure-equilibrium invariant

Suppose at \(t=t_0\) the spatial state is \(p\equiv p_0\) constant, \(u\equiv u_0\) constant, with arbitrary \(\alpha_1(x), T_1(x), T_2(x)\).  Substitution into (2.2)–(2.5) gives \(\partial_t p \equiv 0\) and \(\partial_t u \equiv 0\) as long as the EOS closures keep \(p\) consistent with \((\rho_k,T_k)\); the \(\alpha\) and phase-mass equations reduce to pure advection \(\partial_t f + u_0\partial_x f = 0\).  This \(p_0\)-\(u_0\) state is therefore a manifold of invariants of the continuous PDE.  Discrete preservation of this manifold is what §3.6 (PE target recovery) addresses.

---

## 2. Finite-volume discretization

### 2.1 Cell averaging

Subdivide \([x_L,x_R]\) into \(N\) cells \(C_i = [x_{i-1/2}, x_{i+1/2}]\) of equal width \(\Delta x\).  Define the cell average

$$
\overline{\mathbf{U}}_i(t) = \frac{1}{\Delta x}\int_{C_i}\mathbf{U}(x,t)\,dx .
\tag{3.1}
$$

Integrating (2.2)–(2.4) over \(C_i\) gives the conservative semi-discrete form

$$
\frac{d\overline{\mathbf{U}}_i}{dt} = -\frac{\mathbf{F}_{i+1/2}-\mathbf{F}_{i-1/2}}{\Delta x} + \mathbf{H}_i ,
\tag{3.2}
$$

with the physical flux components

$$
\mathbf{F}(\mathbf{U}) = \bigl(\alpha_1\rho_1 u,\ \alpha_2\rho_2 u,\ \rho u^2 + p,\ (\rho E+p)u,\ 0\bigr)^{\!\top},
\tag{3.3}
$$

and the non-conservative source for \(\alpha_1\),

$$
\mathbf{H}_i = \bigl(0,0,0,0,\ -u_i\,(\partial_x\alpha_1)_i + (\alpha_1+D_1)_i (\partial_x u)_i\bigr)^{\!\top}.
\tag{3.4}
$$

### 2.2 Operator split: explicit material vs. implicit acoustic

The spatial residual is split into

$$
\mathcal{L}(\mathbf{W}) = \mathcal{L}_E(\mathbf{W}) + \mathcal{L}_I(\mathbf{W}),\qquad
\frac{d\mathbf{U}}{dt} + \mathcal{L}_E(\mathbf{W}) + \mathcal{L}_I(\mathbf{W}) = 0,
\tag{3.5}
$$

where the **explicit operator** \(\mathcal{L}_E\) advects \(\alpha\), the phase masses \(q_1,q_2\), the inertial momentum \(\rho u^2\), and the kinetic+APEC energy at the SLAU2 face velocity \(u_f\), and the **implicit operator** \(\mathcal{L}_I\) carries the acoustic pressure-gradient and pressure-work terms

$$
\mathcal{L}_I^{(\rho u)} = \frac{p_{i+1/2}-p_{i-1/2}}{\Delta x},
\qquad
\mathcal{L}_I^{(\rho E)} = \frac{(p u)_{i+1/2}-(p u)_{i-1/2}}{\Delta x}.
\tag{3.6}
$$

The split is chosen so that the acoustic CFL scales with the implicit step (no \(c\Delta t/\Delta x\) restriction) while the material CFL — which scales with \(|u|\Delta t/\Delta x\) — sets the practical step.  This is the standard reason for IMEX in low-Mach two-phase flows.

---

## 3. Time integration: IMEX-SSP3(4,3,3)

### 3.1 Pareschi-Russo SSP3 stage residual

The time integrator is the third-order strong-stability-preserving (SSP) IMEX of Pareschi & Russo, with the SSP3(4,3,3) tableau used by Boscheri & Pareschi.  Sign convention follows (3.5):  \(d\mathbf{U}/dt + \mathcal{L}_E + \mathcal{L}_I = 0\).

**Coefficients** (from `solver/five_eq_IMEX/time_integrator.py`):

$$
\gamma = 0.241694260788213,\qquad
\beta = 0.060423565197050,\qquad
\eta = 0.129152869605900,
$$

$$
\delta = \tfrac12 - \beta - \eta - \gamma .
$$

**Explicit Butcher matrix** \(A_E\) (lower triangular):

$$
A_E =
\begin{pmatrix}
0 & 0 & 0 & 0\\
0 & 0 & 0 & 0\\
0 & 1 & 0 & 0\\
0 & \tfrac14 & \tfrac14 & 0
\end{pmatrix}.
\tag{4.1}
$$

**Implicit Butcher matrix** \(A_I\) (lower triangular with diagonal \(\gamma\)):

$$
A_I =
\begin{pmatrix}
\gamma & 0 & 0 & 0\\
-\gamma & \gamma & 0 & 0\\
0 & 1-\gamma & \gamma & 0\\
\beta & \eta & \delta & \gamma
\end{pmatrix}.
\tag{4.2}
$$

**Final-update weights** (equal for explicit and implicit branches):

$$
b_E = b_I = \bigl(0,\ \tfrac16,\ \tfrac16,\ \tfrac23\bigr).
\tag{4.3}
$$

The diagonal \(\gamma\) is fixed; the equality \(b_E=b_I\) gives stage-blending consistency (explicit and implicit contributions accumulate with the same final weights), which is required for the third-order accuracy and the SSP property.

### 3.2 Stage residual equations

For stage \(s = 1,\dots,4\), define the *stage target* in conservative form:

$$
\mathbf{U}_s^{\!*} = \mathbf{U}^{\,n} - \Delta t \sum_{j<s}\Bigl[(A_E)_{sj}\,\mathcal{L}_E(\mathbf{W}_j) + (A_I)_{sj}\,\mathcal{L}_I(\mathbf{W}_j)\Bigr] .
\tag{4.4}
$$

The unknown stage state \(\mathbf{W}_s\) is the root of the implicit residual

$$
\mathcal{R}_s(\mathbf{W}_s) \;:=\; \frac{\mathbf{U}(\mathbf{W}_s) - \mathbf{U}_s^{\!*}}{(A_I)_{ss}\,\Delta t} \;+\; \mathcal{L}_I(\mathbf{W}_s) \;=\; \mathbf{0}.
\tag{4.5}
$$

Writing \(\Delta\tau \equiv (A_I)_{ss}\Delta t = \gamma\Delta t\), the linearised Newton system at iterate \(\mathbf{W}_s^{(m)}\) is

$$
\Bigl[\frac{1}{\Delta\tau}\,\frac{d\mathbf{U}}{d\mathbf{W}}(\mathbf{W}_s^{(m)}) \;+\; \mathbf{J}_I(\mathbf{W}_s^{(m)})\Bigr]\,\delta\mathbf{W}
\;=\; -\,\mathcal{R}_s(\mathbf{W}_s^{(m)}),
\tag{4.6}
$$

with \(\mathbf{J}_I = \partial\mathcal{L}_I/\partial\mathbf{W}\) the implicit-flux Jacobian (analytic; only the momentum and energy rows are non-zero).  The Newton update is

$$
\mathbf{W}_s^{(m+1)} = \mathbf{W}_s^{(m)} + \lambda^{(m)}\,\delta\mathbf{W},
\tag{4.7}
$$

with line-search step \(\lambda^{(m)}\in(0,1]\) determined by a positivity-preserving guard: \(\alpha_1, T_1, T_2 > 0\) and EOS-admissibility (e.g.\ \(1-b_k\rho_k>0\) for NASG) must all hold at \(\mathbf{W}_s^{(m+1)}\).  If a guard fires, \(\lambda\) is halved.

Convergence criterion: relative update norm

$$
\bigl\|\delta\mathbf{W}\bigr\|_{\!2,\text{rel}} \;:=\; \sqrt{\frac{1}{N}\sum_i\sum_v\Bigl(\frac{\delta W_{v,i}}{\|W_v\|_\infty + \varepsilon_v}\Bigr)^{\!2}}\;<\;10^{-10}.
\tag{4.8}
$$

In the production evidence on the Air-Water 07_B case at \(N=400\), the average Newton iteration count per stage is 2-3 with no line-search activations; on the hypersonic 24_H case at \(N=400\) the count grows to 4-6.

### 3.3 Final SSP combination

After all four stages,

$$
\mathbf{U}^{\,n+1} = \mathbf{U}^{\,n} - \Delta t \sum_{s=1}^{4}\Bigl[(b_E)_s\,\mathcal{L}_E(\mathbf{W}_s) + (b_I)_s\,\mathcal{L}_I(\mathbf{W}_s)\Bigr].
\tag{4.9}
$$

The final \(\mathbf{W}^{\,n+1}\) is recovered from \(\mathbf{U}^{\,n+1}\) by the analytic inversion described in §3.6.

### 3.4 Time-step selection

The production code uses a material-CFL bound:

$$
\Delta t = \mathrm{CFL}\,\frac{\Delta x}{\max_i|u_i|}, \qquad \mathrm{CFL} = 0.4 .
\tag{4.10}
$$

No acoustic-CFL bound is enforced because the implicit operator is L-stable.  The acoustic-CFL **sweep** in §6.7 of the main manuscript is a diagnostic, not a stability requirement.

---

## 4. Spatial discretization details

### 4.1 SLAU2-type material face velocity

`solver/five_eq_IMEX/imex_ad.py::_slau2_faces_np` implements the SLAU2 face state used by every interior face:

$$
v_{\text{avg}} = \frac{\sqrt{\rho_L}\,u_L + \sqrt{\rho_R}\,u_R}{\sqrt{\rho_L}+\sqrt{\rho_R}},\qquad
\bar c = \tfrac12 (c_L + c_R),\qquad
\bar\rho = \tfrac12(\rho_L+\rho_R),
\tag{5.1}
$$

$$
u_{\text{rms}} = \sqrt{\tfrac12(u_L^{2} + u_R^{2})},\qquad
\hat M = \min\!\Bigl(1,\ \frac{u_{\text{rms}}}{\bar c}\Bigr),\qquad
\chi(\hat M) = (1-\hat M)^{2},
\tag{5.2}
$$

$$
\boxed{\;u_f \;=\; v_{\text{avg}} \;-\; \chi(\hat M)\,\frac{p_R - p_L}{\bar\rho\,\bar c} ,\qquad p_f = \tfrac12(p_L + p_R).\;}
\tag{5.3}
$$

The reconstructed left/right states \((u_L,u_R,p_L,p_R,\rho_L,\rho_R)\) come from the T-MLP-u limiter (§4.2) applied component-wise to \((T_1,T_2,u,p)\), with \(\rho_k\) reconstructed via the EOS at the reconstructed \((p,T_k)\).  The Roe-averaged \(v_{\text{avg}}\) preserves the high-Mach Roe limit; the \(\chi(\hat M)\) prefactor vanishes at high Mach (recovering pure Roe flux) and remains active at low Mach where it provides the low-dissipation pressure-velocity coupling.

For the **interface mass flux** the same \(u_f\) is used:

$$
F_{q_k,\,i+1/2} = (\alpha_k)_f\,(\rho_k)_f\,u_f,
\tag{5.4}
$$

with \((\alpha_k)_f\) selected by the upwind sign of \(u_f\) and \((\rho_k)_f\) reconstructed from \((\hat p_f,\hat T_{k,f})\) via the per-phase EOS — the "ACID-style" face thermodynamics in the comments of `face_state.py`.  This avoids bulk-density transfer across high-density-ratio interfaces, where a centred \(\rho_k\) would feed an interface cell with mass that does not belong to it.

### 4.2 T-MLP-u primitive reconstruction

For each cell-centred primitive \(q\in\{T_1,T_2,u,p\}\) the candidate left face value at \(i+\tfrac12\) is the MUSCL form

$$
q_{i+1/2}^{L,*} = q_i + \tfrac12\,\psi(r_i)\,(q_{i+1}-q_i),\qquad
r_i = \frac{q_i - q_{i-1}}{q_{i+1} - q_i + \mathrm{sgn}\,\varepsilon}.
\tag{5.5}
$$

The base limiter is **Roe's superbee**:

$$
\psi_{\text{SB}}(r) = \max\!\bigl(0,\ \min(2r,1),\ \min(r,2)\bigr).
\tag{5.6}
$$

The **T-MLP-u wrapper** then enforces a local maximum-principle (LMP) bound by clipping to the three-cell window of \(q\):

$$
q_{i+1/2}^{L} = \mathrm{clip}\!\Bigl(q_{i+1/2}^{L,*},\ \min(q_{i-1},q_i,q_{i+1}),\ \max(q_{i-1},q_i,q_{i+1})\Bigr).
\tag{5.7}
$$

Equivalently, defining the LMP-derived limiter

$$
\psi_{\text{MLP}} =
\begin{cases}
\dfrac{\max(q_{i-1},q_i,q_{i+1}) - q_i}{\tfrac12(q_{i+1}-q_i)} & \text{if }\delta>0,\\[1.2ex]
\dfrac{\min(q_{i-1},q_i,q_{i+1}) - q_i}{\tfrac12(q_{i+1}-q_i)} & \text{if }\delta<0,
\end{cases}
\tag{5.8}
$$

with \(\delta = \tfrac12(q_{i+1}-q_i)\), the final limiter is

$$
\boxed{\;\psi_{\text{T-MLP-u}}(r) \;=\; \max\!\bigl(0,\ \min(2,\ \psi_{\text{SB}}(r),\ \psi_{\text{MLP}})\bigr).\;}
\tag{5.9}
$$

The wrapper preserves the useful compressive range \(0\le\psi\le 2\) of the base TVD limiter while preventing creation of new primitive extrema across the three-cell stencil.  The right-state \(q_{i+1/2}^{R}\) is constructed symmetrically from the cell to the right with mirrored \(r_{i+1}\).  The corresponding code path is `solver/five_eq_IMEX/limiters.py::t_mlp_u_face_value`.

A **cavitation safeguard** replaces superbee with van Leer (a smoother base) in homogeneous double-rarefaction topologies, identified parameter-free by the local sign pattern of \(u_x\) and \(\alpha_x\).  This is a state-topology rule, not a case-ID switch, and is exercised most prominently in 15_E.

### 4.3 Adaptive-BVD volume-fraction transport

\(\alpha_1\) is reconstructed by an **adaptive-BVD** logic that switches between a CICSAM-style compressive construction near pure 0/1 contacts and a bounded MUSCL-Hancock TVD branch in mixed regions.  The local indicator at face \(i+\tfrac12\) uses the maximum cell-stencil \(\alpha\) gradient and a pure-phase tolerance \(\eta_{\text{pure}}=10^{-12}\):

$$
\text{interface}_{i+1/2} =
\bigl[\,\min(\alpha_{i-1},\alpha_i,\alpha_{i+1},\alpha_{i+2}) < \eta_{\text{pure}}\,\bigr]\;\lor\;
\bigl[\,\max(\alpha_{i-1},\alpha_i,\alpha_{i+1},\alpha_{i+2}) > 1 - \eta_{\text{pure}}\,\bigr].
\tag{5.10}
$$

When the indicator is true the CICSAM compressive branch is used:

$$
\tilde\alpha_C = \frac{\alpha_C - \alpha_U}{\alpha_D - \alpha_U},\qquad
\tilde\alpha_f^{\text{HC}} = \min\!\Bigl(1,\ \frac{\tilde\alpha_C}{\mathrm{Co}_f}\Bigr),
\tag{5.11}
$$

with \(\mathrm{Co}_f = |u_f|\Delta t/\Delta x\) the per-face Courant number; the upwind \((U)\) / centre \((C)\) / downstream \((D)\) labels follow the sign of \(u_f\).  When the indicator is false a MUSCL-Hancock TVD reconstruction is used with a minmod limiter to preserve monotonicity in mixed-composition regions.

**Conservative flux-corrected sharpening.**  The compressive sharpening on \(\alpha_1\) induces corresponding corrections in \((\alpha_1\rho_1, \alpha_2\rho_2, \rho u, \rho E)\) so that conservative variables stay consistent with the sharpened \(\alpha\).  A single local FCT factor

$$
\theta_{i+1/2} \;=\; \min\!\Bigl(1,\ \min_{v\in\{q_1,q_2,q_3,q_4\}}\Theta_v\Bigr)\;\in[0,1]
\tag{5.12}
$$

is computed so that the sharpened update remains within the per-cell admissibility cone (\(0\le\alpha_1\le1\), \(\rho_k>0\), \(p>p_{\text{min}}\), \(T_k>T_{\text{min}}\)).  The same \(\theta\) multiplies all four conservative corrections, so the discrete update is exactly conservative.  This "single-knob FCT" is what prevents sharpening \(\alpha\) alone from creating non-physical density spikes at contacts.

### 4.4 Characteristic-reconstruction policy

The solver may reconstruct in characteristic variables, but **only on composition-uniform stencils** where \(\alpha_1\) is constant across \(\{i-1,i,i+1\}\) to within a tight tolerance.  The detector returns false at any face within a stencil that crosses a material interface; in that case reconstruction falls back to EOS-consistent primitive (or mixture-scalar) reconstruction (§4.2).  This rule is uniform across all validation cases.

The corresponding code path is `solver/five_eq_IMEX/imex_ad.py::_characteristic_recon_allowed` (state-topology detector) and `solver/five_eq_IMEX/imex_ad.py::_characteristic_mixture_lr_states` (the actual characteristic projection).

### 4.5 APEC energy flux

The advective energy flux uses the APEC (Adjoint Phasic Energy Coupling) decomposition,

$$
F_{\rho E,\,i+1/2}^{\,(\text{APEC})}
= \chi_{1,f}\,F_{q_1,f} + \chi_{2,f}\,F_{q_2,f} + \chi_{a,f}\,F_{\alpha,f} + \tfrac12 u_f^{2}\,F_{\rho,f},
\tag{5.13}
$$

with the per-phase enthalpy-like coefficients

$$
\chi_{k,f} = e_{k,f} + \frac{\rho_{k,f}\,e_{k,T,f}}{\rho_{k,T,f}},\qquad
\chi_{a,f} = -\frac{\rho_{1,f}^{2}\,e_{1,T,f}}{\rho_{1,T,f}} + \frac{\rho_{2,f}^{2}\,e_{2,T,f}}{\rho_{2,T,f}},
\tag{5.14}
$$

evaluated face-wise from the EOS at \((\hat p_f, \hat T_{k,f})\).  The fallback for \(|\rho_T|\to 0\) (pure-phase limit) is the simpler Allaire-style \(e_{f,\text{up}}F_{q}\) form, automatically engaged when \((\alpha_k)_f<\eta_\text{pure}\).  The pressure-work term \(p\,u\) is **not** included in \(F_{\rho E}^{\,(\text{APEC})}\) — it lives in the implicit operator \(\mathcal{L}_I\) (3.6).  The corresponding code path is `solver/five_eq_IMEX/energy_flux.py`.

### 4.6 Implicit acoustic flux

The implicit operator (3.6) uses a centred discretisation of \(p\) and \(p u\):

$$
(\mathcal{L}_I^{\,\rho u})_i = \frac{p_{i+1/2} - p_{i-1/2}}{\Delta x},\qquad
(\mathcal{L}_I^{\,\rho E})_i = \frac{(pu)_{i+1/2} - (pu)_{i-1/2}}{\Delta x},
\tag{5.15}
$$

with the face values

$$
p_{i+1/2} = \tfrac12 (p_i + p_{i+1}) + \alpha_{\text{RC}}\,\bigl[\bar\rho\,\bar c\,(u_i - u_{i+1})\bigr],
\tag{5.16}
$$

$$
u_{i+1/2}^{\,\text{(implicit)}} = \tfrac12 (u_i + u_{i+1}) + \alpha_{\text{RC}}\,\bigl[(p_{i+1}-p_i)/(\bar\rho\,\bar c)\bigr].
\tag{5.17}
$$

The \(\alpha_{\text{RC}}\) prefactor (Rhie-Chow-style) is normally set to zero in 1-D since the centred form is already free of checkerboard modes once SLAU2 is used for the explicit branch; it is exposed for diagnostic purposes only.

---

## 5. Pressure-equilibrium target recovery

### 5.1 Continuous invariant restated

From §1.6, a state with constant \(p_0,u_0\) and arbitrary \(\alpha_1(x), T_1(x), T_2(x)\) is an invariant of the continuous five-equation system: \(\partial_t p \equiv 0\), \(\partial_t u \equiv 0\), and \(\alpha_1, T_1, T_2\) are advected at \(u_0\).  Discretisation must preserve this manifold.

### 5.2 Why naive U-to-W inversion fails

In a discrete step, an exact PE state at time \(t^n\) advances to \(t^{n+1}\) by (4.9) and then is converted from \(\mathbf{U}^{\,n+1}\) back to \(\mathbf{W}^{\,n+1}\).  The conversion is a 5-D nonlinear root finding (the EOS for \((\rho_k,T_k,p)\) is implicit); even when the conservative variables satisfy the PE constraint exactly, floating-point round-off of order \(\varepsilon\,\kappa\bigl(d\mathbf{U}/d\mathbf{W}\bigr)\) leaks into \(p\) and \(u\).  Iterated over thousands of steps, this round-off drifts the discrete pressure and velocity above machine precision and contaminates downstream acoustic diagnostics.

### 5.3 PE detector

The PE detector flags the discrete state as on-manifold when

$$
\frac{\max_i|p_i - \langle p\rangle|}{\langle p\rangle} < \tau_p \quad\text{and}\quad
\max_i |u_i - \langle u\rangle| < \tau_u\,\langle c\rangle,
\tag{6.1}
$$

with default tolerances \(\tau_p = 10^{-10}\) and \(\tau_u = 10^{-10}\), and \(\langle\cdot\rangle\) the spatial average.  Only when both conditions hold is the recovery engaged.

### 5.4 Manifold projection

When the detector fires, the conservative-to-primitive conversion is constrained:

$$
p^{\,n+1} \leftarrow p_0,\qquad u^{\,n+1} \leftarrow u_0,
\tag{6.2}
$$

while \(\alpha_1, T_1, T_2\) are recovered cell-locally from the conservative target \((q_1,q_2,q_5)_i\) via the EOS (a 3-equation root-finding that is well-conditioned on the PE manifold).  Concretely:

$$
\alpha_1^{\,n+1}_i = q_5^{\,n+1}_i,
$$

$$
T_1^{\,n+1}_i \text{ such that } \alpha_1\rho_1(p_0,T_1^{\,n+1})_i = q_1^{\,n+1}_i,
$$

$$
T_2^{\,n+1}_i \text{ such that } (1-\alpha_1)\rho_2(p_0,T_2^{\,n+1})_i = q_2^{\,n+1}_i.
$$

This is **not** a spatial remap.  Each cell's recovery is local; no values are copied between cells; the conservative variables are unchanged.  In particular, mass, momentum, and total energy of the cell are exactly preserved by the recovery (it is a re-parameterisation of the same point in state space).

### 5.5 Why this is principled, not a hack

Three independent reasons.  (i) **Continuous invariant.** \(\partial_t p=\partial_t u=0\) is a property of the PDE; the recovery is the discrete realisation of that invariant.  (ii) **Conservative.** The recovery only constrains the primitive *re-parameterisation*; conservative variables and discrete fluxes are unchanged.  (iii) **No spatial remap.** The production evidence runs with `FIVE_EQ_IMEX_UNIFORM_PERIODIC_REMAP=0`; the periodic-remap shortcut is explicitly disabled.  Ablation (§6.6 of the main manuscript) shows that without recovery, p/u Linf in PE cases drift several orders of magnitude above machine epsilon over the integration time.

### 5.6 Pseudocode

```text
function step(U^n, dt):
    # Stage loop
    for s in 1..4:
        U_star = U^n − dt · Σ_{j<s} [ (A_E)_{sj} · L_E(W_j) + (A_I)_{sj} · L_I(W_j) ]
        W_s = NewtonSolve(R_s(W) := (U(W) − U_star) / (γ · dt) + L_I(W) = 0,
                          initial_guess = W_n, jac = (1/(γ·dt))·dU/dW + dL_I/dW,
                          line_search = positivity-preserving)
    # SSP combination
    U^{n+1} = U^n − dt · Σ_{s=1..4} [ b_E_s · L_E(W_s) + b_I_s · L_I(W_s) ]
    # Conservative-to-primitive recovery
    if PE_detector(U^{n+1}):
        W^{n+1} = PE_constrained_inversion(U^{n+1}, p_0, u_0)   # §5.4
    else:
        W^{n+1} = analytic_inversion(U^{n+1})                   # standard path
    return U^{n+1}, W^{n+1}
```

`solver/five_eq_IMEX/time_integrator.py::imex_ssp3_step` is the corresponding routine; PE detection and projection are at `_pe_projection_allowed` and `_project_conservative_target_to_pe`.

---

## 6. Production algorithm summary

```text
PRODUCTION CONFIGURATION (env vars):
    TIME_INTEGRATOR    = imex_ssp3
    ALPHA_SCHEME       = adaptive_bvd
    PRIMITIVE_SCHEME   = tmlpu
    TMLPU_TVD          = superbee
    MATERIAL_FLUX      = slau2
    PRESSURE_CLOSURE   = regime_auto
    CHARACTERISTIC_RECON = 1   (composition-uniform stencils only)
    RUSANOV_FALLBACK   = 0     (disabled)
    UNIFORM_PERIODIC_REMAP = 0 (disabled)

FOR EACH TIME STEP Δt:
    1.  Recover W from U using the analytic dU/dW Jacobian (§1.4).
    2.  Apply boundary states; compute c_k, c_mix, Z_k.
    3.  Reconstruct primitive variables (T_1, T_2, u, p) by T-MLP-u + superbee + LMP (§4.2).
    4.  Reconstruct α_1 by adaptive-BVD with conservative FCT limiting (§4.3).
    5.  At each face: build SLAU2 face velocity u_f and pressure p_f (§4.1).
    6.  IMEX-SSP3 stage loop:
        a.  For s = 1..4: form U_star = U^n − dt Σ_{j<s} (A_E_sj L_E(W_j) + A_I_sj L_I(W_j))
        b.  Newton-solve R_s(W_s) = 0 with line-search positivity guard (§3.2).
    7.  Combine stages: U^{n+1} = U^n − dt Σ (b_E_s L_E(W_s) + b_I_s L_I(W_s)) (§3.3).
    8.  Recover W^{n+1} from U^{n+1}; engage PE recovery only when detector fires (§5).
    9.  Write per-case diagnostics and overwrite results/1D/{case}/diff_vs_exact.png.
```

---

## 7. Dimensional consistency check

The combination is dimensionally consistent.  Sound speed \([c]=L/T\); pressure \([p]=M/(L T^2)\); density \([\rho]=M/L^3\); SLAU2 χ correction is dimensionless; LMP-clipped face value has the same units as \(q\); Newton system in (4.6) has units \([(d\mathbf{U}/d\mathbf{W})]\,/[T]\) on the time-dilation block and \([\partial\mathcal{L}_I/\partial\mathbf{W}]\) on the implicit-flux block, both of which scale as the conservative variable per unit time per primitive — consistent.  The PE-recovery projection is a unitless re-parameterisation in state space.

---

## 8. Implementation file map

| Concept | File | Key symbol |
|---|---|---|
| Governing equations / primitive choice | `solver/five_eq_IMEX/main.py`, `nd_solver.py` | `solve(...)` entry point |
| EOS facade (Ideal/SG/NASG) | `solver/five_eq_IMEX/eos_facade.py` | `to_eos(...)`, `EOSPair` |
| Analytic dU/dW Jacobian | `solver/five_eq_IMEX/jacobian.py` | `dUdW_analytic`, `prim_to_cons_W`, `cons_to_prim_W` |
| IMEX-SSP3 stage residual | `solver/five_eq_IMEX/time_integrator.py` | `imex_ssp3_step`, `SSP3_A_E`, `SSP3_A_I`, `SSP3_B_E`, `SSP3_B_I` |
| SLAU2 face velocity | `solver/five_eq_IMEX/imex_ad.py` | `_slau2_faces_np` (lines 599–658) |
| T-MLP-u + superbee | `solver/five_eq_IMEX/limiters.py` | `t_mlp_u_face_value` |
| Adaptive-BVD α transport | `solver/five_eq_IMEX/face_state.py`, `imex_ad.py` | `face_state(...)`, `_adaptive_bvd_branch` |
| APEC energy flux | `solver/five_eq_IMEX/energy_flux.py` | `total_energy_flux` |
| Implicit acoustic operator | `solver/five_eq_IMEX/imex_ad.py` | `_L_I(...)` |
| Newton + line search | `solver/five_eq_IMEX/newton.py` | `newton_solve(...)` |
| PE detector / projection | `solver/five_eq_IMEX/time_integrator.py` | `_pe_projection_allowed`, `_project_conservative_target_to_pe` |

---

*End of methods document.*
