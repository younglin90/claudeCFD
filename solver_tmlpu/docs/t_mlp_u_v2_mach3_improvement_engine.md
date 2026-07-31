# T-MLP-u-v2 Improvement Engine for Mach 3 Forward-Facing Step, 200x80

## Active Mach 3 Step Contract

The current long-term benchmark contract for this repository lives in
`.omx/specs/autoresearch-mach3-step-strict/CODEX_APP_HANDOFF.md`.

For the current strict Mach 3 forward-facing step work, keep the following
defaults unless the user explicitly changes them:

- Initial state: `rho = 1.4`, `p = 1.0`, `u = 3*c`, `v = 0.0`,
  `c = sqrt(1.4 * 1.0 / 1.4)`
- Boundary conditions: left Dirichlet inflow, right transmissive outflow,
  reflective solid walls
- Mesh family: `triangulate_box_roi_graded`
- Quick/diagnostic grid: `200 x 80` on standard ROI-graded mesh
- Paper/final grid: `480 x 160`
- Upper ROI gate: `x in [0.5, 3.0]`, `y in [0.6, 1.0]`
- Step-top reflected shock gate: `x in [1.0, 1.6]`, `y in [0.2, 0.35]`
- Final time: `t = 4.0 s`
- PASS requires both the upper ROI vortex gate and the step-top shock gate
  to succeed, with no carbuncle and no nonphysical shock split.

## 0. Purpose and Non-Negotiable Rules

This document defines a numerical-method improvement engine for improving T-MLP-u against MLP-u1 on the Mach 3 forward-facing step benchmark on a \(120\times40\) mesh. It does not report computed results. Every metric value is **?�인 ?�요** until produced by a fair run.

Rules:

- Do not fabricate results.
- Do not weaken MLP-u1 artificially.
- Keep governing equations, flux function, time integration, CFL, mesh, boundary conditions, residual tolerance, gradient reconstruction, and post-processing identical unless the comparison explicitly studies that factor for both schemes.
- Any sensor or correction added to T-MLP-u must be mathematically defined and recorded. If the same stabilizer is not available to MLP-u1, the paper claim must say that the comparison is between full schemes, not limiter-only cores.
- Mach 3 step \(120\times40\) is a development grid. Any promising result must be checked on at least one finer grid and at least one different benchmark before SCI-level claims.

## 1. Symbolic Goal Score

Until actual data exists, define a symbolic score rather than assigning invented weights:

\[
\begin{aligned}
S=&\ w_1 S_{\text{shock-stability}}
+w_2 S_{\text{carbuncle}}
+w_3 S_{\text{post-shock}}
+w_4 S_{\text{shock-sharpness}}\\
&+w_5 S_{\text{slip-line}}
+w_6 S_{\text{vortex}}
+w_7 S_{\text{boundedness}}
+w_8 S_{\text{residual}}\\
&-w_9 C_{\text{CPU}}
-w_{10} C_{\text{clipping}} .
\end{aligned}
\]

Recommended measurable proxies:

| Term | Metric proxy | Direction |
|---|---|---|
| \(S_{\text{shock-stability}}\) | shock continuity pass, shock-split band count, pressure-gradient line continuity | higher is better |
| \(S_{\text{carbuncle}}\) | carbuncle blob count, odd-even pressure score, isolated spot count | lower is better |
| \(S_{\text{post-shock}}\) | post-shock \(p,\rho\) line-probe peak-to-peak amplitude | lower is better |
| \(S_{\text{shock-sharpness}}\) | shock transition thickness in cells | lower is better unless oscillatory |
| \(S_{\text{slip-line}}\) | \(|\nabla \rho|\), entropy gradient, or vorticity strength along slip line | higher is better if non-oscillatory |
| \(S_{\text{vortex}}\) | coherent upper shear-layer vortex-pair count and density-rollup count | higher is better |
| \(S_{\text{boundedness}}\) | \(p_{\min}>0\), \(\rho_{\min}>0\), overshoot magnitude | higher is better |
| \(S_{\text{residual}}\) | monotone residual trend, absence of dt collapse | higher is better |
| \(C_{\text{CPU}}\) | wall time normalized by MLP-u1 under same thread count | lower is better |
| \(C_{\text{clipping}}\) | fraction of smooth-region faces with \(\psi<0.9\) | lower is better |

Weights must be chosen after seeing metric sensitivity. A defensible starting interpretation is:

\[
w_1,w_2,w_7,w_8 \text{ are hard-gate dominant},\qquad
w_4,w_5,w_6 \text{ decide quality after stability passes}.
\]

### 1.1 Strict Mach 3 ROI Vortex PASS Criterion

The upper ROI vortex gate is stricter than a vorticity-cluster count. A result
must not pass merely because the shear layer contains small wiggles.

For ROI \(x\in[0.55,2.0]\), \(y\in[0.6,0.85]\), PASS requires all of:

- a resolved swirl core with \(Q>0\), \(\lambda_{ci}>0\), coherent vorticity sign, and at least three connected cells;
- an actual density `tricontour` polyline that continuously hooks or spirals around the swirl core, not only density-gradient tangent alignment on an annulus;
- at least one contour segment with continuous angular sweep \(\ge \pi\), with reference-style strong evidence requiring \(\ge 1.5\pi\);
- the swirl core must lie close enough to the hooked contour that the contour passes through a near-core annulus and has visible vertical development;
- detections forming a nearly horizontal row near the lower ROI boundary are classified as shear-sheet wiggles and rejected;
- The gate is now ROI-vortex-clarity based.  T-MLP-u must show clearer
  density-contour hooks around resolved swirl cores inside the ROI than MLP-u1.
  Downstream density count and pair x-extent are diagnostics only, not pass
  requirements.

The old fields `mach3_step_roi_vortex_count`,
`mach3_step_roi_density_winding_count`, and
`mach3_step_roi_vortex_shape_count` are diagnostics only. The hard gate is
`mach3_step_roi_vortex_shape_pass`, which now combines actual contour-hook
evidence, compact local roll-up support, and shear-sheet rejection.  The
comparative gate is `mach3_step_roi_vortex_clarity_better_than_mlp_u1_pass`.

## 2. Current Scheme Audit

The following audit should be run before every modification. Each item must be tied to a diagnostic field or local histogram.

| Failure mode | Likely mechanism in current T-MLP-u | Diagnostic |
|---|---|---|
| carbuncle 발생 | shock-aligned anti-diffusive transverse increment, insufficient crossflow dissipation | pressure checker score, shock split count |
| shock-aligned instability | \(\psi_{TVD}>1\), \(r\)-based compression near normal shock | shock-normal line probe |
| post-shock pressure wiggle | pressure reconstruction too sharp, \(\alpha r\) singular behavior | \(p\) oscillation amplitude, \(p_{\min}\) |
| contact discontinuity smearing | limiter too active on tangential velocity/contact variables | slip-line \(|\nabla\rho|\), entropy gradient |
| shear layer smearing | \(\psi_L=\min\) clips smooth shear regions | vortex-pair count, smooth-region \(\psi\) histogram |
| vortex damping | shock sensor too broad or velocity limiter too dissipative | vorticity roll-up count |
| over-limiting in smooth region | all face-vertex min limiter dominated by one bad vertex | smooth ROI \(\psi<0.9\) ratio |
| under-limiting near shock | \(r>1\) compression and insufficient pressure positivity control | overshoot/undershoot, \(p_{\min}\) |
| pathological \(t^*\) | \(d_{LR}\cdot n_f\) too small or \(t^*\notin[0,1]\) | \(t^*\) histogram, denominator histogram |
| negative or unstable \(\beta\) | normal orientation or \(\hat e_{LR}\cdot n_f<0\) | \(\beta_{\min}\), negative beta count |
| \(r\) blow-up | fixed \(den_{safe}=10^{-8}\), low-gradient denominator | \(|r|_{p99}\), clipped-r count |
| \(\alpha\) singularity | \(r\)-scaled denominator and \(\Delta\phi\approx0\) | \(\alpha_{p99}\), nonfinite alpha count |
| \(\psi_{TVD}\) over-compression | downwind \(\psi_{TVD}=2\) in compressible shock | shock wiggle, carbuncle spots |
| excessive min-limiter clipping | \(\psi_L=\min_{f,V}\psi_{V}\) too global for cell | smooth-region limiter activation map |

## 3. Candidate Modification Generation

These are candidate modifications, not automatic claims. Each must be tested against MLP-u1 and T-MLP-u-old.

| ID | ?�정 ?�??| ?�정??| ?�결?�려??문제 | ?�상 개선 | ?�상 부?�용 | 구현 ?�이??|
|---|---|---|---|---|---|---|
| C01 | \(r\) limiter | \(r_s=(\Delta_R\Delta_L+\epsilon_r^2)/(\Delta_L^2+\epsilon_r^2)\), then \(r=\max(0,r_s)\) | \(den\approx0\), \(r\) blow-up | smoother TVD response | less compression at contacts | low |
| C02 | \(\alpha\) limiter | \(\psi^{bound}_{V}=\max(0,\min(1,B_V/\Delta\phi_V))\), remove \(r\) from \(\alpha\) denominator | \(\alpha\) singularity | clearer DMP proof | may reduce sharpness | low |
| C03 | \(\psi_{TVD}\) adaptive | \(\psi_{\max}=1+(1-\chi_s)(1-\chi_c)\chi_{sh}\), capped \([1,2]\) only away from shocks | over-compression in shocks | shock stability | lower contact compression | medium |
| C04 | shock sensor | \(\chi_s=\mathrm{clip}(|\Delta p|/(p_L+p_R+\epsilon_p),0,1)\) plus compression gate \(\nabla\cdot u<0\) | shock wiggle, positivity loss | pressure stability | can smear shock if too broad | medium |
| C05 | carbuncle sensor | \(\chi_c=\chi_s\,\mathrm{clip}(|\Delta_\perp p|/(|\Delta_n p|+\epsilon_p),0,1)\) or odd-even local pressure alternation | shock-aligned instability | carbuncle suppression | may add transverse diffusion | high |
| C06 | shear sensor | \(\chi_\omega=|\omega|/(|\omega|+|\nabla\cdot u|+\epsilon_u)\) | shear/vortex damping | relax limiter in vortical shear | may permit wiggle near shocks | medium |
| C07 | smoothness sensor | \(\chi_{sm}=1-\mathrm{clip}(\eta_2/(\eta_1+\epsilon),0,1)\), where \(\eta_2\) is LSQ residual | smooth over-limiting | preserve smooth accuracy | sensor cost | medium |
| C08 | \(t^*\) stabilization | use raw \(t^*\) if \(|d_{LR}\cdot n_f|\ge\tau_g|d_{LR}|\), else LSQ face increment; record flag | pathological geometry | avoids blow-up | fallback must be fair and generic | medium |
| C09 | \(\beta\) clipping | \(\beta=\mathrm{clip}(\max(0,\hat e_{LR}\cdot n_f)/\theta_{\min},0,1)\) | negative beta | symmetry and stability | slightly less correction | low |
| C10 | face-gradient damping | \(\nabla\phi_{corr}=\bar\nabla\phi-\beta(1-\chi_s\chi_c)\delta_g\hat e_{LR}\) | shock anti-diffusion | damps shock wiggle | less exact jump correction at shocks | medium |
| C11 | characteristic-variable limiting | limit \(\delta W\) in primitive characteristic fields of normal Euler Jacobian | variable coupling near shocks | pressure/velocity consistency | higher CPU and complexity | high |
| C12 | positivity-preserving correction | after face states, scale increment by \(\lambda_+=\min(1,(\rho_L-\rho_{floor})/(\rho_L-\rho_f), (p_L-p_{floor})/(p_L-p_f))\) when needed | negative \(\rho,p\) | hard positivity | may hide limiter weakness if overused | medium |
| C13 | vortex-preserving relaxation | \(\psi_L\leftarrow \max(\psi_L,\psi_{vort})\) only if \(\chi_\omega\) high and \(\chi_s\) low, while preserving vertex bound | vortex damping | sharper shear roll-up | DMP proof harder | high |
| C14 | anisotropic limiting | apply strong limiting in shock normal direction, weaker tangential limiting: \(\Delta\phi=\psi_n\Delta_n+\psi_t\Delta_t\) | shock vs shear conflict | stable shocks with sharper slip line | more diagnostics needed | high |
| C15 | limiter aggregation | replace hard cell min with bound-preserving face-local limiter for face state and separate vertex limiter for diagnostics | excessive clipping | reduces smooth clipping | may weaken vertex DMP if wrong | high |

## 4. Selected Best Candidate Set for Mach 3 (diagnostic `200x80`, paper/final checks on `480x160`)

For the next v2 design, choose a conservative set that directly addresses the known Mach 3 failure modes without making the method case-specific.

| ?�택???�정 | ?�택 ?�유 | 직접 겨냥?�는 failure mode | ?�상 trade-off |
|---|---|---|---|
| C02 bounded \(\alpha\) separation | gives a clean DMP argument and removes \(r\)-singularity | \(\alpha\) singularity, bound violation | may reduce artificial sharpness |
| C01 scale-aware \(r\) | prevents denominator blow-up while preserving TVD idea | \(r\) blow-up, post-shock wiggle | less compression in weak-gradient contacts |
| C09 clipped \(\beta\) | fixes sign/orientation risk with negligible cost | negative beta, left/right asymmetry | less correction on badly oriented faces |
| C04 shock sensor | needed for pressure-sensitive Mach shocks | post-shock pressure wiggle, positivity | broad sensor can smear slip line |
| C05 carbuncle sensor | specifically targets shock-aligned odd-even instability with generic local detector | carbuncle, shock split | may add dissipation along grid-aligned shocks |
| C06 shear/vortex sensor | prevents shock sensor from killing vortex roll-up | shear layer smearing, vortex damping | can relax limiter too much if shock/shear overlap |
| C10 face-gradient damping | reduces anti-diffusive non-orthogonal correction in strong shocks | shock-aligned instability | lower linear consistency inside shock |
| C12 positivity correction | hard fail-safe for \(\rho,p\), applied as continuous scaling | negative pressure/density | can increase clipping; diagnostic must report activation |

Do not select C13/C14/C15 in v2 unless v2 is stable but too diffusive. They are promising but require a more careful proof and more implementation risk.

## 5. T-MLP-u-v2 Formula

### 5.1 Geometry and \(t^*\)

\[
d_{LR}=c_R-c_L,\qquad
\hat e_{LR}=\frac{d_{LR}}{|d_{LR}|+\epsilon_x}.
\]

\[
D_n=d_{LR}\cdot n_f.
\]

Define a geometry quality indicator:

\[
q_g=\frac{|D_n|}{|d_{LR}|\,|n_f|+\epsilon_x}.
\]

Use

\[
t^*=
\begin{cases}
\dfrac{(m_f-c_L)\cdot n_f}{D_n}, & q_g\ge q_{g,\min},\\
t^{LSQ}_f, & q_g<q_{g,\min}.
\end{cases}
\]

\(t^{LSQ}_f\) must be a geometry-only face projection already used consistently by both schemes or a generic least-squares face evaluation. It must not be tuned for Mach 3 step. Record `pathological_t_flag=1` when \(q_g<q_{g,\min}\).

### 5.2 Revised \(\beta\)

\[
\beta_0=
\frac{\max(0,\hat e_{LR}\cdot \hat n_f)}{\theta_{\min}},
\qquad
\beta=\mathrm{clip}(\beta_0,0,1).
\]

This prevents negative anti-correction.

### 5.3 Sensors

Pressure shock sensor:

\[
\chi_s
=\mathrm{clip}
\left(
\frac{|p_R-p_L|}
{p_R+p_L+\epsilon_p},
0,1
\right)
\cdot
H(-\nabla\cdot u),
\]

where \(H\) is a smooth compression gate. In practice use \(H_c=\mathrm{clip}(-\nabla\cdot u/(|\nabla\cdot u|+|\omega|+\epsilon_u),0,1)\).

Carbuncle/odd-even sensor:

\[
\chi_c
=\chi_s\,
\mathrm{clip}
\left(
\frac{|\Delta_\perp p|}
{|\Delta_n p|+\epsilon_p},
0,1
\right),
\]

where \(\Delta_n p\) is the face-normal pressure jump and \(\Delta_\perp p\) is a local tangential/neighbor pressure alternation measure. This must be local and geometry-based, not a fixed ROI detector.

Shear/vortex sensor:

\[
\chi_\omega
=
\frac{|\omega|}
{|\omega|+|\nabla\cdot u|+\epsilon_u}.
\]

Smoothness sensor:

\[
\chi_{sm}
=
1-\mathrm{clip}
\left(
\frac{\eta_{LSQ}}
{|\nabla\phi|h+\epsilon_\phi},
0,1
\right),
\]

where \(\eta_{LSQ}\) is a normalized local reconstruction residual.

### 5.4 Revised Face Gradient Correction

Let

\[
\bar\nabla\phi=(1-t^*)\nabla\phi_L+t^*\nabla\phi_R.
\]

Let

\[
\delta_g=
\bar\nabla\phi\cdot\hat e_{LR}
-
\frac{\phi_R-\phi_L}{|d_{LR}|+\epsilon_x}.
\]

Shock/carbuncle-damped correction:

\[
\nabla\phi_{f,corr}
=
\bar\nabla\phi
-
\beta\,D_g\,\delta_g\,\hat e_{LR},
\]

\[
D_g
=
1-\chi_s(1-\chi_\omega)(1+\chi_c)/2.
\]

Thus, the correction is active in smooth/shear regions and damped in strong compressive shocks where anti-diffusive correction can trigger pressure wiggles.

### 5.5 Revised Increment

\[
\Delta\phi_{V_i}
=
t^*(\phi_R-\phi_L)
+\nabla\phi_{f,corr}\cdot(V_i-f_0).
\]

\[
\Delta\phi_f
=
t^*(\phi_R-\phi_L)
+\nabla\phi_{f,corr}\cdot(m_f-f_0).
\]

### 5.6 Revised \(r\)

Use a scale-aware smooth ratio:

\[
\Delta_R=\phi_R-\phi_L,\qquad
\Delta_L=\nabla\phi_L\cdot d_{LR}.
\]

\[
\epsilon_r
=
C_r\epsilon_{mach}
\left(
|\phi_L|+|\phi_R|+|\Delta_L|+|\Delta_R|+1
\right).
\]

\[
r_s
=
\frac{\Delta_R\Delta_L+\epsilon_r^2}
{\Delta_L^2+\epsilon_r^2},
\qquad
r=\max(0,r_s).
\]

This avoids discontinuous sign-preserving fixed \(10^{-8}\) behavior.

### 5.7 Revised \(\alpha\) and Bound Candidate

For each vertex:

\[
B_{V_i}=
\begin{cases}
\phi_{V_i}^{\max}-\phi_L,& \Delta\phi_{V_i}>0,\\
\phi_{V_i}^{\min}-\phi_L,& \Delta\phi_{V_i}<0.
\end{cases}
\]

\[
\psi^{bound}_{V_i}
=
\max
\left(
0,
\min
\left(
1,\frac{B_{V_i}}{\Delta\phi_{V_i}+\epsilon_\Delta\,\mathrm{sign}(\Delta\phi_{V_i})}
\right)
\right).
\]

If

\[
|\Delta\phi_{V_i}|<
C_\Delta\epsilon_{mach}
\left(|\phi_L|+|\phi_R|+\phi_{V_i}^{\max}-\phi_{V_i}^{\min}+1\right),
\]

set \(\psi^{bound}_{V_i}=1\).

### 5.8 Revised \(\psi_{TVD}\)

Use a non-compressive base limiter in shocks and allow mild compression only in non-shock shear/contact regions:

\[
\psi_{TVD}
=
\min
\left[
\psi_{\max},
\phi_{MC}(r)
\right],
\]

where

\[
\phi_{MC}(r)=\max\left(0,\min\left(2r,\frac{1+r}{2},2\right)\right).
\]

Adaptive maximum:

\[
\psi_{\max}
=
1+
(1-\chi_s)(1-\chi_c)\chi_\omega\chi_{sm}.
\]

Thus \(\psi_{\max}\to1\) in shocks/carbuncle-prone regions and \(\psi_{\max}\to2\) only in smooth vortical/contact-like regions.

### 5.9 Final \(\psi_L\)

\[
\psi_{V_i}
=
\min(\psi^{bound}_{V_i},\psi_{TVD}).
\]

\[
\psi_L
=
\min_{f,V_i\in L}\psi_{V_i}.
\]

Optional diagnostic-only values:

\[
\psi_{\text{smoothROI}},\quad
\psi_{\text{shockROI}},\quad
\mathrm{clip\_reason}\in
\{\text{bound},\text{TVD},\text{shock},\text{carbuncle},\text{positivity}\}.
\]

### 5.10 Positivity Correction

For Euler primitive reconstruction, after constructing face primitive state \(W_f=(\rho,u,v,p)\), enforce:

\[
\rho_f> \rho_{floor},\qquad p_f>p_{floor}.
\]

If violated, scale only the increment from cell state:

\[
W_f=W_L+\lambda(W_f^{HO}-W_L),
\]

\[
\lambda=
\min
\left(
1,
\frac{\rho_L-\rho_{floor}}{\rho_L-\rho_f^{HO}+\epsilon_\rho},
\frac{p_L-p_{floor}}{p_L-p_f^{HO}+\epsilon_p}
\right)
\]

using only active ratios where the denominator indicates a violation. Record `positivity_scaled=1`. Floors must be machine/thermodynamic floors, not Mach-3-tuned constants.

### 5.11 Fallback Rule

Fallback is allowed only for mathematically degenerate geometry or positivity, not for a visual ROI:

1. If \(q_g<q_{g,\min}\), use generic LSQ face increment and record `pathological_geometry`.
2. If nonfinite \(r,\alpha,\psi\), set \(\psi=0\) for that face/cell and record `nonfinite_limiter`.
3. If positivity scaling activates, report activation count and magnitude.
4. Never switch only the Mach 3 top wall/outlet ROI to first order in the official comparison unless the identical rule is also applied to MLP-u1 and documented as a boundary treatment study.

## 6. Pseudo-Code

```text
compute_TMLP_u_face_value(
    phi_L, phi_R,
    grad_phi_L, grad_phi_R,
    c_L, c_R, m_f, n_f,
    vertex_list,
    vertex_neighbor_cell_values,
    local_flow_variables,
    parameters
):
    eps_mach = parameters.eps_mach
    theta_min = parameters.theta_min
    qg_min = parameters.qg_min

    diagnostics = {}

    dLR = c_R - c_L
    eLR = dLR / (norm(dLR) + eps_x)
    nhat = n_f / (norm(n_f) + eps_x)
    Dn = dot(dLR, nhat)
    qg = abs(Dn) / (norm(dLR) + eps_x)

    if qg >= qg_min:
        t_star = dot(m_f - c_L, nhat) / Dn
        diagnostics.pathological_t = 0
    else:
        t_star = generic_lsq_face_projection(c_L, c_R, m_f, local_geometry)
        diagnostics.pathological_t = 1

    f0 = c_L + t_star * dLR

    beta0 = max(0, dot(eLR, nhat)) / theta_min
    beta = clip(beta0, 0, 1)

    sensors = compute_local_sensors(local_flow_variables, c_L, c_R, m_f, nhat)
    chi_s = sensors.shock
    chi_c = sensors.carbuncle
    chi_w = sensors.shear_vortex
    chi_sm = sensors.smoothness

    grad_bar = (1 - t_star) * grad_phi_L + t_star * grad_phi_R
    jump_grad = (phi_R - phi_L) / (norm(dLR) + eps_x)
    delta_g = dot(grad_bar, eLR) - jump_grad
    Dg = 1 - chi_s * (1 - chi_w) * (1 + chi_c) / 2
    Dg = clip(Dg, 0, 1)
    grad_corr = grad_bar - beta * Dg * delta_g * eLR

    Delta_R = phi_R - phi_L
    Delta_L = dot(grad_phi_L, dLR)
    eps_r = scale_aware_eps(phi_L, phi_R, Delta_L, Delta_R, eps_mach)
    r_s = (Delta_R * Delta_L + eps_r^2) / (Delta_L^2 + eps_r^2)
    r = max(0, r_s)

    psi_max = 1 + (1 - chi_s) * (1 - chi_c) * chi_w * chi_sm
    psi_max = clip(psi_max, 1, 2)
    psi_tvd = min(psi_max, max(0, min(2*r, 0.5*(1+r), 2)))

    psi_L = 1
    limiting_reason = "inactive"

    for V_i in vertex_list:
        vertex_values = vertex_neighbor_cell_values[V_i]
        phi_v_min = min(vertex_values)
        phi_v_max = max(vertex_values)

        Delta_V = t_star * (phi_R - phi_L) + dot(grad_corr, V_i - f0)
        eps_D = scale_aware_delta_eps(phi_L, phi_R, phi_v_min, phi_v_max)

        if abs(Delta_V) < eps_D:
            psi_bound = 1
        else if Delta_V > 0:
            B = phi_v_max - phi_L
            psi_bound = max(0, min(1, B / (Delta_V + eps_D)))
        else:
            B = phi_v_min - phi_L
            psi_bound = max(0, min(1, B / (Delta_V - eps_D)))

        psi_V = min(psi_bound, psi_tvd)
        if psi_V < psi_L:
            psi_L = psi_V
            limiting_reason = classify_reason(psi_bound, psi_tvd, sensors)

    Delta_f = t_star * (phi_R - phi_L) + dot(grad_corr, m_f - f0)
    phi_f_L = phi_L + psi_L * Delta_f

    if is_euler_primitive(phi_group):
        phi_f_L, lambda_pos, pos_flag = positivity_scale_if_needed(
            phi_L, phi_f_L, local_flow_variables, parameters
        )
        if pos_flag:
            psi_L = psi_L * lambda_pos
            limiting_reason = "positivity"

    diagnostics.psi_L = psi_L
    diagnostics.reason = limiting_reason
    diagnostics.beta = beta
    diagnostics.t_star = t_star
    diagnostics.qg = qg
    diagnostics.r = r
    diagnostics.psi_tvd = psi_tvd
    diagnostics.sensors = sensors

    return phi_f_L, psi_L, diagnostics.flags, sensors
```

## 7. Test and Accept/Reject Logic

Use this comparison table after actual runs:

| Metric | MLP-u1 | TMLP-u-old | TMLP-u-new | ?�단 |
|---|---:|---:|---:|---|
| carbuncle index | ?�인 ?�요 | ?�인 ?�요 | ?�인 ?�요 | new must not exceed MLP-u1 or old |
| shock split band count | ?�인 ?�요 | ?�인 ?�요 | ?�인 ?�요 | lower is better |
| post-shock \(p\) oscillation | ?�인 ?�요 | ?�인 ?�요 | ?�인 ?�요 | reject if increased |
| shock thickness | ?�인 ?�요 | ?�인 ?�요 | ?�인 ?�요 | reject if much thicker without stability gain |
| slip-line sharpness | ?�인 ?�요 | ?�인 ?�요 | ?�인 ?�요 | reject if clearly degraded |
| vortex-pair count | ?�인 ?�요 | ?�인 ?�요 | ?�인 ?�요 | reject if reduced, unless oscillation reduction is decisive |
| \(\rho_{\min}\), \(p_{\min}\) | ?�인 ?�요 | ?�인 ?�요 | ?�인 ?�요 | hard reject if worse bound violation |
| residual/dt stability | ?�인 ?�요 | ?�인 ?�요 | ?�인 ?�요 | reject if dt collapse or residual instability appears |
| CPU cost | ?�인 ?�요 | ?�인 ?�요 | ?�인 ?�요 | reject if excessive without quality gain |
| smooth-region \(\psi<0.9\) ratio | ?�인 ?�요 | ?�인 ?�요 | ?�인 ?�요 | reject if over-clipping grows strongly |

Hard reject rules:

- Reject if carbuncle index increases.
- Reject if pressure/density bound violation increases.
- Reject if shock thickness increases excessively while oscillation is not reduced.
- Reject if slip-line sharpness decreases strongly.
- Reject if vortex preservation decreases, except hold for review when oscillation suppression is very large.
- Reject if CPU cost is excessive relative to MLP-u1 and TMLP-u-old.
- Reject if residual or dt behavior becomes unstable.

Accept rule:

\[
\text{Accept if all hard gates pass and }
S_{\text{new}}>S_{\text{old}}
\text{ under the declared symbolic weights.}
\]

If weights are changed after seeing results, report both the old and new weight interpretation to avoid post-hoc bias.

## 8. Next Iteration Recommendation

| Case | Observation | Modify next | Specific action |
|---|---|---|---|
| A | shock stable but slip line too smeared | shear/vortex relaxation | increase \(\chi_\omega\) protection; reduce shock sensor footprint outside compression; test C13 only if DMP remains intact |
| B | slip line sharp but post-shock oscillation occurs | shock and pressure limiting | reduce \(\psi_{\max}\to1\) in \(\chi_s\) regions; increase C10 damping; add characteristic limiting for pressure/normal velocity |
| C | carbuncle-like instability occurs | carbuncle sensor and transverse damping | strengthen \(\chi_c\); damp transverse correction near aligned shocks; compare robust flux interaction without changing MLP-u1 unfairly |
| D | vortex weakly captured | smooth/shear preservation | lower smooth-region clipping; inspect \(\psi\) histogram; protect tangential velocity/contact variables in \(\chi_\omega\)-high, \(\chi_s\)-low cells |
| E | residual convergence unstable | positivity and limiter smoothness | smooth sensor transitions; remove discontinuous switches; reduce \(\psi_{\max}\); check nonfinite \(r,\alpha,\beta,t^*\) diagnostics |
| F | almost no difference from MLP-u1 | ablation and metric sensitivity | run A1-A5; inspect whether \(t^*\), \(\beta\), gradient correction activate; if not, use harder skewed/non-orthogonal benchmark |

## 9. Overfitting Risk for Mach 3 Step

Mach 3 step `200x80` (with later checks on `480x160`) can overfit the method in several ways:

- The step geometry and top-wall/outlet interactions can reward boundary-specific fixes.
- A coarse grid can make dissipative methods look robust while hiding true resolution loss.
- Carbuncle sensors may accidentally detect the benchmark geometry rather than a general shock-alignment failure.
- Vortex preservation metrics at \(120\times40\) are qualitative and must be checked on \(240\times80\) or finer.
- A flux-specific success with `ausm_slau2_shock` does not prove limiter-level superiority.

SCI-defensible path:

1. Develop v2 on \(120\times40\).
2. Freeze all parameters and formulas.
3. Re-run MLP-u1, TMLP-u-old, and TMLP-u-v2 under identical settings.
4. Validate on \(240\times80\) and one non-step shock benchmark.
5. Report failures and trade-offs honestly.

## 10. Minimum Implementation Diagnostics

Every TMLP-u-v2 run should write these fields:

| Diagnostic | Purpose |
|---|---|
| \(\min\rho,\min p\) | positivity hard gate |
| `dt_collapse_flag` | residual/time-step instability |
| \(t^*\) min/max/p99 and pathological count | geometry robustness |
| \(\beta\) min/max and negative pre-clip count | orientation robustness |
| \(r\) p95/p99/max and nonfinite count | TVD ratio stability |
| \(\psi_L\) histogram by shock and smooth ROI | clipping diagnosis |
| shock/carbuncle/shear/smooth sensor histograms | sensor footprint |
| positivity scaling count and max scaling | hidden clipping detection |
| CPU reconstruction/flux/residual timing | cost attribution |

No final superiority claim should be made without these diagnostics.



