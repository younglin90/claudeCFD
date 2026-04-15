"""
solver/He2024/explicit_mmacm_ex.py

Standalone explicit 5-equation solver with MMACM-Ex interface sharpening.

Governing equations (2-species, 1D, Allaire-Massoni form):
    dQ/dt + dF/dx = S      (conservative part)
    da1/dt + d(a1*u)/dx - (a1 + D1) * du/dx = 0  (volume fraction, non-conservative)

Conservative variables per cell: Q = {a1r1, a2r2, ru, rE}
Volume fraction separately:       a1

Time integration: SSP-RK3 (Shu-Osher 1988)
Spatial reconstruction: TVD van Leer (primitive variables)
Interface flux: HLLC (Toro 1994)
Alpha reconstruction: MMACM-Ex (Zhao et al. 2025, Phys. Fluids 37:076157)
    - H_k characteristic function (Eq. 32)
    - Pure downwind alpha reconstruction (Eq. 30)
    - Conservation consistency corrections (Eq. 27)
    - F_alpha = F_{a1r1} / rho1_upwind (Eq. 26)
    - HLLC face velocity u* (Eq. 25) for alpha source term

Interface cell slope freeze (Eq. 19):
    In cells where eps < a1 < 1-eps AND monotone, set rho1/rho2 slope = 0.
    This prevents NaN from huge density gradients across the interface.

EOS: Stiffened Gas (SG) for both phases (b=0, eta=0 NASG special case):
    p = (gamma - 1) * rho * e - gamma * Pinf
    e = (p + gamma*Pinf) / ((gamma-1)*rho)
    c^2 = gamma * (p + Pinf) / rho

Phase 2-1 setup:
    SG Water (gamma=4.1, Pinf=4.4e8) left, Ideal Air (gamma=1.4) right,
    domain [0,2]m, N=200, CFL=0.4, t_end=8e-4 s (full spec), or 2.4e-4 s (paper).

Ref: Zhao et al. 2025, Phys. Fluids 37:076157
     Toro 1994, Shock Waves 4:25-34
     Johnsen & Colonius 2006, JCP 219:715-759
     CLAUDE.md § He2024 5-Equation
"""

import numpy as np

# ---------------------------------------------------------------------------
# EOS helpers (Stiffened Gas / Ideal Gas)
# ---------------------------------------------------------------------------

_EPS = 1e-30   # small positive for division safety


def _sg_pressure(rho, e, gamma, pinf):
    """p = (gamma-1)*rho*e - gamma*Pinf"""
    return (gamma - 1.0) * rho * e - gamma * pinf


def _sg_sound_speed_sq(p, rho, gamma, pinf):
    """c^2 = gamma*(p+Pinf)/rho"""
    return gamma * (p + pinf) / np.maximum(rho, _EPS)


def _sg_internal_energy(p, rho, gamma, pinf):
    """e = (p + gamma*Pinf) / ((gamma-1)*rho)"""
    return (p + gamma * pinf) / np.maximum((gamma - 1.0) * rho, _EPS)


def _sg_temperature(p, gamma, pinf, kv):
    """T = (p + Pinf) / ((gamma-1)*kv*rho)  — used for initial conditions."""
    # For SG: p = (gamma-1)*rho*kv*T - Pinf  =>  rho = (p+Pinf)/((gamma-1)*kv*T)
    pass


def _sg_density_from_pT(p, T, gamma, pinf, kv):
    """rho = (p+Pinf) / ((gamma-1)*kv*T)"""
    return (p + pinf) / np.maximum((gamma - 1.0) * kv * T, _EPS)


def _sg_specific_internal_energy_from_pT(p, T, gamma, pinf, kv):
    """e = kv*T*(p + gamma*Pinf)/(p+Pinf)"""
    return kv * T * (p + gamma * pinf) / np.maximum(p + pinf, _EPS)


# ---------------------------------------------------------------------------
# Primitive → conservative and back
# ---------------------------------------------------------------------------

def prim_to_cons(rho1, rho2, u, p, a1, ph1, ph2):
    """Convert primitive (rho1, rho2, u, p, a1) → conservative (a1r1, a2r2, ru, rE).

    rho1, rho2 are PHASE densities (not partial densities).
    a1r1 = alpha1 * rho1 (partial density of species 1).
    """
    g1, pinf1 = ph1['gamma'], ph1['pinf']
    g2, pinf2 = ph2['gamma'], ph2['pinf']
    a2 = 1.0 - a1

    a1r1 = a1 * rho1
    a2r2 = a2 * rho2
    rho = a1r1 + a2r2
    ru = rho * u

    e1 = _sg_internal_energy(p, rho1, g1, pinf1)
    e2 = _sg_internal_energy(p, rho2, g2, pinf2)
    rho_e = a1 * rho1 * e1 + a2 * rho2 * e2
    rE = rho_e + 0.5 * rho * u * u

    return a1r1, a2r2, ru, rE


def cons_to_prim(a1r1, a2r2, ru, rE, a1, ph1, ph2):
    """Convert conservative → primitive. Standard 5-eq linear pressure.

    p = (ρe - Σ αₖγₖP∞ₖ/(γₖ-1)) / (Σ αₖ/(γₖ-1))
    Phase densities from conservative variables: ρₖ = αₖρₖ / αₖ
    T from majority phase (diagnostic).
    Ref: Zhao et al. 2025 (MMACM-Ex), standard 5-equation model.
    """
    g1, pinf1 = ph1['gamma'], ph1['pinf']
    g2, pinf2 = ph2['gamma'], ph2['pinf']
    kv1, kv2 = ph1['kv'], ph2['kv']
    gm1, gm2 = g1 - 1.0, g2 - 1.0

    a2 = 1.0 - a1
    rho = a1r1 + a2r2
    u = ru / np.maximum(rho, _EPS)
    rho_e = rE - 0.5 * ru * u

    # Standard 5-eq pressure (linear)
    Gamma_inv = a1 / gm1 + a2 / gm2
    Pi = a1 * g1 * pinf1 / gm1 + a2 * g2 * pinf2 / gm2
    p = (rho_e - Pi) / np.maximum(Gamma_inv, _EPS)
    p = np.maximum(p, 1.0)

    # Phase densities from conservative variables
    _af = 1e-8
    rho1 = a1r1 / np.maximum(a1, _af)
    rho2 = a2r2 / np.maximum(a2, _af)

    # Minority phase protection: use EOS from majority phase T
    T_from_1 = (p + pinf1) / np.maximum(gm1 * kv1 * rho1, _EPS)
    T_from_2 = (p + pinf2) / np.maximum(gm2 * kv2 * rho2, _EPS)
    T = np.where(a1 >= 0.5, T_from_1, T_from_2)
    T = np.maximum(T, 1.0)

    rho1_eos = (p + pinf1) / np.maximum(gm1 * kv1 * T, _EPS)
    rho2_eos = (p + pinf2) / np.maximum(gm2 * kv2 * T, _EPS)
    rho1 = np.where(a1 < _af, rho1_eos, rho1)
    rho2 = np.where(a2 < _af, rho2_eos, rho2)
    rho1 = np.maximum(rho1, _EPS)
    rho2 = np.maximum(rho2, _EPS)

    # Sound speeds
    c1_sq = np.maximum(_sg_sound_speed_sq(p, rho1, g1, pinf1), _EPS)
    c2_sq = np.maximum(_sg_sound_speed_sq(p, rho2, g2, pinf2), _EPS)

    # T-eq effective sound speed (still use c_eff for HLLC consistency with DC)
    c_mix = _ceff_temp_eq(a1, rho1, rho2, p, T, ph1, ph2)

    return p, u, T, rho1, rho2, np.sqrt(c1_sq), np.sqrt(c2_sq), c_mix


# ---------------------------------------------------------------------------
# Ghost cell extension (transmissive / periodic)
# ---------------------------------------------------------------------------

def _ghost(arr, bc_l='transmissive', bc_r='transmissive', ng=1):
    """Extend array (N,) with ng ghost layers."""
    if bc_l == 'periodic':
        left = arr[-ng:]
    else:  # transmissive (zero-gradient)
        left = np.repeat(arr[:1], ng)

    if bc_r == 'periodic':
        right = arr[:ng]
    else:
        right = np.repeat(arr[-1:], ng)

    return np.concatenate([left, arr, right])


def _ghost2(arr, bc_l='transmissive', bc_r='transmissive'):
    """Extend with ng=2 ghost layers."""
    if bc_l == 'periodic':
        left = arr[-2:]
    else:
        left = np.array([arr[0], arr[0]])

    if bc_r == 'periodic':
        right = arr[:2]
    else:
        right = np.array([arr[-1], arr[-1]])

    return np.concatenate([left, arr, right])


# ---------------------------------------------------------------------------
# TVD van Leer limiter reconstruction
# ---------------------------------------------------------------------------

def _van_leer(r):
    """Van Leer limiter: phi = (r + |r|) / (1 + |r|), smooth version."""
    abs_r = np.abs(r)
    return (r + abs_r) / np.maximum(1.0 + abs_r, _EPS)


def _tvd_reconstruct(q, bc_l='transmissive', bc_r='transmissive'):
    """TVD reconstruction with van Leer limiter. Returns (qL, qR) at N+1 faces.

    qL[j] = left state at face j  (from cell j-1)
    qR[j] = right state at face j (from cell j)
    Face index j in [0, N], i.e. N+1 faces.
    """
    N = len(q)
    q_ext = _ghost(q, bc_l, bc_r, ng=2)  # (N+4,) with 2 ghosts each side
    # Indices: q_ext[2:N+2] = q[0:N]
    #          q_ext[i] = q[i-2] (with ghost handling)

    # Differences (use ng=2 extended array)
    dL = q_ext[2:N+2] - q_ext[1:N+1]   # q_i - q_{i-1}  (N,)
    dR = q_ext[3:N+3] - q_ext[2:N+2]   # q_{i+1} - q_i  (N,)

    # Slope ratio
    r = np.where(np.abs(dR) > _EPS, dL / (dR + np.sign(dR + _EPS) * _EPS), 0.0)
    phi = _van_leer(r)
    sigma = 0.5 * phi * dR   # (N,) limited slope

    # Cell-center reconstructed edge values
    qL_cell = q + sigma     # right face of cell i (contributes to face i+1)
    qR_cell = q - sigma     # left face of cell i  (contributes to face i)

    # Assemble face arrays: face j has L from cell j-1 and R from cell j
    # Left state at face j = qL_cell[j-1] (cell j-1 right edge)
    # Right state at face j = qR_cell[j]  (cell j left edge)
    # Face 0: left boundary (use ghost)
    # Face N: right boundary (use ghost)

    # Ghost for qL_cell and qR_cell
    if bc_l == 'periodic':
        qL_ghost_l = qL_cell[-1:]
        qR_ghost_l = qR_cell[-1:]
    else:
        qL_ghost_l = qL_cell[0:1]
        qR_ghost_l = qR_cell[0:1]

    if bc_r == 'periodic':
        qL_ghost_r = qL_cell[0:1]
        qR_ghost_r = qR_cell[0:1]
    else:
        qL_ghost_r = qL_cell[-1:]
        qR_ghost_r = qR_cell[-1:]

    # Face L states: from cells [-1, 0, 1, ..., N-1]
    qL_faces = np.concatenate([qL_ghost_l, qL_cell])   # (N+1,)
    # Face R states: from cells [0, 1, ..., N-1, N]
    qR_faces = np.concatenate([qR_cell, qR_ghost_r])    # (N+1,)

    return qL_faces, qR_faces


# ---------------------------------------------------------------------------
# THINC-BVD reconstruction for volume fraction (Deng et al. 2018 / Shyue & Xiao 2014)
# ---------------------------------------------------------------------------

def _thinc_bvd_reconstruct(q, bc_l='transmissive', bc_r='transmissive',
                            beta=2.0, eps_thinc=1e-4):
    """THINC-BVD reconstruction for volume fraction α₁.

    BVD selection (Deng 2018): pick THINC only when TBV_THINC < TBV_TVD
    AND monotone AND interface cell. Otherwise fall back to TVD.

    Computes BOTH TVD and THINC face values, then selects per-cell
    using the BVD criterion (minimize total boundary variation).

    THINC: tangent-of-hyperbola step-function reconstruction
           q̂(ξ) = q_min + Δq/2 · (1 + tanh(β(ξ - ξ₀)))
           ξ₀ from cell-average constraint.

    BVD: for cell i, pick TVD or THINC to minimize
         |q_{i-1/2,R} - q_{i-1/2,L}| + |q_{i+1/2,R} - q_{i+1/2,L}|

    Returns (qL, qR) at N+1 faces (same convention as _tvd_reconstruct).
    """
    N = len(q)

    # --- TVD reconstruction (baseline) ---
    qL_tvd, qR_tvd = _tvd_reconstruct(q, bc_l, bc_r)

    # --- THINC reconstruction per cell ---
    q_ext = _ghost(q, bc_l, bc_r, ng=2)  # (N+4,): q_ext[2:N+2] = q[0:N]

    # Neighbor values for each cell i (0-based)
    qm1 = q_ext[1:N+1]   # q_{i-1}
    q0  = q_ext[2:N+2]   # q_i  (= q itself)
    qp1 = q_ext[3:N+3]   # q_{i+1}

    q_min = np.minimum(qm1, qp1)
    q_max = np.maximum(qm1, qp1)
    dq = q_max - q_min

    # Normalized cell average
    d = np.where(dq > eps_thinc, (q0 - q_min) / np.maximum(dq, _EPS), 0.5)
    d = np.clip(d, eps_thinc, 1.0 - eps_thinc)

    # Interface direction
    sigma = np.where(qp1 >= qm1, 1.0, -1.0)

    # THINC: solve for interface position ξ₀ from cell average constraint
    # For tanh profile in [0,1]: ξ₀ ≈ 1 - d (first order)
    # Exact: ln(cosh(β(1-ξ₀))/cosh(βξ₀))/(2β) = d - 0.5
    # Using the direct formula:
    #   B = exp(σβ(2d-1))
    #   face_R = q_min + dq * B/(B + exp(σβ))       [right face, ξ=1]
    #   face_L = q_min + dq * B/(B + exp(-σβ))  ... no

    # Direct THINC face values (Deng et al. 2018 formulation):
    # exp_term = exp(2σβ(d - 0.5))  = exp(σβ(2d-1))
    sb = sigma * beta
    exp_sb = np.exp(sb)            # exp(σβ)
    exp_2sd = np.exp(sb * (2.0 * d - 1.0))  # exp(σβ(2d-1))

    # Face values at ξ=0 (left face) and ξ=1 (right face) of cell i:
    # q_L = q_min + dq/2 * (1 + σ * tanh(β(-ξ₀)))
    # q_R = q_min + dq/2 * (1 + σ * tanh(β(1-ξ₀)))
    # Using exp form: tanh(x) = (exp(2x)-1)/(exp(2x)+1)
    # After algebra with cell-avg constraint:
    #   q_R_cell = q_min + dq * (exp_2sd * exp_sb - 1) / (exp_2sd * exp_sb + 1)  ... hmm

    # Simplest stable formulation (Shyue & Xiao 2014):
    # Right face of cell i: ξ = 1
    thinc_R_cell = q_min + dq * 0.5 * (1.0 + sigma * (exp_2sd * exp_sb - 1.0)
                                         / (exp_2sd * exp_sb + 1.0))
    # Left face of cell i: ξ = 0
    inv_exp_sb = 1.0 / np.maximum(exp_sb, _EPS)
    thinc_L_cell = q_min + dq * 0.5 * (1.0 + sigma * (exp_2sd * inv_exp_sb - 1.0)
                                         / (exp_2sd * inv_exp_sb + 1.0))

    # Clip to [0, 1] (global α bounds) — more compressive than [q_min, q_max]
    thinc_R_cell = np.clip(thinc_R_cell, 0.0, 1.0)
    thinc_L_cell = np.clip(thinc_L_cell, 0.0, 1.0)

    # Interface detection + monotonicity (Deng 2018)
    is_intf = (q0 > eps_thinc) & (q0 < 1.0 - eps_thinc)
    is_mono = (qp1 - q0) * (q0 - qm1) > 0.0
    use_thinc_candidate = is_mono & is_intf

    # Assemble THINC face arrays (same convention as TVD)
    # Face j: L from cell j-1, R from cell j
    if bc_l == 'periodic':
        thinc_L_ghost_l = thinc_R_cell[-1:]  # right face of last cell
        thinc_R_ghost_l = thinc_L_cell[-1:]
    else:
        thinc_L_ghost_l = thinc_R_cell[0:1]
        thinc_R_ghost_l = thinc_L_cell[0:1]

    if bc_r == 'periodic':
        thinc_L_ghost_r = thinc_R_cell[0:1]
        thinc_R_ghost_r = thinc_L_cell[0:1]
    else:
        thinc_L_ghost_r = thinc_R_cell[-1:]
        thinc_R_ghost_r = thinc_L_cell[-1:]

    # THINC face L states: right face of cells [-1, 0, ..., N-1]
    qL_thinc = np.concatenate([thinc_L_ghost_l, thinc_R_cell])  # (N+1,)
    # THINC face R states: left face of cells [0, 1, ..., N]
    qR_thinc = np.concatenate([thinc_L_cell, thinc_R_ghost_r])  # (N+1,)

    # --- BVD selection per cell (Deng et al. 2018, Eq. 26-27) ---
    # For cell i: compare TBV when cell i uses THINC vs TVD.
    # KEY: neighbors (cells i-1, i+1) always use TVD as baseline.
    # Only cell i switches between THINC and TVD.
    #
    # Face i: L = qL_tvd[i] (from cell i-1, TVD fixed),
    #         R = cell i's left face (THINC or TVD candidate)
    # Face i+1: L = cell i's right face (THINC or TVD candidate),
    #           R = qR_tvd[i+1] (from cell i+1, TVD fixed)

    # TBV when cell i uses TVD (all TVD)
    tbv_tvd = (np.abs(qL_tvd[0:N] - qR_tvd[0:N])
               + np.abs(qL_tvd[1:N+1] - qR_tvd[1:N+1]))

    # TBV when cell i uses THINC (neighbors stay TVD)
    # thinc_L_cell[i] = cell i's LEFT face from THINC = right state at face i
    # thinc_R_cell[i] = cell i's RIGHT face from THINC = left state at face i+1
    tbv_thinc = (np.abs(qL_tvd[0:N] - thinc_L_cell)
                 + np.abs(thinc_R_cell - qR_tvd[1:N+1]))

    # BVD: pick THINC only when it gives smaller boundary variation
    use_thinc = use_thinc_candidate & (tbv_thinc < tbv_tvd)

    # Build final face values by replacing cell-by-cell
    # Cell i contributes: qL[i+1] (left state at face i+1) and qR[i] (right state at face i)
    qR_cell_final = np.where(use_thinc, thinc_L_cell, q - 0.5 * _van_leer(
        np.where(np.abs(q_ext[3:N+3] - q_ext[2:N+2]) > _EPS,
                 (q_ext[2:N+2] - q_ext[1:N+1]) / (q_ext[3:N+3] - q_ext[2:N+2] + np.sign(q_ext[3:N+3] - q_ext[2:N+2] + _EPS)*_EPS),
                 0.0)) * (q_ext[3:N+3] - q_ext[2:N+2]))
    qL_cell_final = np.where(use_thinc, thinc_R_cell, q + 0.5 * _van_leer(
        np.where(np.abs(q_ext[3:N+3] - q_ext[2:N+2]) > _EPS,
                 (q_ext[2:N+2] - q_ext[1:N+1]) / (q_ext[3:N+3] - q_ext[2:N+2] + np.sign(q_ext[3:N+3] - q_ext[2:N+2] + _EPS)*_EPS),
                 0.0)) * (q_ext[3:N+3] - q_ext[2:N+2]))

    # Simpler: just blend the already-computed TVD and THINC face arrays
    # For cell i: replace qL[i+1] and qR[i]
    # qL[i+1] comes from cell i's right face → qL_cell_final[i]
    # qR[i] comes from cell i's left face → qR_cell_final[i]

    # Start from TVD, then overwrite where THINC is selected
    qL_final = qL_tvd.copy()
    qR_final = qR_tvd.copy()

    # Cell i → qL_final[i+1] (left state at face i+1, from cell i)
    # Cell i → qR_final[i]   (right state at face i, from cell i)
    qL_final[1:N+1] = np.where(use_thinc, thinc_R_cell, qL_tvd[1:N+1])
    qR_final[0:N]   = np.where(use_thinc, thinc_L_cell, qR_tvd[0:N])

    return qL_final, qR_final


def _cicsam_face(q, u_face, dt, dx, bc_l='transmissive', bc_r='transmissive'):
    """CICSAM Hyper-C face reconstruction for α advection.

    Ubbink & Issa (1999) CICSAM — 1D specialization.
    In 1D the interface normal is always parallel to the face normal,
    so the angle-blending factor γ=1 and the scheme reduces to pure Hyper-C.

    Hyper-C NVD formula (Leonard 1991):
      ñ_D = (α_D - α_U) / (α_A - α_U)   [normalized donor value]
      ñ_f = min(ñ_D / Co_f, 1)           [normalized face value]
      α_f = α_U + ñ_f (α_A - α_U)        [de-normalized]

    where D = donor (upwind), A = acceptor (downwind), U = upwind-of-donor.
    Outside NVD range [0, 1]: fall back to 1st-order upwind (α_f = α_D).

    Parameters
    ----------
    q       : cell-centered α (N,)
    u_face  : face velocities (N+1,)
    dt      : current sub-step Δt (scalar) — used for Co = |u|Δt/Δx
    dx      : uniform cell size
    bc_l/r  : boundary condition ('periodic' or 'transmissive')

    Returns
    -------
    alpha_face : (N+1,), clipped to [0, 1]
    """
    N = len(q)
    _eps = 1e-12

    # ng=2 ghost cells: U is 2 cells upstream of face
    q_ext = _ghost2(q, bc_l, bc_r)   # (N+4,): q_ext[0:2]=left gh, q_ext[2:N+2]=q, q_ext[N+2:N+4]=right gh

    Co = np.maximum(np.abs(u_face) * dt / dx, _eps)   # (N+1,)

    # For face f in 0..N:
    #   u > 0: D = cell(f-1) = q_ext[f+1]
    #          A = cell(f)   = q_ext[f+2]
    #          U = cell(f-2) = q_ext[f]
    #   u < 0: D = cell(f)   = q_ext[f+2]
    #          A = cell(f-1) = q_ext[f+1]
    #          U = cell(f+1) = q_ext[f+3]
    aD_pos = q_ext[1:N+2];  aA_pos = q_ext[2:N+3];  aU_pos = q_ext[0:N+1]
    aD_neg = q_ext[2:N+3];  aA_neg = q_ext[1:N+2];  aU_neg = q_ext[3:N+4]

    def _hc(aD, aA, aU, co):
        """Hyper-C face value for one velocity sign."""
        dAU = aA - aU
        # Normalized donor: ñ_D in (-∞, +∞)
        nd = np.where(np.abs(dAU) > _eps, (aD - aU) / dAU, 0.5)
        # Only apply Hyper-C where ñ_D ∈ (0, 1)  [interface region]
        in_range = (nd > 0.0) & (nd < 1.0)
        nf = np.minimum(nd / co, 1.0)              # Hyper-C (capped at 1)
        af_hc = aU + nf * dAU                      # de-normalize
        return np.where(in_range, af_hc, aD)       # 1st-order upwind outside NVD

    af_pos = _hc(aD_pos, aA_pos, aU_pos, Co)
    af_neg = _hc(aD_neg, aA_neg, aU_neg, Co)
    alpha_face = np.where(u_face >= 0.0, af_pos, af_neg)
    return np.clip(alpha_face, 0.0, 1.0)


def _nvd_face(q, u_face, dt, dx, bc_l='transmissive', bc_r='transmissive',
              cds='hyper_c'):
    """Generic NVD face reconstruction with selectable CDS.

    All NVD schemes share the same donor-acceptor stencil (U, D, A).
    They differ in the CDS (compressive) formula.
    In 1D, blending factor γ=1 → pure CDS (no HR blending).

    cds options:
      'hyper_c'  — CICSAM (Ubbink 1999): min(ñ_D/Co, 1)
      'superbee' — STACS (Darwish 2006): piecewise TVD SUPERBEE
      'mstacs'   — MSTACS (Anghan 2021): Hyper-C(Co≤1/3) or min(3ñ_D,1)(Co>1/3)
      'saish'    — SAISH: min(2ñ_D, 1) (bounded downwind, most compressive)
    """
    N = len(q)
    _eps = 1e-12
    q_ext = _ghost2(q, bc_l, bc_r)
    Co = np.maximum(np.abs(u_face) * dt / dx, _eps)

    aD_pos = q_ext[1:N+2]; aA_pos = q_ext[2:N+3]; aU_pos = q_ext[0:N+1]
    aD_neg = q_ext[2:N+3]; aA_neg = q_ext[1:N+2]; aU_neg = q_ext[3:N+4]

    def _cds_face(aD, aA, aU, co):
        dAU = aA - aU
        nd = np.where(np.abs(dAU) > _eps, (aD - aU) / dAU, 0.5)
        in_range = (nd > 0.0) & (nd < 1.0)

        if cds == 'superbee':
            # SUPERBEE (Roe 1985): piecewise in NVD
            nf = np.where(nd < 1./3, 2.0*nd,
                 np.where(nd < 0.5, 0.5 + 0.5*nd,
                 np.where(nd < 2./3, 1.5*nd,
                 1.0)))
        elif cds == 'mstacs':
            # MSTACS (Anghan 2021): Hyper-C at Co≤1/3, 3×downwind at Co>1/3
            nf_hc = np.minimum(nd / co, 1.0)
            nf_3x = np.minimum(3.0 * nd, 1.0)
            nf = np.where(co <= 1./3, nf_hc, nf_3x)
        elif cds == 'saish':
            # SAISH: bounded downwind min(2ñ_D, 1) — most compressive
            nf = np.minimum(2.0 * nd, 1.0)
        else:  # hyper_c (default)
            nf = np.minimum(nd / co, 1.0)

        af = aU + nf * dAU
        return np.where(in_range, af, aD)

    af_pos = _cds_face(aD_pos, aA_pos, aU_pos, Co)
    af_neg = _cds_face(aD_neg, aA_neg, aU_neg, Co)
    return np.clip(np.where(u_face >= 0.0, af_pos, af_neg), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Interface cell detection (Eq. 19)
# ---------------------------------------------------------------------------

def _interface_mask(a1, eps_intf=1e-4):
    """Boolean mask: True in interface cells.

    Interface cell: eps < a1 < 1-eps  AND  (a_{i+1}-a_i)*(a_i-a_{i-1}) > 0 (monotone).
    """
    N = len(a1)
    a_ext = _ghost(a1, 'transmissive', 'transmissive', ng=1)
    dL = a_ext[1:N+1] - a_ext[0:N]
    dR = a_ext[2:N+2] - a_ext[1:N+1]

    in_range = (a1 > eps_intf) & (a1 < 1.0 - eps_intf)
    monotone = (dL * dR) > 0.0
    return in_range & monotone


# ---------------------------------------------------------------------------
# HLLC flux (Toro 1994)
# ---------------------------------------------------------------------------

def _hllc_flux(a1r1_L, a2r2_L, ru_L, rE_L, p_L, c_L,
               a1r1_R, a2r2_R, ru_R, rE_R, p_R, c_R):
    """HLLC numerical flux for 4-variable conservative system.

    Returns: (F_a1r1, F_a2r2, F_ru, F_rE, u_star)
    u_star = S* (contact wave speed) — used for alpha source term (Eq. 25).

    Sign-aware epsilon in (SL - uL)/(SL - S*) to avoid sign flip issues.
    """
    rho_L = a1r1_L + a2r2_L
    rho_R = a1r1_R + a2r2_R
    u_L = ru_L / np.maximum(rho_L, _EPS)
    u_R = ru_R / np.maximum(rho_R, _EPS)
    E_L = rE_L / np.maximum(rho_L, _EPS)
    E_R = rE_R / np.maximum(rho_R, _EPS)
    Y1_L = a1r1_L / np.maximum(rho_L, _EPS)
    Y1_R = a1r1_R / np.maximum(rho_R, _EPS)
    Y2_L = a2r2_L / np.maximum(rho_L, _EPS)
    Y2_R = a2r2_R / np.maximum(rho_R, _EPS)

    # Wave speed estimates (Davis, Eq. 22-23)
    S_L = np.minimum(u_L - c_L, u_R - c_R)
    S_R = np.maximum(u_L + c_L, u_R + c_R)
    s_minus = np.minimum(0.0, S_L)   # s^- = min(0, S^L)
    s_plus  = np.maximum(0.0, S_R)   # s^+ = max(0, S^R)

    # Contact wave speed S* (Toro 1994, Eq. 10.37)
    num_Ss = (p_R - p_L
              + rho_L * u_L * (S_L - u_L)
              - rho_R * u_R * (S_R - u_R))
    den_Ss = rho_L * (S_L - u_L) - rho_R * (S_R - u_R)
    # Avoid division by zero with sign-aware epsilon
    den_Ss_safe = np.where(np.abs(den_Ss) > _EPS, den_Ss,
                           np.sign(den_Ss + _EPS) * _EPS)
    S_star = num_Ss / den_Ss_safe

    # Physical fluxes
    F_a1r1_L = a1r1_L * u_L
    F_a2r2_L = a2r2_L * u_L
    F_ru_L   = ru_L * u_L + p_L
    F_rE_L   = (rE_L + p_L) * u_L

    F_a1r1_R = a1r1_R * u_R
    F_a2r2_R = a2r2_R * u_R
    F_ru_R   = ru_R * u_R + p_R
    F_rE_R   = (rE_R + p_R) * u_R

    # HLLC intermediate state coefficient: rho_K*(S_K-u_K)/(S_K-S*)
    def _coeff_star(rho_K, u_K, S_K):
        # sign-aware epsilon in denominator (task spec requirement)
        denom = S_K - S_star
        denom_safe = np.where(np.abs(denom) > _EPS, denom,
                              np.sign(denom + _EPS) * _EPS)
        return rho_K * (S_K - u_K) / denom_safe

    cL = _coeff_star(rho_L, u_L, S_L)
    cR = _coeff_star(rho_R, u_R, S_R)

    # Star total energy: E* = E + (S*-u)*(S* + p/(rho*(S-u)))
    def _estar(E_K, u_K, p_K, rho_K, S_K):
        denom = rho_K * (S_K - u_K)
        denom_safe = np.where(np.abs(denom) > _EPS, denom,
                              np.sign(denom + _EPS) * _EPS)
        return E_K + (S_star - u_K) * (S_star + p_K / denom_safe)

    EstarL = _estar(E_L, u_L, p_L, rho_L, S_L)
    EstarR = _estar(E_R, u_R, p_R, rho_R, S_R)

    # HLLC flux in left/right star regions
    def _hllc_K(FK, QK, star_coeff, Y1K, Y2K, EstarK, S_K):
        Qstar_a1r1 = star_coeff * Y1K
        Qstar_a2r2 = star_coeff * Y2K
        Qstar_ru   = star_coeff * S_star
        Qstar_rE   = star_coeff * EstarK

        Q_a1r1, Q_a2r2, Q_ru, Q_rE = QK
        F_a1r1K, F_a2r2K, F_ruK, F_rEK = FK

        return (F_a1r1K + S_K * (Qstar_a1r1 - Q_a1r1),
                F_a2r2K + S_K * (Qstar_a2r2 - Q_a2r2),
                F_ruK   + S_K * (Qstar_ru   - Q_ru),
                F_rEK   + S_K * (Qstar_rE   - Q_rE))

    FL = (F_a1r1_L, F_a2r2_L, F_ru_L, F_rE_L)
    FR = (F_a1r1_R, F_a2r2_R, F_ru_R, F_rE_R)
    QL = (a1r1_L, a2r2_L, ru_L, rE_L)
    QR = (a1r1_R, a2r2_R, ru_R, rE_R)

    hllcL = _hllc_K(FL, QL, cL, Y1_L, Y2_L, EstarL, S_L)
    hllcR = _hllc_K(FR, QR, cR, Y1_R, Y2_R, EstarR, S_R)

    # Select region
    region = np.where(S_L >= 0.0, 0,
              np.where(S_star >= 0.0, 1,
              np.where(S_R > 0.0, 2, 3)))

    def _select(fL_phys, hllc_L, hllc_R, fR_phys):
        return np.where(region == 0, fL_phys,
               np.where(region == 1, hllc_L,
               np.where(region == 2, hllc_R, fR_phys)))

    F1 = _select(F_a1r1_L, hllcL[0], hllcR[0], F_a1r1_R)
    F2 = _select(F_a2r2_L, hllcL[1], hllcR[1], F_a2r2_R)
    F3 = _select(F_ru_L,   hllcL[2], hllcR[2], F_ru_R)
    F4 = _select(F_rE_L,   hllcL[3], hllcR[3], F_rE_R)

    # HLLC face velocity ū (Eq. 25, Zhao 2025 / Johnsen & Colonius 2006)
    # This is the velocity consistent with the HLLC flux, NOT simply S*.
    # For s* > 0: ū = u^L + s^- · ((S^L - u^L)/(S^L - S*) - 1)
    # For s* ≤ 0: ū = u^R + s^+ · ((S^R - u^R)/(S^R - S*) - 1)
    denom_L = S_L - S_star
    denom_L_safe = np.where(np.abs(denom_L) > _EPS, denom_L,
                            np.sign(denom_L + _EPS) * _EPS)
    denom_R = S_R - S_star
    denom_R_safe = np.where(np.abs(denom_R) > _EPS, denom_R,
                            np.sign(denom_R + _EPS) * _EPS)

    u_hllc_L = u_L + s_minus * ((S_L - u_L) / denom_L_safe - 1.0)
    u_hllc_R = u_R + s_plus  * ((S_R - u_R) / denom_R_safe - 1.0)

    u_face = np.where(S_star >= 0.0,
                      0.5 * (1.0 + np.sign(S_star + _EPS)) * u_hllc_L,
                      0.5 * (1.0 - np.sign(-S_star + _EPS)) * u_hllc_R)
    # Simplified: select L branch when S* > 0, R branch when S* < 0
    u_face = np.where(S_star >= 0.0, u_hllc_L, u_hllc_R)

    return F1, F2, F3, F4, u_face, S_star


# ---------------------------------------------------------------------------
# Temperature-equilibrium sound speed c_eff (He & Tan 2024 Eq. A.17)
# ---------------------------------------------------------------------------

def _ceff_temp_eq(a1, rho1, rho2, p, T, ph1, ph2):
    """T-equilibrium mixture sound speed c_eff (He & Tan 2024 Eq. A.17/A.18).

    General isentropic sound speed:
        c² = (∂p/∂ρ)_T - (∂p/∂T)_ρ · (∂s/∂ρ)_T / (∂s/∂T)_ρ

    Applied to T-eq mixture (K=2, SG EOS):
        1/(ρ c_eff²) = Σ α_k/(ρ_k c_{s,k}²)  [Wood]
                      + (α₁ρ₁Cp₁)(α₂ρ₂Cp₂)(ζ₂-ζ₁)² / (T·Σα_kρ_kCp_k)  [T-eq cross]

    Returns c_eff (same shape as a1).
    """
    g1, pinf1, kv1 = ph1['gamma'], ph1['pinf'], ph1['kv']
    g2, pinf2, kv2 = ph2['gamma'], ph2['pinf'], ph2['kv']
    a2 = 1.0 - a1

    pp1 = np.maximum(p + pinf1, _EPS)
    pp2 = np.maximum(p + pinf2, _EPS)
    T_safe = np.maximum(T, 1.0)

    # Isentropic sound speeds per phase
    c1_sq = g1 * pp1 / np.maximum(rho1, _EPS)
    c2_sq = g2 * pp2 / np.maximum(rho2, _EPS)

    # Wood (pressure-equilibrium) part
    rho = a1 * rho1 + a2 * rho2
    wood_inv = a1 / np.maximum(rho1 * c1_sq, _EPS) + a2 / np.maximum(rho2 * c2_sq, _EPS)

    # T-eq cross term: Cp_k = γ_k kv_k, ζ_k = (γ_k-1)T/(γ_k(p+P∞_k))
    Cp1, Cp2 = g1 * kv1, g2 * kv2
    zeta1 = (g1 - 1.0) * T_safe / np.maximum(g1 * pp1, _EPS)
    zeta2 = (g2 - 1.0) * T_safe / np.maximum(g2 * pp2, _EPS)

    arCp1 = a1 * rho1 * Cp1
    arCp2 = a2 * rho2 * Cp2
    sum_arCp = arCp1 + arCp2
    cross = arCp1 * arCp2 * (zeta2 - zeta1) ** 2 / np.maximum(T_safe * sum_arCp, _EPS)

    inv_rho_ceff_sq = wood_inv + cross
    c_eff = np.sqrt(1.0 / np.maximum(rho * inv_rho_ceff_sq, _EPS))

    return c_eff


# ---------------------------------------------------------------------------
# Temperature-equilibrium distribution coefficient (He & Tan 2024 Eq. A.19)
# ---------------------------------------------------------------------------

def _lambda_temp_eq(a1, rho1, rho2, p, T, ph1, ph2):
    """Distribution coefficient lambda_1 for temperature equilibrium.

    He & Tan 2024, Appendix A, Eq. A.19, specialized for SG EOS (b=0, eta=0).
    Returns lambda_1 at cell centers (N,).

    SG EOS thermodynamic derivatives:
      A_k = rho_k / (p + Pinf_k)              isothermal compressibility
      D_k = rho_k / T                          thermal expansion
      B_k = -kv_k T (gamma_k-1) Pinf_k / (p + Pinf_k)^2
      C_k = kv_k (p + gamma_k Pinf_k) / (p + Pinf_k)
      Cp_k = gamma_k * kv_k
      zeta_k = (gamma_k - 1) T / (gamma_k (p + Pinf_k))
    """
    g1, pinf1, kv1 = ph1['gamma'], ph1['pinf'], ph1['kv']
    g2, pinf2, kv2 = ph2['gamma'], ph2['pinf'], ph2['kv']
    a2 = 1.0 - a1

    # Safe denominators
    pp1 = np.maximum(p + pinf1, _EPS)
    pp2 = np.maximum(p + pinf2, _EPS)
    T_safe = np.maximum(T, 1.0)

    # Thermodynamic derivatives (SG EOS)
    A1 = rho1 / pp1
    A2 = rho2 / pp2
    D1 = rho1 / T_safe
    D2 = rho2 / T_safe
    B1 = -kv1 * T_safe * (g1 - 1.0) * pinf1 / (pp1 * pp1)
    B2 = -kv2 * T_safe * (g2 - 1.0) * pinf2 / (pp2 * pp2)
    C1 = kv1 * (p + g1 * pinf1) / pp1
    C2 = kv2 * (p + g2 * pinf2) / pp2

    # Cp and zeta for c_eff computation
    Cp1, Cp2 = g1 * kv1, g2 * kv2
    zeta1 = (g1 - 1.0) * T_safe / np.maximum(g1 * pp1, _EPS)
    zeta2 = (g2 - 1.0) * T_safe / np.maximum(g2 * pp2, _EPS)

    # Isentropic sound speeds
    c1_sq = g1 * pp1 / np.maximum(rho1, _EPS)
    c2_sq = g2 * pp2 / np.maximum(rho2, _EPS)

    # Temperature-equilibrium mixture sound speed (Eq. A.17, K=2)
    rho = a1 * rho1 + a2 * rho2
    wood_inv = a1 / np.maximum(rho1 * c1_sq, _EPS) + a2 / np.maximum(rho2 * c2_sq, _EPS)
    arCp1 = a1 * rho1 * Cp1
    arCp2 = a2 * rho2 * Cp2
    sum_arCp = arCp1 + arCp2
    cross = arCp1 * arCp2 * (zeta2 - zeta1) ** 2 / np.maximum(T_safe * sum_arCp, _EPS)
    inv_rho_ceff_sq = wood_inv + cross
    rho_ceff_sq = 1.0 / np.maximum(inv_rho_ceff_sq, _EPS)  # = rho * c_eff^2

    # Sums for lambda formula
    sum_arB = a1 * rho1 * B1 + a2 * rho2 * B2
    sum_arC = a1 * rho1 * C1 + a2 * rho2 * C2
    inv_sum_arC = 1.0 / np.maximum(np.abs(sum_arC), _EPS) * np.sign(sum_arC + _EPS)

    # Lambda_1 (Eq. A.19): for SG, A_k/rho_k = 1/(p+Pinf_k), D_k/rho_k = 1/T
    lambda1 = (1.0 / pp1 + sum_arB * inv_sum_arC / T_safe) * rho_ceff_sq \
              - p * inv_sum_arC / T_safe

    # Clip to physically reasonable range [0, 5]
    lambda1 = np.clip(lambda1, 0.0, 5.0)

    return lambda1


# ---------------------------------------------------------------------------
# Instantaneous temperature relaxation (4-equation T-equilibrium closure)
# ---------------------------------------------------------------------------

def _temperature_relaxation(a1r1, a2r2, ru, rE, a1, ph1, ph2):
    """Enforce T₁ = T₂ by solving the 4-equation T-equilibrium closure.

    He & Tan 2024 Eq. A.20-A.22, specialized for Air (Ideal) + Water (SG).

    Preserves: a1r1, a2r2, ru  (mass & momentum conservation)
    Modifies:  a1, rE           (temperature equilibrium)

    For Ideal Gas (P∞₁=0) + SG (P∞₂≠0), pressure satisfies a quadratic:
        a·p² + b·p + c = 0
    where A_k = (α_k ρ_k) · kv_k:
        a = A₁ + A₂
        b = (A₁+A₂)P∞₂ - [A₁(γ₁-1)+A₂(γ₂-1)]ρe + A₂(γ₂-1)P∞₂
        c = -A₁(γ₁-1)·ρe·P∞₂
    """
    g1, pinf1, kv1 = ph1['gamma'], ph1['pinf'], ph1['kv']
    g2, pinf2, kv2 = ph2['gamma'], ph2['pinf'], ph2['kv']

    rho = a1r1 + a2r2
    rho_safe = np.maximum(rho, _EPS)
    u = ru / rho_safe
    rho_e = rE - 0.5 * ru * u  # internal energy density

    # A_k = partial_density_k * Cv_k
    A1 = np.maximum(a1r1, 0.0) * kv1
    A2 = np.maximum(a2r2, 0.0) * kv2

    # Quadratic coefficients for p
    gm1 = g1 - 1.0
    gm2 = g2 - 1.0
    a_coeff = A1 + A2
    b_coeff = (A1 + A2) * pinf2 - (A1 * gm1 + A2 * gm2) * rho_e + A2 * gm2 * pinf2
    c_coeff = -A1 * gm1 * rho_e * pinf2

    # Solve quadratic: p = (-b + sqrt(b²-4ac)) / (2a)
    disc = b_coeff ** 2 - 4.0 * a_coeff * c_coeff
    disc_safe = np.maximum(disc, 0.0)
    p_eq = (-b_coeff + np.sqrt(disc_safe)) / np.maximum(2.0 * a_coeff, _EPS)
    p_eq = np.maximum(p_eq, 1.0)

    # Temperature from volume constraint:
    # T = 1 / [A₁(γ₁-1)/p + A₂(γ₂-1)/(p+P∞₂)]
    denom_T = A1 * gm1 / np.maximum(p_eq, _EPS) + A2 * gm2 / np.maximum(p_eq + pinf2, _EPS)
    T_eq = 1.0 / np.maximum(denom_T, _EPS)
    T_eq = np.maximum(T_eq, 1.0)

    # Phase densities from (p, T)
    rho1_eq = (p_eq + pinf1) / np.maximum(gm1 * kv1 * T_eq, _EPS)
    rho2_eq = (p_eq + pinf2) / np.maximum(gm2 * kv2 * T_eq, _EPS)

    # New volume fraction: α₁ = a1r1 / ρ₁
    a1_new = np.maximum(a1r1, 0.0) / np.maximum(rho1_eq, _EPS)
    a1_new = np.clip(a1_new, 0.0, 1.0)
    a2_new = 1.0 - a1_new

    # Consistent ρE: Σ α_k(p+γ_k P∞_k)/(γ_k-1) + ½ρu²
    rho_e_new = (a1_new * (p_eq + g1 * pinf1) / gm1
                 + a2_new * (p_eq + g2 * pinf2) / gm2)
    rE_new = rho_e_new + 0.5 * ru * u

    return a1r1, a2r2, ru, rE_new, a1_new


# ---------------------------------------------------------------------------
# MMACM-Ex: H_k characteristic function (Zhao 2025 Eq. 32)
# ---------------------------------------------------------------------------

def _hk_characteristic(a1, bc_l, bc_r, eps_intf=1e-4):
    """H_k at cell centers (N,).

    H_k = (1 - ((1-|r|)/(1+|r|))^4)  if interface cell, else 0.
    r = (a_i - a_{i-1}) / (a_{i+1} - a_i)  (slope ratio)
    |n_x| = 1 in 1D.
    """
    N = len(a1)
    a_ext = _ghost(a1, bc_l, bc_r, ng=1)
    dL = a_ext[1:N+1] - a_ext[0:N]      # a_i - a_{i-1}
    dR = a_ext[2:N+2] - a_ext[1:N+1]    # a_{i+1} - a_i

    # Slope ratio r = dL / dR (sign-safe)
    abs_dR = np.abs(dR)
    sign_dR = np.where(dR >= 0, 1.0, -1.0)
    r = dL * sign_dR / np.maximum(abs_dR, 1e-30)
    abs_r = np.abs(r)

    ratio = (1.0 - abs_r) / np.maximum(1.0 + abs_r, 1e-30)
    H_raw = 1.0 - ratio ** 4

    # Interface detection: eps < a1 < 1-eps AND monotone
    in_range = (a1 > eps_intf) & (a1 < 1.0 - eps_intf)
    monotone = (dL * dR) > 0.0
    is_interface = in_range & monotone

    H = np.where(is_interface, H_raw, 0.0)
    return np.clip(H, 0.0, 1.0)


# ---------------------------------------------------------------------------
# OpenFOAM-style compression flux + Zalesak FCT limiter
# ---------------------------------------------------------------------------

def _compression_flux(a1, u_face, bc_l, bc_r, C_alpha=1.0):
    """OpenFOAM-style anti-diffusion compression flux at N+1 faces.

    F_comp = u_c · α_face · (1 - α_face)
    where u_c = C_α · |u| · sign(∇α)  (compression velocity toward interface).
    α_face is upwinded with respect to u_c (not u).

    Parameters
    ----------
    a1     : (N,) cell-center volume fraction
    u_face : (N+1,) face velocity
    bc_l/r : boundary condition
    C_alpha: compression coefficient (0=none, 1=standard, up to 4)

    Returns
    -------
    F_comp : (N+1,) raw compression flux (before FCT limiting)
    """
    N = len(a1)
    a1_ext = _ghost(a1, bc_l, bc_r, ng=1)  # (N+2,)

    # Interface normal at faces: sign(α_R - α_L)
    grad_alpha = a1_ext[1:N+2] - a1_ext[0:N+1]  # (N+1,)
    n_hat = np.sign(grad_alpha)

    # Compression velocity: pushes toward interface
    u_c = C_alpha * np.abs(u_face) * n_hat  # (N+1,)

    # Upwind α with respect to u_c
    alpha_face = np.where(u_c >= 0.0, a1_ext[0:N+1], a1_ext[1:N+2])

    return u_c * alpha_face * (1.0 - alpha_face)


def _zalesak_fct_limit(F_comp, a1, dx, dt, bc_l, bc_r):
    """Zalesak FCT limiter: limit compression flux to keep α ∈ [0, 1].

    Guarantees boundedness AND conservation (no clip needed).
    For each cell, computes the maximum flux that keeps α in bounds,
    then limits each face flux by the minimum of its two cells' limits.

    Parameters
    ----------
    F_comp : (N+1,) raw compression flux
    a1     : (N,) current cell-center volume fraction
    dx     : cell size
    dt     : current sub-step Δt
    bc_l/r : boundary condition

    Returns
    -------
    F_limited : (N+1,) FCT-limited compression flux
    """
    N = len(a1)
    _eps_fct = 1e-30

    # Net flux contribution to each cell from compression
    # Cell i: dα_i = -(F[i+1] - F[i]) * dt / dx
    #       = (F[i] - F[i+1]) * dt / dx
    #       = contrib_L + contrib_R
    contrib_L = F_comp[0:N] * dt / dx          # from left face (positive = into cell)
    contrib_R = -F_comp[1:N+1] * dt / dx       # from right face (negative F = into cell)

    # Total positive (inflow) and negative (outflow) contributions
    P_plus = np.maximum(contrib_L, 0.0) + np.maximum(contrib_R, 0.0)
    P_minus = np.maximum(-contrib_L, 0.0) + np.maximum(-contrib_R, 0.0)

    # Maximum allowable increase/decrease to stay in [0, 1]
    Q_plus = np.maximum(1.0 - a1, 0.0)    # headroom to α=1
    Q_minus = np.maximum(a1, 0.0)          # headroom to α=0

    # Limiting ratios
    R_plus = np.where(P_plus > _eps_fct, np.minimum(Q_plus / P_plus, 1.0), 1.0)
    R_minus = np.where(P_minus > _eps_fct, np.minimum(Q_minus / P_minus, 1.0), 1.0)

    # Per-face limiter: min of donor's outflow limit and acceptor's inflow limit
    is_periodic = (bc_l == 'periodic')
    iL = np.arange(N + 1) - 1    # left cell of face
    iR = np.arange(N + 1)        # right cell of face

    if is_periodic:
        iL = iL % N
        iR = iR % N
    else:
        iL = np.clip(iL, 0, N - 1)
        iR = np.clip(iR, 0, N - 1)

    # F > 0: flux left→right (left cell loses, right cell gains)
    # F < 0: flux right→left (right cell loses, left cell gains)
    C_k = np.where(
        F_comp > 0,
        np.minimum(R_minus[iL], R_plus[iR]),
        np.where(F_comp < 0,
                 np.minimum(R_plus[iL], R_minus[iR]),
                 1.0))

    return C_k * F_comp


# ---------------------------------------------------------------------------
# MMACM-Ex correction fluxes (Zhao 2025 Eqs. 26-32)
# ---------------------------------------------------------------------------

def _mmacm_ex_correction(a1, a1r1, a2r2, rho1, rho2, p, u_vel, u_face, S_star,
                          F_alpha_base, ph1, ph2,
                          bc_l, bc_r, eps_intf=1e-4):
    """Compute MMACM-Ex sharpening correction G at all N+1 faces.

    Paper-exact implementation (Zhao et al. 2025, Eqs. 27-32):
      1. H_k at cell centers (Eq. 32)
      2. Upwind H at faces (Eq. 28): char_face = H_{upwind_cell}
      3. Pure 1st-order downwind alpha (Eq. 30): a1_down
      4. J_k = H̃ · (ū · α̂ - F̂^α)  (Eq. 29)  — uses HLLC alpha flux F̂^α
      5. Conservation consistency (Eq. 27): G^{a1r1}, G^{a2r2}, G^{ru}, G^{rE}

    rho1, rho2: T-consistent phase densities from cons_to_prim (no alpha division).
    """
    N = len(a1)
    g1, pinf1 = ph1['gamma'], ph1['pinf']
    g2, pinf2 = ph2['gamma'], ph2['pinf']

    # H_k at cell centers
    H_cell = _hk_characteristic(a1, bc_l, bc_r, eps_intf)
    H_ext = _ghost(H_cell, bc_l, bc_r, ng=1)

    # Upwind H at faces (Eq. 28): use sgn(S*) for upwind direction
    char_face = np.where(S_star >= 0.0, H_ext[0:N+1], H_ext[1:N+2])

    # Pure 1st-order downwind alpha (Eq. 30):
    # downwind = cell that flow goes INTO
    a1_ext = _ghost(a1, bc_l, bc_r, ng=1)
    a1_down = np.where(S_star >= 0.0, a1_ext[1:N+2], a1_ext[0:N+1])

    # J_k = H̃ · (ū · α̂ - F̂^α) (Eq. 29)
    # F̂^α = F_alpha_base (HLLC alpha flux from Eq. 26)
    # ū = u_face (HLLC consistent face velocity from Eq. 25)
    G_alpha = char_face * (u_face * a1_down - F_alpha_base)

    # Upwind cell quantities for conservation consistency (Eq. 27)
    p_ext       = _ghost(p,     bc_l, bc_r)
    u_ext       = _ghost(u_vel, bc_l, bc_r)
    rho1_ext    = _ghost(rho1,  bc_l, bc_r)
    rho2_ext    = _ghost(rho2,  bc_l, bc_r)

    p_up    = np.where(S_star >= 0.0, p_ext[0:N+1],    p_ext[1:N+2])
    u_up    = np.where(S_star >= 0.0, u_ext[0:N+1],    u_ext[1:N+2])

    # T-consistent phase densities from cons_to_prim (no α-division, smooth at interface)
    rho1_up = np.where(S_star >= 0.0, rho1_ext[0:N+1], rho1_ext[1:N+2])
    rho2_up = np.where(S_star >= 0.0, rho2_ext[0:N+1], rho2_ext[1:N+2])
    rho1_up = np.maximum(rho1_up, _EPS)
    rho2_up = np.maximum(rho2_up, _EPS)

    # Phase specific internal energies from EOS
    e1_up = _sg_internal_energy(p_up, rho1_up, g1, pinf1)
    e2_up = _sg_internal_energy(p_up, rho2_up, g2, pinf2)
    E1_up = e1_up + 0.5 * u_up ** 2
    E2_up = e2_up + 0.5 * u_up ** 2

    # Conservation consistency corrections (Eq. 27)
    G_a1r1 =  rho1_up * G_alpha
    G_a2r2 = -rho2_up * G_alpha
    G_ru   = (rho1_up - rho2_up) * u_up * G_alpha
    G_rE   = (rho1_up * E1_up - rho2_up * E2_up) * G_alpha

    return G_a1r1, G_a2r2, G_ru, G_rE, G_alpha


# ---------------------------------------------------------------------------
# Compute spatial residual dQ/dt (one full RHS evaluation)
# ---------------------------------------------------------------------------

def _rhs(a1r1, a2r2, ru, rE, a1, ph1, ph2,
         dx, bc_l='transmissive', bc_r='transmissive',
         use_mmacm_ex=True, eps_intf=1e-4,
         alpha_recon='thinc_bvd', dt_sub=None,
         use_compression=False, C_alpha=1.0,
         compress_corrections=False,
         use_apec=False):
    """Compute dQ/dt = -dF/dx + G_correction  for all cells.

    Returns: (da1r1, da2r2, dru, drE, da1) each (N,).

    Alpha equation (non-conservative):
        da1/dt = -(d(a1*u)/dx) + (a1 + D1) * du/dx
    Here D1=0 (Allaire-Massoni), so:
        da1/dt = -(d(a1*u)/dx) + a1 * du/dx = -u * da1/dx
    Implemented as:
        da1/dt = -(F_alpha_{i+1/2} - F_alpha_{i-1/2})/dx + a1_i*(u_{i+1/2}-u_{i-1/2})/dx
    where F_alpha = a1_upwind * u_face  (upwind alpha flux, then corrected by MMACM-Ex).
    """
    N = len(a1)

    # --- Primitive variables at cell centers ---
    p, u_vel, T, rho1, rho2, c1, c2, c_wood = cons_to_prim(
        a1r1, a2r2, ru, rE, a1, ph1, ph2)

    # --- Interface cell: freeze rho1, rho2 slopes ---
    is_intf = _interface_mask(a1, eps_intf)

    # --- Reconstruct primitive variables at faces ---
    # Variables: (ρ₁, ρ₂, u, p, α₁) — He & Zhao 2025 Section IV
    # ρ₁, ρ₂ reconstructed directly so face density is taken straight from TVD.
    # This accurately captures density jumps at contact discontinuities and
    # eliminates density peaks caused by p-inconsistency in T-based reconstruction.
    # Interface cells: ρ₁, ρ₂, p, u slopes frozen (Eq. 19)

    g1, pinf1 = ph1['gamma'], ph1['pinf']
    g2, pinf2 = ph2['gamma'], ph2['pinf']
    kv1, kv2 = ph1['kv'], ph2['kv']
    gm1, gm2 = g1 - 1.0, g2 - 1.0

    # TVD reconstruction of (ρ₁, ρ₂, u, p) — He & Zhao 2025 Section IV
    rho1L, rho1R = _tvd_reconstruct(rho1, bc_l, bc_r)
    rho2L, rho2R = _tvd_reconstruct(rho2, bc_l, bc_r)
    uL,    uR    = _tvd_reconstruct(u_vel, bc_l, bc_r)
    pL,    pR    = _tvd_reconstruct(p, bc_l, bc_r)

    # α₁: selectable reconstruction scheme
    if alpha_recon == 'tvd':
        a1L, a1R = _tvd_reconstruct(a1, bc_l, bc_r)
    elif alpha_recon in ('cicsam', 'mstacs', 'superbee', 'saish'):
        # NVD schemes: estimate face velocity from cell-center u
        u_ext = _ghost(u_vel, bc_l, bc_r, ng=1)
        u_face_est = 0.5 * (u_ext[:-1] + u_ext[1:])  # (N+1,)
        dt_use = dt_sub if dt_sub is not None else dx * 0.4 / np.maximum(
            np.max(np.abs(u_vel) + c_wood), _EPS)
        cds_map = {'cicsam': 'hyper_c', 'mstacs': 'mstacs',
                   'superbee': 'superbee', 'saish': 'saish'}
        alpha_face = _nvd_face(a1, u_face_est, dt_use, dx, bc_l, bc_r,
                               cds=cds_map[alpha_recon])
        a1L = alpha_face.copy()
        a1R = alpha_face.copy()
    else:  # thinc_bvd (default)
        a1L, a1R = _thinc_bvd_reconstruct(a1, bc_l, bc_r, beta=2.0)

    # Physical bounds
    rho1L = np.maximum(rho1L, _EPS); rho1R = np.maximum(rho1R, _EPS)
    rho2L = np.maximum(rho2L, _EPS); rho2R = np.maximum(rho2R, _EPS)
    pL    = np.maximum(pL,    1.0);  pR    = np.maximum(pR,    1.0)
    a1L   = np.clip(a1L, 0.0, 1.0); a1R   = np.clip(a1R, 0.0, 1.0)

    # --- Freeze ρ₁, ρ₂, p, u at interface cells (Eq. 19) ---
    for i in range(N):
        if is_intf[i]:
            rho1R[i] = rho1[i]; rho1L[i+1] = rho1[i]
            rho2R[i] = rho2[i]; rho2L[i+1] = rho2[i]
            pR[i]    = p[i];    pL[i+1]    = p[i]
            uR[i]    = u_vel[i]; uL[i+1]   = u_vel[i]

    # Conservative face states
    a2L = np.maximum(1.0 - a1L, 0.0); a2R = np.maximum(1.0 - a1R, 0.0)
    a1r1_fL = a1L * rho1L;  a1r1_fR = a1R * rho1R
    a2r2_fL = a2L * rho2L;  a2r2_fR = a2R * rho2R
    rho_fL  = a1r1_fL + a2r2_fL
    rho_fR  = a1r1_fR + a2r2_fR
    ru_fL   = rho_fL * uL;  ru_fR  = rho_fR * uR

    # ρE from (p, α) — SG identity: α_k·ρ_k·e_k = α_k·(p+γ_k·P∞_k)/(γ_k-1)
    rho_e_fL = a1L * (pL + g1*pinf1)/(g1-1.0) + a2L * (pL + g2*pinf2)/(g2-1.0)
    rho_e_fR = a1R * (pR + g1*pinf1)/(g1-1.0) + a2R * (pR + g2*pinf2)/(g2-1.0)
    rE_fL = rho_e_fL + 0.5 * rho_fL * uL ** 2
    rE_fR = rho_e_fR + 0.5 * rho_fR * uR ** 2

    # T-equilibrium mixture sound speed at faces (He & Tan 2024 Eq. A.17)
    T_fL = np.where(a1L >= 0.5,
                    (pL + pinf1) / np.maximum(gm1 * kv1 * rho1L, _EPS),
                    (pL + pinf2) / np.maximum(gm2 * kv2 * rho2L, _EPS))
    T_fR = np.where(a1R >= 0.5,
                    (pR + pinf1) / np.maximum(gm1 * kv1 * rho1R, _EPS),
                    (pR + pinf2) / np.maximum(gm2 * kv2 * rho2R, _EPS))
    T_fL = np.maximum(T_fL, 1.0)
    T_fR = np.maximum(T_fR, 1.0)
    c_fL = _ceff_temp_eq(a1L, rho1L, rho2L, pL, T_fL, ph1, ph2)
    c_fR = _ceff_temp_eq(a1R, rho1R, rho2R, pR, T_fR, ph1, ph2)

    # --- HLLC flux ---
    F_a1r1, F_a2r2, F_ru, F_rE, u_face, S_star = _hllc_flux(
        a1r1_fL, a2r2_fL, ru_fL, rE_fL, pL, c_fL,
        a1r1_fR, a2r2_fR, ru_fR, rE_fR, pR, c_fR)

    # --- APEC energy flux (pressure-equilibrium preserving) ---
    # Replaces standard HLLC F_rE with:
    #   F_rE^APEC = ε₁·F_{a1r1} + ε₂·F_{a2r2} + ½ū²·F_ρ + p̄·ū
    # This decomposition preserves p-equilibrium at contacts exactly.
    if use_apec:
        # Upwind specific internal energy per phase
        e1_up = np.where(S_star >= 0.0,
                         _sg_internal_energy(pL, rho1L, g1, pinf1),
                         _sg_internal_energy(pR, rho1R, g1, pinf1))
        e2_up = np.where(S_star >= 0.0,
                         _sg_internal_energy(pL, rho2L, g2, pinf2),
                         _sg_internal_energy(pR, rho2R, g2, pinf2))
        # Upwind pressure and velocity
        p_up = np.where(S_star >= 0.0, pL, pR)
        # APEC energy flux
        F_rho = F_a1r1 + F_a2r2
        F_rE = e1_up * F_a1r1 + e2_up * F_a2r2 + 0.5 * u_face**2 * F_rho + p_up * u_face

    # --- Upwind alpha flux for volume fraction equation (Eq. 26) ---
    # F_alpha = F_{a1r1} / rho1_upwind (Johnsen & Colonius 2006)
    # Use sgn(S*) for upwind direction, reconstructed face density
    rho1_up_face = np.where(S_star >= 0.0, rho1L, rho1R)
    rho1_up_face = np.maximum(rho1_up_face, 1e-2)  # floor=1e-2 kg/m³

    # Alpha flux from mass flux / rho1_upwind (Eq. 26)
    F_alpha_base = F_a1r1 / rho1_up_face

    # --- Step 1: Compression term (applied first, before MMACM) ---
    F_comp = np.zeros(N + 1)
    if use_compression:
        F_comp_raw = _compression_flux(a1, u_face, bc_l, bc_r, C_alpha)
        if dt_sub is not None and dt_sub > 0:
            F_comp = _zalesak_fct_limit(F_comp_raw, a1, dx, dt_sub, bc_l, bc_r)
        else:
            F_comp = F_comp_raw

    # α flux after compression (before MMACM)
    F_alpha_pre = F_alpha_base + F_comp

    # --- Step 2: MMACM-Ex correction (sees compression-modified flux) ---
    if use_mmacm_ex:
        # MMACM computes G_alpha relative to F_alpha_pre (includes compression).
        # G_alpha = H_k * (u·α_down - F_alpha_pre) — only the REMAINING deficit.
        G_a1r1, G_a2r2, G_ru, G_rE, G_alpha = _mmacm_ex_correction(
            a1, a1r1, a2r2, rho1, rho2, p, u_vel, u_face, S_star,
            F_alpha_pre, ph1, ph2, bc_l, bc_r, eps_intf)
        # Full conservation consistency (Eq. 27): G corrections cover G_alpha only
        F_a1r1_total = F_a1r1 + G_a1r1
        F_a2r2_total = F_a2r2 + G_a2r2
        F_ru_total   = F_ru   + G_ru
        F_rE_total   = F_rE   + G_rE
        F_alpha_total = F_alpha_pre + G_alpha
    else:
        F_a1r1_total = F_a1r1
        F_a2r2_total = F_a2r2
        F_ru_total   = F_ru
        F_rE_total   = F_rE
        F_alpha_total = F_alpha_pre

    # --- Step 3: Conservation corrections for compression flux ---
    if use_compression and compress_corrections:
        p_ext    = _ghost(p,     bc_l, bc_r)
        u_ext    = _ghost(u_vel, bc_l, bc_r)
        rho1_ext = _ghost(rho1,  bc_l, bc_r)
        rho2_ext = _ghost(rho2,  bc_l, bc_r)
        p_up    = np.where(S_star >= 0, p_ext[0:N+1],    p_ext[1:N+2])
        u_up    = np.where(S_star >= 0, u_ext[0:N+1],    u_ext[1:N+2])
        r1_up   = np.maximum(np.where(S_star >= 0, rho1_ext[0:N+1], rho1_ext[1:N+2]), _EPS)
        r2_up   = np.maximum(np.where(S_star >= 0, rho2_ext[0:N+1], rho2_ext[1:N+2]), _EPS)
        e1_up   = _sg_internal_energy(p_up, r1_up, g1, pinf1)
        e2_up   = _sg_internal_energy(p_up, r2_up, g2, pinf2)
        E1_up   = e1_up + 0.5 * u_up ** 2
        E2_up   = e2_up + 0.5 * u_up ** 2
        F_a1r1_total = F_a1r1_total + r1_up * F_comp
        F_a2r2_total = F_a2r2_total - r2_up * F_comp
        F_ru_total   = F_ru_total   + (r1_up - r2_up) * u_up * F_comp
        F_rE_total   = F_rE_total   + (r1_up * E1_up - r2_up * E2_up) * F_comp

    # --- Divergence ---
    inv_dx = 1.0 / dx
    d_a1r1 = -(F_a1r1_total[1:N+1] - F_a1r1_total[0:N]) * inv_dx
    d_a2r2 = -(F_a2r2_total[1:N+1] - F_a2r2_total[0:N]) * inv_dx
    d_ru   = -(F_ru_total[1:N+1]   - F_ru_total[0:N])   * inv_dx
    d_rE   = -(F_rE_total[1:N+1]   - F_rE_total[0:N])   * inv_dx

    # --- Volume fraction equation (non-conservative) ---
    # T-equilibrium: da1/dt = -d(a1*u)/dx + a1 * lambda1 * du/dx
    # lambda1 = distribution coefficient from He & Tan 2024 Eq. A.19
    du_dx = (u_face[1:N+1] - u_face[0:N]) * inv_dx
    lambda1 = _lambda_temp_eq(a1, rho1, rho2, p, T, ph1, ph2)
    d_alpha = (-(F_alpha_total[1:N+1] - F_alpha_total[0:N]) * inv_dx
               + a1 * lambda1 * du_dx)

    return d_a1r1, d_a2r2, d_ru, d_rE, d_alpha


# ---------------------------------------------------------------------------
# CFL-based time step
# ---------------------------------------------------------------------------

def _compute_dt(a1r1, a2r2, ru, rE, a1, ph1, ph2, dx, cfl):
    """Compute dt = CFL * dx / max(|u| + c_wood)."""
    p, u_vel, T, rho1, rho2, c1, c2, c_wood = cons_to_prim(
        a1r1, a2r2, ru, rE, a1, ph1, ph2)
    max_speed = np.max(np.abs(u_vel) + c_wood)
    max_speed = max(max_speed, _EPS)
    return cfl * dx / max_speed


# ---------------------------------------------------------------------------
# SSP-RK3 time integration (Shu-Osher 1988)
# ---------------------------------------------------------------------------

def _ssp_rk3_step(a1r1, a2r2, ru, rE, a1, ph1, ph2,
                  dx, dt, bc_l, bc_r, use_mmacm_ex=True, eps_intf=1e-4,
                  alpha_recon='thinc_bvd',
                  use_compression=False, C_alpha=1.0,
                  compress_corrections=False, use_apec=False):
    """One SSP-RK3 step. Returns updated (a1r1, a2r2, ru, rE, a1)."""

    def rhs(q1, q2, q3, q4, q5):
        return _rhs(q1, q2, q3, q4, q5, ph1, ph2, dx, bc_l, bc_r,
                    use_mmacm_ex, eps_intf, alpha_recon, dt,
                    use_compression, C_alpha, compress_corrections, use_apec)

    def apply_bounds(q1, q2, q3, q4, q5):
        q1 = np.maximum(q1, 0.0)
        q2 = np.maximum(q2, 0.0)
        q5 = np.clip(q5, 0.0, 1.0)
        return q1, q2, q3, q4, q5

    # Stage 1: Q^(1) = Q^n + dt * RHS(Q^n)
    k1a, k1b, k1c, k1d, k1e = rhs(a1r1, a2r2, ru, rE, a1)
    q1_a1r1 = a1r1 + dt * k1a
    q1_a2r2 = a2r2 + dt * k1b
    q1_ru   = ru   + dt * k1c
    q1_rE   = rE   + dt * k1d
    q1_a1   = a1   + dt * k1e
    q1_a1r1, q1_a2r2, q1_ru, q1_rE, q1_a1 = apply_bounds(
        q1_a1r1, q1_a2r2, q1_ru, q1_rE, q1_a1)

    # Stage 2: Q^(2) = (3/4)*Q^n + (1/4)*(Q^(1) + dt*RHS(Q^(1)))
    k2a, k2b, k2c, k2d, k2e = rhs(q1_a1r1, q1_a2r2, q1_ru, q1_rE, q1_a1)
    q2_a1r1 = 0.75 * a1r1 + 0.25 * (q1_a1r1 + dt * k2a)
    q2_a2r2 = 0.75 * a2r2 + 0.25 * (q1_a2r2 + dt * k2b)
    q2_ru   = 0.75 * ru   + 0.25 * (q1_ru   + dt * k2c)
    q2_rE   = 0.75 * rE   + 0.25 * (q1_rE   + dt * k2d)
    q2_a1   = 0.75 * a1   + 0.25 * (q1_a1   + dt * k2e)
    q2_a1r1, q2_a2r2, q2_ru, q2_rE, q2_a1 = apply_bounds(
        q2_a1r1, q2_a2r2, q2_ru, q2_rE, q2_a1)

    # Stage 3: Q^{n+1} = (1/3)*Q^n + (2/3)*(Q^(2) + dt*RHS(Q^(2)))
    k3a, k3b, k3c, k3d, k3e = rhs(q2_a1r1, q2_a2r2, q2_ru, q2_rE, q2_a1)
    new_a1r1 = (1.0/3.0) * a1r1 + (2.0/3.0) * (q2_a1r1 + dt * k3a)
    new_a2r2 = (1.0/3.0) * a2r2 + (2.0/3.0) * (q2_a2r2 + dt * k3b)
    new_ru   = (1.0/3.0) * ru   + (2.0/3.0) * (q2_ru   + dt * k3c)
    new_rE   = (1.0/3.0) * rE   + (2.0/3.0) * (q2_rE   + dt * k3d)
    new_a1   = (1.0/3.0) * a1   + (2.0/3.0) * (q2_a1   + dt * k3e)
    new_a1r1, new_a2r2, new_ru, new_rE, new_a1 = apply_bounds(
        new_a1r1, new_a2r2, new_ru, new_rE, new_a1)

    return new_a1r1, new_a2r2, new_ru, new_rE, new_a1


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def solve(ph1, ph2, a1r1_0, a2r2_0, ru_0, rE_0, a1_0,
          dx, t_end, cfl=0.4,
          bc_l='transmissive', bc_r='transmissive',
          use_mmacm_ex=True, eps_intf=1e-4,
          max_steps=100000, print_interval=50,
          alpha_recon='thinc_bvd',
          use_compression=False, C_alpha=1.0,
          compress_corrections=False, use_apec=False):
    """Explicit MMACM-Ex solver main loop."""
    a1r1 = a1r1_0.copy()
    a2r2 = a2r2_0.copy()
    ru    = ru_0.copy()
    rE    = rE_0.copy()
    a1    = a1_0.copy()

    t = 0.0
    step = 0

    while t < t_end and step < max_steps:
        dt = _compute_dt(a1r1, a2r2, ru, rE, a1, ph1, ph2, dx, cfl)
        dt = min(dt, t_end - t)
        if dt <= 0.0:
            break

        a1r1, a2r2, ru, rE, a1 = _ssp_rk3_step(
            a1r1, a2r2, ru, rE, a1, ph1, ph2,
            dx, dt, bc_l, bc_r, use_mmacm_ex, eps_intf, alpha_recon,
            use_compression, C_alpha, compress_corrections, use_apec)

        t += dt
        step += 1

        if step % print_interval == 0:
            p, u_vel, T, rho1, rho2, c1, c2, c_wood = cons_to_prim(
                a1r1, a2r2, ru, rE, a1, ph1, ph2)
            print(f"  step={step:5d}  t={t:.4e}  dt={dt:.3e}  "
                  f"p_max={p.max():.4e}  u_max={u_vel.max():.3f}  "
                  f"a1_range=[{a1.min():.4f},{a1.max():.4f}]")

    print(f"Done: {step} steps, t={t:.4e}")
    return t, a1r1, a2r2, ru, rE, a1


# ---------------------------------------------------------------------------
# Phase 2-1 setup: HP Air (left) / LP Water (right)
# ---------------------------------------------------------------------------

def run_phase2_1(N=200, cfl=0.4, t_end=8.0e-4, use_mmacm_ex=True,
                 print_interval=50, alpha_recon='thinc_bvd',
                 use_compression=False, C_alpha=1.0,
                 compress_corrections=False, use_apec=False):
    """Run Phase 2-1: high-pressure Air / low-pressure SG Water shock tube.

    Domain: [0, 2] m, N=200 cells
    Air  (left,  x < 0.5): Ideal Gas, gamma=1.4, Pinf=0, p_L=1e9 Pa
    Water(right, x >= 0.5): Stiffened Gas, gamma=4.1, Pinf=4.4e8, p_R=1e4 Pa
    T_0 = 300 K everywhere, u_0 = 0 m/s
    CFL = 0.4, t_end = 8e-4 s (full spec), or 2.4e-4 (paper)

    Returns
    -------
    x, t_final, a1r1, a2r2, ru, rE, a1, ph1, ph2
    """
    # EOS parameters
    # Phase 1 = Air (Ideal Gas): alpha_1=1 in left region
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5}
    # Phase 2 = Water (Stiffened Gas): alpha_2=1 in right region
    ph2 = {'gamma': 4.1, 'pinf': 4.4e8, 'kv': 474.2}

    L = 2.0
    dx = L / N
    x = np.linspace(0.5 * dx, L - 0.5 * dx, N)

    # Interface position
    x_intf = 0.5

    # Initial conditions
    T0 = 300.0
    p_L = 1.0e9    # 1 GPa (Air left)
    p_R = 1.0e4    # 10 kPa (Water right)
    u0 = 0.0

    g1, pinf1, kv1 = ph1['gamma'], ph1['pinf'], ph1['kv']
    g2, pinf2, kv2 = ph2['gamma'], ph2['pinf'], ph2['kv']

    # Volume fraction: Air = 1 - eps on left, Water = 1 - eps on right
    eps_pure = 1e-8
    a1 = np.where(x < x_intf, 1.0 - eps_pure, eps_pure)  # a1 = alpha_Air

    # Pressure field
    p_field = np.where(x < x_intf, p_L, p_R)

    # Phase densities from EOS
    rho1 = _sg_density_from_pT(p_field, T0, g1, pinf1, kv1)
    rho2 = _sg_density_from_pT(p_field, T0, g2, pinf2, kv2)

    # Partial densities and conservative variables
    a2 = 1.0 - a1
    a1r1 = a1 * rho1
    a2r2 = a2 * rho2
    rho = a1r1 + a2r2
    ru = rho * u0

    e1 = _sg_internal_energy(p_field, rho1, g1, pinf1)
    e2 = _sg_internal_energy(p_field, rho2, g2, pinf2)
    rho_e = a1 * rho1 * e1 + a2 * rho2 * e2
    rE = rho_e + 0.5 * rho * u0 ** 2

    print(f"Phase 2-1: HP Air / LP Water shock tube")
    print(f"  N={N}, dx={dx:.4f} m, CFL={cfl}, t_end={t_end:.2e} s")
    print(f"  Air: gamma={g1}, Pinf={pinf1}, kv={kv1}")
    print(f"  Water: gamma={g2}, Pinf={pinf2}, kv={kv2}")
    print(f"  p_L={p_L:.2e} Pa, p_R={p_R:.2e} Pa, T0={T0} K")
    print(f"  rho_Air_left ={rho1[0]:.3f} kg/m3")
    print(f"  rho_Water_right={rho2[-1]:.3f} kg/m3")
    print(f"  MMACM-Ex: {use_mmacm_ex}")

    # Run solver
    t_final, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve(
        ph1, ph2, a1r1, a2r2, ru, rE, a1,
        dx, t_end, cfl=cfl,
        bc_l='transmissive', bc_r='transmissive',
        use_mmacm_ex=use_mmacm_ex,
        print_interval=print_interval,
        alpha_recon=alpha_recon,
        use_compression=use_compression, C_alpha=C_alpha,
        compress_corrections=compress_corrections, use_apec=use_apec)

    return x, t_final, a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2


# ---------------------------------------------------------------------------
# Plotting utility
# ---------------------------------------------------------------------------

def _plot_phase2_1(x, t_final, a1r1, a2r2, ru, rE, a1, ph1, ph2,
                   save_path='results/phase2_1_mmacm_ex_paper.png'):
    """Generate 6-panel plot: density, pressure, velocity, Mach, impedance, alpha1."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    p, u_vel, T, rho1, rho2, c1, c2, c_wood = cons_to_prim(
        a1r1, a2r2, ru, rE, a1, ph1, ph2)

    rho = a1r1 + a2r2
    mach = np.abs(u_vel) / np.maximum(c_wood, _EPS)

    a2 = 1.0 - a1
    # Acoustic impedance: Z = rho * c  (mixture)
    Z = rho * c_wood

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f'Phase 2-1: HP Air / LP Water Shock Tube  (t={t_final:.4e} s, '
                 f'MMACM-Ex)', fontsize=13)

    panels = [
        (axes[0, 0], rho,   'Mixture Density (kg/m3)', 'Density'),
        (axes[0, 1], p,     'Pressure (Pa)',            'Pressure'),
        (axes[0, 2], u_vel, 'Velocity (m/s)',           'Velocity'),
        (axes[1, 0], mach,  'Mach Number',              'Mach'),
        (axes[1, 1], Z,     'Acoustic Impedance (kg/m2/s)', 'Impedance'),
        (axes[1, 2], a1,    'Volume Fraction alpha1 (Air)', 'Alpha_1'),
    ]

    for ax, data, ylabel, title in panels:
        ax.plot(x, data, 'b-', linewidth=1.2)
        ax.set_xlabel('x (m)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {save_path}")


# ===========================================================================
# IMPLICIT BACKWARD EULER SOLVER (added on top of explicit solver)
# ===========================================================================
# Uses the SAME spatial discretization (_rhs) as the explicit solver.
# Jacobian computed via dense finite differences (autograd-style column FD).
# For N<=50, dense direct solve is efficient.
# ===========================================================================

def _rhs_1st_order(a1r1, a2r2, ru, rE, a1, ph1, ph2,
                    dx, bc_l='transmissive', bc_r='transmissive'):
    """1st-order upwind spatial residual for implicit BE.

    No TVD, no THINC-BVD, no MMACM-Ex, no interface freeze.
    Smooth Jacobian suitable for Newton iteration.
    """
    N = len(a1)
    g1, pinf1 = ph1['gamma'], ph1['pinf']
    g2, pinf2 = ph2['gamma'], ph2['pinf']
    kv1, kv2 = ph1['kv'], ph2['kv']
    gm1, gm2 = g1 - 1.0, g2 - 1.0

    # --- Cell center primitives (mixture-T, no α-division for smooth Jacobian) ---
    a2 = 1.0 - a1
    rho = a1r1 + a2r2
    u_vel = ru / np.maximum(rho, _EPS)
    rho_e = rE - 0.5 * ru * u_vel

    # Pressure: standard 5-eq linear
    Gamma_inv = a1 / gm1 + a2 / gm2
    Pi = a1 * g1 * pinf1 / gm1 + a2 * g2 * pinf2 / gm2
    p = (rho_e - Pi) / np.maximum(Gamma_inv, _EPS)
    p = np.maximum(p, 1.0)

    # Temperature: mixture formula (no α-division!)
    T_numer = a1 * (p + pinf1) / (gm1 * kv1) + a2 * (p + pinf2) / (gm2 * kv2)
    T = T_numer / np.maximum(rho, _EPS)
    T = np.maximum(T, 1.0)

    # Phase densities from (p, T) — smooth, no α-division
    rho1 = (p + pinf1) / np.maximum(gm1 * kv1 * T, _EPS)
    rho2 = (p + pinf2) / np.maximum(gm2 * kv2 * T, _EPS)

    # --- 1st order reconstruction: face = cell center (upwind ghost) ---
    a1_ext = _ghost(a1, bc_l, bc_r, ng=1)
    rho1_ext = _ghost(rho1, bc_l, bc_r, ng=1)
    rho2_ext = _ghost(rho2, bc_l, bc_r, ng=1)
    p_ext = _ghost(p, bc_l, bc_r, ng=1)
    u_ext = _ghost(u_vel, bc_l, bc_r, ng=1)

    # Left state at face j = cell j-1, Right state = cell j
    a1L = a1_ext[0:N+1]
    a1R = a1_ext[1:N+2]
    rho1L = np.maximum(rho1_ext[0:N+1], _EPS)
    rho1R = np.maximum(rho1_ext[1:N+2], _EPS)
    rho2L = np.maximum(rho2_ext[0:N+1], _EPS)
    rho2R = np.maximum(rho2_ext[1:N+2], _EPS)
    pL = np.maximum(p_ext[0:N+1], 1.0)
    pR = np.maximum(p_ext[1:N+2], 1.0)
    uL = u_ext[0:N+1]
    uR = u_ext[1:N+2]

    # Conservative face states
    a2L = np.maximum(1.0 - a1L, 0.0)
    a2R = np.maximum(1.0 - a1R, 0.0)
    a1r1_fL = a1L * rho1L;  a1r1_fR = a1R * rho1R
    a2r2_fL = a2L * rho2L;  a2r2_fR = a2R * rho2R
    rho_fL = a1r1_fL + a2r2_fL
    rho_fR = a1r1_fR + a2r2_fR
    ru_fL = rho_fL * uL;  ru_fR = rho_fR * uR

    rho_e_fL = a1L*(pL+g1*pinf1)/gm1 + a2L*(pL+g2*pinf2)/gm2
    rho_e_fR = a1R*(pR+g1*pinf1)/gm1 + a2R*(pR+g2*pinf2)/gm2
    rE_fL = rho_e_fL + 0.5*rho_fL*uL**2
    rE_fR = rho_e_fR + 0.5*rho_fR*uR**2

    # Sound speeds at faces (Wood)
    c1L_sq = np.maximum(g1*(pL+pinf1)/np.maximum(rho1L, _EPS), _EPS)
    c1R_sq = np.maximum(g1*(pR+pinf1)/np.maximum(rho1R, _EPS), _EPS)
    c2L_sq = np.maximum(g2*(pL+pinf2)/np.maximum(rho2L, _EPS), _EPS)
    c2R_sq = np.maximum(g2*(pR+pinf2)/np.maximum(rho2R, _EPS), _EPS)
    inv_rc2_fL = a1L/np.maximum(rho1L*c1L_sq, _EPS) + a2L/np.maximum(rho2L*c2L_sq, _EPS)
    inv_rc2_fR = a1R/np.maximum(rho1R*c1R_sq, _EPS) + a2R/np.maximum(rho2R*c2R_sq, _EPS)
    c_fL = np.sqrt(1.0/np.maximum(rho_fL*inv_rc2_fL, _EPS))
    c_fR = np.sqrt(1.0/np.maximum(rho_fR*inv_rc2_fR, _EPS))

    # --- HLLC flux ---
    F_a1r1, F_a2r2, F_ru, F_rE, u_face, S_star = _hllc_flux(
        a1r1_fL, a2r2_fL, ru_fL, rE_fL, pL, c_fL,
        a1r1_fR, a2r2_fR, ru_fR, rE_fR, pR, c_fR)

    # --- Alpha flux (Eq. 26) ---
    rho1_up = np.where(S_star >= 0.0, rho1L, rho1R)
    rho1_up = np.maximum(rho1_up, 1e-2)
    F_alpha = F_a1r1 / rho1_up

    # --- Divergence ---
    inv_dx = 1.0 / dx
    d_a1r1 = -(F_a1r1[1:N+1] - F_a1r1[0:N]) * inv_dx
    d_a2r2 = -(F_a2r2[1:N+1] - F_a2r2[0:N]) * inv_dx
    d_ru   = -(F_ru[1:N+1]   - F_ru[0:N])   * inv_dx
    d_rE   = -(F_rE[1:N+1]   - F_rE[0:N])   * inv_dx

    # --- Alpha equation: da1/dt = -div(F_alpha) + a1 * du/dx ---
    du_dx = (u_face[1:N+1] - u_face[0:N]) * inv_dx
    d_alpha = -(F_alpha[1:N+1] - F_alpha[0:N]) * inv_dx + a1 * du_dx

    return d_a1r1, d_a2r2, d_ru, d_rE, d_alpha


# ---------------------------------------------------------------------------
# Autograd-compatible 1st-order RHS for implicit BE
# ---------------------------------------------------------------------------

import autograd
import autograd.numpy as anp
from autograd import jacobian as _ag_jacobian


def _rhs_1st_order_ag(Q_flat, N, ph1, ph2, dx, bc_l, bc_r):
    """1st-order upwind RHS using autograd.numpy for implicit BE.

    Entire code path uses anp for exact automatic differentiation.
    Standard 5-eq pressure + mixture-T density + HLLC + Eq. 26 alpha flux.
    """
    g1, pinf1, kv1 = ph1['gamma'], ph1['pinf'], ph1['kv']
    g2, pinf2, kv2 = ph2['gamma'], ph2['pinf'], ph2['kv']
    gm1, gm2 = g1 - 1.0, g2 - 1.0

    a1r1 = Q_flat[0:N]
    a2r2 = Q_flat[N:2*N]
    ru   = Q_flat[2*N:3*N]
    rE   = Q_flat[3*N:4*N]
    a1   = Q_flat[4*N:5*N]
    a2 = 1.0 - a1

    rho = a1r1 + a2r2
    u_vel = ru / (rho + _EPS)
    rho_e = rE - 0.5 * ru * u_vel

    # Pressure (standard 5-eq linear)
    Gamma_inv = a1 / gm1 + a2 / gm2
    Pi = a1 * g1 * pinf1 / gm1 + a2 * g2 * pinf2 / gm2
    p = (rho_e - Pi) / (Gamma_inv + _EPS)

    # Temperature (mixture, no α-division)
    T_numer = a1 * (p + pinf1) / (gm1 * kv1) + a2 * (p + pinf2) / (gm2 * kv2)
    T = T_numer / (rho + _EPS)

    # Phase densities from (p, T)
    rho1 = (p + pinf1) / (gm1 * kv1 * T + _EPS)
    rho2 = (p + pinf2) / (gm2 * kv2 * T + _EPS)

    # Ghost cells (2 layers for TVD reconstruction)
    def ghost_p2(arr):
        return anp.concatenate([arr[-2:], arr, arr[:2]])

    def ghost_t2(arr):
        return anp.concatenate([anp.array([arr[0], arr[0]]), arr, anp.array([arr[-1], arr[-1]])])

    ghost2 = ghost_p2 if bc_l == 'periodic' else ghost_t2

    # TVD van Leer reconstruction (autograd-compatible)
    def _tvd_recon_ag(q_cell):
        """TVD reconstruction with van Leer limiter. Returns (qL, qR) at N+1 faces."""
        q_ext = ghost2(q_cell)  # (N+4,): q_ext[2:N+2] = q_cell
        dL = q_ext[2:N+2] - q_ext[1:N+1]   # q_i - q_{i-1}
        dR = q_ext[3:N+3] - q_ext[2:N+2]   # q_{i+1} - q_i
        # van Leer limiter: φ(r) = (r + |r|) / (1 + |r|), r = dL/dR
        r = dL / (dR + anp.sign(dR + _EPS) * _EPS)
        phi = (r + anp.abs(r)) / (1.0 + anp.abs(r) + _EPS)
        sigma = 0.5 * phi * dR  # limited slope
        qL_cell = q_cell + sigma   # right face of cell i → qL[i+1]
        qR_cell = q_cell - sigma   # left face of cell i  → qR[i]
        # Assemble face arrays
        if bc_l == 'periodic':
            qL_faces = anp.concatenate([qL_cell[-1:], qL_cell])
            qR_faces = anp.concatenate([qR_cell, qR_cell[:1]])
        else:
            qL_faces = anp.concatenate([qL_cell[:1], qL_cell])
            qR_faces = anp.concatenate([qR_cell, qR_cell[-1:]])
        return qL_faces, qR_faces

    # TVD reconstruction of primitives (ρ₁, ρ₂, u, p, α₁)
    rho1L, rho1R = _tvd_recon_ag(rho1)
    rho2L, rho2R = _tvd_recon_ag(rho2)
    uL, uR = _tvd_recon_ag(u_vel)
    pL, pR = _tvd_recon_ag(p)
    a1L, a1R = _tvd_recon_ag(a1)

    # Bounds
    a1L = anp.clip(a1L, 0.0, 1.0); a1R = anp.clip(a1R, 0.0, 1.0)
    a2L = 1.0 - a1L; a2R = 1.0 - a1R

    # Conservative face states
    a1r1_fL = a1L * rho1L; a1r1_fR = a1R * rho1R
    a2r2_fL = a2L * rho2L; a2r2_fR = a2R * rho2R
    rho_fL = a1r1_fL + a2r2_fL; rho_fR = a1r1_fR + a2r2_fR
    ru_fL = rho_fL * uL; ru_fR = rho_fR * uR

    rho_e_fL = a1L*(pL+g1*pinf1)/gm1 + a2L*(pL+g2*pinf2)/gm2
    rho_e_fR = a1R*(pR+g1*pinf1)/gm1 + a2R*(pR+g2*pinf2)/gm2
    rE_fL = rho_e_fL + 0.5*rho_fL*uL**2
    rE_fR = rho_e_fR + 0.5*rho_fR*uR**2

    # Sound speeds (Wood)
    c1L_sq = g1*(pL+pinf1)/(rho1L+_EPS)
    c1R_sq = g1*(pR+pinf1)/(rho1R+_EPS)
    c2L_sq = g2*(pL+pinf2)/(rho2L+_EPS)
    c2R_sq = g2*(pR+pinf2)/(rho2R+_EPS)
    inv_rc2_fL = a1L/(rho1L*c1L_sq+_EPS) + a2L/(rho2L*c2L_sq+_EPS)
    inv_rc2_fR = a1R/(rho1R*c1R_sq+_EPS) + a2R/(rho2R*c2R_sq+_EPS)
    c_sq_fL = 1.0/(rho_fL*inv_rc2_fL+_EPS)
    c_sq_fR = 1.0/(rho_fR*inv_rc2_fR+_EPS)
    c_fL = anp.sqrt(anp.abs(c_sq_fL) + _EPS)
    c_fR = anp.sqrt(anp.abs(c_sq_fR) + _EPS)

    # HLLC flux — full autograd using anp.where (exact, no sigmoid approximation)
    _eps_s = 1e-30  # safe epsilon for division

    S_L = anp.minimum(uL - c_fL, uR - c_fR)
    S_R = anp.maximum(uL + c_fL, uR + c_fR)

    # Contact speed S*
    num_Ss = pR - pL + rho_fL*uL*(S_L-uL) - rho_fR*uR*(S_R-uR)
    den_Ss = rho_fL*(S_L-uL) - rho_fR*(S_R-uR)
    den_Ss_safe = anp.where(anp.abs(den_Ss) > _eps_s, den_Ss, _eps_s)
    S_star = num_Ss / den_Ss_safe

    # Physical fluxes
    F_a1r1_L = a1r1_fL*uL; F_a1r1_R = a1r1_fR*uR
    F_a2r2_L = a2r2_fL*uL; F_a2r2_R = a2r2_fR*uR
    F_ru_L = ru_fL*uL+pL;  F_ru_R = ru_fR*uR+pR
    F_rE_L = (rE_fL+pL)*uL; F_rE_R = (rE_fR+pR)*uR

    # Star state coefficients: rho_K * (S_K - u_K) / (S_K - S*)
    denom_L = anp.where(anp.abs(S_L - S_star) > _eps_s, S_L - S_star, _eps_s)
    denom_R = anp.where(anp.abs(S_R - S_star) > _eps_s, S_R - S_star, _eps_s)
    cL_coeff = rho_fL * (S_L - uL) / denom_L
    cR_coeff = rho_fR * (S_R - uR) / denom_R

    Y1L = a1r1_fL / (rho_fL + _eps_s); Y2L = a2r2_fL / (rho_fL + _eps_s)
    Y1R = a1r1_fR / (rho_fR + _eps_s); Y2R = a2r2_fR / (rho_fR + _eps_s)

    EL = rE_fL / (rho_fL + _eps_s); ER = rE_fR / (rho_fR + _eps_s)
    denom_pL = anp.where(anp.abs(rho_fL*(S_L-uL)) > _eps_s, rho_fL*(S_L-uL), _eps_s)
    denom_pR = anp.where(anp.abs(rho_fR*(S_R-uR)) > _eps_s, rho_fR*(S_R-uR), _eps_s)
    EstarL = EL + (S_star - uL) * (S_star + pL / denom_pL)
    EstarR = ER + (S_star - uR) * (S_star + pR / denom_pR)

    # HLLC star-region fluxes
    hL_a1r1 = F_a1r1_L + S_L*(cL_coeff*Y1L - a1r1_fL)
    hL_a2r2 = F_a2r2_L + S_L*(cL_coeff*Y2L - a2r2_fL)
    hL_ru   = F_ru_L   + S_L*(cL_coeff*S_star - ru_fL)
    hL_rE   = F_rE_L   + S_L*(cL_coeff*EstarL - rE_fL)

    hR_a1r1 = F_a1r1_R + S_R*(cR_coeff*Y1R - a1r1_fR)
    hR_a2r2 = F_a2r2_R + S_R*(cR_coeff*Y2R - a2r2_fR)
    hR_ru   = F_ru_R   + S_R*(cR_coeff*S_star - ru_fR)
    hR_rE   = F_rE_R   + S_R*(cR_coeff*EstarR - rE_fR)

    # Region selection via anp.where (exact, autograd-supported)
    def _select4(fL, hL, hR, fR):
        return anp.where(S_L >= 0.0, fL,
               anp.where(S_star >= 0.0, hL,
               anp.where(S_R > 0.0, hR, fR)))

    F_a1r1 = _select4(F_a1r1_L, hL_a1r1, hR_a1r1, F_a1r1_R)
    F_a2r2 = _select4(F_a2r2_L, hL_a2r2, hR_a2r2, F_a2r2_R)
    F_ru   = _select4(F_ru_L,   hL_ru,   hR_ru,   F_ru_R)
    F_rE   = _select4(F_rE_L,   hL_rE,   hR_rE,   F_rE_R)

    # APEC energy flux: replace HLLC F_rE with PE-preserving decomposition
    # F_rE^APEC = ε₁·F_{a1r1} + ε₂·F_{a2r2} + ½ū²·F_ρ + p̄·ū
    e1_up = anp.where(S_star >= 0.0,
                      (pL + g1*pinf1) / (gm1 * anp.maximum(rho1L, _EPS)),
                      (pR + g1*pinf1) / (gm1 * anp.maximum(rho1R, _EPS)))
    e2_up = anp.where(S_star >= 0.0,
                      (pL + g2*pinf2) / (gm2 * anp.maximum(rho2L, _EPS)),
                      (pR + g2*pinf2) / (gm2 * anp.maximum(rho2R, _EPS)))
    p_up = anp.where(S_star >= 0.0, pL, pR)
    u_face = anp.where(S_star >= 0.0, uL, uR)
    F_rho = F_a1r1 + F_a2r2
    F_rE = e1_up * F_a1r1 + e2_up * F_a2r2 + 0.5 * u_face**2 * F_rho + p_up * u_face

    # Alpha flux (Eq. 26): F_alpha = F_a1r1 / rho1_upwind
    rho1_up = anp.where(S_star >= 0.0, rho1L, rho1R)
    rho1_up_safe = anp.maximum(rho1_up, 1e-2)
    F_alpha = F_a1r1 / rho1_up_safe

    # --- MMACM-Ex G corrections (autograd-compatible) ---
    # H_k characteristic at cell centers
    a1g = ghost2(a1)  # (N+4,)
    dL_h = a1g[2:N+2] - a1g[1:N+1]   # a_i - a_{i-1}
    dR_h = a1g[3:N+3] - a1g[2:N+2]   # a_{i+1} - a_i
    abs_dR_h = anp.abs(dR_h)
    sign_dR_h = anp.where(dR_h >= 0, 1.0, -1.0)
    r_h = dL_h * sign_dR_h / anp.maximum(abs_dR_h, _EPS)
    abs_r_h = anp.abs(r_h)
    ratio_h = (1.0 - abs_r_h) / anp.maximum(1.0 + abs_r_h, _EPS)
    H_raw = 1.0 - ratio_h ** 4
    is_intf_h = (a1 > 1e-4) * (a1 < 1.0 - 1e-4)   # use * instead of & for autograd
    is_mono_h = (dL_h * dR_h) > 0.0
    H_cell = anp.where(is_intf_h * is_mono_h, H_raw, 0.0)
    H_cell = anp.clip(H_cell, 0.0, 1.0)

    # H at faces (upwind)
    H_ext = ghost2(H_cell)  # reuse ghost2 pattern
    # ghost2 gives N+4 but we only need ng=1 → use [1:N+2] and [2:N+3]
    H_face = anp.where(S_star >= 0.0, H_ext[1:N+2], H_ext[2:N+3])

    # Downwind alpha at faces (ghost2: [g0,g1, a1[0]..a1[N-1], g2,g3])
    # Face j: left cell=a1g[j+1], right cell=a1g[j+2]
    # S*>=0 (flow L→R): downwind=right cell → a1g[j+2] = a1g[2:N+3]
    # S*<0  (flow R→L): downwind=left cell  → a1g[j+1] = a1g[1:N+2]
    a1_down = anp.where(S_star >= 0.0, a1g[2:N+3], a1g[1:N+2])

    # G_alpha = H_face * (u_face * a1_down - F_alpha)
    G_alpha = H_face * (u_face * a1_down - F_alpha)
    F_alpha = F_alpha + G_alpha

    # Conservation corrections (Eq. 27)
    rho2_up = anp.where(S_star >= 0.0, rho2L, rho2R)
    rho2_up = anp.maximum(rho2_up, _EPS)
    u_up_g = anp.where(S_star >= 0.0, uL, uR)
    E1_up = e1_up + 0.5 * u_up_g ** 2
    E2_up = e2_up + 0.5 * u_up_g ** 2
    F_a1r1 = F_a1r1 + rho1_up * G_alpha
    F_a2r2 = F_a2r2 - rho2_up * G_alpha
    F_ru   = F_ru   + (rho1_up - rho2_up) * u_up_g * G_alpha
    F_rE   = F_rE   + (rho1_up * E1_up - rho2_up * E2_up) * G_alpha

    # Divergence
    inv_dx = 1.0 / dx
    d_a1r1 = -(F_a1r1[1:N+1] - F_a1r1[0:N]) * inv_dx
    d_a2r2 = -(F_a2r2[1:N+1] - F_a2r2[0:N]) * inv_dx
    d_ru = -(F_ru[1:N+1] - F_ru[0:N]) * inv_dx
    d_rE = -(F_rE[1:N+1] - F_rE[0:N]) * inv_dx

    du_dx = (u_face[1:N+1] - u_face[0:N]) * inv_dx
    d_alpha = -(F_alpha[1:N+1] - F_alpha[0:N]) * inv_dx + a1 * du_dx

    return anp.concatenate([d_a1r1, d_a2r2, d_ru, d_rE, d_alpha])


def _pack(a1r1, a2r2, ru, rE, a1):
    """Pack 5 state arrays into flat vector (5N,)."""
    return np.concatenate([a1r1, a2r2, ru, rE, a1])


def _unpack(Q, N):
    """Unpack flat vector (5N,) into 5 state arrays."""
    return Q[0:N], Q[N:2*N], Q[2*N:3*N], Q[3*N:4*N], Q[4*N:5*N]


def _apply_bounds_flat(Q, N):
    """Apply physical bounds on packed vector."""
    Q[0:N]    = np.maximum(Q[0:N], 0.0)     # a1r1 >= 0
    Q[N:2*N]  = np.maximum(Q[N:2*N], 0.0)   # a2r2 >= 0
    Q[4*N:5*N] = np.clip(Q[4*N:5*N], 0.0, 1.0)  # a1 in [0,1]
    return Q


def _be_residual(Q, Q_old, dt, ph1, ph2, dx, bc_l, bc_r, N):
    """Backward Euler residual: R = Q - Q_old - dt * RHS(Q).
    Uses 1st-order spatial discretization for smooth Jacobian.
    """
    a1r1, a2r2, ru, rE, a1 = _unpack(Q, N)
    da1r1, da2r2, dru, drE, da1 = _rhs_1st_order(
        a1r1, a2r2, ru, rE, a1, ph1, ph2, dx, bc_l, bc_r)
    rhs_flat = _pack(da1r1, da2r2, dru, drE, da1)
    return Q - Q_old - dt * rhs_flat


def _fd_jacobian(res_func, Q, eps_fd=1e-7):
    """Dense finite-difference Jacobian with relative perturbation.

    J[i,j] = dR_i/dQ_j ≈ (R(Q + h*e_j) - R(Q)) / h
    where h = eps * max(|Q_j|, 1) for proper scaling.
    """
    R0 = res_func(Q)
    M = len(R0)
    N_vars = len(Q)
    J = np.zeros((M, N_vars))
    for j in range(N_vars):
        h = eps_fd * max(abs(Q[j]), 1.0)
        Q_p = Q.copy()
        Q_p[j] += h
        R_p = res_func(Q_p)
        J[:, j] = (R_p - R0) / h
    return J


def _fd_sparse_jacobian_1d(res_func, Q_k, N, bc_periodic=False, eps_fd=1e-7):
    """FD sparse Jacobian for 5-equation 1D system — vectorized COO assembly.

    Uses 25-color (5 eq × stride=5) graph coloring.
    bc_periodic=True: stencil wraps around at boundaries (for periodic BC).
    bc_periodic=False: transmissive/truncated stencil.
    """
    from scipy.sparse import csc_matrix
    n_eq = 5
    n_dof = n_eq * N
    R0 = np.array(res_func(Q_k), dtype=float)

    rows_all, cols_all, vals_all = [], [], []
    stride = 5
    half = 2
    stencil_offsets = np.arange(-half, half + 1)  # [-2,-1,0,1,2]

    for eq in range(n_eq):
        for offset in range(stride):
            cells = np.arange(offset, N, stride)
            n_cells = len(cells)
            if n_cells == 0:
                continue
            col_indices = eq * N + cells
            Q_pert = Q_k.copy()
            eps_vec = eps_fd * np.maximum(np.abs(Q_k[col_indices]), 1.0)
            Q_pert[col_indices] += eps_vec
            R_pert = np.array(res_func(Q_pert), dtype=float)
            dR = R_pert - R0

            # Stencil cell indices: (n_cells, stencil_size)
            if bc_periodic:
                stencil_cells = (cells[:, None] + stencil_offsets[None, :]) % N
            else:
                stencil_cells = np.clip(
                    cells[:, None] + stencil_offsets[None, :], 0, N - 1)

            # Vectorized extraction for each row equation
            for row_eq in range(n_eq):
                rows = row_eq * N + stencil_cells  # (n_cells, stencil_size)
                cols = np.broadcast_to(
                    col_indices[:, None], rows.shape)  # (n_cells, stencil_size)
                vals = dR[rows] / eps_vec[:, None]     # vectorized

                mask = np.abs(vals) > 1e-30
                rows_all.append(rows[mask].ravel())
                cols_all.append(cols[mask].ravel())
                vals_all.append(vals[mask].ravel())

    rows_cat = np.concatenate(rows_all)
    cols_cat = np.concatenate(cols_all)
    vals_cat = np.concatenate(vals_all)
    return csc_matrix((vals_cat, (rows_cat, cols_cat)), shape=(n_dof, n_dof))


def solve_implicit_be(ph1, ph2, a1r1_0, a2r2_0, ru_0, rE_0, a1_0,
                      dx, t_end, dt=None, cfl=0.5,
                      bc_l='transmissive', bc_r='transmissive',
                      max_steps=100000, max_newton=20, newton_tol=1e-8,
                      print_interval=10,
                      jacobian_method='autograd'):
    """Implicit Backward Euler solver with Newton.

    Parameters
    ----------
    jacobian_method : 'autograd' (default, dense, N<=50) or 'fd_sparse' (N>=50).
    """
    N = len(a1_0)
    a1r1 = a1r1_0.copy()
    a2r2 = a2r2_0.copy()
    ru = ru_0.copy()
    rE = rE_0.copy()
    a1 = a1_0.copy()

    t = 0.0
    step = 0
    dim = 5 * N

    if jacobian_method == 'fd_sparse':
        from scipy.sparse import eye as speye
        from scipy.sparse.linalg import spsolve
        _bc_periodic = (bc_l == 'periodic')
        _fd_sp_jac = lambda f, Q, n: _fd_sparse_jacobian_1d(
            f, Q, n, bc_periodic=_bc_periodic)

    while t < t_end and step < max_steps:
        if dt is not None:
            dt_step = min(dt, t_end - t)
        else:
            dt_step = _compute_dt(a1r1, a2r2, ru, rE, a1, ph1, ph2, dx, cfl)
            dt_step = min(dt_step, t_end - t)
        if dt_step <= 0.0:
            break

        Q_n = _pack(a1r1, a2r2, ru, rE, a1)

        Q_scale = np.ones(dim)
        for i in range(N):
            Q_scale[i]       = max(abs(a1r1[i]), 1.0)
            Q_scale[N+i]     = max(abs(a2r2[i]), 1.0)
            Q_scale[2*N+i]   = max(abs(ru[i]), 1.0)
            Q_scale[3*N+i]   = max(abs(rE[i]), 1.0)
            Q_scale[4*N+i]   = 1.0

        def rhs_scaled(Q_s):
            Q_phys = Q_s * Q_scale
            rhs_phys = _rhs_1st_order_ag(Q_phys, N, ph1, ph2, dx, bc_l, bc_r)
            return rhs_phys / Q_scale

        if jacobian_method == 'autograd':
            J_rhs_scaled_func = _ag_jacobian(rhs_scaled)

        Q_s_n = Q_n / Q_scale
        RHS_s_n = np.array(rhs_scaled(Q_s_n))
        dQ_s = dt_step * RHS_s_n  # explicit Euler predictor

        res_norm = 1.0
        for newton_iter in range(max_newton):
            Q_s_k = Q_s_n + dQ_s
            RHS_s_k = np.array(rhs_scaled(Q_s_k))
            R_s = dQ_s - dt_step * RHS_s_k
            res_norm = np.sqrt(np.mean(R_s ** 2))

            if res_norm < newton_tol:
                break

            if jacobian_method == 'fd_sparse':
                def res_eval(Q_s): return np.array(rhs_scaled(Q_s))
                J_sp = _fd_sp_jac(res_eval, Q_s_k, N)
                A_sp = speye(dim, format='csc') - dt_step * J_sp
                import warnings
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        ddQ_s = spsolve(A_sp, -R_s)
                    if not np.all(np.isfinite(ddQ_s)):
                        raise ValueError("spsolve non-finite")
                except (RuntimeError, ValueError):
                    # spsolve (SuperLU) failed: fall back to dense LAPACK
                    A_dense = A_sp.toarray()
                    try:
                        ddQ_s = np.linalg.solve(A_dense, -R_s)
                    except np.linalg.LinAlgError:
                        ddQ_s, _, _, _ = np.linalg.lstsq(A_dense, -R_s, rcond=None)
            else:
                J_s = np.array(J_rhs_scaled_func(Q_s_k))
                A_s = np.eye(dim) - dt_step * J_s
                try:
                    ddQ_s = np.linalg.solve(A_s, -R_s)
                except np.linalg.LinAlgError:
                    ddQ_s, _, _, _ = np.linalg.lstsq(A_s, -R_s, rcond=None)

            dQ_s = dQ_s + ddQ_s

        Q_new = _apply_bounds_flat((Q_s_n + dQ_s) * Q_scale, N)
        a1r1, a2r2, ru, rE, a1 = _unpack(Q_new, N)

        t += dt_step
        step += 1

        if step % print_interval == 0:
            p, u_vel, T, rho1, rho2, c1, c2, c_mix = cons_to_prim(
                a1r1, a2r2, ru, rE, a1, ph1, ph2)
            print(f"  step={step:5d}  t={t:.4e}  dt={dt_step:.3e}  "
                  f"Newton={newton_iter+1}  res={res_norm:.2e}  "
                  f"p=[{p.min():.2e},{p.max():.2e}]  u_max={np.abs(u_vel).max():.4f}")

    print(f"Done: {step} steps, t={t:.4e}")
    return t, a1r1, a2r2, ru, rE, a1


# ===========================================================================
# IMEX SOLVER: Implicit acoustics + Explicit α transport
# ===========================================================================

def solve_imex(ph1, ph2, a1r1_0, a2r2_0, ru_0, rE_0, a1_0,
               dx, t_end, dt=None, cfl=0.5,
               bc_l='transmissive', bc_r='transmissive',
               use_mmacm_ex=False, eps_intf=1e-4,
               max_steps=100000, max_newton=20, newton_tol=1e-8,
               print_interval=10):
    """IMEX solver: Implicit BE for all + Explicit α post-correction.

    Step 1: Full 5N implicit BE (handles acoustics, p/u preserved)
    Step 2: OVERWRITE α with explicit update from full _rhs (TVD + THINC-BVD)
            → sharp interface restored after each implicit step.

    The implicit step gives stable (a1r1, a2r2, ru, rE) with diffused α.
    The explicit α correction replaces the diffused α with a sharp one.
    """
    N = len(a1_0)
    a1r1 = a1r1_0.copy()
    a2r2 = a2r2_0.copy()
    ru = ru_0.copy()
    rE = rE_0.copy()
    a1 = a1_0.copy()

    t = 0.0
    step = 0
    dim = 5 * N

    while t < t_end and step < max_steps:
        if dt is not None:
            dt_step = min(dt, t_end - t)
        else:
            dt_step = _compute_dt(a1r1, a2r2, ru, rE, a1, ph1, ph2, dx, cfl)
            dt_step = min(dt_step, t_end - t)
        if dt_step <= 0.0:
            break

        # ==== Step 1: Full 5N Implicit BE (working, p/u machine precision) ====
        Q_n = _pack(a1r1, a2r2, ru, rE, a1)

        Q_scale = np.ones(dim)
        for i in range(N):
            Q_scale[i]       = max(abs(a1r1[i]), 1.0)
            Q_scale[N+i]     = max(abs(a2r2[i]), 1.0)
            Q_scale[2*N+i]   = max(abs(ru[i]), 1.0)
            Q_scale[3*N+i]   = max(abs(rE[i]), 1.0)
            Q_scale[4*N+i]   = 1.0

        def rhs_scaled(Q_s):
            Q_phys = Q_s * Q_scale
            rhs_phys = _rhs_1st_order_ag(Q_phys, N, ph1, ph2, dx, bc_l, bc_r)
            return rhs_phys / Q_scale

        J_rhs_scaled_func = _ag_jacobian(rhs_scaled)
        Q_s_n = Q_n / Q_scale

        RHS_s_n = np.array(rhs_scaled(Q_s_n))
        dQ_s = dt_step * RHS_s_n

        for newton_iter in range(max_newton):
            Q_s_k = Q_s_n + dQ_s
            RHS_s_k = np.array(rhs_scaled(Q_s_k))
            R_s = dQ_s - dt_step * RHS_s_k
            res_norm = np.sqrt(np.mean(R_s ** 2))

            if res_norm < newton_tol:
                break

            J_s = np.array(J_rhs_scaled_func(Q_s_k))
            A_s = np.eye(dim) - dt_step * J_s

            try:
                ddQ_s = np.linalg.solve(A_s, -R_s)
            except np.linalg.LinAlgError:
                ddQ_s, _, _, _ = np.linalg.lstsq(A_s, -R_s, rcond=None)

            dQ_s = dQ_s + ddQ_s

        Q_new = _apply_bounds_flat((Q_s_n + dQ_s) * Q_scale, N)
        a1r1, a2r2, ru, rE, a1_implicit = _unpack(Q_new, N)

        # ==== Step 2: Explicit α post-correction ====
        # Use explicit solver's _rhs (TVD + THINC-BVD) to get sharp α update
        # Only α is updated; (a1r1, a2r2, ru, rE) kept from implicit step.
        # The autograd RHS uses mixture-T (no α-division), so the slight
        # inconsistency between α and conservative vars is tolerated.
        _, _, _, _, da1_explicit = _rhs(
            a1r1, a2r2, ru, rE, a1, ph1, ph2, dx, bc_l, bc_r,
            use_mmacm_ex=use_mmacm_ex, eps_intf=eps_intf)

        a1 = a1 + dt_step * da1_explicit
        a1 = np.clip(a1, 0.0, 1.0)

        t += dt_step
        step += 1

        if step % print_interval == 0:
            p, u_vel, T, rho1, rho2, c1, c2, c_mix = cons_to_prim(
                a1r1, a2r2, ru, rE, a1, ph1, ph2)
            print(f"  step={step:5d}  t={t:.4e}  dt={dt_step:.3e}  "
                  f"Newton={newton_iter+1}  res={res_norm:.2e}  "
                  f"p=[{p.min():.2e},{p.max():.2e}]  u_max={np.abs(u_vel).max():.4f}")

    print(f"Done: {step} steps, t={t:.4e}")
    return t, a1r1, a2r2, ru, rE, a1


# ===========================================================================
# SEGREGATED SOLVER: 4N Implicit (α frozen) + α Explicit (TVD+THINC-BVD)
# ===========================================================================

def _pack4(a1r1, a2r2, ru, rE):
    """Pack 4 state arrays into flat vector (4N,)."""
    return np.concatenate([a1r1, a2r2, ru, rE])


def _unpack4(Q, N):
    """Unpack flat vector (4N,) into 4 state arrays."""
    return Q[0:N], Q[N:2*N], Q[2*N:3*N], Q[3*N:4*N]


def _apply_bounds_flat4(Q, N):
    """Apply physical bounds on packed 4N vector."""
    Q[0:N]    = np.maximum(Q[0:N], 0.0)     # a1r1 >= 0
    Q[N:2*N]  = np.maximum(Q[N:2*N], 0.0)   # a2r2 >= 0
    return Q


def _rhs_4N_ag(Q4_flat, N, ph1, ph2, dx, bc_l, bc_r, a1_frozen):
    """4N RHS with α₁ frozen — autograd differentiates only w.r.t. Q4.

    Q4_flat = [α₁ρ₁, α₂ρ₂, ρu, ρE] (4N vector, autograd variable)
    a1_frozen = α₁ array (numpy, treated as constant by autograd)

    Returns 4N vector: [d(α₁ρ₁)/dt, d(α₂ρ₂)/dt, d(ρu)/dt, d(ρE)/dt]
    """
    g1, pinf1, kv1 = ph1['gamma'], ph1['pinf'], ph1['kv']
    g2, pinf2, kv2 = ph2['gamma'], ph2['pinf'], ph2['kv']
    gm1, gm2 = g1 - 1.0, g2 - 1.0

    a1r1 = Q4_flat[0:N]
    a2r2 = Q4_flat[N:2*N]
    ru   = Q4_flat[2*N:3*N]
    rE   = Q4_flat[3*N:4*N]

    # α₁ is frozen (constant for autograd)
    a1 = a1_frozen
    a2 = 1.0 - a1

    rho = a1r1 + a2r2
    u_vel = ru / (rho + _EPS)
    rho_e = rE - 0.5 * ru * u_vel

    # Pressure (standard 5-eq linear, α frozen)
    Gamma_inv = a1 / gm1 + a2 / gm2
    Pi = a1 * g1 * pinf1 / gm1 + a2 * g2 * pinf2 / gm2
    p = (rho_e - Pi) / (Gamma_inv + _EPS)

    # Temperature (mixture, no α-division)
    T_numer = a1 * (p + pinf1) / (gm1 * kv1) + a2 * (p + pinf2) / (gm2 * kv2)
    T = T_numer / (rho + _EPS)

    # Phase densities from (p, T)
    rho1 = (p + pinf1) / (gm1 * kv1 * T + _EPS)
    rho2 = (p + pinf2) / (gm2 * kv2 * T + _EPS)

    # Ghost cells (2 layers for TVD reconstruction)
    def ghost_p2(arr):
        return anp.concatenate([arr[-2:], arr, arr[:2]])

    def ghost_t2(arr):
        return anp.concatenate([anp.array([arr[0], arr[0]]), arr, anp.array([arr[-1], arr[-1]])])

    ghost2 = ghost_p2 if bc_l == 'periodic' else ghost_t2

    # TVD van Leer reconstruction (autograd-compatible)
    def _tvd_recon_ag(q_cell):
        q_ext = ghost2(q_cell)
        dL = q_ext[2:N+2] - q_ext[1:N+1]
        dR = q_ext[3:N+3] - q_ext[2:N+2]
        r = dL / (dR + anp.sign(dR + _EPS) * _EPS)
        phi = (r + anp.abs(r)) / (1.0 + anp.abs(r) + _EPS)
        sigma = 0.5 * phi * dR
        qL_cell = q_cell + sigma
        qR_cell = q_cell - sigma
        if bc_l == 'periodic':
            qL_faces = anp.concatenate([qL_cell[-1:], qL_cell])
            qR_faces = anp.concatenate([qR_cell, qR_cell[:1]])
        else:
            qL_faces = anp.concatenate([qL_cell[:1], qL_cell])
            qR_faces = anp.concatenate([qR_cell, qR_cell[-1:]])
        return qL_faces, qR_faces

    # TVD reconstruction of primitives
    rho1L, rho1R = _tvd_recon_ag(rho1)
    rho2L, rho2R = _tvd_recon_ag(rho2)
    uL, uR = _tvd_recon_ag(u_vel)
    pL, pR = _tvd_recon_ag(p)

    # α reconstruction (frozen, constant for autograd)
    # Use numpy ghost/TVD for α since it's not differentiated
    def ghost_p2_np(arr):
        return np.concatenate([arr[-2:], arr, arr[:2]])

    def ghost_t2_np(arr):
        return np.concatenate([np.array([arr[0], arr[0]]), arr, np.array([arr[-1], arr[-1]])])

    ghost2_np = ghost_p2_np if bc_l == 'periodic' else ghost_t2_np

    a1_ext = ghost2_np(a1)
    dL_a = a1_ext[2:N+2] - a1_ext[1:N+1]
    dR_a = a1_ext[3:N+3] - a1_ext[2:N+2]
    r_a = dL_a / (dR_a + np.sign(dR_a + _EPS) * _EPS)
    phi_a = (r_a + np.abs(r_a)) / (1.0 + np.abs(r_a) + _EPS)
    sigma_a = 0.5 * phi_a * dR_a
    a1L_cell = a1 + sigma_a
    a1R_cell = a1 - sigma_a
    if bc_l == 'periodic':
        a1L = np.concatenate([a1L_cell[-1:], a1L_cell])
        a1R = np.concatenate([a1R_cell, a1R_cell[:1]])
    else:
        a1L = np.concatenate([a1L_cell[:1], a1L_cell])
        a1R = np.concatenate([a1R_cell, a1R_cell[-1:]])

    a1L = np.clip(a1L, 0.0, 1.0); a1R = np.clip(a1R, 0.0, 1.0)
    a2L = 1.0 - a1L; a2R = 1.0 - a1R

    # Conservative face states
    a1r1_fL = a1L * rho1L; a1r1_fR = a1R * rho1R
    a2r2_fL = a2L * rho2L; a2r2_fR = a2R * rho2R
    rho_fL = a1r1_fL + a2r2_fL; rho_fR = a1r1_fR + a2r2_fR
    ru_fL = rho_fL * uL; ru_fR = rho_fR * uR

    rho_e_fL = a1L*(pL+g1*pinf1)/gm1 + a2L*(pL+g2*pinf2)/gm2
    rho_e_fR = a1R*(pR+g1*pinf1)/gm1 + a2R*(pR+g2*pinf2)/gm2
    rE_fL = rho_e_fL + 0.5*rho_fL*uL**2
    rE_fR = rho_e_fR + 0.5*rho_fR*uR**2

    # Sound speeds (Wood)
    c1L_sq = g1*(pL+pinf1)/(rho1L+_EPS)
    c1R_sq = g1*(pR+pinf1)/(rho1R+_EPS)
    c2L_sq = g2*(pL+pinf2)/(rho2L+_EPS)
    c2R_sq = g2*(pR+pinf2)/(rho2R+_EPS)
    inv_rc2_fL = a1L/(rho1L*c1L_sq+_EPS) + a2L/(rho2L*c2L_sq+_EPS)
    inv_rc2_fR = a1R/(rho1R*c1R_sq+_EPS) + a2R/(rho2R*c2R_sq+_EPS)
    c_sq_fL = 1.0/(rho_fL*inv_rc2_fL+_EPS)
    c_sq_fR = 1.0/(rho_fR*inv_rc2_fR+_EPS)
    c_fL = anp.sqrt(anp.abs(c_sq_fL) + _EPS)
    c_fR = anp.sqrt(anp.abs(c_sq_fR) + _EPS)

    # HLLC flux — anp.where for autograd
    _eps_s = 1e-30

    S_L = anp.minimum(uL - c_fL, uR - c_fR)
    S_R = anp.maximum(uL + c_fL, uR + c_fR)

    num_Ss = pR - pL + rho_fL*uL*(S_L-uL) - rho_fR*uR*(S_R-uR)
    den_Ss = rho_fL*(S_L-uL) - rho_fR*(S_R-uR)
    den_Ss_safe = anp.where(anp.abs(den_Ss) > _eps_s, den_Ss, _eps_s)
    S_star = num_Ss / den_Ss_safe

    # Physical fluxes
    F_a1r1_L = a1r1_fL*uL; F_a1r1_R = a1r1_fR*uR
    F_a2r2_L = a2r2_fL*uL; F_a2r2_R = a2r2_fR*uR
    F_ru_L = ru_fL*uL+pL;  F_ru_R = ru_fR*uR+pR
    F_rE_L = (rE_fL+pL)*uL; F_rE_R = (rE_fR+pR)*uR

    # Star state coefficients
    denom_L = anp.where(anp.abs(S_L - S_star) > _eps_s, S_L - S_star, _eps_s)
    denom_R = anp.where(anp.abs(S_R - S_star) > _eps_s, S_R - S_star, _eps_s)
    cL_coeff = rho_fL * (S_L - uL) / denom_L
    cR_coeff = rho_fR * (S_R - uR) / denom_R

    Y1L = a1r1_fL / (rho_fL + _eps_s); Y2L = a2r2_fL / (rho_fL + _eps_s)
    Y1R = a1r1_fR / (rho_fR + _eps_s); Y2R = a2r2_fR / (rho_fR + _eps_s)

    EL = rE_fL / (rho_fL + _eps_s); ER = rE_fR / (rho_fR + _eps_s)
    denom_pL = anp.where(anp.abs(rho_fL*(S_L-uL)) > _eps_s, rho_fL*(S_L-uL), _eps_s)
    denom_pR = anp.where(anp.abs(rho_fR*(S_R-uR)) > _eps_s, rho_fR*(S_R-uR), _eps_s)
    EstarL = EL + (S_star - uL) * (S_star + pL / denom_pL)
    EstarR = ER + (S_star - uR) * (S_star + pR / denom_pR)

    # HLLC star-region fluxes
    hL_a1r1 = F_a1r1_L + S_L*(cL_coeff*Y1L - a1r1_fL)
    hL_a2r2 = F_a2r2_L + S_L*(cL_coeff*Y2L - a2r2_fL)
    hL_ru   = F_ru_L   + S_L*(cL_coeff*S_star - ru_fL)
    hL_rE   = F_rE_L   + S_L*(cL_coeff*EstarL - rE_fL)

    hR_a1r1 = F_a1r1_R + S_R*(cR_coeff*Y1R - a1r1_fR)
    hR_a2r2 = F_a2r2_R + S_R*(cR_coeff*Y2R - a2r2_fR)
    hR_ru   = F_ru_R   + S_R*(cR_coeff*S_star - ru_fR)
    hR_rE   = F_rE_R   + S_R*(cR_coeff*EstarR - rE_fR)

    def _select4(fL, hL, hR, fR):
        return anp.where(S_L >= 0.0, fL,
               anp.where(S_star >= 0.0, hL,
               anp.where(S_R > 0.0, hR, fR)))

    F_a1r1 = _select4(F_a1r1_L, hL_a1r1, hR_a1r1, F_a1r1_R)
    F_a2r2 = _select4(F_a2r2_L, hL_a2r2, hR_a2r2, F_a2r2_R)
    F_ru   = _select4(F_ru_L,   hL_ru,   hR_ru,   F_ru_R)
    F_rE   = _select4(F_rE_L,   hL_rE,   hR_rE,   F_rE_R)

    # Divergence (4 equations only, no α equation)
    inv_dx = 1.0 / dx
    d_a1r1 = -(F_a1r1[1:N+1] - F_a1r1[0:N]) * inv_dx
    d_a2r2 = -(F_a2r2[1:N+1] - F_a2r2[0:N]) * inv_dx
    d_ru = -(F_ru[1:N+1] - F_ru[0:N]) * inv_dx
    d_rE = -(F_rE[1:N+1] - F_rE[0:N]) * inv_dx

    return anp.concatenate([d_a1r1, d_a2r2, d_ru, d_rE])


def solve_segregated(ph1, ph2, a1r1_0, a2r2_0, ru_0, rE_0, a1_0,
                     dx, t_end, dt=None, cfl=0.5,
                     bc_l='transmissive', bc_r='transmissive',
                     use_mmacm_ex=False, eps_intf=1e-4,
                     max_steps=100000, max_newton=20, newton_tol=1e-8,
                     print_interval=10,
                     n_alpha_subcycle=1, cfl_alpha=0.4,
                     thinc_beta=2.0,
                     alpha_scheme='thinc_bvd',
                     jacobian_method='autograd',
                     use_compression=False, C_alpha=1.0):
    """Segregated solver: 5N implicit BE + α explicit (sub-cycled SSP-RK3).

    Step 1: Full 5N implicit BE (p/u machine precision, acoustic CFL free).
    Step 2: Extract (p, u, T) from implicit result.
    Step 3: α update via SSP-RK3 with n_alpha_subcycle sub-steps.
    Step 4: Reconstruct all conservative vars from (p, u, T, α_new).

    Parameters
    ----------
    n_alpha_subcycle : int or 'auto'
        Number of explicit α sub-steps per implicit step.
        'auto' → compute from advective CFL condition.
    cfl_alpha : float
        Max CFL for α sub-cycling (used when n_alpha_subcycle='auto').
    thinc_beta : float
        THINC sharpness parameter (used when alpha_scheme='thinc_bvd').
        β=2.0: default — best for BVD selection. Higher β (>~5) causes BVD
        to reject THINC for diffused interfaces, resulting in MORE diffusion.
    alpha_scheme : str
        'thinc_bvd' (default) — THINC-BVD reconstruction.
        'cicsam'              — CICSAM Hyper-C (Ubbink & Issa 1999, 1D pure Hyper-C).
        CICSAM requires dt_sub for the face Courant number.
    jacobian_method : str
        'autograd' (default) or 'fd_sparse'.
        fd_sparse uses 25-color graph coloring; feasible for N≥50.
    """
    g1, pinf1 = ph1['gamma'], ph1['pinf']
    g2, pinf2 = ph2['gamma'], ph2['pinf']
    gm1, gm2 = g1 - 1.0, g2 - 1.0

    N = len(a1_0)
    a1r1 = a1r1_0.copy()
    a2r2 = a2r2_0.copy()
    ru = ru_0.copy()
    rE = rE_0.copy()
    a1 = a1_0.copy()

    t = 0.0
    step = 0
    dim = 5 * N

    # fd_sparse: lazy import to avoid circular dependency
    if jacobian_method == 'fd_sparse':
        from scipy.sparse import eye as speye
        from scipy.sparse.linalg import spsolve
        _bc_periodic = (bc_l == 'periodic')
        _fd_sp_jac = lambda f, Q, n: _fd_sparse_jacobian_1d(
            f, Q, n, bc_periodic=_bc_periodic)

    while t < t_end and step < max_steps:
        if dt is not None:
            dt_step = min(dt, t_end - t)
        else:
            dt_step = _compute_dt(a1r1, a2r2, ru, rE, a1, ph1, ph2, dx, cfl)
            dt_step = min(dt_step, t_end - t)
        if dt_step <= 0.0:
            break

        # ==== Step 1: Full 5N Implicit BE ====
        a1_old = a1.copy()

        Q_n = _pack(a1r1, a2r2, ru, rE, a1)

        Q_scale = np.ones(dim)
        for i in range(N):
            Q_scale[i]       = max(abs(a1r1[i]), 1.0)
            Q_scale[N+i]     = max(abs(a2r2[i]), 1.0)
            Q_scale[2*N+i]   = max(abs(ru[i]), 1.0)
            Q_scale[3*N+i]   = max(abs(rE[i]), 1.0)
            Q_scale[4*N+i]   = 1.0

        def rhs_scaled(Q_s):
            Q_phys = Q_s * Q_scale
            rhs_phys = _rhs_1st_order_ag(Q_phys, N, ph1, ph2, dx, bc_l, bc_r)
            return rhs_phys / Q_scale

        if jacobian_method == 'autograd':
            J_func = _ag_jacobian(rhs_scaled)

        Q_s_n = Q_n / Q_scale
        RHS_s_n = np.array(rhs_scaled(Q_s_n))
        dQ_s = dt_step * RHS_s_n  # explicit Euler predictor

        res_norm = 1.0
        for newton_iter in range(max_newton):
            Q_s_k = Q_s_n + dQ_s
            RHS_s_k = np.array(rhs_scaled(Q_s_k))
            R_s = dQ_s - dt_step * RHS_s_k
            res_norm = np.sqrt(np.mean(R_s ** 2))

            if res_norm < newton_tol:
                break

            if jacobian_method == 'fd_sparse':
                def res_eval(Q_s): return np.array(rhs_scaled(Q_s))
                J_sp = _fd_sp_jac(res_eval, Q_s_k, N)
                A_sp = speye(dim, format='csc') - dt_step * J_sp
                import warnings
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        ddQ_s = spsolve(A_sp, -R_s)
                    if not np.all(np.isfinite(ddQ_s)):
                        raise ValueError("spsolve non-finite")
                except (RuntimeError, ValueError):
                    # spsolve (SuperLU) failed: fall back to dense LAPACK
                    A_dense = A_sp.toarray()
                    try:
                        ddQ_s = np.linalg.solve(A_dense, -R_s)
                    except np.linalg.LinAlgError:
                        ddQ_s, _, _, _ = np.linalg.lstsq(A_dense, -R_s, rcond=None)
            else:
                J_s = np.array(J_func(Q_s_k))
                A_s = np.eye(dim) - dt_step * J_s
                try:
                    ddQ_s = np.linalg.solve(A_s, -R_s)
                except np.linalg.LinAlgError:
                    ddQ_s, _, _, _ = np.linalg.lstsq(A_s, -R_s, rcond=None)

            dQ_s = dQ_s + ddQ_s

        Q_new = _apply_bounds_flat((Q_s_n + dQ_s) * Q_scale, N)
        a1r1, a2r2, ru, rE, a1_impl = _unpack(Q_new, N)

        # ==== Step 2: Extract (p, u, T) from implicit result ====
        p_impl, u_impl, T_impl, _, _, _, _, _ = cons_to_prim(
            a1r1, a2r2, ru, rE, a1_impl, ph1, ph2)

        # ==== Step 3: α SSP-RK3 with sub-cycling ====
        kv1, kv2 = ph1['kv'], ph2['kv']

        def _make_cons_from_prim(a1_cur):
            """Reconstruct conservative vars from (p_impl, u_impl, T_impl, α)."""
            a2_cur = 1.0 - a1_cur
            r1 = (p_impl + pinf1) / np.maximum(gm1 * kv1 * T_impl, _EPS)
            r2 = (p_impl + pinf2) / np.maximum(gm2 * kv2 * T_impl, _EPS)
            ar1 = a1_cur * r1
            ar2 = a2_cur * r2
            rho_cur = ar1 + ar2
            ru_cur = rho_cur * u_impl
            re_cur = a1_cur*(p_impl+g1*pinf1)/gm1 + a2_cur*(p_impl+g2*pinf2)/gm2
            rE_cur = re_cur + 0.5 * rho_cur * u_impl**2
            return ar1, ar2, ru_cur, rE_cur

        # Compute number of sub-steps (before defining _alpha_adv_rhs
        # so CICSAM can capture dt_sub for the Courant number)
        if n_alpha_subcycle == 'auto':
            u_max = np.abs(u_impl).max() + _EPS
            dt_alpha_max = cfl_alpha * dx / u_max
            n_sub = max(1, int(np.ceil(dt_step / dt_alpha_max)))
        else:
            n_sub = max(1, int(n_alpha_subcycle))

        dt_sub = dt_step / n_sub

        # Pre-compute face velocities (same for all sub-steps since u_impl is fixed)
        _u_g = _ghost(u_impl, bc_l, bc_r)             # N+2
        _u_face = 0.5 * (_u_g[:-1] + _u_g[1:])        # N+1

        def _alpha_adv_rhs(a1_cur):
            """α advection RHS: scheme selected by alpha_scheme parameter.

            THINC-BVD: BVD selects between TVD and THINC per cell.
              β=2.0 optimal — higher β causes BVD to reject THINC.
            CICSAM Hyper-C: ñ_f = min(ñ_D/Co_f, 1), most compressive
              scheme in NVD; Co_f = |u|·dt_sub/dx.
            """
            if alpha_scheme in ('cicsam', 'stacs', 'mstacs', 'saish'):
                # NVD-based schemes (all use _nvd_face with different CDS)
                cds_map = {'cicsam': 'hyper_c', 'stacs': 'superbee',
                           'mstacs': 'mstacs', 'saish': 'saish'}
                alpha_face = _nvd_face(
                    a1_cur, _u_face, dt_sub, dx, bc_l, bc_r,
                    cds=cds_map[alpha_scheme])
            else:  # thinc_bvd (default)
                a1_L, a1_R = _thinc_bvd_reconstruct(
                    a1_cur, bc_l=bc_l, bc_r=bc_r, beta=thinc_beta)
                alpha_face = np.where(_u_face >= 0.0, a1_L, a1_R)

            F_alpha = alpha_face * _u_face

            # Compression term (OpenFOAM-style anti-diffusion)
            if use_compression:
                F_comp_raw = _compression_flux(a1_cur, _u_face, bc_l, bc_r, C_alpha)
                F_comp = _zalesak_fct_limit(F_comp_raw, a1_cur, dx, dt_sub, bc_l, bc_r)
                F_alpha = F_alpha + F_comp

            return -(F_alpha[1:] - F_alpha[:-1]) / dx    # N

        a1_cur = a1_old.copy()

        # Sub-cycle loop: SSP-RK3 at each sub-step
        for _ in range(n_sub):
            # Stage 1
            da1_1 = _alpha_adv_rhs(a1_cur)
            a1_s1 = np.clip(a1_cur + dt_sub * da1_1, _EPS, 1.0 - _EPS)
            # Stage 2
            da1_2 = _alpha_adv_rhs(a1_s1)
            a1_s2 = np.clip(0.75*a1_cur + 0.25*(a1_s1 + dt_sub*da1_2),
                             _EPS, 1.0 - _EPS)
            # Stage 3
            da1_3 = _alpha_adv_rhs(a1_s2)
            a1_cur = np.clip((1.0/3.0)*a1_cur + (2.0/3.0)*(a1_s2 + dt_sub*da1_3),
                              _EPS, 1.0 - _EPS)

        a1 = a1_cur

        # ==== Step 4: Reconstruct ALL conservative vars from (p, u, T, α_new) ====
        a1r1, a2r2, ru, rE = _make_cons_from_prim(a1)

        t += dt_step
        step += 1

        if step % print_interval == 0:
            p, u_vel, T, rho1, rho2, c1, c2, c_mix = cons_to_prim(
                a1r1, a2r2, ru, rE, a1, ph1, ph2)
            print(f"  step={step:5d}  t={t:.4e}  dt={dt_step:.3e}  "
                  f"Newton={newton_iter+1}  res={res_norm:.2e}  "
                  f"p=[{p.min():.2e},{p.max():.2e}]  u_max={np.abs(u_vel).max():.4f}")

    print(f"Done: {step} steps, t={t:.4e}")
    return t, a1r1, a2r2, ru, rE, a1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import os
    os.makedirs('results', exist_ok=True)

    x, t_final, a1r1, a2r2, ru, rE, a1, ph1, ph2 = run_phase2_1(
        N=200, cfl=0.4, t_end=8.0e-4,
        use_mmacm_ex=True, print_interval=50)

    _plot_phase2_1(x, t_final, a1r1, a2r2, ru, rE, a1, ph1, ph2,
                   save_path='results/phase2_1_mmacm_ex_paper.png')
