# solver/denner_1d/assembly_5eq_ad.py
# Ref: Solver_fully_coupled_5_equation.md — §4, §5, §14
#      CLAUDE.md — 5-equation model
#
# Autograd-compatible residual for 5-equation conservative backward Euler solver.
#
# Conservative variables per cell:
#   U = {α₁ρ₁,  α₂ρ₂,  ρu,  ρE,  α₁}   (5 DOFs per cell, 5N total)
#
# Governing equations (BDF1 residual form):
#   R₁ = (α₁ρ₁)^{n+1}/dt - (α₁ρ₁)^n/dt + ∂(α₁ρ₁·u_f)/∂x = 0
#   R₂ = (α₂ρ₂)^{n+1}/dt - (α₂ρ₂)^n/dt + ∂(α₂ρ₂·u_f)/∂x = 0
#   R₃ = (ρu)^{n+1}/dt   - (ρu)^n/dt   + ∂(ρu·u_f + p)/∂x = 0
#   R₄ = (ρE)^{n+1}/dt   - (ρE)^n/dt   + ∂((ρE+p)·u_f)/∂x = 0
#   R₅ = α₁^{n+1}/dt     - α₁^n/dt     + ∂(α₁·u_f)/∂x
#        - (α₁+K)·∂u_f/∂x = 0
#
# Jacobian strategy:
#   - Temporal block:    1/dt · I  (exact, always)
#   - Spatial Jacobian:  computed via autograd.jacobian(residual_ad)(U_k)
#   - CICSAM γ_f / face topology:  LAGGED (outer Picard, frozen inside Newton)
#   - Face velocity θ:   LAGGED (recomputed every outer Picard, frozen inside Newton)
#   - K coefficient:     LAGGED (recomputed every outer Picard)
#   - EOS p(U):          autograd-differentiable via primitive variable recovery
#
# EOS inversion strategy for autograd:
#   Given U = {α₁ρ₁, α₂ρ₂, ρu, ρE, α₁}, primitive recovery is:
#     ρ = α₁ρ₁ + α₂ρ₂
#     u = ρu / ρ
#     ρ₁ = α₁ρ₁ / α₁  (for pure-phase density)
#     ρ₂ = α₂ρ₂ / α₂
#     e = (ρE - ½ρu²) / ρ  (mixture specific internal energy)
#   Then p and T from 2x2 Newton (NOT autograd-differentiated — lagged).
#   Only the EOS evaluation p(p_lag, T_lag) itself is lagged.
#   The autograd-differentiable part is the flux assembly with LAGGED p, theta.
#
# Design: since CICSAM γ_f, θ, K, and p are all lagged (outer Picard), the
# autograd residual function receives U as its live variable, and uses:
#   - theta_lag (N+1,)  — frozen face velocity
#   - p_lag (N,)        — frozen cell pressure
#   - K_lag (N,)        — frozen K coefficient
#   - alpha_f_lag (N+1,) — frozen CICSAM face α₁
# These are captured as closure variables. The Jacobian then reflects only
# the U-dependence of the mass, momentum, energy, and alpha time terms plus
# the upwind flux dependence on the conserved variables.

import numpy as np

try:
    import autograd.numpy as anp
    from autograd import jacobian as ad_jacobian
    _AUTOGRAD_AVAILABLE = True
except ImportError:
    _AUTOGRAD_AVAILABLE = False
    anp = np   # fallback: use numpy (Jacobian won't be available)

from .eos.eos_class import create_eos


# ---------------------------------------------------------------------------
# Autograd-compatible EOS wrappers
# ---------------------------------------------------------------------------

def _nasg_rho_anp(p, T, gamma, pinf, b, kv):
    """NASG density: ρ = (p+p∞) / [κᵥ·T·(γ-1) + b·(p+p∞)].
    Uses anp operations for autograd compatibility."""
    gm1 = gamma - 1.0
    A = kv * T * gm1 + b * (p + pinf) + 1e-300
    return (p + pinf) / A


def _nasg_h_anp(p, T, gamma, pinf, b, kv, eta):
    """NASG specific enthalpy: h = γ·κᵥ·T + b·p + η."""
    return gamma * kv * T + b * p + eta


def _nasg_evol_anp(p, T, gamma, pinf, b, kv, eta):
    """NASG volumetric internal energy: ρe = ρh − p."""
    rho = _nasg_rho_anp(p, T, gamma, pinf, b, kv)
    h   = _nasg_h_anp(p, T, gamma, pinf, b, kv, eta)
    return rho * h - p


def _stiffened_rho_anp(p, T, gamma, pinf, cv):
    """Stiffened Gas density: ρ = (p+p∞) / [(γ-1)·cv·T]."""
    gm1 = gamma - 1.0
    R   = gm1 * cv
    return (p + pinf) / (R * T + 1e-300)


def _stiffened_h_anp(p, T, gamma, pinf, cv, q):
    """Stiffened Gas specific enthalpy: h = γ·cv·T + q."""
    return gamma * cv * T + q


def _stiffened_evol_anp(p, T, gamma, pinf, cv, q):
    """Stiffened Gas volumetric internal energy: ρe = ρh − p."""
    rho = _stiffened_rho_anp(p, T, gamma, pinf, cv)
    h   = _stiffened_h_anp(p, T, gamma, pinf, cv, q)
    return rho * h - p


def _get_eos_params(ph):
    """Extract EOS type and parameters from phase dict or EOS object."""
    from .eos.eos_class import NasgEOS, StiffenedGasEOS, EOS
    if isinstance(ph, NasgEOS):
        return ('nasg', ph.gamma, ph.pinf, ph.b, ph.kv, ph.eta)
    if isinstance(ph, StiffenedGasEOS):
        return ('stiffened', ph.gamma, ph.pinf, ph.cv, ph.q)
    if isinstance(ph, EOS):
        # Generic EOS — fall back to numerical
        return ('generic', ph)
    # dict
    eos_type = ph.get('eos_type', ph.get('type', 'nasg'))
    if eos_type == 'stiffened':
        return ('stiffened',
                float(ph['gamma']),
                float(ph.get('pinf', ph.get('Pi', 0.0))),
                float(ph['cv']),
                float(ph.get('q', 0.0)))
    return ('nasg',
            float(ph['gamma']),
            float(ph.get('pinf', ph.get('p_inf', 0.0))),
            float(ph.get('b',    ph.get('b_covolume', 0.0))),
            float(ph.get('kv',   ph.get('cv', ph.get('c_v', 1.0)))),
            float(ph.get('eta',  ph.get('q', 0.0))))


def _eos_rho_anp(p, T, params):
    """EOS density using anp — autograd-differentiable."""
    kind = params[0]
    if kind == 'nasg':
        _, gamma, pinf, b, kv, eta = params
        return _nasg_rho_anp(p, T, gamma, pinf, b, kv)
    elif kind == 'stiffened':
        _, gamma, pinf, cv, q = params
        return _stiffened_rho_anp(p, T, gamma, pinf, cv)
    else:
        # Generic: call EOS object directly (breaks autograd graph)
        eos = params[1]
        return eos.rho(p, T)


def _eos_evol_anp(p, T, params):
    """EOS volumetric internal energy ρe using anp — autograd-differentiable."""
    kind = params[0]
    if kind == 'nasg':
        _, gamma, pinf, b, kv, eta = params
        return _nasg_evol_anp(p, T, gamma, pinf, b, kv, eta)
    elif kind == 'stiffened':
        _, gamma, pinf, cv, q = params
        return _stiffened_evol_anp(p, T, gamma, pinf, cv, q)
    else:
        eos = params[1]
        return eos.e_vol(p, T)


# ---------------------------------------------------------------------------
# 2nd-order TVD reconstruction (autograd-compatible)
# ---------------------------------------------------------------------------
# Denner 2018 (JCP 348) Eq. (9) / ACID 2 (JCP 367) Eq. (40):
#   phi_f = phi_U + (xi_f / 2) (phi_D - phi_U)
# with the minmod flux limiter xi(r) acting on the upwind slope ratio
#   r = (phi_U - phi_UU) / (phi_D - phi_U)
# This is the canonical 2nd-order TVD reconstruction.  We express it
# directly in the symmetric two-slope form
#   phi_f = phi_U + 0.5 * minmod(phi_U - phi_UU, phi_D - phi_U)
# which is equivalent for the minmod limiter and avoids dividing by a
# potentially tiny denominator inside the autograd graph.


def _minmod_anp(a, b):
    """Two-argument symmetric minmod, autograd-compatible.

    Returns 0 if a and b have opposite signs; otherwise the smaller-
    magnitude value with the common sign.
    """
    # anp.sign returns -1, 0, +1.  Use .where for the sign-agreement test
    # so the autograd graph stays smooth.
    same_sign = (a * b) > 0
    sign = anp.where(a >= 0, 1.0, -1.0)
    mag = anp.minimum(anp.abs(a), anp.abs(b))
    return anp.where(same_sign, sign * mag, 0.0)


def _recon_minmod_anp(phi_UU, phi_U, phi_D):
    """2nd-order TVD reconstruction at a face using the minmod limiter.

    Inputs may be scalars or arrays.  Returns phi_face = phi_U + 0.5 * slope
    where slope = minmod(phi_U - phi_UU, phi_D - phi_U).
    """
    d_back = phi_U - phi_UU
    d_forw = phi_D - phi_U
    return phi_U + 0.5 * _minmod_anp(d_back, d_forw)


# ---------------------------------------------------------------------------
# ACID energy flux helper (standard numpy, for conservative residual)
# ---------------------------------------------------------------------------

def _acid_rH_face(a1_cell, p_up, T_up, u_up, ph1_params, ph2_params):
    """ACID total enthalpy density at face: ρH = Σ α_cell · ρ_k(p_up,T_up) · H_k.

    Uses cell i's α₁ (not upwind) with upwind (p,T,u) for PE-preserving
    energy flux. At uniform (p,T,u), both face evaluations are identical
    and the net energy flux is exactly zero regardless of the α₁ field.

    Parameters
    ----------
    a1_cell   : float  cell i's volume fraction of phase 1
    p_up, T_up, u_up : float  upwind pressure, temperature, velocity
    ph1_params, ph2_params : tuple  from _get_eos_params()

    Returns
    -------
    rH_acid : float  ρH = ρE_acid + p  (total enthalpy density at face)
    """
    a2_cell = 1.0 - a1_cell
    r1 = float(_eos_rho_anp(float(p_up), float(T_up), ph1_params))
    r2 = float(_eos_rho_anp(float(p_up), float(T_up), ph2_params))
    evol1 = float(_eos_evol_anp(float(p_up), float(T_up), ph1_params))
    evol2 = float(_eos_evol_anp(float(p_up), float(T_up), ph2_params))
    rho_acid = a1_cell * r1 + a2_cell * r2
    rE_acid  = a1_cell * evol1 + a2_cell * evol2 + 0.5 * rho_acid * float(u_up)**2
    return rE_acid + float(p_up)   # = ρH_acid


# ---------------------------------------------------------------------------
# Ghost cell extension — autograd-compatible (array concatenation)
# ---------------------------------------------------------------------------

def _ghost_extend_anp(arr, bc_l, bc_r, n_ghost=1):
    """Extend 1D array with ghost cells using array ops (autograd-compatible).

    Uses anp.concatenate and indexing only — avoids np.broadcast_to which
    may break the autograd graph.

    Parameters
    ----------
    arr    : array (N,)
    bc_l   : str  'periodic' | 'transmissive' | 'wall'
    bc_r   : str
    n_ghost: int  number of ghost layers each side

    Returns
    -------
    arr_ext : array (N + 2*n_ghost,)
    """
    N = arr.shape[0]
    ng = n_ghost

    if bc_l == 'periodic':
        # last ng cells wrap around
        left_ghosts = arr[N - ng:]
    else:
        # transmissive/wall: repeat first cell ng times
        # arr[:1] is shape (1,); repeat by multiplying elementwise with ones
        left_ghosts = arr[:1] * anp.ones(ng)

    if bc_r == 'periodic':
        right_ghosts = arr[:ng]
    else:
        right_ghosts = arr[-1:] * anp.ones(ng)

    return anp.concatenate([left_ghosts, arr, right_ghosts])


def _ghost_extend_vel_anp(u, bc_l, bc_r, n_ghost=1):
    """Ghost cell extension for velocity (wall BC: negate)."""
    N = u.shape[0]
    ng = n_ghost

    if bc_l == 'periodic':
        left_ghosts = u[N - ng:]
    elif bc_l == 'wall':
        left_ghosts = -u[:1] * anp.ones(ng)
    else:
        left_ghosts = u[:1] * anp.ones(ng)

    if bc_r == 'periodic':
        right_ghosts = u[:ng]
    elif bc_r == 'wall':
        right_ghosts = -u[-1:] * anp.ones(ng)
    else:
        right_ghosts = u[-1:] * anp.ones(ng)

    return anp.concatenate([left_ghosts, u, right_ghosts])


# ---------------------------------------------------------------------------
# Core autograd-compatible residual (primitive variable form)
# ---------------------------------------------------------------------------

def make_residual_prim_ad(
        p_n, u_n, T_n, a1_n,
        N, dx, dt,
        ph1_params, ph2_params,
        bc_l, bc_r,
        theta_lag,
        K_lag):
    """Build autograd-compatible residual closure for primitive-variable 5-eq.

    The residual closure R(W) accepts W = concat([p, u, T, α₁])  (4N,)
    and returns R  (4N,).

    Closure captures (frozen):
        p_n, u_n, T_n, a1_n  — old-time primitives
        theta_lag             — MWI face velocity (N+1,), lagged
        K_lag                 — Wood K coefficient (N,), lagged

    Parameters
    ----------
    p_n, u_n, T_n, a1_n : ndarray (N,)  old-time primitives
    N, dx, dt            : int, float, float
    ph1_params, ph2_params : tuple from _get_eos_params()
    bc_l, bc_r           : str
    theta_lag            : ndarray (N+1,)  MWI face velocity (lagged)
    K_lag                : ndarray (N,)   Wood K coefficient (lagged)

    Returns
    -------
    residual_func : callable  R(W) → ndarray (4N,)
    """
    # Convert old-time to anp-compatible arrays (constants in residual)
    _p_n  = anp.array(p_n)
    _u_n  = anp.array(u_n)
    _T_n  = anp.array(T_n)
    _a1_n = anp.array(a1_n)
    _theta = anp.array(theta_lag)
    _K     = anp.array(K_lag)

    # ---------- Physics-based residual normalization scales ----------
    # The 4 residual blocks (mass1, momentum, energy, alpha) carry
    # natural physical magnitudes that differ by ~10 orders of magnitude
    # for air at 1 atm.  Without normalization the LM-regularized linear
    # solve produces a dW whose mass/energy components are O(1) but
    # whose alpha component is O(1e-4), so the volume fraction barely
    # advects between Newton iterations even when the iterate has
    # otherwise converged on the PE manifold.
    #
    # We rescale each residual block by an old-time, cell-wise reference
    # quantity:
    #   R_m1   /= rho_n                    (phase-1 mass density)
    #   R_mom  /= rho_n * c_n              (momentum flux scale)
    #   R_en   /= rho_n * c_n^2 + p_n      (energy flux scale)
    #   R_a1   /= 1.0                      (already O(1))
    #
    # Scales are *lagged* (from W_n only), so they are constants for
    # the autograd graph and add no Jacobian cost.  The Newton system
    # is mathematically equivalent up to a fixed row scaling; the
    # converged solution is unchanged, but the descent direction at
    # each iteration now puts equal weight on the alpha equation.
    _g1, _pinf1 = float(ph1_params[1]), float(ph1_params[2])
    _g2, _pinf2 = float(ph2_params[1]), float(ph2_params[2])
    _rho1_n_phys = _eos_rho_anp(_p_n, _T_n, ph1_params)
    _rho2_n_phys = _eos_rho_anp(_p_n, _T_n, ph2_params)
    _a1_n_safe   = anp.maximum(anp.minimum(_a1_n, 1.0 - 1e-10), 1e-10)
    _rho_n_phys  = (_a1_n_safe * _rho1_n_phys
                    + (1.0 - _a1_n_safe) * _rho2_n_phys)
    _c1_sq = anp.maximum(_g1 * (_p_n + _pinf1) / (_rho1_n_phys + 1e-300), 1.0)
    _c2_sq = anp.maximum(_g2 * (_p_n + _pinf2) / (_rho2_n_phys + 1e-300), 1.0)
    _c_n_phys = anp.sqrt(anp.maximum(_c1_sq, _c2_sq))

    _scale_m1  = anp.maximum(_rho_n_phys, 1e-3)
    _scale_mom = anp.maximum(_rho_n_phys * _c_n_phys, 1.0)
    _scale_en  = anp.maximum(_rho_n_phys * _c_n_phys * _c_n_phys + _p_n, 1.0)
    _scale_a1  = anp.ones(N)

    # Precompute integer index arrays for ghost-extended array access.
    # We use ng=2 ghost layers so 2nd-order TVD reconstruction at every
    # face has a full 3-cell stencil (phi_UU, phi_U, phi_D) available.
    # Extended size = N + 2*ng = N + 4, indices 0..N+3.
    # Cell i maps to ext index i + ng = i + 2.
    #
    # Right face of cell i is f_R = i + 1/2 with theta index f_R = i+1.
    #   If theta[f_R] >= 0:  upwind=cell i, downwind=cell i+1
    #     stencil ext indices:  UU=i+1, U=i+2, D=i+3
    #   If theta[f_R] <  0:  upwind=cell i+1, downwind=cell i
    #     stencil ext indices:  UU=i+4, U=i+3, D=i+2
    #
    # Left face of cell i is f_L = i - 1/2 with theta index f_L = i.
    #   If theta[f_L] >= 0:  upwind=cell i-1, downwind=cell i
    #     stencil ext indices:  UU=i,   U=i+1, D=i+2
    #   If theta[f_L] <  0:  upwind=cell i,   downwind=cell i-1
    #     stencil ext indices:  UU=i+3, U=i+2, D=i+1
    _i_arr     = np.arange(N, dtype=int)       # 0..N-1

    # Right face — positive theta branch (upwind = cell i)
    _idx_R_UU_p = _i_arr + 1
    _idx_R_U_p  = _i_arr + 2
    _idx_R_D_p  = _i_arr + 3
    # Right face — negative theta branch (upwind = cell i+1)
    _idx_R_UU_n = _i_arr + 4
    _idx_R_U_n  = _i_arr + 3
    _idx_R_D_n  = _i_arr + 2

    # Left face — positive theta branch (upwind = cell i-1)
    _idx_L_UU_p = _i_arr
    _idx_L_U_p  = _i_arr + 1
    _idx_L_D_p  = _i_arr + 2
    # Left face — negative theta branch (upwind = cell i)
    _idx_L_UU_n = _i_arr + 3
    _idx_L_U_n  = _i_arr + 2
    _idx_L_D_n  = _i_arr + 1

    # Face indices in the (N+1,) theta array.
    _idx_R_face = _i_arr + 1                   # right face of cell i (= i+1/2)
    _idx_L_face = _i_arr                       # left  face of cell i (= i-1/2)

    # Backwards-compatible index aliases used further down for cell-centred
    # arithmetic averages of pressure on the face (still safe with ng=2:
    # the immediate face neighbours are still ext idx i+2, i+3).
    _idx_R_L   = _i_arr + 2                    # cell i  (left  of right face)
    _idx_R_R   = _i_arr + 3                    # cell i+1 (right of right face)
    _idx_L_L   = _i_arr + 1                    # cell i-1 (left  of left face)
    _idx_L_R   = _i_arr + 2                    # cell i  (right of left face)

    # ----------------- Lagged minmod slope correction ----------------
    # Deferred-correction (ACID 2 Eq. 47 / Denner 2018 Eq. 9) form:
    #   phi_f = phi_U + 0.5 * slope_lagged
    # where slope_lagged = minmod(phi_U - phi_UU, phi_D - phi_U) is
    # evaluated *only on the old-time field W_n* using plain numpy.
    # This keeps the Newton residual function smooth in W^(n+1) (the
    # only autograd variable) while still delivering 2nd-order TVD
    # accuracy in space.  An additional outer Picard iteration refreshes
    # the slope using the latest W^(n+1) estimate; the inner Newton sees
    # frozen slopes.
    #
    # Empirically (FD-vs-autograd Jacobian check), the previous in-graph
    # minmod produced rel error 33% on Jacobian entries because the two
    # nested anp.where calls deliver wrong gradients at the sign-change
    # surfaces.  Lagging the slope eliminates that problem entirely.
    def _np_minmod(a, b):
        sign_a = np.sign(a); sign_b = np.sign(b)
        same = (sign_a * sign_b) > 0
        mag = np.minimum(np.abs(a), np.abs(b))
        return np.where(same, sign_a * mag, 0.0)

    def _np_lagged_slope(field_n, idx_UU, idx_U, idx_D, bc_l_local, bc_r_local,
                          is_vel=False):
        # Build a numpy ext array (ng=2) of the old-time field.
        N_ = field_n.shape[0]
        ng_ = 2
        if bc_l_local == 'periodic':
            left_g = field_n[N_ - ng_:]
        elif bc_l_local == 'wall' and is_vel:
            left_g = -field_n[:1] * np.ones(ng_)
        else:
            left_g = field_n[:1] * np.ones(ng_)
        if bc_r_local == 'periodic':
            right_g = field_n[:ng_]
        elif bc_r_local == 'wall' and is_vel:
            right_g = -field_n[-1:] * np.ones(ng_)
        else:
            right_g = field_n[-1:] * np.ones(ng_)
        ext = np.concatenate([left_g, field_n, right_g])
        d_back = ext[idx_U]  - ext[idx_UU]
        d_forw = ext[idx_D]  - ext[idx_U]
        return _np_minmod(d_back, d_forw)

    # Lagged slopes for each (face × theta-sign × primitive) combination.
    # field_n is taken at the *current Newton outer iterate* via theta-lagged
    # closure variables (_p_n, _u_n, _T_n, _a1_n).  For the first inner
    # iteration these equal W_n; subsequent outer-Picard refreshes can
    # override them by rebuilding the residual closure.
    def _slopes_for(field_n, bc_l_, bc_r_, is_vel_=False):
        return {
            'R_p': _np_lagged_slope(field_n, _idx_R_UU_p, _idx_R_U_p, _idx_R_D_p,
                                     bc_l_, bc_r_, is_vel_),
            'R_n': _np_lagged_slope(field_n, _idx_R_UU_n, _idx_R_U_n, _idx_R_D_n,
                                     bc_l_, bc_r_, is_vel_),
            'L_p': _np_lagged_slope(field_n, _idx_L_UU_p, _idx_L_U_p, _idx_L_D_p,
                                     bc_l_, bc_r_, is_vel_),
            'L_n': _np_lagged_slope(field_n, _idx_L_UU_n, _idx_L_U_n, _idx_L_D_n,
                                     bc_l_, bc_r_, is_vel_),
        }

    _slope_p  = _slopes_for(p_n,  bc_l, bc_r, False)
    _slope_T  = _slopes_for(T_n,  bc_l, bc_r, False)
    _slope_u  = _slopes_for(u_n,  bc_l, bc_r, True)
    _slope_a1 = _slopes_for(a1_n, bc_l, bc_r, False)

    # Express the four lagged slopes as theta-blended numpy constants
    # so the in-graph reconstruction reduces to a smooth affine step.
    _tR_pos = (np.asarray(theta_lag)[_i_arr + 1] >= 0).astype(float)
    _tL_pos = (np.asarray(theta_lag)[_i_arr])     >= 0
    _tL_pos = _tL_pos.astype(float)

    def _blend(slope_dict, side):
        # side is 'R' or 'L'
        sp = slope_dict[f'{side}_p']; sn = slope_dict[f'{side}_n']
        flag = _tR_pos if side == 'R' else _tL_pos
        return flag * sp + (1.0 - flag) * sn

    _slope_p_R  = anp.array(_blend(_slope_p,  'R'))
    _slope_p_L  = anp.array(_blend(_slope_p,  'L'))
    _slope_T_R  = anp.array(_blend(_slope_T,  'R'))
    _slope_T_L  = anp.array(_blend(_slope_T,  'L'))
    _slope_u_R  = anp.array(_blend(_slope_u,  'R'))
    _slope_u_L  = anp.array(_blend(_slope_u,  'L'))
    _slope_a1_R = anp.array(_blend(_slope_a1, 'R'))
    _slope_a1_L = anp.array(_blend(_slope_a1, 'L'))

    def residual_func(W):
        """Autograd-compatible residual R(W) for 5-eq primitive Newton.

        W : array (4N,)  packed as [p(0..N-1) | u(N..2N-1) | T(2N..3N-1) | α₁(3N..4N-1)]

        Returns R : array (4N,)
        Block ordering:
          R[0*N : 1*N] = R_mass1   (phase-1 mass)
          R[1*N : 2*N] = R_mom     (mixture momentum)
          R[2*N : 3*N] = R_energy  (total energy)
          R[3*N : 4*N] = R_alpha   (volume fraction)
        """
        p_k  = W[0*N:1*N]
        u_k  = W[1*N:2*N]
        T_k  = W[2*N:3*N]
        a1_k = W[3*N:4*N]

        # α₁ bounds: do NOT clamp inside residual — kills Jacobian gradient.
        # Bounds are enforced in _enforce_bounds() after each Newton trial step.
        a2_k  = 1.0 - a1_k
        _a1_n_c = anp.maximum(anp.minimum(_a1_n, 1.0 - 1e-10), 1e-10)
        _a2_n_c = 1.0 - _a1_n_c

        # --------------- Cell-centre EOS evaluations ---------------
        rho1_k = _eos_rho_anp(p_k, T_k, ph1_params)
        rho2_k = _eos_rho_anp(p_k, T_k, ph2_params)
        rho_k  = a1_k * rho1_k + a2_k * rho2_k

        rho1_n = _eos_rho_anp(_p_n, _T_n, ph1_params)
        rho2_n = _eos_rho_anp(_p_n, _T_n, ph2_params)
        rho_n  = _a1_n_c * rho1_n + _a2_n_c * rho2_n

        evol1_k = _eos_evol_anp(p_k, T_k, ph1_params)
        evol2_k = _eos_evol_anp(p_k, T_k, ph2_params)
        evol1_n = _eos_evol_anp(_p_n, _T_n, ph1_params)
        evol2_n = _eos_evol_anp(_p_n, _T_n, ph2_params)

        # Total energy density  ρE = Σ αₖρₖeₖ + ½ρu²
        rE_k = a1_k * evol1_k + a2_k * evol2_k + 0.5 * rho_k * u_k**2
        rE_n = _a1_n_c * evol1_n + _a2_n_c * evol2_n + 0.5 * rho_n * _u_n**2

        # Phase-1 partial mass  α₁ρ₁
        a1r1_k = a1_k * rho1_k
        a1r1_n = _a1_n_c * rho1_n

        # Mixture momentum  ρu
        ru_k = rho_k * u_k
        ru_n = rho_n * _u_n

        # --------------- Ghost cell extensions ---------------
        # ng=2: extended size = N+4, indices 0..(N+3)
        # Cell i maps to ext index i+2 (ng=2).
        # The extra ghost layer is needed for the 3-cell stencil
        # (phi_UU, phi_U, phi_D) of the 2nd-order TVD reconstruction.
        ng = 2
        p_ext  = _ghost_extend_anp(p_k,  bc_l, bc_r, ng)
        T_ext  = _ghost_extend_anp(T_k,  bc_l, bc_r, ng)
        a1_ext = _ghost_extend_anp(a1_k, bc_l, bc_r, ng)
        u_ext  = _ghost_extend_vel_anp(u_k, bc_l, bc_r, ng)

        # --------------- Integer index arrays (plain numpy — not autograd) ---------------
        # These are constant indices used to index into the extended arrays.
        # They must be plain Python/numpy ints, not autograd boxes.
        # _idx_* are precomputed in the closure (captured from outer scope).
        #
        # For right face of cell i: f_R = i+1
        #   left_ext  index = f_R      = i+1  → _idx_R_L[i]
        #   right_ext index = f_R + 1  = i+2  → _idx_R_R[i]
        # For left face of cell i: f_L = i
        #   left_ext  index = f_L      = i    → _idx_L_L[i]
        #   right_ext index = f_L + 1  = i+1  → _idx_L_R[i]

        # Face velocity (lagged — numpy constants, not autograd)
        tR = _theta[_idx_R_face]   # (N,) theta at right face of each cell
        tL = _theta[_idx_L_face]   # (N,) theta at left  face of each cell

        # ============================================================
        # 2nd-order TVD face values using *lagged* minmod slopes.
        # Deferred-correction form (ACID 2 Eq. 47 / Denner 2018 Eq. 9):
        #
        #   phi_f = phi_U(W^{n+1}) + 0.5 * slope_lagged(W_n)
        #
        # The slope is precomputed in plain numpy from the old-time
        # field and captured as a constant in the closure, so the
        # autograd graph only differentiates through phi_U.  Picking
        # the right-face vs left-face index sets below keeps the
        # gradient flow on the proper cell.  The theta-sign blending
        # was already done when building _slope_*_R/L above.
        # ============================================================

        # Right face upwind cell (theta-sign aware) — autograd variable.
        p_upR  = (anp.where(tR >= 0, p_ext[_idx_R_U_p],  p_ext[_idx_R_U_n])
                  + 0.5 * _slope_p_R)
        T_upR  = (anp.where(tR >= 0, T_ext[_idx_R_U_p],  T_ext[_idx_R_U_n])
                  + 0.5 * _slope_T_R)
        a1_upR = (anp.where(tR >= 0, a1_ext[_idx_R_U_p], a1_ext[_idx_R_U_n])
                  + 0.5 * _slope_a1_R)
        u_upR  = (anp.where(tR >= 0, u_ext[_idx_R_U_p],  u_ext[_idx_R_U_n])
                  + 0.5 * _slope_u_R)

        # Left face upwind cell.
        p_upL  = (anp.where(tL >= 0, p_ext[_idx_L_U_p],  p_ext[_idx_L_U_n])
                  + 0.5 * _slope_p_L)
        T_upL  = (anp.where(tL >= 0, T_ext[_idx_L_U_p],  T_ext[_idx_L_U_n])
                  + 0.5 * _slope_T_L)
        a1_upL = (anp.where(tL >= 0, a1_ext[_idx_L_U_p], a1_ext[_idx_L_U_n])
                  + 0.5 * _slope_a1_L)
        u_upL  = (anp.where(tL >= 0, u_ext[_idx_L_U_p],  u_ext[_idx_L_U_n])
                  + 0.5 * _slope_u_L)

        # Cell-centred arithmetic-mean pressure on the face (used as the
        # central pressure term in the momentum flux).  ng=2 maps cell i
        # to ext idx i+2, so _idx_R_L / _idx_R_R etc. were rebuilt above.
        p_ext_R_L  = p_ext[_idx_R_L];   p_ext_R_R  = p_ext[_idx_R_R]
        p_ext_L_L  = p_ext[_idx_L_L];   p_ext_L_R  = p_ext[_idx_L_R]

        # Also need a1_ext at the immediate face neighbours for the
        # non-conservative alpha-advection term below.  Keep these as
        # plain upwind selects (1st-order) for the alpha source — the
        # 2nd-order reconstruction is already used for the conservative
        # flux above; mixing recon styles on the alpha source previously
        # had no analytical justification in Allaire/Kapila.
        a1_ext_R_L = a1_ext[_idx_R_L];  a1_ext_R_R = a1_ext[_idx_R_R]
        a1_ext_L_L = a1_ext[_idx_L_L];  a1_ext_L_R = a1_ext[_idx_L_R]

        # Upwind phase densities and volumetric internal energies
        r1_upR = _eos_rho_anp(p_upR, T_upR, ph1_params)
        r2_upR = _eos_rho_anp(p_upR, T_upR, ph2_params)
        r1_upL = _eos_rho_anp(p_upL, T_upL, ph1_params)
        r2_upL = _eos_rho_anp(p_upL, T_upL, ph2_params)

        a2_upR = 1.0 - a1_upR
        a2_upL = 1.0 - a1_upL

        rho_upR = a1_upR * r1_upR + a2_upR * r2_upR
        rho_upL = a1_upL * r1_upL + a2_upL * r2_upL

        # Phase-1 partial mass at faces
        a1r1_upR = a1_upR * r1_upR
        a1r1_upL = a1_upL * r1_upL

        # Momentum at upwind cells
        ru_upR = rho_upR * u_upR
        ru_upL = rho_upL * u_upL

        # Pressure at faces (arithmetic mean, using extended array values)
        # Right face: p_fR = 0.5*(p[i] + p[iR])  (i=cell-left-of-face, iR=cell-right-of-face)
        p_fR = 0.5 * (p_ext_R_L + p_ext_R_R)
        p_fL = 0.5 * (p_ext_L_L + p_ext_L_R)

        # Volumetric internal energy at upwind (p,T)
        evol1_upR = _eos_evol_anp(p_upR, T_upR, ph1_params)
        evol2_upR = _eos_evol_anp(p_upR, T_upR, ph2_params)
        evol1_upL = _eos_evol_anp(p_upL, T_upL, ph1_params)
        evol2_upL = _eos_evol_anp(p_upL, T_upL, ph2_params)

        # ACID energy flux: use cell i's α₁ (not upwind) for face enthalpy composition.
        # At uniform (p,T,u): both faces evaluate identical H_acid → net flux = 0.
        # This is the PE-preserving (pressure equilibrium) energy discretisation.
        # r1_upR, r2_upR, r1_upL, r2_upL are already computed above (lines ~381-384).
        # a2_k is already computed above as 1.0 - a1_k.
        rho_acid_R = a1_k * r1_upR + a2_k * r2_upR
        rho_acid_L = a1_k * r1_upL + a2_k * r2_upL
        rE_acid_R = a1_k * evol1_upR + a2_k * evol2_upR + 0.5 * rho_acid_R * u_upR**2
        rE_acid_L = a1_k * evol1_upL + a2_k * evol2_upL + 0.5 * rho_acid_L * u_upL**2

        # Conservative fluxes
        flux_m1_R = a1r1_upR * tR           # phase-1 mass right
        flux_m1_L = a1r1_upL * tL           # phase-1 mass left
        flux_mom_R = ru_upR * tR + p_fR     # momentum right
        flux_mom_L = ru_upL * tL + p_fL     # momentum left
        flux_E_R  = (rE_acid_R + p_upR) * tR  # ACID energy right (PE-preserving)
        flux_E_L  = (rE_acid_L + p_upL) * tL  # ACID energy left

        # Alpha equation: non-conservative with K source term
        # R₅ = ∂α₁/∂t + (θ·α₁_upwind_R - θ·α₁_upwind_L)/dx
        #      - α₁·div_θ - K·div_θ = 0
        # (spec §9.1)
        a1_adv_R = anp.where(tR >= 0, a1_ext_R_L, a1_ext_R_R)
        a1_adv_L = anp.where(tL >= 0, a1_ext_L_L, a1_ext_L_R)
        div_theta = (tR - tL) / dx

        flux_a1_R = a1_adv_R * tR
        flux_a1_L = a1_adv_L * tL

        # --------------- Residuals (BDF1) ---------------
        # R₁: phase-1 mass conservation
        R_m1 = (a1r1_k - a1r1_n) / dt + (flux_m1_R - flux_m1_L) / dx

        # R₂: mixture momentum conservation
        R_mom = (ru_k - ru_n) / dt + (flux_mom_R - flux_mom_L) / dx

        # R₃: total energy conservation
        R_en = (rE_k - rE_n) / dt + (flux_E_R - flux_E_L) / dx

        # R₄: volume fraction transport (non-conservative + K compression)
        R_a1 = ((a1_k - _a1_n_c) / dt
                + (flux_a1_R - flux_a1_L) / dx
                - a1_k * div_theta
                - _K * div_theta)

        # Apply lagged per-block residual scaling so the four blocks
        # contribute on comparable footing in the LM normal equations
        # (Marquardt 1963; Knoll & Keyes 2004 §3.5).
        R_m1  = R_m1  / _scale_m1
        R_mom = R_mom / _scale_mom
        R_en  = R_en  / _scale_en
        R_a1  = R_a1  / _scale_a1

        return anp.concatenate([R_m1, R_mom, R_en, R_a1])

    return residual_func


# ---------------------------------------------------------------------------
# Autograd Jacobian computation
# ---------------------------------------------------------------------------

def compute_jacobian_ad(residual_func, W_k):
    """Compute full 4N×4N Jacobian via autograd.

    Parameters
    ----------
    residual_func : callable  R(W) → (4N,)
    W_k           : ndarray (4N,)  current Newton iterate

    Returns
    -------
    J : ndarray (4N, 4N)  dense Jacobian

    Notes
    -----
    autograd.jacobian returns ∂R_i/∂W_j as a (4N, 4N) dense matrix.
    For N=10 this is 40×40, for N=200 this is 800×800 — manageable.
    """
    if not _AUTOGRAD_AVAILABLE:
        raise RuntimeError(
            "autograd is not installed. Install it with: pip install autograd")
    J_func = ad_jacobian(residual_func)
    return J_func(W_k)


# ---------------------------------------------------------------------------
# Helper: pack/unpack primitive vector
# ---------------------------------------------------------------------------

def pack_W(p, u, T, a1):
    """Pack 4 arrays into W = [p | u | T | α₁] (4N,)."""
    return np.concatenate([p, u, T, a1])


def unpack_W(W, N):
    """Unpack W (4N,) into (p, u, T, a1) each (N,)."""
    return W[0*N:1*N], W[1*N:2*N], W[2*N:3*N], W[3*N:4*N]


# ---------------------------------------------------------------------------
# K coefficient (Wood's sound speed) — standard numpy (lagged)
# ---------------------------------------------------------------------------

def compute_K_ad(a1, p, T, ph1, ph2):
    """Compute Wood's K closure coefficient (lagged, standard numpy).

    K₁ = α₁·ρ·c_Wood² / (ρ₁·c₁²)

    Parameters
    ----------
    a1       : ndarray (N,)  volume fraction of phase 1
    p, T     : ndarray (N,)  pressure, temperature
    ph1, ph2 : EOS dicts or objects

    Returns
    -------
    K1 : ndarray (N,)
    """
    eos1 = create_eos(ph1)
    eos2 = create_eos(ph2)

    a2 = np.maximum(1.0 - a1, 0.0)

    rho1 = eos1.rho(p, T)
    c1   = eos1.c(p, T)
    rho2 = eos2.rho(p, T)
    c2   = eos2.c(p, T)
    rho  = a1 * rho1 + a2 * rho2

    rho1c1sq = rho1 * c1**2 + 1e-300
    rho2c2sq = rho2 * c2**2 + 1e-300
    inv_rho_c2 = a1 / rho1c1sq + a2 / rho2c2sq
    rho_cw2  = 1.0 / (inv_rho_c2 + 1e-300)

    K1 = a1 * rho_cw2 / rho1c1sq
    return K1


# ---------------------------------------------------------------------------
# MWI face velocity (lagged) — standard numpy
# ---------------------------------------------------------------------------

def compute_face_velocity_ad(u_k, p_k, rho_k, dx, dt, bc_l, bc_r,
                              theta_old=None, u_bar_old=None, rho_star_old=None):
    """Compute MWI face velocity theta (N+1,) — lagged, not autograd.

    θ_f = ū_f - d̂_f·(p_R - p_L)/dx  + transient correction

    Parameters
    ----------
    u_k, p_k, rho_k : ndarray (N,)
    dx, dt           : float
    bc_l, bc_r       : str
    theta_old, u_bar_old, rho_star_old : optional transient correction data

    Returns
    -------
    theta   : ndarray (N+1,)
    u_bar   : ndarray (N+1,)
    d_hat   : ndarray (N+1,)
    rho_star: ndarray (N+1,)
    """
    from .flux.mwi import harmonic_face_density, mwi_face_coeff_denner
    from .boundary import apply_ghost, apply_ghost_velocity

    rho_star = harmonic_face_density(rho_k, bc_l, bc_r)
    e_diag   = rho_k / dt  # momentum diagonal ≈ ρ/dt
    d_hat    = mwi_face_coeff_denner(e_diag, rho_star, dx, dt, bc_l, bc_r)

    N  = len(u_k)
    ng = 2
    u_ext = apply_ghost_velocity(u_k, bc_l, bc_r, ng)
    p_ext = apply_ghost(p_k, bc_l, bc_r, ng)
    theta = np.empty(N + 1)
    u_bar = np.empty(N + 1)
    for f in range(N + 1):
        iL = ng + f - 1
        iR = ng + f
        ub = 0.5 * (u_ext[iL] + u_ext[iR])
        dp = (p_ext[iR] - p_ext[iL]) / dx
        theta[f] = ub - d_hat[f] * dp
        u_bar[f] = ub

    # Transient correction (Denner 2018 Eq. 20)
    if (theta_old is not None and u_bar_old is not None
            and rho_star_old is not None and dt is not None):
        theta += d_hat * (rho_star_old / dt) * (theta_old - u_bar_old)

    return theta, u_bar, d_hat, rho_star


# ---------------------------------------------------------------------------
# EOS inversion (lagged) — standard numpy, 2×2 Newton per cell
# ---------------------------------------------------------------------------

def invert_primitives(a1, rho, rho_e, p_guess, T_guess, ph1_params, ph2_params,
                      tol=1e-10, max_iter=50):
    """Recover (p, T) from mixture conserved state (α₁, ρ, ρe).

    2×2 Newton (vectorised over all N cells):
      f₁ = α₁·ρ₁(p,T) + α₂·ρ₂(p,T) - ρ  = 0
      f₂ = α₁·(ρe)₁(p,T) + α₂·(ρe)₂(p,T) - ρe = 0

    Parameters
    ----------
    a1           : ndarray (N,)  volume fraction
    rho          : ndarray (N,)  mixture density
    rho_e        : ndarray (N,)  mixture volumetric internal energy
    p_guess      : ndarray (N,)
    T_guess      : ndarray (N,)
    ph1_params   : tuple  from _get_eos_params()
    ph2_params   : tuple
    tol, max_iter: convergence

    Returns
    -------
    p : ndarray (N,)
    T : ndarray (N,)
    """
    from .eos.eos_class import create_eos

    # We use the regular EOS objects for the inversion Newton (numpy, not autograd)
    kind1 = ph1_params[0]
    kind2 = ph2_params[0]

    if kind1 == 'nasg':
        _, g1, pi1, b1, kv1, eta1 = ph1_params
        def rho1_f(p, T): return _nasg_rho_anp(p, T, g1, pi1, b1, kv1)
        def evol1_f(p, T): return _nasg_evol_anp(p, T, g1, pi1, b1, kv1, eta1)
        def drho1_dp(p, T):
            gm1 = g1 - 1.0
            A = kv1 * T * gm1 + b1 * (p + pi1) + 1e-300
            return kv1 * T * gm1 / (A * A + 1e-300)
        def drho1_dT(p, T):
            gm1 = g1 - 1.0
            A = kv1 * T * gm1 + b1 * (p + pi1) + 1e-300
            return -(p + pi1) * kv1 * gm1 / (A * A + 1e-300)
        def devol1_dp(p, T):
            r = rho1_f(p, T);  h = _nasg_h_anp(p, T, g1, pi1, b1, kv1, eta1)
            return r * b1 + h * drho1_dp(p, T) - 1.0
        def devol1_dT(p, T):
            r = rho1_f(p, T);  h = _nasg_h_anp(p, T, g1, pi1, b1, kv1, eta1)
            cp = g1 * kv1
            return r * cp + h * drho1_dT(p, T)
    elif kind1 == 'stiffened':
        _, g1, pi1, cv1, q1 = ph1_params
        cp1 = g1 * cv1; R1 = (g1 - 1.0) * cv1
        def rho1_f(p, T): return (p + pi1) / (R1 * T + 1e-300)
        def evol1_f(p, T): return rho1_f(p, T) * (cp1 * T + q1) - p
        def drho1_dp(p, T): return 1.0 / (R1 * T + 1e-300)
        def drho1_dT(p, T): return -(p + pi1) / (R1 * T**2 + 1e-300)
        def devol1_dp(p, T):
            r = rho1_f(p, T);  h = cp1 * T + q1
            return r * 0.0 + h * drho1_dp(p, T) - 1.0
        def devol1_dT(p, T):
            r = rho1_f(p, T);  h = cp1 * T + q1
            return r * cp1 + h * drho1_dT(p, T)
    else:
        eos1 = ph1_params[1]
        def rho1_f(p, T): return eos1.rho(p, T)
        def evol1_f(p, T): return eos1.e_vol(p, T)
        def drho1_dp(p, T): return eos1.drho_dp(p, T)
        def drho1_dT(p, T): return eos1.drho_dT(p, T)
        def devol1_dp(p, T): return eos1.de_vol_dp(p, T)
        def devol1_dT(p, T): return eos1.de_vol_dT(p, T)

    if kind2 == 'nasg':
        _, g2, pi2, b2, kv2, eta2 = ph2_params
        def rho2_f(p, T): return _nasg_rho_anp(p, T, g2, pi2, b2, kv2)
        def evol2_f(p, T): return _nasg_evol_anp(p, T, g2, pi2, b2, kv2, eta2)
        def drho2_dp(p, T):
            gm1 = g2 - 1.0
            A = kv2 * T * gm1 + b2 * (p + pi2) + 1e-300
            return kv2 * T * gm1 / (A * A + 1e-300)
        def drho2_dT(p, T):
            gm1 = g2 - 1.0
            A = kv2 * T * gm1 + b2 * (p + pi2) + 1e-300
            return -(p + pi2) * kv2 * gm1 / (A * A + 1e-300)
        def devol2_dp(p, T):
            r = rho2_f(p, T);  h = _nasg_h_anp(p, T, g2, pi2, b2, kv2, eta2)
            return r * b2 + h * drho2_dp(p, T) - 1.0
        def devol2_dT(p, T):
            r = rho2_f(p, T);  h = _nasg_h_anp(p, T, g2, pi2, b2, kv2, eta2)
            cp = g2 * kv2
            return r * cp + h * drho2_dT(p, T)
    elif kind2 == 'stiffened':
        _, g2, pi2, cv2, q2 = ph2_params
        cp2 = g2 * cv2; R2 = (g2 - 1.0) * cv2
        def rho2_f(p, T): return (p + pi2) / (R2 * T + 1e-300)
        def evol2_f(p, T): return rho2_f(p, T) * (cp2 * T + q2) - p
        def drho2_dp(p, T): return 1.0 / (R2 * T + 1e-300)
        def drho2_dT(p, T): return -(p + pi2) / (R2 * T**2 + 1e-300)
        def devol2_dp(p, T):
            r = rho2_f(p, T);  h = cp2 * T + q2
            return r * 0.0 + h * drho2_dp(p, T) - 1.0
        def devol2_dT(p, T):
            r = rho2_f(p, T);  h = cp2 * T + q2
            return r * cp2 + h * drho2_dT(p, T)
    else:
        eos2 = ph2_params[1]
        def rho2_f(p, T): return eos2.rho(p, T)
        def evol2_f(p, T): return eos2.e_vol(p, T)
        def drho2_dp(p, T): return eos2.drho_dp(p, T)
        def drho2_dT(p, T): return eos2.drho_dT(p, T)
        def devol2_dp(p, T): return eos2.de_vol_dp(p, T)
        def devol2_dT(p, T): return eos2.de_vol_dT(p, T)

    a2 = np.maximum(1.0 - a1, 0.0)
    p  = p_guess.copy()
    T  = T_guess.copy()

    for _ in range(max_iter):
        r1  = rho1_f(p, T);   r2  = rho2_f(p, T)
        e1  = evol1_f(p, T);  e2  = evol2_f(p, T)

        f1  = a1 * r1 + a2 * r2 - rho
        f2  = a1 * e1 + a2 * e2 - rho_e

        rel1 = np.abs(f1) / (np.abs(rho)   + 1e-300)
        rel2 = np.abs(f2) / (np.abs(rho_e) + 1e-300)
        if np.max(rel1) < tol and np.max(rel2) < tol:
            break

        J00 = a1 * drho1_dp(p, T) + a2 * drho2_dp(p, T)
        J01 = a1 * drho1_dT(p, T) + a2 * drho2_dT(p, T)
        J10 = a1 * devol1_dp(p, T) + a2 * devol2_dp(p, T)
        J11 = a1 * devol1_dT(p, T) + a2 * devol2_dT(p, T)

        det = J00 * J11 - J01 * J10 + 1e-300
        dp  = ( J11 * f1 - J01 * f2) / det
        dT  = (-J10 * f1 + J00 * f2) / det

        p -= dp
        T -= dT
        p = np.maximum(p, 1.0)
        T = np.maximum(T, 1e-3)

    return p, T


# ---------------------------------------------------------------------------
# Conservative-variable residual wrapper + FD Jacobian
# ---------------------------------------------------------------------------

def make_residual_cons(Q_n, N, dx, dt, ph1_params, ph2_params,
                       bc_l, bc_r, theta_lag, K_lag, p_guess, T_guess):
    """Build a conservative-variable residual closure R(Q) → (5N,).

    Q = [α₁ρ₁ | α₂ρ₂ | ρu | ρE | α₁]  (5N,)

    Inside the residual, (p, T, u) are recovered from Q via EOS inversion.
    theta and K are lagged (frozen from outer Picard).

    Parameters
    ----------
    Q_n            : ndarray (5N,)  old-time conservative state
    N, dx, dt      : int, float, float
    ph1_params, ph2_params : tuple from _get_eos_params()
    bc_l, bc_r     : str
    theta_lag      : ndarray (N+1,) frozen face velocity
    K_lag          : ndarray (N,)   frozen K coefficient
    p_guess, T_guess : ndarray (N,)  initial guess for EOS inversion

    Returns
    -------
    residual_func : callable  R(Q) → ndarray (5N,)
    """
    from .boundary import apply_ghost, apply_ghost_velocity

    _Q_n = Q_n.copy()
    _theta = theta_lag.copy()
    _K = K_lag.copy()
    _p_guess = p_guess.copy()
    _T_guess = T_guess.copy()

    def residual_func(Q):
        a1r1 = Q[0*N:1*N]
        a2r2 = Q[1*N:2*N]
        ru   = Q[2*N:3*N]
        rE   = Q[3*N:4*N]
        a1   = Q[4*N:5*N]

        a1 = np.clip(a1, 1e-14, 1.0 - 1e-14)
        a2 = 1.0 - a1

        # Primitive recovery
        rho = np.maximum(a1r1 + a2r2, 1e-300)
        u   = ru / rho
        rho_e = rE - 0.5 * rho * u**2

        p, T = invert_primitives(a1, rho, rho_e, _p_guess, _T_guess,
                                  ph1_params, ph2_params)

        # Old-time
        a1r1_n = _Q_n[0*N:1*N]
        a2r2_n = _Q_n[1*N:2*N]
        ru_n   = _Q_n[2*N:3*N]
        rE_n   = _Q_n[3*N:4*N]
        a1_n   = _Q_n[4*N:5*N]

        # Ghost extensions (1 layer)
        ng = 1
        a1r1_ext = apply_ghost(a1r1, bc_l, bc_r, ng)
        a2r2_ext = apply_ghost(a2r2, bc_l, bc_r, ng)
        ru_ext   = apply_ghost(ru,   bc_l, bc_r, ng)
        rE_ext   = apply_ghost(rE,   bc_l, bc_r, ng)
        a1_ext   = apply_ghost(a1,   bc_l, bc_r, ng)
        p_ext    = apply_ghost(p,    bc_l, bc_r, ng)
        T_ext    = apply_ghost(T,    bc_l, bc_r, ng)
        u_ext    = apply_ghost(u,    bc_l, bc_r, ng)

        R = np.zeros(5 * N)

        for i in range(N):
            f_R = i + 1
            f_L = i

            iL_ext = ng + i - 1
            i_ext  = ng + i
            iR_ext = ng + i + 1

            tR = _theta[f_R]
            tL = _theta[f_L]

            # Upwind mass flux (conservative: use a1r1, a2r2 directly)
            a1r1_fR = a1r1_ext[i_ext] if tR >= 0 else a1r1_ext[iR_ext]
            a2r2_fR = a2r2_ext[i_ext] if tR >= 0 else a2r2_ext[iR_ext]
            a1r1_fL = a1r1_ext[iL_ext] if tL >= 0 else a1r1_ext[i_ext]
            a2r2_fL = a2r2_ext[iL_ext] if tL >= 0 else a2r2_ext[i_ext]

            # R1: phase 1 mass
            R[0*N+i] = (a1r1[i] - a1r1_n[i]) / dt + (a1r1_fR * tR - a1r1_fL * tL) / dx

            # R2: phase 2 mass
            R[1*N+i] = (a2r2[i] - a2r2_n[i]) / dt + (a2r2_fR * tR - a2r2_fL * tL) / dx

            # R3: momentum (upwind ρu·θ + face pressure)
            ru_fR = ru_ext[i_ext]   if tR >= 0 else ru_ext[iR_ext]
            ru_fL = ru_ext[iL_ext]  if tL >= 0 else ru_ext[i_ext]
            pR_face = 0.5 * (p_ext[i_ext] + p_ext[iR_ext])
            pL_face = 0.5 * (p_ext[iL_ext] + p_ext[i_ext])
            R[2*N+i] = (ru[i] - ru_n[i]) / dt + (ru_fR * tR - ru_fL * tL + pR_face - pL_face) / dx

            # R4: energy — ACID: use cell i's α₁ with upwind (p,T,u).
            # PE-preserving: at uniform (p,T,u) the net energy flux is zero.
            p_up_R = p_ext[i_ext]   if tR >= 0 else p_ext[iR_ext]
            T_up_R = T_ext[i_ext]   if tR >= 0 else T_ext[iR_ext]
            u_up_R = u_ext[i_ext]   if tR >= 0 else u_ext[iR_ext]
            p_up_L = p_ext[iL_ext]  if tL >= 0 else p_ext[i_ext]
            T_up_L = T_ext[iL_ext]  if tL >= 0 else T_ext[i_ext]
            u_up_L = u_ext[iL_ext]  if tL >= 0 else u_ext[i_ext]
            rH_acid_R = _acid_rH_face(a1[i], p_up_R, T_up_R, u_up_R, ph1_params, ph2_params)
            rH_acid_L = _acid_rH_face(a1[i], p_up_L, T_up_L, u_up_L, ph1_params, ph2_params)
            R[3*N+i] = (rE[i] - rE_n[i]) / dt + (rH_acid_R * tR - rH_acid_L * tL) / dx

            # R5: alpha (non-conservative + K)
            a1_advR = a1_ext[i_ext]   if tR >= 0 else a1_ext[iR_ext]
            a1_advL = a1_ext[iL_ext]  if tL >= 0 else a1_ext[i_ext]
            div_theta = (tR - tL) / dx
            R[4*N+i] = ((a1[i] - a1_n[i]) / dt
                        + (tR * a1_advR - tL * a1_advL) / dx
                        - a1[i] * div_theta
                        - _K[i] * div_theta)

        return R

    return residual_func


def fd_jacobian_cons(residual_func, Q_k, N, eps_rel=1e-7):
    """Compute 5N×5N Jacobian via forward finite differences.

    Parameters
    ----------
    residual_func : callable  R(Q) → (5N,)
    Q_k           : ndarray (5N,)
    N             : int
    eps_rel       : float  relative perturbation size

    Returns
    -------
    J : ndarray (5N, 5N)
    """
    n5 = 5 * N
    R0 = residual_func(Q_k)
    J = np.zeros((n5, n5))

    for j in range(n5):
        Q_pert = Q_k.copy()
        eps_j = eps_rel * max(abs(Q_k[j]), 1.0)
        Q_pert[j] += eps_j
        R_pert = residual_func(Q_pert)
        J[:, j] = (R_pert - R0) / eps_j

    return J


def pack_Q(a1r1, a2r2, ru, rE, a1):
    """Pack 5 arrays into Q = [α₁ρ₁ | α₂ρ₂ | ρu | ρE | α₁] (5N,)."""
    return np.concatenate([a1r1, a2r2, ru, rE, a1])


def unpack_Q(Q, N):
    """Unpack Q (5N,) into (a1r1, a2r2, ru, rE, a1) each (N,)."""
    return Q[0*N:1*N], Q[1*N:2*N], Q[2*N:3*N], Q[3*N:4*N], Q[4*N:5*N]


# ---------------------------------------------------------------------------
# Algebraic EOS inversion — autograd-compatible
# ---------------------------------------------------------------------------

def _mixture_p_from_Q_anp(a1r1, a2r2, ru, rE, a1, ph1_params, ph2_params):
    """Algebraic pressure recovery from conservative Q — autograd-compatible.

    p = (ρe - B) / A
    where A = Σ α_k·C_k/(γ_k-1), B = Σ [γ_k·α_k·C_k·p∞_k/(γ_k-1) + η_k·(α_k·ρ_k)]
    C_k = 1 - b_k·ρ_k,  and α_k·C_k = α_k - b_k·(α_k·ρ_k)  (Q-safe: no div by α_k)

    For ideal gas (b=0, p∞=0, η=0): p = ρe · (γ-1) / α₂  (no iteration needed).
    For NASG water: γ=1.187, p∞=7.028e8, b=6.61e-4, κᵥ=3610, η=-1.177788e6.

    Parameters
    ----------
    a1r1, a2r2, ru, rE : array (N,)  conservative variables per cell
    a1                  : array (N,)  volume fraction of phase 1
    ph1_params, ph2_params : tuple from _get_eos_params()

    Returns
    -------
    p : array (N,)  pressure
    """
    a2 = 1.0 - a1
    rho = a1r1 + a2r2
    u = ru / (rho + 1e-300)
    rho_e = rE - 0.5 * ru * u      # ρe = ρE - ½ρu²

    # --- Phase 1 parameters ---
    kind1 = ph1_params[0]
    if kind1 == 'nasg':
        _, g1, pi1, b1, kv1, eta1 = ph1_params
    elif kind1 == 'stiffened':
        _, g1, pi1, cv1, q1 = ph1_params
        b1   = 0.0
        eta1 = q1
        kv1  = cv1
    else:
        raise ValueError(f"Unsupported EOS type for algebraic p recovery: {kind1}")

    # --- Phase 2 parameters ---
    kind2 = ph2_params[0]
    if kind2 == 'nasg':
        _, g2, pi2, b2, kv2, eta2 = ph2_params
    elif kind2 == 'stiffened':
        _, g2, pi2, cv2, q2 = ph2_params
        b2   = 0.0
        eta2 = q2
        kv2  = cv2
    else:
        raise ValueError(f"Unsupported EOS type for algebraic p recovery: {kind2}")

    # Q-safe computation: α_k·C_k = α_k - b_k·(α_k·ρ_k)
    # avoids division by α_k which blows up near pure-phase cells
    a1C1 = a1   - b1 * a1r1
    a2C2 = a2   - b2 * a2r2

    A = a1C1 / (g1 - 1.0) + a2C2 / (g2 - 1.0)
    B = (g1 * a1C1 * pi1 / (g1 - 1.0) + eta1 * a1r1
       + g2 * a2C2 * pi2 / (g2 - 1.0) + eta2 * a2r2)

    p = (rho_e - B) / (A + 1e-300)
    return p


def _mixture_T_from_Q_anp(a1r1, a2r2, a1, p, ph1_params, ph2_params):
    """Algebraic temperature recovery from Q and p — autograd-compatible.

    T_k = (p + p∞_k)·(1 - b_k·ρ_k) / [ρ_k·κᵥ_k·(γ_k-1)]
    Returns the volume-fraction-weighted average of T₁ and T₂.

    Parameters
    ----------
    a1r1, a2r2 : array (N,)  partial mass densities
    a1          : array (N,)  volume fraction of phase 1
    p           : array (N,)  pressure (already recovered)
    ph1_params, ph2_params : tuple from _get_eos_params()

    Returns
    -------
    T : array (N,)  temperature
    """
    a2 = 1.0 - a1

    # --- Phase 1 ---
    kind1 = ph1_params[0]
    if kind1 == 'nasg':
        _, g1, pi1, b1, kv1, eta1 = ph1_params
    elif kind1 == 'stiffened':
        _, g1, pi1, cv1, q1 = ph1_params
        b1  = 0.0
        kv1 = cv1
    else:
        raise ValueError(f"Unsupported EOS type for T recovery: {kind1}")

    # --- Phase 2 ---
    kind2 = ph2_params[0]
    if kind2 == 'nasg':
        _, g2, pi2, b2, kv2, eta2 = ph2_params
    elif kind2 == 'stiffened':
        _, g2, pi2, cv2, q2 = ph2_params
        b2  = 0.0
        kv2 = cv2
    else:
        raise ValueError(f"Unsupported EOS type for T recovery: {kind2}")

    # ρ_k = (α_k·ρ_k) / α_k — avoid div-by-zero near pure-phase boundaries
    rho1 = a1r1 / (a1 + 1e-300)
    rho2 = a2r2 / (a2 + 1e-300)

    # T_k = (p + p∞_k)·(1 - b_k·ρ_k) / [ρ_k·κᵥ_k·(γ_k-1)]
    T1 = (p + pi1) * (1.0 - b1 * rho1) / (rho1 * kv1 * (g1 - 1.0) + 1e-300)
    T2 = (p + pi2) * (1.0 - b2 * rho2) / (rho2 * kv2 * (g2 - 1.0) + 1e-300)

    # Volume-weighted average (numerically stable near α→0)
    T = a1 * T1 + a2 * T2
    return T


# ---------------------------------------------------------------------------
# Autograd-compatible conservative-variable residual
# ---------------------------------------------------------------------------

def make_residual_cons_ad(Q_n, N, dx, dt, ph1_params, ph2_params,
                           bc_l, bc_r, theta_lag, K_lag,
                           p_lag=None, T_lag=None):
    """Build autograd-differentiable conservative residual closure R(Q) → (5N,).

    Q = [α₁ρ₁ | α₂ρ₂ | ρu | ρE | α₁]  (5N,)

    Strategy: "partially lagged" — the direct conservative variables (a1r1,
    a2r2, ru, rE, a1) are the LIVE autograd variables. All EOS-derived
    quantities (p, T, phase densities, phase energies) used in ACID energy
    flux and face pressure are LAGGED from the outer Picard iteration.

    This keeps the autograd graph simple (temporal I/dt + upwind advection)
    while preserving PE through ACID with lagged (p,T) and live α₁.

    Closure captures (frozen constants):
        Q_n        — old-time conservative state
        theta_lag  — MWI face velocity (N+1,), lagged
        K_lag      — Wood K coefficient (N,), lagged
        p_lag      — pressure (N,), lagged from outer Picard
        T_lag      — temperature (N,), lagged from outer Picard

    Parameters
    ----------
    Q_n            : ndarray (5N,)  old-time conservative state
    N, dx, dt      : int, float, float
    ph1_params, ph2_params : tuple from _get_eos_params()
    bc_l, bc_r     : str
    theta_lag      : ndarray (N+1,)
    K_lag          : ndarray (N,)
    p_lag          : ndarray (N,) or None  lagged pressure
    T_lag          : ndarray (N,) or None  lagged temperature

    Returns
    -------
    residual_func : callable  R(Q) → anp.array (5N,)
    """
    # Constants captured in closure (plain numpy arrays)
    _a1r1_n = anp.array(Q_n[0*N:1*N])
    _a2r2_n = anp.array(Q_n[1*N:2*N])
    _ru_n   = anp.array(Q_n[2*N:3*N])
    _rE_n   = anp.array(Q_n[3*N:4*N])
    _a1_n   = anp.array(Q_n[4*N:5*N])
    _theta  = anp.array(theta_lag)
    _K      = anp.array(K_lag)

    # Lagged p, T for ACID energy flux and face pressure (numpy constants)
    _p_lag = anp.array(p_lag) if p_lag is not None else None
    _T_lag = anp.array(T_lag) if T_lag is not None else None

    # Precompute ACID quantities from lagged (p,T) — pure numpy, frozen
    if _p_lag is not None and _T_lag is not None:
        from .boundary import apply_ghost, apply_ghost_velocity
        # Phase densities and volumetric energies at lagged (p,T)
        _r1_lag = np.array(_eos_rho_anp(np.array(_p_lag), np.array(_T_lag), ph1_params))
        _r2_lag = np.array(_eos_rho_anp(np.array(_p_lag), np.array(_T_lag), ph2_params))
        _evol1_lag = np.array(_eos_evol_anp(np.array(_p_lag), np.array(_T_lag), ph1_params))
        _evol2_lag = np.array(_eos_evol_anp(np.array(_p_lag), np.array(_T_lag), ph2_params))
        # Ghost-extend lagged quantities for face access
        _p_lag_ext = apply_ghost(np.array(_p_lag), bc_l, bc_r, 1)
        _r1_lag_ext = apply_ghost(_r1_lag, bc_l, bc_r, 1)
        _r2_lag_ext = apply_ghost(_r2_lag, bc_l, bc_r, 1)
        _evol1_lag_ext = apply_ghost(_evol1_lag, bc_l, bc_r, 1)
        _evol2_lag_ext = apply_ghost(_evol2_lag, bc_l, bc_r, 1)
        _has_lag = True
    else:
        _has_lag = False

    # Precompute integer index arrays for ghost-extended access (ng=1)
    _i_arr      = np.arange(N, dtype=int)
    _idx_R_L    = _i_arr + 1   # left-of-right-face ext index
    _idx_R_R    = _i_arr + 2   # right-of-right-face ext index
    _idx_L_L    = _i_arr        # left-of-left-face ext index
    _idx_L_R    = _i_arr + 1   # right-of-left-face ext index
    _idx_R_face = _i_arr + 1   # right face index in theta array
    _idx_L_face = _i_arr        # left face index in theta array

    def residual_func(Q):
        """Autograd-compatible residual R(Q) for 5-eq conservative Newton.

        Q : array (5N,) packed as:
            [α₁ρ₁(0..N-1) | α₂ρ₂(N..2N-1) | ρu(2N..3N-1) | ρE(3N..4N-1) | α₁(4N..5N-1)]

        Returns R : array (5N,)
        Block ordering:
          R[0*N:1*N] = R_m1    (phase-1 mass)
          R[1*N:2*N] = R_m2    (phase-2 mass)
          R[2*N:3*N] = R_mom   (mixture momentum)
          R[3*N:4*N] = R_en    (total energy)
          R[4*N:5*N] = R_a1    (volume fraction)
        """
        a1r1 = Q[0*N:1*N]
        a2r2 = Q[1*N:2*N]
        ru   = Q[2*N:3*N]
        rE   = Q[3*N:4*N]
        a1   = Q[4*N:5*N]
        a2   = 1.0 - a1

        # Derived: velocity from conservative variables (live, autograd)
        rho = a1r1 + a2r2
        u   = ru / (rho + 1e-300)

        # --------------- Ghost cell extensions (ng=1, autograd-compatible) ------
        ng = 1
        a1r1_ext = _ghost_extend_anp(a1r1, bc_l, bc_r, ng)
        a2r2_ext = _ghost_extend_anp(a2r2, bc_l, bc_r, ng)
        a1_ext   = _ghost_extend_anp(a1,   bc_l, bc_r, ng)
        u_ext    = _ghost_extend_vel_anp(u, bc_l, bc_r, ng)

        # Face velocities (lagged — numpy constants)
        tR = _theta[_idx_R_face]
        tL = _theta[_idx_L_face]

        # --------------- Gather face values ---------------
        # Right face of cell i
        a1r1_R_L = a1r1_ext[_idx_R_L];  a1r1_R_R = a1r1_ext[_idx_R_R]
        a2r2_R_L = a2r2_ext[_idx_R_L];  a2r2_R_R = a2r2_ext[_idx_R_R]
        a1_R_L   = a1_ext[_idx_R_L];    a1_R_R   = a1_ext[_idx_R_R]
        u_R_L    = u_ext[_idx_R_L];     u_R_R    = u_ext[_idx_R_R]

        # Left face of cell i
        a1r1_L_L = a1r1_ext[_idx_L_L];  a1r1_L_R = a1r1_ext[_idx_L_R]
        a2r2_L_L = a2r2_ext[_idx_L_L];  a2r2_L_R = a2r2_ext[_idx_L_R]
        a1_L_L   = a1_ext[_idx_L_L];    a1_L_R   = a1_ext[_idx_L_R]
        u_L_L    = u_ext[_idx_L_L];     u_L_R    = u_ext[_idx_L_R]

        # --------------- Upwind selection (anp.where for autograd) ------
        a1r1_upR = anp.where(tR >= 0, a1r1_R_L, a1r1_R_R)
        a2r2_upR = anp.where(tR >= 0, a2r2_R_L, a2r2_R_R)
        u_upR    = anp.where(tR >= 0, u_R_L,    u_R_R)

        a1r1_upL = anp.where(tL >= 0, a1r1_L_L, a1r1_L_R)
        a2r2_upL = anp.where(tL >= 0, a2r2_L_L, a2r2_L_R)
        u_upL    = anp.where(tL >= 0, u_L_L,    u_L_R)

        # Upwind mixture momentum: ρu = ρ·u
        rho_upR = a1r1_upR + a2r2_upR
        rho_upL = a1r1_upL + a2r2_upL
        ru_upR  = rho_upR * u_upR
        ru_upL  = rho_upL * u_upL

        # --------------- Face pressure (LAGGED from Picard) ---------------
        # Pressure is embedded in MWI θ (lagged), so face pressure is also
        # lagged for consistency. This keeps J ≈ I/dt (perfect conditioning).
        if _has_lag:
            p_fR = 0.5 * (_p_lag_ext[_idx_R_L] + _p_lag_ext[_idx_R_R])
            p_fL = 0.5 * (_p_lag_ext[_idx_L_L] + _p_lag_ext[_idx_L_R])
        else:
            p_fR = anp.zeros(N)
            p_fL = anp.zeros(N)

        # --------------- ACID energy flux (PE-preserving) ---------------
        # ACID phase decomposition (ρ_k, evol_k, p) fully LAGGED from outer
        # Picard. Cell i's α₁ (LIVE from Q) selects the composition.
        # At uniform (p,T,u): lagged = live → net ACID flux = 0 → PE preserved.
        if _has_lag:
            # Upwind lagged phase properties (pure numpy, frozen)
            r1_upR_lag  = np.where(theta_lag[_idx_R_face] >= 0,
                                   _r1_lag_ext[_idx_R_L], _r1_lag_ext[_idx_R_R])
            r2_upR_lag  = np.where(theta_lag[_idx_R_face] >= 0,
                                   _r2_lag_ext[_idx_R_L], _r2_lag_ext[_idx_R_R])
            ev1_upR_lag = np.where(theta_lag[_idx_R_face] >= 0,
                                   _evol1_lag_ext[_idx_R_L], _evol1_lag_ext[_idx_R_R])
            ev2_upR_lag = np.where(theta_lag[_idx_R_face] >= 0,
                                   _evol2_lag_ext[_idx_R_L], _evol2_lag_ext[_idx_R_R])
            p_upR_lag   = np.where(theta_lag[_idx_R_face] >= 0,
                                   _p_lag_ext[_idx_R_L], _p_lag_ext[_idx_R_R])

            r1_upL_lag  = np.where(theta_lag[_idx_L_face] >= 0,
                                   _r1_lag_ext[_idx_L_L], _r1_lag_ext[_idx_L_R])
            r2_upL_lag  = np.where(theta_lag[_idx_L_face] >= 0,
                                   _r2_lag_ext[_idx_L_L], _r2_lag_ext[_idx_L_R])
            ev1_upL_lag = np.where(theta_lag[_idx_L_face] >= 0,
                                   _evol1_lag_ext[_idx_L_L], _evol1_lag_ext[_idx_L_R])
            ev2_upL_lag = np.where(theta_lag[_idx_L_face] >= 0,
                                   _evol2_lag_ext[_idx_L_L], _evol2_lag_ext[_idx_L_R])
            p_upL_lag   = np.where(theta_lag[_idx_L_face] >= 0,
                                   _p_lag_ext[_idx_L_L], _p_lag_ext[_idx_L_R])

            # ACID: cell i's α₁ (LIVE) × lagged phase properties
            rho_acid_R = a1 * r1_upR_lag + a2 * r2_upR_lag
            rho_acid_L = a1 * r1_upL_lag + a2 * r2_upL_lag
            rE_acid_R  = (a1 * ev1_upR_lag + a2 * ev2_upR_lag
                          + 0.5 * rho_acid_R * u_upR**2)
            rE_acid_L  = (a1 * ev1_upL_lag + a2 * ev2_upL_lag
                          + 0.5 * rho_acid_L * u_upL**2)
            rH_acid_R  = rE_acid_R + p_upR_lag
            rH_acid_L  = rE_acid_L + p_upL_lag
        else:
            # No lagged data — fall back to ρH = ρE + p from algebraic p(Q)
            p_live = _mixture_p_from_Q_anp(a1r1, a2r2, ru, rE, a1,
                                            ph1_params, ph2_params)
            rH_acid_R = (rE + p_live) * anp.ones(N)
            rH_acid_L = (rE + p_live) * anp.ones(N)

        # --------------- Alpha advection ---------------
        a1_advR = anp.where(tR >= 0, a1_R_L, a1_R_R)
        a1_advL = anp.where(tL >= 0, a1_L_L, a1_L_R)
        div_theta = (tR - tL) / dx

        # --------------- Residuals (BDF1) ---------------
        # R_m1: phase-1 mass conservation
        R_m1 = (a1r1 - _a1r1_n) / dt + (a1r1_upR * tR - a1r1_upL * tL) / dx

        # R_m2: phase-2 mass conservation
        R_m2 = (a2r2 - _a2r2_n) / dt + (a2r2_upR * tR - a2r2_upL * tL) / dx

        # R_mom: mixture momentum conservation (face pressure lagged)
        R_mom = (ru - _ru_n) / dt + (ru_upR * tR - ru_upL * tL + p_fR - p_fL) / dx

        # R_en: total energy conservation (ACID with lagged p,T, live α₁,u)
        R_en = (rE - _rE_n) / dt + (rH_acid_R * tR - rH_acid_L * tL) / dx

        # R_a1: volume fraction transport (non-conservative + K source)
        R_a1 = ((a1 - _a1_n) / dt
                + (a1_advR * tR - a1_advL * tL) / dx
                - a1 * div_theta
                - _K * div_theta)

        return anp.concatenate([R_m1, R_m2, R_mom, R_en, R_a1])

    return residual_func
