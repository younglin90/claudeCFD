# solver/denner_1d/flux/consistent.py
# Ref: DENNER_SCHEME.md § 4.2, Eq.(16a-16c)
#
# Consistent transport chain: psi_face -> rho_face -> rhoU_face -> H_face
# This ensures mass-momentum-energy consistency across the interface.

import numpy as np
from ..boundary import apply_ghost, apply_ghost_velocity
from .limiter import van_leer_face, minmod_face
from ..eos.base import compute_phase_props


def consistent_face_rho(psi_face, rho1_f, rho2_f):
    """
    Compute CICSAM-consistent face density.
    rho_tilde_f = psi_face * rho1_f + (1 - psi_face) * rho2_f  [Eq. 4.2, chain step 2]

    Parameters
    ----------
    psi_face : ndarray (N+1,)   CICSAM face VOF values
    rho1_f   : ndarray (N+1,)   phase 1 density at faces (from face p, T)
    rho2_f   : ndarray (N+1,)   phase 2 density at faces

    Returns
    -------
    rho_face_consistent : ndarray (N+1,)
    """
    return psi_face * rho1_f + (1.0 - psi_face) * rho2_f


def consistent_face_momentum(rho_face_cicsam, u_ext, u_face_vol,
                               bc_l, bc_r, n_ghost=2):
    """
    Compute consistent face momentum flux (rho_tilde * u_tilde)_f.

    u_tilde_f is obtained via Minmod limiter (Favre-TVD upwinding).
    rhoU_face = rho_face_cicsam * u_tilde_face  [Eq. 4.2, chain step 4]

    Parameters
    ----------
    rho_face_cicsam : ndarray (N+1,)   consistent (CICSAM) face density
    u_ext           : ndarray (N+2*ng,) velocity with ghost cells
    u_face_vol      : ndarray (N+1,)   volume flux face velocity (for upwind direction)
    bc_l, bc_r      : str
    n_ghost         : int

    Returns
    -------
    rhoU_face : ndarray (N+1,)   deferred face momentum flux
    u_face_tvd: ndarray (N+1,)   Minmod face velocity
    """
    # Minmod-limited face velocity for Favre-TVD
    u_face_tvd = minmod_face(u_ext, u_face_vol, n_ghost)
    rhoU_face  = rho_face_cicsam * u_face_tvd
    return rhoU_face, u_face_tvd


def consistent_face_enthalpy(E_total_ext, p_ext, u_face_vol, n_ghost=2):
    """
    Compute consistent face total enthalpy H_tilde_f = E_tilde_f + p_bar_f.
    E_tilde_f: van Leer limited total energy face value.
    p_bar_f  : arithmetic mean pressure.  [Eq. 4.2, chain step 5]

    Parameters
    ----------
    E_total_ext : ndarray (N+2*ng,)  total energy density with ghost cells
    p_ext       : ndarray (N+2*ng,)  pressure with ghost cells
    u_face_vol  : ndarray (N+1,)     volume flux face velocity (for upwind direction)
    n_ghost     : int

    Returns
    -------
    H_face : ndarray (N+1,)   deferred face total enthalpy
    """
    ng = n_ghost
    N = len(u_face_vol) - 1

    # van Leer face total energy
    E_face = van_leer_face(E_total_ext, u_face_vol, n_ghost)

    # arithmetic mean pressure
    p_face = np.empty(N + 1)
    for f in range(N + 1):
        iL = ng + f - 1
        iR = ng + f
        p_face[f] = 0.5 * (p_ext[iL] + p_ext[iR])

    H_face = E_face + p_face
    return H_face


def compute_all_face_quantities(p_k, u_k, T_k, psi_new,
                                 ph1, ph2,
                                 rho_face_acid, d_hat,
                                 u_face_vol, dx,
                                 bc_l, bc_r, n_ghost=2, dt=None,
                                 psi_face_given=None):
    """
    Compute all deferred face quantities needed for assembly.

    Parameters
    ----------
    p_k, u_k, T_k   : ndarray (N,)   Picard-iterate state
    psi_new          : ndarray (N,)   updated VOF field
    ph1, ph2         : dict           EOS parameters
    rho_face_acid    : ndarray (N+1,) ACID face density
    d_hat            : ndarray (N+1,) MWI coefficient
    u_face_vol       : ndarray (N+1,) volume face velocity (for upwind)
    dx               : float
    bc_l, bc_r       : str
    n_ghost          : int
    psi_face_given   : ndarray (N+1,) or None
        If given, use this as psi_face instead of recomputing CICSAM.
        Pass the VOF-step's psi_face to guarantee mass-flux consistency:
        the continuity equation then sees the SAME face VOF as the VOF step,
        eliminating the ~47 kg/(m³s) inconsistency residual that otherwise
        drives spurious pressure oscillations.

    Returns
    -------
    dict with:
        psi_face    : (N+1,)  CICSAM VOF face values
        rho_face    : (N+1,)  consistent face density
        rhoU_face   : (N+1,)  consistent face momentum flux
        H_face      : (N+1,)  consistent face enthalpy
        u_face_tvd  : (N+1,)  Minmod face velocity (for diagnostics)
    """
    from ..boundary import apply_ghost, apply_ghost_velocity
    from ..interface.cicsam import cicsam_face

    ng = n_ghost
    N  = len(p_k)

    # Extend with ghost cells
    psi_ext     = apply_ghost(psi_new, bc_l, bc_r, ng)
    p_ext       = apply_ghost(p_k,     bc_l, bc_r, ng)
    T_ext       = apply_ghost(T_k,     bc_l, bc_r, ng)
    u_ext       = apply_ghost_velocity(u_k, bc_l, bc_r, ng)

    # Face p, T by arithmetic interpolation (for EOS at faces)
    p_face_arith = np.empty(N + 1)
    T_face_arith = np.empty(N + 1)
    for f in range(N + 1):
        iL = ng + f - 1
        iR = ng + f
        p_face_arith[f] = 0.5 * (p_ext[iL] + p_ext[iR])
        T_face_arith[f] = 0.5 * (T_ext[iL] + T_ext[iR])

    # Ensure physical bounds for face EOS evaluation
    p_face_arith = np.maximum(p_face_arith, 1.0)
    T_face_arith = np.maximum(T_face_arith, 1e-3)

    # Phase properties at faces
    pr1_f = compute_phase_props(p_face_arith, T_face_arith, ph1)
    pr2_f = compute_phase_props(p_face_arith, T_face_arith, ph2)

    # 1. Face VOF: use the VOF-step psi_face if provided (guarantees mass
    #    flux consistency with the explicit VOF advance); otherwise CICSAM.
    if psi_face_given is not None:
        psi_face = psi_face_given.copy()
    elif dt is not None:
        psi_face = cicsam_face(psi_ext, u_face_vol, dt, dx, ng)
    else:
        psi_face = np.empty(N + 1)
        for f in range(N + 1):
            iL = ng + f - 1
            iR = ng + f
            psi_face[f] = 0.5 * (psi_ext[iL] + psi_ext[iR])

    # 2. Consistent face density from CICSAM face VOF
    rho_face_cicsam = consistent_face_rho(psi_face, pr1_f['rho'], pr2_f['rho'])

    # Use ACID for momentum/energy flux weight (physics-based)
    # but CICSAM-consistent density for conservative transport
    rho_face_transport = rho_face_cicsam  # use for momentum flux

    # 3. Consistent face momentum
    rhoU_face, u_face_tvd = consistent_face_momentum(
        rho_face_transport, u_ext, u_face_vol, bc_l, bc_r, ng)

    # 4. Consistent face total enthalpy: H̃_f = ψ̃_f*E1_f + (1-ψ̃_f)*E2_f + KE_f + p̄_f
    #
    # KEY: Use psi_face (= psi_face_vof from VOF step) for phase blending.
    # This ensures that the energy flux at each face reflects the phase
    # distribution ACTUALLY transported by the VOF step, making the temporal
    # term (E_n(psi_n) - E_k(psi_new))/dt cancel exactly with the flux
    # divergence for a uniform (p, T, u) state.
    #
    # Van Leer upwinding of cell-centre E_total gives E_total(psi_new) at
    # the upwind cell, which FAILS this cancellation because psi_new != psi_n
    # at interface cells (e.g. E_air for a cell that was water at psi_n=1
    # but psi_new=0 after VOF). The resulting spurious residual is O(E_water/dt)
    # ≈ 10^10 W/m³ — the T explosion.
    #
    # With face-VOF blending: H̃_f = psi_face_vof * H1_f + (1-psi_face_vof) * H2_f
    # where H1_f and H2_f are per-phase face enthalpies at face (p, T).
    E1_face = pr1_f['E']   # E_int for phase 1 at face (p_f, T_f)
    E2_face = pr2_f['E']   # E_int for phase 2 at face (p_f, T_f)
    # Blended internal energy at face using VOF face values
    E_int_face = psi_face * E1_face + (1.0 - psi_face) * E2_face
    # Kinetic energy at face: consistent density × face velocity magnitude
    ke_face = 0.5 * rho_face_cicsam * u_face_vol**2
    # Total enthalpy = E_int + KE + p
    H_face = E_int_face + ke_face + p_face_arith

    return {
        'psi_face':   psi_face,
        'rho_face':   rho_face_cicsam,
        'rhoU_face':  rhoU_face,
        'H_face':     H_face,
        'u_face_tvd': u_face_tvd,
    }
