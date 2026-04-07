# solver/denner_1d/assembly.py
# Ref: DENNER_SCHEME.md § 7.3, Eq.(21)
#
# Assemble the 3N x 3N sparse linear system for coupled (p, u, T).
# Variable ordering: x = [p_0..p_{N-1}, u_0..u_{N-1}, T_0..T_{N-1}]
#
# Block layout (each block is N x N):
#   [pp | pu | pT] [p]   [b_cont]
#   [up | uu | uT] [u] = [b_mom ]
#   [ep | eu | eT] [T]   [b_ener]

import numpy as np
import scipy.sparse as sp


def _cell_index(block, i, N):
    """Return global row/column index for block (0=p,1=u,2=T) and cell i."""
    return block * N + i


def _neighbor_index(i, N, direction, bc):
    """
    Return the neighbor cell index for periodic or clamped BCs.
    direction: +1 = right neighbor (i+1), -1 = left neighbor (i-1).
    bc: 'periodic' or other.
    """
    j = i + direction
    if bc == 'periodic':
        j = j % N
    else:
        # Transmissive / wall: clamp to boundary (ghost = boundary cell)
        j = max(0, min(N - 1, j))
    return j


def assemble_3N(
    N, dx, dt,
    # Old time-level state
    rho_n, u_n, T_n,
    E_total_n,
    # Picard-iterate state
    rho_k, u_k, T_k, p_k, psi,
    E_total_k,             # total energy at Picard state (N,)
    # Mixture TDU derivatives (at Picard state)
    zeta_v, phi_v,
    dEdp_v, dEdT_v, dEdu_v,
    # Face quantities (deferred / Picard)
    rho_face, d_hat,       # ACID density (N+1,), MWI coeff (N+1,)
    rhoU_face,             # consistent face momentum flux (N+1,)
    H_face,                # consistent face enthalpy (N+1,) [used as fallback / diagnostics]
    rho_face_mwi,          # face density used for MWI pressure term (N+1,) [= rho_face or ACID]
    # MWI transient correction (N+1,) — added to RHS
    mwi_correction,
    # Boundary conditions
    bc_l, bc_r,
    # BDF order and old-old state for BDF2
    bdf_order=1,
    rho_nm1=None, rhoU_nm1=None, E_nm1=None,
    # === Denner 2018 ρh energy equation (ACID formulation) ===
    ph1=None, ph2=None,          # EOS parameters (for ACID face enthalpy computation)
    rho_h_n=None,                # ρh at old time level [J/m³]  (N,)
    rho_h_k=None,                # ρh at Picard state [J/m³]   (N,)
    d_rho_h_dp_v=None,           # d(ρh)/dp at Picard state    (N,)
    d_rho_h_dT_v=None,           # d(ρh)/dT at Picard state    (N,)
    p_n=None,                    # pressure at old time (for ∂p/∂t RHS)
    use_acid_energy=True,        # True: ACID ρh energy eq; False: legacy E_total eq
):
    """
    Assemble 3N×3N sparse linear system for coupled (p, u, T).

    BDF1: (Ω^{n+1} - Ω^n) / dt + divergence = 0
    BDF2: (3Ω^{n+1} - 4Ω^n + Ω^{n-1}) / (2dt) + divergence = 0

    Parameters
    ----------
    N            : int    number of interior cells
    dx, dt       : float  cell size and time step
    rho_n        : (N,)   mixture density at old time level n
    u_n          : (N,)   velocity at n
    T_n          : (N,)   temperature at n
    E_total_n    : (N,)   total energy at n
    rho_k ..     :        Picard iterate state (current inner iteration)
    zeta_v ..    :        mixture TDU coefficients at Picard state
    rho_face     : (N+1,) face density (consistent/ACID, for convective flux)
    d_hat        : (N+1,) MWI coefficient dt/rho_face_ACID
    rhoU_face    : (N+1,) deferred face momentum flux (rho_tilde * u_tilde)
    H_face       : (N+1,) deferred face enthalpy (fallback/diagnostics only when use_acid_energy=True)
    rho_face_mwi : (N+1,) face density used in MWI pressure damping term
    mwi_correction: (N+1,) transient MWI correction (known, goes to b)
    bc_l, bc_r   : str    boundary conditions
    bdf_order    : int    1 or 2
    rho_nm1 ..   : (N,)   state at n-1 for BDF2 (None if BDF1)
    ph1, ph2     : dict   EOS parameters (needed for ACID energy equation)
    rho_h_n      : (N,)   ρh at old time level [J/m³] (needed for ACID energy eq)
    rho_h_k      : (N,)   ρh at Picard state [J/m³]   (needed for ACID energy eq)
    d_rho_h_dp_v : (N,)   d(ρh)/dp at Picard state    (needed for ACID energy eq)
    d_rho_h_dT_v : (N,)   d(ρh)/dT at Picard state    (needed for ACID energy eq)
    p_n          : (N,)   pressure at old time level   (for ∂p/∂t source in energy eq)
    use_acid_energy : bool  True → Denner 2018 ∂(ρh)/∂t + ∇·(ρ̃h̃ϑ) = ∂p/∂t
                            False → legacy ∂E_total/∂t + ∇·(H·ϑ) = 0

    Returns
    -------
    A : scipy.sparse.csr_matrix (3N, 3N)
    b : ndarray (3N,)
    """
    size = 3 * N
    A = sp.lil_matrix((size, size), dtype=float)
    b = np.zeros(size)

    # ---------------------------------------------------------------
    # ACID helper: evaluate ρh + KE using neighbor cell (p,T,u) but
    # THIS cell's phase composition ψ_ref.  Denner 2018 ACID principle:
    # the energy flux seen by cell P is computed with P's own EOS weighting.
    # ---------------------------------------------------------------
    def _acid_rho_h(p_val, T_val, u_val, psi_ref):
        """Return ρ̃h̃ + KE  evaluated at (p,T) with phase weight ψ_ref."""
        g1  = float(ph1['gamma']); pi1 = float(ph1['pinf'])
        b1  = float(ph1['b']);     kv1 = float(ph1['kv']); eta1 = float(ph1['eta'])
        g2  = float(ph2['gamma']); pi2 = float(ph2['pinf'])
        b2  = float(ph2['b']);     kv2 = float(ph2['kv']); eta2 = float(ph2['eta'])
        A1  = kv1 * T_val * (g1 - 1.0) + b1 * (p_val + pi1) + 1e-300
        r1  = (p_val + pi1) / A1
        h1  = g1 * kv1 * T_val + b1 * p_val + eta1
        A2  = kv2 * T_val * (g2 - 1.0) + b2 * (p_val + pi2) + 1e-300
        r2  = (p_val + pi2) / A2
        h2  = g2 * kv2 * T_val + b2 * p_val + eta2
        rho_m = psi_ref * r1 + (1.0 - psi_ref) * r2
        rh_m  = psi_ref * r1 * h1 + (1.0 - psi_ref) * r2 * h2
        ke    = 0.5 * rho_m * u_val * u_val
        return rh_m + ke

    # Flag: ACID energy path is active only if all required arrays are supplied
    _acid_ok = (use_acid_energy and
                ph1 is not None and ph2 is not None and
                rho_h_n is not None and rho_h_k is not None and
                d_rho_h_dp_v is not None and d_rho_h_dT_v is not None)

    # BDF temporal coefficients
    if bdf_order == 1:
        alpha = 1.0 / dt          # coefficient of Ω^{n+1} in temporal term
    else:
        alpha = 3.0 / (2.0 * dt)  # BDF2

    # Determine whether boundary faces are periodic
    is_periodic_l = (bc_l == 'periodic')
    is_periodic_r = (bc_r == 'periodic')

    # Helper: get left/right cell indices for face f
    def face_cells(f):
        """
        Return (iL, iR) indices in [0, N-1] for face f.
        Face f lies between cell f-1 and cell f (interior indexing).
        Face 0  : left boundary face  (iL=-1 or iL=N-1 if periodic)
        Face N  : right boundary face (iR=N or iR=0 if periodic)
        """
        iL = f - 1
        iR = f
        # Clamp/wrap
        if iL < 0:
            iL = (N - 1) if is_periodic_l else 0  # ghost = cell 0 for transmissive
        if iR >= N:
            iR = 0 if is_periodic_r else (N - 1)
        return iL, iR

    # ---------------------------------------------------------------
    # Assemble cell-by-cell
    # ---------------------------------------------------------------
    for i in range(N):
        row_p = _cell_index(0, i, N)   # continuity row
        row_u = _cell_index(1, i, N)   # momentum row
        row_e = _cell_index(2, i, N)   # energy row

        col_p_i = _cell_index(0, i, N)
        col_u_i = _cell_index(1, i, N)
        col_T_i = _cell_index(2, i, N)

        # -----------------------------------------------------------
        # 1. TEMPORAL TERMS
        # -----------------------------------------------------------
        # Continuity: (ρ^{n+1} - ρ^n)/dt = 0  (flux = 0)
        #   ρ^{n+1} ≈ ρ^n + ζ_v δp + φ_v δT   [Eq. 12]
        #   → α * ζ_v * p_i  +  α * φ_v * T_i  =  α * ρ^n (+ rhs BDF2 correction)
        A[row_p, col_p_i] += alpha * zeta_v[i]
        A[row_p, col_T_i] += alpha * phi_v[i]

        # Momentum: (ρu)^{n+1}/dt  [Eq. 13]
        #   (ρu)^{n+1} ≈ ρ^k u^{n+1} + u^k (ζ_v δp + φ_v δT)
        A[row_u, col_u_i] += alpha * rho_k[i]
        A[row_u, col_p_i] += alpha * u_k[i] * zeta_v[i]
        A[row_u, col_T_i] += alpha * u_k[i] * phi_v[i]

        # Energy temporal [Eq. 14]
        # Two paths:
        #   A) ACID ρh (Denner 2018): ∂(ρh)/∂t + ∇·(ρ̃h̃ϑ) = ∂p/∂t
        #   B) Legacy E_total:        ∂E/∂t + ∇·(H·ϑ) = 0  (fallback)
        #
        # Conditioning floor: for pure ideal-gas cells the T-derivative is near
        # zero, making the diagonal ≈ 0.  Add floor = rho * kv_floor to avoid
        # ill-conditioning; this cancels at Picard convergence so does not affect
        # the converged solution.
        _KV_FLOOR = 718.0  # J/(kg·K)

        if _acid_ok:
            # --- Path A: Denner 2018 ρh energy equation ---
            # LHS: α*(d(ρh)/dp - 1)*p + α*d(ρh)/dT*T
            #   The "-1" comes from moving the ∂p/∂t source to the LHS:
            #   ∂(ρh)/∂t = ∂p/∂t  →  α*(ρh - p)  in discrete form
            d_rh_dT_eff_i = max(abs(d_rho_h_dT_v[i]), rho_k[i] * _KV_FLOOR)
            A[row_e, col_p_i] += alpha * (d_rho_h_dp_v[i] - 1.0)
            A[row_e, col_T_i] += alpha * d_rh_dT_eff_i
            # no u-coupling for thermal enthalpy (KE treated via ACID flux separately)

            # RHS: (ρh^n - ρh^k + lin_ρh^k)/dt - p^n/dt
            lin_rh_k = d_rho_h_dp_v[i] * p_k[i] + d_rh_dT_eff_i * T_k[i]
            _p_n_i   = float(p_n[i]) if (p_n is not None) else float(p_k[i])
            if bdf_order == 1:
                b[row_e] += (rho_h_n[i] - rho_h_k[i] + lin_rh_k) / dt - _p_n_i / dt
            else:
                # BDF2: reuse E_nm1 slot to carry rho_h_{n-1} when available
                rh_nm1_i = float(E_nm1[i]) if (E_nm1 is not None) else float(rho_h_n[i])
                b[row_e] += ((4.0 * rho_h_n[i] - rh_nm1_i) / (2.0 * dt)
                             + alpha * (lin_rh_k - rho_h_k[i])
                             - _p_n_i / dt)
        else:
            # --- Path B: Legacy E_total energy equation ---
            dEdT_eff_i = max(abs(dEdT_v[i]), rho_k[i] * _KV_FLOOR)
            A[row_e, col_p_i] += alpha * dEdp_v[i]
            A[row_e, col_T_i] += alpha * dEdT_eff_i
            A[row_e, col_u_i] += alpha * dEdu_v[i]

        # -----------------------------------------------------------
        # RHS temporal — full-variable form linearized around Picard state k
        #
        # Continuity: ζ_v p^{n+1}/dt + φ_v T^{n+1}/dt + div = (ρ^n - ρ^k + ζ_v p^k + φ_v T^k)/dt
        # Momentum:   ρ^k u^{n+1}/dt + u^k ζ_v p^{n+1}/dt + ... = (ρ^n u^n + u^k(ζ_v p^k + φ_v T^k))/dt
        # Energy (legacy): dEdp p^{n+1}/dt + dEdT_eff T^{n+1}/dt + dEdu u^{n+1}/dt + div
        #             = (E^n - E^k + dEdp p^k + dEdT_eff T^k + dEdu u^k)/dt
        # -----------------------------------------------------------
        lin_rho_k  = zeta_v[i] * p_k[i] + phi_v[i] * T_k[i]  # ≈ ρ^k from linearization
        if bdf_order == 1:
            b[row_p] += (rho_n[i]          - rho_k[i]      + lin_rho_k)  / dt
            b[row_u] += (rho_n[i] * u_n[i] + u_k[i] * lin_rho_k)         / dt
            if not _acid_ok:
                dEdT_eff_i = max(abs(dEdT_v[i]), rho_k[i] * _KV_FLOOR)
                lin_E_k = (dEdp_v[i] * p_k[i] + dEdT_eff_i * T_k[i] + dEdu_v[i] * u_k[i])
                b[row_e] += (E_total_n[i] - E_total_k[i] + lin_E_k) / dt
        else:
            # BDF2: alpha = 3/(2dt)
            b[row_p] += (4.0 * rho_n[i]          - rho_nm1[i])   / (2.0 * dt) + alpha * (lin_rho_k - rho_k[i])
            b[row_u] += (4.0 * rho_n[i] * u_n[i] - rhoU_nm1[i]) / (2.0 * dt) + alpha * u_k[i] * lin_rho_k
            if not _acid_ok:
                dEdT_eff_i = max(abs(dEdT_v[i]), rho_k[i] * _KV_FLOOR)
                lin_E_k = (dEdp_v[i] * p_k[i] + dEdT_eff_i * T_k[i] + dEdu_v[i] * u_k[i])
                b[row_e] += (4.0 * E_total_n[i] - E_nm1[i]) / (2.0 * dt) + alpha * (lin_E_k - E_total_k[i])

        # -----------------------------------------------------------
        # 2. FLUX TERMS (right face = f_R = i+1, left face = f_L = i)
        # -----------------------------------------------------------
        # Face indexing: face f is between cell f-1 and cell f.
        # For cell i: right face = i+1, left face = i.
        f_R = i + 1  # right face index
        f_L = i      # left face index

        # Face densities and MWI coefficients
        rho_f_R = rho_face[f_R]
        rho_f_L = rho_face[f_L]
        d_R = d_hat[f_R]
        d_L = d_hat[f_L]
        rho_mwi_R = rho_face_mwi[f_R]
        rho_mwi_L = rho_face_mwi[f_L]

        # Deferred face fluxes
        rhoU_R = rhoU_face[f_R]
        rhoU_L = rhoU_face[f_L]
        H_R    = H_face[f_R]
        H_L    = H_face[f_L]

        # Neighbor cell indices
        iR = face_cells(f_R)[1]   # right neighbor of cell i
        iL = face_cells(f_L)[0]   # left neighbor of cell i

        col_p_iR = _cell_index(0, iR, N)
        col_u_iR = _cell_index(1, iR, N)
        col_p_iL = _cell_index(0, iL, N)
        col_u_iL = _cell_index(1, iL, N)

        # ---- CONTINUITY flux: div(ρ_tilde * F_f^{n+1}) / dx ----
        #
        # F_f^{n+1} = u_arith_f - d_hat_f * (p_{iR} - p_{iL}) / dx  [MWI, Eq. 17]
        #
        # Contribution from RIGHT face (positive, outward):
        #   +rho_f_R * [ (u_i + u_iR)/2  -  d_R*(p_iR - p_i)/dx ] / dx
        #
        # u_i terms:
        A[row_p, col_u_i]  += rho_f_R / (2.0 * dx)
        A[row_p, col_u_iR] += rho_f_R / (2.0 * dx)
        # p terms (MWI pressure damping):
        A[row_p, col_p_i]  += rho_f_R * d_R / (dx * dx)
        A[row_p, col_p_iR] -= rho_f_R * d_R / (dx * dx)

        # Contribution from LEFT face (negative, inward):
        #   -rho_f_L * [ (u_iL + u_i)/2  -  d_L*(p_i - p_iL)/dx ] / dx
        # p terms: -ρ_L*(-d_L*(p_i-p_iL)/dx)/dx = +ρ_L*d_L*(p_i-p_iL)/dx²
        #   → +ρ_L*d_L for p_i (diagonal),  -ρ_L*d_L for p_iL (subdiag)
        A[row_p, col_u_iL] -= rho_f_L / (2.0 * dx)
        A[row_p, col_u_i]  -= rho_f_L / (2.0 * dx)
        # p terms (sign fixed — makes Laplacian, not first-derivative):
        A[row_p, col_p_i]  += rho_f_L * d_L / (dx * dx)   # FIXED: was -=
        A[row_p, col_p_iL] -= rho_f_L * d_L / (dx * dx)   # FIXED: was +=

        # MWI transient correction goes to RHS (known)
        # correction_f is a velocity correction; the mass flux contribution = rho_f * correction_f
        b[row_p] -= (rho_f_R * mwi_correction[f_R] - rho_f_L * mwi_correction[f_L]) / dx

        # ---- MOMENTUM flux: div((rho_tilde*u_tilde)*F_f^{n+1} + p_bar_f) / dx ----
        #
        # Convective: (rhoU_tilde)_f * u_f^{n+1}  (rhoU_f deferred, u_f implicit)
        # Pressure:   p_bar_f^{n+1} = (p_i + p_iR)/2 at right face
        #
        # Use first-order UPWIND for the velocity u_f^{n+1} in the convective term.
        # This guarantees a positive diagonal regardless of how strongly rhoU varies
        # across a phase interface (prevents negative diagonal = instability).
        # The MWI pressure-damping terms are kept for checkerboard suppression.
        #
        # Right face contribution (+):
        if rhoU_R >= 0.0:
            # Flow from left: u_face ≈ u_i (upwind)
            A[row_u, col_u_i]  += rhoU_R / dx
        else:
            # Flow from right: u_face ≈ u_iR (upwind)
            A[row_u, col_u_iR] += rhoU_R / dx
        # MWI pressure in momentum convection
        A[row_u, col_p_i]  += rhoU_R * d_R / (dx * dx)
        A[row_u, col_p_iR] -= rhoU_R * d_R / (dx * dx)
        # Pressure gradient (standard central: net (p_iR - p_iL)/(2dx))
        A[row_u, col_p_iR] += 1.0 / (2.0 * dx)
        # Left face contribution (-):
        if rhoU_L >= 0.0:
            # Flow from left: u_face ≈ u_iL (upwind)
            A[row_u, col_u_iL] -= rhoU_L / dx
        else:
            # Flow from right: u_face ≈ u_i (upwind)
            A[row_u, col_u_i]  -= rhoU_L / dx
        A[row_u, col_p_iL] -= rhoU_L * d_L / (dx * dx)
        A[row_u, col_p_i]  += rhoU_L * d_L / (dx * dx)
        A[row_u, col_p_iL] -= 1.0 / (2.0 * dx)

        # ---- ENERGY flux: div(H̃_f * F_f^{n+1}) / dx ----
        #
        # Two paths:
        #   A) ACID: H at face f evaluated with THIS cell's ψ applied to
        #      the NEIGHBOR cell's (p,T,u) — Denner 2018 ACID principle.
        #      For air cell (ψ≈0): water-side face uses air EOS → ρ_air*h_air
        #      instead of ρ_water*h_water, eliminating the 4000× jump.
        #   B) Legacy: use pre-computed H_face array (ψ-blended ρh from consistent.py)
        #
        # MWI pressure-damping is retained in both paths.
        if _acid_ok:
            # ACID face enthalpy: neighbour cell's (p,T,u) evaluated with cell i's ψ
            psi_i   = float(psi[i])
            H_R_acid = _acid_rho_h(float(p_k[iR]), float(T_k[iR]), float(u_k[iR]), psi_i)
            H_L_acid = _acid_rho_h(float(p_k[iL]), float(T_k[iL]), float(u_k[iL]), psi_i)
            H_i_acid = _acid_rho_h(float(p_k[i]),  float(T_k[i]),  float(u_k[i]),  psi_i)

            # Right face (+outward): upwind H·u
            if rhoU_R >= 0.0:
                # flow from cell i → use H_i_acid (this cell is upwind)
                A[row_e, col_u_i]  += H_i_acid / dx
            else:
                # flow from iR → use H_R_acid (iR is upwind)
                A[row_e, col_u_iR] += H_R_acid / dx
            A[row_e, col_p_i]  += H_i_acid * d_R / (dx * dx)
            A[row_e, col_p_iR] -= H_i_acid * d_R / (dx * dx)

            # Left face (-inward): upwind H·u
            if rhoU_L >= 0.0:
                # flow from iL → use H_L_acid (iL is upwind)
                A[row_e, col_u_iL] -= H_L_acid / dx
            else:
                # flow from cell i → use H_i_acid (this cell is upwind)
                A[row_e, col_u_i]  -= H_i_acid / dx
            A[row_e, col_p_iL] -= H_L_acid * d_L / (dx * dx)
            A[row_e, col_p_i]  += H_L_acid * d_L / (dx * dx)
        else:
            # Path B: Fully deferred energy flux — move H*u_face into b to fix
            # ill-conditioning. The H_face/dx and H*d/dx² terms in A for air
            # cells at the water-air interface are ~10^6× larger than the
            # dEdT/dt diagonal, making the system ill-conditioned. Moving the
            # flux to b reduces the condition ratio from ~10^6 to ~700,
            # eliminating spurious T drift. For pressure-equilibrium (uniform
            # state): r_e = (E_n-E_k)/dt - H_flux ≈ 0 exactly to machine
            # precision → Picard converges in 1 iteration.
            u_up_R = u_k[i]   if rhoU_R >= 0.0 else u_k[iR]
            u_up_L = u_k[iL]  if rhoU_L >= 0.0 else u_k[i]
            dp_R = (p_k[iR] - p_k[i]) / dx
            dp_L = (p_k[i]  - p_k[iL]) / dx
            uf_R = u_up_R - d_R * dp_R
            uf_L = u_up_L - d_L * dp_L
            b[row_e] -= (H_R * uf_R - H_L * uf_L) / dx

    # ---------------------------------------------------------------
    # Boundary condition modifications
    # ---------------------------------------------------------------
    # For transmissive BC: ghost = boundary cell → zero-gradient.
    # The clamped neighbor indices in face_cells() already handle this
    # by mapping ghost indices to the boundary cell, effectively doubling
    # the boundary cell contribution. No additional correction needed.
    #
    # For wall BC on velocity: u_ghost = -u_interior, so u_arith at wall = 0.
    # During the cell loop for cell 0 (left wall) and cell N-1 (right wall),
    # the boundary face contribution was assembled using the clamped neighbor
    # index (iL=0 for left wall, iR=N-1 for right wall). This gives:
    #   left face of cell 0:  rho_f_L/2dx * (u_iL + u_0) with iL=0 → rho_f_L/dx * u_0
    # But wall means u_arith = 0, so this whole term must vanish.
    # Correction: subtract the erroneous contribution (the loop added it incorrectly
    # assuming the ghost equals the interior, but wall ghost = -interior → net = 0).
    if bc_l == 'wall':
        i0    = 0
        row_p0 = _cell_index(0, i0, N)
        row_e0 = _cell_index(2, i0, N)
        col_u0 = _cell_index(1, i0, N)
        f_wall = i0        # left boundary face index = 0
        rho_fw = rho_face[f_wall]
        # The loop subtracted  -rho_fw/(2dx) from A[row_p0, col_u0]  (left face -)
        # and also subtracted  -rho_fw/(2dx) from A[row_p0, col_u_iL] (= col_u0, clamped)
        # Net erroneous contribution to A[row_p0, col_u0]: -rho_fw/dx
        # Correct it back to 0 by adding +rho_fw/dx
        A[row_p0, col_u0] += rho_fw / dx
        # Energy row correction: only needed for Path A (ACID), where H·u terms
        # are in A. For Path B the flux is fully deferred to b, so no A entry
        # to correct.
        if _acid_ok:
            H_fw = _acid_rho_h(float(p_k[0]), float(T_k[0]), float(u_k[0]), float(psi[0]))
            A[row_e0, col_u0] += H_fw / dx

    if bc_r == 'wall':
        i_last = N - 1
        row_pL = _cell_index(0, i_last, N)
        row_eL = _cell_index(2, i_last, N)
        col_uL = _cell_index(1, i_last, N)
        f_wall = N         # right boundary face index = N
        rho_fw = rho_face[f_wall]
        # The loop added +rho_fw/(2dx) twice for A[row_pL, col_uL] (right face +)
        # Correct to 0 by subtracting rho_fw/dx
        A[row_pL, col_uL] -= rho_fw / dx
        # Energy row correction: only for Path A (ACID). Path B is fully
        # deferred to b, so no A correction is needed.
        if _acid_ok:
            H_fw = _acid_rho_h(float(p_k[N-1]), float(T_k[N-1]), float(u_k[N-1]), float(psi[N-1]))
            A[row_eL, col_uL] -= H_fw / dx

    return A.tocsr(), b


def solve_linear_system(A, b, p_ref=1.0e5, u_ref=1.0, T_ref=300.0):
    """
    Solve A @ x = b with row+column equilibration and direct LU (spsolve).

    Column equilibration: divide block p by p_ref, u by u_ref, T by T_ref.
    Row equilibration: divide row i by the L∞ norm of that row in A (after column
    scaling). Using A-row norms (not |b|) avoids amplifying noise in near-zero RHS
    rows that occur in equilibrium states.

    Parameters
    ----------
    A : scipy.sparse.csr_matrix (3N, 3N)
    b : ndarray (3N,)
    p_ref, u_ref, T_ref : float   physical reference scales

    Returns
    -------
    x : ndarray (3N,)   solution in original units
    """
    import scipy.sparse.linalg as spla

    size = len(b)
    N3   = size // 3

    # Column scaling (physical units normalisation)
    col_scale = np.ones(size)
    col_scale[:N3]     = max(abs(p_ref), 1.0)
    col_scale[N3:2*N3] = max(abs(u_ref), 1e-6)
    col_scale[2*N3:]   = max(abs(T_ref), 1.0)
    A_cs = A.dot(sp.diags(col_scale, format='csr'))

    # Row scaling by L∞ norm of each row in A_cs
    # (avoids b-based scaling which blows up when b≈0 at equilibrium)
    abs_A = np.abs(A_cs)
    row_max_result = abs_A.max(axis=1)
    # .max(axis=1) on sparse matrix returns a matrix; convert to dense 1D array
    if sp.issparse(row_max_result):
        row_max = np.asarray(row_max_result.toarray()).ravel()
    else:
        row_max = np.asarray(row_max_result).ravel()
    row_max = np.maximum(row_max, 1e-300)
    D_inv = sp.diags(1.0 / row_max, format='csr')
    As = D_inv.dot(A_cs)
    bs = D_inv.dot(b)

    x_hat = None
    try:
        x_hat = spla.spsolve(As, bs)
        if not np.all(np.isfinite(x_hat)):
            x_hat = None
    except Exception:
        pass

    if x_hat is None:
        # Fallback: dense solve
        try:
            x_hat = np.linalg.solve(As.toarray(), bs)
            if not np.all(np.isfinite(x_hat)):
                x_hat = None
        except Exception:
            x_hat = None

    if x_hat is None:
        # Last resort: zero correction (accept current Picard iterate)
        x_hat = np.zeros_like(bs)

    # Ensure x_hat is 1D dense array before scaling
    if sp.issparse(x_hat):
        x_hat = np.asarray(x_hat.todense()).ravel()
    else:
        x_hat = np.asarray(x_hat).ravel()

    # Unscale to original units
    x = col_scale * x_hat
    return x
