# solver/denner_1d/assembly.py
# Ref: Denner 2018 — Newton linearisation, Eqs. 25, 29, 30
#
# Variable ordering: x = [p_0..p_{N-1}, u_0..u_{N-1}, h_0..h_{N-1}]
#
# System: A*x = b.  Solver: r = b - A*x_k; solve A*δx = r; x += δx.
#
# Discrete BDF1 equations:
#   Continuity:  ζ*(p^{n+1}-p^n)/dt + div(ρ̃·ϑ^{n+1}) = 0
#   Momentum:    Newton[ρu]/dt + div(ρ̃·ϑ·u^{n+1}) + ∇p^{n+1} = 0
#   Enthalpy:    Newton[ρh]/dt + div(ρ̃·ϑ·h^{n+1}) - ∂p/∂t = 0
#
# Newton linearisation of ρ·χ around iterate (ρ_k, χ_k), Eq. 29:
#   ρ^{n+1}·χ^{n+1} ≈ ρ_k·χ^{n+1} + ζ_k·(p^{n+1}-p_k)·χ_k
#
# MWI face velocity (implicit in u and p), Eq. 20:
#   ϑ_f^{n+1} = ū_f^{n+1} − d̂_f·∇p_f^{n+1}
#   → contributions to A from u_L, u_R (arithmetic mean) and p_L, p_R (d̂·Laplacian)

import numpy as np
import scipy.sparse as sp
from .eos.eos_class import create_eos


def _ci(block, i, N):
    return block * N + i


def assemble_newton_3N(
    N, dx, dt,
    rho_old, u_old, h_old, p_old,   # old-time (n) quantities
    rho_k, u_k, h_k, p_k, T_k, psi_k,
    zeta_k,           # dρ/dp at iterate  (N,)
    rho_face_acid,    # ACID face density  (N+1,)
    d_hat,            # MWI coefficient   (N+1,)
    theta_k,          # face velocity at x_k (N+1,)
    ph1, ph2,
    bc_l, bc_r,
    freeze_h=False,
    third_var='h',    # 'h' = (p,u,h), 'T' = (p,u,T)
    T_old=None,       # needed when third_var='T'
    phi_k=None,       # dρ/dT (needed for third_var='T')
    mixing_type='volume',  # 'volume' (ψ-based) or 'mass' (Y-based) ACID helpers
):
    """
    Newton-linearised (p, u, h) system.
    Returns A (csr), b (ndarray).
    """
    size = 3 * N
    A = sp.lil_matrix((size, size), dtype=float)
    b = np.zeros(size)

    is_per_l = (bc_l == 'periodic')
    is_per_r = (bc_r == 'periodic')

    def face_lr(f):
        iL = f - 1
        iR = f
        iL = (N - 1 if is_per_l else 0) if iL < 0 else iL
        iR = (0 if is_per_r else N - 1) if iR >= N else iR
        return iL, iR

    # ACID EOS helpers: evaluate partial densities/enthalpies at (p,T) with ψ_ref
    # Build EOS objects once; works for any EOS (NASG, RKPR, etc.)
    eos1 = create_eos(ph1)
    eos2 = create_eos(ph2)

    def _acid_rho(p_val, T_val, psi_ref):
        """ACID density at (p_val,T_val) with this-cell ψ_ref (Eq. 37)."""
        r1 = eos1.rho(p_val, T_val)
        r2 = eos2.rho(p_val, T_val)
        return psi_ref * r1 + (1.0 - psi_ref) * r2

    def _acid_rh(p_val, T_val, u_val, psi_ref):
        """ACID ρH_total at (p,T,u) with this-cell ψ_ref (Denner 2018 Eq. 45-49).
        H = ρ★·h★ where h★ = h_static + ½u² (total specific enthalpy)."""
        r1 = eos1.rho(p_val, T_val)
        h1 = eos1.h(p_val, T_val) + 0.5 * u_val * u_val
        r2 = eos2.rho(p_val, T_val)
        h2 = eos2.h(p_val, T_val) + 0.5 * u_val * u_val
        return psi_ref * r1 * h1 + (1.0 - psi_ref) * r2 * h2

    def _acid_cp(p_val, T_val, psi_ref):
        """ACID mixture cp (Denner 2018 Eq. 46): density-weighted average."""
        r1 = eos1.rho(p_val, T_val)
        r2 = eos2.rho(p_val, T_val)
        rho_mix = psi_ref * r1 + (1.0 - psi_ref) * r2 + 1e-300
        return (psi_ref * r1 * eos1.cp(p_val, T_val) +
                (1.0 - psi_ref) * r2 * eos2.cp(p_val, T_val)) / rho_mix

    def _acid_bm(p_val, T_val, psi_ref):
        """ACID mixture b_mix = dh_static/dp (density-weighted)."""
        r1 = eos1.rho(p_val, T_val)
        r2 = eos2.rho(p_val, T_val)
        rho_mix = psi_ref * r1 + (1.0 - psi_ref) * r2 + 1e-300
        return (psi_ref * r1 * eos1.dh_dp(p_val, T_val) +
                (1.0 - psi_ref) * r2 * eos2.dh_dp(p_val, T_val)) / rho_mix

    # --- Y-based (mass fraction) ACID helpers ---
    def _acid_rho_Y(p_val, T_val, Y_ref):
        """Harmonic mixture density: 1/(Y/r1 + (1-Y)/r2)."""
        r1 = eos1.rho(p_val, T_val)
        r2 = eos2.rho(p_val, T_val)
        inv_rho = Y_ref / (r1 + 1e-300) + (1.0 - Y_ref) / (r2 + 1e-300)
        return 1.0 / (inv_rho + 1e-300)

    def _acid_rh_Y(p_val, T_val, u_val, Y_ref):
        """Mass-weighted total enthalpy density: rho_star * (Y*h1 + (1-Y)*h2 + 0.5*u^2)."""
        r1 = eos1.rho(p_val, T_val)
        h1 = eos1.h(p_val, T_val)
        r2 = eos2.rho(p_val, T_val)
        h2 = eos2.h(p_val, T_val)
        inv_rho = Y_ref / (r1 + 1e-300) + (1.0 - Y_ref) / (r2 + 1e-300)
        rho_star = 1.0 / (inv_rho + 1e-300)
        h_static = Y_ref * h1 + (1.0 - Y_ref) * h2
        return rho_star * (h_static + 0.5 * u_val * u_val)

    def _acid_cp_Y(p_val, T_val, Y_ref):
        """Mass-weighted mixture cp: Y*cp₁ + (1-Y)*cp₂."""
        return Y_ref * eos1.cp(p_val, T_val) + (1.0 - Y_ref) * eos2.cp(p_val, T_val)

    def _acid_bm_Y(p_val, T_val, Y_ref):
        """Mass-weighted mixture b_mix: Y*∂h₁/∂p + (1-Y)*∂h₂/∂p."""
        return Y_ref * eos1.dh_dp(p_val, T_val) + (1.0 - Y_ref) * eos2.dh_dp(p_val, T_val)

    for i in range(N):
        rp = _ci(0, i, N)
        ru = _ci(1, i, N)
        rh_row = _ci(2, i, N)
        cp = _ci(0, i, N)
        cu = _ci(1, i, N)
        ch = _ci(2, i, N)

        f_R = i + 1
        f_L = i
        iL, _ = face_lr(f_L)
        _, iR  = face_lr(f_R)

        cp_L = _ci(0, iL, N);  cu_L = _ci(1, iL, N);  ch_L = _ci(2, iL, N)
        cp_R = _ci(0, iR, N);  cu_R = _ci(1, iR, N);  ch_R = _ci(2, iR, N)

        rho_i  = rho_k[i]
        zeta_i = zeta_k[i]
        u_i    = u_k[i]
        h_i    = h_k[i]
        psi_i  = float(psi_k[i])   # volume fraction (or Y if mixing_type='mass')

        tR = theta_k[f_R]
        tL = theta_k[f_L]
        dR  = d_hat[f_R]
        dL  = d_hat[f_L]

        # ACID face density: computed with cell i's ψ (or Y) applied to neighbor's (p,T)
        # At uniform (p,T): rfR = rfL = _acid_rho(p, T, psi_i) → div=0 ✓
        if mixing_type == 'mass':
            rfR = _acid_rho_Y(float(p_k[iR]), float(T_k[iR]), psi_i)
            rfL = _acid_rho_Y(float(p_k[iL]), float(T_k[iL]), psi_i)
        else:
            rfR = _acid_rho(float(p_k[iR]), float(T_k[iR]), psi_i)
            rfL = _acid_rho(float(p_k[iL]), float(T_k[iL]), psi_i)

        # Deferred mass fluxes at x_k
        mR = rfR * tR
        mL = rfL * tL

        # -----------------------------------------------------------
        # CONTINUITY
        # -----------------------------------------------------------
        # Discrete: ζ*(p^{n+1}-p^n)/dt + div(ρ̃·ϑ^{n+1}) = 0
        # A*x = b:  ζ/dt·p_i + div_impl(u,p) = ζ/dt·p^n
        A[rp, cp]   += zeta_i / dt
        b[rp]       += zeta_i * p_old[i] / dt
        # MWI right face: ρ̃_fR·ϑ_fR / dx
        A[rp, cu]   += rfR / (2.0 * dx)
        A[rp, cu_R] += rfR / (2.0 * dx)
        A[rp, cp]   += rfR * dR / (dx * dx)
        A[rp, cp_R] -= rfR * dR / (dx * dx)
        # MWI left face: −ρ̃_fL·ϑ_fL / dx
        A[rp, cu_L] -= rfL / (2.0 * dx)
        A[rp, cu]   -= rfL / (2.0 * dx)
        A[rp, cp]   += rfL * dL / (dx * dx)
        A[rp, cp_L] -= rfL * dL / (dx * dx)

        # -----------------------------------------------------------
        # MOMENTUM
        # -----------------------------------------------------------
        # Discrete: [ρ_k·u^{n+1} + ζ·(p^{n+1}-p_k)·u_k − ρ^n·u^n]/dt
        #           + div(ρ̃·ϑ·u^{n+1}) + ∇p^{n+1} = 0
        # A*x = b:  ρ_k/dt·u_i + ζ·u_k/dt·p_i + conv + ∇p = ρ^n·u^n/dt + ζ·u_k/dt·p_k
        A[ru, cu] += rho_i / dt
        A[ru, cp] += zeta_i * u_i / dt
        b[ru]     += rho_old[i] * u_old[i] / dt + zeta_i * u_i * p_k[i] / dt

        # Convective: ρ̃·ϑ_k·u^{n+1} (upwind u) + ρ̃·u_k·ϑ^{n+1,p-part}
        # Right face (+):
        if mR >= 0.0:
            A[ru, cu]  += mR / dx
        else:
            A[ru, cu_R] += mR / dx
        # MWI implicit p part: ρ̃·u_k·(−d̂·(p_R−p_L)/dx) / dx
        A[ru, cp]   += rfR * u_i * dR / (dx * dx)
        A[ru, cp_R] -= rfR * u_i * dR / (dx * dx)
        # Left face (−):
        if mL >= 0.0:
            A[ru, cu_L] -= mL / dx
        else:
            A[ru, cu]   -= mL / dx
        A[ru, cp_L] -= rfL * u_i * dL / (dx * dx)
        A[ru, cp]   += rfL * u_i * dL / (dx * dx)

        # Pressure gradient: −(p_R − p_L)/(2dx)  [note: +1/(2dx) to right, −1/(2dx) to left]
        A[ru, cp_R] += 1.0 / (2.0 * dx)
        A[ru, cp_L] -= 1.0 / (2.0 * dx)

        # -----------------------------------------------------------
        # ENERGY EQUATION (block 2)
        # -----------------------------------------------------------
        if freeze_h:
            # Inner barotropic loop: third variable frozen (identity row)
            A[rh_row, ch] = 1.0
            b[rh_row]     = h_k[i]
        elif third_var == 'h':
            # --- (p, u, h) mode: Denner 2018 enthalpy equation ---
            # (ρ^{n+1}h^{n+1} - ρ^n·h^n)/dt + div(ρ̃ϑh) = (p^{n+1}-p^n)/dt
            # Newton: ρ_k/dt·h + (ζ·h_k - 1)/dt·p = ρ^n·h^n/dt - p^n/dt + ζ·h_k·p_k/dt
            A[rh_row, ch] += rho_i / dt
            A[rh_row, cp] += zeta_i * h_i / dt - 1.0 / dt
            b[rh_row]     += (rho_old[i] * h_old[i] / dt
                              - p_old[i] / dt
                              + zeta_i * h_i * p_k[i] / dt)
            # Convective + ACID
            if mixing_type == 'mass':
                H_R_acid = _acid_rh_Y(float(p_k[iR]), float(T_k[iR]), float(u_k[iR]), psi_i)
                H_L_acid = _acid_rh_Y(float(p_k[iL]), float(T_k[iL]), float(u_k[iL]), psi_i)
            else:
                H_R_acid = _acid_rh(float(p_k[iR]), float(T_k[iR]), float(u_k[iR]), psi_i)
                H_L_acid = _acid_rh(float(p_k[iL]), float(T_k[iL]), float(u_k[iL]), psi_i)
            h_up_R = h_k[i]   if mR >= 0.0 else h_k[iR]
            h_up_L = h_k[iL]  if mL >= 0.0 else h_k[i]
            if mR >= 0.0: A[rh_row, ch]   += mR / dx
            else:         A[rh_row, ch_R] += mR / dx
            if mL >= 0.0: A[rh_row, ch_L] -= mL / dx
            else:         A[rh_row, ch]   -= mL / dx
            acid_corr_R = (H_R_acid - rfR * h_up_R) * tR / dx
            acid_corr_L = (H_L_acid - rfL * h_up_L) * tL / dx
            b[rh_row] -= (acid_corr_R - acid_corr_L)
        else:
            # --- (p, u, T) mode: same ρh energy eq but T is variable ---
            # h_total = cp_mix·T + ½u² → linearize: h ≈ cp·T + u_k·u - ½u_k²
            # Temporal: ρ^{n+1}·h^{n+1} ≈ ρ_k·(cp·T + u_k·u) + (ζ·δp + φ·δT)·h_k
            # Full Newton product (ρ·h): ρ_k·h^{n+1} + ρ^{n+1}·h_k - ρ_k·h_k
            #   where ρ^{n+1} = ρ_k + ζ·δp + φ·δT
            #   and   h^{n+1} = cp·T + u_k·u - ½u_k² (linearized around x_k)
            #
            # T-coefficient: ρ_k·cp/dt + h_k·φ/dt   (≈ ρ_k·cp for ideal gas)
            # u-coefficient: ρ_k·u_k/dt               (from d(½u²)/du)
            # p-coefficient: (ζ·h_k - 1)/dt           (from Newton ρ·h + dp/dt)
            phi_i = float(phi_k[i]) if phi_k is not None else 0.0
            if mixing_type == 'mass':
                cp_i = _acid_cp_Y(float(p_k[i]), float(T_k[i]), psi_i)
                bm_i = _acid_bm_Y(float(p_k[i]), float(T_k[i]), psi_i)
            else:
                cp_i = _acid_cp(float(p_k[i]), float(T_k[i]), psi_i)
                bm_i = _acid_bm(float(p_k[i]), float(T_k[i]), psi_i)
            T_i   = T_k[i]

            # Newton product-rule linearization of ρ·h where h = cp*T + b*p + η + ½u²
            # d(ρh)/dT = ρ_k·cp + h_k·φ (= ρ·dh/dT + h·dρ/dT)
            # d(ρh)/dp = ρ_k·b_mix + h_k·ζ (= ρ·dh/dp + h·dρ/dp)
            # d(ρh)/du = ρ_k·u_k
            # Full eq: d(ρh)/dT·T + d(ρh)/dp·p + d(ρh)/du·u − dp/dt
            #        = ρ^n·h^n + [d(ρh)/dT·T_k + d(ρh)/dp·p_k + d(ρh)/du·u_k − ρ_k·h_k] − p^n/dt
            drhdt = rho_i * cp_i + h_i * phi_i
            drhdp = rho_i * bm_i + h_i * zeta_i
            drhdu = rho_i * u_i

            A[rh_row, ch] += drhdt / dt             # T column
            A[rh_row, cu] += drhdu / dt              # u coupling from ½u²
            A[rh_row, cp] += (drhdp - 1.0) / dt     # p coupling (−1 from dp/dt source)
            b[rh_row]     += (rho_old[i] * h_old[i] / dt
                              - p_old[i] / dt
                              + (drhdt * T_i + drhdp * p_k[i] + drhdu * u_i
                                 - rho_i * h_i) / dt)

            # Convective for T-mode: use ACID face enthalpy DIRECTLY.
            # Split H_acid = rfR·(cp_i·T + rest). Implicit: cp_i·T; deferred: rest.
            # At uniform (p,T,u): both faces give same H_acid → net flux = 0 ✓
            if mixing_type == 'mass':
                H_R_acid = _acid_rh_Y(float(p_k[iR]), float(T_k[iR]), float(u_k[iR]), psi_i)
                H_L_acid = _acid_rh_Y(float(p_k[iL]), float(T_k[iL]), float(u_k[iL]), psi_i)
                cp_i_acid = _acid_cp_Y(float(p_k[i]), float(T_k[i]), psi_i)
            else:
                H_R_acid = _acid_rh(float(p_k[iR]), float(T_k[iR]), float(u_k[iR]), psi_i)
                H_L_acid = _acid_rh(float(p_k[iL]), float(T_k[iL]), float(u_k[iL]), psi_i)
                cp_i_acid = _acid_cp(float(p_k[i]), float(T_k[i]), psi_i)
            # Full ACID flux deferred to b:
            b[rh_row] -= (H_R_acid * tR - H_L_acid * tL) / dx
            # Implicit cp·T part in A (upwind T):
            if mR >= 0.0: A[rh_row, ch]   += mR * cp_i_acid / dx
            else:         A[rh_row, ch_R] += mR * cp_i_acid / dx
            if mL >= 0.0: A[rh_row, ch_L] -= mL * cp_i_acid / dx
            else:         A[rh_row, ch]   -= mL * cp_i_acid / dx
            # Subtract the deferred cp·T part (already in b via full flux):
            T_up_R = T_k[i]  if mR >= 0.0 else T_k[iR]
            T_up_L = T_k[iL] if mL >= 0.0 else T_k[i]
            b[rh_row] += (mR * cp_i_acid * T_up_R - mL * cp_i_acid * T_up_L) / dx

    return A.tocsr(), b


def assemble_newton_4N(
    N, dx, dt,
    rho_old, u_old, h_old, p_old, phi_old,  # old-time
    rho_k, u_k, h_k, p_k, T_k, phi_k,      # iterate
    zeta_k, phi_T_k,         # dρ/dp, dρ/dT
    alpha_k,                 # dρ/dφ (N,)
    d_rho_h_dphi_k,          # d(ρh)/dφ (N,)
    rho_face_acid, d_hat, theta_k,
    beta_k,                  # CICSAM blending factor (N+1,)
    ph1, ph2, bc_l, bc_r,
    mixing_type='volume',
    use_compress=False,
    C_k=None, n_hat_k=None, u_face_vof=None,
):
    """
    Fully coupled Newton-linearised (p, u, T, φ) 4N system.
    Block ordering: [p_0..p_{N-1}, u_0..u_{N-1}, T_0..T_{N-1}, phi_0..phi_{N-1}]
    Returns A (csr), b (ndarray).
    """
    size = 4 * N
    A = sp.lil_matrix((size, size), dtype=float)
    b = np.zeros(size)

    is_per_l = (bc_l == 'periodic')
    is_per_r = (bc_r == 'periodic')

    def face_lr(f):
        iL = f - 1
        iR = f
        iL = (N - 1 if is_per_l else 0) if iL < 0 else iL
        iR = (0 if is_per_r else N - 1) if iR >= N else iR
        return iL, iR

    # ACID EOS helpers (same interface as in assemble_newton_3N)
    # Build EOS objects once; works for any EOS (NASG, RKPR, etc.)
    eos1 = create_eos(ph1)
    eos2 = create_eos(ph2)

    def _acid_rho(p_val, T_val, psi_ref):
        r1 = eos1.rho(p_val, T_val)
        r2 = eos2.rho(p_val, T_val)
        return psi_ref * r1 + (1.0 - psi_ref) * r2

    def _acid_rh(p_val, T_val, u_val, psi_ref):
        r1 = eos1.rho(p_val, T_val)
        h1_val = eos1.h(p_val, T_val) + 0.5 * u_val * u_val
        r2 = eos2.rho(p_val, T_val)
        h2_val = eos2.h(p_val, T_val) + 0.5 * u_val * u_val
        return psi_ref * r1 * h1_val + (1.0 - psi_ref) * r2 * h2_val

    def _acid_cp(p_val, T_val, psi_ref):
        r1 = eos1.rho(p_val, T_val)
        r2 = eos2.rho(p_val, T_val)
        rho_mix = psi_ref * r1 + (1.0 - psi_ref) * r2 + 1e-300
        return (psi_ref * r1 * eos1.cp(p_val, T_val) +
                (1.0 - psi_ref) * r2 * eos2.cp(p_val, T_val)) / rho_mix

    def _acid_bm(p_val, T_val, psi_ref):
        r1 = eos1.rho(p_val, T_val)
        r2 = eos2.rho(p_val, T_val)
        rho_mix = psi_ref * r1 + (1.0 - psi_ref) * r2 + 1e-300
        return (psi_ref * r1 * eos1.dh_dp(p_val, T_val) +
                (1.0 - psi_ref) * r2 * eos2.dh_dp(p_val, T_val)) / rho_mix

    def _acid_rho_Y(p_val, T_val, Y_ref):
        r1 = eos1.rho(p_val, T_val)
        r2 = eos2.rho(p_val, T_val)
        inv_rho = Y_ref / (r1 + 1e-300) + (1.0 - Y_ref) / (r2 + 1e-300)
        return 1.0 / (inv_rho + 1e-300)

    def _acid_rh_Y(p_val, T_val, u_val, Y_ref):
        r1 = eos1.rho(p_val, T_val)
        h1_val = eos1.h(p_val, T_val)
        r2 = eos2.rho(p_val, T_val)
        h2_val = eos2.h(p_val, T_val)
        inv_rho = Y_ref / (r1 + 1e-300) + (1.0 - Y_ref) / (r2 + 1e-300)
        rho_star = 1.0 / (inv_rho + 1e-300)
        h_static = Y_ref * h1_val + (1.0 - Y_ref) * h2_val
        return rho_star * (h_static + 0.5 * u_val * u_val)

    def _acid_cp_Y(p_val, T_val, Y_ref):
        return Y_ref * eos1.cp(p_val, T_val) + (1.0 - Y_ref) * eos2.cp(p_val, T_val)

    def _acid_bm_Y(p_val, T_val, Y_ref):
        return Y_ref * eos1.dh_dp(p_val, T_val) + (1.0 - Y_ref) * eos2.dh_dp(p_val, T_val)

    for i in range(N):
        rp = _ci(0, i, N)
        ru = _ci(1, i, N)
        rT = _ci(2, i, N)
        rv = _ci(3, i, N)

        cp = _ci(0, i, N)
        cu = _ci(1, i, N)
        cT = _ci(2, i, N)
        cv = _ci(3, i, N)

        f_R = i + 1
        f_L = i
        iL, _ = face_lr(f_L)
        _, iR  = face_lr(f_R)

        cp_L = _ci(0, iL, N);  cu_L = _ci(1, iL, N);  cT_L = _ci(2, iL, N);  cv_L = _ci(3, iL, N)
        cp_R = _ci(0, iR, N);  cu_R = _ci(1, iR, N);  cT_R = _ci(2, iR, N);  cv_R = _ci(3, iR, N)

        rho_i   = rho_k[i]
        zeta_i  = zeta_k[i]
        phi_T_i = float(phi_T_k[i]) if phi_T_k is not None else 0.0
        alpha_i = float(alpha_k[i])
        drh_dphi_i = float(d_rho_h_dphi_k[i])
        u_i     = u_k[i]
        h_i     = h_k[i]
        T_i     = T_k[i]
        psi_i   = float(phi_k[i])  # volume or mass fraction for ACID

        tR = theta_k[f_R]
        tL = theta_k[f_L]
        dR  = d_hat[f_R]
        dL  = d_hat[f_L]

        # ACID face density
        if mixing_type == 'mass':
            rfR = _acid_rho_Y(float(p_k[iR]), float(T_k[iR]), psi_i)
            rfL = _acid_rho_Y(float(p_k[iL]), float(T_k[iL]), psi_i)
        else:
            rfR = _acid_rho(float(p_k[iR]), float(T_k[iR]), psi_i)
            rfL = _acid_rho(float(p_k[iL]), float(T_k[iL]), psi_i)

        mR = rfR * tR
        mL = rfL * tL

        # -----------------------------------------------------------
        # CONTINUITY (block 0) — extended with φ coupling
        # -----------------------------------------------------------
        A[rp, cp]   += zeta_i / dt
        b[rp]       += zeta_i * p_old[i] / dt
        # φ coupling: α·φ/dt
        A[rp, cv]   += alpha_i / dt
        b[rp]       += alpha_i * phi_old[i] / dt
        # MWI right face
        A[rp, cu]   += rfR / (2.0 * dx)
        A[rp, cu_R] += rfR / (2.0 * dx)
        A[rp, cp]   += rfR * dR / (dx * dx)
        A[rp, cp_R] -= rfR * dR / (dx * dx)
        # MWI left face
        A[rp, cu_L] -= rfL / (2.0 * dx)
        A[rp, cu]   -= rfL / (2.0 * dx)
        A[rp, cp]   += rfL * dL / (dx * dx)
        A[rp, cp_L] -= rfL * dL / (dx * dx)

        # -----------------------------------------------------------
        # MOMENTUM (block 1) — extended with φ coupling
        # -----------------------------------------------------------
        A[ru, cu] += rho_i / dt
        A[ru, cp] += zeta_i * u_i / dt
        A[ru, cv] += alpha_i * u_i / dt
        b[ru]     += rho_old[i] * u_old[i] / dt + zeta_i * u_i * p_k[i] / dt + alpha_i * u_i * phi_k[i] / dt

        # Convective right face
        if mR >= 0.0:
            A[ru, cu]   += mR / dx
        else:
            A[ru, cu_R] += mR / dx
        A[ru, cp]   += rfR * u_i * dR / (dx * dx)
        A[ru, cp_R] -= rfR * u_i * dR / (dx * dx)
        # Convective left face
        if mL >= 0.0:
            A[ru, cu_L] -= mL / dx
        else:
            A[ru, cu]   -= mL / dx
        A[ru, cp_L] -= rfL * u_i * dL / (dx * dx)
        A[ru, cp]   += rfL * u_i * dL / (dx * dx)
        # Pressure gradient
        A[ru, cp_R] += 1.0 / (2.0 * dx)
        A[ru, cp_L] -= 1.0 / (2.0 * dx)

        # -----------------------------------------------------------
        # ENERGY (block 2) — T-mode with φ coupling
        # From assemble_newton_3N T-mode (L257-316), extended with φ column
        # -----------------------------------------------------------
        if mixing_type == 'mass':
            cp_i    = _acid_cp_Y(float(p_k[i]), float(T_k[i]), psi_i)
            bm_i    = _acid_bm_Y(float(p_k[i]), float(T_k[i]), psi_i)
        else:
            cp_i    = _acid_cp(float(p_k[i]), float(T_k[i]), psi_i)
            bm_i    = _acid_bm(float(p_k[i]), float(T_k[i]), psi_i)

        drhdt = rho_i * cp_i + h_i * phi_T_i
        drhdp = rho_i * bm_i + h_i * zeta_i
        drhdu = rho_i * u_i

        A[rT, cT] += drhdt / dt
        A[rT, cu] += drhdu / dt
        A[rT, cp] += (drhdp - 1.0) / dt
        # φ coupling for energy: d(ρh)/dφ
        A[rT, cv] += drh_dphi_i / dt
        b[rT]     += (rho_old[i] * h_old[i] / dt
                      - p_old[i] / dt
                      + (drhdt * T_i + drhdp * p_k[i] + drhdu * u_i
                         - rho_i * h_i + drh_dphi_i * phi_k[i]) / dt)

        # Convective for T-mode: ACID face enthalpy
        if mixing_type == 'mass':
            H_R_acid    = _acid_rh_Y(float(p_k[iR]), float(T_k[iR]), float(u_k[iR]), psi_i)
            H_L_acid    = _acid_rh_Y(float(p_k[iL]), float(T_k[iL]), float(u_k[iL]), psi_i)
            cp_i_acid   = _acid_cp_Y(float(p_k[i]), float(T_k[i]), psi_i)
        else:
            H_R_acid    = _acid_rh(float(p_k[iR]), float(T_k[iR]), float(u_k[iR]), psi_i)
            H_L_acid    = _acid_rh(float(p_k[iL]), float(T_k[iL]), float(u_k[iL]), psi_i)
            cp_i_acid   = _acid_cp(float(p_k[i]), float(T_k[i]), psi_i)
        b[rT] -= (H_R_acid * tR - H_L_acid * tL) / dx
        if mR >= 0.0: A[rT, cT]   += mR * cp_i_acid / dx
        else:         A[rT, cT_R] += mR * cp_i_acid / dx
        if mL >= 0.0: A[rT, cT_L] -= mL * cp_i_acid / dx
        else:         A[rT, cT]   -= mL * cp_i_acid / dx
        T_up_R = T_k[i]  if mR >= 0.0 else T_k[iR]
        T_up_L = T_k[iL] if mL >= 0.0 else T_k[i]
        b[rT] += (mR * cp_i_acid * T_up_R - mL * cp_i_acid * T_up_L) / dx

        # -----------------------------------------------------------
        # VOF / SPECIES TRANSPORT (block 3)
        # Implicit CICSAM using frozen beta_k
        # -----------------------------------------------------------
        # Temporal
        A[rv, cv] += 1.0 / dt
        b[rv]     += phi_old[i] / dt

        # Advection with implicit CICSAM beta
        # Right face (f_R = i+1):
        tR_vof = tR   # same face velocity (or u_face_vof if provided separately)
        tL_vof = tL
        beta_R = float(beta_k[f_R])
        beta_L = float(beta_k[f_L])

        if tR_vof >= 0:
            # donor=i, acceptor=iR
            A[rv, cv]    += (1.0 - beta_R) * tR_vof / dx
            A[rv, cv_R]  += beta_R * tR_vof / dx
        else:
            # donor=iR, acceptor=i
            A[rv, cv_R]  += (1.0 - beta_R) * tR_vof / dx
            A[rv, cv]    += beta_R * tR_vof / dx

        # Left face (f_L = i):
        if tL_vof >= 0:
            # donor=iL, acceptor=i
            A[rv, cv_L]  -= (1.0 - beta_L) * tL_vof / dx
            A[rv, cv]    -= beta_L * tL_vof / dx
        else:
            # donor=i, acceptor=iL
            A[rv, cv]    -= (1.0 - beta_L) * tL_vof / dx
            A[rv, cv_L]  -= beta_L * tL_vof / dx

        # Source: -φ·∇·θ (divergence-free correction, implicit in φ)
        div_theta = (tR_vof - tL_vof) / dx
        A[rv, cv] -= div_theta

        # Compression (linearized φ(1-φ) around φ_k)
        if use_compress and C_k is not None and n_hat_k is not None and u_face_vof is not None:
            for face, sign_mult in [(f_R, -1.0), (f_L, 1.0)]:
                ck = float(C_k[face])
                nh = float(n_hat_k[face])
                u_f = float(u_face_vof[face])
                if abs(ck * nh) < 1e-15:
                    continue
                coeff = sign_mult * ck * abs(u_f) * nh / dx
                # Determine compression donor cell
                # n_hat * |u| >= 0: donor = left cell of face
                # face f_R: left cell = i, right cell = iR
                # face f_L: left cell = iL, right cell = i
                if face == f_R:
                    j_donor = i   if nh * abs(u_f) >= 0 else iR
                else:
                    j_donor = iL  if nh * abs(u_f) >= 0 else i
                phi_d   = float(phi_k[j_donor])
                cv_d    = _ci(3, j_donor, N)
                A[rv, cv_d] += coeff * (1.0 - 2.0 * phi_d)
                b[rv]       -= coeff * phi_d * phi_d

    return A.tocsr(), b


def assemble_newton_Ns(
    N, dx, dt, N_s,
    rho_old, u_old, h_old, p_old, phi_old_arr,  # phi_old_arr: (N_s-1, N)
    rho_k, u_k, h_k, p_k, T_k, phi_k_arr,      # phi_k_arr: (N_s-1, N)
    zeta_k, phi_T_k,       # dρ/dp, dρ/dT (N,)
    alpha_k_arr,            # list of N_s-1: ∂ρ/∂φₖ (N,)
    d_rho_h_dphi_k_arr,     # list of N_s-1: ∂(ρh)/∂φₖ (N,)
    rho_face_acid, d_hat, theta_k,
    beta_k_arr,             # list of N_s-1: CICSAM beta per species (N+1,)
    phases,                 # list of N_s EOS objects
    bc_l, bc_r,
    mixing_type='volume',
    use_compress=False,
    C_k_arr=None,           # list of N_s-1: Zalesak limiter (N+1,)
    n_hat_k_arr=None,       # list of N_s-1: interface normal (N+1,)
    u_face_vof=None,
):
    """
    Fully coupled Newton-linearised (p, u, T, φ₀, ..., φ_{N_s-2}) system.
    Block ordering: [p_0..p_{N-1}, u_0..u_{N-1}, T_0..T_{N-1},
                     phi0_0..phi0_{N-1}, ..., phi{N_s-2}_0..phi{N_s-2}_{N-1}]
    Matrix size: (2+N_s)*N x (2+N_s)*N
    Returns A (csr), b (ndarray).
    """
    size = (2 + N_s) * N
    A = sp.lil_matrix((size, size), dtype=float)
    b = np.zeros(size)

    is_per_l = (bc_l == 'periodic')
    is_per_r = (bc_r == 'periodic')

    def face_lr(f):
        iL = f - 1
        iR = f
        iL = (N - 1 if is_per_l else 0) if iL < 0 else iL
        iR = (0 if is_per_r else N - 1) if iR >= N else iR
        return iL, iR

    # Build EOS objects
    eos_list = [create_eos(ph) for ph in phases]

    # ACID helpers for volume fraction mixing
    def _acid_rho_Ns(p_val, T_val, phi_ref):
        """phi_ref: length N_s, cell i's fractions (sum=1)."""
        return sum(phi_ref[k] * eos_list[k].rho(p_val, T_val) for k in range(N_s))

    def _acid_rh_Ns(p_val, T_val, u_val, phi_ref):
        """ACID rhoH_total."""
        total = 0.0
        ke = 0.5 * u_val * u_val
        for k in range(N_s):
            r = eos_list[k].rho(p_val, T_val)
            h = eos_list[k].h(p_val, T_val)
            total += phi_ref[k] * r * (h + ke)
        return total

    def _acid_cp_Ns(p_val, T_val, phi_ref):
        """Density-weighted mixture cp."""
        num = sum(phi_ref[k] * eos_list[k].rho(p_val, T_val) * eos_list[k].cp(p_val, T_val)
                  for k in range(N_s))
        den = sum(phi_ref[k] * eos_list[k].rho(p_val, T_val) for k in range(N_s)) + 1e-300
        return num / den

    def _acid_bm_Ns(p_val, T_val, phi_ref):
        """Density-weighted mixture dh/dp."""
        num = sum(phi_ref[k] * eos_list[k].rho(p_val, T_val) * eos_list[k].dh_dp(p_val, T_val)
                  for k in range(N_s))
        den = sum(phi_ref[k] * eos_list[k].rho(p_val, T_val) for k in range(N_s)) + 1e-300
        return num / den

    # ACID helpers for mass fraction mixing
    def _acid_rho_Ns_mass(p_val, T_val, Y_ref):
        inv_rho = sum(Y_ref[k] / (eos_list[k].rho(p_val, T_val) + 1e-300) for k in range(N_s))
        return 1.0 / (inv_rho + 1e-300)

    def _acid_rh_Ns_mass(p_val, T_val, u_val, Y_ref):
        rho_star = _acid_rho_Ns_mass(p_val, T_val, Y_ref)
        h_static = sum(Y_ref[k] * eos_list[k].h(p_val, T_val) for k in range(N_s))
        return rho_star * (h_static + 0.5 * u_val * u_val)

    def _acid_cp_Ns_mass(p_val, T_val, Y_ref):
        return sum(Y_ref[k] * eos_list[k].cp(p_val, T_val) for k in range(N_s))

    def _acid_bm_Ns_mass(p_val, T_val, Y_ref):
        return sum(Y_ref[k] * eos_list[k].dh_dp(p_val, T_val) for k in range(N_s))

    # Select helpers based on mixing_type
    if mixing_type == 'mass':
        _acid_rho_f = _acid_rho_Ns_mass
        _acid_rh_f  = _acid_rh_Ns_mass
        _acid_cp_f  = _acid_cp_Ns_mass
        _acid_bm_f  = _acid_bm_Ns_mass
    else:
        _acid_rho_f = _acid_rho_Ns
        _acid_rh_f  = _acid_rh_Ns
        _acid_cp_f  = _acid_cp_Ns
        _acid_bm_f  = _acid_bm_Ns

    for i in range(N):
        rp     = _ci(0, i, N)
        ru     = _ci(1, i, N)
        rT_row = _ci(2, i, N)

        cp = _ci(0, i, N)
        cu = _ci(1, i, N)
        cT = _ci(2, i, N)

        f_R = i + 1
        f_L = i
        iL, _ = face_lr(f_L)
        _, iR  = face_lr(f_R)

        cp_L = _ci(0, iL, N);  cu_L = _ci(1, iL, N);  cT_L = _ci(2, iL, N)
        cp_R = _ci(0, iR, N);  cu_R = _ci(1, iR, N);  cT_R = _ci(2, iR, N)

        rho_i   = rho_k[i]
        zeta_i  = zeta_k[i]
        phi_T_i = float(phi_T_k[i]) if phi_T_k is not None else 0.0
        u_i     = u_k[i]
        h_i     = h_k[i]
        T_i     = T_k[i]

        tR = theta_k[f_R]
        tL = theta_k[f_L]
        dR  = d_hat[f_R]
        dL  = d_hat[f_L]

        # Build full phi vector for cell i
        phi_i_full = np.zeros(N_s)
        for k in range(N_s - 1):
            phi_i_full[k] = float(phi_k_arr[k][i])
        phi_i_full[N_s - 1] = 1.0 - sum(phi_i_full[:N_s - 1])
        phi_i_full = np.clip(phi_i_full, 0.0, 1.0)

        # ACID face densities
        phi_iR_full = np.zeros(N_s)
        for k in range(N_s - 1):
            phi_iR_full[k] = float(phi_k_arr[k][iR])
        phi_iR_full[N_s - 1] = 1.0 - sum(phi_iR_full[:N_s - 1])
        phi_iR_full = np.clip(phi_iR_full, 0.0, 1.0)

        phi_iL_full = np.zeros(N_s)
        for k in range(N_s - 1):
            phi_iL_full[k] = float(phi_k_arr[k][iL])
        phi_iL_full[N_s - 1] = 1.0 - sum(phi_iL_full[:N_s - 1])
        phi_iL_full = np.clip(phi_iL_full, 0.0, 1.0)

        rfR = _acid_rho_f(float(p_k[iR]), float(T_k[iR]), phi_i_full)
        rfL = _acid_rho_f(float(p_k[iL]), float(T_k[iL]), phi_i_full)

        mR = rfR * tR
        mL = rfL * tL

        # -----------------------------------------------------------
        # CONTINUITY (block 0)
        # -----------------------------------------------------------
        A[rp, cp] += zeta_i / dt
        b[rp]     += zeta_i * p_old[i] / dt
        # phi column coupling
        for k in range(N_s - 1):
            cv_k = _ci(3 + k, i, N)
            alpha_i_k = float(alpha_k_arr[k][i])
            A[rp, cv_k] += alpha_i_k / dt
            b[rp]       += alpha_i_k * float(phi_old_arr[k][i]) / dt
        # MWI right face
        A[rp, cu]   += rfR / (2.0 * dx)
        A[rp, cu_R] += rfR / (2.0 * dx)
        A[rp, cp]   += rfR * dR / (dx * dx)
        A[rp, cp_R] -= rfR * dR / (dx * dx)
        # MWI left face
        A[rp, cu_L] -= rfL / (2.0 * dx)
        A[rp, cu]   -= rfL / (2.0 * dx)
        A[rp, cp]   += rfL * dL / (dx * dx)
        A[rp, cp_L] -= rfL * dL / (dx * dx)

        # -----------------------------------------------------------
        # MOMENTUM (block 1)
        # -----------------------------------------------------------
        A[ru, cu] += rho_i / dt
        A[ru, cp] += zeta_i * u_i / dt
        b[ru]     += rho_old[i] * u_old[i] / dt + zeta_i * u_i * p_k[i] / dt
        # phi column coupling
        for k in range(N_s - 1):
            cv_k = _ci(3 + k, i, N)
            alpha_i_k = float(alpha_k_arr[k][i])
            A[ru, cv_k] += alpha_i_k * u_i / dt
            b[ru]       += alpha_i_k * u_i * float(phi_k_arr[k][i]) / dt
        # Convective right face
        if mR >= 0.0:
            A[ru, cu]   += mR / dx
        else:
            A[ru, cu_R] += mR / dx
        A[ru, cp]   += rfR * u_i * dR / (dx * dx)
        A[ru, cp_R] -= rfR * u_i * dR / (dx * dx)
        # Convective left face
        if mL >= 0.0:
            A[ru, cu_L] -= mL / dx
        else:
            A[ru, cu]   -= mL / dx
        A[ru, cp_L] -= rfL * u_i * dL / (dx * dx)
        A[ru, cp]   += rfL * u_i * dL / (dx * dx)
        # Pressure gradient
        A[ru, cp_R] += 1.0 / (2.0 * dx)
        A[ru, cp_L] -= 1.0 / (2.0 * dx)

        # -----------------------------------------------------------
        # ENERGY (block 2) — T-mode with phi coupling
        # -----------------------------------------------------------
        cp_i = _acid_cp_f(float(p_k[i]), float(T_k[i]), phi_i_full)
        bm_i = _acid_bm_f(float(p_k[i]), float(T_k[i]), phi_i_full)

        drhdt = rho_i * cp_i + h_i * phi_T_i
        drhdp = rho_i * bm_i + h_i * zeta_i
        drhdu = rho_i * u_i

        A[rT_row, cT] += drhdt / dt
        A[rT_row, cu] += drhdu / dt
        A[rT_row, cp] += (drhdp - 1.0) / dt
        # phi column coupling
        b[rT_row] += (rho_old[i] * h_old[i] / dt
                      - p_old[i] / dt
                      + (drhdt * T_i + drhdp * p_k[i] + drhdu * u_i
                         - rho_i * h_i) / dt)
        for k in range(N_s - 1):
            cv_k = _ci(3 + k, i, N)
            drh_dphi_i_k = float(d_rho_h_dphi_k_arr[k][i])
            A[rT_row, cv_k] += drh_dphi_i_k / dt
            b[rT_row]       += drh_dphi_i_k * float(phi_k_arr[k][i]) / dt

        # Convective: ACID face enthalpy
        H_R_acid  = _acid_rh_f(float(p_k[iR]), float(T_k[iR]), float(u_k[iR]), phi_i_full)
        H_L_acid  = _acid_rh_f(float(p_k[iL]), float(T_k[iL]), float(u_k[iL]), phi_i_full)
        b[rT_row] -= (H_R_acid * tR - H_L_acid * tL) / dx
        if mR >= 0.0: A[rT_row, cT]   += mR * cp_i / dx
        else:         A[rT_row, cT_R] += mR * cp_i / dx
        if mL >= 0.0: A[rT_row, cT_L] -= mL * cp_i / dx
        else:         A[rT_row, cT]   -= mL * cp_i / dx
        T_up_R = T_k[i]  if mR >= 0.0 else T_k[iR]
        T_up_L = T_k[iL] if mL >= 0.0 else T_k[i]
        b[rT_row] += (mR * cp_i * T_up_R - mL * cp_i * T_up_L) / dx

        # -----------------------------------------------------------
        # SPECIES TRANSPORT (blocks 3..2+N_s)
        # -----------------------------------------------------------
        for k in range(N_s - 1):
            rv  = _ci(3 + k, i, N)
            cv  = _ci(3 + k, i, N)
            cv_L = _ci(3 + k, iL, N)
            cv_R = _ci(3 + k, iR, N)

            beta_R = float(beta_k_arr[k][f_R])
            beta_L = float(beta_k_arr[k][f_L])

            # Temporal
            A[rv, cv] += 1.0 / dt
            b[rv]     += float(phi_old_arr[k][i]) / dt

            # Implicit CICSAM advection
            # Right face
            if tR >= 0:
                A[rv, cv]   += (1.0 - beta_R) * tR / dx
                A[rv, cv_R] += beta_R * tR / dx
            else:
                A[rv, cv_R] += (1.0 - beta_R) * tR / dx
                A[rv, cv]   += beta_R * tR / dx
            # Left face
            if tL >= 0:
                A[rv, cv_L] -= (1.0 - beta_L) * tL / dx
                A[rv, cv]   -= beta_L * tL / dx
            else:
                A[rv, cv]   -= (1.0 - beta_L) * tL / dx
                A[rv, cv_L] -= beta_L * tL / dx
            # Source: -phi * div(theta)
            div_theta = (tR - tL) / dx
            A[rv, cv] -= div_theta

            # Compression (linearized phi(1-phi) around phi_k)
            if (use_compress and C_k_arr is not None and n_hat_k_arr is not None
                    and u_face_vof is not None):
                C_k    = C_k_arr[k]
                n_hat  = n_hat_k_arr[k]
                for face, sign_mult in [(f_R, -1.0), (f_L, 1.0)]:
                    ck  = float(C_k[face])
                    nh  = float(n_hat[face])
                    u_f = float(u_face_vof[face])
                    if abs(ck * nh) < 1e-15:
                        continue
                    coeff = sign_mult * ck * abs(u_f) * nh / dx
                    if face == f_R:
                        j_donor = i   if nh * abs(u_f) >= 0 else iR
                    else:
                        j_donor = iL  if nh * abs(u_f) >= 0 else i
                    phi_d = float(phi_k_arr[k][j_donor])
                    cv_d  = _ci(3 + k, j_donor, N)
                    A[rv, cv_d] += coeff * (1.0 - 2.0 * phi_d)
                    b[rv]       -= coeff * phi_d * phi_d

    return A.tocsr(), b


def solve_linear_system(A, b, p_ref=1.0e5, u_ref=1.0, h_ref=3.0e5, phi_ref=None,
                        n_blocks=None):
    """Solve A @ x = b with column + row equilibration."""
    import scipy.sparse.linalg as spla

    size = len(b)

    col_scale = np.ones(size)
    if n_blocks is not None:
        # N_s-species general case: n_blocks = 2 + N_s
        NB = size // n_blocks
        col_scale[:NB]       = max(abs(p_ref), 1.0)
        col_scale[NB:2*NB]   = max(abs(u_ref), 1e-6)
        col_scale[2*NB:3*NB] = max(abs(h_ref), 1.0)
        for kb in range(3, n_blocks):
            col_scale[kb*NB:(kb+1)*NB] = max(abs(phi_ref) if phi_ref is not None else 1.0, 1e-6)
    elif phi_ref is not None:
        N4 = size // 4
        col_scale[:N4]        = max(abs(p_ref), 1.0)
        col_scale[N4:2*N4]    = max(abs(u_ref), 1e-6)
        col_scale[2*N4:3*N4]  = max(abs(h_ref), 1.0)
        col_scale[3*N4:]      = max(abs(phi_ref), 1e-10)
    else:
        N3 = size // 3
        col_scale[:N3]      = max(abs(p_ref), 1.0)
        col_scale[N3:2*N3]  = max(abs(u_ref), 1e-6)
        col_scale[2*N3:]    = max(abs(h_ref), 1.0)
    A_cs = A.dot(sp.diags(col_scale, format='csr'))

    abs_A = np.abs(A_cs)
    row_max_r = abs_A.max(axis=1)
    if sp.issparse(row_max_r):
        row_max = np.asarray(row_max_r.toarray()).ravel()
    else:
        row_max = np.asarray(row_max_r).ravel()
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
        try:
            x_hat = np.linalg.solve(As.toarray(), bs)
            if not np.all(np.isfinite(x_hat)):
                x_hat = None
        except Exception:
            x_hat = None

    if x_hat is None:
        x_hat = np.zeros_like(bs)

    if sp.issparse(x_hat):
        x_hat = np.asarray(x_hat.todense()).ravel()
    else:
        x_hat = np.asarray(x_hat).ravel()

    return col_scale * x_hat
