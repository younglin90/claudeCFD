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

        # ACID face density — Full Newton: upwind face primitives + face derivatives
        # Ref: Denner 2018 Eq. 25, 29, 30; Full Newton linearisation of ρ̃
        if mixing_type == 'mass':
            _ar = _acid_rho_Y
        else:
            _ar = _acid_rho

        # --- Upwind face primitive variables and column indices ---
        # Right face (f_R): upwind direction determined by theta sign
        if tR >= 0:
            p_fR = float(p_k[i]);  T_fR = float(T_k[i])
            cp_up_R = cp;   ch_up_R = ch    # upwind column = cell i
        else:
            p_fR = float(p_k[iR]); T_fR = float(T_k[iR])
            cp_up_R = cp_R; ch_up_R = ch_R  # upwind column = cell iR

        # Left face (f_L): upwind direction determined by theta sign
        if tL >= 0:
            p_fL = float(p_k[iL]); T_fL = float(T_k[iL])
            cp_up_L = cp_L; ch_up_L = ch_L  # upwind column = cell iL
        else:
            p_fL = float(p_k[i]);  T_fL = float(T_k[i])
            cp_up_L = cp;   ch_up_L = ch    # upwind column = cell i

        # ACID density at face using upwind (p,T) with cell i's ψ
        rfR = _ar(p_fR, T_fR, psi_i)
        rfL = _ar(p_fL, T_fL, psi_i)

        # Face density derivatives ∂ρ̃/∂p and ∂ρ̃/∂T for Full Newton of ρ̃·θ
        if mixing_type == 'volume':
            zeta_fR = psi_i * eos1.drho_dp(p_fR, T_fR) + (1.0 - psi_i) * eos2.drho_dp(p_fR, T_fR)
            phi_fR  = psi_i * eos1.drho_dT(p_fR, T_fR) + (1.0 - psi_i) * eos2.drho_dT(p_fR, T_fR)
            zeta_fL = psi_i * eos1.drho_dp(p_fL, T_fL) + (1.0 - psi_i) * eos2.drho_dp(p_fL, T_fL)
            phi_fL  = psi_i * eos1.drho_dT(p_fL, T_fL) + (1.0 - psi_i) * eos2.drho_dT(p_fL, T_fL)
        else:
            # Harmonic mixing: 1/ρ = ψ/ρ₁ + (1-ψ)/ρ₂
            # ∂ρ/∂p = ρ²·(ψ·ζ₁/ρ₁² + (1-ψ)·ζ₂/ρ₂²)
            # ∂ρ/∂T = ρ²·(ψ·φ₁/ρ₁² + (1-ψ)·φ₂/ρ₂²)
            r1R = eos1.rho(p_fR, T_fR); r2R = eos2.rho(p_fR, T_fR)
            z1R = eos1.drho_dp(p_fR, T_fR); z2R = eos2.drho_dp(p_fR, T_fR)
            g1R = eos1.drho_dT(p_fR, T_fR); g2R = eos2.drho_dT(p_fR, T_fR)
            zeta_fR = rfR**2 * (psi_i * z1R / (r1R**2 + 1e-300) + (1.0 - psi_i) * z2R / (r2R**2 + 1e-300))
            phi_fR  = rfR**2 * (psi_i * g1R / (r1R**2 + 1e-300) + (1.0 - psi_i) * g2R / (r2R**2 + 1e-300))
            r1L = eos1.rho(p_fL, T_fL); r2L = eos2.rho(p_fL, T_fL)
            z1L = eos1.drho_dp(p_fL, T_fL); z2L = eos2.drho_dp(p_fL, T_fL)
            g1L = eos1.drho_dT(p_fL, T_fL); g2L = eos2.drho_dT(p_fL, T_fL)
            zeta_fL = rfL**2 * (psi_i * z1L / (r1L**2 + 1e-300) + (1.0 - psi_i) * z2L / (r2L**2 + 1e-300))
            phi_fL  = rfL**2 * (psi_i * g1L / (r1L**2 + 1e-300) + (1.0 - psi_i) * g2L / (r2L**2 + 1e-300))

        # Deferred mass fluxes at x_k
        mR = rfR * tR
        mL = rfL * tL

        # Determine if T-coupling should be included (third_var=='T' and phi_k provided)
        use_T_coupling = (third_var == 'T' and phi_k is not None and not freeze_h)
        phi_i_val = float(phi_k[i]) if (phi_k is not None) else 0.0

        # -----------------------------------------------------------
        # CONTINUITY — Full Newton
        # Ref: Denner 2018 Eq. 25
        # (ρ^{n+1} - ρ^n)/dt + div(ρ̃^{n+1}·ϑ^{n+1}) = 0
        # Full Newton: ρ^{n+1} = ρ_k + ζ·δp + φ·δT
        #              ρ̃^{n+1}·ϑ^{n+1} = ρ̃_k·ϑ^{n+1} + ρ̃^{n+1}·ϑ_k - ρ̃_k·ϑ_k
        # -----------------------------------------------------------
        # Temporal: A·ζ/dt·p + (φ/dt·T if T-mode)
        # b:  ρ_old/dt + (ζ·p_k + φ·T_k - ρ_k)/dt  → residual = (ρ_old - ρ_k)/dt ✓
        A[rp, cp] += zeta_i / dt
        b[rp]     += rho_old[i] / dt + (zeta_i * float(p_k[i]) - rho_i) / dt
        if use_T_coupling:
            A[rp, ch] += phi_i_val / dt          # ∂ρ/∂T temporal (Full Newton)
            b[rp]     += phi_i_val * float(T_k[i]) / dt

        # Term 1: ρ̃_k · ϑ^{n+1} — MWI implicit (ū and −d̂·∇p)
        # Right face:
        A[rp, cu]   += rfR / (2.0 * dx)
        A[rp, cu_R] += rfR / (2.0 * dx)
        A[rp, cp]   += rfR * dR / (dx * dx)
        A[rp, cp_R] -= rfR * dR / (dx * dx)
        # Left face:
        A[rp, cu_L] -= rfL / (2.0 * dx)
        A[rp, cu]   -= rfL / (2.0 * dx)
        A[rp, cp]   += rfL * dL / (dx * dx)
        A[rp, cp_L] -= rfL * dL / (dx * dx)

        # Term 2: ρ̃^{n+1} · ϑ_k (Newton sensitivity of ρ̃ to p and T)
        # A adds Jacobian; b adds same evaluated at x_k → residual unchanged
        A[rp, cp_up_R] += zeta_fR * tR / dx
        A[rp, cp_up_L] -= zeta_fL * tL / dx
        b[rp] += zeta_fR * tR * p_fR / dx - zeta_fL * tL * p_fL / dx
        if use_T_coupling:
            A[rp, ch_up_R] += phi_fR * tR / dx
            A[rp, ch_up_L] -= phi_fL * tL / dx
            b[rp] += phi_fR * tR * T_fR / dx - phi_fL * tL * T_fL / dx

        # -----------------------------------------------------------
        # MOMENTUM — Full Newton
        # Ref: Denner 2018 Eq. 29, 30
        # (ρ·u)^{n+1} = ρ_k·u^{n+1} + ρ^{n+1}·u_k - ρ_k·u_k
        # Full advection: ρ̃·ϑ·ũ — linearize ρ̃ as well
        # -----------------------------------------------------------
        # Temporal: ρ_k/dt·u + ζ·u_k/dt·p + (φ·u_k/dt·T if T-mode)
        # b: ρ_old·u_old/dt + (ζ·u_k·p_k + φ·u_k·T_k - ρ_k·u_k)/dt
        A[ru, cu] += rho_i / dt
        A[ru, cp] += zeta_i * u_i / dt
        b[ru]     += rho_old[i] * u_old[i] / dt + zeta_i * u_i * float(p_k[i]) / dt
        if use_T_coupling:
            A[ru, ch] += phi_i_val * u_i / dt
            b[ru]     += phi_i_val * u_i * float(T_k[i]) / dt

        # Convective Term 1: ρ̃_k·ϑ_k·ũ^{n+1} (upwind u implicit)
        if mR >= 0.0:
            A[ru, cu]   += mR / dx
        else:
            A[ru, cu_R] += mR / dx
        if mL >= 0.0:
            A[ru, cu_L] -= mL / dx
        else:
            A[ru, cu]   -= mL / dx

        # Convective Term 2: ρ̃_k·ϑ^{n+1}·ũ_k — MWI implicit p-part (θ = ū - d̂·∇p)
        # u_k deferred, d̂·∇p part is implicit
        A[ru, cp]   += rfR * u_i * dR / (dx * dx)
        A[ru, cp_R] -= rfR * u_i * dR / (dx * dx)
        A[ru, cp_L] -= rfL * u_i * dL / (dx * dx)
        A[ru, cp]   += rfL * u_i * dL / (dx * dx)

        # Convective Term 3: ρ̃^{n+1}·ϑ_k·ũ_k (Newton sensitivity of ρ̃)
        u_up_R = float(u_k[i])  if tR >= 0 else float(u_k[iR])
        u_up_L = float(u_k[iL]) if tL >= 0 else float(u_k[i])
        A[ru, cp_up_R] += zeta_fR * tR * u_up_R / dx
        A[ru, cp_up_L] -= zeta_fL * tL * u_up_L / dx
        b[ru] += zeta_fR * tR * u_up_R * p_fR / dx - zeta_fL * tL * u_up_L * p_fL / dx
        if use_T_coupling:
            A[ru, ch_up_R] += phi_fR * tR * u_up_R / dx
            A[ru, ch_up_L] -= phi_fL * tL * u_up_L / dx
            b[ru] += phi_fR * tR * u_up_R * T_fR / dx - phi_fL * tL * u_up_L * T_fL / dx

        # Pressure gradient: −(p_R − p_L)/(2dx)
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
            # Convective + ACID: use upwind (p,T,u) consistent with rfR/rfL
            # For volume fraction mixing, H_acid(p,T,u,ψ) ≡ rfR·h_up when same (p,T)
            # → acid_corr = 0 in single-phase regions (eliminates spurious oscillation)
            if mixing_type == 'mass':
                H_R_acid = _acid_rh_Y(p_fR, T_fR, u_up_R, psi_i)
                H_L_acid = _acid_rh_Y(p_fL, T_fL, u_up_L, psi_i)
            else:
                H_R_acid = _acid_rh(p_fR, T_fR, u_up_R, psi_i)
                H_L_acid = _acid_rh(p_fL, T_fL, u_up_L, psi_i)
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
            # Regularization: when d(ρh)/dT ≈ 0 (ideal gas, stiffened gas with q=0),
            # the T-diagonal vanishes and the matrix becomes ill-conditioned.
            # Add a small ε·ρ·cp term to prevent zero diagonal.
            # This is equivalent to adding ε·ρ·cp·(T-T_k)/dt to the energy equation,
            # which vanishes at convergence (T=T_k) and doesn't affect the solution.
            if abs(drhdt) < 1e-3 * rho_i * (abs(cp_i) + 1e-300):
                drhdt = 1e-3 * rho_i * cp_i
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
                H_R_acid = _acid_rh_Y(p_fR, T_fR, u_up_R, psi_i)
                H_L_acid = _acid_rh_Y(p_fL, T_fL, u_up_L, psi_i)
                cp_i_acid = _acid_cp_Y(float(p_k[i]), float(T_k[i]), psi_i)
            else:
                H_R_acid = _acid_rh(p_fR, T_fR, u_up_R, psi_i)
                H_L_acid = _acid_rh(p_fL, T_fL, u_up_L, psi_i)
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
    # Newton-CICSAM Jacobian data (optional; if None, fallback to Picard)
    psi_face=None,
    jac_D=None, jac_A=None, jac_UU=None,
    idx_D=None, idx_A=None, idx_UU=None,
    third_var='T',   # 'T' = (p,u,T,φ) mode (default), 'h' = (p,u,h,φ) mode
    scale_continuity=False,  # optional: scale continuity rows to balance p vs Y columns
    picard_advection=False,  # Picard advection: skip spatial ACID Y-Jacobian (dRho_R/L terms)
    use_acid=True,   # True: ACID face density (cell-i ψ); False: upwind Y consistent flux
):
    """
    Fully coupled Newton-linearised (p, u, T/h, φ) 4N system.
    Block ordering: [p_0..p_{N-1}, u_0..u_{N-1}, T_0(or h_0)..T_{N-1}, phi_0..phi_{N-1}]
    third_var='T': energy unknown is temperature T (default, backward-compatible).
    third_var='h': energy unknown is specific total enthalpy h (Denner 2018 h-mode).
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

        # Face density: ACID (cell-i ψ) or upwind-Y consistent flux
        if use_acid:
            # ACID: cell i의 ψ로 face density 계산 (PE 보장, 기존 동작)
            Y_upR = psi_i
            Y_upL = psi_i
        else:
            # Non-ACID: upwind cell의 Y로 face density 계산 (Jacobian Y-sensitivity 확보)
            Y_upR = psi_i if (tR >= 0) else float(phi_k[iR])
            Y_upL = float(phi_k[iL]) if (tL >= 0) else psi_i

        if mixing_type == 'mass':
            rfR = _acid_rho_Y(float(p_k[iR]), float(T_k[iR]), Y_upR)
            rfL = _acid_rho_Y(float(p_k[iL]), float(T_k[iL]), Y_upL)
        else:
            rfR = _acid_rho(float(p_k[iR]), float(T_k[iR]), Y_upR)
            rfL = _acid_rho(float(p_k[iL]), float(T_k[iL]), Y_upL)

        mR = rfR * tR
        mL = rfL * tL

        # Spatial Jacobian: ∂ρ̃_f/∂ψ_i
        # Pre-compute EOS values at face neighbors (always needed)
        r1R = eos1.rho(float(p_k[iR]), float(T_k[iR]))
        r2R = eos2.rho(float(p_k[iR]), float(T_k[iR]))
        r1L = eos1.rho(float(p_k[iL]), float(T_k[iL]))
        r2L = eos2.rho(float(p_k[iL]), float(T_k[iL]))

        if use_acid:
            # ACID Jacobian: ∂ρ̃_f/∂ψ_i — always nonzero (cell i's ψ used for all faces)
            if mixing_type != 'mass':
                dRho_R = r1R - r2R
                dRho_L = r1L - r2L
            else:
                dRho_R = rfR * rfR * (1.0/(r2R+1e-300) - 1.0/(r1R+1e-300))
                dRho_L = rfL * rfL * (1.0/(r2L+1e-300) - 1.0/(r1L+1e-300))
        else:
            # Non-ACID Jacobian: ∂ρ_f/∂Y_i nonzero only when cell i is upwind
            if mixing_type != 'mass':
                # Right face: cell i is upwind when tR >= 0
                dRho_R = (r1R - r2R) if (tR >= 0) else 0.0
                # Left face: cell i is downwind when tL >= 0 (upwind = iL)
                dRho_L = 0.0 if (tL >= 0) else (r1L - r2L)
            else:
                # Harmonic mixing: ∂ρ_harm/∂Y = ρ²·(1/ρ₂ - 1/ρ₁)
                dRho_R = (rfR * rfR * (1.0/(r2R+1e-300) - 1.0/(r1R+1e-300))) if (tR >= 0) else 0.0
                dRho_L = 0.0 if (tL >= 0) else (rfL * rfL * (1.0/(r2L+1e-300) - 1.0/(r1L+1e-300)))

        # -----------------------------------------------------------
        # CONTINUITY (block 0) — extended with φ coupling
        # -----------------------------------------------------------
        A[rp, cp]   += zeta_i / dt
        b[rp]       += zeta_i * p_old[i] / dt
        # α·dψ/dt: temporal ψ-density coupling
        A[rp, cv]   += alpha_i / dt
        b[rp]       += alpha_i * float(phi_old[i]) / dt
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
        # Spatial ACID ψ Jacobian: ∂(ρ̃_f·θ_f)/∂ψ_i = Δρ_f · θ_f
        # picard_advection=True: skip (deferred — already in residual, temporal coupling retained)
        if not picard_advection:
            A[rp, cv] += dRho_R * tR / dx
            A[rp, cv] -= dRho_L * tL / dx

        # -----------------------------------------------------------
        # MOMENTUM (block 1) — extended with φ coupling
        # -----------------------------------------------------------
        A[ru, cu] += rho_i / dt
        A[ru, cp] += zeta_i * u_i / dt
        # α·u_k·dψ/dt: temporal ψ-density coupling for momentum
        A[ru, cv] += alpha_i * u_i / dt
        b[ru]     += (rho_old[i] * u_old[i] / dt + zeta_i * u_i * p_k[i] / dt
                      + alpha_i * u_i * float(phi_old[i]) / dt)

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
        # Spatial ACID ψ Jacobian: ∂(ρ̃_f·θ_f·ũ_f)/∂ψ_i
        u_up_R = float(u_k[i]) if mR >= 0.0 else float(u_k[iR])
        u_up_L = float(u_k[iL]) if mL >= 0.0 else float(u_k[i])
        # picard_advection=True: skip (deferred — temporal coupling retained)
        if not picard_advection:
            A[ru, cv] += dRho_R * tR * u_up_R / dx
            A[ru, cv] -= dRho_L * tL * u_up_L / dx

        # -----------------------------------------------------------
        # ENERGY (block 2) — h-mode or T-mode with φ coupling
        # -----------------------------------------------------------
        if third_var == 'h':
            # --- h-mode: Denner 2018 enthalpy equation ---
            # (ρ^{n+1}h^{n+1} - ρ^n·h^n)/dt + div(ρ̃ϑh) = (p^{n+1}-p^n)/dt
            # Newton: ρ_k/dt·h + (ζ·h_k - 1)/dt·p = ρ^n·h^n/dt - p^n/dt + ζ·h_k·p_k/dt
            A[rT, cT] += rho_i / dt
            A[rT, cp] += (zeta_i * h_i - 1.0) / dt
            # d(ρh)/dψ temporal coupling
            A[rT, cv] += drh_dphi_i / dt
            b[rT]     += (rho_old[i] * h_old[i] / dt
                          - p_old[i] / dt
                          + zeta_i * h_i * float(p_k[i]) / dt
                          + drh_dphi_i * float(phi_old[i]) / dt)
            # Convective + ACID/consistent (h-mode): upwind (p,T,u) face enthalpy
            h_up_R = h_k[i]  if mR >= 0.0 else h_k[iR]
            h_up_L = h_k[iL] if mL >= 0.0 else h_k[i]
            if mR >= 0.0: A[rT, cT]   += mR / dx
            else:         A[rT, cT_R] += mR / dx
            if mL >= 0.0: A[rT, cT_L] -= mL / dx
            else:         A[rT, cT]   -= mL / dx
            if use_acid:
                # ACID: face enthalpy evaluated with cell-i ψ (PE-preserving)
                if mixing_type == 'mass':
                    H_R_acid = _acid_rh_Y(float(p_k[iR]), float(T_k[iR]), u_up_R, psi_i)
                    H_L_acid = _acid_rh_Y(float(p_k[iL]), float(T_k[iL]), u_up_L, psi_i)
                else:
                    H_R_acid = _acid_rh(float(p_k[iR]), float(T_k[iR]), u_up_R, psi_i)
                    H_L_acid = _acid_rh(float(p_k[iL]), float(T_k[iL]), u_up_L, psi_i)
                acid_corr_R = (H_R_acid - rfR * h_up_R) * tR / dx
                acid_corr_L = (H_L_acid - rfL * h_up_L) * tL / dx
                b[rT] -= (acid_corr_R - acid_corr_L)
            else:
                # Non-ACID: consistent flux with upwind Y — H evaluated with same Y_up as rfR/rfL
                if mixing_type == 'mass':
                    H_R_consistent = _acid_rh_Y(float(p_k[iR]), float(T_k[iR]), u_up_R, Y_upR)
                    H_L_consistent = _acid_rh_Y(float(p_k[iL]), float(T_k[iL]), u_up_L, Y_upL)
                else:
                    H_R_consistent = _acid_rh(float(p_k[iR]), float(T_k[iR]), u_up_R, Y_upR)
                    H_L_consistent = _acid_rh(float(p_k[iL]), float(T_k[iL]), u_up_L, Y_upL)
                # correction: H_consistent - rfR·h_up_R accounts for species-mixture h difference
                acid_corr_R = (H_R_consistent - rfR * h_up_R) * tR / dx
                acid_corr_L = (H_L_consistent - rfL * h_up_L) * tL / dx
                b[rT] -= (acid_corr_R - acid_corr_L)
            # Spatial ψ Jacobian: ∂(H̃_f·θ_f)/∂ψ_i
            # picard_advection=True: skip (deferred — temporal coupling retained)
            if not picard_advection:
                if mixing_type != 'mass':
                    h1R_e = eos1.h(float(p_k[iR]), float(T_k[iR])) + 0.5*float(u_k[iR])**2
                    h2R_e = eos2.h(float(p_k[iR]), float(T_k[iR])) + 0.5*float(u_k[iR])**2
                    dH_R_full = r1R * h1R_e - r2R * h2R_e
                    h1L_e = eos1.h(float(p_k[iL]), float(T_k[iL])) + 0.5*float(u_k[iL])**2
                    h2L_e = eos2.h(float(p_k[iL]), float(T_k[iL])) + 0.5*float(u_k[iL])**2
                    dH_L_full = r1L * h1L_e - r2L * h2L_e
                    if use_acid:
                        dH_R = dH_R_full
                        dH_L = dH_L_full
                    else:
                        # Non-ACID: ∂H_f/∂Y_i nonzero only when cell i is upwind
                        dH_R = dH_R_full if (tR >= 0) else 0.0
                        dH_L = 0.0 if (tL >= 0) else dH_L_full
                else:
                    dH_R = 0.0
                    dH_L = 0.0
                A[rT, cv] += dH_R * tR / dx
                A[rT, cv] -= dH_L * tL / dx
        else:
            # --- T-mode (default): From assemble_newton_3N T-mode, extended with φ column ---
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
            # d(ρh)/dψ temporal coupling
            A[rT, cv] += drh_dphi_i / dt
            b[rT]     += (rho_old[i] * h_old[i] / dt
                          - p_old[i] / dt
                          + (drhdt * T_i + drhdp * p_k[i] + drhdu * u_i
                             + drh_dphi_i * psi_i
                             - rho_i * h_i) / dt)

            # Convective for T-mode: ACID or consistent face enthalpy
            if use_acid:
                Y_face_R = psi_i
                Y_face_L = psi_i
            else:
                Y_face_R = Y_upR
                Y_face_L = Y_upL
            if mixing_type == 'mass':
                H_R_acid    = _acid_rh_Y(float(p_k[iR]), float(T_k[iR]), float(u_k[iR]), Y_face_R)
                H_L_acid    = _acid_rh_Y(float(p_k[iL]), float(T_k[iL]), float(u_k[iL]), Y_face_L)
                cp_i_acid   = _acid_cp_Y(float(p_k[i]), float(T_k[i]), psi_i)
            else:
                H_R_acid    = _acid_rh(float(p_k[iR]), float(T_k[iR]), float(u_k[iR]), Y_face_R)
                H_L_acid    = _acid_rh(float(p_k[iL]), float(T_k[iL]), float(u_k[iL]), Y_face_L)
                cp_i_acid   = _acid_cp(float(p_k[i]), float(T_k[i]), psi_i)
            b[rT] -= (H_R_acid * tR - H_L_acid * tL) / dx
            if mR >= 0.0: A[rT, cT]   += mR * cp_i_acid / dx
            else:         A[rT, cT_R] += mR * cp_i_acid / dx
            if mL >= 0.0: A[rT, cT_L] -= mL * cp_i_acid / dx
            else:         A[rT, cT]   -= mL * cp_i_acid / dx
            T_up_R = T_k[i]  if mR >= 0.0 else T_k[iR]
            T_up_L = T_k[iL] if mL >= 0.0 else T_k[i]
            b[rT] += (mR * cp_i_acid * T_up_R - mL * cp_i_acid * T_up_L) / dx
            # Spatial ψ Jacobian: ∂(H̃_f·θ_f)/∂ψ_i
            # picard_advection=True: skip (deferred — temporal coupling retained)
            if not picard_advection:
                if mixing_type != 'mass':
                    h1R_e = eos1.h(float(p_k[iR]), float(T_k[iR])) + 0.5*float(u_k[iR])**2
                    h2R_e = eos2.h(float(p_k[iR]), float(T_k[iR])) + 0.5*float(u_k[iR])**2
                    dH_R_full = r1R * h1R_e - r2R * h2R_e
                    h1L_e = eos1.h(float(p_k[iL]), float(T_k[iL])) + 0.5*float(u_k[iL])**2
                    h2L_e = eos2.h(float(p_k[iL]), float(T_k[iL])) + 0.5*float(u_k[iL])**2
                    dH_L_full = r1L * h1L_e - r2L * h2L_e
                    if use_acid:
                        dH_R = dH_R_full
                        dH_L = dH_L_full
                    else:
                        # Non-ACID: ∂H_f/∂Y_i nonzero only when cell i is upwind
                        dH_R = dH_R_full if (tR >= 0) else 0.0
                        dH_L = 0.0 if (tL >= 0) else dH_L_full
                else:
                    dH_R = 0.0
                    dH_L = 0.0
                A[rT, cv] += dH_R * tR / dx
                A[rT, cv] -= dH_L * tL / dx

        # -----------------------------------------------------------
        # VOF / SPECIES TRANSPORT (block 3)
        # -----------------------------------------------------------
        # Temporal: ψ^{n+1}/dt
        A[rv, cv] += 1.0 / dt
        b[rv]     += phi_old[i] / dt

        if psi_face is not None and jac_D is not None:
            # --- Newton-CICSAM: exact Jacobian for ψ̃_f ---
            psi_fR = float(psi_face[f_R])
            psi_fL = float(psi_face[f_L])

            # (u,p) columns: (ψ̃_f^k - ψ_i^k)·θ_f^{n+1} implicit
            coeff_R = (psi_fR - psi_i) / dx
            coeff_L = -(psi_fL - psi_i) / dx
            A[rv, cu]   += coeff_R * 0.5
            A[rv, cu_R] += coeff_R * 0.5
            A[rv, cp]   += coeff_R * dR / dx
            A[rv, cp_R] -= coeff_R * dR / dx
            A[rv, cu_L] += coeff_L * 0.5
            A[rv, cu]   += coeff_L * 0.5
            A[rv, cp_L] += coeff_L * dL / dx
            A[rv, cp]   -= coeff_L * dL / dx

            # ψ columns: θ_f^k · Σ J_s · ψ_s^{n+1} (advection Newton Jacobian)
            A[rv, _ci(3, int(idx_D[f_R]), N)]  += float(jac_D[f_R]) * tR / dx
            A[rv, _ci(3, int(idx_A[f_R]), N)]  += float(jac_A[f_R]) * tR / dx
            A[rv, _ci(3, int(idx_UU[f_R]), N)] += float(jac_UU[f_R]) * tR / dx
            A[rv, _ci(3, int(idx_D[f_L]), N)]  -= float(jac_D[f_L]) * tL / dx
            A[rv, _ci(3, int(idx_A[f_L]), N)]  -= float(jac_A[f_L]) * tL / dx
            A[rv, _ci(3, int(idx_UU[f_L]), N)] -= float(jac_UU[f_L]) * tL / dx
            # Source: -ψ_i · div(θ)
            A[rv, cv] -= (tR - tL) / dx
        else:
            # --- Picard fallback: deferred CICSAM beta ---
            beta_R = float(beta_k[f_R])
            beta_L = float(beta_k[f_L])
            if tR >= 0:
                psi_face_R = (1.0 - beta_R) * psi_i + beta_R * float(phi_k[iR])
            else:
                psi_face_R = (1.0 - beta_R) * float(phi_k[iR]) + beta_R * psi_i
            if tL >= 0:
                psi_face_L = (1.0 - beta_L) * float(phi_k[iL]) + beta_L * psi_i
            else:
                psi_face_L = (1.0 - beta_L) * psi_i + beta_L * float(phi_k[iL])
            coeff_R = (psi_face_R - psi_i) / dx
            coeff_L = -(psi_face_L - psi_i) / dx
            A[rv, cu]   += coeff_R * 0.5
            A[rv, cu_R] += coeff_R * 0.5
            A[rv, cp]   += coeff_R * dR / dx
            A[rv, cp_R] -= coeff_R * dR / dx
            A[rv, cu_L] += coeff_L * 0.5
            A[rv, cu]   += coeff_L * 0.5
            A[rv, cp_L] += coeff_L * dL / dx
            A[rv, cp]   -= coeff_L * dL / dx

    # --- Optional: equation scaling for continuity rows ---
    # Scale continuity rows so that zeta (dρ/dp) and alpha (dρ/dφ) coefficients
    # are balanced, improving conditioning when alpha/zeta >> 1.
    if scale_continuity and alpha_k is not None and zeta_k is not None:
        A_lil = A.tolil()
        for ii in range(N):
            row_i = _ci(0, ii, N)
            alpha_i_abs = abs(float(alpha_k[ii])) + 1e-300
            zeta_i_abs  = abs(float(zeta_k[ii]))  + 1e-300
            scale_factor = zeta_i_abs / alpha_i_abs
            A_lil[row_i, :] = A_lil[row_i, :] * scale_factor
            b[row_i]        *= scale_factor
        A = A_lil.tocsr()
    else:
        A = A.tocsr()

    return A, b


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


def solve_schur_4N(A_4N, b_4N, N, p_ref=1.0e5, u_ref=1.0, h_ref=3.0e5):
    """Solve 4N system via Schur complement block elimination of ψ.

    Partitions:
        [A_ff(3N×3N)  A_fψ(3N×N)] [x_f ]   [b_f ]
        [A_ψf(N×3N)  A_ψψ(N×N) ] [x_ψ ] = [b_ψ ]

    1. Factor A_ψψ (N×N, well-conditioned)
    2. Schur complement: S = A_ff − A_fψ · A_ψψ⁻¹ · A_ψf  (3N×3N)
    3. Solve S · x_f = b_f − A_fψ · A_ψψ⁻¹ · b_ψ
    4. Back-substitute: x_ψ = A_ψψ⁻¹ · (b_ψ − A_ψf · x_f)
    """
    import scipy.sparse.linalg as spla

    n3 = 3 * N
    n4 = 4 * N

    # Extract blocks
    A_ff   = A_4N[:n3, :n3]     # 3N×3N
    A_fpsi = A_4N[:n3, n3:n4]   # 3N×N
    A_psif = A_4N[n3:n4, :n3]   # N×3N
    A_psipsi = A_4N[n3:n4, n3:n4]  # N×N
    b_f   = b_4N[:n3].copy()
    b_psi = b_4N[n3:n4].copy()

    # Step 1: Factor A_ψψ
    try:
        psi_lu = spla.splu(A_psipsi.tocsc())
    except Exception:
        # If factorization fails, fall back to full 4N direct solve
        return solve_linear_system(A_4N, b_4N, p_ref=p_ref, u_ref=u_ref,
                                   h_ref=h_ref, phi_ref=1.0, n_blocks=4)

    # Step 2: Compute A_ψψ⁻¹ · A_ψf  (N×3N) and A_ψψ⁻¹ · b_ψ (N,)
    # Solve column by column: A_ψψ · X = A_ψf
    A_psif_dense = A_psif.toarray()  # N×3N
    inv_psif = np.zeros((N, n3))
    for j in range(n3):
        col = A_psif_dense[:, j]
        if np.any(col != 0):
            inv_psif[:, j] = psi_lu.solve(col)
    inv_b_psi = psi_lu.solve(b_psi)

    # Step 3: Schur complement S = A_ff - A_fψ · inv_psif
    A_fpsi_dense = A_fpsi.toarray()  # 3N×N
    correction = A_fpsi_dense @ inv_psif  # 3N×3N
    S = A_ff.toarray() - correction
    S_sparse = sp.csr_matrix(S)

    b_s = b_f - A_fpsi_dense @ inv_b_psi

    # Step 4: Solve S · x_f = b_s (3N system — reuse existing solver)
    x_f = solve_linear_system(S_sparse, b_s, p_ref=p_ref, u_ref=u_ref,
                              h_ref=h_ref, n_blocks=3)

    # Step 5: Back-substitute ψ = A_ψψ⁻¹ · (b_ψ - A_ψf · x_f)
    x_psi = psi_lu.solve(b_psi - A_psif.dot(x_f))

    return np.concatenate([x_f, x_psi])


def solve_block_schur_4N(A, b, N, p_ref=1e5, u_ref=1.0, h_ref=3e5):
    """Block Schur preconditioned solver for 4N coupled system.

    Partitions 4N system into flow (3N: p,u,T/h) and species (N: Y/ψ) blocks.
    Uses lower-triangular block factorization as preconditioner for BiCGSTAB.
    Falls back to direct solve on failure.

    Block layout (same as assemble_newton_4N):
        [J_FF (3N×3N)  J_FY (3N×N)] [x_F]   [b_F]
        [J_YF (N×3N)   J_YY (N×N) ] [x_Y] = [b_Y]
    """
    import scipy.sparse.linalg as spla

    n3, n4 = 3 * N, 4 * N
    A_csc = A.tocsc()

    # Extract blocks
    J_FF = A_csc[:n3, :n3]     # 3N x 3N (flow)
    J_FY = A_csc[:n3, n3:n4]   # 3N x N  (flow <- species)
    J_YF = A_csc[n3:n4, :n3]   # N x 3N  (species <- flow)
    J_YY = A_csc[n3:n4, n3:n4] # N x N   (species)

    try:
        YY_lu = spla.splu(J_YY)
        FF_lu = spla.splu(J_FF)
    except Exception:
        # Singular block — fall back to full direct solve
        try:
            return spla.spsolve(A_csc, b)
        except Exception:
            return np.zeros_like(b)

    def precond_matvec(r):
        r_F = r[:n3]
        r_Y = r[n3:n4]
        # 1. Solve species block
        z_Y = YY_lu.solve(r_Y)
        # 2. Modify flow RHS with species contribution
        r_F_mod = r_F - J_FY.dot(z_Y)
        # 3. Solve flow block
        z_F = FF_lu.solve(r_F_mod)
        return np.concatenate([z_F, z_Y])

    M = spla.LinearOperator((n4, n4), matvec=precond_matvec)

    # Try BiCGSTAB with block preconditioner
    x, info = spla.bicgstab(A_csc, b, M=M, rtol=1e-8, maxiter=500)
    if info != 0 or not np.all(np.isfinite(x)):
        # BiCGSTAB failed — try GMRES
        x, info = spla.gmres(A_csc, b, M=M, rtol=1e-8, maxiter=500)
    if info != 0 or not np.all(np.isfinite(x)):
        # All iterative solvers failed — direct solve
        try:
            x = spla.spsolve(A_csc, b)
            if not np.all(np.isfinite(x)):
                x = np.zeros_like(b)
        except Exception:
            x = np.zeros_like(b)
    return x


def solve_linear_system(A, b, p_ref=1.0e5, u_ref=1.0, h_ref=3.0e5, phi_ref=None,
                        n_blocks=None):
    """Solve A @ x = b with GMRES + ILU preconditioner (most robust default).

    Column + row equilibration is applied for scaling.
    Primary solver: GMRES with ILU(0) preconditioner.
    Fallback 1: BiCGSTAB with Block-Jacobi preconditioner (Denner 2018 §6).
    Fallback 2: direct sparse solver (spsolve).
    Fallback 3: LSMR (robust for ill-conditioned / near-singular).
    """
    import scipy.sparse.linalg as spla

    size = len(b)

    # --- Column scaling ---
    col_scale = np.ones(size)
    if n_blocks is not None:
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

    # --- Row scaling ---
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

    # --- GMRES + ILU(0) preconditioner (most robust default) ---
    try:
        ilu = spla.spilu(As.tocsc(), drop_tol=1e-4, fill_factor=10)
        M = spla.LinearOperator(As.shape, ilu.solve)
        x_hat, info = spla.gmres(As, bs, M=M, rtol=1e-6, maxiter=200, restart=50)
        if info != 0 or not np.all(np.isfinite(x_hat)):
            x_hat = None
    except Exception:
        pass

    # --- Fallback 1: BiCGSTAB + Block-Jacobi preconditioner (Denner 2018 §6) ---
    # Block-Jacobi: invert each diagonal block (N×N) independently.
    # For 3N system (p,u,T): M^{-1} = diag(App^{-1}, Auu^{-1}, ATT^{-1})
    if x_hat is None:
        try:
            N_blk = n_blocks if n_blocks is not None else (4 if phi_ref is not None else 3)
            NB = size // N_blk
            block_solvers = []
            for k in range(N_blk):
                diag_block = As[k*NB:(k+1)*NB, k*NB:(k+1)*NB].tocsc()
                block_solvers.append(spla.splu(diag_block))

            def block_jacobi_solve(r):
                x_out = np.empty_like(r)
                for k in range(N_blk):
                    x_out[k*NB:(k+1)*NB] = block_solvers[k].solve(r[k*NB:(k+1)*NB])
                return x_out

            M_bj = spla.LinearOperator(As.shape, block_jacobi_solve)
            x_hat, info = spla.bicgstab(As, bs, M=M_bj, rtol=1e-6, maxiter=200)
            if info != 0 or not np.all(np.isfinite(x_hat)):
                x_hat = None
        except Exception:
            pass

    # --- Fallback 2: direct sparse solver ---
    if x_hat is None:
        try:
            x_hat = spla.spsolve(As.tocsc(), bs)
            if not np.all(np.isfinite(x_hat)):
                x_hat = None
        except Exception:
            pass

    # --- Fallback 3: LSMR (robust for ill-conditioned / near-singular) ---
    if x_hat is None:
        try:
            result = spla.lsmr(As, bs, atol=1e-8, btol=1e-8, maxiter=1000)
            x_hat = result[0]
            if not np.all(np.isfinite(x_hat)):
                x_hat = None
        except Exception:
            pass

    if x_hat is None:
        x_hat = np.zeros_like(bs)

    if sp.issparse(x_hat):
        x_hat = np.asarray(x_hat.todense()).ravel()
    else:
        x_hat = np.asarray(x_hat).ravel()

    return col_scale * x_hat


def residual_4N(x_4N, N, dx, dt,
                rho_old, u_old, h_old, p_old, phi_old,
                ph1, ph2, bc_l, bc_r,
                theta_old=None, u_bar_old=None, rho_star_old=None,
                mixing_type='volume', use_K=False, use_acid=True,
                acid_temporal=False):
    """Compute conservative residual R(x) for the fully-coupled 4N system.

    x_4N = [p_0..p_{N-1}, u_0..u_{N-1}, T_0..T_{N-1}, ψ_0..ψ_{N-1}]

    Returns R (4N vector):
      R[0:N]     = mass:     (ρ_new − ρ_old)/dt + div(ρ̃·θ)
      R[N:2N]    = momentum: (ρu_new − ρu_old)/dt + div(ρ̃·θ·u) + ∂p/∂x
      R[2N:3N]   = energy:   (ρE_new − ρE_old)/dt + div((ρE+p)·θ)
      R[3N:4N]   = VOF:      (ψ_new − ψ_old)/dt + div(ψ·θ) + (ψ+K)·∇·u
    """
    from .eos.base import compute_mixture_props, compute_mixture_props_Y
    from .flux.mwi import acid_face_density, harmonic_face_density, mwi_face_coeff_denner
    from .boundary import apply_ghost, apply_ghost_velocity
    from .interface.cicsam import cicsam_face_beta

    p_k = x_4N[0:N]
    u_k = x_4N[N:2*N]
    T_k = x_4N[2*N:3*N]
    phi_k = np.clip(x_4N[3*N:4*N], 0.0, 1.0)

    is_per_l = (bc_l == 'periodic')
    is_per_r = (bc_r == 'periodic')

    # Mixture properties at current state
    if mixing_type == 'mass':
        props = compute_mixture_props_Y(p_k, u_k, T_k, phi_k, ph1, ph2)
    else:
        props = compute_mixture_props(p_k, u_k, T_k, phi_k, ph1, ph2)
    rho_k = props['rho']
    E_k = props['E_total']  # ρE per cell

    # ACID face density at current (p_k, T_k, ψ_k)
    rho_face = acid_face_density(rho_k, props['c_mix'], phi_k, bc_l, bc_r)

    # MWI face velocity
    rho_star = harmonic_face_density(rho_k, bc_l, bc_r)
    e_diag = rho_k / dt  # momentum diagonal ≈ ρ/dt
    from .flux.mwi import mwi_face_coeff_denner as _mwi
    d_hat = _mwi(e_diag, rho_star, dx, dt, bc_l, bc_r)

    ng = 2
    u_ext = apply_ghost_velocity(u_k, bc_l, bc_r, ng)
    p_ext = apply_ghost(p_k, bc_l, bc_r, ng)

    theta = np.empty(N + 1)
    for f in range(N + 1):
        iL_g = ng + f - 1
        iR_g = ng + f
        ub = 0.5 * (u_ext[iL_g] + u_ext[iR_g])
        dp_dx = (p_ext[iR_g] - p_ext[iL_g]) / dx
        theta[f] = ub - d_hat[f] * dp_dx
    # Transient correction
    if theta_old is not None and u_bar_old is not None and rho_star_old is not None:
        theta += d_hat * (rho_star_old / dt) * (theta_old - u_bar_old)

    # CICSAM face VOF (upwind for now — simple and robust)
    phi_ext = apply_ghost(phi_k, bc_l, bc_r, ng)
    psi_face = np.empty(N + 1)
    for f in range(N + 1):
        iL_g = ng + f - 1
        iR_g = ng + f
        if theta[f] >= 0:
            psi_face[f] = phi_ext[iL_g]
        else:
            psi_face[f] = phi_ext[iR_g]

    # EOS objects for ACID face enthalpy
    eos1 = create_eos(ph1)
    eos2 = create_eos(ph2)

    # K factor (compressibility correction for VOF)
    if use_K:
        c1_arr = np.array([eos1.c(float(p_k[i]), float(T_k[i])) for i in range(N)])
        c2_arr = np.array([eos2.c(float(p_k[i]), float(T_k[i])) for i in range(N)])
        r1_arr = np.array([eos1.rho(float(p_k[i]), float(T_k[i])) for i in range(N)])
        r2_arr = np.array([eos2.rho(float(p_k[i]), float(T_k[i])) for i in range(N)])
        rc1 = r1_arr * c1_arr**2
        rc2 = r2_arr * c2_arr**2
        rc_mix = phi_k * rc1 + (1 - phi_k) * rc2 + 1e-300
        K_arr = phi_k * (rc1 / rc_mix - 1.0)
    else:
        K_arr = np.zeros(N)

    # Old-time references for temporal terms
    rho_old_use = rho_old
    h_old_use = h_old

    # Compute fluxes and residual
    R = np.zeros(4 * N)

    # Pressure ghost for gradient
    T_ext = apply_ghost(T_k, bc_l, bc_r, ng)

    def face_lr(f):
        iL = f - 1
        iR = f
        iL = (N - 1 if is_per_l else 0) if iL < 0 else iL
        iR = (0 if is_per_r else N - 1) if iR >= N else iR
        return iL, iR

    for i in range(N):
        f_R = i + 1
        f_L = i
        iL, _ = face_lr(f_L)
        _, iR = face_lr(f_R)

        tR = theta[f_R]
        tL = theta[f_L]

        # Face density and enthalpy
        psi_i = float(phi_k[i])
        psi_iR = float(phi_k[iR])
        psi_iL = float(phi_k[iL])

        if use_acid:
            # ACID: ρ̃(p_neighbor, T_neighbor, ψ_i) — cell i's ψ for both faces
            psi_fR = psi_i
            psi_fL = psi_i
        else:
            # Non-ACID: upwind ψ at face — face density responds to ψ changes
            if tR >= 0:
                psi_fR = psi_i
            else:
                psi_fR = psi_iR
            if tL >= 0:
                psi_fL = psi_iL
            else:
                psi_fL = psi_i

        if mixing_type == 'mass':
            rfR = 1.0 / (psi_fR / (eos1.rho(float(p_k[iR]), float(T_k[iR])) + 1e-300) +
                         (1 - psi_fR) / (eos2.rho(float(p_k[iR]), float(T_k[iR])) + 1e-300) + 1e-300)
            rfL = 1.0 / (psi_fL / (eos1.rho(float(p_k[iL]), float(T_k[iL])) + 1e-300) +
                         (1 - psi_fL) / (eos2.rho(float(p_k[iL]), float(T_k[iL])) + 1e-300) + 1e-300)
        else:
            rfR = psi_fR * eos1.rho(float(p_k[iR]), float(T_k[iR])) + (1 - psi_fR) * eos2.rho(float(p_k[iR]), float(T_k[iR]))
            rfL = psi_fL * eos1.rho(float(p_k[iL]), float(T_k[iL])) + (1 - psi_fL) * eos2.rho(float(p_k[iL]), float(T_k[iL]))

        mR = rfR * tR
        mL = rfL * tL

        # Face enthalpy (total): H̃ = ρ̃·h̃_total
        if mixing_type == 'mass':
            h1R = eos1.h(float(p_k[iR]), float(T_k[iR]))
            h2R = eos2.h(float(p_k[iR]), float(T_k[iR]))
            HR = rfR * (psi_fR * h1R + (1 - psi_fR) * h2R + 0.5 * float(u_k[iR])**2)
            h1L = eos1.h(float(p_k[iL]), float(T_k[iL]))
            h2L = eos2.h(float(p_k[iL]), float(T_k[iL]))
            HL = rfL * (psi_fL * h1L + (1 - psi_fL) * h2L + 0.5 * float(u_k[iL])**2)
        else:
            r1R_v = eos1.rho(float(p_k[iR]), float(T_k[iR]))
            r2R_v = eos2.rho(float(p_k[iR]), float(T_k[iR]))
            h1R = eos1.h(float(p_k[iR]), float(T_k[iR])) + 0.5 * float(u_k[iR])**2
            h2R = eos2.h(float(p_k[iR]), float(T_k[iR])) + 0.5 * float(u_k[iR])**2
            HR = psi_fR * r1R_v * h1R + (1 - psi_fR) * r2R_v * h2R
            r1L_v = eos1.rho(float(p_k[iL]), float(T_k[iL]))
            r2L_v = eos2.rho(float(p_k[iL]), float(T_k[iL]))
            h1L = eos1.h(float(p_k[iL]), float(T_k[iL])) + 0.5 * float(u_k[iL])**2
            h2L = eos2.h(float(p_k[iL]), float(T_k[iL])) + 0.5 * float(u_k[iL])**2
            HL = psi_fL * r1L_v * h1L + (1 - psi_fL) * r2L_v * h2L

        # Upwind velocity for momentum convection
        u_up_R = float(u_k[i]) if mR >= 0 else float(u_k[iR])
        u_up_L = float(u_k[iL]) if mL >= 0 else float(u_k[i])

        # --- Mass residual ---
        R[i] = (rho_k[i] - rho_old_use[i]) / dt + (mR - mL) / dx

        # --- Momentum residual ---
        R[N + i] = ((rho_k[i] * u_k[i] - rho_old_use[i] * u_old[i]) / dt
                    + (mR * u_up_R - mL * u_up_L) / dx
                    + (p_k[iR] - p_k[iL]) / (2 * dx))

        # --- Energy residual ---
        E_old_i = rho_old_use[i] * h_old_use[i] - p_old[i]
        flux_E_R = HR * tR
        flux_E_L = HL * tL
        R[2*N + i] = ((E_k[i] - E_old_i) / dt
                      + (flux_E_R - flux_E_L) / dx)

        # --- VOF residual ---
        psi_fR = float(psi_face[f_R])
        psi_fL = float(psi_face[f_L])
        div_theta = (tR - tL) / dx
        R[3*N + i] = ((phi_k[i] - phi_old[i]) / dt
                      + (psi_fR * tR - psi_fL * tL) / dx
                      + (psi_i + K_arr[i]) * div_theta
                      - psi_i * div_theta)
        # Simplified: ∂ψ/∂t + div(ψ_face·θ) + K·div(θ) = 0
        # = (ψ_new - ψ_old)/dt + (ψ_fR·θR - ψ_fL·θL)/dx + K·div(θ)
        # Actually: ∂ψ/∂t + ∇·(ψu) + (ψ+K)∇·u = 0
        #         = ∂ψ/∂t + div(ψ_face·θ) + K·div(θ) = 0
        R[3*N + i] = ((phi_k[i] - phi_old[i]) / dt
                      + (psi_fR * tR - psi_fL * tL) / dx
                      + K_arr[i] * div_theta)

    return R


def solve_jfnk_4N(residual_fn, x0, N,
                   max_newton=50, newton_tol=1e-6,
                   max_gmres=100, gmres_tol=1e-3,
                   omega=1.0, verbose=False,
                   precond_fn=None):  # NEW: optional ILU preconditioner factory
    """JFNK solver for 4N fully-coupled system.

    Parameters
    ----------
    residual_fn : callable  x (4N,) → R (4N,)
    x0 : ndarray (4N,) initial guess [p, u, T, ψ]
    N  : int  number of cells

    Returns
    -------
    x_converged, info_dict
    """
    import scipy.sparse.linalg as spla

    x_k = x0.copy()
    size = 4 * N
    info = {'converged': False, 'outer_iters': 0, 'residuals': []}

    # Reference scales for convergence check
    p_ref = max(np.mean(np.abs(x_k[:N])), 1.0)
    u_ref = max(np.mean(np.abs(x_k[N:2*N])), 1.0)
    T_ref = max(np.mean(np.abs(x_k[2*N:3*N])), 1.0)

    for k in range(max_newton):
        R_k = residual_fn(x_k)
        r_norm = np.linalg.norm(R_k)
        info['residuals'].append(r_norm)

        # Convergence check: scaled residual
        scale = np.ones(size)
        scale[:N] = max(p_ref * 1e-7, 1e-10)  # mass: ρ scale / dt
        scale[N:2*N] = max(p_ref * u_ref * 1e-7, 1e-10)  # momentum
        scale[2*N:3*N] = max(p_ref * 1e-3, 1e-10)  # energy
        scale[3*N:] = 1.0  # VOF
        res_scaled = np.max(np.abs(R_k) / (np.abs(scale) + 1e-300))

        if verbose and (k < 5 or k % 10 == 0):
            print(f"    JFNK {k:3d}: |R|={r_norm:.3e}  scaled={res_scaled:.3e}")

        if r_norm < newton_tol * max(r_norm if k == 0 else info['residuals'][0], 1e-10):
            info['converged'] = True
            info['outer_iters'] = k + 1
            break
        if k == 0:
            r0_norm = r_norm

        # --- GMRES for J·δx = -R_k ---
        # J·v ≈ [R(x_k + ε·v) - R_k] / ε
        x_norm = np.linalg.norm(x_k)
        eps_base = np.sqrt(np.finfo(float).eps) * max(x_norm, 1.0)

        def jvp(v):
            eps = eps_base / (np.linalg.norm(v) + 1e-300)
            return (residual_fn(x_k + eps * v) - R_k) / eps

        J_op = spla.LinearOperator((size, size), matvec=jvp)

        # Build ILU preconditioner from approximate Jacobian (if provided)
        M_op = None
        if precond_fn is not None:
            try:
                J_approx = precond_fn(x_k)
                ilu = spla.spilu(J_approx.tocsc(), drop_tol=1e-4)
                M_op = spla.LinearOperator((size, size), matvec=ilu.solve)
            except Exception:
                M_op = None

        # Solve J·δx = -R_k with GMRES
        dx, gmres_info = spla.gmres(J_op, -R_k,
                                     maxiter=max_gmres,
                                     rtol=gmres_tol,
                                     M=M_op)
        if not np.all(np.isfinite(dx)):
            dx = np.zeros(size)

        # Line search with damping
        dp = dx[0:N]; du = dx[N:2*N]
        dT = dx[2*N:3*N]; dphi = dx[3*N:4*N]

        omega_k = omega * min(1.0,
            0.5 * p_ref / (np.max(np.abs(dp)) + 1e-300),
            500.0 / (np.max(np.abs(du)) + 1e-300),
            0.5 * T_ref / (np.max(np.abs(dT)) + 1e-300))

        x_k[:N] = np.maximum(x_k[:N] + omega_k * dp, 1.0)  # p floor
        x_k[N:2*N] += omega_k * du
        x_k[2*N:3*N] = np.maximum(x_k[2*N:3*N] + omega_k * dT, 1e-3)  # T floor
        x_k[3*N:] = np.clip(x_k[3*N:] + omega_k * dphi, 0.0, 1.0)

        info['outer_iters'] = k + 1

    return x_k, info
