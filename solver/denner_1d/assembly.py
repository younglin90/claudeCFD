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
    g1 = float(ph1['gamma']); pi1 = float(ph1['pinf'])
    b1 = float(ph1['b']);     kv1 = float(ph1['kv']); eta1 = float(ph1['eta'])
    g2 = float(ph2['gamma']); pi2 = float(ph2['pinf'])
    b2 = float(ph2['b']);     kv2 = float(ph2['kv']); eta2 = float(ph2['eta'])

    def _acid_rho(p_val, T_val, psi_ref):
        """ACID density at (p_val,T_val) with this-cell ψ_ref (Eq. 37)."""
        A1 = kv1 * T_val * (g1 - 1.0) + b1 * (p_val + pi1) + 1e-300
        r1 = (p_val + pi1) / A1
        A2 = kv2 * T_val * (g2 - 1.0) + b2 * (p_val + pi2) + 1e-300
        r2 = (p_val + pi2) / A2
        return psi_ref * r1 + (1.0 - psi_ref) * r2

    def _acid_rh(p_val, T_val, psi_ref):
        """ACID ρh at (p_val,T_val) with this-cell ψ_ref."""
        A1 = kv1 * T_val * (g1 - 1.0) + b1 * (p_val + pi1) + 1e-300
        r1 = (p_val + pi1) / A1
        h1 = g1 * kv1 * T_val + b1 * p_val + eta1
        A2 = kv2 * T_val * (g2 - 1.0) + b2 * (p_val + pi2) + 1e-300
        r2 = (p_val + pi2) / A2
        h2 = g2 * kv2 * T_val + b2 * p_val + eta2
        return psi_ref * r1 * h1 + (1.0 - psi_ref) * r2 * h2

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
        psi_i  = float(psi_k[i])

        tR = theta_k[f_R]
        tL = theta_k[f_L]
        dR  = d_hat[f_R]
        dL  = d_hat[f_L]

        # ACID face density: computed with cell i's ψ applied to neighbor's (p,T)
        # At uniform (p,T): rfR = rfL = _acid_rho(p, T, psi_i) → div=0 ✓
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
        # ENTHALPY ENERGY
        # -----------------------------------------------------------
        if freeze_h:
            # Inner loop: h row = identity
            A[rh_row, ch] = 1.0
            b[rh_row]     = h_k[i]
        else:
            # Discrete: [ρ_k·h^{n+1} + ζ·(p^{n+1}-p_k)·h_k − ρ^n·h^n]/dt
            #           + div(ρ̃·ϑ·h^{n+1}) − (p^{n+1}−p^n)/dt = 0
            # Rearranged A*x = b:
            #   (ρ_k/dt + ζ·h_k/dt − 1/dt)·... wait, expand:
            #   ρ_k/dt·h_i + (ζ·h_k/dt − 1/dt)·p_i + conv = ρ^n·h^n/dt + ζ·h_k·p_k/dt + p^n/dt
            # Note: the −1/dt·p_i comes from −(p^{n+1}−p^n)/dt

            # Discrete energy: (ρ^{n+1}h^{n+1} - ρ^n·h^n)/dt + div - (p^{n+1}-p^n)/dt = 0
            # Newton: ρ_k/dt·h + (ζ·h_k - 1)/dt·p = ρ^n·h^n/dt - p^n/dt + ζ·h_k·p_k/dt
            A[rh_row, ch] += rho_i / dt
            A[rh_row, cp] += zeta_i * h_i / dt - 1.0 / dt
            b[rh_row]     += (rho_old[i] * h_old[i] / dt
                              - p_old[i] / dt
                              + zeta_i * h_i * p_k[i] / dt)

            # Convective: ρ̃·ϑ·h (upwind h, ACID deferred face enthalpy)
            psi_i    = float(psi_k[i])
            H_R_acid = _acid_rh(float(p_k[iR]), float(T_k[iR]), psi_i)
            H_L_acid = _acid_rh(float(p_k[iL]), float(T_k[iL]), psi_i)

            # The upwind h^{n+1}·mR part goes into A; the ACID face enthalpy
            # correction (H_acid − rho·h_upwind) is deferred to b:
            h_up_R = h_k[i]   if mR >= 0.0 else h_k[iR]
            h_up_L = h_k[iL]  if mL >= 0.0 else h_k[i]

            # Standard upwind (implicit h):
            if mR >= 0.0:
                A[rh_row, ch]  += mR / dx
            else:
                A[rh_row, ch_R] += mR / dx
            if mL >= 0.0:
                A[rh_row, ch_L] -= mL / dx
            else:
                A[rh_row, ch]   -= mL / dx

            # ACID deferred correction: (H_acid - rho*h_upwind_k) * theta / dx
            # At uniform state: H_acid = rho*h_upwind → correction = 0 ✓
            acid_corr_R = (H_R_acid - rfR * h_up_R) * tR / dx
            acid_corr_L = (H_L_acid - rfL * h_up_L) * tL / dx
            b[rh_row] -= (acid_corr_R - acid_corr_L)

    return A.tocsr(), b


def solve_linear_system(A, b, p_ref=1.0e5, u_ref=1.0, h_ref=3.0e5):
    """Solve A @ x = b with column + row equilibration."""
    import scipy.sparse.linalg as spla

    size = len(b)
    N3   = size // 3

    col_scale = np.ones(size)
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
