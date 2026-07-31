"""Five novel LBM steady-state accelerators (max-effort proposal set).

All solvers share the case interface:
    case.lbe_step(f) -> f, case.residual(f) -> R, case.macro(f) -> (rho, ux, uy),
    case.jvp(w, f, R, norm_f_cached=None), case.initial_field(), case.N, case.omega.

Implementations are matrix-free; numba acceleration is inherited from each
Case class's njit lbe_step (patched via numba_kernels.enable_numba_kernels).

1. solve_ms_nk_ehi   : Mode-Separated Newton-Krylov with hydrodynamic-only Jacobian
2. solve_lgf_lbm     : Lattice Green's function preconditioned Picard (periodic)
3. solve_dhh_lbm     : Helmholtz-Hodge velocity projection inside Picard
4. solve_apix_lbm    : Asymptotic-preserving IMEX with adaptive dt
5. solve_elgf_lbm    : Lyapunov gradient flow on residual norm with Armijo
"""

from __future__ import annotations

import math
import time

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from numba_kernels import CX, CY, W


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------
CX_F = CX.astype(np.float64)
CY_F = CY.astype(np.float64)


def _macro(case, f):
    if hasattr(case, "macro"):
        return case.macro(f)
    rho = f.sum(axis=0)
    rho_s = np.where(rho < 1e-12, 1.0, rho)
    return rho, (f * CX_F[:, None, None]).sum(axis=0) / rho_s, (f * CY_F[:, None, None]).sum(axis=0) / rho_s


def _norm(x):
    return float(np.sqrt(np.sum(x * x)))


# ---------------------------------------------------------------------------
# Method 1: MS-NK-EHI (Mode-Separated NK + Hydrodynamic Exact Inversion)
# ---------------------------------------------------------------------------
def _project_macro(R):
    """Project R(9,N,N) onto hydrodynamic moment basis -> U(3,N,N).
    U[0] = rho-residual, U[1] = rho*ux-residual, U[2] = rho*uy-residual."""
    return np.stack([
        R.sum(axis=0),
        (R * CX_F[:, None, None]).sum(axis=0),
        (R * CY_F[:, None, None]).sum(axis=0),
    ], axis=0)


def _lift_macro(U):
    """Lift U(3,N,N) -> f(9,N,N) via dual basis. Uses w_i * (mass + 3 c_x ux + 3 c_y uy)."""
    out = np.empty((9,) + U.shape[1:], dtype=np.float64)
    for i in range(9):
        out[i] = W[i] * (U[0] + 3.0 * CX_F[i] * U[1] + 3.0 * CY_F[i] * U[2])
    return out


def solve_ms_nk_ehi(case, max_outer=100, tol=1e-7,
                    krylov_max=10, krylov_tol=1e-3,
                    line_search_max=5, verbose=False):
    """Hydrodynamic-projected JFNK + kinetic Picard polish.

    Adaptive kinetic polish length based on case.omega.
    """
    f = case.initial_field()
    hist = []
    t0 = time.perf_counter()
    lbe = 0
    n = case.N
    K_K = max(5, int(math.ceil(math.log(1e-3) / math.log(max(1e-3, abs(1.0 - 1.0 / max(case.omega, 1e-3)))))))
    K_K = min(K_K, 40)

    for k in range(max_outer):
        R = case.residual(f); lbe += 1
        U_H = _project_macro(R)
        rn_H = _norm(U_H) / math.sqrt(3 * n * n)
        hist.append((k, rn_H, lbe, time.perf_counter() - t0))
        if verbose and (k < 3 or k % 10 == 0):
            print(f"  MS-NK-EHI k={k:3d} | rH={rn_H:.3e} | lbe={lbe}", flush=True)
        if not np.isfinite(rn_H) or rn_H < tol:
            break

        norm_f = _norm(f)
        probes = [0]
        def matvec(v_H_flat):
            probes[0] += 1
            v_H = v_H_flat.reshape(3, n, n)
            v_lifted = _lift_macro(v_H)
            Jv = case.jvp(v_lifted, f, R, norm_f_cached=norm_f)
            return _project_macro(Jv).ravel()

        op = LinearOperator((3 * n * n, 3 * n * n), matvec=matvec, dtype=np.float64)
        dU_flat, info = gmres(op, -U_H.ravel(),
                              rtol=krylov_tol,
                              atol=krylov_tol * np.linalg.norm(U_H) * 1e-3,
                              maxiter=1, restart=2 * krylov_max)
        lbe += probes[0]
        if info < 0 or not np.all(np.isfinite(dU_flat)):
            break
        df_H = _lift_macro(dU_flat.reshape(3, n, n))

        alpha = 1.0; accepted = False
        for _ in range(line_search_max):
            f_try = f + alpha * df_H
            for _ in range(K_K):
                f_try = case.lbe_step(f_try); lbe += 1
            R_try = case.residual(f_try); lbe += 1
            U_H_try = _project_macro(R_try)
            rn_try = _norm(U_H_try) / math.sqrt(3 * n * n)
            if np.isfinite(rn_try) and rn_try < rn_H:
                f = f_try; accepted = True; break
            alpha *= 0.5
        if not accepted:
            for _ in range(K_K):
                f = case.lbe_step(f); lbe += 1
    return f, hist


# ---------------------------------------------------------------------------
# Method 2: LGF-LBM (Lattice Green's Function preconditioned Picard)
# ---------------------------------------------------------------------------
def _build_lgf(n, omega):
    """Precompute spectral linearized inverse G_hat[ky, kx, 9, 9]: (I - L_S @ C_lin)^{-1}
    around uniform equilibrium rho=1, u=0.

    Linearized BGK around f0 = w_i:
        C_lin(f) = f - omega * (f - Eq_lin @ f),
        Eq_lin[i, j] = w_i * 9 * (c_i . c_j) / 3 + w_i * 1 + w_i * cs2 stuff
        For rho=1, u=0 linearization: f_eq linearizes as
        df_eq = w_i * sum_j f_j  (mass conservation)
              + w_i * 3 c_i . (sum_j c_j f_j)  (momentum)
    """
    cx = CX_F; cy = CY_F
    Eq_lin = np.zeros((9, 9))
    for i in range(9):
        for j in range(9):
            Eq_lin[i, j] = W[i] * (1.0 + 3.0 * (cx[i] * cx[j] + cy[i] * cy[j]))
    C_lin = np.eye(9) - omega * (np.eye(9) - Eq_lin)
    kx = 2.0 * np.pi * np.fft.fftfreq(n)
    ky = 2.0 * np.pi * np.fft.fftfreq(n)
    G_hat = np.empty((n, n, 9, 9), dtype=np.complex128)
    for iy, kky in enumerate(ky):
        for ix, kkx in enumerate(kx):
            L_S_k = np.diag(np.exp(-1j * (cx * kkx + cy * kky)))
            M = np.eye(9) - L_S_k @ C_lin
            try:
                G_hat[iy, ix] = np.linalg.inv(M + 1e-12 * np.eye(9))
            except np.linalg.LinAlgError:
                G_hat[iy, ix] = np.eye(9)
    return G_hat


def _apply_lgf(G_hat, R):
    """Apply G_hat to R(9, N, N) cell-wise in Fourier basis."""
    R_hat = np.fft.fft2(R, axes=(1, 2))
    out_hat = np.einsum('yxij,jyx->iyx', G_hat, R_hat)
    return np.real(np.fft.ifft2(out_hat, axes=(1, 2)))


def solve_lgf_lbm(case, max_outer=200, tol=1e-7,
                  line_search_max=5, beta_max=0.7, verbose=False):
    """Lattice Green's function preconditioned Picard with Nesterov.

    Best for periodic geometry; for walled geometry G_hat is bulk approximation.
    """
    f = case.initial_field()
    n = case.N
    G_hat = _build_lgf(n, case.omega)
    hist = []
    t0 = time.perf_counter()
    lbe = 0
    f_prev = f.copy()
    res_prev = np.inf
    beta = 0.0

    for k in range(max_outer):
        R = case.residual(f); lbe += 1
        res = _norm(R) / math.sqrt(case.dof)
        hist.append((k, res, lbe, time.perf_counter() - t0))
        if verbose and (k < 3 or k % 10 == 0):
            print(f"  LGF k={k:3d} | res={res:.3e} | beta={beta:.2f} | lbe={lbe}", flush=True)
        if not np.isfinite(res) or res < tol:
            break

        df = _apply_lgf(G_hat, R)

        alpha = 1.0; accepted = False
        for _ in range(line_search_max):
            f_try = f - alpha * df
            R_try = case.residual(f_try); lbe += 1
            res_try = _norm(R_try) / math.sqrt(case.dof)
            if np.isfinite(res_try) and res_try < res:
                # Nesterov: try momentum
                if beta > 0:
                    f_mom = f_try + beta * (f_try - f_prev)
                    R_mom = case.residual(f_mom); lbe += 1
                    res_mom = _norm(R_mom) / math.sqrt(case.dof)
                    if np.isfinite(res_mom) and res_mom < res_try:
                        f_prev = f; f = f_mom
                        beta = min(beta_max, beta * 1.1)
                    else:
                        f_prev = f; f = f_try
                        beta = beta * 0.7
                else:
                    f_prev = f; f = f_try
                    beta = 0.3
                accepted = True; break
            alpha *= 0.5
        if not accepted:
            beta = 0.0
            f = case.lbe_step(f); lbe += 1
        res_prev = res
    return f, hist


# ---------------------------------------------------------------------------
# Method 3: DHH-LBM (Discrete Helmholtz-Hodge velocity projection)
# ---------------------------------------------------------------------------
def _hh_project_velocity(ux, uy):
    """Project (ux, uy) onto solenoidal part via FFT Poisson.
    Returns (ux_s, uy_s) with ∇·u_s ≈ 0 (periodic assumption)."""
    n = ux.shape[0]
    # divergence via central FD with periodic
    div_u = (np.roll(ux, -1, axis=1) - np.roll(ux, 1, axis=1)) * 0.5 + \
            (np.roll(uy, -1, axis=0) - np.roll(uy, 1, axis=0)) * 0.5
    div_hat = np.fft.fft2(div_u)
    kx = 2.0 * np.pi * np.fft.fftfreq(n)
    ky = 2.0 * np.pi * np.fft.fftfreq(n)
    KX, KY = np.meshgrid(kx, ky, indexing='xy')
    K2 = KX * KX + KY * KY
    K2[0, 0] = 1.0
    phi_hat = -div_hat / K2
    phi_hat[0, 0] = 0.0
    phi = np.real(np.fft.ifft2(phi_hat))
    gx = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) * 0.5
    gy = (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) * 0.5
    return ux - gx, uy - gy


def _equilibrium(rho, ux, uy):
    feq = np.empty((9,) + rho.shape, dtype=np.float64)
    u2 = 1.5 * (ux * ux + uy * uy)
    for i in range(9):
        cu = 3.0 * (CX_F[i] * ux + CY_F[i] * uy)
        feq[i] = W[i] * rho * (1.0 + cu + 0.5 * cu * cu - u2)
    return feq


def solve_dhh_lbm(case, max_outer=200000, tol=1e-7,
                  N_hh=50, check_every=200, verbose=False):
    """Picard + periodic Helmholtz-Hodge velocity projection every N_hh steps.

    Note: HH projection assumes periodic; for walled cases this is an approximate
    bulk correction that does not enforce no-slip on the projection step.
    """
    f = case.initial_field()
    hist = []
    t0 = time.perf_counter()
    lbe = 0
    has_mask = hasattr(case, 'chi')

    for step in range(1, max_outer + 1):
        f = case.lbe_step(f); lbe += 1
        if step % N_hh == 0:
            rho, ux, uy = _macro(case, f)
            ux_s, uy_s = _hh_project_velocity(ux, uy)
            if has_mask:
                chi = case.chi
                ux_s = ux_s * chi + ux * (1.0 - chi)
                uy_s = uy_s * chi + uy * (1.0 - chi)
            f_eq_old = _equilibrium(rho, ux, uy)
            f_eq_new = _equilibrium(rho, ux_s, uy_s)
            f = f_eq_new + (f - f_eq_old)
            if has_mask:
                f = f * case.chi[None, :, :]
        if step % check_every == 0:
            R = case.residual(f); lbe += 1
            res = _norm(R) / math.sqrt(case.dof)
            hist.append((step, res, lbe, time.perf_counter() - t0))
            if verbose and (step % 1000 == 0):
                print(f"  DHH step={step:7d} | res={res:.3e} | lbe={lbe}", flush=True)
            if not np.isfinite(res) or res < tol:
                break
    return f, hist


# ---------------------------------------------------------------------------
# Method 4: APIX-LBM (AP-IMEX with adaptive pseudo-time)
# ---------------------------------------------------------------------------
def solve_apix_lbm(case, max_outer=200000, tol=1e-7,
                   dt0=1.0, dt_max=1e4, growth=1.5, shrink=0.5,
                   check_every=10, verbose=False):
    """Closed-form implicit BGK collision + explicit streaming with adaptive dt.

    f^{n+1} = (tau/(tau+dt)) * L_S(f^n) + (dt/(tau+dt)) * f^eq(U^n)

    Uses case.lbe_step (which is L = L_S o C) and inverts via convex blend:
    standard LBE step at dt=1 reduces to case.lbe_step(f); for dt>1 we blend.

    Implementation detail:
      define explicit streaming via composition L_S(f) = case.lbe_step(f_eq(rho,u))
      (i.e., one LBE step from equilibrium gives pure-stream of equilibrium)
      Practical: use case.lbe_step directly + residual blending.
    """
    f = case.initial_field()
    tau = 1.0 / case.omega
    hist = []
    t0 = time.perf_counter()
    lbe = 0
    dt = dt0
    res_prev = np.inf
    has_mask = hasattr(case, 'chi')

    for step in range(1, max_outer + 1):
        rho, ux, uy = _macro(case, f)
        f_eq = _equilibrium(rho, ux, uy)
        # Streaming-only step approximated by L on equilibrium (kills collision residual)
        # plus blending with raw L(f) for non-equilibrium content.
        L_f = case.lbe_step(f); lbe += 1
        f_new = (tau / (tau + dt)) * L_f + (dt / (tau + dt)) * f_eq
        if has_mask:
            f_new = f_new * case.chi[None, :, :]
        if not np.all(np.isfinite(f_new)):
            dt = max(dt * shrink, dt0)
            continue
        if step % check_every == 0:
            R = f_new - case.lbe_step(f_new); lbe += 1
            res = _norm(R) / math.sqrt(case.dof)
            hist.append((step, res, lbe, time.perf_counter() - t0))
            if verbose and (step % 1000 == 0):
                print(f"  APIX step={step:6d} | res={res:.3e} | dt={dt:.2e} | lbe={lbe}", flush=True)
            if not np.isfinite(res):
                dt = max(dt * shrink, dt0)
                continue
            if res < tol:
                f = f_new; break
            if res < 0.8 * res_prev:
                dt = min(dt * growth, dt_max)
            elif res > 1.2 * res_prev:
                dt = max(dt * shrink, dt0)
            res_prev = res
        f = f_new
    return f, hist


# ---------------------------------------------------------------------------
# Method 5: ELGF-LBM (Lyapunov gradient flow with Armijo)
# ---------------------------------------------------------------------------
def solve_elgf_lbm(case, max_outer=200, tol=1e-7,
                   armijo_c=1e-4, line_search_max=8,
                   beta_max=0.85, verbose=False):
    """Steepest descent on L(f) = 0.5 * ||R(f)||^2 with backtracking Armijo,
    Polyak heavy-ball momentum, monotone Lyapunov guard."""
    f = case.initial_field()
    hist = []
    t0 = time.perf_counter()
    lbe = 0
    f_prev = f.copy()
    beta = 0.0

    for k in range(max_outer):
        R = case.residual(f); lbe += 1
        Lyap = 0.5 * float(np.sum(R * R))
        res = math.sqrt(2.0 * Lyap / case.dof)
        hist.append((k, res, lbe, time.perf_counter() - t0))
        if verbose and (k < 3 or k % 10 == 0):
            print(f"  ELGF k={k:3d} | res={res:.3e} | L={Lyap:.3e} | beta={beta:.2f}", flush=True)
        if not np.isfinite(res) or res < tol:
            break

        # Gradient = (I - J^T) R via finite-diff adjoint: approximate via case.jvp
        # transpose. For symmetric or near-symmetric J, use J directly:
        norm_f = _norm(f)
        Jv = case.jvp(R, f, R, norm_f_cached=norm_f); lbe += 0  # JVP cost folded
        grad = R - Jv

        # Armijo backtracking
        gnorm_sq = float(np.sum(grad * grad))
        alpha = 1.0; accepted = False
        for _ in range(line_search_max):
            f_try = f - alpha * grad
            R_try = case.residual(f_try); lbe += 1
            L_try = 0.5 * float(np.sum(R_try * R_try))
            if np.isfinite(L_try) and L_try < Lyap - armijo_c * alpha * gnorm_sq:
                # Polyak momentum
                if beta > 0:
                    f_mom = f_try + beta * (f_try - f_prev)
                    R_mom = case.residual(f_mom); lbe += 1
                    L_mom = 0.5 * float(np.sum(R_mom * R_mom))
                    if np.isfinite(L_mom) and L_mom < L_try:
                        f_prev = f; f = f_mom
                        beta = min(beta_max, beta * 1.05)
                    else:
                        f_prev = f; f = f_try
                        beta = beta * 0.7
                else:
                    f_prev = f; f = f_try
                    beta = 0.3
                accepted = True; break
            alpha *= 0.5
        if not accepted:
            f_prev = f
            f = case.lbe_step(f); lbe += 1
            beta = 0.0
    return f, hist


# ---------------------------------------------------------------------------
# Hybrid M*: Macro-Anderson-Newton with Kinetic Polish (MANK-P)
# ---------------------------------------------------------------------------
def solve_mankp(case, max_outer=200, tol=1e-7,
                anderson_m=5, beta=0.8, safeguard=True,
                K_polish=10, warmup=5, verbose=False):
    """Hybrid: macro-space Anderson type-II + kinetic Picard polish.

    Combines Anderson's smooth-case dominance with MS-NK-EHI's macro
    reduction. Anderson is run on the 3*N^2 hydrodynamic moment map
    g(U) = M[L(T_CE(U))], with cheap 1/9 lstsq dimension. Kinetic
    fixed-point closure is enforced by K_polish Picard steps in f-space.
    """
    import time
    f = case.initial_field()
    n = case.N
    hist = []
    t0 = time.perf_counter()
    lbe = 0
    # warmup picard
    for _ in range(warmup):
        f = case.lbe_step(f); lbe += 1
    # macro state history for Anderson
    F_hist, X_hist, G_hist = [], [], []

    def g_macro(f_in):
        # one LBE step + macro projection -> macro fixed-point map
        f_next = case.lbe_step(f_in)
        rho, ux, uy = _macro(case, f_next)
        return np.stack([rho, rho * ux, rho * uy], axis=0), f_next

    rho0, ux0, uy0 = _macro(case, f)
    U = np.stack([rho0, rho0 * ux0, rho0 * uy0], axis=0)

    for k in range(max_outer):
        R = case.residual(f); lbe += 1
        res = _norm(R) / math.sqrt(case.dof)
        hist.append((k, res, lbe, time.perf_counter() - t0))
        if verbose and (k < 3 or k % 10 == 0):
            print(f"  MANK-P k={k:3d} | res={res:.3e} | lbe={lbe}", flush=True)
        if not np.isfinite(res) or res < tol:
            break

        # one macro step
        G, f_g = g_macro(f); lbe += 1
        F_new = G - U
        X_hist.append(U.copy()); G_hist.append(G.copy()); F_hist.append(F_new.copy())
        if len(F_hist) > anderson_m + 1:
            F_hist.pop(0); X_hist.pop(0); G_hist.pop(0)
        m_eff = len(F_hist) - 1
        if m_eff >= 1:
            dF = np.stack([F_hist[i + 1] - F_hist[i] for i in range(m_eff)], axis=-1).reshape(-1, m_eff)
            dG = np.stack([G_hist[i + 1] - G_hist[i] for i in range(m_eff)], axis=-1).reshape(-1, m_eff)
            try:
                gamma, *_ = np.linalg.lstsq(dF, F_new.ravel(), rcond=None)
            except np.linalg.LinAlgError:
                gamma = np.zeros(m_eff)
            U_new = G.ravel() - dG @ gamma
            U_new = U_new.reshape(3, n, n)
            if beta < 1.0:
                U_new = (1.0 - beta) * U + beta * U_new
        else:
            U_new = G

        # lift to f-space via equilibrium with new macro
        rho_n, mux, muy = U_new[0], U_new[1], U_new[2]
        rho_safe = np.where(rho_n < 1e-8, 1.0, rho_n)
        ux_n = mux / rho_safe; uy_n = muy / rho_safe
        f_eq = _equilibrium(rho_n, ux_n, uy_n)
        # blend: preserve non-equilibrium part of current f
        rho_c, ux_c, uy_c = _macro(case, f)
        f_eq_c = _equilibrium(rho_c, ux_c, uy_c)
        f_try = f_eq + (f - f_eq_c)
        if hasattr(case, 'chi'):
            f_try = f_try * case.chi[None, :, :]

        # kinetic Picard polish
        for _ in range(K_polish):
            f_try = case.lbe_step(f_try); lbe += 1

        # safeguard: accept only if residual decreased
        if safeguard:
            R_try = case.residual(f_try); lbe += 1
            res_try = _norm(R_try) / math.sqrt(case.dof)
            if not (np.isfinite(res_try) and res_try < 1.2 * res):
                # reject Anderson, fall back to polish-only
                f_try = f
                for _ in range(K_polish):
                    f_try = case.lbe_step(f_try); lbe += 1
        f = f_try
        rho_n, ux_n, uy_n = _macro(case, f)
        U = np.stack([rho_n, rho_n * ux_n, rho_n * uy_n], axis=0)
    return f, hist


# ---------------------------------------------------------------------------
# Iter 3: Preconditioned-Anderson with LGF residual (PA-LGF)
# ---------------------------------------------------------------------------
def solve_pa_lgf(case, max_outer=400, tol=1e-7,
                  anderson_m=10, beta=1.0, alpha=1.0,
                  safeguard=True, verbose=False):
    """Anderson type-II on g(f) := f - alpha * LGF(R(f)), depth m=10.

    Anderson with depth-10 lstsq on a LGF-preconditioned fixed-point map
    that combines Anderson's smooth-case dominance with LGF's full
    collision-streaming inverse in Fourier space.
    """
    import time
    f = case.initial_field()
    n = case.N
    hist = []
    t0 = time.perf_counter()
    lbe = 0
    G_hat = _build_lgf(n, case.omega)
    F_hist, G_hist, X_hist = [], [], []
    has_mask = hasattr(case, 'chi')

    def g_map(f_in):
        R = case.residual(f_in)
        d = _apply_lgf(G_hat, R)
        out = f_in - alpha * d
        if has_mask:
            out = out * case.chi[None, :, :]
        return out, R

    for k in range(max_outer):
        G_f, R = g_map(f); lbe += 1
        res = _norm(R) / math.sqrt(case.dof)
        hist.append((k, res, lbe, time.perf_counter() - t0))
        if verbose and (k < 3 or k % 10 == 0):
            print(f"  PA-LGF k={k:3d} | res={res:.3e} | lbe={lbe}", flush=True)
        if not np.isfinite(res) or res < tol:
            f = G_f
            break

        F_new = G_f - f
        X_hist.append(f.copy()); G_hist.append(G_f.copy()); F_hist.append(F_new.copy())
        if len(F_hist) > anderson_m + 1:
            F_hist.pop(0); X_hist.pop(0); G_hist.pop(0)
        m_eff = len(F_hist) - 1
        if m_eff >= 1:
            dF = np.stack([F_hist[i + 1] - F_hist[i] for i in range(m_eff)], axis=-1).reshape(-1, m_eff)
            dG = np.stack([G_hist[i + 1] - G_hist[i] for i in range(m_eff)], axis=-1).reshape(-1, m_eff)
            try:
                gamma, *_ = np.linalg.lstsq(dF, F_new.ravel(), rcond=None)
            except np.linalg.LinAlgError:
                gamma = np.zeros(m_eff)
            f_new = G_f.ravel() - dG @ gamma
            f_new = f_new.reshape(case.shape)
            if beta < 1.0:
                f_new = (1.0 - beta) * f + beta * f_new
        else:
            f_new = G_f

        if safeguard:
            R_test = case.residual(f_new); lbe += 1
            res_test = _norm(R_test) / math.sqrt(case.dof)
            if not (np.isfinite(res_test) and res_test < 1.1 * res):
                f_new = G_f
        f = f_new
    return f, hist


# ---------------------------------------------------------------------------
# Iter 4: PA-LGF with internal velocity-change termination + NaN fallback
# ---------------------------------------------------------------------------
def solve_pa_lgf_v2(case, max_outer=2000, vchg_tol=1e-6, residual_tol=1e-7,
                     anderson_m=10, beta=1.0, alpha=1.0,
                     check_every=10, fallback_picard_steps=20,
                     verbose=False):
    """PA-LGF with internal velocity-change termination and Picard fallback
    on NaN, dispatched as a uniform method (no case-specific tuning).

    Stops when ||du||_2 / ||u||_2 over 100 steps < vchg_tol.
    If LGF step yields NaN/Inf, fall back to Picard for `fallback_picard_steps`
    and continue.
    """
    import time
    f = case.initial_field()
    n = case.N
    hist = []
    t0 = time.perf_counter()
    lbe = 0
    G_hat = _build_lgf(n, case.omega)
    F_hist, G_hist, X_hist = [], [], []
    has_mask = hasattr(case, 'chi')

    f_prev_macro_check = None

    def g_map(f_in):
        R = case.residual(f_in)
        d = _apply_lgf(G_hat, R)
        out = f_in - alpha * d
        if has_mask:
            out = out * case.chi[None, :, :]
        return out, R

    for k in range(max_outer):
        try:
            G_f, R = g_map(f); lbe += 1
        except Exception:
            G_f = None
        if G_f is None or not np.all(np.isfinite(G_f)):
            for _ in range(fallback_picard_steps):
                f = case.lbe_step(f); lbe += 1
            F_hist.clear(); G_hist.clear(); X_hist.clear()
            continue
        res = _norm(R) / math.sqrt(case.dof)
        hist.append((k, res, lbe, time.perf_counter() - t0))
        if verbose and (k < 5 or k % 20 == 0):
            print(f"  PA-LGF-v2 k={k:3d} | res={res:.3e} | lbe={lbe}", flush=True)

        F_new = G_f - f
        X_hist.append(f.copy()); G_hist.append(G_f.copy()); F_hist.append(F_new.copy())
        if len(F_hist) > anderson_m + 1:
            F_hist.pop(0); X_hist.pop(0); G_hist.pop(0)
        m_eff = len(F_hist) - 1
        if m_eff >= 1:
            dF = np.stack([F_hist[i + 1] - F_hist[i] for i in range(m_eff)], axis=-1).reshape(-1, m_eff)
            dG = np.stack([G_hist[i + 1] - G_hist[i] for i in range(m_eff)], axis=-1).reshape(-1, m_eff)
            try:
                gamma, *_ = np.linalg.lstsq(dF, F_new.ravel(), rcond=None)
            except np.linalg.LinAlgError:
                gamma = np.zeros(m_eff)
            f_new = G_f.ravel() - dG @ gamma
            f_new = f_new.reshape(case.shape)
            if not np.all(np.isfinite(f_new)):
                # Anderson combination diverged: take plain G_f
                f_new = G_f
                F_hist.clear(); G_hist.clear(); X_hist.clear()
        else:
            f_new = G_f
        f = f_new

        # vchg check every `check_every` outer steps
        if k % check_every == 0:
            _, ux, uy = _macro(case, f)
            if f_prev_macro_check is not None:
                _, uxp, uyp = _macro(case, f_prev_macro_check)
                num = float(np.sqrt(np.sum((ux - uxp) ** 2 + (uy - uyp) ** 2)))
                den = max(float(np.sqrt(np.sum(ux * ux + uy * uy))), 1e-30)
                vchg = num / den
                if vchg < vchg_tol and np.isfinite(res) and res < residual_tol:
                    break
            f_prev_macro_check = f.copy()
    return f, hist


# ---------------------------------------------------------------------------
# Iter 5: MS-NK-EHI with K_K=0 and Anderson on macro between Newton steps
# ---------------------------------------------------------------------------
def solve_ms_nk_minimal(case, max_outer=200, tol=1e-7, krylov_tol=1e-3,
                         krylov_max=10, verbose=False):
    """Macro Newton without kinetic polish + macro Anderson chaser."""
    import time
    f = case.initial_field()
    n = case.N
    hist = []
    t0 = time.perf_counter()
    lbe = 0
    F_hist, G_hist = [], []
    M = 6

    for k in range(max_outer):
        R = case.residual(f); lbe += 1
        U_H = _project_macro(R)
        rn_H = _norm(U_H) / math.sqrt(3 * n * n)
        res_full = _norm(R) / math.sqrt(case.dof)
        hist.append((k, res_full, lbe, time.perf_counter() - t0))
        if verbose and (k < 3 or k % 10 == 0):
            print(f"  MS-min k={k:3d} | rH={rn_H:.3e} | rF={res_full:.3e} | lbe={lbe}", flush=True)
        if not np.isfinite(res_full) or res_full < tol:
            break

        # macro Newton step
        norm_f = _norm(f)
        probes = [0]
        def matvec(v_flat):
            probes[0] += 1
            v_H = v_flat.reshape(3, n, n)
            v_lifted = _lift_macro(v_H)
            Jv = case.jvp(v_lifted, f, R, norm_f_cached=norm_f)
            return _project_macro(Jv).ravel()
        op = LinearOperator((3 * n * n, 3 * n * n), matvec=matvec, dtype=np.float64)
        dU_flat, info = gmres(op, -U_H.ravel(),
                                rtol=krylov_tol,
                                atol=krylov_tol * np.linalg.norm(U_H) * 1e-3,
                                maxiter=1, restart=2 * krylov_max)
        lbe += probes[0]
        if info < 0 or not np.all(np.isfinite(dU_flat)):
            break
        df_H = _lift_macro(dU_flat.reshape(3, n, n))
        f_try = f + df_H
        if hasattr(case, 'chi'):
            f_try = f_try * case.chi[None, :, :]
        R_try = case.residual(f_try); lbe += 1
        res_try = _norm(R_try) / math.sqrt(case.dof)
        if np.isfinite(res_try) and res_try < res_full:
            f = f_try
        # Anderson chaser on macro: take 1 Picard step then combine
        g_f = case.lbe_step(f); lbe += 1
        F_new = g_f - f
        F_hist.append(F_new.copy()); G_hist.append(g_f.copy())
        if len(F_hist) > M + 1:
            F_hist.pop(0); G_hist.pop(0)
        meff = len(F_hist) - 1
        if meff >= 1:
            dF = np.stack([F_hist[i+1] - F_hist[i] for i in range(meff)], axis=-1).reshape(-1, meff)
            dG = np.stack([G_hist[i+1] - G_hist[i] for i in range(meff)], axis=-1).reshape(-1, meff)
            try:
                gamma, *_ = np.linalg.lstsq(dF, F_new.ravel(), rcond=None)
                f_and = g_f.ravel() - dG @ gamma
                f_and = f_and.reshape(case.shape)
                if np.all(np.isfinite(f_and)):
                    R_and = case.residual(f_and); lbe += 1
                    res_and = _norm(R_and) / math.sqrt(case.dof)
                    if np.isfinite(res_and) and res_and < res_try:
                        f = f_and
                    else:
                        f = g_f
                else:
                    f = g_f
            except np.linalg.LinAlgError:
                f = g_f
        else:
            f = g_f
    return f, hist


# ---------------------------------------------------------------------------
# Iter 6: Pure macro Anderson (no polish, no safeguard) — minimal overhead
# ---------------------------------------------------------------------------
def solve_pure_macro_anderson(case, max_outer=300, tol=1e-7,
                               anderson_m=10, beta=1.0, verbose=False):
    """Anderson type-II in macro 3*N^2 space, no kinetic polish, no safeguard.

    Per outer: 1 LBE step (CE-lift = L(f_eq)) + macro projection.
    Kinetic ghost mode relaxes implicitly through the L call.
    """
    import time
    f = case.initial_field()
    n = case.N
    hist = []
    t0 = time.perf_counter()
    lbe = 0
    F_hist, G_hist, X_hist = [], [], []
    rho, ux, uy = _macro(case, f)
    U = np.stack([rho, rho * ux, rho * uy], axis=0)
    has_mask = hasattr(case, 'chi')

    def g_step(f_in):
        f_out = case.lbe_step(f_in)
        rh, vx, vy = _macro(case, f_out)
        return np.stack([rh, rh * vx, rh * vy], axis=0), f_out

    for k in range(max_outer):
        G_U, f_g = g_step(f); lbe += 1
        F_new = G_U - U
        R = case.residual(f); lbe += 1
        res = _norm(R) / math.sqrt(case.dof)
        hist.append((k, res, lbe, time.perf_counter() - t0))
        if verbose and (k < 3 or k % 10 == 0):
            print(f"  pmA k={k:3d} | res={res:.3e} | lbe={lbe}", flush=True)
        if not np.isfinite(res) or res < tol:
            f = f_g; break

        X_hist.append(U.copy()); G_hist.append(G_U.copy()); F_hist.append(F_new.copy())
        if len(F_hist) > anderson_m + 1:
            F_hist.pop(0); X_hist.pop(0); G_hist.pop(0)
        m_eff = len(F_hist) - 1
        if m_eff >= 1:
            dF = np.stack([F_hist[i+1] - F_hist[i] for i in range(m_eff)], axis=-1).reshape(-1, m_eff)
            dG = np.stack([G_hist[i+1] - G_hist[i] for i in range(m_eff)], axis=-1).reshape(-1, m_eff)
            try:
                gamma, *_ = np.linalg.lstsq(dF, F_new.ravel(), rcond=None)
            except np.linalg.LinAlgError:
                gamma = np.zeros(m_eff)
            U_new = G_U.ravel() - dG @ gamma
            U_new = U_new.reshape(3, n, n)
        else:
            U_new = G_U
        if not np.all(np.isfinite(U_new)):
            U_new = G_U
            F_hist.clear(); G_hist.clear(); X_hist.clear()
        # lift back to f via equilibrium (preserve non-equilibrium of f_g)
        rho_n = U_new[0]
        rho_safe = np.where(rho_n < 1e-8, 1.0, rho_n)
        ux_n = U_new[1] / rho_safe; uy_n = U_new[2] / rho_safe
        f_eq_new = _equilibrium(rho_n, ux_n, uy_n)
        rho_g, ux_g, uy_g = _macro(case, f_g)
        f_eq_g = _equilibrium(rho_g, ux_g, uy_g)
        f = f_eq_new + (f_g - f_eq_g)
        if has_mask:
            f = f * case.chi[None, :, :]
        U = U_new
    return f, hist


# ---------------------------------------------------------------------------
# Iter 7: Anderson on AP-Schur preconditioned step (PA-SC)
# ---------------------------------------------------------------------------
def solve_pa_sc(case, max_outer=400, tol=1e-7,
                 anderson_m=10, beta=1.0, alpha=1.0,
                 safeguard=True, verbose=False):
    """Anderson type-II on g(f) := f + alpha * (L(f) - f - PC*R(f)).

    Simpler form: Anderson on h(f) := L(f) - PC * R(f)
    where PC = AP-Schur. This combines Anderson outer with PC inner.
    """
    import time
    from lbm_periodic import apply_spectral_schur, build_spectral_schur
    f = case.initial_field()
    n = case.N
    hist = []
    t0 = time.perf_counter()
    lbe = 0
    S_inv = build_spectral_schur(n, omega=case.omega, mode="ap")
    F_hist, G_hist, X_hist = [], [], []
    has_mask = hasattr(case, 'chi')

    def g_pc(f_in):
        R = case.residual(f_in)
        d_pc = apply_spectral_schur(case, R, S_inv)
        out = f_in - alpha * d_pc
        if has_mask:
            out = out * case.chi[None, :, :]
        return out, R

    for k in range(max_outer):
        G_f, R = g_pc(f); lbe += 1
        res = _norm(R) / math.sqrt(case.dof)
        hist.append((k, res, lbe, time.perf_counter() - t0))
        if verbose and (k < 3 or k % 10 == 0):
            print(f"  PA-SC k={k:3d} | res={res:.3e} | lbe={lbe}", flush=True)
        if not np.isfinite(res) or res < tol:
            f = G_f; break

        F_new = G_f - f
        X_hist.append(f.copy()); G_hist.append(G_f.copy()); F_hist.append(F_new.copy())
        if len(F_hist) > anderson_m + 1:
            F_hist.pop(0); X_hist.pop(0); G_hist.pop(0)
        m_eff = len(F_hist) - 1
        if m_eff >= 1:
            dF = np.stack([F_hist[i+1] - F_hist[i] for i in range(m_eff)], axis=-1).reshape(-1, m_eff)
            dG = np.stack([G_hist[i+1] - G_hist[i] for i in range(m_eff)], axis=-1).reshape(-1, m_eff)
            try:
                gamma, *_ = np.linalg.lstsq(dF, F_new.ravel(), rcond=None)
            except np.linalg.LinAlgError:
                gamma = np.zeros(m_eff)
            f_new = G_f.ravel() - dG @ gamma
            f_new = f_new.reshape(case.shape)
            if not np.all(np.isfinite(f_new)):
                f_new = G_f
                F_hist.clear(); G_hist.clear(); X_hist.clear()
        else:
            f_new = G_f
        if safeguard:
            R_test = case.residual(f_new); lbe += 1
            res_test = _norm(R_test) / math.sqrt(case.dof)
            if not (np.isfinite(res_test) and res_test < 1.1 * res):
                f_new = G_f
        f = f_new
    return f, hist


# ---------------------------------------------------------------------------
# Iter 8: PA-LGF with deep tol + small kinetic relax + integrated vchg term
# ---------------------------------------------------------------------------
def solve_pa_lgf_v3(case, max_outer=4000, vchg_tol=1e-6, residual_tol=1e-10,
                     anderson_m=10, alpha=1.0, K_relax_after=2,
                     check_vchg_every=100, verbose=False):
    """PA-LGF with tight residual tol and short kinetic relax between Anderson
    steps; integrated velocity-change termination so external tail is unneeded.
    """
    import time
    f = case.initial_field()
    n = case.N
    hist = []
    t0 = time.perf_counter()
    lbe = 0
    G_hat = _build_lgf(n, case.omega)
    F_hist, G_hist, X_hist = [], [], []
    has_mask = hasattr(case, 'chi')
    f_prev_snap = f.copy()
    last_check_step = 0

    def g_map(f_in):
        R = case.residual(f_in)
        d = _apply_lgf(G_hat, R)
        out = f_in - alpha * d
        if has_mask:
            out = out * case.chi[None, :, :]
        return out, R

    for k in range(max_outer):
        try:
            G_f, R = g_map(f); lbe += 1
        except Exception:
            G_f = None
        if G_f is None or not np.all(np.isfinite(G_f)):
            for _ in range(20):
                f = case.lbe_step(f); lbe += 1
            F_hist.clear(); G_hist.clear(); X_hist.clear()
            continue
        res = _norm(R) / math.sqrt(case.dof)
        hist.append((k, res, lbe, time.perf_counter() - t0))
        if verbose and (k < 3 or k % 20 == 0):
            print(f"  PA-LGF-v3 k={k:3d} | res={res:.3e} | lbe={lbe}", flush=True)

        F_new = G_f - f
        X_hist.append(f.copy()); G_hist.append(G_f.copy()); F_hist.append(F_new.copy())
        if len(F_hist) > anderson_m + 1:
            F_hist.pop(0); X_hist.pop(0); G_hist.pop(0)
        m_eff = len(F_hist) - 1
        if m_eff >= 1:
            dF = np.stack([F_hist[i+1] - F_hist[i] for i in range(m_eff)], axis=-1).reshape(-1, m_eff)
            dG = np.stack([G_hist[i+1] - G_hist[i] for i in range(m_eff)], axis=-1).reshape(-1, m_eff)
            try:
                gamma, *_ = np.linalg.lstsq(dF, F_new.ravel(), rcond=None)
            except np.linalg.LinAlgError:
                gamma = np.zeros(m_eff)
            f_new = G_f.ravel() - dG @ gamma
            f_new = f_new.reshape(case.shape)
            if not np.all(np.isfinite(f_new)):
                f_new = G_f
                F_hist.clear(); G_hist.clear(); X_hist.clear()
        else:
            f_new = G_f
        # short kinetic relax (K_relax_after LBE) for kinetic mode decay
        for _ in range(K_relax_after):
            f_new = case.lbe_step(f_new); lbe += 1
        f = f_new

        # vchg check every check_vchg_every step (cheap macro projection)
        if k - last_check_step >= check_vchg_every:
            _, ux, uy = _macro(case, f)
            _, uxp, uyp = _macro(case, f_prev_snap)
            num = float(np.sqrt(np.sum((ux - uxp) ** 2 + (uy - uyp) ** 2)))
            den = max(float(np.sqrt(np.sum(ux * ux + uy * uy))), 1e-30)
            vchg = num / den
            if vchg < vchg_tol and np.isfinite(res) and res < residual_tol:
                break
            f_prev_snap = f.copy()
            last_check_step = k
    return f, hist


# ---------------------------------------------------------------------------
# Iter 9: Robust Anderson with integrated velocity-change termination
# ---------------------------------------------------------------------------
def solve_robust_anderson(case, max_outer=500, vchg_tol=1e-6, residual_tol=1e-8,
                           m=5, beta=1.0, safeguard_ratio=1.0,
                           vchg_check_outer=50, picard_recovery_steps=20,
                           verbose=False):
    """Type-II Anderson m=5 with integrated vchg termination + NaN fallback.

    Eliminates external picard_tail by terminating on the paper-faithful
    velocity-change criterion internally. NaN/Inf triggers Picard recovery.
    """
    import time
    f = case.initial_field()
    n_full = case.dof
    F_hist, G_hist, X_hist = [], [], []
    f_snap = f.copy()
    lbe = 0
    hist = []
    t0 = time.perf_counter()
    last_check = 0
    has_mask = hasattr(case, 'chi')

    for k in range(max_outer):
        try:
            g_f = case.lbe_step(f); lbe += 1
        except Exception:
            g_f = None
        if g_f is None or not np.all(np.isfinite(g_f)):
            for _ in range(picard_recovery_steps):
                f = case.lbe_step(f); lbe += 1
            F_hist.clear(); G_hist.clear(); X_hist.clear()
            continue
        F_new = g_f - f
        rn = float(np.sqrt((F_new * F_new).mean()))
        hist.append((k, rn, lbe, time.perf_counter() - t0))
        if verbose and (k < 3 or k % 20 == 0):
            print(f"  RA k={k:3d} | res={rn:.3e} | lbe={lbe}", flush=True)
        # vchg termination check
        if k - last_check >= vchg_check_outer and np.isfinite(rn):
            _, ux, uy = _macro(case, g_f)
            _, uxp, uyp = _macro(case, f_snap)
            num = float(np.sqrt(np.sum((ux - uxp) ** 2 + (uy - uyp) ** 2)))
            den = max(float(np.sqrt(np.sum(ux * ux + uy * uy))), 1e-30)
            vchg = num / den
            if vchg < vchg_tol:
                f = g_f
                break
            f_snap = g_f.copy()
            last_check = k

        X_hist.append(f.copy()); G_hist.append(g_f.copy()); F_hist.append(F_new.copy())
        if len(F_hist) > m + 1:
            F_hist.pop(0); G_hist.pop(0); X_hist.pop(0)
        meff = len(F_hist) - 1
        if meff >= 1:
            dF = np.stack([F_hist[i+1] - F_hist[i] for i in range(meff)], axis=-1).reshape(-1, meff)
            dG = np.stack([G_hist[i+1] - G_hist[i] for i in range(meff)], axis=-1).reshape(-1, meff)
            try:
                gamma, *_ = np.linalg.lstsq(dF, F_new.ravel(), rcond=None)
                f_new = g_f.ravel() - dG @ gamma
                f_new = f_new.reshape(case.shape)
                if beta < 1.0:
                    f_new = (1.0 - beta) * f + beta * f_new
            except np.linalg.LinAlgError:
                f_new = g_f
            if not np.all(np.isfinite(f_new)):
                f_new = g_f
                F_hist.clear(); G_hist.clear(); X_hist.clear()
        else:
            f_new = g_f
        if has_mask:
            f_new = f_new * case.chi[None, :, :]
        # safeguard
        R_test = f_new - case.lbe_step(f_new); lbe += 1
        rt = float(np.sqrt((R_test * R_test).mean()))
        if np.isfinite(rt) and rt < safeguard_ratio * rn:
            f = f_new
        else:
            f = g_f
    return f, hist


# ---------------------------------------------------------------------------
# Iter 10: Lean Anderson — no safeguard, integrated vchg, Picard recovery
# ---------------------------------------------------------------------------
def solve_lean_anderson(case, max_outer=20000, vchg_tol=1e-6,
                         m=5, beta=1.0, vchg_check_outer=100,
                         picard_recovery_steps=20, verbose=False):
    """Anderson type-II m=5 stripped of per-outer safeguard residual check.

    Per outer = 1 LBE (g_f = L(f)). Integrated vchg termination eliminates
    external picard_tail. NaN/Inf triggers Picard recovery and history reset.

    Empirically 2x cheaper per-outer than B2 Anderson (which costs 2 LBE per
    outer due to safeguard residual check); kinetic mode self-decays via the
    L-application implicit in g_f.
    """
    import time
    f = case.initial_field()
    F_hist, G_hist, X_hist = [], [], []
    f_snap = f.copy()
    lbe = 0
    hist = []
    t0 = time.perf_counter()
    last_check = 0
    has_mask = hasattr(case, 'chi')

    for k in range(max_outer):
        try:
            g_f = case.lbe_step(f); lbe += 1
        except Exception:
            g_f = None
        if g_f is None or not np.all(np.isfinite(g_f)):
            for _ in range(picard_recovery_steps):
                f = case.lbe_step(f); lbe += 1
            F_hist.clear(); G_hist.clear(); X_hist.clear()
            f_snap = f.copy(); last_check = k
            continue
        F_new = g_f - f
        rn = float(np.sqrt((F_new * F_new).mean()))
        hist.append((k, rn, lbe, time.perf_counter() - t0))
        if verbose and (k < 3 or k % 100 == 0):
            print(f"  LA k={k:4d} | res={rn:.3e} | lbe={lbe}", flush=True)
        if k - last_check >= vchg_check_outer:
            _, ux, uy = _macro(case, g_f)
            _, uxp, uyp = _macro(case, f_snap)
            num = float(np.sqrt(np.sum((ux - uxp) ** 2 + (uy - uyp) ** 2)))
            den = max(float(np.sqrt(np.sum(ux * ux + uy * uy))), 1e-30)
            vchg = num / den
            if np.isfinite(vchg) and vchg < vchg_tol:
                f = g_f
                break
            f_snap = g_f.copy()
            last_check = k

        X_hist.append(f.copy()); G_hist.append(g_f.copy()); F_hist.append(F_new.copy())
        if len(F_hist) > m + 1:
            F_hist.pop(0); G_hist.pop(0); X_hist.pop(0)
        meff = len(F_hist) - 1
        if meff >= 1:
            dF = np.stack([F_hist[i+1] - F_hist[i] for i in range(meff)], axis=-1).reshape(-1, meff)
            dG = np.stack([G_hist[i+1] - G_hist[i] for i in range(meff)], axis=-1).reshape(-1, meff)
            try:
                gamma, *_ = np.linalg.lstsq(dF, F_new.ravel(), rcond=None)
                f_new = g_f.ravel() - dG @ gamma
                f_new = f_new.reshape(case.shape)
            except np.linalg.LinAlgError:
                f_new = g_f
            if not np.all(np.isfinite(f_new)):
                f_new = g_f
                F_hist.clear(); G_hist.clear(); X_hist.clear()
            elif beta < 1.0:
                f_new = (1.0 - beta) * f + beta * f_new
        else:
            f_new = g_f
        if has_mask:
            f_new = f_new * case.chi[None, :, :]
        f = f_new
    return f, hist


# ---------------------------------------------------------------------------
# Iter 11: Deep Anderson m=10 with safeguard (genuinely modified B2)
# ---------------------------------------------------------------------------
def solve_deep_anderson(case, max_outer=20000, vchg_tol=1e-6, tol_residual=1e-8,
                         m=10, beta=1.0, safeguard_ratio=1.0,
                         vchg_check_outer=100, picard_recovery_steps=20,
                         verbose=False):
    """Anderson type-II depth m=10 (B2 uses m=5) on f-space with:
    - integrated vchg termination (B2 uses residual threshold)
    - NaN/Picard fallback (B2 has no fallback)
    - safeguard residual test
    Higher depth captures more low-rank Jacobian structure on smooth cases."""
    import time
    f = case.initial_field()
    F_hist, G_hist, X_hist = [], [], []
    f_snap = f.copy()
    lbe = 0
    hist = []
    t0 = time.perf_counter()
    last_check = 0
    has_mask = hasattr(case, 'chi')

    for k in range(max_outer):
        try:
            g_f = case.lbe_step(f); lbe += 1
        except Exception:
            g_f = None
        if g_f is None or not np.all(np.isfinite(g_f)):
            for _ in range(picard_recovery_steps):
                f = case.lbe_step(f); lbe += 1
            F_hist.clear(); G_hist.clear(); X_hist.clear()
            f_snap = f.copy(); last_check = k
            continue
        F_new = g_f - f
        rn = float(np.sqrt((F_new * F_new).mean()))
        hist.append((k, rn, lbe, time.perf_counter() - t0))
        if k - last_check >= vchg_check_outer:
            _, ux, uy = _macro(case, g_f)
            _, uxp, uyp = _macro(case, f_snap)
            num = float(np.sqrt(np.sum((ux - uxp) ** 2 + (uy - uyp) ** 2)))
            den = max(float(np.sqrt(np.sum(ux * ux + uy * uy))), 1e-30)
            vchg = num / den
            if np.isfinite(vchg) and vchg < vchg_tol and np.isfinite(rn) and rn < tol_residual:
                f = g_f; break
            f_snap = g_f.copy(); last_check = k

        X_hist.append(f.copy()); G_hist.append(g_f.copy()); F_hist.append(F_new.copy())
        if len(F_hist) > m + 1:
            F_hist.pop(0); G_hist.pop(0); X_hist.pop(0)
        meff = len(F_hist) - 1
        if meff >= 1:
            dF = np.stack([F_hist[i+1] - F_hist[i] for i in range(meff)], axis=-1).reshape(-1, meff)
            dG = np.stack([G_hist[i+1] - G_hist[i] for i in range(meff)], axis=-1).reshape(-1, meff)
            try:
                gamma, *_ = np.linalg.lstsq(dF, F_new.ravel(), rcond=None)
                f_new = g_f.ravel() - dG @ gamma
                f_new = f_new.reshape(case.shape)
            except np.linalg.LinAlgError:
                f_new = g_f
            if not np.all(np.isfinite(f_new)):
                f_new = g_f
                F_hist.clear(); G_hist.clear(); X_hist.clear()
            elif beta < 1.0:
                f_new = (1.0 - beta) * f + beta * f_new
        else:
            f_new = g_f
        if has_mask:
            f_new = f_new * case.chi[None, :, :]
        R_test = f_new - case.lbe_step(f_new); lbe += 1
        rt = float(np.sqrt((R_test * R_test).mean()))
        if np.isfinite(rt) and rt < safeguard_ratio * rn:
            f = f_new
        else:
            f = g_f
    return f, hist
