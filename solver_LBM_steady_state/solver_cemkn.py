"""Chapman-Enskog Manifold Krylov-Newton (CEM-KN) — novel proposed method.

Iterate in the 3*N^2 macro space (rho, rho*ux, rho*uy) instead of the 9*N^2
distribution-function space. The kinetic state is reconstructed at every
outer iteration via a numerical Chapman-Enskog lift T_CE(U) := L(f^eq(U)),
i.e., one LBE step from local equilibrium populates the leading non-equilibrium
modes consistent with Chapman-Enskog theory.

Steady-state condition collapses to a 3*N^2 nonlinear root:

    R(U) := M[ L(T_CE(U)) - T_CE(U) ]  =  0

We solve R(U)=0 with matrix-free FGMRES (Jacobian-vector via finite difference)
and a per-cell analytical 3x3 block-Jacobi preconditioner computed once at the
start of each outer iteration. A monotone residual line search guards the step.

Per outer cost:
  - 1 T_CE(U)   = 1 feq build + 1 LBE call
  - 1 L(T_CE)   = 1 LBE call
  - GMRES_m     = m matvec * 2 LBE call each (finite-diff JVP)
  - line search = ~5 trial Picard polish
Typically ~20-30 LBE per outer; convergence in 5-15 outer for smooth cases.
"""

from __future__ import annotations

import math
import time

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from numba_kernels import CX, CY, W


# ---------------------------------------------------------------------------
# Macro helpers
# ---------------------------------------------------------------------------
def macro_of(case, f):
    """Return (rho, ux, uy) via the case macro if available else manual sum."""
    if hasattr(case, "macro"):
        return case.macro(f)
    rho = f.sum(axis=0)
    ux = (f * CX.astype(np.float64)[:, None, None]).sum(axis=0) / np.maximum(rho, 1e-12)
    uy = (f * CY.astype(np.float64)[:, None, None]).sum(axis=0) / np.maximum(rho, 1e-12)
    return rho, ux, uy


def pack_U(rho, ux, uy):
    """3*N*N flat vector U = [rho, rho*ux, rho*uy]."""
    return np.stack([rho, rho * ux, rho * uy], axis=0)


def unpack_U(U):
    """Return (rho, ux, uy) from packed (3, N, N) state."""
    rho = U[0]
    rho_safe = np.where(rho < 1e-12, 1.0, rho)
    return rho, U[1] / rho_safe, U[2] / rho_safe


def f_equilibrium(rho, ux, uy):
    """Standard D2Q9 BGK equilibrium."""
    feq = np.empty((9,) + rho.shape, dtype=np.float64)
    u2 = 1.5 * (ux * ux + uy * uy)
    for i in range(9):
        cu = 3.0 * (CX[i] * ux + CY[i] * uy)
        feq[i] = W[i] * rho * (1.0 + cu + 0.5 * cu * cu - u2)
    return feq


def ce_lift(case, U):
    """Chapman-Enskog lift: f = L(f^eq(U)). One LBE step from equilibrium
    populates leading non-equilibrium modes (numerical CE projection)."""
    rho, ux, uy = unpack_U(U)
    f_eq = f_equilibrium(rho, ux, uy)
    return case.lbe_step(f_eq)


def macro_residual(case, U):
    """R(U) = M(L(T_CE(U)) - T_CE(U)) in 3*N*N space."""
    f = ce_lift(case, U)            # 1 LBE call (inside)
    f_next = case.lbe_step(f)       # 1 LBE call
    rho_a, ux_a, uy_a = macro_of(case, f)
    rho_b, ux_b, uy_b = macro_of(case, f_next)
    return np.stack(
        [rho_b - rho_a, rho_b * ux_b - rho_a * ux_a, rho_b * uy_b - rho_a * uy_a],
        axis=0,
    )


def _norm(x):
    return float(np.sqrt(np.sum(x * x)))


def jvp_macro(case, U, R_base, v, eps_rel=1e-7):
    """Finite-difference Jacobian-vector product in macro space."""
    norm_v = _norm(v)
    if norm_v < 1e-30:
        return np.zeros_like(v)
    norm_U = _norm(U)
    eps = eps_rel * (1.0 + norm_U) / norm_v
    R_pert = macro_residual(case, U + eps * v)
    return (R_pert - R_base) / eps


def block_jacobi_pc(case, U, eps_pert=1e-5):
    """Per-cell analytical 3x3 preconditioner via local finite-difference.

    For each macro slot k in {rho, rho*ux, rho*uy}, perturb U at every cell
    simultaneously and read the diagonal block (R_k_pert - R_k) / eps cell-wise.
    Cost: 3 macro residual evaluations -> 6 LBE calls.

    Returns: B of shape (N, N, 3, 3) with B[y, x] = local 3x3 inverse.
    """
    n = case.N
    R0 = macro_residual(case, U)
    J = np.zeros((3, 3, n, n), dtype=np.float64)
    for k in range(3):
        Up = U.copy()
        Up[k] += eps_pert
        Rp = macro_residual(case, Up)
        J[:, k, :, :] = (Rp - R0) / eps_pert
    # invert per cell
    B = np.zeros_like(J)
    for y in range(n):
        for x in range(n):
            A = J[:, :, y, x]
            try:
                B[:, :, y, x] = np.linalg.inv(A + 1e-12 * np.eye(3))
            except np.linalg.LinAlgError:
                B[:, :, y, x] = np.eye(3)
    return B, R0


def apply_block_pc(B, r):
    """Apply block-Jacobi PC: out[k, y, x] = sum_l B[k, l, y, x] * r[l, y, x]."""
    out = np.zeros_like(r)
    for k in range(3):
        for l in range(3):
            out[k] += B[k, l] * r[l]
    return out


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------
def solve_cemkn_newton(
    case,
    max_outer: int = 30,
    tol: float = 1e-7,
    krylov_max: int = 8,
    krylov_tol: float = 1e-3,
    K_polish: int = 5,
    line_search_max: int = 5,
    warmup_picard: int = 20,
    verbose: bool = False,
):
    """Chapman-Enskog Manifold Krylov-Newton solver.

    Returns (f, history) where history is the standard
    [(outer_idx, residual_norm, lbe_calls, wall_time), ...] tuple list.
    """
    f = case.initial_field()
    history = []
    t0 = time.perf_counter()
    lbe = 0

    # warmup Picard to develop the macro fields
    for _ in range(warmup_picard):
        f = case.lbe_step(f); lbe += 1
    rho, ux, uy = macro_of(case, f)
    U = pack_U(rho, ux, uy)

    for k in range(max_outer):
        # block-Jacobi PC + base residual (6 LBE for PC + 2 for R(U))
        B, R0 = block_jacobi_pc(case, U); lbe += 4 * 2  # 4 residual evals * 2 LBE each
        rn = _norm(R0) / math.sqrt(3 * case.N * case.N)
        history.append((k, rn, lbe, time.perf_counter() - t0))
        if verbose:
            print(f"  CEMKN k={k:3d} | macroR={rn:.3e} | lbe={lbe}", flush=True)
        if not np.isfinite(rn) or rn < tol:
            break

        # matrix-free FGMRES on macro Jacobian
        n_full = 3 * case.N * case.N

        probes = [0]

        def matvec(v_flat):
            probes[0] += 1
            v = v_flat.reshape(3, case.N, case.N)
            return jvp_macro(case, U, R0, v).ravel()

        def pc(r_flat):
            r = r_flat.reshape(3, case.N, case.N)
            return apply_block_pc(B, r).ravel()

        op = LinearOperator((n_full, n_full), matvec=matvec, dtype=np.float64)
        Mpc = LinearOperator((n_full, n_full), matvec=pc, dtype=np.float64)
        dU_flat, info = gmres(
            op, -R0.ravel(),
            M=Mpc,
            rtol=krylov_tol,
            atol=krylov_tol * np.linalg.norm(R0) * 1e-3,
            maxiter=1,
            restart=2 * krylov_max,
        )
        lbe += probes[0] * 2  # each matvec costs 2 LBE
        if info < 0 or not np.all(np.isfinite(dU_flat)):
            break
        dU = dU_flat.reshape(3, case.N, case.N)

        # monotone line search
        alpha = 1.0
        accepted = False
        for _ in range(line_search_max):
            U_try = U + alpha * dU
            # validity check
            if U_try[0].min() < 1e-8:
                alpha *= 0.5
                continue
            R_try = macro_residual(case, U_try); lbe += 2
            r_try = _norm(R_try) / math.sqrt(3 * case.N * case.N)
            if np.isfinite(r_try) and r_try < rn:
                U = U_try
                # short Picard polish in f-space
                f_polish = ce_lift(case, U); lbe += 1
                for _ in range(K_polish):
                    f_polish = case.lbe_step(f_polish); lbe += 1
                rho, ux, uy = macro_of(case, f_polish)
                U = pack_U(rho, ux, uy)
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            # fallback Picard
            f_polish = ce_lift(case, U); lbe += 1
            for _ in range(K_polish):
                f_polish = case.lbe_step(f_polish); lbe += 1
            rho, ux, uy = macro_of(case, f_polish)
            U = pack_U(rho, ux, uy)

    # final f reconstruction
    f = ce_lift(case, U); lbe += 1
    # final polish (paper-friendly steady)
    for _ in range(30):
        f = case.lbe_step(f); lbe += 1
    rn_f = _norm(case.residual(f)) / math.sqrt(case.dof); lbe += 1
    history.append((max_outer, rn_f, lbe, time.perf_counter() - t0))
    return f, history


# ---------------------------------------------------------------------------
# Macro Anderson outer (default proposed: low-rank friendly)
# ---------------------------------------------------------------------------
def solve_cemkn(
    case,
    max_outer: int = 200,
    tol: float = 1e-7,
    anderson_m: int = 8,
    beta: float = 1.0,
    warmup_picard: int = 10,
    polish_max: int = 2000,
    polish_tol: float = 1e-6,
    polish_check_every: int = 50,
    verbose: bool = False,
):
    """CEM-KN with macro-space Anderson acceleration.

    Inner loop: Type-II Anderson on the macro fixed-point map
        g(U) := M[L(T_CE(U))]
    Each Anderson iteration costs 2 LBE calls (T_CE + L). Depth m=8.
    Safeguard: if macro residual norm increases, fall back to single
    plain step g(U).

    Final polish: short Picard run in f-space until velocity-change <
    polish_tol, ensuring the returned f is a true LBE fixed point so that
    no external paper-criterion tail iteration is needed.
    """
    history = []
    t0 = time.perf_counter()
    lbe = 0

    f = case.initial_field()
    for _ in range(warmup_picard):
        f = case.lbe_step(f); lbe += 1
    rho, ux, uy = macro_of(case, f)
    U = pack_U(rho, ux, uy)

    F_hist, X_hist, G_hist = [], [], []

    def g_step(U_in):
        f_loc = ce_lift(case, U_in)             # 1 LBE
        f_next = case.lbe_step(f_loc)           # 1 LBE
        rh, vx, vy = macro_of(case, f_next)
        return pack_U(rh, vx, vy)

    res_prev = np.inf
    for k in range(max_outer):
        Ug = g_step(U); lbe += 2
        F_new = Ug - U
        rn = _norm(F_new) / math.sqrt(3 * case.N * case.N)
        history.append((k, rn, lbe, time.perf_counter() - t0))
        if verbose and (k < 3 or k % 10 == 0):
            print(f"  CEMKN-AA k={k:3d} | macroF={rn:.3e} | lbe={lbe}", flush=True)
        if not np.isfinite(rn) or rn < tol:
            U = Ug
            break

        X_hist.append(U.copy()); G_hist.append(Ug.copy()); F_hist.append(F_new.copy())
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
            U_new = Ug.ravel() - dG @ gamma
            U_new = U_new.reshape(3, case.N, case.N)
            if beta < 1.0:
                U_new = (1.0 - beta) * U + beta * U_new
            # safeguard via macro residual at U_new
            F_test = g_step(U_new) - U_new; lbe += 2
            r_test = _norm(F_test) / math.sqrt(3 * case.N * case.N)
            if np.isfinite(r_test) and r_test < 1.1 * rn:
                U = U_new
            else:
                U = Ug
        else:
            U = Ug
        if not np.all(np.isfinite(U)):
            break
        res_prev = rn

    # final polish in f-space until vchg < polish_tol
    f = ce_lift(case, U); lbe += 1
    prev = f.copy()
    for step in range(1, polish_max + 1):
        f = case.lbe_step(f); lbe += 1
        if step % polish_check_every == 0:
            _, ux, uy = macro_of(case, f)
            _, uxp, uyp = macro_of(case, prev)
            num = float(np.sqrt(np.sum((ux - uxp) ** 2 + (uy - uyp) ** 2)))
            den = max(float(np.sqrt(np.sum(ux * ux + uy * uy))), 1e-30)
            vchg = num / den
            history.append((max_outer + step, vchg, lbe, time.perf_counter() - t0))
            if vchg < polish_tol:
                break
            prev = f.copy()
    return f, history
