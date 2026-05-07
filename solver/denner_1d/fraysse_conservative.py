# solver/denner_1d/fraysse_conservative.py
# Fraysse Conservative solver: Q = {ρY₁, ρY₂, ρu, ρE}
# Newton on conserved variables, HLLC/Rusanov flux, autograd Jacobian.

import numpy as np
from .fraysse_common import (
    anp, _AD, ad_jacobian,
    _get_ph_params, pack_fraysse, unpack_fraysse,
    mixture_eos_anp, hllc_flux_anp, rusanov_flux_anp, _ghost_anp,
)


def make_residual_fraysse(Q_n, N, dx, dt, ph1, ph2, bc_l, bc_r, flux_type='hllc'):
    """Build autograd-differentiable residual closure for one timestep.

    R(Q) = (Q - Q_n)/dt + (F_{i+1/2} - F_{i-1/2})/dx = 0
    """
    _Q_n = anp.array(Q_n, dtype=float)

    def residual(Q):
        rY1, rY2, ru, rE = unpack_fraysse(Q, N)
        p, T, u, c = mixture_eos_anp(rY1, rY2, ru, rE, ph1, ph2)

        vars_ext = [_ghost_anp(v, bc_l, bc_r) for v in [rY1, rY2, ru, rE, p, c]]
        L = [v[0:N+1] for v in vars_ext]
        R = [v[1:N+2] for v in vars_ext]

        if flux_type == 'hllc':
            Ff = hllc_flux_anp(L[0], L[1], L[2], L[3], L[4], L[5],
                               R[0], R[1], R[2], R[3], R[4], R[5])
        else:
            Ff = rusanov_flux_anp(L[0], L[1], L[2], L[3], L[4], L[5],
                                  R[0], R[1], R[2], R[3], R[4], R[5])

        Q_n_split = unpack_fraysse(_Q_n, N)

        res = []
        for k in range(4):
            Qk = [rY1, rY2, ru, rE][k]
            Qk_n = Q_n_split[k]
            res.append((Qk - Qk_n) / dt + (Ff[k][1:N+1] - Ff[k][0:N]) / dx)

        return anp.concatenate(res)

    return residual


def newton_fraysse(Q_n, N, dx, dt, ph1, ph2, bc_l, bc_r,
                   max_newton=20, tol=1e-10, verbose=False, flux_type='hllc'):
    """Single Newton loop with row-column equilibration."""
    if not _AD:
        raise ImportError("autograd required for newton_fraysse")

    res_func = make_residual_fraysse(Q_n, N, dx, dt, ph1, ph2, bc_l, bc_r,
                                     flux_type=flux_type)
    jac_func = ad_jacobian(res_func)

    Q_k = Q_n.copy().astype(float)

    for niter in range(max_newton):
        R = np.array(res_func(Q_k), dtype=float)
        R_norm = np.linalg.norm(R)

        if verbose:
            print(f"    Newton {niter:2d}: |R| = {R_norm:.3e}")

        if R_norm < tol:
            return Q_k, {
                'converged': True,
                'newton_iters': niter,
                'final_residual': float(R_norm),
            }

        J = np.array(jac_func(Q_k), dtype=float)

        D_row = 1.0 / (np.max(np.abs(J), axis=1) + 1e-300)
        Q_scale = np.maximum(np.abs(Q_k), 1.0)
        J_eq = np.diag(D_row) @ J @ np.diag(Q_scale)

        try:
            dQ_eq = np.linalg.solve(J_eq, -D_row * R)
            dQ = Q_scale * dQ_eq
        except np.linalg.LinAlgError:
            return Q_k, {
                'converged': False,
                'newton_iters': niter,
                'final_residual': float(R_norm),
            }

        omega = 1.0
        for _ls in range(12):
            Q_trial = Q_k + omega * dQ
            Q_trial[0:N]   = np.maximum(Q_trial[0:N], 1e-10)
            Q_trial[N:2*N] = np.maximum(Q_trial[N:2*N], 1e-10)
            try:
                R_trial = np.array(res_func(Q_trial), dtype=float)
                if np.linalg.norm(R_trial) < R_norm:
                    break
            except Exception:
                pass
            omega *= 0.5

        Q_k = Q_k + omega * dQ
        Q_k[0:N]   = np.maximum(Q_k[0:N], 1e-10)
        Q_k[N:2*N] = np.maximum(Q_k[N:2*N], 1e-10)

    R_final = float(np.linalg.norm(np.array(res_func(Q_k), dtype=float)))
    return Q_k, {
        'converged': R_final < tol,
        'newton_iters': max_newton,
        'final_residual': R_final,
    }


def step_fraysse(N, dx, Q_prev, ph1, ph2, bc_l, bc_r, cfg):
    """Advance solution by one timestep using the Fraysse Conservative solver."""
    CFL = float(cfg.get('CFL', 0.5))

    rY1, rY2, ru, rE = unpack_fraysse(Q_prev, N)
    p_k, T_k, u_k, c_k = mixture_eos_anp(rY1, rY2, ru, rE, ph1, ph2)
    u_k = np.array(u_k, dtype=float)
    c_k = np.array(c_k, dtype=float)

    max_speed = float(np.max(np.abs(u_k) + c_k))
    if max_speed < 1e-300:
        max_speed = 1e-300

    if 'dt_fixed' in cfg:
        dt = float(cfg['dt_fixed'])
    else:
        dt = CFL * dx / max_speed

    flux_type = cfg.get('flux_type', 'hllc')

    Q_new, info = newton_fraysse(
        Q_prev, N, dx, dt, ph1, ph2, bc_l, bc_r,
        max_newton=int(cfg.get('max_newton', 20)),
        tol=float(cfg.get('newton_tol', 1e-10)),
        verbose=bool(cfg.get('verbose', False)),
        flux_type=flux_type,
    )

    return Q_new, dt, info
