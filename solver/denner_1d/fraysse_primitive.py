# solver/denner_1d/fraysse_primitive.py
# Fraysse Primitive solver: Q_prim = {p, u, T, Y₁}
# Newton on primitive variables, GMRES+ILU, correction-based convergence.
# Optional THINC donor-side reconstruction for sharp interface.

import numpy as np
from .fraysse_common import (
    anp, _AD, ad_jacobian,
    _get_ph_params, mixture_eos_anp,
    hllc_flux_anp, rusanov_flux_anp,
    _ghost_anp, _ghost_anp2, _thinc_face_anp,
)


# ---------------------------------------------------------------------------
# EOS: primitive → conservative  (autograd-compatible)
# ---------------------------------------------------------------------------

def primitive_to_conservative_anp(p, u, T, Y1, ph1, ph2):
    """Convert (p, u, T, Y1) to conservative variables."""
    g1, pinf1, b1, kv1, eta1 = _get_ph_params(ph1)
    g2, pinf2, b2, kv2, eta2 = _get_ph_params(ph2)

    _eps = 1e-300
    Y2 = 1.0 - Y1

    rho1 = (p + pinf1) / (kv1 * (g1 - 1.0) * T + b1 * (p + pinf1) + _eps)
    rho2 = (p + pinf2) / (kv2 * (g2 - 1.0) * T + b2 * (p + pinf2) + _eps)

    rho = 1.0 / (Y1 / (rho1 + _eps) + Y2 / (rho2 + _eps) + _eps)

    e1 = kv1 * T * (p + g1 * pinf1) / (p + pinf1 + _eps) + eta1
    e2 = kv2 * T * (p + g2 * pinf2) / (p + pinf2 + _eps) + eta2
    e_mix = Y1 * e1 + Y2 * e2

    rhoY1 = rho * Y1
    rhoY2 = rho * Y2
    rhou  = rho * u
    rhoE  = rho * (e_mix + 0.5 * u * u)

    alpha1 = Y1 * rho / (rho1 + _eps)
    alpha1 = anp.maximum(alpha1, 0.0)
    alpha1 = anp.minimum(alpha1, 1.0)
    alpha2 = 1.0 - alpha1

    c1_sq = g1 * (p + pinf1) / (rho1 * (1.0 - b1 * rho1) + _eps)
    c2_sq = g2 * (p + pinf2) / (rho2 + _eps)
    c1_sq = anp.maximum(c1_sq, _eps)
    c2_sq = anp.maximum(c2_sq, _eps)

    inv_rho_c2 = (alpha1 / (rho1 * c1_sq + _eps)
                + alpha2 / (rho2 * c2_sq + _eps))
    c_wood = anp.sqrt(1.0 / (rho * inv_rho_c2 + _eps))

    return rho, rhou, rhoE, rhoY1, rhoY2, c_wood


# ---------------------------------------------------------------------------
# Residual factory  (primitive variable input)
# ---------------------------------------------------------------------------

def make_residual_primitive(p_n, u_n, T_n, Y1_n, N, dx, dt, ph1, ph2,
                             bc_l, bc_r, flux_type='hllc', use_cicsam=False,
                             _cicsam_corr=None, use_thinc=False, thinc_beta=2.5):
    """Build autograd-differentiable residual closure for primitive variable Newton."""
    _rho_n, _rhou_n, _rhoE_n, _rhoY1_n, _rhoY2_n, _ = primitive_to_conservative_anp(
        anp.array(p_n,  dtype=float),
        anp.array(u_n,  dtype=float),
        anp.array(T_n,  dtype=float),
        anp.array(Y1_n, dtype=float),
        ph1, ph2,
    )

    def residual(Q_prim):
        p  = Q_prim[0:N]
        u  = Q_prim[N:2*N]
        T  = Q_prim[2*N:3*N]
        Y1 = Q_prim[3*N:4*N]

        rho, rhou, rhoE, rhoY1, rhoY2, c = primitive_to_conservative_anp(
            p, u, T, Y1, ph1, ph2)

        vars_all = [rhoY1, rhoY2, rhou, rhoE, p, c]
        vars_ext = [_ghost_anp(v, bc_l, bc_r) for v in vars_all]
        rY1_e, rY2_e, ru_e, rE_e, p_e, c_e = vars_ext

        L = [v[0:N+1] for v in vars_ext]
        R = [v[1:N+2] for v in vars_ext]

        if flux_type == 'hllc':
            Ff = hllc_flux_anp(L[0], L[1], L[2], L[3], L[4], L[5],
                               R[0], R[1], R[2], R[3], R[4], R[5])
        else:
            Ff = rusanov_flux_anp(L[0], L[1], L[2], L[3], L[4], L[5],
                                  R[0], R[1], R[2], R[3], R[4], R[5])

        # CICSAM deferred correction
        if use_cicsam and _cicsam_corr is not None:
            m_face = Ff[0] + Ff[1]
            Y1_ext1 = _ghost_anp(Y1, bc_l, bc_r)
            u_ext1  = _ghost_anp(u, bc_l, bc_r)
            u_face  = 0.5 * (u_ext1[0:N+1] + u_ext1[1:N+2])
            Y1_upwind = anp.where(u_face >= 0,
                                   Y1_ext1[0:N+1], Y1_ext1[1:N+2])
            Y1_face_total = Y1_upwind + anp.array(_cicsam_corr)
            Y1_face_total = anp.maximum(anp.minimum(Y1_face_total, 1.0), 0.0)
            Ff_Y1 = Y1_face_total * m_face
            Ff_Y2 = (1.0 - Y1_face_total) * m_face
            Ff = (Ff_Y1, Ff_Y2, Ff[2], Ff[3])

        # THINC donor-side reconstruction
        if use_thinc:
            _eps = 1e-300
            Y1_ext2 = _ghost_anp2(Y1, bc_l, bc_r)
            u_ext1  = _ghost_anp(u, bc_l, bc_r)
            u_face  = 0.5 * (u_ext1[0:N+1] + u_ext1[1:N+2])
            Y1_thinc = _thinc_face_anp(Y1_ext2, u_face, beta=thinc_beta)

            p_ext = _ghost_anp(p, bc_l, bc_r)
            T_ext = _ghost_anp(T, bc_l, bc_r)
            p_don = anp.where(u_face >= 0, p_ext[0:N+1], p_ext[1:N+2])
            u_don = anp.where(u_face >= 0, u_ext1[0:N+1], u_ext1[1:N+2])
            T_don = anp.where(u_face >= 0, T_ext[0:N+1], T_ext[1:N+2])

            rho_d, ru_d, rE_d, rY1_d, rY2_d, c_d = \
                primitive_to_conservative_anp(p_don, u_don, T_don, Y1_thinc,
                                              ph1, ph2)

            is_right = (u_face >= 0)
            new_L = [anp.where(is_right, rY1_d, L[0]),
                     anp.where(is_right, rY2_d, L[1]),
                     anp.where(is_right, ru_d,  L[2]),
                     anp.where(is_right, rE_d,  L[3]),
                     anp.where(is_right, p_don, L[4]),
                     anp.where(is_right, c_d,   L[5])]
            new_R = [anp.where(is_right, R[0], rY1_d),
                     anp.where(is_right, R[1], rY2_d),
                     anp.where(is_right, R[2], ru_d),
                     anp.where(is_right, R[3], rE_d),
                     anp.where(is_right, R[4], p_don),
                     anp.where(is_right, R[5], c_d)]

            if flux_type == 'hllc':
                Ff = hllc_flux_anp(new_L[0], new_L[1], new_L[2], new_L[3],
                                   new_L[4], new_L[5],
                                   new_R[0], new_R[1], new_R[2], new_R[3],
                                   new_R[4], new_R[5])
            else:
                Ff = rusanov_flux_anp(new_L[0], new_L[1], new_L[2], new_L[3],
                                      new_L[4], new_L[5],
                                      new_R[0], new_R[1], new_R[2], new_R[3],
                                      new_R[4], new_R[5])

        Q_now_list  = [rhoY1, rhoY2, rhou, rhoE]
        Q_n_list    = [_rhoY1_n, _rhoY2_n, _rhou_n, _rhoE_n]

        res = []
        for k in range(4):
            res.append(
                (Q_now_list[k] - Q_n_list[k]) / dt
                + (Ff[k][1:N+1] - Ff[k][0:N]) / dx
            )

        return anp.concatenate(res)

    return residual


# ---------------------------------------------------------------------------
# CICSAM correction helper
# ---------------------------------------------------------------------------

def _compute_cicsam_correction(Y1, u, N, dx, dt, bc_l, bc_r):
    """Compute frozen CICSAM correction: (Y1_cicsam - Y1_upwind) at faces."""
    from .boundary import apply_ghost, apply_ghost_velocity

    u_ext = apply_ghost_velocity(np.asarray(u, dtype=float), bc_l, bc_r, n_ghost=2)
    u_face = np.array([0.5 * (u_ext[2 + f - 1] + u_ext[2 + f]) for f in range(N + 1)])

    Y1_np = np.asarray(Y1, dtype=float)
    Y1_ext1 = apply_ghost(Y1_np, bc_l, bc_r, n_ghost=1)
    Y1_upwind = np.where(u_face >= 0, Y1_ext1[0:N+1], Y1_ext1[1:N+2])

    from .interface.cicsam import cicsam_face
    Y1_ext2 = apply_ghost(Y1_np, bc_l, bc_r, n_ghost=2)
    Y1_cicsam = cicsam_face(Y1_ext2, u_face, dt, dx, n_ghost=2)

    return Y1_cicsam - Y1_upwind


# ---------------------------------------------------------------------------
# Newton solver  (primitive variable)
# ---------------------------------------------------------------------------

def newton_primitive(p_n, u_n, T_n, Y1_n, N, dx, dt, ph1, ph2, bc_l, bc_r,
                     max_newton=20, tol=1e-6, verbose=False, flux_type='hllc',
                     use_cicsam=False, use_thinc=False, thinc_beta=2.5):
    """Primitive variable Newton solver with GMRES+ILU.

    Convergence criterion: max relative correction of primitive variables.
    """
    if not _AD:
        raise ImportError("autograd required for newton_primitive")

    from scipy.sparse import csc_matrix
    from scipy.sparse.linalg import spilu, gmres, LinearOperator

    Q_k = np.concatenate([
        np.array(p_n,  dtype=float),
        np.array(u_n,  dtype=float),
        np.array(T_n,  dtype=float),
        np.array(Y1_n, dtype=float),
    ])

    p_ref = max(float(np.mean(np.abs(p_n))), 1.0)
    u_ref = max(float(np.max(np.abs(u_n))), 1.0)
    T_ref = max(float(np.mean(np.abs(T_n))), 1.0)

    max_picard = 5 if use_cicsam else 1
    total_newton = 0

    for picard in range(max_picard):
        if use_cicsam:
            cicsam_corr = _compute_cicsam_correction(
                Q_k[3*N:4*N], Q_k[N:2*N], N, dx, dt, bc_l, bc_r)
        else:
            cicsam_corr = None

        res_func = make_residual_primitive(
            p_n, u_n, T_n, Y1_n, N, dx, dt, ph1, ph2, bc_l, bc_r,
            flux_type=flux_type, use_cicsam=use_cicsam,
            _cicsam_corr=cicsam_corr,
            use_thinc=use_thinc, thinc_beta=thinc_beta,
        )
        jac_func = ad_jacobian(res_func)

        Q_k_start = Q_k.copy()
        converged_inner = False

        for niter in range(max_newton):
            R = np.array(res_func(Q_k), dtype=float)
            R_norm = np.linalg.norm(R)

            if verbose:
                pic_str = f"[P{picard}]" if use_cicsam else ""
                print(f"    Newton{pic_str} {niter:2d}: |R| = {R_norm:.3e}")

            J = np.array(jac_func(Q_k), dtype=float)

            D_row   = 1.0 / (np.max(np.abs(J), axis=1) + 1e-300)
            Q_scale = np.maximum(np.abs(Q_k), 1.0)
            J_eq    = np.diag(D_row) @ J @ np.diag(Q_scale)
            b_eq    = -D_row * R

            try:
                J_sp = csc_matrix(J_eq)
                ilu  = spilu(J_sp, fill_factor=10)
                M    = LinearOperator(J_sp.shape, matvec=ilu.solve)
                dQ_eq, info_gmres = gmres(J_sp, b_eq, M=M, atol=1e-12, maxiter=200)
                if info_gmres != 0:
                    raise RuntimeError("GMRES did not converge")
            except Exception:
                try:
                    dQ_eq = np.linalg.solve(J_eq, b_eq)
                except np.linalg.LinAlgError:
                    return Q_k, {
                        'converged': False, 'newton_iters': total_newton + niter,
                        'final_residual': float(R_norm),
                    }

            dQ = Q_scale * dQ_eq

            omega = 1.0
            for _ls in range(12):
                Q_trial = Q_k + omega * dQ
                Q_trial[0:N]       = np.maximum(Q_trial[0:N],   1.0)
                Q_trial[2*N:3*N]   = np.maximum(Q_trial[2*N:3*N], 1.0)
                Q_trial[3*N:4*N]   = np.clip(Q_trial[3*N:4*N], 0.0, 1.0)
                try:
                    R_trial = np.array(res_func(Q_trial), dtype=float)
                    if np.linalg.norm(R_trial) < R_norm:
                        break
                except Exception:
                    pass
                omega *= 0.5

            Q_k = Q_k + omega * dQ
            Q_k[0:N]       = np.maximum(Q_k[0:N],   1.0)
            Q_k[2*N:3*N]   = np.maximum(Q_k[2*N:3*N], 1.0)
            Q_k[3*N:4*N]   = np.clip(Q_k[3*N:4*N], 0.0, 1.0)

            dQ_actual = omega * dQ
            corr_p  = float(np.max(np.abs(dQ_actual[0:N])))     / p_ref
            corr_u  = float(np.max(np.abs(dQ_actual[N:2*N])))   / u_ref
            corr_T  = float(np.max(np.abs(dQ_actual[2*N:3*N]))) / T_ref
            corr_Y1 = float(np.max(np.abs(dQ_actual[3*N:4*N])))
            corr_max = max(corr_p, corr_u, corr_T, corr_Y1)

            if verbose:
                print(f"      ω={omega:.3f} corr={corr_max:.2e}")

            if niter >= 2 and corr_max < tol:
                converged_inner = True
                total_newton += niter + 1
                break

        if not converged_inner:
            total_newton += max_newton

        if not use_cicsam or max_picard <= 1:
            break
        picard_change = float(np.max(np.abs(Q_k - Q_k_start)))
        if verbose:
            print(f"  Picard {picard}: change={picard_change:.3e}")
        if picard_change < tol * p_ref and picard >= 1:
            break

    R_final = float(np.linalg.norm(np.array(res_func(Q_k), dtype=float)))
    return Q_k, {
        'converged': converged_inner,
        'newton_iters': total_newton,
        'final_residual': R_final,
    }


# ---------------------------------------------------------------------------
# Step function  (primitive variable)
# ---------------------------------------------------------------------------

def step_fraysse_primitive(N, dx, p_prev, u_prev, T_prev, Y1_prev,
                           ph1, ph2, bc_l, bc_r, cfg):
    """Advance one timestep with the primitive variable Newton solver."""
    CFL = float(cfg.get('CFL', 0.5))

    _, _, _, _, _, c_k = primitive_to_conservative_anp(
        anp.array(p_prev,  dtype=float),
        anp.array(u_prev,  dtype=float),
        anp.array(T_prev,  dtype=float),
        anp.array(Y1_prev, dtype=float),
        ph1, ph2,
    )
    u_k = np.array(u_prev, dtype=float)
    c_k = np.array(c_k,    dtype=float)

    max_speed = float(np.max(np.abs(u_k) + c_k))
    if max_speed < 1e-300:
        max_speed = 1e-300

    if 'dt_fixed' in cfg:
        dt = float(cfg['dt_fixed'])
    else:
        dt = CFL * dx / max_speed

    flux_type  = cfg.get('flux_type', 'hllc')
    use_cicsam = bool(cfg.get('use_cicsam', False))
    use_thinc  = bool(cfg.get('use_thinc', False))
    thinc_beta = float(cfg.get('thinc_beta', 2.5))

    Q_prim_new, info = newton_primitive(
        p_prev, u_prev, T_prev, Y1_prev,
        N, dx, dt, ph1, ph2, bc_l, bc_r,
        max_newton=int(cfg.get('max_newton', 20)),
        tol=float(cfg.get('newton_tol', 1e-6)),
        verbose=bool(cfg.get('verbose', False)),
        flux_type=flux_type,
        use_cicsam=use_cicsam,
        use_thinc=use_thinc,
        thinc_beta=thinc_beta,
    )

    p_new  = Q_prim_new[0:N]
    u_new  = Q_prim_new[N:2*N]
    T_new  = Q_prim_new[2*N:3*N]
    Y1_new = Q_prim_new[3*N:4*N]

    return (p_new, u_new, T_new, Y1_new), dt, info
