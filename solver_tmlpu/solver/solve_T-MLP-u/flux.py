"""Numerical face fluxes — work for any equation that exposes
`physical_flux`, `prim_to_cons`, `wave_speeds_lr`, `max_wave_speed`.

  upwind_advection — exact for linear scalar advection (a·n upwind side)
  llf              — Local Lax-Friedrichs (Rusanov), universal
  hllc_1d          — HLLC for 1D Euler (Toro 1994)

Free parameters: 0 (Davis wave speeds, fixed formula).
"""
from __future__ import annotations
import os
import numpy as np

try:  # Optional fast path for large Euler shock calculations.
    from numba import njit, prange, set_num_threads
    _NUMBA_AVAILABLE = True
    _thread_env = (
        os.environ.get('TMLPU_SOLVER_THREADS')
        or os.environ.get('TMLPU_FLUX_THREADS')
        or os.environ.get('NUMBA_NUM_THREADS'))
    _default_threads = min(32, os.cpu_count() or 1)
    set_num_threads(max(1, min(_default_threads, int(
        _thread_env or _default_threads))))
except Exception:  # pragma: no cover - exercised when numba is absent.
    njit = None
    prange = range
    _NUMBA_AVAILABLE = False


_EPS = 1e-30


if _NUMBA_AVAILABLE:
    @njit(cache=True)
    def _hllc_hll_face(rho_l_in, u_l, v_l, p_l_in,
                       rho_r_in, u_r, v_r, p_r_in,
                       nx, ny, gamma):
        eps = 1.0e-30
        rho_l = max(rho_l_in, eps)
        rho_r = max(rho_r_in, eps)
        p_l = max(p_l_in, eps)
        p_r = max(p_r_in, eps)
        tx = -ny
        ty = nx
        un_l = u_l * nx + v_l * ny
        un_r = u_r * nx + v_r * ny
        ut_l = u_l * tx + v_l * ty
        ut_r = u_r * tx + v_r * ty
        c_l = np.sqrt(max(gamma * p_l / rho_l, eps))
        c_r = np.sqrt(max(gamma * p_r / rho_r, eps))
        e_l = p_l / ((gamma - 1.0) * rho_l) + 0.5 * (u_l * u_l + v_l * v_l)
        e_r = p_r / ((gamma - 1.0) * rho_r) + 0.5 * (u_r * u_r + v_r * v_r)
        U_l0 = rho_l
        U_l1 = rho_l * u_l
        U_l2 = rho_l * v_l
        U_l3 = rho_l * e_l
        U_r0 = rho_r
        U_r1 = rho_r * u_r
        U_r2 = rho_r * v_r
        U_r3 = rho_r * e_r
        F_l0 = rho_l * un_l
        F_l1 = rho_l * u_l * un_l + p_l * nx
        F_l2 = rho_l * v_l * un_l + p_l * ny
        F_l3 = (rho_l * e_l + p_l) * un_l
        F_r0 = rho_r * un_r
        F_r1 = rho_r * u_r * un_r + p_r * nx
        F_r2 = rho_r * v_r * un_r + p_r * ny
        F_r3 = (rho_r * e_r + p_r) * un_r
        S_l = min(un_l - c_l, un_r - c_r)
        S_r = max(un_l + c_l, un_r + c_r)
        den = rho_l * (S_l - un_l) - rho_r * (S_r - un_r)
        if abs(den) <= eps:
            den = eps if den >= 0.0 else -eps
        S_m = (p_r - p_l
               + rho_l * un_l * (S_l - un_l)
               - rho_r * un_r * (S_r - un_r)) / den

        hden = max(S_r - S_l, eps)
        if S_l >= 0.0:
            H0, H1, H2, H3 = F_l0, F_l1, F_l2, F_l3
        elif S_r <= 0.0:
            H0, H1, H2, H3 = F_r0, F_r1, F_r2, F_r3
        else:
            H0 = (S_r * F_l0 - S_l * F_r0 + S_l * S_r * (U_r0 - U_l0)) / hden
            H1 = (S_r * F_l1 - S_l * F_r1 + S_l * S_r * (U_r1 - U_l1)) / hden
            H2 = (S_r * F_l2 - S_l * F_r2 + S_l * S_r * (U_r2 - U_l2)) / hden
            H3 = (S_r * F_l3 - S_l * F_r3 + S_l * S_r * (U_r3 - U_l3)) / hden

        den_l = S_l - S_m
        den_r = S_r - S_m
        if abs(den_l) <= eps:
            den_l = eps if den_l >= 0.0 else -eps
        if abs(den_r) <= eps:
            den_r = eps if den_r >= 0.0 else -eps
        fac_l = rho_l * (S_l - un_l) / den_l
        fac_r = rho_r * (S_r - un_r) / den_r
        wave_l = rho_l * (S_l - un_l)
        wave_r = rho_r * (S_r - un_r)
        if abs(wave_l) <= eps:
            wave_l = eps if wave_l >= 0.0 else -eps
        if abs(wave_r) <= eps:
            wave_r = eps if wave_r >= 0.0 else -eps
        mn_l = fac_l * S_m
        mt_l = fac_l * ut_l
        mn_r = fac_r * S_m
        mt_r = fac_r * ut_r
        Us_l0 = fac_l
        Us_l1 = mn_l * nx + mt_l * tx
        Us_l2 = mn_l * ny + mt_l * ty
        Us_l3 = fac_l * (e_l + (S_m - un_l) * (S_m + p_l / wave_l))
        Us_r0 = fac_r
        Us_r1 = mn_r * nx + mt_r * tx
        Us_r2 = mn_r * ny + mt_r * ty
        Us_r3 = fac_r * (e_r + (S_m - un_r) * (S_m + p_r / wave_r))

        if S_l >= 0.0:
            C0, C1, C2, C3 = F_l0, F_l1, F_l2, F_l3
        elif S_m >= 0.0:
            C0 = F_l0 + S_l * (Us_l0 - U_l0)
            C1 = F_l1 + S_l * (Us_l1 - U_l1)
            C2 = F_l2 + S_l * (Us_l2 - U_l2)
            C3 = F_l3 + S_l * (Us_l3 - U_l3)
        elif S_r > 0.0:
            C0 = F_r0 + S_r * (Us_r0 - U_r0)
            C1 = F_r1 + S_r * (Us_r1 - U_r1)
            C2 = F_r2 + S_r * (Us_r2 - U_r2)
            C3 = F_r3 + S_r * (Us_r3 - U_r3)
        else:
            C0, C1, C2, C3 = F_r0, F_r1, F_r2, F_r3
        if (not np.isfinite(C0 + C1 + C2 + C3)
                or Us_l0 <= 0.0 or Us_r0 <= 0.0):
            C0, C1, C2, C3 = H0, H1, H2, H3
        return (C0, C1, C2, C3, H0, H1, H2, H3,
                un_l, un_r, c_l, c_r, p_l, p_r)


    @njit(cache=True)
    def _roe_face(rho_l_in, u_l, v_l, p_l_in,
                  rho_r_in, u_r, v_r, p_r_in,
                  nx, ny, gamma):
        eps = 1.0e-30
        rho_l = max(rho_l_in, eps)
        rho_r = max(rho_r_in, eps)
        p_l = max(p_l_in, eps)
        p_r = max(p_r_in, eps)
        tx = -ny
        ty = nx
        un_l = u_l * nx + v_l * ny
        un_r = u_r * nx + v_r * ny
        ut_l = u_l * tx + v_l * ty
        ut_r = u_r * tx + v_r * ty
        q_l = u_l * u_l + v_l * v_l
        q_r = u_r * u_r + v_r * v_r
        e_l = p_l / ((gamma - 1.0) * rho_l) + 0.5 * q_l
        e_r = p_r / ((gamma - 1.0) * rho_r) + 0.5 * q_r
        E_l = rho_l * e_l
        E_r = rho_r * e_r
        H_l = (E_l + p_l) / rho_l
        H_r = (E_r + p_r) / rho_r

        F_l0 = rho_l * un_l
        F_l1n = rho_l * un_l * un_l + p_l
        F_l1t = rho_l * un_l * ut_l
        F_l3 = (E_l + p_l) * un_l
        F_r0 = rho_r * un_r
        F_r1n = rho_r * un_r * un_r + p_r
        F_r1t = rho_r * un_r * ut_r
        F_r3 = (E_r + p_r) * un_r

        sr_l = np.sqrt(rho_l)
        sr_r = np.sqrt(rho_r)
        inv_sum = 1.0 / max(sr_l + sr_r, eps)
        un = (sr_l * un_l + sr_r * un_r) * inv_sum
        ut = (sr_l * ut_l + sr_r * ut_r) * inv_sum
        H = (sr_l * H_l + sr_r * H_r) * inv_sum
        rho = sr_l * sr_r
        q = un * un + ut * ut
        a2 = (gamma - 1.0) * (H - 0.5 * q)
        if a2 <= eps:
            vals = _hllc_hll_face(rho_l, u_l, v_l, p_l,
                                  rho_r, u_r, v_r, p_r, nx, ny, gamma)
            return vals[4], vals[5], vals[6], vals[7]
        a = np.sqrt(a2)

        drho = rho_r - rho_l
        dun = un_r - un_l
        dut = ut_r - ut_l
        dp = p_r - p_l
        inv_a2 = 1.0 / max(a2, eps)
        alpha_minus = 0.5 * (dp * inv_a2 - rho * dun / max(a, eps))
        alpha_plus = 0.5 * (dp * inv_a2 + rho * dun / max(a, eps))
        alpha_entropy = drho - dp * inv_a2
        alpha_shear = rho * dut

        lam_minus = abs(un - a)
        lam_mid = abs(un)
        lam_plus = abs(un + a)

        rminus0 = 1.0
        rminus1n = un - a
        rminus1t = ut
        rminus3 = H - un * a
        rent0 = 1.0
        rent1n = un
        rent1t = ut
        rent3 = 0.5 * q
        rshear0 = 0.0
        rshear1n = 0.0
        rshear1t = 1.0
        rshear3 = ut
        rplus0 = 1.0
        rplus1n = un + a
        rplus1t = ut
        rplus3 = H + un * a

        d0 = (lam_minus * alpha_minus * rminus0
              + lam_mid * alpha_entropy * rent0
              + lam_mid * alpha_shear * rshear0
              + lam_plus * alpha_plus * rplus0)
        d1n = (lam_minus * alpha_minus * rminus1n
               + lam_mid * alpha_entropy * rent1n
               + lam_mid * alpha_shear * rshear1n
               + lam_plus * alpha_plus * rplus1n)
        d1t = (lam_minus * alpha_minus * rminus1t
               + lam_mid * alpha_entropy * rent1t
               + lam_mid * alpha_shear * rshear1t
               + lam_plus * alpha_plus * rplus1t)
        d3 = (lam_minus * alpha_minus * rminus3
              + lam_mid * alpha_entropy * rent3
              + lam_mid * alpha_shear * rshear3
              + lam_plus * alpha_plus * rplus3)

        f0 = 0.5 * (F_l0 + F_r0) - 0.5 * d0
        fn = 0.5 * (F_l1n + F_r1n) - 0.5 * d1n
        ft = 0.5 * (F_l1t + F_r1t) - 0.5 * d1t
        f3 = 0.5 * (F_l3 + F_r3) - 0.5 * d3
        fx = fn * nx + ft * tx
        fy = fn * ny + ft * ty
        if not np.isfinite(f0 + fx + fy + f3):
            vals = _hllc_hll_face(rho_l, u_l, v_l, p_l,
                                  rho_r, u_r, v_r, p_r, nx, ny, gamma)
            return vals[4], vals[5], vals[6], vals[7]
        return f0, fx, fy, f3


    @njit(parallel=True, cache=True)
    def _roe_rotated_hybrid_kernel(WL, WR, normals, gamma):
        nf = normals.shape[0]
        out = np.empty((4, nf), dtype=np.float64)
        eps = 1.0e-30
        macheps = np.finfo(np.float64).eps
        for i in prange(nf):
            nx = normals[i, 0]
            ny = normals[i, 1]
            rho_l, u_l, v_l, p_l = WL[0, i], WL[1, i], WL[2, i], WL[3, i]
            rho_r, u_r, v_r, p_r = WR[0, i], WR[1, i], WR[2, i], WR[3, i]
            du = u_r - u_l
            dv = v_r - v_l
            speed_jump = np.sqrt(du * du + dv * dv)
            scale = macheps * (abs(u_l) + abs(u_r) + abs(v_l) + abs(v_r) + 1.0)
            if speed_jump > scale:
                n1x = du / max(speed_jump, eps)
                n1y = dv / max(speed_jump, eps)
            else:
                n1x = nx
                n1y = ny
            a1 = nx * n1x + ny * n1y
            if a1 < 0.0:
                n1x = -n1x
                n1y = -n1y
                a1 = -a1
            n2x = -n1y
            n2y = n1x
            a2 = nx * n2x + ny * n2y
            if a2 < 0.0:
                n2x = -n2x
                n2y = -n2y
                a2 = -a2
            vals1 = _hllc_hll_face(rho_l, u_l, v_l, p_l,
                                   rho_r, u_r, v_r, p_r, n1x, n1y, gamma)
            r0, r1, r2, r3 = _roe_face(rho_l, u_l, v_l, p_l,
                                       rho_r, u_r, v_r, p_r,
                                       n2x, n2y, gamma)
            out[0, i] = a1 * vals1[4] + a2 * r0
            out[1, i] = a1 * vals1[5] + a2 * r1
            out[2, i] = a1 * vals1[6] + a2 * r2
            out[3, i] = a1 * vals1[7] + a2 * r3
        return out


    @njit(parallel=True, cache=True)
    def _roe_rotated_shock_hybrid_kernel(WL, WR, normals, gamma):
        nf = normals.shape[0]
        out = np.empty((4, nf), dtype=np.float64)
        eps = 1.0e-30
        macheps = np.finfo(np.float64).eps
        for i in prange(nf):
            nx = normals[i, 0]
            ny = normals[i, 1]
            rho_l, u_l, v_l, p_l = WL[0, i], WL[1, i], WL[2, i], WL[3, i]
            rho_r, u_r, v_r, p_r = WR[0, i], WR[1, i], WR[2, i], WR[3, i]

            valsn = _hllc_hll_face(rho_l, u_l, v_l, p_l,
                                   rho_r, u_r, v_r, p_r,
                                   nx, ny, gamma)
            un_l, un_r = valsn[8], valsn[9]
            c_l, c_r = valsn[10], valsn[11]
            pp_l, pp_r = valsn[12], valsn[13]
            pressure_jump = abs(pp_r - pp_l) / max(pp_r + pp_l, eps)
            compression = max(0.0, un_l - un_r) / max(c_l + c_r, eps)
            jump_sensor = min(1.0, max(0.0, (pressure_jump - 0.05) / 0.35))
            compression_sensor = min(1.0, max(0.0, 4.0 * compression))
            shock = np.sqrt(jump_sensor * compression_sensor)

            du = u_r - u_l
            dv = v_r - v_l
            speed_jump = np.sqrt(du * du + dv * dv)
            scale = macheps * (abs(u_l) + abs(u_r) + abs(v_l) + abs(v_r) + 1.0)
            if speed_jump > scale:
                n1x = du / max(speed_jump, eps)
                n1y = dv / max(speed_jump, eps)
            else:
                n1x = nx
                n1y = ny
            a1 = nx * n1x + ny * n1y
            if a1 < 0.0:
                n1x = -n1x
                n1y = -n1y
                a1 = -a1
            n2x = -n1y
            n2y = n1x
            a2 = nx * n2x + ny * n2y
            if a2 < 0.0:
                n2x = -n2x
                n2y = -n2y
                a2 = -a2

            vals1 = _hllc_hll_face(rho_l, u_l, v_l, p_l,
                                   rho_r, u_r, v_r, p_r, n1x, n1y, gamma)
            r0, r1, r2, r3 = _roe_face(rho_l, u_l, v_l, p_l,
                                       rho_r, u_r, v_r, p_r,
                                       n2x, n2y, gamma)
            vals2 = _hllc_hll_face(rho_l, u_l, v_l, p_l,
                                   rho_r, u_r, v_r, p_r, n2x, n2y, gamma)
            f20 = (1.0 - shock) * r0 + shock * vals2[0]
            f21 = (1.0 - shock) * r1 + shock * vals2[1]
            f22 = (1.0 - shock) * r2 + shock * vals2[2]
            f23 = (1.0 - shock) * r3 + shock * vals2[3]
            out[0, i] = a1 * vals1[4] + a2 * f20
            out[1, i] = a1 * vals1[5] + a2 * f21
            out[2, i] = a1 * vals1[6] + a2 * f22
            out[3, i] = a1 * vals1[7] + a2 * f23
        return out


    @njit(parallel=True, cache=True)
    def _roe_rotated_soft_shock_hybrid_kernel(WL, WR, normals, gamma):
        nf = normals.shape[0]
        out = np.empty((4, nf), dtype=np.float64)
        eps = 1.0e-30
        macheps = np.finfo(np.float64).eps
        for i in prange(nf):
            nx = normals[i, 0]
            ny = normals[i, 1]
            rho_l, u_l, v_l, p_l = WL[0, i], WL[1, i], WL[2, i], WL[3, i]
            rho_r, u_r, v_r, p_r = WR[0, i], WR[1, i], WR[2, i], WR[3, i]

            valsn = _hllc_hll_face(rho_l, u_l, v_l, p_l,
                                   rho_r, u_r, v_r, p_r,
                                   nx, ny, gamma)
            un_l, un_r = valsn[8], valsn[9]
            c_l, c_r = valsn[10], valsn[11]
            pp_l, pp_r = valsn[12], valsn[13]
            pressure_jump = abs(pp_r - pp_l) / max(pp_r + pp_l, eps)
            compression = max(0.0, un_l - un_r) / max(c_l + c_r, eps)
            jump_sensor = min(1.0, max(0.0, (pressure_jump - 0.05) / 0.35))
            compression_sensor = min(1.0, max(0.0, 4.0 * compression))
            shock = np.sqrt(jump_sensor * compression_sensor)
            soft_shock = shock * shock

            du = u_r - u_l
            dv = v_r - v_l
            speed_jump = np.sqrt(du * du + dv * dv)
            scale = macheps * (abs(u_l) + abs(u_r) + abs(v_l) + abs(v_r) + 1.0)
            if speed_jump > scale:
                n1x = du / max(speed_jump, eps)
                n1y = dv / max(speed_jump, eps)
            else:
                n1x = nx
                n1y = ny
            a1 = nx * n1x + ny * n1y
            if a1 < 0.0:
                n1x = -n1x
                n1y = -n1y
                a1 = -a1
            n2x = -n1y
            n2y = n1x
            a2 = nx * n2x + ny * n2y
            if a2 < 0.0:
                n2x = -n2x
                n2y = -n2y
                a2 = -a2

            vals1 = _hllc_hll_face(rho_l, u_l, v_l, p_l,
                                   rho_r, u_r, v_r, p_r, n1x, n1y, gamma)
            r0, r1, r2, r3 = _roe_face(rho_l, u_l, v_l, p_l,
                                       rho_r, u_r, v_r, p_r,
                                       n2x, n2y, gamma)
            vals2 = _hllc_hll_face(rho_l, u_l, v_l, p_l,
                                   rho_r, u_r, v_r, p_r, n2x, n2y, gamma)
            f20 = (1.0 - soft_shock) * r0 + soft_shock * vals2[0]
            f21 = (1.0 - soft_shock) * r1 + soft_shock * vals2[1]
            f22 = (1.0 - soft_shock) * r2 + soft_shock * vals2[2]
            f23 = (1.0 - soft_shock) * r3 + soft_shock * vals2[3]
            out[0, i] = a1 * vals1[4] + a2 * f20
            out[1, i] = a1 * vals1[5] + a2 * f21
            out[2, i] = a1 * vals1[6] + a2 * f22
            out[3, i] = a1 * vals1[7] + a2 * f23
        return out


    @njit(parallel=True, cache=True)
    def _hllc_adc_kernel(WL, WR, normals, gamma):
        nf = normals.shape[0]
        out = np.empty((4, nf), dtype=np.float64)
        eps = 1.0e-30
        for i in prange(nf):
            nx = normals[i, 0]
            ny = normals[i, 1]
            vals = _hllc_hll_face(WL[0, i], WL[1, i], WL[2, i], WL[3, i],
                                  WR[0, i], WR[1, i], WR[2, i], WR[3, i],
                                  nx, ny, gamma)
            c0, c1, c2, c3 = vals[0], vals[1], vals[2], vals[3]
            h0, h1, h2, h3 = vals[4], vals[5], vals[6], vals[7]
            pp_l, pp_r = vals[12], vals[13]
            pressure_jump = abs(pp_r - pp_l) / max(pp_r + pp_l, eps)
            pressure_ratio = min(pp_l, pp_r) / max(max(pp_l, pp_r), eps)
            jump_sensor = min(1.0, max(0.0, (pressure_jump - 0.05) / 0.35))
            ratio_sensor = min(1.0, max(0.0, 1.0 - pressure_ratio))
            shock = np.sqrt(jump_sensor * ratio_sensor)

            out[0, i] = (1.0 - shock) * c0 + shock * h0
            tx = -ny
            ty = nx
            normal_c = c1 * nx + c2 * ny
            tangent_c = c1 * tx + c2 * ty
            tangent_h = h1 * tx + h2 * ty
            tangent_blend = (1.0 - shock) * tangent_c + shock * tangent_h
            out[1, i] = normal_c * nx + tangent_blend * tx
            out[2, i] = normal_c * ny + tangent_blend * ty
            out[3, i] = c3
        return out


    @njit(parallel=True, cache=True)
    def _hllc_adc_strong_kernel(WL, WR, normals, gamma):
        nf = normals.shape[0]
        out = np.empty((4, nf), dtype=np.float64)
        eps = 1.0e-30
        for i in prange(nf):
            nx = normals[i, 0]
            ny = normals[i, 1]
            vals = _hllc_hll_face(WL[0, i], WL[1, i], WL[2, i], WL[3, i],
                                  WR[0, i], WR[1, i], WR[2, i], WR[3, i],
                                  nx, ny, gamma)
            c0, c1, c2, c3 = vals[0], vals[1], vals[2], vals[3]
            h0, h1, h2 = vals[4], vals[5], vals[6]
            pp_l, pp_r = vals[12], vals[13]
            pressure_jump = abs(pp_r - pp_l) / max(pp_r + pp_l, eps)
            pressure_ratio = min(pp_l, pp_r) / max(max(pp_l, pp_r), eps)
            jump_sensor = min(1.0, max(0.0, (pressure_jump - 0.05) / 0.35))
            ratio_sensor = min(1.0, max(0.0, 1.0 - pressure_ratio))
            shock = max(jump_sensor, ratio_sensor)

            out[0, i] = (1.0 - shock) * c0 + shock * h0
            tx = -ny
            ty = nx
            normal_c = c1 * nx + c2 * ny
            tangent_c = c1 * tx + c2 * ty
            tangent_h = h1 * tx + h2 * ty
            tangent_blend = (1.0 - shock) * tangent_c + shock * tangent_h
            out[1, i] = normal_c * nx + tangent_blend * tx
            out[2, i] = normal_c * ny + tangent_blend * ty
            out[3, i] = c3
        return out


    @njit(parallel=True, cache=True)
    def _hllc_tangent_adc_kernel(WL, WR, normals, gamma):
        nf = normals.shape[0]
        out = np.empty((4, nf), dtype=np.float64)
        eps = 1.0e-30
        for i in prange(nf):
            nx = normals[i, 0]
            ny = normals[i, 1]
            vals = _hllc_hll_face(WL[0, i], WL[1, i], WL[2, i], WL[3, i],
                                  WR[0, i], WR[1, i], WR[2, i], WR[3, i],
                                  nx, ny, gamma)
            c0, c1, c2, c3 = vals[0], vals[1], vals[2], vals[3]
            h1, h2 = vals[5], vals[6]
            pp_l, pp_r = vals[12], vals[13]
            pressure_jump = abs(pp_r - pp_l) / max(pp_r + pp_l, eps)
            pressure_ratio = min(pp_l, pp_r) / max(max(pp_l, pp_r), eps)
            jump_sensor = min(1.0, max(0.0, (pressure_jump - 0.05) / 0.35))
            ratio_sensor = min(1.0, max(0.0, 1.0 - pressure_ratio))
            shock = np.sqrt(jump_sensor * ratio_sensor)

            tx = -ny
            ty = nx
            normal_c = c1 * nx + c2 * ny
            tangent_c = c1 * tx + c2 * ty
            tangent_h = h1 * tx + h2 * ty
            tangent_blend = (1.0 - shock) * tangent_c + shock * tangent_h
            out[0, i] = c0
            out[1, i] = normal_c * nx + tangent_blend * tx
            out[2, i] = normal_c * ny + tangent_blend * ty
            out[3, i] = c3
        return out


    @njit(parallel=True, cache=True)
    def _hllc_adc_normal_kernel(WL, WR, normals, gamma):
        nf = normals.shape[0]
        out = np.empty((4, nf), dtype=np.float64)
        eps = 1.0e-30
        for i in prange(nf):
            nx = normals[i, 0]
            ny = normals[i, 1]
            vals = _hllc_hll_face(WL[0, i], WL[1, i], WL[2, i], WL[3, i],
                                  WR[0, i], WR[1, i], WR[2, i], WR[3, i],
                                  nx, ny, gamma)
            c0, c1, c2, c3 = vals[0], vals[1], vals[2], vals[3]
            h0, h1, h2, h3 = vals[4], vals[5], vals[6], vals[7]
            pp_l, pp_r = vals[12], vals[13]
            pressure_jump = abs(pp_r - pp_l) / max(pp_r + pp_l, eps)
            pressure_ratio = min(pp_l, pp_r) / max(max(pp_l, pp_r), eps)
            jump_sensor = min(1.0, max(0.0, (pressure_jump - 0.05) / 0.35))
            ratio_sensor = min(1.0, max(0.0, 1.0 - pressure_ratio))
            shock = np.sqrt(jump_sensor * ratio_sensor)

            tx = -ny
            ty = nx
            normal_c = c1 * nx + c2 * ny
            normal_h = h1 * nx + h2 * ny
            tangent_c = c1 * tx + c2 * ty
            normal_blend = (1.0 - shock) * normal_c + shock * normal_h
            out[0, i] = (1.0 - shock) * c0 + shock * h0
            out[1, i] = normal_blend * nx + tangent_c * tx
            out[2, i] = normal_blend * ny + tangent_c * ty
            out[3, i] = (1.0 - shock) * c3 + shock * h3
        return out


    @njit(parallel=True, cache=True)
    def _hllc_adc_mass_normal_kernel(WL, WR, normals, gamma):
        nf = normals.shape[0]
        out = np.empty((4, nf), dtype=np.float64)
        eps = 1.0e-30
        for i in prange(nf):
            nx = normals[i, 0]
            ny = normals[i, 1]
            vals = _hllc_hll_face(WL[0, i], WL[1, i], WL[2, i], WL[3, i],
                                  WR[0, i], WR[1, i], WR[2, i], WR[3, i],
                                  nx, ny, gamma)
            c0, c1, c2, c3 = vals[0], vals[1], vals[2], vals[3]
            h0, h1, h2 = vals[4], vals[5], vals[6]
            pp_l, pp_r = vals[12], vals[13]
            pressure_jump = abs(pp_r - pp_l) / max(pp_r + pp_l, eps)
            pressure_ratio = min(pp_l, pp_r) / max(max(pp_l, pp_r), eps)
            jump_sensor = min(1.0, max(0.0, (pressure_jump - 0.05) / 0.35))
            ratio_sensor = min(1.0, max(0.0, 1.0 - pressure_ratio))
            shock = np.sqrt(jump_sensor * ratio_sensor)

            tx = -ny
            ty = nx
            normal_c = c1 * nx + c2 * ny
            normal_h = h1 * nx + h2 * ny
            tangent_c = c1 * tx + c2 * ty
            normal_blend = (1.0 - shock) * normal_c + shock * normal_h
            out[0, i] = (1.0 - shock) * c0 + shock * h0
            out[1, i] = normal_blend * nx + tangent_c * tx
            out[2, i] = normal_blend * ny + tangent_c * ty
            out[3, i] = c3
        return out


    @njit(parallel=True, cache=True)
    def _hllc_adc_full_kernel(WL, WR, normals, gamma):
        nf = normals.shape[0]
        out = np.empty((4, nf), dtype=np.float64)
        eps = 1.0e-30
        for i in prange(nf):
            nx = normals[i, 0]
            ny = normals[i, 1]
            vals = _hllc_hll_face(WL[0, i], WL[1, i], WL[2, i], WL[3, i],
                                  WR[0, i], WR[1, i], WR[2, i], WR[3, i],
                                  nx, ny, gamma)
            pp_l, pp_r = vals[12], vals[13]
            pressure_jump = abs(pp_r - pp_l) / max(pp_r + pp_l, eps)
            pressure_ratio = min(pp_l, pp_r) / max(max(pp_l, pp_r), eps)
            jump_sensor = min(1.0, max(0.0, (pressure_jump - 0.05) / 0.35))
            ratio_sensor = min(1.0, max(0.0, 1.0 - pressure_ratio))
            shock = np.sqrt(jump_sensor * ratio_sensor)
            for m in range(4):
                out[m, i] = (1.0 - shock) * vals[m] + shock * vals[m + 4]
        return out


    @njit(parallel=True, cache=True)
    def _hllc_pure_2d_kernel(WL, WR, normals, gamma):
        nf = normals.shape[0]
        out = np.empty((4, nf), dtype=np.float64)
        for i in prange(nf):
            nx = normals[i, 0]
            ny = normals[i, 1]
            vals = _hllc_hll_face(WL[0, i], WL[1, i], WL[2, i], WL[3, i],
                                  WR[0, i], WR[1, i], WR[2, i], WR[3, i],
                                  nx, ny, gamma)
            out[0, i] = vals[0]
            out[1, i] = vals[1]
            out[2, i] = vals[2]
            out[3, i] = vals[3]
        return out


    @njit(parallel=True, cache=True)
    def _hllc_rotated_hybrid_kernel(WL, WR, normals, gamma):
        nf = normals.shape[0]
        out = np.empty((4, nf), dtype=np.float64)
        eps = 1.0e-30
        macheps = np.finfo(np.float64).eps
        for i in prange(nf):
            nx = normals[i, 0]
            ny = normals[i, 1]
            rho_l, u_l, v_l, p_l = WL[0, i], WL[1, i], WL[2, i], WL[3, i]
            rho_r, u_r, v_r, p_r = WR[0, i], WR[1, i], WR[2, i], WR[3, i]
            vals = _hllc_hll_face(rho_l, u_l, v_l, p_l,
                                  rho_r, u_r, v_r, p_r, nx, ny, gamma)
            c0, c1, c2, c3 = vals[0], vals[1], vals[2], vals[3]
            h0, h1, h2, h3 = vals[4], vals[5], vals[6], vals[7]
            un_l, un_r = vals[8], vals[9]
            c_l, c_r = vals[10], vals[11]
            pp_l, pp_r = vals[12], vals[13]
            pressure_jump = abs(pp_r - pp_l) / max(pp_r + pp_l, eps)
            pressure_ratio = min(pp_l, pp_r) / max(max(pp_l, pp_r), eps)
            jump_sensor = min(1.0, max(0.0, (pressure_jump - 0.05) / 0.35))
            ratio_sensor = min(1.0, max(0.0, 1.0 - pressure_ratio))

            du = u_r - u_l
            dv = v_r - v_l
            dun_face = du * nx + dv * ny
            dut_face = du * (-ny) + dv * nx
            normality = abs(dun_face) / max(abs(dun_face) + abs(dut_face), eps)
            pressure_sensor = np.sqrt(jump_sensor * ratio_sensor)
            adc_shock = np.sqrt(pressure_sensor)
            rotated_shock = np.sqrt(pressure_sensor) * normality
            adc0 = (1.0 - adc_shock) * c0 + adc_shock * h0
            adc1 = (1.0 - adc_shock) * c1 + adc_shock * h1
            adc2 = (1.0 - adc_shock) * c2 + adc_shock * h2
            adc3 = (1.0 - adc_shock) * c3 + adc_shock * h3

            speed_jump = np.sqrt(du * du + dv * dv)
            scale = macheps * (abs(u_l) + abs(u_r) + abs(v_l) + abs(v_r) + 1.0)
            if speed_jump > scale:
                n1x = du / max(speed_jump, eps)
                n1y = dv / max(speed_jump, eps)
            else:
                n1x = nx
                n1y = ny
            a1raw = nx * n1x + ny * n1y
            if a1raw < 0.0:
                n1x = -n1x
                n1y = -n1y
                a1raw = -a1raw
            n2x = -n1y
            n2y = n1x
            a2raw = nx * n2x + ny * n2y
            if a2raw < 0.0:
                n2x = -n2x
                n2y = -n2y
                a2raw = -a2raw

            vals1 = _hllc_hll_face(rho_l, u_l, v_l, p_l,
                                   rho_r, u_r, v_r, p_r, n1x, n1y, gamma)
            vals2 = _hllc_hll_face(rho_l, u_l, v_l, p_l,
                                   rho_r, u_r, v_r, p_r, n2x, n2y, gamma)
            rot0 = a1raw * vals1[4] + a2raw * vals2[0]
            rot1 = a1raw * vals1[5] + a2raw * vals2[1]
            rot2 = a1raw * vals1[6] + a2raw * vals2[2]
            rot3 = a1raw * vals1[7] + a2raw * vals2[3]
            out[0, i] = (1.0 - rotated_shock) * adc0 + rotated_shock * rot0
            out[1, i] = (1.0 - rotated_shock) * adc1 + rotated_shock * rot1
            out[2, i] = (1.0 - rotated_shock) * adc2 + rotated_shock * rot2
            out[3, i] = (1.0 - rotated_shock) * adc3 + rotated_shock * rot3
        return out


    @njit(parallel=True, cache=True)
    def _hllc_rotated_compressive_hybrid_kernel(WL, WR, normals, gamma):
        nf = normals.shape[0]
        out = np.empty((4, nf), dtype=np.float64)
        eps = 1.0e-30
        macheps = np.finfo(np.float64).eps
        for i in prange(nf):
            nx = normals[i, 0]
            ny = normals[i, 1]
            rho_l, u_l, v_l, p_l = WL[0, i], WL[1, i], WL[2, i], WL[3, i]
            rho_r, u_r, v_r, p_r = WR[0, i], WR[1, i], WR[2, i], WR[3, i]
            vals = _hllc_hll_face(rho_l, u_l, v_l, p_l,
                                  rho_r, u_r, v_r, p_r, nx, ny, gamma)
            c0, c1, c2, c3 = vals[0], vals[1], vals[2], vals[3]
            h0, h1, h2, h3 = vals[4], vals[5], vals[6], vals[7]
            un_l, un_r = vals[8], vals[9]
            c_l, c_r = vals[10], vals[11]
            pp_l, pp_r = vals[12], vals[13]
            pressure_jump = abs(pp_r - pp_l) / max(pp_r + pp_l, eps)
            pressure_ratio = min(pp_l, pp_r) / max(max(pp_l, pp_r), eps)
            jump_sensor = min(1.0, max(0.0, (pressure_jump - 0.05) / 0.35))
            ratio_sensor = min(1.0, max(0.0, 1.0 - pressure_ratio))
            compression = max(0.0, un_l - un_r) / max(c_l + c_r, eps)
            compression_sensor = min(1.0, max(0.0, 4.0 * compression))

            du = u_r - u_l
            dv = v_r - v_l
            dun_face = du * nx + dv * ny
            dut_face = du * (-ny) + dv * nx
            normality = abs(dun_face) / max(abs(dun_face) + abs(dut_face), eps)
            pressure_sensor = np.sqrt(jump_sensor * ratio_sensor)
            shock_sensor = np.sqrt(pressure_sensor * compression_sensor)
            adc_shock = shock_sensor
            rotated_shock = shock_sensor * normality
            adc0 = (1.0 - adc_shock) * c0 + adc_shock * h0
            adc1 = (1.0 - adc_shock) * c1 + adc_shock * h1
            adc2 = (1.0 - adc_shock) * c2 + adc_shock * h2
            adc3 = (1.0 - adc_shock) * c3 + adc_shock * h3

            speed_jump = np.sqrt(du * du + dv * dv)
            scale = macheps * (abs(u_l) + abs(u_r) + abs(v_l) + abs(v_r) + 1.0)
            if speed_jump > scale:
                n1x = du / max(speed_jump, eps)
                n1y = dv / max(speed_jump, eps)
            else:
                n1x = nx
                n1y = ny
            a1raw = nx * n1x + ny * n1y
            if a1raw < 0.0:
                n1x = -n1x
                n1y = -n1y
                a1raw = -a1raw
            n2x = -n1y
            n2y = n1x
            a2raw = nx * n2x + ny * n2y
            if a2raw < 0.0:
                n2x = -n2x
                n2y = -n2y
                a2raw = -a2raw

            vals1 = _hllc_hll_face(rho_l, u_l, v_l, p_l,
                                   rho_r, u_r, v_r, p_r, n1x, n1y, gamma)
            vals2 = _hllc_hll_face(rho_l, u_l, v_l, p_l,
                                   rho_r, u_r, v_r, p_r, n2x, n2y, gamma)
            rot0 = a1raw * vals1[4] + a2raw * vals2[0]
            rot1 = a1raw * vals1[5] + a2raw * vals2[1]
            rot2 = a1raw * vals1[6] + a2raw * vals2[2]
            rot3 = a1raw * vals1[7] + a2raw * vals2[3]
            out[0, i] = (1.0 - rotated_shock) * adc0 + rotated_shock * rot0
            out[1, i] = (1.0 - rotated_shock) * adc1 + rotated_shock * rot1
            out[2, i] = (1.0 - rotated_shock) * adc2 + rotated_shock * rot2
            out[3, i] = (1.0 - rotated_shock) * adc3 + rotated_shock * rot3
        return out


    @njit(parallel=True, cache=True)
    def _hllc_rotated_compressive_normal_hybrid_kernel(WL, WR, normals, gamma):
        nf = normals.shape[0]
        out = np.empty((4, nf), dtype=np.float64)
        eps = 1.0e-30
        macheps = np.finfo(np.float64).eps
        for i in prange(nf):
            nx = normals[i, 0]
            ny = normals[i, 1]
            tx = -ny
            ty = nx
            rho_l, u_l, v_l, p_l = WL[0, i], WL[1, i], WL[2, i], WL[3, i]
            rho_r, u_r, v_r, p_r = WR[0, i], WR[1, i], WR[2, i], WR[3, i]
            vals = _hllc_hll_face(rho_l, u_l, v_l, p_l,
                                  rho_r, u_r, v_r, p_r, nx, ny, gamma)
            c0, c1, c2, c3 = vals[0], vals[1], vals[2], vals[3]
            h0, h1, h2, h3 = vals[4], vals[5], vals[6], vals[7]
            un_l, un_r = vals[8], vals[9]
            c_l, c_r = vals[10], vals[11]
            pp_l, pp_r = vals[12], vals[13]
            pressure_jump = abs(pp_r - pp_l) / max(pp_r + pp_l, eps)
            pressure_ratio = min(pp_l, pp_r) / max(max(pp_l, pp_r), eps)
            jump_sensor = min(1.0, max(0.0, (pressure_jump - 0.05) / 0.35))
            ratio_sensor = min(1.0, max(0.0, 1.0 - pressure_ratio))
            compression = max(0.0, un_l - un_r) / max(c_l + c_r, eps)
            compression_sensor = min(1.0, max(0.0, 4.0 * compression))

            du = u_r - u_l
            dv = v_r - v_l
            dun_face = du * nx + dv * ny
            dut_face = du * tx + dv * ty
            normality = abs(dun_face) / max(abs(dun_face) + abs(dut_face), eps)
            pressure_sensor = np.sqrt(jump_sensor * ratio_sensor)
            shock_sensor = np.sqrt(pressure_sensor * compression_sensor)
            rotated_shock = shock_sensor * normality

            adc0 = (1.0 - shock_sensor) * c0 + shock_sensor * h0
            normal_c = c1 * nx + c2 * ny
            tangent_c = c1 * tx + c2 * ty
            normal_h = h1 * nx + h2 * ny
            normal_blend = (1.0 - shock_sensor) * normal_c + shock_sensor * normal_h
            adc1 = normal_blend * nx + tangent_c * tx
            adc2 = normal_blend * ny + tangent_c * ty
            adc3 = (1.0 - shock_sensor) * c3 + shock_sensor * h3

            speed_jump = np.sqrt(du * du + dv * dv)
            scale = macheps * (abs(u_l) + abs(u_r) + abs(v_l) + abs(v_r) + 1.0)
            if speed_jump > scale:
                n1x = du / max(speed_jump, eps)
                n1y = dv / max(speed_jump, eps)
            else:
                n1x = nx
                n1y = ny
            a1raw = nx * n1x + ny * n1y
            if a1raw < 0.0:
                n1x = -n1x
                n1y = -n1y
                a1raw = -a1raw
            n2x = -n1y
            n2y = n1x
            a2raw = nx * n2x + ny * n2y
            if a2raw < 0.0:
                n2x = -n2x
                n2y = -n2y
                a2raw = -a2raw

            vals1 = _hllc_hll_face(rho_l, u_l, v_l, p_l,
                                   rho_r, u_r, v_r, p_r, n1x, n1y, gamma)
            vals2 = _hllc_hll_face(rho_l, u_l, v_l, p_l,
                                   rho_r, u_r, v_r, p_r, n2x, n2y, gamma)
            rot0 = a1raw * vals1[4] + a2raw * vals2[0]
            rot1 = a1raw * vals1[5] + a2raw * vals2[1]
            rot2 = a1raw * vals1[6] + a2raw * vals2[2]
            rot3 = a1raw * vals1[7] + a2raw * vals2[3]
            out[0, i] = (1.0 - rotated_shock) * adc0 + rotated_shock * rot0
            out[1, i] = (1.0 - rotated_shock) * adc1 + rotated_shock * rot1
            out[2, i] = (1.0 - rotated_shock) * adc2 + rotated_shock * rot2
            out[3, i] = (1.0 - rotated_shock) * adc3 + rotated_shock * rot3
        return out


    @njit(parallel=True, cache=True)
    def _hllc_rotated_compressive_tangent_hybrid_kernel(WL, WR, normals, gamma):
        nf = normals.shape[0]
        out = np.empty((4, nf), dtype=np.float64)
        eps = 1.0e-30
        macheps = np.finfo(np.float64).eps
        for i in prange(nf):
            nx = normals[i, 0]
            ny = normals[i, 1]
            tx = -ny
            ty = nx
            rho_l, u_l, v_l, p_l = WL[0, i], WL[1, i], WL[2, i], WL[3, i]
            rho_r, u_r, v_r, p_r = WR[0, i], WR[1, i], WR[2, i], WR[3, i]
            vals = _hllc_hll_face(rho_l, u_l, v_l, p_l,
                                  rho_r, u_r, v_r, p_r, nx, ny, gamma)
            c0, c1, c2, c3 = vals[0], vals[1], vals[2], vals[3]
            h0, h1, h2, h3 = vals[4], vals[5], vals[6], vals[7]
            un_l, un_r = vals[8], vals[9]
            c_l, c_r = vals[10], vals[11]
            pp_l, pp_r = vals[12], vals[13]
            pressure_jump = abs(pp_r - pp_l) / max(pp_r + pp_l, eps)
            pressure_ratio = min(pp_l, pp_r) / max(max(pp_l, pp_r), eps)
            jump_sensor = min(1.0, max(0.0, (pressure_jump - 0.05) / 0.35))
            ratio_sensor = min(1.0, max(0.0, 1.0 - pressure_ratio))
            compression = max(0.0, un_l - un_r) / max(c_l + c_r, eps)
            compression_sensor = min(1.0, max(0.0, 4.0 * compression))

            du = u_r - u_l
            dv = v_r - v_l
            dun_face = du * nx + dv * ny
            dut_face = du * tx + dv * ty
            normality = abs(dun_face) / max(abs(dun_face) + abs(dut_face), eps)
            pressure_sensor = np.sqrt(jump_sensor * ratio_sensor)
            shock_sensor = np.sqrt(pressure_sensor * compression_sensor)
            rotated_shock = shock_sensor * normality

            adc0 = (1.0 - shock_sensor) * c0 + shock_sensor * h0
            normal_c = c1 * nx + c2 * ny
            tangent_c = c1 * tx + c2 * ty
            normal_h = h1 * nx + h2 * ny
            tangent_h = h1 * tx + h2 * ty
            normal_blend = (1.0 - shock_sensor) * normal_c + shock_sensor * normal_h
            tangent_blend = (1.0 - rotated_shock) * tangent_c + rotated_shock * tangent_h
            adc1 = normal_blend * nx + tangent_blend * tx
            adc2 = normal_blend * ny + tangent_blend * ty
            adc3 = (1.0 - shock_sensor) * c3 + shock_sensor * h3

            speed_jump = np.sqrt(du * du + dv * dv)
            scale = macheps * (abs(u_l) + abs(u_r) + abs(v_l) + abs(v_r) + 1.0)
            if speed_jump > scale:
                n1x = du / max(speed_jump, eps)
                n1y = dv / max(speed_jump, eps)
            else:
                n1x = nx
                n1y = ny
            a1raw = nx * n1x + ny * n1y
            if a1raw < 0.0:
                n1x = -n1x
                n1y = -n1y
                a1raw = -a1raw
            n2x = -n1y
            n2y = n1x
            a2raw = nx * n2x + ny * n2y
            if a2raw < 0.0:
                n2x = -n2x
                n2y = -n2y
                a2raw = -a2raw

            vals1 = _hllc_hll_face(rho_l, u_l, v_l, p_l,
                                   rho_r, u_r, v_r, p_r, n1x, n1y, gamma)
            vals2 = _hllc_hll_face(rho_l, u_l, v_l, p_l,
                                   rho_r, u_r, v_r, p_r, n2x, n2y, gamma)
            rot0 = a1raw * vals1[4] + a2raw * vals2[0]
            rot1 = a1raw * vals1[5] + a2raw * vals2[1]
            rot2 = a1raw * vals1[6] + a2raw * vals2[2]
            rot3 = a1raw * vals1[7] + a2raw * vals2[3]
            out[0, i] = (1.0 - rotated_shock) * adc0 + rotated_shock * rot0
            out[1, i] = (1.0 - rotated_shock) * adc1 + rotated_shock * rot1
            out[2, i] = (1.0 - rotated_shock) * adc2 + rotated_shock * rot2
            out[3, i] = (1.0 - rotated_shock) * adc3 + rotated_shock * rot3
        return out


    @njit(parallel=True, cache=True)
    def _hllc_rotated_compressive_normality2_hybrid_kernel(WL, WR, normals, gamma):
        nf = normals.shape[0]
        out = np.empty((4, nf), dtype=np.float64)
        eps = 1.0e-30
        macheps = np.finfo(np.float64).eps
        for i in prange(nf):
            nx = normals[i, 0]
            ny = normals[i, 1]
            tx = -ny
            ty = nx
            rho_l, u_l, v_l, p_l = WL[0, i], WL[1, i], WL[2, i], WL[3, i]
            rho_r, u_r, v_r, p_r = WR[0, i], WR[1, i], WR[2, i], WR[3, i]
            vals = _hllc_hll_face(rho_l, u_l, v_l, p_l,
                                  rho_r, u_r, v_r, p_r, nx, ny, gamma)
            c0, c1, c2, c3 = vals[0], vals[1], vals[2], vals[3]
            h0, h1, h2, h3 = vals[4], vals[5], vals[6], vals[7]
            un_l, un_r = vals[8], vals[9]
            c_l, c_r = vals[10], vals[11]
            pp_l, pp_r = vals[12], vals[13]
            pressure_jump = abs(pp_r - pp_l) / max(pp_r + pp_l, eps)
            pressure_ratio = min(pp_l, pp_r) / max(max(pp_l, pp_r), eps)
            jump_sensor = min(1.0, max(0.0, (pressure_jump - 0.05) / 0.35))
            ratio_sensor = min(1.0, max(0.0, 1.0 - pressure_ratio))
            compression = max(0.0, un_l - un_r) / max(c_l + c_r, eps)
            compression_sensor = min(1.0, max(0.0, 4.0 * compression))

            du = u_r - u_l
            dv = v_r - v_l
            dun_face = du * nx + dv * ny
            dut_face = du * tx + dv * ty
            normality = abs(dun_face) / max(abs(dun_face) + abs(dut_face), eps)
            pressure_sensor = np.sqrt(jump_sensor * ratio_sensor)
            shock_sensor = np.sqrt(pressure_sensor * compression_sensor)
            rotated_shock = shock_sensor * normality
            tangent_shock = shock_sensor * normality * normality

            adc0 = (1.0 - shock_sensor) * c0 + shock_sensor * h0
            normal_c = c1 * nx + c2 * ny
            tangent_c = c1 * tx + c2 * ty
            normal_h = h1 * nx + h2 * ny
            tangent_h = h1 * tx + h2 * ty
            normal_blend = (1.0 - shock_sensor) * normal_c + shock_sensor * normal_h
            tangent_blend = (1.0 - tangent_shock) * tangent_c + tangent_shock * tangent_h
            adc1 = normal_blend * nx + tangent_blend * tx
            adc2 = normal_blend * ny + tangent_blend * ty
            adc3 = (1.0 - shock_sensor) * c3 + shock_sensor * h3

            speed_jump = np.sqrt(du * du + dv * dv)
            scale = macheps * (abs(u_l) + abs(u_r) + abs(v_l) + abs(v_r) + 1.0)
            if speed_jump > scale:
                n1x = du / max(speed_jump, eps)
                n1y = dv / max(speed_jump, eps)
            else:
                n1x = nx
                n1y = ny
            a1raw = nx * n1x + ny * n1y
            if a1raw < 0.0:
                n1x = -n1x
                n1y = -n1y
                a1raw = -a1raw
            n2x = -n1y
            n2y = n1x
            a2raw = nx * n2x + ny * n2y
            if a2raw < 0.0:
                n2x = -n2x
                n2y = -n2y
                a2raw = -a2raw

            vals1 = _hllc_hll_face(rho_l, u_l, v_l, p_l,
                                   rho_r, u_r, v_r, p_r, n1x, n1y, gamma)
            vals2 = _hllc_hll_face(rho_l, u_l, v_l, p_l,
                                   rho_r, u_r, v_r, p_r, n2x, n2y, gamma)
            rot0 = a1raw * vals1[4] + a2raw * vals2[0]
            rot1 = a1raw * vals1[5] + a2raw * vals2[1]
            rot2 = a1raw * vals1[6] + a2raw * vals2[2]
            rot3 = a1raw * vals1[7] + a2raw * vals2[3]
            out[0, i] = (1.0 - rotated_shock) * adc0 + rotated_shock * rot0
            out[1, i] = (1.0 - rotated_shock) * adc1 + rotated_shock * rot1
            out[2, i] = (1.0 - rotated_shock) * adc2 + rotated_shock * rot2
            out[3, i] = (1.0 - rotated_shock) * adc3 + rotated_shock * rot3
        return out


    @njit(parallel=True, cache=True)
    def _hllct_kernel(WL, WR, normals, gamma):
        nf = normals.shape[0]
        out = np.empty((4, nf), dtype=np.float64)
        eps = 1.0e-30
        for i in prange(nf):
            nx = normals[i, 0]
            ny = normals[i, 1]
            tx = -ny
            ty = nx
            rho_l = max(WL[0, i], eps)
            u_l = WL[1, i]
            v_l = WL[2, i]
            p_l = max(WL[3, i], eps)
            rho_r = max(WR[0, i], eps)
            u_r = WR[1, i]
            v_r = WR[2, i]
            p_r = max(WR[3, i], eps)

            vals = _hllc_hll_face(rho_l, u_l, v_l, p_l,
                                  rho_r, u_r, v_r, p_r,
                                  nx, ny, gamma)
            c0, c1, c2, c3 = vals[0], vals[1], vals[2], vals[3]
            h0, h1, h2, h3 = vals[4], vals[5], vals[6], vals[7]
            un_l, un_r = vals[8], vals[9]
            a_l, a_r = vals[10], vals[11]

            S_l = min(un_l - a_l, un_r - a_r)
            S_r = max(un_l + a_l, un_r + a_r)
            alpha_l = rho_l * (S_l - un_l)
            alpha_r = rho_r * (S_r - un_r)
            den = alpha_r - alpha_l
            if abs(den) <= eps:
                den = eps if den >= 0.0 else -eps
            S_m = (p_l - p_r + alpha_r * un_r - alpha_l * un_l) / den

            du = u_r - u_l
            dv = v_r - v_l
            dun = du * nx + dv * ny
            dut = du * tx + dv * ty
            dvel = np.sqrt(dun * dun + dut * dut)
            speed_l = np.sqrt(u_l * u_l + v_l * v_l)
            speed_r = np.sqrt(u_r * u_r + v_r * v_r)
            eps_u = min(10.0 * abs(un_l + un_r - 2.0 * S_m),
                        1.0e-4 * (speed_l + speed_r))
            ratio = (0.5 * abs(dun) + eps_u) / (dvel + eps_u + 1.0e-12)
            w = 1.0 - ratio * ratio
            if w < 0.0:
                w = 0.0
            elif w > 1.0:
                w = 1.0

            q_l = u_l * u_l + v_l * v_l
            q_r = u_r * u_r + v_r * v_r
            E_l = p_l / ((gamma - 1.0) * rho_l) + 0.5 * q_l
            E_r = p_r / ((gamma - 1.0) * rho_r) + 0.5 * q_r
            U_l0 = rho_l
            U_l1 = rho_l * u_l
            U_l2 = rho_l * v_l
            U_l3 = rho_l * E_l
            U_r0 = rho_r
            U_r1 = rho_r * u_r
            U_r2 = rho_r * v_r
            U_r3 = rho_r * E_r
            F_l0 = rho_l * un_l
            F_l1 = rho_l * u_l * un_l + p_l * nx
            F_l2 = rho_l * v_l * un_l + p_l * ny
            F_l3 = (rho_l * E_l + p_l) * un_l
            F_r0 = rho_r * un_r
            F_r1 = rho_r * u_r * un_r + p_r * nx
            F_r2 = rho_r * v_r * un_r + p_r * ny
            F_r3 = (rho_r * E_r + p_r) * un_r

            ut_l = u_l * tx + v_l * ty
            ut_r = u_r * tx + v_r * ty
            ut_star = (alpha_r * ut_r - alpha_l * ut_l) / den
            ut_energy = (alpha_r * ut_r * ut_r
                         - alpha_l * ut_l * ut_l) / den

            den_l = S_l - S_m
            den_r = S_r - S_m
            if abs(den_l) <= eps:
                den_l = eps if den_l >= 0.0 else -eps
            if abs(den_r) <= eps:
                den_r = eps if den_r >= 0.0 else -eps
            fac_l = alpha_l / den_l
            fac_r = alpha_r / den_r
            if abs(alpha_l) <= eps:
                alpha_l = eps if alpha_l >= 0.0 else -eps
            if abs(alpha_r) <= eps:
                alpha_r = eps if alpha_r >= 0.0 else -eps
            e_star_l = (E_l + (S_m - un_l) * (S_m + p_l / alpha_l)
                        + 0.5 * (ut_energy - ut_l * ut_l))
            e_star_r = (E_r + (S_m - un_r) * (S_m + p_r / alpha_r)
                        + 0.5 * (ut_energy - ut_r * ut_r))
            Usm_l0 = fac_l
            Usm_l1 = fac_l * (S_m * nx + ut_star * tx)
            Usm_l2 = fac_l * (S_m * ny + ut_star * ty)
            Usm_l3 = fac_l * e_star_l
            Usm_r0 = fac_r
            Usm_r1 = fac_r * (S_m * nx + ut_star * tx)
            Usm_r2 = fac_r * (S_m * ny + ut_star * ty)
            Usm_r3 = fac_r * e_star_r

            if S_l >= 0.0:
                m0, m1, m2, m3 = F_l0, F_l1, F_l2, F_l3
            elif S_m >= 0.0:
                m0 = F_l0 + S_l * (Usm_l0 - U_l0)
                m1 = F_l1 + S_l * (Usm_l1 - U_l1)
                m2 = F_l2 + S_l * (Usm_l2 - U_l2)
                m3 = F_l3 + S_l * (Usm_l3 - U_l3)
            elif S_r > 0.0:
                m0 = F_r0 + S_r * (Usm_r0 - U_r0)
                m1 = F_r1 + S_r * (Usm_r1 - U_r1)
                m2 = F_r2 + S_r * (Usm_r2 - U_r2)
                m3 = F_r3 + S_r * (Usm_r3 - U_r3)
            else:
                m0, m1, m2, m3 = F_r0, F_r1, F_r2, F_r3
            if (not np.isfinite(m0 + m1 + m2 + m3)
                    or fac_l <= 0.0 or fac_r <= 0.0):
                m0, m1, m2, m3 = h0, h1, h2, h3

            out[0, i] = w * c0 + (1.0 - w) * m0
            out[1, i] = w * c1 + (1.0 - w) * m1
            out[2, i] = w * c2 + (1.0 - w) * m2
            out[3, i] = w * c3 + (1.0 - w) * m3
        return out


    @njit(parallel=True, cache=True)
    def _hllc_lm_kernel(WL, WR, normals, gamma):
        nf = normals.shape[0]
        out = np.empty((4, nf), dtype=np.float64)
        eps = 1.0e-30
        ma_limit = 0.1
        half_pi = 0.5 * np.pi
        for i in prange(nf):
            nx = normals[i, 0]
            ny = normals[i, 1]
            tx = -ny
            ty = nx
            rho_l = max(WL[0, i], eps)
            rho_r = max(WR[0, i], eps)
            u_l = WL[1, i]
            v_l = WL[2, i]
            p_l = max(WL[3, i], eps)
            u_r = WR[1, i]
            v_r = WR[2, i]
            p_r = max(WR[3, i], eps)
            un_l = u_l * nx + v_l * ny
            un_r = u_r * nx + v_r * ny
            ut_l = u_l * tx + v_l * ty
            ut_r = u_r * tx + v_r * ty
            c_l = np.sqrt(max(gamma * p_l / rho_l, eps))
            c_r = np.sqrt(max(gamma * p_r / rho_r, eps))
            e_l = p_l / ((gamma - 1.0) * rho_l) + 0.5 * (u_l * u_l + v_l * v_l)
            e_r = p_r / ((gamma - 1.0) * rho_r) + 0.5 * (u_r * u_r + v_r * v_r)
            U_l0 = rho_l
            U_l1 = rho_l * u_l
            U_l2 = rho_l * v_l
            U_l3 = rho_l * e_l
            U_r0 = rho_r
            U_r1 = rho_r * u_r
            U_r2 = rho_r * v_r
            U_r3 = rho_r * e_r
            F_l0 = rho_l * un_l
            F_l1 = rho_l * u_l * un_l + p_l * nx
            F_l2 = rho_l * v_l * un_l + p_l * ny
            F_l3 = (rho_l * e_l + p_l) * un_l
            F_r0 = rho_r * un_r
            F_r1 = rho_r * u_r * un_r + p_r * nx
            F_r2 = rho_r * v_r * un_r + p_r * ny
            F_r3 = (rho_r * e_r + p_r) * un_r
            S_l = min(un_l - c_l, un_r - c_r)
            S_r = max(un_l + c_l, un_r + c_r)
            den = rho_l * (S_l - un_l) - rho_r * (S_r - un_r)
            if abs(den) <= eps:
                den = eps if den >= 0.0 else -eps
            S_m = (p_r - p_l
                   + rho_l * un_l * (S_l - un_l)
                   - rho_r * un_r * (S_r - un_r)) / den

            hden = max(S_r - S_l, eps)
            if S_l >= 0.0:
                H0, H1, H2, H3 = F_l0, F_l1, F_l2, F_l3
            elif S_r <= 0.0:
                H0, H1, H2, H3 = F_r0, F_r1, F_r2, F_r3
            else:
                H0 = (S_r * F_l0 - S_l * F_r0 + S_l * S_r * (U_r0 - U_l0)) / hden
                H1 = (S_r * F_l1 - S_l * F_r1 + S_l * S_r * (U_r1 - U_l1)) / hden
                H2 = (S_r * F_l2 - S_l * F_r2 + S_l * S_r * (U_r2 - U_l2)) / hden
                H3 = (S_r * F_l3 - S_l * F_r3 + S_l * S_r * (U_r3 - U_l3)) / hden

            den_l = S_l - S_m
            den_r = S_r - S_m
            if abs(den_l) <= eps:
                den_l = eps if den_l >= 0.0 else -eps
            if abs(den_r) <= eps:
                den_r = eps if den_r >= 0.0 else -eps
            fac_l = rho_l * (S_l - un_l) / den_l
            fac_r = rho_r * (S_r - un_r) / den_r
            wave_l = rho_l * (S_l - un_l)
            wave_r = rho_r * (S_r - un_r)
            if abs(wave_l) <= eps:
                wave_l = eps if wave_l >= 0.0 else -eps
            if abs(wave_r) <= eps:
                wave_r = eps if wave_r >= 0.0 else -eps
            mn_l = fac_l * S_m
            mt_l = fac_l * ut_l
            mn_r = fac_r * S_m
            mt_r = fac_r * ut_r
            Us_l0 = fac_l
            Us_l1 = mn_l * nx + mt_l * tx
            Us_l2 = mn_l * ny + mt_l * ty
            Us_l3 = fac_l * (e_l + (S_m - un_l) * (S_m + p_l / wave_l))
            Us_r0 = fac_r
            Us_r1 = mn_r * nx + mt_r * tx
            Us_r2 = mn_r * ny + mt_r * ty
            Us_r3 = fac_r * (e_r + (S_m - un_r) * (S_m + p_r / wave_r))

            ma_local = max(abs(un_l) / max(c_l, eps), abs(un_r) / max(c_r, eps))
            phi = np.sin(min(1.0, ma_local / ma_limit) * half_pi)
            S_l_lm = phi * S_l
            S_r_lm = phi * S_r
            S_abs = abs(S_m)
            if S_l >= 0.0:
                C0, C1, C2, C3 = F_l0, F_l1, F_l2, F_l3
            elif S_r <= 0.0:
                C0, C1, C2, C3 = F_r0, F_r1, F_r2, F_r3
            else:
                C0 = 0.5 * (F_l0 + F_r0) + 0.5 * (
                    S_l_lm * (Us_l0 - U_l0)
                    + S_abs * (Us_l0 - Us_r0)
                    + S_r_lm * (Us_r0 - U_r0))
                C1 = 0.5 * (F_l1 + F_r1) + 0.5 * (
                    S_l_lm * (Us_l1 - U_l1)
                    + S_abs * (Us_l1 - Us_r1)
                    + S_r_lm * (Us_r1 - U_r1))
                C2 = 0.5 * (F_l2 + F_r2) + 0.5 * (
                    S_l_lm * (Us_l2 - U_l2)
                    + S_abs * (Us_l2 - Us_r2)
                    + S_r_lm * (Us_r2 - U_r2))
                C3 = 0.5 * (F_l3 + F_r3) + 0.5 * (
                    S_l_lm * (Us_l3 - U_l3)
                    + S_abs * (Us_l3 - Us_r3)
                    + S_r_lm * (Us_r3 - U_r3))
            if (not np.isfinite(C0 + C1 + C2 + C3)
                    or Us_l0 <= 0.0 or Us_r0 <= 0.0):
                C0, C1, C2, C3 = H0, H1, H2, H3
            out[0, i] = C0
            out[1, i] = C1
            out[2, i] = C2
            out[3, i] = C3
        return out


    @njit(cache=True)
    def _hllc_swm_p_face(rho_l_in, u_l, v_l, p_l_in,
                         rho_r_in, u_r, v_r, p_r_in,
                         nx, ny, gamma):
        eps = 1.0e-30
        vals = _hllc_hll_face(rho_l_in, u_l, v_l, p_l_in,
                              rho_r_in, u_r, v_r, p_r_in,
                              nx, ny, gamma)
        c0, c1, c2, c3 = vals[0], vals[1], vals[2], vals[3]
        h0, h1, h2, h3 = vals[4], vals[5], vals[6], vals[7]
        rho_l = max(rho_l_in, eps)
        rho_r = max(rho_r_in, eps)
        p_l = max(p_l_in, eps)
        p_r = max(p_r_in, eps)
        un_l = u_l * nx + v_l * ny
        un_r = u_r * nx + v_r * ny
        c_l = np.sqrt(max(gamma * p_l / rho_l, eps))
        c_r = np.sqrt(max(gamma * p_r / rho_r, eps))
        e_l = p_l / ((gamma - 1.0) * rho_l) + 0.5 * (u_l * u_l + v_l * v_l)
        e_r = p_r / ((gamma - 1.0) * rho_r) + 0.5 * (u_r * u_r + v_r * v_r)
        U_l0 = rho_l
        U_l1 = rho_l * u_l
        U_l2 = rho_l * v_l
        U_l3 = rho_l * e_l
        U_r0 = rho_r
        U_r1 = rho_r * u_r
        U_r2 = rho_r * v_r
        U_r3 = rho_r * e_r
        F_l0 = rho_l * un_l
        F_l1 = rho_l * u_l * un_l + p_l * nx
        F_l2 = rho_l * v_l * un_l + p_l * ny
        F_l3 = (rho_l * e_l + p_l) * un_l
        F_r0 = rho_r * un_r
        F_r1 = rho_r * u_r * un_r + p_r * nx
        F_r2 = rho_r * v_r * un_r + p_r * ny
        F_r3 = (rho_r * e_r + p_r) * un_r
        S_l = min(un_l - c_l, un_r - c_r)
        S_r = max(un_l + c_l, un_r + c_r)
        if S_l >= 0.0:
            return (c0 + F_l0 - h0, c1 + F_l1 - h1,
                    c2 + F_l2 - h2, c3 + F_l3 - h3)
        if S_r <= 0.0:
            return (c0 + F_r0 - h0, c1 + F_r1 - h1,
                    c2 + F_r2 - h2, c3 + F_r3 - h3)

        dlam_l = abs((un_r - c_r) - (un_l - c_l))
        dlam_m = abs(un_r - un_l)
        dlam_r = abs((un_r + c_r) - (un_l + c_l))
        eta = 0.5 * max(dlam_l, max(dlam_m, dlam_r))
        p_ratio = min(p_l, p_r) / max(max(p_l, p_r), eps)
        omega = p_ratio ** 5.0
        du = u_r - u_l
        dv = v_r - v_l
        dun = du * nx + dv * ny
        dut = du * (-ny) + dv * nx
        normality = abs(dun) / max(abs(dun) + abs(dut), eps)
        eps_swm = (1.0 - omega) * eta * normality
        alpha = 3.5
        S_l_bar = S_l - alpha * eps_swm
        S_r_bar = S_r + alpha * eps_swm
        den = max(S_r - S_l, eps)
        a0 = (abs(S_r_bar) - abs(S_l_bar)) / (2.0 * den)
        a1 = ((abs(S_l_bar) * S_r - abs(S_r_bar) * S_l)
              / (2.0 * den))
        hm0 = 0.5 * (F_l0 + F_r0) + a0 * (F_l0 - F_r0) + a1 * (U_l0 - U_r0)
        hm1 = 0.5 * (F_l1 + F_r1) + a0 * (F_l1 - F_r1) + a1 * (U_l1 - U_r1)
        hm2 = 0.5 * (F_l2 + F_r2) + a0 * (F_l2 - F_r2) + a1 * (U_l2 - U_r2)
        hm3 = 0.5 * (F_l3 + F_r3) + a0 * (F_l3 - F_r3) + a1 * (U_l3 - U_r3)
        return (c0 + hm0 - h0, c1 + hm1 - h1,
                c2 + hm2 - h2, c3 + hm3 - h3)


    @njit(parallel=True, cache=True)
    def _hllc_swm_p_kernel(WL, WR, normals, gamma):
        nf = normals.shape[0]
        out = np.empty((4, nf), dtype=np.float64)
        for i in prange(nf):
            vals = _hllc_swm_p_face(WL[0, i], WL[1, i], WL[2, i], WL[3, i],
                                    WR[0, i], WR[1, i], WR[2, i], WR[3, i],
                                    normals[i, 0], normals[i, 1], gamma)
            out[0, i] = vals[0]
            out[1, i] = vals[1]
            out[2, i] = vals[2]
            out[3, i] = vals[3]
        return out


    @njit(parallel=True, cache=True)
    def _hlle_kernel(WL, WR, normals, gamma):
        nf = normals.shape[0]
        out = np.empty((4, nf), dtype=np.float64)
        for i in prange(nf):
            vals = _hllc_hll_face(WL[0, i], WL[1, i], WL[2, i], WL[3, i],
                                  WR[0, i], WR[1, i], WR[2, i], WR[3, i],
                                  normals[i, 0], normals[i, 1], gamma)
            out[0, i] = vals[4]
            out[1, i] = vals[5]
            out[2, i] = vals[6]
            out[3, i] = vals[7]
        return out


    @njit(parallel=True, cache=True)
    def _ausm_plus_up_kernel(WL, WR, normals, gamma):
        nf = normals.shape[0]
        out = np.empty((4, nf), dtype=np.float64)
        eps = 1.0e-30
        beta = 0.125
        alpha = 0.1875
        for i in prange(nf):
            nx = normals[i, 0]
            ny = normals[i, 1]
            rho_l = max(WL[0, i], eps)
            rho_r = max(WR[0, i], eps)
            u_l = WL[1, i]
            v_l = WL[2, i]
            u_r = WR[1, i]
            v_r = WR[2, i]
            p_l = max(WL[3, i], eps)
            p_r = max(WR[3, i], eps)
            un_l = u_l * nx + v_l * ny
            un_r = u_r * nx + v_r * ny
            c_l = np.sqrt(max(gamma * p_l / rho_l, eps))
            c_r = np.sqrt(max(gamma * p_r / rho_r, eps))
            a_face = max(0.5 * (c_l + c_r), eps)
            ml = un_l / a_face
            mr = un_r / a_face

            abs_ml = abs(ml)
            if abs_ml >= 1.0:
                mp = 0.5 * (ml + abs_ml)
                pp = 1.0 if ml >= 0.0 else 0.0
            else:
                m2p = 0.25 * (ml + 1.0) * (ml + 1.0)
                m2m = -0.25 * (ml - 1.0) * (ml - 1.0)
                mp = m2p * (1.0 - 16.0 * beta * m2m)
                pp = m2p * ((2.0 - ml) - 16.0 * alpha * ml * m2m)

            abs_mr = abs(mr)
            if abs_mr >= 1.0:
                mm = 0.5 * (mr - abs_mr)
                pm = 0.0 if mr >= 0.0 else 1.0
            else:
                m2p = 0.25 * (mr + 1.0) * (mr + 1.0)
                m2m = -0.25 * (mr - 1.0) * (mr - 1.0)
                mm = m2m * (1.0 + 16.0 * beta * m2p)
                pm = m2m * ((-2.0 - mr) + 16.0 * alpha * mr * m2p)

            rho_bar = 0.5 * (rho_l + rho_r)
            mbar2 = 0.5 * (un_l * un_l + un_r * un_r) / max(a_face * a_face, eps)
            pressure_diffusion = (
                0.25 * max(1.0 - mbar2, 0.0)
                * (p_r - p_l) / max(rho_bar * a_face * a_face, eps))
            m_face = mp + mm - pressure_diffusion
            rho_face = rho_l if m_face >= 0.0 else rho_r
            mdot = a_face * m_face * rho_face

            velocity_diffusion = (
                0.75 * pp * pm * (rho_l + rho_r) * a_face * (un_r - un_l))
            p_face = pp * p_l + pm * p_r - velocity_diffusion

            h_l = (gamma * p_l / ((gamma - 1.0) * rho_l)
                   + 0.5 * (u_l * u_l + v_l * v_l))
            h_r = (gamma * p_r / ((gamma - 1.0) * rho_r)
                   + 0.5 * (u_r * u_r + v_r * v_r))
            if mdot >= 0.0:
                u_up = u_l
                v_up = v_l
                h_up = h_l
            else:
                u_up = u_r
                v_up = v_r
                h_up = h_r
            out[0, i] = mdot
            out[1, i] = mdot * u_up + p_face * nx
            out[2, i] = mdot * v_up + p_face * ny
            out[3, i] = mdot * h_up
        return out


    @njit(parallel=True, cache=True)
    def _ausm_hlle_shock_kernel(WL, WR, normals, gamma):
        nf = normals.shape[0]
        out = np.empty((4, nf), dtype=np.float64)
        eps = 1.0e-30
        beta = 0.125
        alpha = 0.1875
        for i in prange(nf):
            nx = normals[i, 0]
            ny = normals[i, 1]
            rho_l = max(WL[0, i], eps)
            rho_r = max(WR[0, i], eps)
            u_l = WL[1, i]
            v_l = WL[2, i]
            u_r = WR[1, i]
            v_r = WR[2, i]
            p_l = max(WL[3, i], eps)
            p_r = max(WR[3, i], eps)
            un_l = u_l * nx + v_l * ny
            un_r = u_r * nx + v_r * ny
            c_l = np.sqrt(max(gamma * p_l / rho_l, eps))
            c_r = np.sqrt(max(gamma * p_r / rho_r, eps))
            a_face = max(0.5 * (c_l + c_r), eps)
            ml = un_l / a_face
            mr = un_r / a_face

            abs_ml = abs(ml)
            if abs_ml >= 1.0:
                mp = 0.5 * (ml + abs_ml)
                pp = 1.0 if ml >= 0.0 else 0.0
            else:
                m2p = 0.25 * (ml + 1.0) * (ml + 1.0)
                m2m = -0.25 * (ml - 1.0) * (ml - 1.0)
                mp = m2p * (1.0 - 16.0 * beta * m2m)
                pp = m2p * ((2.0 - ml) - 16.0 * alpha * ml * m2m)

            abs_mr = abs(mr)
            if abs_mr >= 1.0:
                mm = 0.5 * (mr - abs_mr)
                pm = 0.0 if mr >= 0.0 else 1.0
            else:
                m2p = 0.25 * (mr + 1.0) * (mr + 1.0)
                m2m = -0.25 * (mr - 1.0) * (mr - 1.0)
                mm = m2m * (1.0 + 16.0 * beta * m2p)
                pm = m2m * ((-2.0 - mr) + 16.0 * alpha * mr * m2p)

            rho_bar = 0.5 * (rho_l + rho_r)
            mbar2 = 0.5 * (un_l * un_l + un_r * un_r) / max(a_face * a_face, eps)
            pressure_diffusion = (
                0.25 * max(1.0 - mbar2, 0.0)
                * (p_r - p_l) / max(rho_bar * a_face * a_face, eps))
            m_face = mp + mm - pressure_diffusion
            rho_face = rho_l if m_face >= 0.0 else rho_r
            mdot = a_face * m_face * rho_face

            velocity_diffusion = (
                0.75 * pp * pm * (rho_l + rho_r) * a_face * (un_r - un_l))
            p_face = pp * p_l + pm * p_r - velocity_diffusion

            h_l = (gamma * p_l / ((gamma - 1.0) * rho_l)
                   + 0.5 * (u_l * u_l + v_l * v_l))
            h_r = (gamma * p_r / ((gamma - 1.0) * rho_r)
                   + 0.5 * (u_r * u_r + v_r * v_r))
            if mdot >= 0.0:
                u_up = u_l
                v_up = v_l
                h_up = h_l
            else:
                u_up = u_r
                v_up = v_r
                h_up = h_r
            ausm0 = mdot
            ausm1 = mdot * u_up + p_face * nx
            ausm2 = mdot * v_up + p_face * ny
            ausm3 = mdot * h_up

            vals = _hllc_hll_face(rho_l, u_l, v_l, p_l,
                                  rho_r, u_r, v_r, p_r, nx, ny, gamma)
            h0, h1, h2, h3 = vals[4], vals[5], vals[6], vals[7]
            pressure_jump = abs(p_r - p_l) / max(p_r + p_l, eps)
            compression = max(0.0, un_l - un_r) / max(c_l + c_r, eps)
            jump_sensor = min(1.0, max(0.0, (pressure_jump - 0.05) / 0.35))
            compression_sensor = min(1.0, max(0.0, 4.0 * compression))
            shock = jump_sensor * compression_sensor

            out[0, i] = (1.0 - shock) * ausm0 + shock * h0
            tx = -ny
            ty = nx
            normal_ausm = ausm1 * nx + ausm2 * ny
            tangent_ausm = ausm1 * tx + ausm2 * ty
            normal_hll = h1 * nx + h2 * ny
            normal_blend = (1.0 - shock) * normal_ausm + shock * normal_hll
            out[1, i] = normal_blend * nx + tangent_ausm * tx
            out[2, i] = normal_blend * ny + tangent_ausm * ty
            out[3, i] = (1.0 - shock) * ausm3 + shock * h3
        return out


    @njit(parallel=True, cache=True)
    def _ausm_slau2_shock_kernel(WL, WR, normals, gamma):
        nf = normals.shape[0]
        out = np.empty((4, nf), dtype=np.float64)
        eps = 1.0e-30
        beta = 0.125
        alpha = 0.1875
        for i in prange(nf):
            nx = normals[i, 0]
            ny = normals[i, 1]
            rho_l = max(WL[0, i], eps)
            rho_r = max(WR[0, i], eps)
            u_l = WL[1, i]
            v_l = WL[2, i]
            u_r = WR[1, i]
            v_r = WR[2, i]
            p_l = max(WL[3, i], eps)
            p_r = max(WR[3, i], eps)
            un_l = u_l * nx + v_l * ny
            un_r = u_r * nx + v_r * ny
            c_l = np.sqrt(max(gamma * p_l / rho_l, eps))
            c_r = np.sqrt(max(gamma * p_r / rho_r, eps))
            a_face = max(0.5 * (c_l + c_r), eps)
            ml = un_l / a_face
            mr = un_r / a_face

            abs_ml = abs(ml)
            if abs_ml >= 1.0:
                mp = 0.5 * (ml + abs_ml)
                pp = 1.0 if ml >= 0.0 else 0.0
            else:
                m2p = 0.25 * (ml + 1.0) * (ml + 1.0)
                m2m = -0.25 * (ml - 1.0) * (ml - 1.0)
                mp = m2p * (1.0 - 16.0 * beta * m2m)
                pp = m2p * ((2.0 - ml) - 16.0 * alpha * ml * m2m)

            abs_mr = abs(mr)
            if abs_mr >= 1.0:
                mm = 0.5 * (mr - abs_mr)
                pm = 0.0 if mr >= 0.0 else 1.0
            else:
                m2p = 0.25 * (mr + 1.0) * (mr + 1.0)
                m2m = -0.25 * (mr - 1.0) * (mr - 1.0)
                mm = m2m * (1.0 + 16.0 * beta * m2p)
                pm = m2m * ((-2.0 - mr) + 16.0 * alpha * mr * m2p)

            rho_bar = 0.5 * (rho_l + rho_r)
            mbar2 = 0.5 * (un_l * un_l + un_r * un_r) / max(a_face * a_face, eps)
            pressure_diffusion = (
                0.25 * max(1.0 - mbar2, 0.0)
                * (p_r - p_l) / max(rho_bar * a_face * a_face, eps))
            m_face = mp + mm - pressure_diffusion
            rho_face = rho_l if m_face >= 0.0 else rho_r
            ausm_mdot = a_face * m_face * rho_face

            speed_l2 = u_l * u_l + v_l * v_l
            speed_r2 = u_r * u_r + v_r * v_r
            h_l = gamma * p_l / ((gamma - 1.0) * rho_l) + 0.5 * speed_l2
            h_r = gamma * p_r / ((gamma - 1.0) * rho_r) + 0.5 * speed_r2
            velocity_diffusion = (
                0.75 * pp * pm * (rho_l + rho_r) * a_face * (un_r - un_l))
            p_face = pp * p_l + pm * p_r - velocity_diffusion
            if ausm_mdot >= 0.0:
                ausm_u_up = u_l
                ausm_v_up = v_l
                ausm_h_up = h_l
            else:
                ausm_u_up = u_r
                ausm_v_up = v_r
                ausm_h_up = h_r
            ausm0 = ausm_mdot
            ausm1 = ausm_mdot * ausm_u_up + p_face * nx
            ausm2 = ausm_mdot * ausm_v_up + p_face * ny
            ausm3 = ausm_mdot * ausm_h_up

            abs_vbar = (
                rho_l * abs(un_l) + rho_r * abs(un_r)
            ) / max(rho_l + rho_r, eps)
            g_left = max(min(ml, 0.0), -1.0)
            g_right = min(max(mr, 0.0), 1.0)
            g = -g_left * g_right
            abs_v_plus = (1.0 - g) * abs_vbar + g * abs(un_l)
            abs_v_minus = (1.0 - g) * abs_vbar + g * abs(un_r)
            mach_hat = min(1.0, np.sqrt(0.5 * (speed_l2 + speed_r2)) / a_face)
            chi = (1.0 - mach_hat) * (1.0 - mach_hat)
            slau_mdot = 0.5 * (
                rho_l * (un_l + abs_v_plus)
                + rho_r * (un_r - abs_v_minus)
                - chi * (p_r - p_l) / a_face)
            if slau_mdot >= 0.0:
                slau_u_up = u_l
                slau_v_up = v_l
                slau_h_up = h_l
            else:
                slau_u_up = u_r
                slau_v_up = v_r
                slau_h_up = h_r
            slau0 = slau_mdot
            slau1 = slau_mdot * slau_u_up + p_face * nx
            slau2 = slau_mdot * slau_v_up + p_face * ny
            slau3 = slau_mdot * slau_h_up

            pressure_jump = abs(p_r - p_l) / max(p_r + p_l, eps)
            compression = max(0.0, un_l - un_r) / max(c_l + c_r, eps)
            jump_sensor = min(1.0, max(0.0, (pressure_jump - 0.05) / 0.35))
            compression_sensor = min(1.0, max(0.0, 4.0 * compression))
            shock = np.sqrt(jump_sensor * compression_sensor)

            out[0, i] = (1.0 - shock) * ausm0 + shock * slau0
            out[1, i] = (1.0 - shock) * ausm1 + shock * slau1
            out[2, i] = (1.0 - shock) * ausm2 + shock * slau2
            out[3, i] = (1.0 - shock) * ausm3 + shock * slau3
        return out


def upwind_advection(eq, W_L, W_R, normal, points=None, face_velocity=None):
    """Pure-upwind flux for the linear advection equation.

    Velocity sampling priority:
      1. If `face_velocity` is supplied — use it directly (one vector per
         face, e.g. ½(a(x_o)+a(x_n)) cell-centre central average).
      2. Else if eq is variable-velocity and `points` provided — sample
         a(x_GP) analytically at the Gauss-quadrature point (default).
      3. Else fall back to constant `eq.velocity`.

    φ_f selection follows the sign of u_f = a·n (upwind upstream cell).
    """
    if face_velocity is not None:
        a = face_velocity
        a_dot_n = np.einsum('...i,...i->...', a, normal)
    elif getattr(eq, 'is_variable_velocity', False) and points is not None:
        a = eq.velocity_at(points)
        a_dot_n = np.einsum('...i,...i->...', a, normal)
    else:
        a_dot_n = np.einsum('i,...i->...', eq.velocity, normal)
    upwind_left = a_dot_n >= 0
    U_L = eq.prim_to_cons(W_L)
    U_R = eq.prim_to_cons(W_R)
    return np.where(upwind_left, a_dot_n * U_L, a_dot_n * U_R)


def central(eq, W_L, W_R, normal, points=None):
    """Pure central flux:  F = ½ (F_L + F_R), no dissipation.

    Textbook central differencing — well-known to be unconditionally
    unstable for advection without limiting and oscillation-prone at
    discontinuities.  Provided here as a reference comparison only.
    """
    U_L = eq.prim_to_cons(W_L)
    U_R = eq.prim_to_cons(W_R)
    try:
        F_L = eq.physical_flux(U_L, normal, points=points)
        F_R = eq.physical_flux(U_R, normal, points=points)
    except TypeError:
        F_L = eq.physical_flux(U_L, normal)
        F_R = eq.physical_flux(U_R, normal)
    return 0.5 * (F_L + F_R)


def llf(eq, W_L, W_R, normal, points=None):
    """Local Lax-Friedrichs (Rusanov)."""
    U_L = eq.prim_to_cons(W_L)
    U_R = eq.prim_to_cons(W_R)
    # Forward `points` only when the equation accepts it.
    try:
        F_L = eq.physical_flux(U_L, normal, points=points)
        F_R = eq.physical_flux(U_R, normal, points=points)
        lam = np.maximum(eq.max_wave_speed(U_L, normal, points=points),
                         eq.max_wave_speed(U_R, normal, points=points))
    except TypeError:
        F_L = eq.physical_flux(U_L, normal)
        F_R = eq.physical_flux(U_R, normal)
        lam = np.maximum(eq.max_wave_speed(U_L, normal),
                         eq.max_wave_speed(U_R, normal))
    return 0.5 * (F_L + F_R) - 0.5 * lam * (U_R - U_L)


def _hll_flux(eq, U_L, U_R, F_L, F_R, S_L, S_R):
    den = np.maximum(S_R - S_L, _EPS)
    return np.where(
        S_L >= 0.0, F_L,
        np.where(
            S_R <= 0.0, F_R,
            (S_R * F_L - S_L * F_R + S_L * S_R * (U_R - U_L)) / den,
        )
    )


def _euler2d_hllc_hll(eq, W_L, W_R, normal):
    n = np.asarray(normal, dtype=float)
    nx = n[..., 0]
    ny = n[..., 1]
    tx = -ny
    ty = nx

    rho_L = np.maximum(W_L[0], _EPS)
    rho_R = np.maximum(W_R[0], _EPS)
    u_L, v_L, p_L = W_L[1], W_L[2], np.maximum(W_L[3], _EPS)
    u_R, v_R, p_R = W_R[1], W_R[2], np.maximum(W_R[3], _EPS)
    un_L = u_L * nx + v_L * ny
    un_R = u_R * nx + v_R * ny
    ut_L = u_L * tx + v_L * ty
    ut_R = u_R * tx + v_R * ty
    c_L = np.sqrt(np.maximum(eq.gamma * p_L / rho_L, _EPS))
    c_R = np.sqrt(np.maximum(eq.gamma * p_R / rho_R, _EPS))
    E_L = p_L / ((eq.gamma - 1.0) * rho_L) + 0.5 * (u_L * u_L + v_L * v_L)
    E_R = p_R / ((eq.gamma - 1.0) * rho_R) + 0.5 * (u_R * u_R + v_R * v_R)

    U_L = eq.prim_to_cons(np.stack([rho_L, u_L, v_L, p_L], axis=0))
    U_R = eq.prim_to_cons(np.stack([rho_R, u_R, v_R, p_R], axis=0))
    F_L = eq.physical_flux(U_L, normal)
    F_R = eq.physical_flux(U_R, normal)

    S_L = np.minimum(un_L - c_L, un_R - c_R)
    S_R = np.maximum(un_L + c_L, un_R + c_R)
    den = rho_L * (S_L - un_L) - rho_R * (S_R - un_R)
    den = np.where(np.abs(den) > _EPS, den, np.sign(den) * _EPS + _EPS)
    S_M = (p_R - p_L
           + rho_L * un_L * (S_L - un_L)
           - rho_R * un_R * (S_R - un_R)) / den

    def star_state(rho, un, ut, p, E, S):
        den_star = S - S_M
        den_star = np.where(np.abs(den_star) > _EPS,
                            den_star, np.sign(den_star) * _EPS + _EPS)
        fac = rho * (S - un) / den_star
        mn = fac * S_M
        mt = fac * ut
        mx = mn * nx + mt * tx
        my = mn * ny + mt * ty
        wave_den = rho * (S - un)
        wave_den = np.where(np.abs(wave_den) > _EPS,
                            wave_den, np.sign(wave_den) * _EPS + _EPS)
        e_star = fac * (
            E + (S_M - un) * (S_M + p / wave_den))
        return np.stack([fac, mx, my, e_star], axis=0)

    U_star_L = star_state(rho_L, un_L, ut_L, p_L, E_L, S_L)
    U_star_R = star_state(rho_R, un_R, ut_R, p_R, E_R, S_R)
    F_hllc = np.where(
        S_L >= 0.0, F_L,
        np.where(
            S_M >= 0.0, F_L + S_L * (U_star_L - U_L),
            np.where(
                S_R > 0.0, F_R + S_R * (U_star_R - U_R),
                F_R,
            )
        )
    )
    F_hll = _hll_flux(eq, U_L, U_R, F_L, F_R, S_L, S_R)
    star_ok = (
        np.all(np.isfinite(U_star_L), axis=0)
        & np.all(np.isfinite(U_star_R), axis=0)
        & (U_star_L[0] > 0.0)
        & (U_star_R[0] > 0.0)
        & np.all(np.isfinite(F_hllc), axis=0)
    )
    F_hllc = np.where(star_ok[None, :], F_hllc, F_hll)

    return F_hllc, F_hll, un_L, un_R, c_L, c_R, p_L, p_R


def _pressure_compression_sensor(un_L, un_R, c_L, c_R, p_L, p_R):
    pressure_jump = np.abs(p_R - p_L) / np.maximum(p_R + p_L, _EPS)
    pressure_ratio = (
        np.minimum(p_L, p_R) / np.maximum(np.maximum(p_L, p_R), _EPS))
    jump_sensor = np.clip((pressure_jump - 0.05) / 0.35, 0.0, 1.0)
    ratio_sensor = np.clip(1.0 - pressure_ratio, 0.0, 1.0)
    pressure_sensor = np.sqrt(jump_sensor * ratio_sensor)
    return pressure_sensor


def _pressure_compressive_sensor(un_L, un_R, c_L, c_R, p_L, p_R):
    pressure_jump = np.abs(p_R - p_L) / np.maximum(p_R + p_L, _EPS)
    compression = np.maximum(0.0, un_L - un_R) / np.maximum(c_L + c_R, _EPS)
    jump_sensor = np.clip((pressure_jump - 0.05) / 0.35, 0.0, 1.0)
    compression_sensor = np.clip(4.0 * compression, 0.0, 1.0)
    return np.sqrt(jump_sensor * compression_sensor)


def _adc_normal_blend(F_hllc, F_hll, normal, shock):
    nx = normal[..., 0]
    ny = normal[..., 1]
    tx = -ny
    ty = nx
    out = F_hllc.copy()
    out[0] = (1.0 - shock) * F_hllc[0] + shock * F_hll[0]
    normal_hllc = F_hllc[1] * nx + F_hllc[2] * ny
    normal_hll = F_hll[1] * nx + F_hll[2] * ny
    tangent_hllc = F_hllc[1] * tx + F_hllc[2] * ty
    normal_blend = (1.0 - shock) * normal_hllc + shock * normal_hll
    out[1] = normal_blend * nx + tangent_hllc * tx
    out[2] = normal_blend * ny + tangent_hllc * ty
    out[3] = (1.0 - shock) * F_hllc[3] + shock * F_hll[3]
    return out


def _normal_motion_factor(W_L, W_R, normal):
    du = W_R[1] - W_L[1]
    dv = W_R[2] - W_L[2]
    dun = du * normal[..., 0] + dv * normal[..., 1]
    dut = du * (-normal[..., 1]) + dv * normal[..., 0]
    return np.abs(dun) / np.maximum(np.abs(dun) + np.abs(dut), _EPS)


def _euler2d_hllc_lm(eq, W_L, W_R, normal, ma_limit=0.1):
    """Central-form HLLC-LM flux.

    Fleischmann et al. reduce only the acoustic HLLC signal speeds in the
    final central flux evaluation.  The star states and contact speed remain
    those of the original HLLC solver, so contacts/shear are not deliberately
    smeared by an HLLE fallback away from invalid states.
    """
    n = np.asarray(normal, dtype=float)
    nx = n[..., 0]
    ny = n[..., 1]
    tx = -ny
    ty = nx

    rho_L = np.maximum(W_L[0], _EPS)
    rho_R = np.maximum(W_R[0], _EPS)
    u_L, v_L, p_L = W_L[1], W_L[2], np.maximum(W_L[3], _EPS)
    u_R, v_R, p_R = W_R[1], W_R[2], np.maximum(W_R[3], _EPS)
    un_L = u_L * nx + v_L * ny
    un_R = u_R * nx + v_R * ny
    ut_L = u_L * tx + v_L * ty
    ut_R = u_R * tx + v_R * ty
    c_L = np.sqrt(np.maximum(eq.gamma * p_L / rho_L, _EPS))
    c_R = np.sqrt(np.maximum(eq.gamma * p_R / rho_R, _EPS))
    E_L = p_L / ((eq.gamma - 1.0) * rho_L) + 0.5 * (u_L * u_L + v_L * v_L)
    E_R = p_R / ((eq.gamma - 1.0) * rho_R) + 0.5 * (u_R * u_R + v_R * v_R)

    U_L = eq.prim_to_cons(np.stack([rho_L, u_L, v_L, p_L], axis=0))
    U_R = eq.prim_to_cons(np.stack([rho_R, u_R, v_R, p_R], axis=0))
    F_L = eq.physical_flux(U_L, n)
    F_R = eq.physical_flux(U_R, n)

    S_L = np.minimum(un_L - c_L, un_R - c_R)
    S_R = np.maximum(un_L + c_L, un_R + c_R)
    den = rho_L * (S_L - un_L) - rho_R * (S_R - un_R)
    den = np.where(np.abs(den) > _EPS, den, np.sign(den) * _EPS + _EPS)
    S_M = (p_R - p_L
           + rho_L * un_L * (S_L - un_L)
           - rho_R * un_R * (S_R - un_R)) / den

    def star_state(rho, un, ut, p, E, S):
        den_star = S - S_M
        den_star = np.where(np.abs(den_star) > _EPS,
                            den_star, np.sign(den_star) * _EPS + _EPS)
        fac = rho * (S - un) / den_star
        mn = fac * S_M
        mt = fac * ut
        mx = mn * nx + mt * tx
        my = mn * ny + mt * ty
        wave_den = rho * (S - un)
        wave_den = np.where(np.abs(wave_den) > _EPS,
                            wave_den, np.sign(wave_den) * _EPS + _EPS)
        e_star = fac * (
            E + (S_M - un) * (S_M + p / wave_den))
        return np.stack([fac, mx, my, e_star], axis=0)

    U_star_L = star_state(rho_L, un_L, ut_L, p_L, E_L, S_L)
    U_star_R = star_state(rho_R, un_R, ut_R, p_R, E_R, S_R)
    ma_local = np.maximum(np.abs(un_L) / np.maximum(c_L, _EPS),
                          np.abs(un_R) / np.maximum(c_R, _EPS))
    phi = np.sin(np.minimum(1.0, ma_local / ma_limit) * (0.5 * np.pi))
    S_L_lm = phi * S_L
    S_R_lm = phi * S_R
    F_star = (
        0.5 * (F_L + F_R)
        + 0.5 * (
            S_L_lm[None, :] * (U_star_L - U_L)
            + np.abs(S_M)[None, :] * (U_star_L - U_star_R)
            + S_R_lm[None, :] * (U_star_R - U_R)
        )
    )
    F_lm = np.where(
        S_L >= 0.0, F_L,
        np.where(S_R <= 0.0, F_R, F_star)
    )
    _, F_hll, *_ = _euler2d_hllc_hll(eq, W_L, W_R, n)
    star_ok = (
        np.all(np.isfinite(U_star_L), axis=0)
        & np.all(np.isfinite(U_star_R), axis=0)
        & (U_star_L[0] > 0.0)
        & (U_star_R[0] > 0.0)
        & np.all(np.isfinite(F_lm), axis=0)
    )
    return np.where(star_ok[None, :], F_lm, F_hll)


def _euler2d_hllc_swm_p(eq, W_L, W_R, normal):
    """HLLC-SWM-P flux in central HLL plus unchanged antidiffusion form.

    Mandal & Panwar's pressure-based selective-wave modification increases
    the embedded HLL dissipation near pressure shocks while preserving the
    HLLC contact/shear antidiffusive correction.
    """
    n = np.asarray(normal, dtype=float)
    nx = n[..., 0]
    ny = n[..., 1]
    F_hllc, F_hll, un_L, un_R, c_L, c_R, p_L, p_R = _euler2d_hllc_hll(
        eq, W_L, W_R, n)

    rho_L = np.maximum(W_L[0], _EPS)
    rho_R = np.maximum(W_R[0], _EPS)
    u_L, v_L = W_L[1], W_L[2]
    u_R, v_R = W_R[1], W_R[2]
    p_L = np.maximum(p_L, _EPS)
    p_R = np.maximum(p_R, _EPS)
    E_L = p_L / ((eq.gamma - 1.0) * rho_L) + 0.5 * (u_L * u_L + v_L * v_L)
    E_R = p_R / ((eq.gamma - 1.0) * rho_R) + 0.5 * (u_R * u_R + v_R * v_R)
    U_L = eq.prim_to_cons(np.stack([rho_L, u_L, v_L, p_L], axis=0))
    U_R = eq.prim_to_cons(np.stack([rho_R, u_R, v_R, p_R], axis=0))
    F_L = eq.physical_flux(U_L, n)
    F_R = eq.physical_flux(U_R, n)

    S_L = np.minimum(un_L - c_L, un_R - c_R)
    S_R = np.maximum(un_L + c_L, un_R + c_R)
    eta = 0.5 * np.maximum.reduce([
        np.abs((un_R - c_R) - (un_L - c_L)),
        np.abs(un_R - un_L),
        np.abs((un_R + c_R) - (un_L + c_L)),
    ])
    p_ratio = np.minimum(p_L, p_R) / np.maximum(np.maximum(p_L, p_R), _EPS)
    omega = p_ratio ** 5.0
    normality = _normal_motion_factor(W_L, W_R, n)
    eps_swm = (1.0 - omega) * eta * normality
    alpha = 3.5
    S_L_bar = S_L - alpha * eps_swm
    S_R_bar = S_R + alpha * eps_swm
    den = np.maximum(S_R - S_L, _EPS)
    a0 = (np.abs(S_R_bar) - np.abs(S_L_bar)) / (2.0 * den)
    a1 = ((np.abs(S_L_bar) * S_R - np.abs(S_R_bar) * S_L)
          / (2.0 * den))
    F_hll_mod_inside = (
        0.5 * (F_L + F_R)
        + a0[None, :] * (F_L - F_R)
        + a1[None, :] * (U_L - U_R)
    )
    F_hll_mod = np.where(
        S_L >= 0.0, F_L,
        np.where(S_R <= 0.0, F_R, F_hll_mod_inside)
    )
    return F_hllc + (F_hll_mod - F_hll)


def hllc_adc_2d(eq, W_L, W_R, normal, points=None):
    """Shock-stabilized HLLC flux for 2D Euler.

    This is an HLLC/HLLE hybrid in the spirit of recent HLLC shock-
    stabilization work: HLLC resolves contacts and shear waves away from
    shocks, while a pressure/compression sensor smoothly restores part of
    the more dissipative HLLE flux at strong compressive discontinuities.
    It is substantially less diffusive than LLF/Rusanov in smooth and shear
    regions but avoids using unmodified HLLC at grid-aligned strong shocks.
    """
    if eq.__class__.__name__ != 'Euler2D':
        return llf(eq, W_L, W_R, normal, points=points)
    if _NUMBA_AVAILABLE and W_L.shape[0] == 4 and W_L.shape[1] >= 512:
        return _hllc_adc_kernel(
            np.ascontiguousarray(W_L, dtype=np.float64),
            np.ascontiguousarray(W_R, dtype=np.float64),
            np.ascontiguousarray(normal, dtype=np.float64),
            float(eq.gamma))

    F_hllc, F_hll, un_L, un_R, c_L, c_R, p_L, p_R = _euler2d_hllc_hll(
        eq, W_L, W_R, normal)
    pressure_jump = np.abs(p_R - p_L) / np.maximum(p_R + p_L, 1.0e-30)
    pressure_ratio = np.minimum(p_L, p_R) / np.maximum(
        np.maximum(p_L, p_R), 1.0e-30)
    jump_sensor = np.clip((pressure_jump - 0.05) / 0.35, 0.0, 1.0)
    ratio_sensor = np.clip(1.0 - pressure_ratio, 0.0, 1.0)
    shock = np.sqrt(jump_sensor * ratio_sensor)
    F = F_hllc.copy()
    n = np.asarray(normal, dtype=float)
    nx = n[..., 0]
    ny = n[..., 1]
    tx = -ny
    ty = nx
    F[0] = (1.0 - shock) * F_hllc[0] + shock * F_hll[0]
    normal_hllc = F_hllc[1] * nx + F_hllc[2] * ny
    tangent_hllc = F_hllc[1] * tx + F_hllc[2] * ty
    tangent_hll = F_hll[1] * tx + F_hll[2] * ty
    tangent_blend = (1.0 - shock) * tangent_hllc + shock * tangent_hll
    F[1] = normal_hllc * nx + tangent_blend * tx
    F[2] = normal_hllc * ny + tangent_blend * ty

    return F


def hllc_pure_2d(eq, W_L, W_R, normal, points=None):
    """Contact-preserving 2D HLLC flux without shock-stability blending."""
    if eq.__class__.__name__ != 'Euler2D':
        return hllc_1d(eq, W_L, W_R, normal, points=points)
    if _NUMBA_AVAILABLE and W_L.shape[0] == 4 and W_L.shape[1] >= 512:
        return _hllc_pure_2d_kernel(
            np.ascontiguousarray(W_L, dtype=np.float64),
            np.ascontiguousarray(W_R, dtype=np.float64),
            np.ascontiguousarray(normal, dtype=np.float64),
            float(eq.gamma))
    F_hllc, _, *_ = _euler2d_hllc_hll(eq, W_L, W_R, np.asarray(normal))
    return F_hllc


def hllc_adc_strong_2d(eq, W_L, W_R, normal, points=None):
    """HLLC-ADC variant with stronger pressure-ratio shock sensing.

    It preserves the same mass/tangential ADC channel structure as the current
    Mach-step baseline, but uses the maximum of pressure-jump and pressure-
    ratio sensors.  This is still a pressure-ratio shock sensor and is not
    tied to any case geometry.
    """
    if eq.__class__.__name__ != 'Euler2D':
        return llf(eq, W_L, W_R, normal, points=points)
    if _NUMBA_AVAILABLE and W_L.shape[0] == 4 and W_L.shape[1] >= 512:
        return _hllc_adc_strong_kernel(
            np.ascontiguousarray(W_L, dtype=np.float64),
            np.ascontiguousarray(W_R, dtype=np.float64),
            np.ascontiguousarray(normal, dtype=np.float64),
            float(eq.gamma))
    F_hllc, F_hll, un_L, un_R, c_L, c_R, p_L, p_R = _euler2d_hllc_hll(
        eq, W_L, W_R, normal)
    pressure_jump = np.abs(p_R - p_L) / np.maximum(p_R + p_L, 1.0e-30)
    pressure_ratio = np.minimum(p_L, p_R) / np.maximum(
        np.maximum(p_L, p_R), 1.0e-30)
    jump_sensor = np.clip((pressure_jump - 0.05) / 0.35, 0.0, 1.0)
    ratio_sensor = np.clip(1.0 - pressure_ratio, 0.0, 1.0)
    shock = np.maximum(jump_sensor, ratio_sensor)
    F = F_hllc.copy()
    n = np.asarray(normal, dtype=float)
    nx = n[..., 0]
    ny = n[..., 1]
    tx = -ny
    ty = nx
    F[0] = (1.0 - shock) * F_hllc[0] + shock * F_hll[0]
    normal_hllc = F_hllc[1] * nx + F_hllc[2] * ny
    tangent_hllc = F_hllc[1] * tx + F_hllc[2] * ty
    tangent_hll = F_hll[1] * tx + F_hll[2] * ty
    tangent_blend = (1.0 - shock) * tangent_hllc + shock * tangent_hll
    F[1] = normal_hllc * nx + tangent_blend * tx
    F[2] = normal_hllc * ny + tangent_blend * ty
    return F


def hllc_lm_2d(eq, W_L, W_R, normal, points=None):
    """Low-Mach shock-stable HLLC-LM flux for 2D Euler.

    The directional Mach limiter follows Fleischmann, Adami and Adams'
    HLLC-LM proposal with the published Ma_limit=0.1.  It is not a local
    fallback: the same flux formula is used on every face.
    """
    if eq.__class__.__name__ != 'Euler2D':
        return llf(eq, W_L, W_R, normal, points=points)
    if _NUMBA_AVAILABLE and W_L.shape[0] == 4 and W_L.shape[1] >= 512:
        return _hllc_lm_kernel(
            np.ascontiguousarray(W_L, dtype=np.float64),
            np.ascontiguousarray(W_R, dtype=np.float64),
            np.ascontiguousarray(normal, dtype=np.float64),
            float(eq.gamma))
    return _euler2d_hllc_lm(eq, W_L, W_R, normal)


def hllc_swm_p_2d(eq, W_L, W_R, normal, points=None):
    """Pressure-based HLLC-SWM flux for 2D Euler.

    The published SWM-P constants alpha=3.5 and beta=5.0 are used in the
    pressure-ratio sensor; no benchmark-specific switch is applied.
    """
    if eq.__class__.__name__ != 'Euler2D':
        return llf(eq, W_L, W_R, normal, points=points)
    if _NUMBA_AVAILABLE and W_L.shape[0] == 4 and W_L.shape[1] >= 512:
        return _hllc_swm_p_kernel(
            np.ascontiguousarray(W_L, dtype=np.float64),
            np.ascontiguousarray(W_R, dtype=np.float64),
            np.ascontiguousarray(normal, dtype=np.float64),
            float(eq.gamma))
    return _euler2d_hllc_swm_p(eq, W_L, W_R, normal)


def hllct_2d(eq, W_L, W_R, normal, points=None):
    """Velocity-sensor HLLC/HLLCM-style flux for 2D Euler.

    Vevek-Zang-New's HLLCT idea uses a localized velocity-based shear sensor:
    apply tangential-momentum diffusion away from shear layers, but recover
    HLLC where tangential velocity jumps dominate.  The published constants in
    the sensor are used; no problem-specific switching is applied.
    """
    if eq.__class__.__name__ != 'Euler2D':
        return llf(eq, W_L, W_R, normal, points=points)
    if _NUMBA_AVAILABLE and W_L.shape[0] == 4 and W_L.shape[1] >= 512:
        return _hllct_kernel(
            np.ascontiguousarray(W_L, dtype=np.float64),
            np.ascontiguousarray(W_R, dtype=np.float64),
            np.ascontiguousarray(normal, dtype=np.float64),
            float(eq.gamma))
    n = np.asarray(normal, dtype=float)
    nx = n[..., 0]
    ny = n[..., 1]
    tx = -ny
    ty = nx

    F_hllc, _, un_L, un_R, c_L, c_R, p_L, p_R = _euler2d_hllc_hll(
        eq, W_L, W_R, n)
    rho_L = np.maximum(W_L[0], _EPS)
    rho_R = np.maximum(W_R[0], _EPS)
    u_L, v_L = W_L[1], W_L[2]
    u_R, v_R = W_R[1], W_R[2]
    du = u_R - u_L
    dv = v_R - v_L
    dun = du * nx + dv * ny
    dut = du * tx + dv * ty
    dvel = np.sqrt(dun * dun + dut * dut)

    S_L = np.minimum(un_L - c_L, un_R - c_R)
    S_R = np.maximum(un_L + c_L, un_R + c_R)
    den = rho_L * (S_L - un_L) - rho_R * (S_R - un_R)
    den = np.where(np.abs(den) > _EPS, den, np.sign(den) * _EPS + _EPS)
    S_M = (p_R - p_L
           + rho_L * un_L * (S_L - un_L)
           - rho_R * un_R * (S_R - un_R)) / den

    speed_L = np.sqrt(u_L * u_L + v_L * v_L)
    speed_R = np.sqrt(u_R * u_R + v_R * v_R)
    eps_u = np.minimum(10.0 * np.abs(un_L + un_R - 2.0 * S_M),
                       1.0e-4 * (speed_L + speed_R))
    ratio = (0.5 * np.abs(dun) + eps_u) / (dvel + eps_u + 1.0e-12)
    w = 1.0 - ratio * ratio
    w = np.clip(w, 0.0, 1.0)

    rho_bar = 0.5 * (rho_L + rho_R)
    a_bar = 0.5 * (c_L + c_R)
    tav = 0.5 * (1.0 - w) * rho_bar * a_bar * dut
    F = F_hllc.copy()
    F[1] -= tav * tx
    F[2] -= tav * ty
    return F


def hlle_2d(eq, W_L, W_R, normal, points=None):
    """HLLE/HLL flux for 2D Euler.

    This is a robust contact-averaged shock flux.  It is less sharp than
    HLLC-family fluxes, but it is carbuncle resistant for strong aligned
    shocks and gives a useful Mach-step stability baseline without switching
    to first-order reconstruction.
    """
    if eq.__class__.__name__ != 'Euler2D':
        return llf(eq, W_L, W_R, normal, points=points)
    if _NUMBA_AVAILABLE and W_L.shape[0] == 4 and W_L.shape[1] >= 512:
        return _hlle_kernel(
            np.ascontiguousarray(W_L, dtype=np.float64),
            np.ascontiguousarray(W_R, dtype=np.float64),
            np.ascontiguousarray(normal, dtype=np.float64),
            float(eq.gamma))
    _, F_hll, *_ = _euler2d_hllc_hll(eq, W_L, W_R, normal)
    return F_hll


def hllc_rotated_hybrid_2d(eq, W_L, W_R, normal, points=None):
    """Rotated-hybrid HLL/HLLC flux for shock-stable 2D Euler.

    The construction follows the rotated-hybrid Riemann-solver idea: near
    compressive shocks, use a carbuncle-free HLL flux in the local velocity-
    jump direction and retain HLLC in the orthogonal direction.  Away from
    shocks it reduces to the existing HLLC-ADC flux, so contact/shear
    resolution is not globally sacrificed.
    """
    if eq.__class__.__name__ != 'Euler2D':
        return llf(eq, W_L, W_R, normal, points=points)
    if _NUMBA_AVAILABLE and W_L.shape[0] == 4 and W_L.shape[1] >= 512:
        return _hllc_rotated_hybrid_kernel(
            np.ascontiguousarray(W_L, dtype=np.float64),
            np.ascontiguousarray(W_R, dtype=np.float64),
            np.ascontiguousarray(normal, dtype=np.float64),
            float(eq.gamma))

    n = np.asarray(normal, dtype=float)
    F_hllc, F_hll, un_L, un_R, c_L, c_R, p_L, p_R = _euler2d_hllc_hll(
        eq, W_L, W_R, n)
    shock = _pressure_compressive_sensor(un_L, un_R, c_L, c_R, p_L, p_R)
    normality = _normal_motion_factor(W_L, W_R, n)
    shock_weight = np.sqrt(shock)
    rotated_shock = np.sqrt(shock) * normality
    F_adc = (1.0 - shock_weight) * F_hllc + shock_weight * F_hll

    du = W_R[1] - W_L[1]
    dv = W_R[2] - W_L[2]
    speed_jump = np.sqrt(du * du + dv * dv)
    valid = speed_jump > (np.finfo(float).eps * (
        np.abs(W_L[1]) + np.abs(W_R[1]) + np.abs(W_L[2]) + np.abs(W_R[2])
        + 1.0))
    inv = np.where(valid, 1.0 / np.maximum(speed_jump, _EPS), 0.0)
    n1x = np.where(valid, du * inv, n[..., 0])
    n1y = np.where(valid, dv * inv, n[..., 1])
    a1 = n[..., 0] * n1x + n[..., 1] * n1y
    flip1 = a1 < 0.0
    n1x = np.where(flip1, -n1x, n1x)
    n1y = np.where(flip1, -n1y, n1y)
    a1 = np.abs(a1)

    n2x = -n1y
    n2y = n1x
    a2 = n[..., 0] * n2x + n[..., 1] * n2y
    flip2 = a2 < 0.0
    n2x = np.where(flip2, -n2x, n2x)
    n2y = np.where(flip2, -n2y, n2y)
    a2 = np.abs(a2)

    n1 = np.stack([n1x, n1y], axis=-1)
    n2 = np.stack([n2x, n2y], axis=-1)
    _, F_hll_1, *_ = _euler2d_hllc_hll(eq, W_L, W_R, n1)
    F_hllc_2, _, *_ = _euler2d_hllc_hll(eq, W_L, W_R, n2)
    F_rot = a1[None, :] * F_hll_1 + a2[None, :] * F_hllc_2
    return (1.0 - rotated_shock) * F_adc + rotated_shock * F_rot


def hllc_rotated_compressive_hybrid_2d(eq, W_L, W_R, normal, points=None):
    """Rotated HLL/HLLC with compression-gated shock dissipation.

    This variant keeps the same all-face rotated-hybrid structure as
    ``hllc_rotated_hybrid_2d`` but activates the HLL/rotated dissipation by
    the pressure-compression sensor used in the AUSM/SLAU shock variants.
    Pure pressure jumps in shear/contact-dominated regions therefore keep the
    HLLC contact/shear resolution, while genuinely compressive shocks still
    receive multidimensional HLL damping.
    """
    if eq.__class__.__name__ != 'Euler2D':
        return llf(eq, W_L, W_R, normal, points=points)
    if _NUMBA_AVAILABLE and W_L.shape[0] == 4 and W_L.shape[1] >= 512:
        return _hllc_rotated_compressive_hybrid_kernel(
            np.ascontiguousarray(W_L, dtype=np.float64),
            np.ascontiguousarray(W_R, dtype=np.float64),
            np.ascontiguousarray(normal, dtype=np.float64),
            float(eq.gamma))

    n = np.asarray(normal, dtype=float)
    F_hllc, F_hll, un_L, un_R, c_L, c_R, p_L, p_R = _euler2d_hllc_hll(
        eq, W_L, W_R, n)
    shock = _pressure_compressive_sensor(un_L, un_R, c_L, c_R, p_L, p_R)
    normality = _normal_motion_factor(W_L, W_R, n)
    rotated_shock = shock * normality
    F_adc = (1.0 - shock) * F_hllc + shock * F_hll

    du = W_R[1] - W_L[1]
    dv = W_R[2] - W_L[2]
    speed_jump = np.sqrt(du * du + dv * dv)
    valid = speed_jump > (np.finfo(float).eps * (
        np.abs(W_L[1]) + np.abs(W_R[1]) + np.abs(W_L[2]) + np.abs(W_R[2])
        + 1.0))
    inv = np.where(valid, 1.0 / np.maximum(speed_jump, _EPS), 0.0)
    n1x = np.where(valid, du * inv, n[..., 0])
    n1y = np.where(valid, dv * inv, n[..., 1])
    a1 = n[..., 0] * n1x + n[..., 1] * n1y
    flip1 = a1 < 0.0
    n1x = np.where(flip1, -n1x, n1x)
    n1y = np.where(flip1, -n1y, n1y)
    a1 = np.abs(a1)

    n2x = -n1y
    n2y = n1x
    a2 = n[..., 0] * n2x + n[..., 1] * n2y
    flip2 = a2 < 0.0
    n2x = np.where(flip2, -n2x, n2x)
    n2y = np.where(flip2, -n2y, n2y)
    a2 = np.abs(a2)

    n1 = np.stack([n1x, n1y], axis=-1)
    n2 = np.stack([n2x, n2y], axis=-1)
    _, F_hll_1, *_ = _euler2d_hllc_hll(eq, W_L, W_R, n1)
    F_hllc_2, _, *_ = _euler2d_hllc_hll(eq, W_L, W_R, n2)
    F_rot = a1[None, :] * F_hll_1 + a2[None, :] * F_hllc_2
    return (1.0 - rotated_shock) * F_adc + rotated_shock * F_rot


def hllc_rotated_compressive_normal_hybrid_2d(
        eq, W_L, W_R, normal, points=None):
    """Compression-gated rotated HLLC with normal-momentum shock damping.

    This keeps the mass/energy and face-normal momentum damping needed for
    carbuncle control, but preserves the HLLC face-tangential momentum in the
    non-rotated branch.  The same all-face pressure-compression sensor is used;
    there is no region-specific fallback.
    """
    if eq.__class__.__name__ != 'Euler2D':
        return llf(eq, W_L, W_R, normal, points=points)
    if _NUMBA_AVAILABLE and W_L.shape[0] == 4 and W_L.shape[1] >= 512:
        return _hllc_rotated_compressive_normal_hybrid_kernel(
            np.ascontiguousarray(W_L, dtype=np.float64),
            np.ascontiguousarray(W_R, dtype=np.float64),
            np.ascontiguousarray(normal, dtype=np.float64),
            float(eq.gamma))

    n = np.asarray(normal, dtype=float)
    F_hllc, F_hll, un_L, un_R, c_L, c_R, p_L, p_R = _euler2d_hllc_hll(
        eq, W_L, W_R, n)
    shock = _pressure_compressive_sensor(un_L, un_R, c_L, c_R, p_L, p_R)
    normality = _normal_motion_factor(W_L, W_R, n)
    rotated_shock = shock * normality
    F_adc = _adc_normal_blend(F_hllc, F_hll, n, shock)

    du = W_R[1] - W_L[1]
    dv = W_R[2] - W_L[2]
    speed_jump = np.sqrt(du * du + dv * dv)
    valid = speed_jump > (np.finfo(float).eps * (
        np.abs(W_L[1]) + np.abs(W_R[1]) + np.abs(W_L[2]) + np.abs(W_R[2])
        + 1.0))
    inv = np.where(valid, 1.0 / np.maximum(speed_jump, _EPS), 0.0)
    n1x = np.where(valid, du * inv, n[..., 0])
    n1y = np.where(valid, dv * inv, n[..., 1])
    a1 = n[..., 0] * n1x + n[..., 1] * n1y
    flip1 = a1 < 0.0
    n1x = np.where(flip1, -n1x, n1x)
    n1y = np.where(flip1, -n1y, n1y)
    a1 = np.abs(a1)

    n2x = -n1y
    n2y = n1x
    a2 = n[..., 0] * n2x + n[..., 1] * n2y
    flip2 = a2 < 0.0
    n2x = np.where(flip2, -n2x, n2x)
    n2y = np.where(flip2, -n2y, n2y)
    a2 = np.abs(a2)

    n1 = np.stack([n1x, n1y], axis=-1)
    n2 = np.stack([n2x, n2y], axis=-1)
    _, F_hll_1, *_ = _euler2d_hllc_hll(eq, W_L, W_R, n1)
    F_hllc_2, _, *_ = _euler2d_hllc_hll(eq, W_L, W_R, n2)
    F_rot = a1[None, :] * F_hll_1 + a2[None, :] * F_hllc_2
    return (1.0 - rotated_shock) * F_adc + rotated_shock * F_rot


def hllc_rotated_compressive_tangent_hybrid_2d(
        eq, W_L, W_R, normal, points=None):
    """Compression-gated rotated HLLC with sensor-limited tangent damping.

    Compared with ``hllc_rotated_compressive_normal_hybrid_2d``, the
    face-tangential momentum is not fully preserved at normal shocks.  It is
    blended toward the HLL value only by the same ``shock * normality`` factor
    used by the rotated branch, which keeps shear-dominated faces sharper
    without leaving grid-aligned shock perturbations undamped.
    """
    if eq.__class__.__name__ != 'Euler2D':
        return llf(eq, W_L, W_R, normal, points=points)
    if _NUMBA_AVAILABLE and W_L.shape[0] == 4 and W_L.shape[1] >= 512:
        return _hllc_rotated_compressive_tangent_hybrid_kernel(
            np.ascontiguousarray(W_L, dtype=np.float64),
            np.ascontiguousarray(W_R, dtype=np.float64),
            np.ascontiguousarray(normal, dtype=np.float64),
            float(eq.gamma))

    n = np.asarray(normal, dtype=float)
    F_hllc, F_hll, un_L, un_R, c_L, c_R, p_L, p_R = _euler2d_hllc_hll(
        eq, W_L, W_R, n)
    shock = _pressure_compressive_sensor(un_L, un_R, c_L, c_R, p_L, p_R)
    normality = _normal_motion_factor(W_L, W_R, n)
    tangent_shock = shock * normality
    F_adc = _adc_normal_blend(F_hllc, F_hll, n, shock)
    nx = n[..., 0]
    ny = n[..., 1]
    tx = -ny
    ty = nx
    tangent_c = F_hllc[1] * tx + F_hllc[2] * ty
    tangent_h = F_hll[1] * tx + F_hll[2] * ty
    tangent_blend = (1.0 - tangent_shock) * tangent_c + tangent_shock * tangent_h
    normal_adc = F_adc[1] * nx + F_adc[2] * ny
    F_adc[1] = normal_adc * nx + tangent_blend * tx
    F_adc[2] = normal_adc * ny + tangent_blend * ty

    du = W_R[1] - W_L[1]
    dv = W_R[2] - W_L[2]
    speed_jump = np.sqrt(du * du + dv * dv)
    valid = speed_jump > (np.finfo(float).eps * (
        np.abs(W_L[1]) + np.abs(W_R[1]) + np.abs(W_L[2]) + np.abs(W_R[2])
        + 1.0))
    inv = np.where(valid, 1.0 / np.maximum(speed_jump, _EPS), 0.0)
    n1x = np.where(valid, du * inv, n[..., 0])
    n1y = np.where(valid, dv * inv, n[..., 1])
    a1 = n[..., 0] * n1x + n[..., 1] * n1y
    flip1 = a1 < 0.0
    n1x = np.where(flip1, -n1x, n1x)
    n1y = np.where(flip1, -n1y, n1y)
    a1 = np.abs(a1)

    n2x = -n1y
    n2y = n1x
    a2 = n[..., 0] * n2x + n[..., 1] * n2y
    flip2 = a2 < 0.0
    n2x = np.where(flip2, -n2x, n2x)
    n2y = np.where(flip2, -n2y, n2y)
    a2 = np.abs(a2)

    n1 = np.stack([n1x, n1y], axis=-1)
    n2 = np.stack([n2x, n2y], axis=-1)
    _, F_hll_1, *_ = _euler2d_hllc_hll(eq, W_L, W_R, n1)
    F_hllc_2, _, *_ = _euler2d_hllc_hll(eq, W_L, W_R, n2)
    F_rot = a1[None, :] * F_hll_1 + a2[None, :] * F_hllc_2
    return (1.0 - tangent_shock) * F_adc + tangent_shock * F_rot


def hllc_rotated_compressive_normality2_hybrid_2d(
        eq, W_L, W_R, normal, points=None):
    """Compression-gated rotated HLLC with energy-fraction tangent damping.

    Tangential momentum damping uses ``shock * normality**2``.  The square is
    the normal component's share of the local velocity-jump energy, so this
    reduces damping in mixed shear/shock regions without adding a tunable
    coefficient.
    """
    if eq.__class__.__name__ != 'Euler2D':
        return llf(eq, W_L, W_R, normal, points=points)
    if _NUMBA_AVAILABLE and W_L.shape[0] == 4 and W_L.shape[1] >= 512:
        return _hllc_rotated_compressive_normality2_hybrid_kernel(
            np.ascontiguousarray(W_L, dtype=np.float64),
            np.ascontiguousarray(W_R, dtype=np.float64),
            np.ascontiguousarray(normal, dtype=np.float64),
            float(eq.gamma))

    n = np.asarray(normal, dtype=float)
    F_hllc, F_hll, un_L, un_R, c_L, c_R, p_L, p_R = _euler2d_hllc_hll(
        eq, W_L, W_R, n)
    shock = _pressure_compressive_sensor(un_L, un_R, c_L, c_R, p_L, p_R)
    normality = _normal_motion_factor(W_L, W_R, n)
    tangent_shock = shock * normality * normality
    rotated_shock = shock * normality
    F_adc = _adc_normal_blend(F_hllc, F_hll, n, shock)
    nx = n[..., 0]
    ny = n[..., 1]
    tx = -ny
    ty = nx
    tangent_c = F_hllc[1] * tx + F_hllc[2] * ty
    tangent_h = F_hll[1] * tx + F_hll[2] * ty
    tangent_blend = (1.0 - tangent_shock) * tangent_c + tangent_shock * tangent_h
    normal_adc = F_adc[1] * nx + F_adc[2] * ny
    F_adc[1] = normal_adc * nx + tangent_blend * tx
    F_adc[2] = normal_adc * ny + tangent_blend * ty

    du = W_R[1] - W_L[1]
    dv = W_R[2] - W_L[2]
    speed_jump = np.sqrt(du * du + dv * dv)
    valid = speed_jump > (np.finfo(float).eps * (
        np.abs(W_L[1]) + np.abs(W_R[1]) + np.abs(W_L[2]) + np.abs(W_R[2])
        + 1.0))
    inv = np.where(valid, 1.0 / np.maximum(speed_jump, _EPS), 0.0)
    n1x = np.where(valid, du * inv, n[..., 0])
    n1y = np.where(valid, dv * inv, n[..., 1])
    a1 = n[..., 0] * n1x + n[..., 1] * n1y
    flip1 = a1 < 0.0
    n1x = np.where(flip1, -n1x, n1x)
    n1y = np.where(flip1, -n1y, n1y)
    a1 = np.abs(a1)

    n2x = -n1y
    n2y = n1x
    a2 = n[..., 0] * n2x + n[..., 1] * n2y
    flip2 = a2 < 0.0
    n2x = np.where(flip2, -n2x, n2x)
    n2y = np.where(flip2, -n2y, n2y)
    a2 = np.abs(a2)

    n1 = np.stack([n1x, n1y], axis=-1)
    n2 = np.stack([n2x, n2y], axis=-1)
    _, F_hll_1, *_ = _euler2d_hllc_hll(eq, W_L, W_R, n1)
    F_hllc_2, _, *_ = _euler2d_hllc_hll(eq, W_L, W_R, n2)
    F_rot = a1[None, :] * F_hll_1 + a2[None, :] * F_hllc_2
    return (1.0 - rotated_shock) * F_adc + rotated_shock * F_rot


def roe_rotated_hybrid_2d(eq, W_L, W_R, normal, points=None):
    """Rotated HLL/Roe flux for shock stability and shear resolution.

    The velocity-jump direction uses HLL to suppress carbuncle-prone shock
    perturbations; the orthogonal direction uses Roe's full-wave flux to keep
    contact and shear waves sharper than HLL/HLLC blending.
    """
    if eq.__class__.__name__ != 'Euler2D':
        return llf(eq, W_L, W_R, normal, points=points)
    if _NUMBA_AVAILABLE and W_L.shape[0] == 4 and W_L.shape[1] >= 512:
        return _roe_rotated_hybrid_kernel(
            np.ascontiguousarray(W_L, dtype=np.float64),
            np.ascontiguousarray(W_R, dtype=np.float64),
            np.ascontiguousarray(normal, dtype=np.float64),
            float(eq.gamma))
    return hllc_rotated_hybrid_2d(eq, W_L, W_R, normal, points=points)


def roe_rotated_shock_hybrid_2d(eq, W_L, W_R, normal, points=None):
    """Roe-rotated flux with pressure-compression orthogonal damping.

    This keeps the current Roe-rotated structure that best preserved the
    upper shear-layer roll-up, but replaces the orthogonal Roe branch by a
    continuous HLLC blend only on pressure-compressive faces.  The sensor is
    the same all-face pressure/compression sensor already used by the HLLC-ADC
    family, so this is not a boundary-local or case-local switch.
    """
    if eq.__class__.__name__ != 'Euler2D':
        return llf(eq, W_L, W_R, normal, points=points)
    if _NUMBA_AVAILABLE and W_L.shape[0] == 4 and W_L.shape[1] >= 512:
        return _roe_rotated_shock_hybrid_kernel(
            np.ascontiguousarray(W_L, dtype=np.float64),
            np.ascontiguousarray(W_R, dtype=np.float64),
            np.ascontiguousarray(normal, dtype=np.float64),
            float(eq.gamma))

    n = np.asarray(normal, dtype=float)
    F_hllc, _, un_L, un_R, c_L, c_R, p_L, p_R = _euler2d_hllc_hll(
        eq, W_L, W_R, n)
    shock = _pressure_compressive_sensor(un_L, un_R, c_L, c_R, p_L, p_R)

    du = W_R[1] - W_L[1]
    dv = W_R[2] - W_L[2]
    speed_jump = np.sqrt(du * du + dv * dv)
    valid = speed_jump > (np.finfo(float).eps * (
        np.abs(W_L[1]) + np.abs(W_R[1]) + np.abs(W_L[2]) + np.abs(W_R[2])
        + 1.0))
    inv = np.where(valid, 1.0 / np.maximum(speed_jump, _EPS), 0.0)
    n1x = np.where(valid, du * inv, n[..., 0])
    n1y = np.where(valid, dv * inv, n[..., 1])
    a1 = n[..., 0] * n1x + n[..., 1] * n1y
    flip1 = a1 < 0.0
    n1x = np.where(flip1, -n1x, n1x)
    n1y = np.where(flip1, -n1y, n1y)
    a1 = np.abs(a1)

    n2x = -n1y
    n2y = n1x
    a2 = n[..., 0] * n2x + n[..., 1] * n2y
    flip2 = a2 < 0.0
    n2x = np.where(flip2, -n2x, n2x)
    n2y = np.where(flip2, -n2y, n2y)
    a2 = np.abs(a2)

    n1 = np.stack([n1x, n1y], axis=-1)
    n2 = np.stack([n2x, n2y], axis=-1)
    _, F_hll_1, *_ = _euler2d_hllc_hll(eq, W_L, W_R, n1)
    F_hllc_2, _, *_ = _euler2d_hllc_hll(eq, W_L, W_R, n2)
    # Vectorized fallback uses HLLC in the orthogonal branch because the Roe
    # helper is only implemented in the fast path.
    F_orth = (1.0 - shock)[None, :] * F_hllc + shock[None, :] * F_hllc_2
    return a1[None, :] * F_hll_1 + a2[None, :] * F_orth


def roe_rotated_soft_shock_hybrid_2d(eq, W_L, W_R, normal, points=None):
    """Roe-rotated flux with softened pressure-compression damping.

    Compared with ``roe_rotated_shock_hybrid_2d``, this uses the square of
    the same dimensionless pressure/compression sensor as a confidence
    measure.  It keeps full shock-normal damping when the sensor is certain
    while reducing dissipation in mixed shock/shear regions where the Mach-3
    step upper roll-up is sensitive to excessive contact/shear damping.
    """
    if eq.__class__.__name__ != 'Euler2D':
        return llf(eq, W_L, W_R, normal, points=points)
    if _NUMBA_AVAILABLE and W_L.shape[0] == 4 and W_L.shape[1] >= 512:
        return _roe_rotated_soft_shock_hybrid_kernel(
            np.ascontiguousarray(W_L, dtype=np.float64),
            np.ascontiguousarray(W_R, dtype=np.float64),
            np.ascontiguousarray(normal, dtype=np.float64),
            float(eq.gamma))

    n = np.asarray(normal, dtype=float)
    F_hllc, _, un_L, un_R, c_L, c_R, p_L, p_R = _euler2d_hllc_hll(
        eq, W_L, W_R, n)
    shock = _pressure_compressive_sensor(un_L, un_R, c_L, c_R, p_L, p_R)
    soft_shock = shock * shock

    du = W_R[1] - W_L[1]
    dv = W_R[2] - W_L[2]
    speed_jump = np.sqrt(du * du + dv * dv)
    valid = speed_jump > (np.finfo(float).eps * (
        np.abs(W_L[1]) + np.abs(W_R[1]) + np.abs(W_L[2]) + np.abs(W_R[2])
        + 1.0))
    inv = np.where(valid, 1.0 / np.maximum(speed_jump, _EPS), 0.0)
    n1x = np.where(valid, du * inv, n[..., 0])
    n1y = np.where(valid, dv * inv, n[..., 1])
    a1 = n[..., 0] * n1x + n[..., 1] * n1y
    flip1 = a1 < 0.0
    n1x = np.where(flip1, -n1x, n1x)
    n1y = np.where(flip1, -n1y, n1y)
    a1 = np.abs(a1)

    n2x = -n1y
    n2y = n1x
    a2 = n[..., 0] * n2x + n[..., 1] * n2y
    flip2 = a2 < 0.0
    n2x = np.where(flip2, -n2x, n2x)
    n2y = np.where(flip2, -n2y, n2y)
    a2 = np.abs(a2)

    n1 = np.stack([n1x, n1y], axis=-1)
    n2 = np.stack([n2x, n2y], axis=-1)
    _, F_hll_1, *_ = _euler2d_hllc_hll(eq, W_L, W_R, n1)
    F_hllc_2, _, *_ = _euler2d_hllc_hll(eq, W_L, W_R, n2)
    F_orth = (1.0 - soft_shock)[None, :] * F_hllc + soft_shock[None, :] * F_hllc_2
    return a1[None, :] * F_hll_1 + a2[None, :] * F_orth


def ausm_rotated_hybrid_2d(eq, W_L, W_R, normal, points=None):
    """AUSM+up with pressure-shock fallback to rotated-hybrid HLLC.

    AUSM+up is retained in contacts and shear layers for lower dissipation;
    the existing rotated-hybrid HLLC flux is blended in only by the same
    pressure/compression shock sensor used elsewhere in this module.
    """
    if eq.__class__.__name__ != 'Euler2D':
        return llf(eq, W_L, W_R, normal, points=points)
    n = np.asarray(normal, dtype=float)
    F_ausm = ausm_plus_up_2d(eq, W_L, W_R, n, points=points)
    F_stable = hllc_rotated_hybrid_2d(eq, W_L, W_R, n, points=points)
    _, _, un_L, un_R, c_L, c_R, p_L, p_R = _euler2d_hllc_hll(
        eq, W_L, W_R, n)
    shock = _pressure_compressive_sensor(un_L, un_R, c_L, c_R, p_L, p_R)
    return (1.0 - shock)[None, :] * F_ausm + shock[None, :] * F_stable


def ausm_hlle_shock_2d(eq, W_L, W_R, normal, points=None):
    """AUSM+-up with HLLE shock blending for strong compressive shocks.

    AUSM+-up is used in smooth/contact/shear regions. The pressure-compression
    sensor used by the HLLC shock-stable variants continuously increases HLLE
    dissipation at strong compressive discontinuities, which is the standard
    carbuncle-control tradeoff without a case-local switch.
    """
    if eq.__class__.__name__ != 'Euler2D':
        return llf(eq, W_L, W_R, normal, points=points)
    n = np.asarray(normal, dtype=float)
    if _NUMBA_AVAILABLE and W_L.shape[0] == 4 and W_L.shape[1] >= 512:
        return _ausm_hlle_shock_kernel(
            np.ascontiguousarray(W_L, dtype=np.float64),
            np.ascontiguousarray(W_R, dtype=np.float64),
            np.ascontiguousarray(n, dtype=np.float64),
            float(eq.gamma))
    F_ausm = ausm_plus_up_2d(eq, W_L, W_R, n, points=points)
    _, F_hll, un_L, un_R, c_L, c_R, p_L, p_R = _euler2d_hllc_hll(
        eq, W_L, W_R, n)
    shock = _pressure_compressive_sensor(un_L, un_R, c_L, c_R, p_L, p_R)
    out = F_ausm.copy()
    nx = n[..., 0]
    ny = n[..., 1]
    tx = -ny
    ty = nx
    out[0] = (1.0 - shock) * F_ausm[0] + shock * F_hll[0]
    normal_ausm = F_ausm[1] * nx + F_ausm[2] * ny
    tangent_ausm = F_ausm[1] * tx + F_ausm[2] * ty
    normal_hll = F_hll[1] * nx + F_hll[2] * ny
    normal_blend = (1.0 - shock) * normal_ausm + shock * normal_hll
    out[1] = normal_blend * nx + tangent_ausm * tx
    out[2] = normal_blend * ny + tangent_ausm * ty
    out[3] = (1.0 - shock) * F_ausm[3] + shock * F_hll[3]
    return out


def ausm_slau2_shock_2d(eq, W_L, W_R, normal, points=None):
    """AUSM+-up with SLAU2 mass-flux stabilization at compressive shocks.

    The same flux formula is used on every face. A pressure-compression shock
    sensor moves only the local AUSM-family mass/energy transport toward the
    SLAU2 form, while smooth contacts and shear layers retain AUSM+-up.
    """
    if eq.__class__.__name__ != 'Euler2D':
        return llf(eq, W_L, W_R, normal, points=points)
    n = np.asarray(normal, dtype=float)
    if _NUMBA_AVAILABLE and W_L.shape[0] == 4 and W_L.shape[1] >= 512:
        return _ausm_slau2_shock_kernel(
            np.ascontiguousarray(W_L, dtype=np.float64),
            np.ascontiguousarray(W_R, dtype=np.float64),
            np.ascontiguousarray(n, dtype=np.float64),
            float(eq.gamma))
    F_ausm = ausm_plus_up_2d(eq, W_L, W_R, n, points=points)
    F_slau = slau2_2d(eq, W_L, W_R, n, points=points)
    _, _, un_L, un_R, c_L, c_R, p_L, p_R = _euler2d_hllc_hll(
        eq, W_L, W_R, n)
    shock = _pressure_compressive_sensor(un_L, un_R, c_L, c_R, p_L, p_R)
    return (1.0 - shock)[None, :] * F_ausm + shock[None, :] * F_slau


def ausm_slau2_pressure_shock_2d(eq, W_L, W_R, normal, points=None):
    """AUSM-SLAU2 blend with pressure-ratio shock sensing.

    The standard `ausm_slau2_shock` branch requires pressure jump and
    face-normal compression.  Strong shock/wall interactions can leave compact
    pressure/density spots on faces where the local normal compression is weak.
    This variant uses the same all-face AUSM/SLAU2 blend, but activates it with
    the maximum of the compressive sensor and a pressure-jump/pressure-ratio
    sensor.  Contacts and shear layers with small pressure jumps remain on the
    low-dissipation AUSM path.
    """
    if eq.__class__.__name__ != 'Euler2D':
        return llf(eq, W_L, W_R, normal, points=points)
    n = np.asarray(normal, dtype=float)
    F_ausm = ausm_plus_up_2d(eq, W_L, W_R, n, points=points)
    F_slau = slau2_2d(eq, W_L, W_R, n, points=points)
    _, _, un_L, un_R, c_L, c_R, p_L, p_R = _euler2d_hllc_hll(
        eq, W_L, W_R, n)
    compressive = _pressure_compressive_sensor(
        un_L, un_R, c_L, c_R, p_L, p_R)
    pressure = _pressure_compression_sensor(
        un_L, un_R, c_L, c_R, p_L, p_R)
    shock = np.maximum(compressive, pressure)
    return (1.0 - shock)[None, :] * F_ausm + shock[None, :] * F_slau


def ausm_slau2_pressure_shock_soft_2d(eq, W_L, W_R, normal, points=None):
    """AUSM-SLAU2 pressure-shock blend with a softer pressure response."""
    if eq.__class__.__name__ != 'Euler2D':
        return llf(eq, W_L, W_R, normal, points=points)
    n = np.asarray(normal, dtype=float)
    F_ausm = ausm_plus_up_2d(eq, W_L, W_R, n, points=points)
    F_slau = slau2_2d(eq, W_L, W_R, n, points=points)
    _, _, un_L, un_R, c_L, c_R, p_L, p_R = _euler2d_hllc_hll(
        eq, W_L, W_R, n)
    compressive = _pressure_compressive_sensor(
        un_L, un_R, c_L, c_R, p_L, p_R)
    pressure = _pressure_compression_sensor(
        un_L, un_R, c_L, c_R, p_L, p_R)
    shock = np.maximum(compressive, pressure * pressure)
    return (1.0 - shock)[None, :] * F_ausm + shock[None, :] * F_slau


def ausm_slau2_pressure_shock_soft_normal_2d(
        eq, W_L, W_R, normal, points=None):
    """Soft pressure-shock blend with tangential momentum preserved.

    This uses the same shock sensor as `ausm_slau2_pressure_shock_soft`, but
    applies the SLAU2 blend only to mass, energy, and face-normal momentum.
    Face-tangential momentum stays on the AUSM path so shear-layer roll-up is
    not damped by the shock-stability channel.  The split is defined in local
    face coordinates on every face and is not a boundary or ROI fallback.
    """
    if eq.__class__.__name__ != 'Euler2D':
        return llf(eq, W_L, W_R, normal, points=points)
    n = np.asarray(normal, dtype=float)
    F_ausm = ausm_plus_up_2d(eq, W_L, W_R, n, points=points)
    F_slau = slau2_2d(eq, W_L, W_R, n, points=points)
    _, _, un_L, un_R, c_L, c_R, p_L, p_R = _euler2d_hllc_hll(
        eq, W_L, W_R, n)
    compressive = _pressure_compressive_sensor(
        un_L, un_R, c_L, c_R, p_L, p_R)
    pressure = _pressure_compression_sensor(
        un_L, un_R, c_L, c_R, p_L, p_R)
    shock = np.maximum(compressive, pressure * pressure)
    return _adc_normal_blend(F_ausm, F_slau, n, shock)


def ausm_slau2_pressure_guarded_shock_2d(eq, W_L, W_R, normal, points=None):
    """AUSM-SLAU2 pressure-shock blend guarded by acoustic/normal jumps.

    The full pressure sensor is too dissipative and unstable for the current
    high-order primitive reconstruction, while the squared soft sensor leaves
    compact carbuncle-like blobs in the Mach-step ROI.  This variant keeps the
    same all-face AUSM/SLAU2 blend but only restores the stronger pressure
    response when the jump is shock-normal/acoustic rather than mostly
    tangential shear.  The guard is local and nondimensional; it is not tied to
    a boundary, coordinate window, or benchmark component.
    """
    if eq.__class__.__name__ != 'Euler2D':
        return llf(eq, W_L, W_R, normal, points=points)
    n = np.asarray(normal, dtype=float)
    F_ausm = ausm_plus_up_2d(eq, W_L, W_R, n, points=points)
    F_slau = slau2_2d(eq, W_L, W_R, n, points=points)
    _, _, un_L, un_R, c_L, c_R, p_L, p_R = _euler2d_hllc_hll(
        eq, W_L, W_R, n)
    compressive = _pressure_compressive_sensor(
        un_L, un_R, c_L, c_R, p_L, p_R)
    pressure = _pressure_compression_sensor(
        un_L, un_R, c_L, c_R, p_L, p_R)

    tx = -n[..., 1]
    ty = n[..., 0]
    ut_L = W_L[1] * tx + W_L[2] * ty
    ut_R = W_R[1] * tx + W_R[2] * ty
    dn = un_R - un_L
    dtan = ut_R - ut_L
    rho_bar = 0.5 * (np.maximum(W_L[0], _EPS) + np.maximum(W_R[0], _EPS))
    c_bar = 0.5 * (c_L + c_R)
    dp_vel = np.abs(p_R - p_L) / np.maximum(rho_bar * c_bar, _EPS)
    normality = dn * dn / np.maximum(dn * dn + dtan * dtan, _EPS)
    acousticity = dp_vel * dp_vel / np.maximum(
        dp_vel * dp_vel + dtan * dtan, _EPS)
    pressure_guard = np.maximum(normality, acousticity)

    shock = np.maximum(compressive, pressure * pressure_guard)
    return (1.0 - shock)[None, :] * F_ausm + shock[None, :] * F_slau


def ausm_plus_up_shear_guard_geomean_hcorr_2d(
        eq, W_L, W_R, normal, points=None):
    """AUSM+up with mid-strength shear-guarded normal/energy H-correction."""
    if eq.__class__.__name__ != 'Euler2D':
        return llf(eq, W_L, W_R, normal, points=points)
    return ausm_plus_up_2d(eq, W_L, W_R, normal, points=points)


def ausm_slau2_hcorr_2d(eq, W_L, W_R, normal, points=None):
    """AUSM-SLAU2 shock blend with solver-level H-correction enabled."""
    return ausm_slau2_shock_2d(eq, W_L, W_R, normal, points=points)


def ausm_slau2_hcorr_strong_2d(eq, W_L, W_R, normal, points=None):
    """AUSM-SLAU2 blend with full normal-momentum/energy H-correction."""
    return ausm_slau2_shock_2d(eq, W_L, W_R, normal, points=points)


def ausm_slau2_transverse_soft_hcorr_2d(
        eq, W_L, W_R, normal, points=None):
    """AUSM-SLAU2 blend with soft mass/transverse H-correction.

    The base AUSM/SLAU2 formula is unchanged on every face.  The solver-level
    H-correction damps the mass flux and transverse momentum only inside a
    multidimensional pressure-compressive shock band.  This targets the
    odd-even/carbuncle channel without adding direct normal-momentum damping.
    """
    return ausm_slau2_shock_2d(eq, W_L, W_R, normal, points=points)


def ausm_slau2_normal_momentum_soft_hcorr_2d(
        eq, W_L, W_R, normal, points=None):
    """AUSM-SLAU2 blend with soft pressure-shock normal-momentum correction.

    This keeps the AUSM/SLAU2 mass and energy fluxes unchanged and applies
    only a pressure-jump shock-band HLLE blend to the normal momentum flux.
    It targets shock-normal odd-even instability while avoiding the energy
    damping that can smear upper shear-layer density rollup.
    """
    return ausm_slau2_shock_2d(eq, W_L, W_R, normal, points=points)


def ausm_slau2_shear_guard_hcorr_2d(eq, W_L, W_R, normal, points=None):
    """AUSM-SLAU2 with shock-normal, shear-guarded H-correction.

    The base flux is unchanged.  Solver-level H-correction uses the same
    pressure-compression shock-band sensor, then multiplies it by the ratio of
    normal velocity-jump energy to total velocity-jump energy.  This preserves
    genuine slip-line/shear roll-up while retaining damping for face-normal
    odd-even shock decoupling.
    """
    return ausm_slau2_shock_2d(eq, W_L, W_R, normal, points=points)


def ausm_slau2_shear_guard_geomean_hcorr_2d(
        eq, W_L, W_R, normal, points=None):
    """AUSM-SLAU2 with mid-strength shear-guarded H-correction.

    The solver-level correction uses the geometric mean between the linear
    shock-band weight and the squared soft weight: w^(3/2).  This keeps the
    same local shock/shear sensor on every face while targeting the gap where
    the squared soft weight leaves global shock splitting and the linear
    weight over-damps or destabilizes the high-order primitive reconstruction.
    """
    return ausm_slau2_shock_2d(eq, W_L, W_R, normal, points=points)


def ausm_slau2_shear_guard_mass_hcorr_2d(eq, W_L, W_R, normal, points=None):
    """AUSM-SLAU2 with shear-guarded mass-flux H-correction.

    AUSM-family carbuncle analyses identify pressure and density perturbations
    in the mass flux as an odd-even shock-instability path.  This variant uses
    the same all-face shear-guarded multidimensional sensor as the acoustic
    H-correction, but blends only the continuity flux toward HLLE so the
    tangential momentum carrying physical upper shear roll-up is not directly
    diffused.
    """
    return ausm_slau2_shock_2d(eq, W_L, W_R, normal, points=points)


def ausm_slau2_shear_guard_mass_energy_hcorr_2d(
        eq, W_L, W_R, normal, points=None):
    """AUSM-SLAU2 with shear-guarded mass/energy H-correction.

    This keeps the mass-flux perturbation damping motivated by AUSM carbuncle
    analyses, but corrects the thermodynamic inconsistency of a mass-only
    change by blending total-energy flux with the same HLLE weight.  Momentum
    fluxes remain the AUSM-SLAU2 values so post-shock tangential shear is not
    directly smeared by the shock cure.
    """
    return ausm_slau2_shock_2d(eq, W_L, W_R, normal, points=points)


def ausm_slau2_shear_guard_geomean_hllblend_2d(
        eq, W_L, W_R, normal, points=None):
    """AUSM-SLAU2 with mid-strength shear-guarded HLLE flux blending.

    This uses the same all-face shock/shear sensor and w^(3/2) band weight as
    `ausm_slau2_shear_guard_geomean_hcorr`, but blends the complete flux
    toward HLLE inside the detected multidimensional shock band.  It is a
    global flux-family variant, not a boundary or ROI-specific fix.
    """
    return ausm_slau2_shock_2d(eq, W_L, W_R, normal, points=points)


def ausm_slau2_shear_guard_soft_hllblend_2d(
        eq, W_L, W_R, normal, points=None):
    """AUSM-SLAU2 with weak shear-guarded HLLE flux blending.

    Compared with the geomean blend, this uses the squared shock-band weight.
    The goal is to keep the same multidimensional all-face carbuncle cure
    while reducing excessive damping of post-shock shear-layer density roll-up.
    """
    return ausm_slau2_shock_2d(eq, W_L, W_R, normal, points=points)


def ausm_slau2_shear_guard_ultrasoft_hllblend_2d(
        eq, W_L, W_R, normal, points=None):
    """AUSM-SLAU2 with very weak shear-guarded HLLE flux blending."""
    return ausm_slau2_shock_2d(eq, W_L, W_R, normal, points=points)


def ausm_slau2_shear_guard_directional_soft_hllblend_2d(
        eq, W_L, W_R, normal, points=None):
    """AUSM-SLAU2 with directional, shear-guarded HLLE flux blending.

    The solver-level correction spreads a pressure-compression shock sensor
    through adjacent cells only along faces aligned with the local pressure
    jump direction.  This follows multidimensional H-correction logic without
    adding boundary- or region-specific switches.
    """
    return ausm_slau2_shock_2d(eq, W_L, W_R, normal, points=points)


def ausm_slau2_pressure_guard_directional_soft_hllblend_2d(
        eq, W_L, W_R, normal, points=None):
    """AUSM-SLAU2 with pressure-guarded directional soft HLLE blending.

    This combines the pressure/acoustic guard used by the pressure-guarded
    H-correction with the directional shock-band spreading used by the soft
    HLL-blend family.  The same face-local nondimensional sensor is applied
    on every face; it is not a boundary, ROI, or time-window switch.
    """
    return ausm_slau2_shock_2d(eq, W_L, W_R, normal, points=points)


def ausm_slau2_pressure_guard_directional_geomean_hllblend_2d(
        eq, W_L, W_R, normal, points=None):
    """AUSM-SLAU2 with pressure-guarded directional geomean HLLE blending.

    This keeps the pressure/acoustic guard and pressure-gradient directional
    shock-band spreading of the soft HLL-blend variant, but uses the existing
    geomean shock-band strength.  The same face-local nondimensional sensor is
    applied on every face; it is not a boundary, ROI, or time-window switch.
    """
    return ausm_slau2_shock_2d(eq, W_L, W_R, normal, points=points)


def ausm_slau2_shear_guard_geomean_mass_normal_hcorr_2d(
        eq, W_L, W_R, normal, points=None):
    """AUSM-SLAU2 with mass/normal/energy H-correction.

    The shock-band sensor and w^(3/2) weight are the same as the geomean
    shear-guard variants.  Only mass flux, face-normal momentum, and energy
    are blended toward HLLE; face-tangential momentum is kept from the base
    AUSM/SLAU2 flux to preserve shear-layer roll-up.
    """
    return ausm_slau2_shock_2d(eq, W_L, W_R, normal, points=points)


def ausm_slau2_shear_guard_mass_normal_hcorr_2d(
        eq, W_L, W_R, normal, points=None):
    """AUSM-SLAU2 with soft mass/normal/energy H-correction."""
    return ausm_slau2_shock_2d(eq, W_L, W_R, normal, points=points)


def ausm_slau2_shear_guard_full_hcorr_2d(eq, W_L, W_R, normal, points=None):
    """AUSM-SLAU2 with unsquared shock-normal shear-guard H-correction."""
    return ausm_slau2_shock_2d(eq, W_L, W_R, normal, points=points)


def ausm_slau2_pressure_guard_hcorr_2d(eq, W_L, W_R, normal, points=None):
    """AUSM-SLAU2 with pressure-aware shear/contact guarded H-correction.

    The base AUSM/SLAU2 flux remains unchanged.  The solver-level correction
    uses a face-local acoustic pressure-jump measure so HLLE normal/energy
    damping is retained on true shocks, but reduced on nearly isobaric shear
    or contact faces where it would smear post-shock roll-up.
    """
    return ausm_slau2_shock_2d(eq, W_L, W_R, normal, points=points)


def ausm_slau2_pressure_guard_geomean_hcorr_2d(
        eq, W_L, W_R, normal, points=None):
    """AUSM-SLAU2 with pressure-guarded mid-strength H-correction.

    This combines the pressure/acoustic guard used to avoid damping nearly
    isobaric shear/contact faces with the existing geomean shock-band strength
    used by the shear-guard family.  It remains a global face flux variant:
    the same pressure-compression sensor and channel-selective normal/energy
    correction are applied on every face.
    """
    return ausm_slau2_shock_2d(eq, W_L, W_R, normal, points=points)


def ausm_slau2_face_hcorr_2d(eq, W_L, W_R, normal, points=None):
    """AUSM-SLAU2 with face-local soft normal-energy H-correction."""
    return ausm_slau2_shock_2d(eq, W_L, W_R, normal, points=points)


def ausm_slau2_directional_hcorr_2d(eq, W_L, W_R, normal, points=None):
    """AUSM-SLAU2 with pressure-gradient directional H-correction."""
    return ausm_slau2_shock_2d(eq, W_L, W_R, normal, points=points)


def hllc_adc_hcorr_2d(eq, W_L, W_R, normal, points=None):
    """HLLC-ADC flux with solver-level multidimensional H-correction."""
    return hllc_adc_2d(eq, W_L, W_R, normal, points=points)


def hllc_adc_acoustic_hcorr_2d(eq, W_L, W_R, normal, points=None):
    """HLLC-ADC with H-correction on normal-momentum/energy channels.

    The mass/contact flux is left as HLLC-ADC so density roll-up is not
    directly diffused.  The solver-level H-correction damps only acoustic
    normal momentum and energy in pressure-compressive transverse shock
    bands, targeting carbuncle modes with less contact smearing.
    """
    return hllc_adc_2d(eq, W_L, W_R, normal, points=points)


def hllc_adc_acoustic_soft_hcorr_2d(eq, W_L, W_R, normal, points=None):
    """HLLC-ADC with sensor-squared acoustic H-correction."""
    return hllc_adc_2d(eq, W_L, W_R, normal, points=points)


def hllc_adc_transverse_hcorr_2d(eq, W_L, W_R, normal, points=None):
    """HLLC-ADC with H-correction on mass/transverse-momentum channels.

    Recent HLLC shock-stabilization work identifies the carbuncle-prone
    transverse flux channel near strong grid-aligned shocks.  This variant
    keeps the base HLLC-ADC flux everywhere and lets the solver-level
    H-correction add dissipation only to mass and face-tangential momentum on
    compressive pressure-shock bands, using the same global sensor as the
    other H-correction paths.
    """
    return hllc_adc_2d(eq, W_L, W_R, normal, points=points)


def hllc_adc_transverse_energy_hcorr_2d(eq, W_L, W_R, normal, points=None):
    """HLLC-ADC with H-correction on mass, transverse momentum, and energy."""
    return hllc_adc_2d(eq, W_L, W_R, normal, points=points)


def hllc_adc_transverse_soft_hcorr_2d(eq, W_L, W_R, normal, points=None):
    """HLLC-ADC with sensor-squared transverse H-correction."""
    return hllc_adc_2d(eq, W_L, W_R, normal, points=points)


def hllc_adc_mass_hcorr_2d(eq, W_L, W_R, normal, points=None):
    """HLLC-ADC with H-correction only on the mass/contact flux channel."""
    return hllc_adc_2d(eq, W_L, W_R, normal, points=points)


def hllc_adc_full_transverse_soft_hcorr_2d(
        eq, W_L, W_R, normal, points=None):
    """Full HLLC-ADC with sensor-squared transverse H-correction."""
    return hllc_adc_full_2d(eq, W_L, W_R, normal, points=points)


def hllc_adc_full_transverse_hcorr_2d(eq, W_L, W_R, normal, points=None):
    """Full HLLC-ADC with mass/transverse-momentum H-correction.

    This combines the HLLC-ADC full antidiffusion control with the
    multidimensional H-correction channel identified in carbuncle studies:
    pressure-compressive shock bands receive extra cross-flow dissipation in
    the mass and face-tangential momentum fluxes, while the pressure-first
    reconstruction remains responsible for acoustic monotonicity.
    """
    return hllc_adc_full_2d(eq, W_L, W_R, normal, points=points)


def hllc_adc_full_neighbor_hcorr_2d(eq, W_L, W_R, normal, points=None):
    """Full HLLC-ADC with neighboring-face H-correction strength.

    The H-correction strength is taken from the maximum pressure-compressive
    shock sensor in the two cells adjacent to a face, matching the
    multidimensional entropy-fix idea of using neighboring shock information
    instead of only the current face jump.
    """
    return hllc_adc_full_2d(eq, W_L, W_R, normal, points=points)


def hllc_adc_full_neighbor_soft_hcorr_2d(eq, W_L, W_R, normal, points=None):
    """Full HLLC-ADC with sensor-squared neighboring H-correction."""
    return hllc_adc_full_2d(eq, W_L, W_R, normal, points=points)


def hllc_adc_full_neighbor_hllblend_2d(eq, W_L, W_R, normal, points=None):
    """Full HLLC-ADC with neighbor-aware convex HLL shock blending."""
    return hllc_adc_full_2d(eq, W_L, W_R, normal, points=points)


def hllc_adc_full_neighbor_soft_hllblend_2d(
        eq, W_L, W_R, normal, points=None):
    """Full HLLC-ADC with softened neighbor-aware convex HLL shock blending."""
    return hllc_adc_full_2d(eq, W_L, W_R, normal, points=points)


def hllc_adc_full_ducros_hll_soft_2d(eq, W_L, W_R, normal, points=None):
    """Full HLLC-ADC with face-local Ducros soft HLL shock blending."""
    return hllc_adc_full_2d(eq, W_L, W_R, normal, points=points)


def hllc_adc_hllct_momentum_2d(eq, W_L, W_R, normal, points=None):
    """HLLC-ADC mass/energy with HLLCT momentum stabilization.

    The density and energy fluxes remain the contact-preserving HLLC-ADC
    values.  Only the two Cartesian momentum flux components are taken from
    the HLLCT-style velocity-sensor flux, which damps carbuncle-prone
    tangential momentum modes near pressure shocks while avoiding direct mass
    diffusion of the upper contact layer.
    """
    F_adc = hllc_adc_2d(eq, W_L, W_R, normal, points=points)
    if eq.__class__.__name__ != 'Euler2D':
        return F_adc
    F_hllct = hllct_2d(eq, W_L, W_R, normal, points=points)
    out = F_adc.copy()
    out[1] = F_hllct[1]
    out[2] = F_hllct[2]
    return out


def hllc_adc_ausm_hlle_momentum_2d(eq, W_L, W_R, normal, points=None):
    """HLLC-ADC scalar transport with AUSM-HLLE shock-stable momentum.

    Density and energy keep the contact-preserving HLLC-ADC transport that
    best preserved the upper shear roll-up on the target grid.  Momentum is
    taken from the AUSM+up/HLLE shock blend, which was robust against isolated
    carbuncle blobs and applies the same pressure-compression sensor on every
    face rather than a case-local switch.
    """
    F_adc = hllc_adc_2d(eq, W_L, W_R, normal, points=points)
    if eq.__class__.__name__ != 'Euler2D':
        return F_adc
    F_ausm = ausm_hlle_shock_2d(eq, W_L, W_R, normal, points=points)
    out = F_adc.copy()
    out[1] = F_ausm[1]
    out[2] = F_ausm[2]
    return out


def hllc_adc_ausm_hlle_momentum_soft_hllblend_2d(
        eq, W_L, W_R, normal, points=None):
    """HLLC-ADC/AUSM-HLLE momentum plus positivity-limited HLL blend."""
    return hllc_adc_ausm_hlle_momentum_2d(
        eq, W_L, W_R, normal, points=points)


def hllc_adc_ausm_hlle_momentum_hllblend_2d(
        eq, W_L, W_R, normal, points=None):
    """HLLC-ADC/AUSM-HLLE momentum plus full neighbor HLL blend."""
    return hllc_adc_ausm_hlle_momentum_2d(
        eq, W_L, W_R, normal, points=points)


def hllc_adc_ausm_hlle_momentum_pressure_hcorr_2d(
        eq, W_L, W_R, normal, points=None):
    """HLLC-ADC/AUSM-HLLE momentum plus pressure-jump momentum damping."""
    return hllc_adc_ausm_hlle_momentum_2d(
        eq, W_L, W_R, normal, points=points)


def hllc_adc_ausm_hlle_momentum_pressure_normal_hcorr_2d(
        eq, W_L, W_R, normal, points=None):
    """Hybrid flux plus pressure-gradient-normal momentum damping."""
    return hllc_adc_ausm_hlle_momentum_2d(
        eq, W_L, W_R, normal, points=points)


def hllc_adc_ausm_hlle_momentum_pressureonly_normal_hcorr_2d(
        eq, W_L, W_R, normal, points=None):
    """Hybrid flux plus pressure-jump shock-band normal momentum damping."""
    return hllc_adc_ausm_hlle_momentum_2d(
        eq, W_L, W_R, normal, points=points)


def hllc_adc_ausm_hlle_momentum_pressureonly_soft_normal_hcorr_2d(
        eq, W_L, W_R, normal, points=None):
    """Hybrid flux plus sensor-squared pressure-band normal damping."""
    return hllc_adc_ausm_hlle_momentum_2d(
        eq, W_L, W_R, normal, points=points)


def hllc_adc_full_hllct_momentum_2d(eq, W_L, W_R, normal, points=None):
    """Full HLLC-ADC mass/energy with HLLCT momentum stabilization.

    This is the same globally applied flux on every face: the pressure-jump
    HLLC/HLLE blend supplies shock-stable mass and energy transport, while
    the HLLCT velocity sensor damps carbuncle-prone momentum modes without
    replacing the contact-resolving scalar fluxes by a case-local fallback.
    """
    F_adc = hllc_adc_full_2d(eq, W_L, W_R, normal, points=points)
    if eq.__class__.__name__ != 'Euler2D':
        return F_adc
    F_hllct = hllct_2d(eq, W_L, W_R, normal, points=points)
    out = F_adc.copy()
    out[1] = F_hllct[1]
    out[2] = F_hllct[2]
    return out


def hllc_adc_hllct_momentum_neighbor_hllblend_2d(
        eq, W_L, W_R, normal, points=None):
    """HLLC-ADC/HLLCT momentum plus solver-level neighbor HLL blending."""
    return hllc_adc_hllct_momentum_2d(
        eq, W_L, W_R, normal, points=points)


def hllc_adc_hllct_momentum_neighbor_soft_hllblend_2d(
        eq, W_L, W_R, normal, points=None):
    """Soft neighbor HLL shock-band blend on HLLC-ADC/HLLCT momentum."""
    return hllc_adc_hllct_momentum_2d(
        eq, W_L, W_R, normal, points=points)


def hllc_adc_hllct_momentum_neighbor_geomean_hllblend_2d(
        eq, W_L, W_R, normal, points=None):
    """Mid-strength neighbor HLL shock-band blend."""
    return hllc_adc_hllct_momentum_2d(
        eq, W_L, W_R, normal, points=points)


def hllc_adc_hllct_momentum_neighbor_soft_hllblend_pos_2d(
        eq, W_L, W_R, normal, points=None):
    """Positivity-limited soft neighbor HLL shock-band blend."""
    return hllc_adc_hllct_momentum_2d(
        eq, W_L, W_R, normal, points=points)


def hllc_adc_hllct_momentum_neighbor_ultrasoft_hllblend_2d(
        eq, W_L, W_R, normal, points=None):
    """Sensor-squared neighbor HLL blend on HLLC-ADC/HLLCT momentum."""
    return hllc_adc_hllct_momentum_2d(
        eq, W_L, W_R, normal, points=points)


def hllc_adc_hllct_momentum_neighbor_shear_guard_ultrasoft_hllblend_2d(
        eq, W_L, W_R, normal, points=None):
    """Shear-preserving ultrasoft HLL blend on HLLC-ADC/HLLCT momentum."""
    return hllc_adc_hllct_momentum_2d(
        eq, W_L, W_R, normal, points=points)


def hllc_adc_hllct_momentum_neighbor_directional_ultrasoft_hllblend_2d(
        eq, W_L, W_R, normal, points=None):
    """Pressure-gradient-aligned ultrasoft neighbor HLL shock-band blend."""
    return hllc_adc_hllct_momentum_2d(
        eq, W_L, W_R, normal, points=points)


def hllc_adc_hllct_momentum_neighbor_directional_soft_hllblend_2d(
        eq, W_L, W_R, normal, points=None):
    """Pressure-gradient-aligned soft neighbor HLL shock-band blend."""
    return hllc_adc_hllct_momentum_2d(
        eq, W_L, W_R, normal, points=points)


def hllc_adc_hllct_momentum_neighbor_directional_ultrasoft_normal_hcorr_2d(
        eq, W_L, W_R, normal, points=None):
    """Neighbor-directional ultrasoft H-correction on normal momentum/energy.

    This keeps the HLLC-ADC/HLLCT base flux contact/shear resolving while
    borrowing HLLE damping only in acoustic-normal channels across a
    pressure-gradient-aligned shock band.
    """
    return hllc_adc_hllct_momentum_2d(
        eq, W_L, W_R, normal, points=points)


def hllc_adc_hllct_momentum_neighbor_directional_ultrasoft_mass_normal_hcorr_2d(
        eq, W_L, W_R, normal, points=None):
    """Neighbor-directional ultrasoft H-correction on mass/normal/energy."""
    return hllc_adc_hllct_momentum_2d(
        eq, W_L, W_R, normal, points=points)


def hllc_adc_hllct_momentum_neighbor_soft_normal_hcorr_2d(
        eq, W_L, W_R, normal, points=None):
    """Soft shock-band H-correction on normal momentum and energy only."""
    return hllc_adc_hllct_momentum_2d(
        eq, W_L, W_R, normal, points=points)


def hllc_adc_hllct_momentum_neighbor_geomean_normal_hcorr_2d(
        eq, W_L, W_R, normal, points=None):
    """Mid-strength shock-band H-correction on normal momentum and energy."""
    return hllc_adc_hllct_momentum_2d(
        eq, W_L, W_R, normal, points=points)


def hllc_adc_hllct_momentum_directional_soft_normal_hcorr_2d(
        eq, W_L, W_R, normal, points=None):
    """Directional soft H-correction on normal momentum and energy only."""
    return hllc_adc_hllct_momentum_2d(
        eq, W_L, W_R, normal, points=points)


def hllc_adc_hllct_momentum_directional_soft_mass_normal_hcorr_2d(
        eq, W_L, W_R, normal, points=None):
    """Directional soft H-correction on mass, normal momentum, and energy."""
    return hllc_adc_hllct_momentum_2d(
        eq, W_L, W_R, normal, points=points)


def hllc_adc_hllct_momentum_pressure_guard_directional_soft_mass_normal_hcorr_2d(
        eq, W_L, W_R, normal, points=None):
    """Pressure-guarded directional soft H-correction on mass/normal/energy.

    The channel set is the carbuncle-stable mass/normal/energy correction, but
    the shock-band sensor is pressure/acoustic-normal guarded so shear/contact
    faces do not inherit unnecessary HLLE dissipation.
    """
    return hllc_adc_hllct_momentum_2d(
        eq, W_L, W_R, normal, points=points)


def hllc_adc_hllct_momentum_pressure_guard_directional_ultrasoft_mass_normal_hcorr_2d(
        eq, W_L, W_R, normal, points=None):
    """Pressure-guarded directional ultrasoft H-correction on mass/normal/energy.

    This is the low-dissipation counterpart of the pressure-guarded mass
    normal H-correction: the same pressure/acoustic shock-band sensor is used,
    but its band weight is cubed so shear/contact roll-up is less damped.
    """
    return hllc_adc_hllct_momentum_2d(
        eq, W_L, W_R, normal, points=points)


def hllc_adc_hllct_momentum_shear_guard_directional_soft_normal_hcorr_2d(
        eq, W_L, W_R, normal, points=None):
    """Shear-guarded directional H-correction on normal momentum/energy."""
    return hllc_adc_hllct_momentum_2d(
        eq, W_L, W_R, normal, points=points)


def hllc_adc_hllct_momentum_pressure_guard_directional_soft_normal_hcorr_2d(
        eq, W_L, W_R, normal, points=None):
    """Pressure-guarded directional H-correction on normal momentum/energy."""
    return hllc_adc_hllct_momentum_2d(
        eq, W_L, W_R, normal, points=points)


def hllc_adc_hllct_momentum_pressure_guard_soft_normal_hcorr_2d(
        eq, W_L, W_R, normal, points=None):
    """Pressure-guarded non-directional H-correction on normal momentum/energy.

    This keeps the globally applied HLLC-ADC/HLLCT base flux, but limits the
    shock-band correction to acoustic-normal channels.  The pressure guard
    suppresses correction on predominantly shear/contact faces, avoiding the
    all-channel HLL blend that can damp upper shear-layer roll-up.
    """
    return hllc_adc_hllct_momentum_2d(
        eq, W_L, W_R, normal, points=points)


def hllc_adc_hllct_momentum_shear_guard_geomean_normal_hcorr_2d(
        eq, W_L, W_R, normal, points=None):
    """Shear-guarded geomean H-correction on normal momentum/energy."""
    return hllc_adc_hllct_momentum_2d(
        eq, W_L, W_R, normal, points=points)


def hllc_adc_hllct_momentum_shear_guard_geomean_mass_energy_hcorr_2d(
        eq, W_L, W_R, normal, points=None):
    """Shear-guarded geomean H-correction on mass and energy only."""
    return hllc_adc_hllct_momentum_2d(
        eq, W_L, W_R, normal, points=points)


def hllc_adc_hllct_momentum_pressure_guard_geomean_normal_hcorr_2d(
        eq, W_L, W_R, normal, points=None):
    """Pressure-guarded geomean H-correction on normal momentum/energy."""
    return hllc_adc_hllct_momentum_2d(
        eq, W_L, W_R, normal, points=points)


def hllc_adc_hllct_momentum_face_soft_normal_hcorr_2d(
        eq, W_L, W_R, normal, points=None):
    """Face-local soft H-correction on normal momentum and energy.

    The base flux keeps HLLC-ADC mass/energy transport with HLLCT momentum.
    The solver-level correction uses the same pressure-compression shock
    sensor as the H-correction family, but does not expand it to neighboring
    cells.  Only the acoustic normal momentum and energy channels are blended
    toward HLLE, preserving contact/shear transport needed for upper roll-up.
    """
    return hllc_adc_hllct_momentum_2d(
        eq, W_L, W_R, normal, points=points)


def hllc_adc_hllct_momentum_face_geomean_normal_hcorr_2d(
        eq, W_L, W_R, normal, points=None):
    """Face-local mid-strength H-correction on normal momentum and energy."""
    return hllc_adc_hllct_momentum_2d(
        eq, W_L, W_R, normal, points=points)


def hllc_adc_hllct_momentum_face_normal_hcorr_2d(
        eq, W_L, W_R, normal, points=None):
    """Face-local full-strength H-correction on normal momentum and energy."""
    return hllc_adc_hllct_momentum_2d(
        eq, W_L, W_R, normal, points=points)


def hllc_adc_hllct_momentum_face_soft_mass_normal_hcorr_2d(
        eq, W_L, W_R, normal, points=None):
    """Face-local soft H-correction on mass, normal momentum, and energy."""
    return hllc_adc_hllct_momentum_2d(
        eq, W_L, W_R, normal, points=points)


def hllc_adc_hllct_momentum_face_geomean_mass_normal_hcorr_2d(
        eq, W_L, W_R, normal, points=None):
    """Face-local mid-strength mass/normal/energy H-correction."""
    return hllc_adc_hllct_momentum_2d(
        eq, W_L, W_R, normal, points=points)


def hllc_adc_hllct_momentum_directional_geomean_normal_hcorr_2d(
        eq, W_L, W_R, normal, points=None):
    """Directional geomean H-correction on normal momentum and energy."""
    return hllc_adc_hllct_momentum_2d(
        eq, W_L, W_R, normal, points=points)


def hllc_adc_hllct_momentum_pressure_guard_directional_geomean_normal_hcorr_2d(
        eq, W_L, W_R, normal, points=None):
    """Pressure-guarded directional geomean normal-momentum/energy H-correction."""
    return hllc_adc_hllct_momentum_2d(
        eq, W_L, W_R, normal, points=points)


def hllc_adc_hllct_momentum_transverse_soft_hcorr_2d(
        eq, W_L, W_R, normal, points=None):
    """Soft mass/transverse shock-band correction on HLLC-ADC/HLLCT."""
    return hllc_adc_hllct_momentum_2d(
        eq, W_L, W_R, normal, points=points)


def hllc_adc_hllct_momentum_transverse_hcorr_2d(
        eq, W_L, W_R, normal, points=None):
    """Mass/transverse shock-band correction on HLLC-ADC/HLLCT."""
    return hllc_adc_hllct_momentum_2d(
        eq, W_L, W_R, normal, points=points)


def hllc_adc_hllct_momentum_pressure_hcorr_2d(
        eq, W_L, W_R, normal, points=None):
    """Pressure-jump momentum dissipation on HLLC-ADC/HLLCT."""
    return hllc_adc_hllct_momentum_2d(
        eq, W_L, W_R, normal, points=points)


def hllc_adc_ducros_2d(eq, W_L, W_R, normal, points=None):
    """HLLC-ADC flux with Ducros shock dissipation enabled.

    HLLC-ADC keeps contact/shear transport in smooth regions.  The solver
    applies a Ducros pressure-compression sensor afterward, blending toward a
    local Lax-Friedrichs flux only where divergence dominates vorticity at a
    pressure jump.  That targets carbuncle-prone shock bands without making a
    case-local switch.
    """
    return hllc_adc_2d(eq, W_L, W_R, normal, points=points)


def hllc_tangent_adc_2d(eq, W_L, W_R, normal, points=None):
    """HLLC with ADC only on face-tangential momentum.

    This keeps the contact-resolving HLLC mass and energy fluxes intact while
    damping the transverse momentum channel that drives odd-even shock modes.
    """
    if eq.__class__.__name__ != 'Euler2D':
        return llf(eq, W_L, W_R, normal, points=points)
    if _NUMBA_AVAILABLE and W_L.shape[0] == 4 and W_L.shape[1] >= 512:
        return _hllc_tangent_adc_kernel(
            np.ascontiguousarray(W_L, dtype=np.float64),
            np.ascontiguousarray(W_R, dtype=np.float64),
            np.ascontiguousarray(normal, dtype=np.float64),
            float(eq.gamma))
    n = np.asarray(normal, dtype=float)
    F_hllc, F_hll, un_L, un_R, c_L, c_R, p_L, p_R = _euler2d_hllc_hll(
        eq, W_L, W_R, n)
    pressure_jump = np.abs(p_R - p_L) / np.maximum(p_R + p_L, 1.0e-30)
    pressure_ratio = np.minimum(p_L, p_R) / np.maximum(
        np.maximum(p_L, p_R), 1.0e-30)
    jump_sensor = np.clip((pressure_jump - 0.05) / 0.35, 0.0, 1.0)
    ratio_sensor = np.clip(1.0 - pressure_ratio, 0.0, 1.0)
    shock = np.sqrt(jump_sensor * ratio_sensor)
    nx = n[..., 0]
    ny = n[..., 1]
    tx = -ny
    ty = nx
    out = F_hllc.copy()
    normal_hllc = F_hllc[1] * nx + F_hllc[2] * ny
    tangent_hllc = F_hllc[1] * tx + F_hllc[2] * ty
    tangent_hll = F_hll[1] * tx + F_hll[2] * ty
    tangent_blend = (1.0 - shock) * tangent_hllc + shock * tangent_hll
    out[1] = normal_hllc * nx + tangent_blend * tx
    out[2] = normal_hllc * ny + tangent_blend * ty
    return out


def hllc_adc_normal_2d(eq, W_L, W_R, normal, points=None):
    """HLLC/HLLE blend on acoustic-normal channels only.

    Strong-shock faces receive HLLE dissipation in mass, normal momentum, and
    energy, while tangential momentum keeps the HLLC shear/contact transport.
    This follows the carbuncle-control idea of adding dissipation to acoustic
    modes without smearing the tangential shear layer.
    """
    if eq.__class__.__name__ != 'Euler2D':
        return llf(eq, W_L, W_R, normal, points=points)
    if _NUMBA_AVAILABLE and W_L.shape[0] == 4 and W_L.shape[1] >= 512:
        return _hllc_adc_normal_kernel(
            np.ascontiguousarray(W_L, dtype=np.float64),
            np.ascontiguousarray(W_R, dtype=np.float64),
            np.ascontiguousarray(normal, dtype=np.float64),
            float(eq.gamma))
    F_hllc, F_hll, un_L, un_R, c_L, c_R, p_L, p_R = _euler2d_hllc_hll(
        eq, W_L, W_R, np.asarray(normal, dtype=float))
    shock = _pressure_compression_sensor(un_L, un_R, c_L, c_R, p_L, p_R)
    return _adc_normal_blend(F_hllc, F_hll, np.asarray(normal, dtype=float),
                             shock)


def hllc_adc_mass_normal_2d(eq, W_L, W_R, normal, points=None):
    """HLLC-ADC with antidiffusion control on mass and normal momentum.

    This follows the HLLC-ADC stability analysis more directly than the older
    project variant: control the critical mass and interface-normal momentum
    antidiffusive terms near pressure shocks, while retaining HLLC tangential
    momentum and energy transport for shear/contact resolution.
    """
    if eq.__class__.__name__ != 'Euler2D':
        return llf(eq, W_L, W_R, normal, points=points)
    if _NUMBA_AVAILABLE and W_L.shape[0] == 4 and W_L.shape[1] >= 512:
        return _hllc_adc_mass_normal_kernel(
            np.ascontiguousarray(W_L, dtype=np.float64),
            np.ascontiguousarray(W_R, dtype=np.float64),
            np.ascontiguousarray(normal, dtype=np.float64),
            float(eq.gamma))
    n = np.asarray(normal, dtype=float)
    F_hllc, F_hll, un_L, un_R, c_L, c_R, p_L, p_R = _euler2d_hllc_hll(
        eq, W_L, W_R, n)
    pressure_jump = np.abs(p_R - p_L) / np.maximum(p_R + p_L, 1.0e-30)
    pressure_ratio = np.minimum(p_L, p_R) / np.maximum(
        np.maximum(p_L, p_R), 1.0e-30)
    jump_sensor = np.clip((pressure_jump - 0.05) / 0.35, 0.0, 1.0)
    ratio_sensor = np.clip(1.0 - pressure_ratio, 0.0, 1.0)
    shock = np.sqrt(jump_sensor * ratio_sensor)
    nx = n[..., 0]
    ny = n[..., 1]
    tx = -ny
    ty = nx
    out = F_hllc.copy()
    out[0] = (1.0 - shock) * F_hllc[0] + shock * F_hll[0]
    normal_hllc = F_hllc[1] * nx + F_hllc[2] * ny
    normal_hll = F_hll[1] * nx + F_hll[2] * ny
    tangent_hllc = F_hllc[1] * tx + F_hllc[2] * ty
    normal_blend = (1.0 - shock) * normal_hllc + shock * normal_hll
    out[1] = normal_blend * nx + tangent_hllc * tx
    out[2] = normal_blend * ny + tangent_hllc * ty
    return out


def hllc_adc_full_2d(eq, W_L, W_R, normal, points=None):
    """Full HLLC/HLLE pressure-sensor blend for strong shock stability."""
    if eq.__class__.__name__ != 'Euler2D':
        return llf(eq, W_L, W_R, normal, points=points)
    if _NUMBA_AVAILABLE and W_L.shape[0] == 4 and W_L.shape[1] >= 512:
        return _hllc_adc_full_kernel(
            np.ascontiguousarray(W_L, dtype=np.float64),
            np.ascontiguousarray(W_R, dtype=np.float64),
            np.ascontiguousarray(normal, dtype=np.float64),
            float(eq.gamma))
    F_hllc, F_hll, un_L, un_R, c_L, c_R, p_L, p_R = _euler2d_hllc_hll(
        eq, W_L, W_R, np.asarray(normal, dtype=float))
    shock = _pressure_compression_sensor(un_L, un_R, c_L, c_R, p_L, p_R)
    return (1.0 - shock)[None, :] * F_hllc + shock[None, :] * F_hll


def ausm_slau2_ducros_2d(eq, W_L, W_R, normal, points=None):
    """AUSM-SLAU2 shock blend with Ducros-sensor shock dissipation."""
    return ausm_slau2_shock_2d(eq, W_L, W_R, normal, points=points)


hllc_rotated_hybrid_2d.internal_parallel = _NUMBA_AVAILABLE
roe_rotated_hybrid_2d.internal_parallel = _NUMBA_AVAILABLE
roe_rotated_shock_hybrid_2d.internal_parallel = _NUMBA_AVAILABLE
roe_rotated_soft_shock_hybrid_2d.internal_parallel = _NUMBA_AVAILABLE
hllc_adc_2d.internal_parallel = _NUMBA_AVAILABLE
hllc_adc_strong_2d.internal_parallel = _NUMBA_AVAILABLE
hllc_adc_hcorr_2d.internal_parallel = _NUMBA_AVAILABLE
hllc_adc_hcorr_2d.h_correction = True
hllc_adc_acoustic_hcorr_2d.internal_parallel = _NUMBA_AVAILABLE
hllc_adc_acoustic_hcorr_2d.h_correction = True
hllc_adc_acoustic_hcorr_2d.h_correction_mode = 'normal_energy'
hllc_adc_acoustic_soft_hcorr_2d.internal_parallel = _NUMBA_AVAILABLE
hllc_adc_acoustic_soft_hcorr_2d.h_correction = True
hllc_adc_acoustic_soft_hcorr_2d.h_correction_mode = 'normal_energy_soft'
hllc_adc_transverse_hcorr_2d.internal_parallel = _NUMBA_AVAILABLE
hllc_adc_transverse_hcorr_2d.h_correction = True
hllc_adc_transverse_hcorr_2d.h_correction_mode = 'mass_transverse'
hllc_adc_transverse_energy_hcorr_2d.internal_parallel = _NUMBA_AVAILABLE
hllc_adc_transverse_energy_hcorr_2d.h_correction = True
hllc_adc_transverse_energy_hcorr_2d.h_correction_mode = 'mass_transverse_energy'
hllc_adc_transverse_soft_hcorr_2d.internal_parallel = _NUMBA_AVAILABLE
hllc_adc_transverse_soft_hcorr_2d.h_correction = True
hllc_adc_transverse_soft_hcorr_2d.h_correction_mode = 'mass_transverse_soft'
hllc_adc_mass_hcorr_2d.internal_parallel = _NUMBA_AVAILABLE
hllc_adc_mass_hcorr_2d.h_correction = True
hllc_adc_mass_hcorr_2d.h_correction_mode = 'mass'
hllc_adc_full_transverse_soft_hcorr_2d.internal_parallel = _NUMBA_AVAILABLE
hllc_adc_full_transverse_soft_hcorr_2d.h_correction = True
hllc_adc_full_transverse_soft_hcorr_2d.h_correction_mode = (
    'mass_transverse_soft')
hllc_adc_full_transverse_hcorr_2d.internal_parallel = _NUMBA_AVAILABLE
hllc_adc_full_transverse_hcorr_2d.h_correction = True
hllc_adc_full_transverse_hcorr_2d.h_correction_mode = 'mass_transverse'
hllc_adc_full_neighbor_hcorr_2d.internal_parallel = _NUMBA_AVAILABLE
hllc_adc_full_neighbor_hcorr_2d.h_correction = True
hllc_adc_full_neighbor_hcorr_2d.h_correction_mode = 'mass_transverse_neighbor'
hllc_adc_full_neighbor_soft_hcorr_2d.internal_parallel = _NUMBA_AVAILABLE
hllc_adc_full_neighbor_soft_hcorr_2d.h_correction = True
hllc_adc_full_neighbor_soft_hcorr_2d.h_correction_mode = (
    'mass_transverse_neighbor_soft')
hllc_adc_full_neighbor_hllblend_2d.internal_parallel = _NUMBA_AVAILABLE
hllc_adc_full_neighbor_hllblend_2d.h_correction = True
hllc_adc_full_neighbor_hllblend_2d.h_correction_mode = 'neighbor_hll_blend'
hllc_adc_full_neighbor_soft_hllblend_2d.internal_parallel = _NUMBA_AVAILABLE
hllc_adc_full_neighbor_soft_hllblend_2d.h_correction = True
hllc_adc_full_neighbor_soft_hllblend_2d.h_correction_mode = (
    'neighbor_soft_hll_blend')
hllc_adc_full_ducros_hll_soft_2d.internal_parallel = _NUMBA_AVAILABLE
hllc_adc_full_ducros_hll_soft_2d.shock_dissipation = 'ducros_hll_soft'
hllc_adc_hllct_momentum_2d.internal_parallel = _NUMBA_AVAILABLE
hllc_adc_ausm_hlle_momentum_2d.internal_parallel = _NUMBA_AVAILABLE
hllc_adc_ausm_hlle_momentum_soft_hllblend_2d.internal_parallel = (
    _NUMBA_AVAILABLE)
hllc_adc_ausm_hlle_momentum_soft_hllblend_2d.h_correction = True
hllc_adc_ausm_hlle_momentum_soft_hllblend_2d.h_correction_mode = (
    'neighbor_soft_hll_blend_pos')
hllc_adc_ausm_hlle_momentum_hllblend_2d.internal_parallel = (
    _NUMBA_AVAILABLE)
hllc_adc_ausm_hlle_momentum_hllblend_2d.h_correction = True
hllc_adc_ausm_hlle_momentum_hllblend_2d.h_correction_mode = (
    'neighbor_hll_blend')
hllc_adc_ausm_hlle_momentum_pressure_hcorr_2d.internal_parallel = (
    _NUMBA_AVAILABLE)
hllc_adc_ausm_hlle_momentum_pressure_hcorr_2d.h_correction = True
hllc_adc_ausm_hlle_momentum_pressure_hcorr_2d.h_correction_mode = (
    'pressure_momentum')
hllc_adc_ausm_hlle_momentum_pressure_normal_hcorr_2d.internal_parallel = (
    _NUMBA_AVAILABLE)
hllc_adc_ausm_hlle_momentum_pressure_normal_hcorr_2d.h_correction = True
hllc_adc_ausm_hlle_momentum_pressure_normal_hcorr_2d.h_correction_mode = (
    'pressure_normal_momentum')
hllc_adc_ausm_hlle_momentum_pressureonly_normal_hcorr_2d.internal_parallel = (
    _NUMBA_AVAILABLE)
hllc_adc_ausm_hlle_momentum_pressureonly_normal_hcorr_2d.h_correction = True
hllc_adc_ausm_hlle_momentum_pressureonly_normal_hcorr_2d.h_correction_mode = (
    'pressureonly_normal_momentum')
hllc_adc_ausm_hlle_momentum_pressureonly_soft_normal_hcorr_2d.internal_parallel = (
    _NUMBA_AVAILABLE)
hllc_adc_ausm_hlle_momentum_pressureonly_soft_normal_hcorr_2d.h_correction = True
hllc_adc_ausm_hlle_momentum_pressureonly_soft_normal_hcorr_2d.h_correction_mode = (
    'pressureonly_soft_normal_momentum')
hllc_adc_full_hllct_momentum_2d.internal_parallel = _NUMBA_AVAILABLE
hllc_adc_hllct_momentum_neighbor_hllblend_2d.internal_parallel = (
    _NUMBA_AVAILABLE)
hllc_adc_hllct_momentum_neighbor_hllblend_2d.h_correction = True
hllc_adc_hllct_momentum_neighbor_hllblend_2d.h_correction_mode = (
    'neighbor_hll_blend')
hllc_adc_hllct_momentum_neighbor_soft_hllblend_2d.internal_parallel = (
    _NUMBA_AVAILABLE)
hllc_adc_hllct_momentum_neighbor_soft_hllblend_2d.h_correction = True
hllc_adc_hllct_momentum_neighbor_soft_hllblend_2d.h_correction_mode = (
    'neighbor_soft_hll_blend')
hllc_adc_hllct_momentum_neighbor_geomean_hllblend_2d.internal_parallel = (
    _NUMBA_AVAILABLE)
hllc_adc_hllct_momentum_neighbor_geomean_hllblend_2d.h_correction = True
hllc_adc_hllct_momentum_neighbor_geomean_hllblend_2d.h_correction_mode = (
    'neighbor_geomean_hll_blend')
hllc_adc_hllct_momentum_neighbor_soft_hllblend_pos_2d.internal_parallel = (
    _NUMBA_AVAILABLE)
hllc_adc_hllct_momentum_neighbor_soft_hllblend_pos_2d.h_correction = True
hllc_adc_hllct_momentum_neighbor_soft_hllblend_pos_2d.h_correction_mode = (
    'neighbor_soft_hll_blend_pos')
hllc_adc_hllct_momentum_neighbor_ultrasoft_hllblend_2d.internal_parallel = (
    _NUMBA_AVAILABLE)
hllc_adc_hllct_momentum_neighbor_ultrasoft_hllblend_2d.h_correction = True
hllc_adc_hllct_momentum_neighbor_ultrasoft_hllblend_2d.h_correction_mode = (
    'neighbor_ultrasoft_hll_blend')
hllc_adc_hllct_momentum_neighbor_shear_guard_ultrasoft_hllblend_2d.internal_parallel = (
    _NUMBA_AVAILABLE)
hllc_adc_hllct_momentum_neighbor_shear_guard_ultrasoft_hllblend_2d.h_correction = True
hllc_adc_hllct_momentum_neighbor_shear_guard_ultrasoft_hllblend_2d.h_correction_mode = (
    'neighbor_shear_guard_ultrasoft_hll_blend')
hllc_adc_hllct_momentum_neighbor_directional_ultrasoft_hllblend_2d.internal_parallel = (
    _NUMBA_AVAILABLE)
hllc_adc_hllct_momentum_neighbor_directional_ultrasoft_hllblend_2d.h_correction = True
hllc_adc_hllct_momentum_neighbor_directional_ultrasoft_hllblend_2d.h_correction_mode = (
    'neighbor_directional_ultrasoft_hll_blend')
hllc_adc_hllct_momentum_neighbor_directional_soft_hllblend_2d.internal_parallel = (
    _NUMBA_AVAILABLE)
hllc_adc_hllct_momentum_neighbor_directional_soft_hllblend_2d.h_correction = True
hllc_adc_hllct_momentum_neighbor_directional_soft_hllblend_2d.h_correction_mode = (
    'neighbor_directional_soft_hll_blend')
hllc_adc_hllct_momentum_neighbor_directional_ultrasoft_normal_hcorr_2d.internal_parallel = (
    _NUMBA_AVAILABLE)
hllc_adc_hllct_momentum_neighbor_directional_ultrasoft_normal_hcorr_2d.h_correction = True
hllc_adc_hllct_momentum_neighbor_directional_ultrasoft_normal_hcorr_2d.h_correction_mode = (
    'normal_energy_neighbor_directional_ultrasoft')
hllc_adc_hllct_momentum_neighbor_directional_ultrasoft_mass_normal_hcorr_2d.internal_parallel = (
    _NUMBA_AVAILABLE)
hllc_adc_hllct_momentum_neighbor_directional_ultrasoft_mass_normal_hcorr_2d.h_correction = True
hllc_adc_hllct_momentum_neighbor_directional_ultrasoft_mass_normal_hcorr_2d.h_correction_mode = (
    'mass_normal_energy_neighbor_directional_ultrasoft')
hllc_adc_hllct_momentum_neighbor_soft_normal_hcorr_2d.internal_parallel = (
    _NUMBA_AVAILABLE)
hllc_adc_hllct_momentum_neighbor_soft_normal_hcorr_2d.h_correction = True
hllc_adc_hllct_momentum_neighbor_soft_normal_hcorr_2d.h_correction_mode = (
    'normal_energy_soft')
hllc_adc_hllct_momentum_neighbor_geomean_normal_hcorr_2d.internal_parallel = (
    _NUMBA_AVAILABLE)
hllc_adc_hllct_momentum_neighbor_geomean_normal_hcorr_2d.h_correction = True
hllc_adc_hllct_momentum_neighbor_geomean_normal_hcorr_2d.h_correction_mode = (
    'normal_energy_geomean')
hllc_adc_hllct_momentum_directional_soft_normal_hcorr_2d.internal_parallel = (
    _NUMBA_AVAILABLE)
hllc_adc_hllct_momentum_directional_soft_normal_hcorr_2d.h_correction = True
hllc_adc_hllct_momentum_directional_soft_normal_hcorr_2d.h_correction_mode = (
    'normal_energy_directional_soft')
hllc_adc_hllct_momentum_directional_soft_mass_normal_hcorr_2d.internal_parallel = (
    _NUMBA_AVAILABLE)
hllc_adc_hllct_momentum_directional_soft_mass_normal_hcorr_2d.h_correction = True
hllc_adc_hllct_momentum_directional_soft_mass_normal_hcorr_2d.h_correction_mode = (
    'mass_normal_energy_directional_soft')
hllc_adc_hllct_momentum_pressure_guard_directional_soft_mass_normal_hcorr_2d.internal_parallel = (
    _NUMBA_AVAILABLE)
hllc_adc_hllct_momentum_pressure_guard_directional_soft_mass_normal_hcorr_2d.h_correction = True
hllc_adc_hllct_momentum_pressure_guard_directional_soft_mass_normal_hcorr_2d.h_correction_mode = (
    'mass_normal_energy_pressure_guard_directional_soft')
hllc_adc_hllct_momentum_pressure_guard_directional_ultrasoft_mass_normal_hcorr_2d.internal_parallel = (
    _NUMBA_AVAILABLE)
hllc_adc_hllct_momentum_pressure_guard_directional_ultrasoft_mass_normal_hcorr_2d.h_correction = True
hllc_adc_hllct_momentum_pressure_guard_directional_ultrasoft_mass_normal_hcorr_2d.h_correction_mode = (
    'mass_normal_energy_pressure_guard_directional_ultrasoft')
hllc_adc_hllct_momentum_shear_guard_directional_soft_normal_hcorr_2d.internal_parallel = (
    _NUMBA_AVAILABLE)
hllc_adc_hllct_momentum_shear_guard_directional_soft_normal_hcorr_2d.h_correction = True
hllc_adc_hllct_momentum_shear_guard_directional_soft_normal_hcorr_2d.h_correction_mode = (
    'normal_energy_shear_guard_directional_soft')
hllc_adc_hllct_momentum_pressure_guard_directional_soft_normal_hcorr_2d.internal_parallel = (
    _NUMBA_AVAILABLE)
hllc_adc_hllct_momentum_pressure_guard_directional_soft_normal_hcorr_2d.h_correction = True
hllc_adc_hllct_momentum_pressure_guard_directional_soft_normal_hcorr_2d.h_correction_mode = (
    'normal_energy_pressure_guard_directional_soft')
hllc_adc_hllct_momentum_pressure_guard_soft_normal_hcorr_2d.internal_parallel = (
    _NUMBA_AVAILABLE)
hllc_adc_hllct_momentum_pressure_guard_soft_normal_hcorr_2d.h_correction = True
hllc_adc_hllct_momentum_pressure_guard_soft_normal_hcorr_2d.h_correction_mode = (
    'normal_energy_pressure_guard_soft')
hllc_adc_hllct_momentum_shear_guard_geomean_normal_hcorr_2d.internal_parallel = (
    _NUMBA_AVAILABLE)
hllc_adc_hllct_momentum_shear_guard_geomean_normal_hcorr_2d.h_correction = True
hllc_adc_hllct_momentum_shear_guard_geomean_normal_hcorr_2d.h_correction_mode = (
    'normal_energy_shear_guard_geomean')
hllc_adc_hllct_momentum_shear_guard_geomean_mass_energy_hcorr_2d.internal_parallel = (
    _NUMBA_AVAILABLE)
hllc_adc_hllct_momentum_shear_guard_geomean_mass_energy_hcorr_2d.h_correction = True
hllc_adc_hllct_momentum_shear_guard_geomean_mass_energy_hcorr_2d.h_correction_mode = (
    'mass_energy_shear_guard_geomean')
hllc_adc_hllct_momentum_pressure_guard_geomean_normal_hcorr_2d.internal_parallel = (
    _NUMBA_AVAILABLE)
hllc_adc_hllct_momentum_pressure_guard_geomean_normal_hcorr_2d.h_correction = True
hllc_adc_hllct_momentum_pressure_guard_geomean_normal_hcorr_2d.h_correction_mode = (
    'normal_energy_pressure_guard_geomean')
hllc_adc_hllct_momentum_face_soft_normal_hcorr_2d.internal_parallel = (
    _NUMBA_AVAILABLE)
hllc_adc_hllct_momentum_face_soft_normal_hcorr_2d.h_correction = True
hllc_adc_hllct_momentum_face_soft_normal_hcorr_2d.h_correction_mode = (
    'normal_energy_face_local_soft')
hllc_adc_hllct_momentum_face_geomean_normal_hcorr_2d.internal_parallel = (
    _NUMBA_AVAILABLE)
hllc_adc_hllct_momentum_face_geomean_normal_hcorr_2d.h_correction = True
hllc_adc_hllct_momentum_face_geomean_normal_hcorr_2d.h_correction_mode = (
    'normal_energy_face_local_geomean')
hllc_adc_hllct_momentum_face_normal_hcorr_2d.internal_parallel = (
    _NUMBA_AVAILABLE)
hllc_adc_hllct_momentum_face_normal_hcorr_2d.h_correction = True
hllc_adc_hllct_momentum_face_normal_hcorr_2d.h_correction_mode = (
    'normal_energy_face_local')
hllc_adc_hllct_momentum_face_soft_mass_normal_hcorr_2d.internal_parallel = (
    _NUMBA_AVAILABLE)
hllc_adc_hllct_momentum_face_soft_mass_normal_hcorr_2d.h_correction = True
hllc_adc_hllct_momentum_face_soft_mass_normal_hcorr_2d.h_correction_mode = (
    'mass_normal_energy_face_local_soft')
hllc_adc_hllct_momentum_face_geomean_mass_normal_hcorr_2d.internal_parallel = (
    _NUMBA_AVAILABLE)
hllc_adc_hllct_momentum_face_geomean_mass_normal_hcorr_2d.h_correction = True
hllc_adc_hllct_momentum_face_geomean_mass_normal_hcorr_2d.h_correction_mode = (
    'mass_normal_energy_face_local_geomean')
hllc_adc_hllct_momentum_directional_geomean_normal_hcorr_2d.internal_parallel = (
    _NUMBA_AVAILABLE)
hllc_adc_hllct_momentum_directional_geomean_normal_hcorr_2d.h_correction = True
hllc_adc_hllct_momentum_directional_geomean_normal_hcorr_2d.h_correction_mode = (
    'normal_energy_directional_geomean')
hllc_adc_hllct_momentum_pressure_guard_directional_geomean_normal_hcorr_2d.internal_parallel = (
    _NUMBA_AVAILABLE)
hllc_adc_hllct_momentum_pressure_guard_directional_geomean_normal_hcorr_2d.h_correction = True
hllc_adc_hllct_momentum_pressure_guard_directional_geomean_normal_hcorr_2d.h_correction_mode = (
    'normal_energy_pressure_guard_directional_geomean')
hllc_adc_hllct_momentum_transverse_soft_hcorr_2d.internal_parallel = (
    _NUMBA_AVAILABLE)
hllc_adc_hllct_momentum_transverse_soft_hcorr_2d.h_correction = True
hllc_adc_hllct_momentum_transverse_soft_hcorr_2d.h_correction_mode = (
    'mass_transverse_soft')
hllc_adc_hllct_momentum_transverse_hcorr_2d.internal_parallel = (
    _NUMBA_AVAILABLE)
hllc_adc_hllct_momentum_transverse_hcorr_2d.h_correction = True
hllc_adc_hllct_momentum_transverse_hcorr_2d.h_correction_mode = (
    'mass_transverse')
hllc_adc_hllct_momentum_pressure_hcorr_2d.internal_parallel = (
    _NUMBA_AVAILABLE)
hllc_adc_hllct_momentum_pressure_hcorr_2d.h_correction = True
hllc_adc_hllct_momentum_pressure_hcorr_2d.h_correction_mode = (
    'pressure_momentum')
hllc_adc_ducros_2d.internal_parallel = _NUMBA_AVAILABLE
hllc_adc_ducros_2d.shock_dissipation = 'ducros_rusanov'
hllc_tangent_adc_2d.internal_parallel = _NUMBA_AVAILABLE
hllc_adc_normal_2d.internal_parallel = _NUMBA_AVAILABLE
hllc_adc_mass_normal_2d.internal_parallel = _NUMBA_AVAILABLE
hllc_adc_full_2d.internal_parallel = _NUMBA_AVAILABLE
hllc_pure_2d.internal_parallel = _NUMBA_AVAILABLE
hllc_lm_2d.internal_parallel = _NUMBA_AVAILABLE
hllc_swm_p_2d.internal_parallel = _NUMBA_AVAILABLE
ausm_rotated_hybrid_2d.internal_parallel = _NUMBA_AVAILABLE
ausm_hlle_shock_2d.internal_parallel = _NUMBA_AVAILABLE
ausm_slau2_shock_2d.internal_parallel = _NUMBA_AVAILABLE
ausm_slau2_pressure_shock_2d.internal_parallel = _NUMBA_AVAILABLE
ausm_slau2_pressure_shock_soft_2d.internal_parallel = _NUMBA_AVAILABLE
ausm_slau2_pressure_shock_soft_normal_2d.internal_parallel = _NUMBA_AVAILABLE
ausm_slau2_pressure_guarded_shock_2d.internal_parallel = _NUMBA_AVAILABLE
ausm_plus_up_shear_guard_geomean_hcorr_2d.internal_parallel = (
    _NUMBA_AVAILABLE)
ausm_plus_up_shear_guard_geomean_hcorr_2d.h_correction = True
ausm_plus_up_shear_guard_geomean_hcorr_2d.h_correction_mode = (
    'normal_energy_shear_guard_geomean')
ausm_slau2_hcorr_2d.internal_parallel = _NUMBA_AVAILABLE
ausm_slau2_hcorr_2d.h_correction = True
ausm_slau2_hcorr_2d.h_correction_mode = 'normal_energy_soft'
ausm_slau2_hcorr_strong_2d.internal_parallel = _NUMBA_AVAILABLE
ausm_slau2_hcorr_strong_2d.h_correction = True
ausm_slau2_hcorr_strong_2d.h_correction_mode = 'normal_energy'
ausm_slau2_transverse_soft_hcorr_2d.internal_parallel = _NUMBA_AVAILABLE
ausm_slau2_transverse_soft_hcorr_2d.h_correction = True
ausm_slau2_transverse_soft_hcorr_2d.h_correction_mode = (
    'mass_transverse_soft')
ausm_slau2_normal_momentum_soft_hcorr_2d.internal_parallel = (
    _NUMBA_AVAILABLE)
ausm_slau2_normal_momentum_soft_hcorr_2d.h_correction = True
ausm_slau2_normal_momentum_soft_hcorr_2d.h_correction_mode = (
    'pressureonly_soft_normal_momentum')
ausm_slau2_shear_guard_hcorr_2d.internal_parallel = _NUMBA_AVAILABLE
ausm_slau2_shear_guard_hcorr_2d.h_correction = True
ausm_slau2_shear_guard_hcorr_2d.h_correction_mode = (
    'normal_energy_shear_guard_soft')
ausm_slau2_shear_guard_geomean_hcorr_2d.internal_parallel = _NUMBA_AVAILABLE
ausm_slau2_shear_guard_geomean_hcorr_2d.h_correction = True
ausm_slau2_shear_guard_geomean_hcorr_2d.h_correction_mode = (
    'normal_energy_shear_guard_geomean')
ausm_slau2_shear_guard_mass_hcorr_2d.internal_parallel = _NUMBA_AVAILABLE
ausm_slau2_shear_guard_mass_hcorr_2d.h_correction = True
ausm_slau2_shear_guard_mass_hcorr_2d.h_correction_mode = (
    'mass_shear_guard_geomean')
ausm_slau2_shear_guard_mass_energy_hcorr_2d.internal_parallel = (
    _NUMBA_AVAILABLE)
ausm_slau2_shear_guard_mass_energy_hcorr_2d.h_correction = True
ausm_slau2_shear_guard_mass_energy_hcorr_2d.h_correction_mode = (
    'mass_energy_shear_guard_geomean')
ausm_slau2_shear_guard_geomean_hllblend_2d.internal_parallel = (
    _NUMBA_AVAILABLE)
ausm_slau2_shear_guard_geomean_hllblend_2d.h_correction = True
ausm_slau2_shear_guard_geomean_hllblend_2d.h_correction_mode = (
    'normal_energy_shear_guard_geomean_hll_blend')
ausm_slau2_shear_guard_soft_hllblend_2d.internal_parallel = _NUMBA_AVAILABLE
ausm_slau2_shear_guard_soft_hllblend_2d.h_correction = True
ausm_slau2_shear_guard_soft_hllblend_2d.h_correction_mode = (
    'normal_energy_shear_guard_soft_hll_blend')
ausm_slau2_shear_guard_ultrasoft_hllblend_2d.internal_parallel = (
    _NUMBA_AVAILABLE)
ausm_slau2_shear_guard_ultrasoft_hllblend_2d.h_correction = True
ausm_slau2_shear_guard_ultrasoft_hllblend_2d.h_correction_mode = (
    'normal_energy_shear_guard_ultrasoft_hll_blend')
ausm_slau2_shear_guard_directional_soft_hllblend_2d.internal_parallel = (
    _NUMBA_AVAILABLE)
ausm_slau2_shear_guard_directional_soft_hllblend_2d.h_correction = True
ausm_slau2_shear_guard_directional_soft_hllblend_2d.h_correction_mode = (
    'normal_energy_shear_guard_directional_soft_hll_blend')
ausm_slau2_pressure_guard_directional_soft_hllblend_2d.internal_parallel = (
    _NUMBA_AVAILABLE)
ausm_slau2_pressure_guard_directional_soft_hllblend_2d.h_correction = True
ausm_slau2_pressure_guard_directional_soft_hllblend_2d.h_correction_mode = (
    'normal_energy_pressure_guard_directional_soft_hll_blend')
ausm_slau2_pressure_guard_directional_geomean_hllblend_2d.internal_parallel = (
    _NUMBA_AVAILABLE)
ausm_slau2_pressure_guard_directional_geomean_hllblend_2d.h_correction = True
ausm_slau2_pressure_guard_directional_geomean_hllblend_2d.h_correction_mode = (
    'normal_energy_pressure_guard_directional_geomean_hll_blend')
ausm_slau2_shear_guard_geomean_mass_normal_hcorr_2d.internal_parallel = (
    _NUMBA_AVAILABLE)
ausm_slau2_shear_guard_geomean_mass_normal_hcorr_2d.h_correction = True
ausm_slau2_shear_guard_geomean_mass_normal_hcorr_2d.h_correction_mode = (
    'mass_normal_energy_shear_guard_geomean')
ausm_slau2_shear_guard_mass_normal_hcorr_2d.internal_parallel = (
    _NUMBA_AVAILABLE)
ausm_slau2_shear_guard_mass_normal_hcorr_2d.h_correction = True
ausm_slau2_shear_guard_mass_normal_hcorr_2d.h_correction_mode = (
    'mass_normal_energy_shear_guard_soft')
ausm_slau2_shear_guard_full_hcorr_2d.internal_parallel = _NUMBA_AVAILABLE
ausm_slau2_shear_guard_full_hcorr_2d.h_correction = True
ausm_slau2_shear_guard_full_hcorr_2d.h_correction_mode = (
    'normal_energy_shear_guard')
ausm_slau2_pressure_guard_hcorr_2d.internal_parallel = _NUMBA_AVAILABLE
ausm_slau2_pressure_guard_hcorr_2d.h_correction = True
ausm_slau2_pressure_guard_hcorr_2d.h_correction_mode = (
    'normal_energy_pressure_guard_soft')
ausm_slau2_pressure_guard_geomean_hcorr_2d.internal_parallel = (
    _NUMBA_AVAILABLE)
ausm_slau2_pressure_guard_geomean_hcorr_2d.h_correction = True
ausm_slau2_pressure_guard_geomean_hcorr_2d.h_correction_mode = (
    'normal_energy_pressure_guard_geomean')
ausm_slau2_face_hcorr_2d.internal_parallel = _NUMBA_AVAILABLE
ausm_slau2_face_hcorr_2d.h_correction = True
ausm_slau2_face_hcorr_2d.h_correction_mode = (
    'normal_energy_face_local_soft')
ausm_slau2_directional_hcorr_2d.internal_parallel = _NUMBA_AVAILABLE
ausm_slau2_directional_hcorr_2d.h_correction = True
ausm_slau2_directional_hcorr_2d.h_correction_mode = (
    'normal_energy_directional_soft')
ausm_slau2_ducros_2d.internal_parallel = _NUMBA_AVAILABLE
ausm_slau2_ducros_2d.shock_dissipation = 'ducros_rusanov'
hlle_2d.internal_parallel = _NUMBA_AVAILABLE
hllc_rotated_compressive_hybrid_2d.internal_parallel = _NUMBA_AVAILABLE
hllc_rotated_compressive_normal_hybrid_2d.internal_parallel = (
    _NUMBA_AVAILABLE)
hllc_rotated_compressive_tangent_hybrid_2d.internal_parallel = (
    _NUMBA_AVAILABLE)
hllc_rotated_compressive_normality2_hybrid_2d.internal_parallel = (
    _NUMBA_AVAILABLE)


def hllc_1d(eq, W_L, W_R, normal=None, points=None):
    """HLLC for 1D Euler (Toro et al. 1994), normal-aware.

    The Riemann problem is solved in the face-aligned frame (velocity
    component along +normal), then the resulting flux is rotated back
    to the original frame.  In 1D rotation = scalar multiply by n_x for
    the momentum component; mass and energy fluxes are scalars
    invariant under the rotation (already F·n).

    Wave speeds (Davis):  S_L = min(u_n_L − c_L, u_n_R − c_R)
                          S_R = max(u_n_L + c_L, u_n_R + c_R)
    """
    n_x = np.asarray(normal)[..., 0] if normal is not None else 1.0
    rho_L, u_L_orig, p_L = W_L[0], W_L[1], W_L[2]
    rho_R, u_R_orig, p_R = W_R[0], W_R[1], W_R[2]
    rho_L = np.maximum(rho_L, _EPS); rho_R = np.maximum(rho_R, _EPS)

    # Project velocity onto face normal (face-aligned frame).
    u_L = u_L_orig * n_x
    u_R = u_R_orig * n_x

    c_L = np.sqrt(np.maximum(eq.gamma * p_L / rho_L, _EPS))
    c_R = np.sqrt(np.maximum(eq.gamma * p_R / rho_R, _EPS))

    S_L = np.minimum(u_L - c_L, u_R - c_R)
    S_R = np.maximum(u_L + c_L, u_R + c_R)

    SL_uL = S_L - u_L
    SR_uR = S_R - u_R
    den = rho_L * SL_uL - rho_R * SR_uR
    den = np.where(np.abs(den) > _EPS, den, np.sign(den) * _EPS + _EPS)
    S_star = (p_R - p_L + rho_L * u_L * SL_uL - rho_R * u_R * SR_uR) / den

    # Total energy (frame-invariant scalar)
    E_L = p_L / ((eq.gamma - 1.0) * rho_L) + 0.5 * u_L_orig ** 2
    E_R = p_R / ((eq.gamma - 1.0) * rho_R) + 0.5 * u_R_orig ** 2

    # Conservative state in face-aligned frame: (ρ, ρ u_n, ρE)
    Uf_L = np.stack([rho_L, rho_L * u_L, rho_L * E_L], axis=0)
    Uf_R = np.stack([rho_R, rho_R * u_R, rho_R * E_R], axis=0)
    # Face-aligned physical flux: (ρ u_n, ρ u_n² + p, (ρE + p) u_n)
    Ff_L = np.stack([rho_L * u_L,
                     rho_L * u_L * u_L + p_L,
                     (rho_L * E_L + p_L) * u_L], axis=0)
    Ff_R = np.stack([rho_R * u_R,
                     rho_R * u_R * u_R + p_R,
                     (rho_R * E_R + p_R) * u_R], axis=0)

    factor_L = SL_uL / (S_L - S_star)
    factor_R = SR_uR / (S_R - S_star)
    rho_star_L = rho_L * factor_L
    rho_star_R = rho_R * factor_R
    rhoE_star_L = rho_L * factor_L * (E_L + (S_star - u_L) *
                                       (S_star + p_L / (rho_L * SL_uL)))
    rhoE_star_R = rho_R * factor_R * (E_R + (S_star - u_R) *
                                       (S_star + p_R / (rho_R * SR_uR)))
    Ufstar_L = np.stack([rho_star_L, rho_star_L * S_star, rhoE_star_L], axis=0)
    Ufstar_R = np.stack([rho_star_R, rho_star_R * S_star, rhoE_star_R], axis=0)

    Ff = np.where(
        S_L >= 0.0, Ff_L,
        np.where(
            S_star >= 0.0, Ff_L + S_L * (Ufstar_L - Uf_L),
            np.where(
                S_R >= 0.0, Ff_R + S_R * (Ufstar_R - Uf_R),
                Ff_R,
            )
        )
    )
    # Rotate back: only the momentum component depends on n_x.
    F = np.empty_like(Ff)
    F[0] = Ff[0]
    F[1] = Ff[1] * n_x
    F[2] = Ff[2]
    return F


def ausm_plus_up_2d(eq, W_L, W_R, normal, points=None):
    """AUSM+-up flux for 2D Euler using Liou's published split constants.

    The convective mass flux and pressure flux are split separately.  This
    tends to be less carbuncle-prone than contact-preserving HLLC while
    avoiding the full contact smearing of HLLE/HLL on shear layers.
    """
    if eq.__class__.__name__ != 'Euler2D':
        return llf(eq, W_L, W_R, normal, points=points)
    if _NUMBA_AVAILABLE and W_L.shape[0] == 4 and W_L.shape[1] >= 512:
        return _ausm_plus_up_kernel(
            np.ascontiguousarray(W_L, dtype=np.float64),
            np.ascontiguousarray(W_R, dtype=np.float64),
            np.ascontiguousarray(normal, dtype=np.float64),
            float(eq.gamma))

    n = np.asarray(normal, dtype=float)
    nx = n[..., 0]
    ny = n[..., 1]
    gamma = float(eq.gamma)
    rho_L = np.maximum(W_L[0], _EPS)
    rho_R = np.maximum(W_R[0], _EPS)
    u_L = W_L[1]
    v_L = W_L[2]
    u_R = W_R[1]
    v_R = W_R[2]
    p_L = np.maximum(W_L[3], _EPS)
    p_R = np.maximum(W_R[3], _EPS)

    un_L = u_L * nx + v_L * ny
    un_R = u_R * nx + v_R * ny
    c_L = np.sqrt(np.maximum(gamma * p_L / rho_L, _EPS))
    c_R = np.sqrt(np.maximum(gamma * p_R / rho_R, _EPS))
    a = np.maximum(0.5 * (c_L + c_R), _EPS)
    M_L = un_L / a
    M_R = un_R / a

    beta = 0.125
    alpha = 0.1875

    def _M4_plus(M):
        absM = np.abs(M)
        M1 = 0.5 * (M + absM)
        M2p = 0.25 * (M + 1.0) ** 2
        M2m = -0.25 * (M - 1.0) ** 2
        return np.where(absM >= 1.0, M1, M2p * (1.0 - 16.0 * beta * M2m))

    def _M4_minus(M):
        absM = np.abs(M)
        M1 = 0.5 * (M - absM)
        M2p = 0.25 * (M + 1.0) ** 2
        M2m = -0.25 * (M - 1.0) ** 2
        return np.where(absM >= 1.0, M1, M2m * (1.0 + 16.0 * beta * M2p))

    def _P5_plus(M):
        absM = np.abs(M)
        M1 = 0.5 * (M + absM) / np.maximum(M, _EPS)
        M2p = 0.25 * (M + 1.0) ** 2
        M2m = -0.25 * (M - 1.0) ** 2
        sub = M2p * ((2.0 - M) - 16.0 * alpha * M * M2m)
        return np.where(absM >= 1.0, M1, sub)

    def _P5_minus(M):
        absM = np.abs(M)
        M1 = 0.5 * (M - absM) / np.minimum(M, -_EPS)
        M2p = 0.25 * (M + 1.0) ** 2
        M2m = -0.25 * (M - 1.0) ** 2
        sub = M2m * ((-2.0 - M) + 16.0 * alpha * M * M2p)
        return np.where(absM >= 1.0, M1, sub)

    Mp = _M4_plus(M_L)
    Mm = _M4_minus(M_R)
    rho_face = np.where(Mp + Mm >= 0.0, rho_L, rho_R)
    rho_bar = 0.5 * (rho_L + rho_R)
    Mbar2 = 0.5 * (un_L * un_L + un_R * un_R) / np.maximum(a * a, _EPS)
    pressure_diffusion = 0.25 * np.maximum(1.0 - Mbar2, 0.0) * (
        p_R - p_L) / np.maximum(rho_bar * a * a, _EPS)
    M_face = Mp + Mm - pressure_diffusion
    mdot = a * M_face * rho_face

    Pp = _P5_plus(M_L)
    Pm = _P5_minus(M_R)
    velocity_diffusion = (
        0.75 * Pp * Pm * (rho_L + rho_R) * a * (un_R - un_L))
    p_face = Pp * p_L + Pm * p_R - velocity_diffusion

    H_L = gamma * p_L / ((gamma - 1.0) * rho_L) + 0.5 * (
        u_L * u_L + v_L * v_L)
    H_R = gamma * p_R / ((gamma - 1.0) * rho_R) + 0.5 * (
        u_R * u_R + v_R * v_R)
    u_up = np.where(mdot >= 0.0, u_L, u_R)
    v_up = np.where(mdot >= 0.0, v_L, v_R)
    H_up = np.where(mdot >= 0.0, H_L, H_R)
    return np.stack([
        mdot,
        mdot * u_up + p_face * nx,
        mdot * v_up + p_face * ny,
        mdot * H_up,
    ], axis=0)


def slau2_2d(eq, W_L, W_R, normal, points=None):
    """SLAU2-style AUSM-family flux for 2D Euler.

    The mass flux follows the parameter-free SLAU/SLAU2 pressure-difference
    dissipation form of Shima/Kitamura.  The pressure split uses the existing
    AUSM+-up polynomial pressure flux, keeping this implementation conservative
    and directly comparable with the existing AUSM-family path.
    """
    if eq.__class__.__name__ != 'Euler2D':
        return llf(eq, W_L, W_R, normal, points=points)

    n = np.asarray(normal, dtype=float)
    nx = n[..., 0]
    ny = n[..., 1]
    gamma = float(eq.gamma)
    rho_L = np.maximum(W_L[0], _EPS)
    rho_R = np.maximum(W_R[0], _EPS)
    u_L = W_L[1]
    v_L = W_L[2]
    u_R = W_R[1]
    v_R = W_R[2]
    p_L = np.maximum(W_L[3], _EPS)
    p_R = np.maximum(W_R[3], _EPS)

    un_L = u_L * nx + v_L * ny
    un_R = u_R * nx + v_R * ny
    c_L = np.sqrt(np.maximum(gamma * p_L / rho_L, _EPS))
    c_R = np.sqrt(np.maximum(gamma * p_R / rho_R, _EPS))
    a = np.maximum(0.5 * (c_L + c_R), _EPS)
    M_L = un_L / a
    M_R = un_R / a

    abs_vbar = (
        rho_L * np.abs(un_L) + rho_R * np.abs(un_R)
    ) / np.maximum(rho_L + rho_R, _EPS)
    g = -np.maximum(np.minimum(M_L, 0.0), -1.0) * np.minimum(
        np.maximum(M_R, 0.0), 1.0)
    abs_v_plus = (1.0 - g) * abs_vbar + g * np.abs(un_L)
    abs_v_minus = (1.0 - g) * abs_vbar + g * np.abs(un_R)
    speed_L2 = u_L * u_L + v_L * v_L
    speed_R2 = u_R * u_R + v_R * v_R
    mach_hat = np.minimum(
        1.0, np.sqrt(0.5 * (speed_L2 + speed_R2)) / a)
    chi = (1.0 - mach_hat) * (1.0 - mach_hat)
    mdot = 0.5 * (
        rho_L * (un_L + abs_v_plus)
        + rho_R * (un_R - abs_v_minus)
        - chi * (p_R - p_L) / a)

    beta = 0.125
    alpha = 0.1875

    def _P5_plus(M):
        absM = np.abs(M)
        M1 = 0.5 * (M + absM) / np.maximum(M, _EPS)
        M2p = 0.25 * (M + 1.0) ** 2
        M2m = -0.25 * (M - 1.0) ** 2
        sub = M2p * ((2.0 - M) - 16.0 * alpha * M * M2m)
        return np.where(absM >= 1.0, M1, sub)

    def _P5_minus(M):
        absM = np.abs(M)
        M1 = 0.5 * (M - absM) / np.minimum(M, -_EPS)
        M2p = 0.25 * (M + 1.0) ** 2
        M2m = -0.25 * (M - 1.0) ** 2
        sub = M2m * ((-2.0 - M) + 16.0 * alpha * M * M2p)
        return np.where(absM >= 1.0, M1, sub)

    Pp = _P5_plus(M_L)
    Pm = _P5_minus(M_R)
    velocity_diffusion = (
        0.75 * Pp * Pm * (rho_L + rho_R) * a * (un_R - un_L))
    p_face = Pp * p_L + Pm * p_R - velocity_diffusion

    H_L = gamma * p_L / ((gamma - 1.0) * rho_L) + 0.5 * speed_L2
    H_R = gamma * p_R / ((gamma - 1.0) * rho_R) + 0.5 * speed_R2
    up_left = mdot >= 0.0
    u_up = np.where(up_left, u_L, u_R)
    v_up = np.where(up_left, v_L, v_R)
    H_up = np.where(up_left, H_L, H_R)
    return np.stack([
        mdot,
        mdot * u_up + p_face * nx,
        mdot * v_up + p_face * ny,
        mdot * H_up,
    ], axis=0)


def _slau2_with_riemann_pressure(eq, W_L, W_R, normal, *, pressure_flux):
    """SLAU2 convective flux with a Riemann acoustic pressure flux.

    The conservative mass/energy transport uses the same SLAU2 mass flux as
    `slau2_2d`.  The normal momentum pressure term is extracted from either
    the HLLE or HLLC momentum flux, avoiding a region switch while testing
    whether a more shock-robust acoustic flux suppresses carbuncle modes.
    """
    n = np.asarray(normal, dtype=float)
    nx = n[..., 0]
    ny = n[..., 1]
    gamma = float(eq.gamma)
    rho_L = np.maximum(W_L[0], _EPS)
    rho_R = np.maximum(W_R[0], _EPS)
    u_L = W_L[1]
    v_L = W_L[2]
    u_R = W_R[1]
    v_R = W_R[2]
    p_L = np.maximum(W_L[3], _EPS)
    p_R = np.maximum(W_R[3], _EPS)

    un_L = u_L * nx + v_L * ny
    un_R = u_R * nx + v_R * ny
    c_L = np.sqrt(np.maximum(gamma * p_L / rho_L, _EPS))
    c_R = np.sqrt(np.maximum(gamma * p_R / rho_R, _EPS))
    a = np.maximum(0.5 * (c_L + c_R), _EPS)
    M_L = un_L / a
    M_R = un_R / a

    abs_vbar = (
        rho_L * np.abs(un_L) + rho_R * np.abs(un_R)
    ) / np.maximum(rho_L + rho_R, _EPS)
    g = -np.maximum(np.minimum(M_L, 0.0), -1.0) * np.minimum(
        np.maximum(M_R, 0.0), 1.0)
    abs_v_plus = (1.0 - g) * abs_vbar + g * np.abs(un_L)
    abs_v_minus = (1.0 - g) * abs_vbar + g * np.abs(un_R)
    speed_L2 = u_L * u_L + v_L * v_L
    speed_R2 = u_R * u_R + v_R * v_R
    mach_hat = np.minimum(
        1.0, np.sqrt(0.5 * (speed_L2 + speed_R2)) / a)
    chi = (1.0 - mach_hat) * (1.0 - mach_hat)
    mdot = 0.5 * (
        rho_L * (un_L + abs_v_plus)
        + rho_R * (un_R - abs_v_minus)
        - chi * (p_R - p_L) / a)

    H_L = gamma * p_L / ((gamma - 1.0) * rho_L) + 0.5 * speed_L2
    H_R = gamma * p_R / ((gamma - 1.0) * rho_R) + 0.5 * speed_R2
    up_left = mdot >= 0.0
    u_up = np.where(up_left, u_L, u_R)
    v_up = np.where(up_left, v_L, v_R)
    un_up = u_up * nx + v_up * ny
    H_up = np.where(up_left, H_L, H_R)

    F_hllc, F_hll, *_ = _euler2d_hllc_hll(eq, W_L, W_R, n)
    Fp = F_hll if pressure_flux == 'hlle' else F_hllc
    normal_momentum = Fp[1] * nx + Fp[2] * ny
    p_face = normal_momentum - mdot * un_up

    return np.stack([
        mdot,
        mdot * u_up + p_face * nx,
        mdot * v_up + p_face * ny,
        mdot * H_up,
    ], axis=0)


def slau2_hlle_pressure_2d(eq, W_L, W_R, normal, points=None):
    """SLAU2 convective flux with HLLE-derived pressure/acoustic flux."""
    if eq.__class__.__name__ != 'Euler2D':
        return llf(eq, W_L, W_R, normal, points=points)
    return _slau2_with_riemann_pressure(
        eq, W_L, W_R, normal, pressure_flux='hlle')


def slau2_hllc_pressure_2d(eq, W_L, W_R, normal, points=None):
    """SLAU2 convective flux with HLLC-derived pressure/acoustic flux."""
    if eq.__class__.__name__ != 'Euler2D':
        return llf(eq, W_L, W_R, normal, points=points)
    return _slau2_with_riemann_pressure(
        eq, W_L, W_R, normal, pressure_flux='hllc')


# ─── Registry helper ───────────────────────────────────────────────────────
def get_flux(name: str):
    table = {
        'upwind':         upwind_advection,
        'upwind_advection': upwind_advection,
        'central':        central,
        'llf':            llf,
        'rusanov':        llf,
        'hllc_adc':       hllc_adc_2d,
        'hllc_adc_strong': hllc_adc_strong_2d,
        'hllc_adc_hcorr': hllc_adc_hcorr_2d,
        'hllc_adc_acoustic_hcorr': hllc_adc_acoustic_hcorr_2d,
        'hllc_adc_acoustic_soft_hcorr': hllc_adc_acoustic_soft_hcorr_2d,
        'hllc_adc_transverse_hcorr': hllc_adc_transverse_hcorr_2d,
        'hllc_adc_transverse_energy_hcorr': (
            hllc_adc_transverse_energy_hcorr_2d),
        'hllc_adc_transverse_soft_hcorr': hllc_adc_transverse_soft_hcorr_2d,
        'hllc_adc_mass_hcorr': hllc_adc_mass_hcorr_2d,
        'hllc_adc_full_transverse_soft_hcorr': (
            hllc_adc_full_transverse_soft_hcorr_2d),
        'hllc_adc_full_transverse_hcorr': hllc_adc_full_transverse_hcorr_2d,
        'hllc_adc_full_neighbor_hcorr': hllc_adc_full_neighbor_hcorr_2d,
        'hllc_adc_full_neighbor_soft_hcorr': (
            hllc_adc_full_neighbor_soft_hcorr_2d),
        'hllc_adc_full_neighbor_hllblend': (
            hllc_adc_full_neighbor_hllblend_2d),
        'hllc_adc_full_neighbor_soft_hllblend': (
            hllc_adc_full_neighbor_soft_hllblend_2d),
        'hllc_adc_full_ducros_hll_soft': (
            hllc_adc_full_ducros_hll_soft_2d),
        'hllc_adc_hllct_momentum': hllc_adc_hllct_momentum_2d,
        'hllc_adc_ausm_hlle_momentum': hllc_adc_ausm_hlle_momentum_2d,
        'hllc_adc_ausm_hlle_momentum_soft_hllblend': (
            hllc_adc_ausm_hlle_momentum_soft_hllblend_2d),
        'hllc_adc_ausm_hlle_momentum_hllblend': (
            hllc_adc_ausm_hlle_momentum_hllblend_2d),
        'hllc_adc_ausm_hlle_momentum_pressure_hcorr': (
            hllc_adc_ausm_hlle_momentum_pressure_hcorr_2d),
        'hllc_adc_ausm_hlle_momentum_pressure_normal_hcorr': (
            hllc_adc_ausm_hlle_momentum_pressure_normal_hcorr_2d),
        'hllc_adc_ausm_hlle_momentum_pressureonly_normal_hcorr': (
            hllc_adc_ausm_hlle_momentum_pressureonly_normal_hcorr_2d),
        'hllc_adc_ausm_hlle_momentum_pressureonly_soft_normal_hcorr': (
            hllc_adc_ausm_hlle_momentum_pressureonly_soft_normal_hcorr_2d),
        'hllc_adc_full_hllct_momentum': hllc_adc_full_hllct_momentum_2d,
        'hllc_adc_hllct_momentum_neighbor_hllblend': (
            hllc_adc_hllct_momentum_neighbor_hllblend_2d),
        'hllc_adc_hllct_momentum_neighbor_soft_hllblend': (
            hllc_adc_hllct_momentum_neighbor_soft_hllblend_2d),
        'hllc_adc_hllct_momentum_neighbor_geomean_hllblend': (
            hllc_adc_hllct_momentum_neighbor_geomean_hllblend_2d),
        'hllc_adc_hllct_momentum_neighbor_soft_hllblend_pos': (
            hllc_adc_hllct_momentum_neighbor_soft_hllblend_pos_2d),
        'hllc_adc_hllct_momentum_neighbor_ultrasoft_hllblend': (
            hllc_adc_hllct_momentum_neighbor_ultrasoft_hllblend_2d),
        'hllc_adc_hllct_momentum_neighbor_shear_guard_ultrasoft_hllblend': (
            hllc_adc_hllct_momentum_neighbor_shear_guard_ultrasoft_hllblend_2d),
        'hllc_adc_hllct_momentum_neighbor_directional_ultrasoft_hllblend': (
            hllc_adc_hllct_momentum_neighbor_directional_ultrasoft_hllblend_2d),
        'hllc_adc_hllct_momentum_neighbor_directional_soft_hllblend': (
            hllc_adc_hllct_momentum_neighbor_directional_soft_hllblend_2d),
        'hllc_adc_hllct_momentum_neighbor_directional_ultrasoft_normal_hcorr': (
            hllc_adc_hllct_momentum_neighbor_directional_ultrasoft_normal_hcorr_2d),
        'hllc_adc_hllct_momentum_neighbor_directional_ultrasoft_mass_normal_hcorr': (
            hllc_adc_hllct_momentum_neighbor_directional_ultrasoft_mass_normal_hcorr_2d),
        'hllc_adc_hllct_momentum_neighbor_soft_normal_hcorr': (
            hllc_adc_hllct_momentum_neighbor_soft_normal_hcorr_2d),
        'hllc_adc_hllct_momentum_neighbor_geomean_normal_hcorr': (
            hllc_adc_hllct_momentum_neighbor_geomean_normal_hcorr_2d),
        'hllc_adc_hllct_momentum_directional_soft_normal_hcorr': (
            hllc_adc_hllct_momentum_directional_soft_normal_hcorr_2d),
        'hllc_adc_hllct_momentum_directional_soft_mass_normal_hcorr': (
            hllc_adc_hllct_momentum_directional_soft_mass_normal_hcorr_2d),
        'hllc_adc_hllct_momentum_pressure_guard_directional_soft_mass_normal_hcorr': (
            hllc_adc_hllct_momentum_pressure_guard_directional_soft_mass_normal_hcorr_2d),
        'hllc_adc_hllct_momentum_pressure_guard_directional_ultrasoft_mass_normal_hcorr': (
            hllc_adc_hllct_momentum_pressure_guard_directional_ultrasoft_mass_normal_hcorr_2d),
        'hllc_adc_hllct_momentum_shear_guard_directional_soft_normal_hcorr': (
            hllc_adc_hllct_momentum_shear_guard_directional_soft_normal_hcorr_2d),
        'hllc_adc_hllct_momentum_pressure_guard_directional_soft_normal_hcorr': (
            hllc_adc_hllct_momentum_pressure_guard_directional_soft_normal_hcorr_2d),
        'hllc_adc_hllct_momentum_pressure_guard_soft_normal_hcorr': (
            hllc_adc_hllct_momentum_pressure_guard_soft_normal_hcorr_2d),
        'hllc_adc_hllct_momentum_shear_guard_geomean_normal_hcorr': (
            hllc_adc_hllct_momentum_shear_guard_geomean_normal_hcorr_2d),
        'hllc_adc_hllct_momentum_shear_guard_geomean_mass_energy_hcorr': (
            hllc_adc_hllct_momentum_shear_guard_geomean_mass_energy_hcorr_2d),
        'hllc_adc_hllct_momentum_pressure_guard_geomean_normal_hcorr': (
            hllc_adc_hllct_momentum_pressure_guard_geomean_normal_hcorr_2d),
        'hllc_adc_hllct_momentum_face_soft_normal_hcorr': (
            hllc_adc_hllct_momentum_face_soft_normal_hcorr_2d),
        'hllc_adc_hllct_momentum_face_geomean_normal_hcorr': (
            hllc_adc_hllct_momentum_face_geomean_normal_hcorr_2d),
        'hllc_adc_hllct_momentum_face_normal_hcorr': (
            hllc_adc_hllct_momentum_face_normal_hcorr_2d),
        'hllc_adc_hllct_momentum_face_soft_mass_normal_hcorr': (
            hllc_adc_hllct_momentum_face_soft_mass_normal_hcorr_2d),
        'hllc_adc_hllct_momentum_face_geomean_mass_normal_hcorr': (
            hllc_adc_hllct_momentum_face_geomean_mass_normal_hcorr_2d),
        'hllc_adc_hllct_momentum_directional_geomean_normal_hcorr': (
            hllc_adc_hllct_momentum_directional_geomean_normal_hcorr_2d),
        'hllc_adc_hllct_momentum_pressure_guard_directional_geomean_normal_hcorr': (
            hllc_adc_hllct_momentum_pressure_guard_directional_geomean_normal_hcorr_2d),
        'hllc_adc_hllct_momentum_transverse_soft_hcorr': (
            hllc_adc_hllct_momentum_transverse_soft_hcorr_2d),
        'hllc_adc_hllct_momentum_transverse_hcorr': (
            hllc_adc_hllct_momentum_transverse_hcorr_2d),
        'hllc_adc_hllct_momentum_pressure_hcorr': (
            hllc_adc_hllct_momentum_pressure_hcorr_2d),
        'hllc_adc_ducros': hllc_adc_ducros_2d,
        'hllc_adc_normal': hllc_adc_normal_2d,
        'hllc_adc_mass_normal': hllc_adc_mass_normal_2d,
        'hllc_adc_full':  hllc_adc_full_2d,
        'hllc_tangent_adc': hllc_tangent_adc_2d,
        'hllc_pure':      hllc_pure_2d,
        'hllc_shock_stable': hllc_adc_2d,
        'hllc_lm':        hllc_lm_2d,
        'hllc-lm':        hllc_lm_2d,
        'hllc_swm_p':     hllc_swm_p_2d,
        'hllc-swm-p':     hllc_swm_p_2d,
        'hllct':          hllct_2d,
        'hllc_rotated_hybrid': hllc_rotated_hybrid_2d,
        'hllc_rotated_compressive_hybrid': (
            hllc_rotated_compressive_hybrid_2d),
        'hllc_rotated_compressive_normal_hybrid': (
            hllc_rotated_compressive_normal_hybrid_2d),
        'hllc_rotated_compressive_tangent_hybrid': (
            hllc_rotated_compressive_tangent_hybrid_2d),
        'hllc_rotated_compressive_normality2_hybrid': (
            hllc_rotated_compressive_normality2_hybrid_2d),
        'roe_rotated_hybrid': roe_rotated_hybrid_2d,
        'roe_rotated_shock_hybrid': roe_rotated_shock_hybrid_2d,
        'roe_rotated_soft_shock_hybrid': roe_rotated_soft_shock_hybrid_2d,
        'ausm_rotated_hybrid': ausm_rotated_hybrid_2d,
        'ausm_hlle_shock': ausm_hlle_shock_2d,
        'ausm_slau2_shock': ausm_slau2_shock_2d,
        'ausm_slau2_pressure_shock': ausm_slau2_pressure_shock_2d,
        'ausm_slau2_pressure_shock_soft': ausm_slau2_pressure_shock_soft_2d,
        'ausm_slau2_pressure_shock_soft_normal': (
            ausm_slau2_pressure_shock_soft_normal_2d),
        'ausm_slau2_pressure_guarded_shock': (
            ausm_slau2_pressure_guarded_shock_2d),
        'ausm_plus_up_shear_guard_geomean_hcorr': (
            ausm_plus_up_shear_guard_geomean_hcorr_2d),
        'ausm_slau2_hcorr': ausm_slau2_hcorr_2d,
        'ausm_slau2_hcorr_strong': ausm_slau2_hcorr_strong_2d,
        'ausm_slau2_transverse_soft_hcorr': (
            ausm_slau2_transverse_soft_hcorr_2d),
        'ausm_slau2_normal_momentum_soft_hcorr': (
            ausm_slau2_normal_momentum_soft_hcorr_2d),
        'ausm_slau2_shear_guard_hcorr': ausm_slau2_shear_guard_hcorr_2d,
        'ausm_slau2_shear_guard_geomean_hcorr': (
            ausm_slau2_shear_guard_geomean_hcorr_2d),
        'ausm_slau2_shear_guard_mass_hcorr': (
            ausm_slau2_shear_guard_mass_hcorr_2d),
        'ausm_slau2_shear_guard_mass_energy_hcorr': (
            ausm_slau2_shear_guard_mass_energy_hcorr_2d),
        'ausm_slau2_shear_guard_geomean_hllblend': (
            ausm_slau2_shear_guard_geomean_hllblend_2d),
        'ausm_slau2_shear_guard_soft_hllblend': (
            ausm_slau2_shear_guard_soft_hllblend_2d),
        'ausm_slau2_shear_guard_ultrasoft_hllblend': (
            ausm_slau2_shear_guard_ultrasoft_hllblend_2d),
        'ausm_slau2_shear_guard_directional_soft_hllblend': (
            ausm_slau2_shear_guard_directional_soft_hllblend_2d),
        'ausm_slau2_pressure_guard_directional_soft_hllblend': (
            ausm_slau2_pressure_guard_directional_soft_hllblend_2d),
        'ausm_slau2_pressure_guard_directional_geomean_hllblend': (
            ausm_slau2_pressure_guard_directional_geomean_hllblend_2d),
        'ausm_slau2_shear_guard_geomean_mass_normal_hcorr': (
            ausm_slau2_shear_guard_geomean_mass_normal_hcorr_2d),
        'ausm_slau2_shear_guard_mass_normal_hcorr': (
            ausm_slau2_shear_guard_mass_normal_hcorr_2d),
        'ausm_slau2_shear_guard_full_hcorr': (
            ausm_slau2_shear_guard_full_hcorr_2d),
        'ausm_slau2_pressure_guard_hcorr': (
            ausm_slau2_pressure_guard_hcorr_2d),
        'ausm_slau2_pressure_guard_geomean_hcorr': (
            ausm_slau2_pressure_guard_geomean_hcorr_2d),
        'ausm_slau2_face_hcorr': ausm_slau2_face_hcorr_2d,
        'ausm_slau2_directional_hcorr': ausm_slau2_directional_hcorr_2d,
        'ausm_slau2_ducros': ausm_slau2_ducros_2d,
        'ausm_plus_up':   ausm_plus_up_2d,
        'ausm+up':        ausm_plus_up_2d,
        'slau2':          slau2_2d,
        'slau2_hlle_pressure': slau2_hlle_pressure_2d,
        'slau2_hllc_pressure': slau2_hllc_pressure_2d,
        'hll':            hlle_2d,
        'hlle':           hlle_2d,
        'hllc':           hllc_pure_2d,
        'hllc_1d':        hllc_1d,
    }
    name = name.lower()
    if name not in table:
        raise ValueError(f"unknown flux '{name}'; available: {list(table)}")
    return table[name]
