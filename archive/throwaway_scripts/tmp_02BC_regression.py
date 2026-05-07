#!/usr/bin/env python3
"""
Test B/C 회귀 체크 (기존 설정 + alpha_scheme='cicsam').
Test A FAIL로 인해 Phase 2는 진행하지 않지만, Test B/C는 달성 가능한지 확인.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

sys.path.insert(0, '/home/younglin90/work/claude_code/claudeCFD')

from solver.He2024.explicit_mmacm_ex import solve_IMEX, cons_to_prim
from solver.He2024.eos_general import IdealEOS, NASGEOS
from solver.He2024.kapila_k import solve_kapila_K, cons_to_prim_K

def run_02B():
    """Test B: 3-species gas advection (alpha_scheme='cicsam')"""
    print('\n[02-B Test B: 3-species air/He/SF6, u=100 m/s, alpha=cicsam]')
    start = time.time()

    try:
        eos_list = [IdealEOS(gamma=1.4, kv=717.5),
                    IdealEOS(gamma=1.667, kv=3116.0),
                    IdealEOS(gamma=1.094, kv=665.0)]
        ph_list = [{'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5},
                   {'gamma': 1.667, 'pinf': 0.0, 'kv': 3116.0},
                   {'gamma': 1.094, 'pinf': 0.0, 'kv': 665.0}]

        N, L = 100, 1.0
        dx = L/N
        x = np.linspace(dx/2, L-dx/2, N)
        p0, u0, T0 = 1.0e5, 100.0, 300.0

        a = np.zeros((3, N))
        a[0] = ((x < 0.25) | (x >= 0.75)).astype(float)
        a[1] = ((x >= 0.25) & (x < 0.50)).astype(float)
        a[2] = ((x >= 0.50) & (x < 0.75)).astype(float)
        eps = 1e-6
        a = np.clip(a, eps, 1.0 - eps)
        a = a / np.sum(a, axis=0, keepdims=True)

        rho = np.array([e.density(np.full(N, p0), np.full(N, T0)) for e in eos_list])
        ar = a * rho
        u = np.full(N, u0)
        p = np.full(N, p0)
        rho_mix = np.sum(ar, axis=0)
        ru = rho_mix * u
        e = np.array([eos_list[k].energy(rho[k], p) for k in range(3)])
        rE = np.sum(ar * e, axis=0) + 0.5 * rho_mix * u**2

        res = solve_kapila_K(
            eos_list, list(ar), ru, rE, list(a),
            dx=dx, t_end=1e-2, cfl=0.4,
            bc_l='periodic', bc_r='periodic',
            max_steps=20000, print_interval=999999)
        t, ar_f_list, ru_f, rE_f, a_f_list = res
        ar_f = np.array(ar_f_list)
        a_f = np.array(a_f_list)

        prim = cons_to_prim_K(list(ar_f), ru_f, rE_f, list(a_f), eos_list)
        p_f, u_f, T_f = prim[0], prim[1], prim[2]

        err_p = float(np.max(np.abs(p_f - p0)) / p0)
        err_u = float(np.max(np.abs(u_f - u0)))

        status = 'PASS' if (err_p < 1e-6 and err_u < 1e-2) else 'FAIL'
        wall = time.time() - start

        print(f'  Status: {status}')
        print(f'  err_p={err_p:.3e}, err_u={err_u:.3e}, t={t:.3e}')
        print(f'  wall_time={wall:.3f}s')

        return status, err_p, err_u, wall

    except Exception as e:
        import traceback
        traceback.print_exc()
        return 'FAIL', np.nan, np.nan, time.time()-start


def run_02C():
    """Test C: Moving contact u=100, p=1e9 (alpha_scheme='cicsam')"""
    print('\n[02-C Test C: Moving contact u=100 m/s, p=1e9 Pa, alpha=cicsam]')
    start = time.time()

    try:
        ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5}
        ph2 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5}
        eos1 = IdealEOS(gamma=1.4, kv=717.5)
        eos2 = IdealEOS(gamma=1.4, kv=717.5)

        N, L = 200, 1.0
        dx = L/N
        x = np.linspace(dx/2, L-dx/2, N)
        p0, u0, T0 = 1.0e9, 100.0, 300.0

        a1 = np.where(x < 0.5, 1 - 1e-6, 1e-6)
        rho1 = eos1.density(np.full(N, p0), np.full(N, T0))
        rho2 = eos2.density(np.full(N, p0), np.full(N, T0))
        u = np.full(N, u0)
        p = np.full(N, p0)

        a1r1 = a1 * rho1
        a2r2 = (1 - a1) * rho2
        rho = a1r1 + a2r2
        ru = rho * u
        e1 = eos1.energy(rho1, p)
        e2 = eos2.energy(rho2, p)
        rE = a1r1 * e1 + a2r2 * e2 + 0.5 * rho * u**2

        t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
            ph1, ph2, a1r1, a2r2, ru, rE, a1.copy(),
            dx=dx, t_end=0.01, cfl=0.4,
            bc_l='periodic', bc_r='periodic',
            max_steps=20000, print_interval=999999,
            alpha_scheme='cicsam',  # Changed to cicsam
            use_mmacm_ex=False)

        p_f, u_f, T_f, rho1_f, rho2_f, *_ = cons_to_prim(
            a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)

        err_p = float(np.max(np.abs(p_f - p0)) / p0)
        err_u = float(np.max(np.abs(u_f - u0)))
        err_T = float(np.max(np.abs(T_f - T0)) / T0)

        status = 'PASS' if (err_p < 1e-10 and err_u < 1e-8) else 'FAIL'
        wall = time.time() - start

        print(f'  Status: {status}')
        print(f'  err_p={err_p:.3e}, err_u={err_u:.3e}, err_T={err_T:.3e}, t={t:.3e}')
        print(f'  wall_time={wall:.3f}s')

        return status, err_p, err_u, wall

    except Exception as e:
        import traceback
        traceback.print_exc()
        return 'FAIL', np.nan, np.nan, time.time()-start


if __name__ == '__main__':
    print("="*90)
    print("Test B/C Regression Check (alpha_scheme='cicsam')")
    print("="*90)

    status_b, err_p_b, err_u_b, wall_b = run_02B()
    status_c, err_p_c, err_u_c, wall_c = run_02C()

    print("\n" + "="*90)
    print("Summary")
    print("="*90)
    print(f"Test B: {status_b} (err_p={err_p_b:.3e}, err_u={err_u_b:.3e}, wall={wall_b:.3f}s)")
    print(f"Test C: {status_c} (err_p={err_p_c:.3e}, err_u={err_u_c:.3e}, wall={wall_c:.3f}s)")
