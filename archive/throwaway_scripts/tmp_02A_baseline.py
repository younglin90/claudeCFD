#!/usr/bin/env python3
"""
Test A baseline: 기존 PASS 설정 확인 (use_material_cfl=False, cfl=0.2)
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

def run_baseline():
    """기존 작동하는 설정"""
    start = time.time()
    try:
        # EOS
        ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5}  # Air
        ph2 = {'gamma': 1.187, 'pinf': 7.028e8, 'kv': 3610.0, 'b': 6.61e-4, 'eta': -1.177788e6}  # Water

        eos1 = IdealEOS(gamma=1.4, kv=717.5)
        eos2 = NASGEOS(gamma=1.187, pinf=7.028e8, kv=3610.0, b=6.61e-4, eta=-1.177788e6)

        N, L = 10, 1.0
        dx = L / N
        x = np.linspace(dx/2, L-dx/2, N)
        p0, u0, T0 = 1.0e5, 1.0, 300.0

        a_water = ((x >= 0.4) & (x <= 0.6)).astype(float)
        a1 = (1 - a_water) * (1 - 1e-6) + a_water * 1e-6

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

        print("Initial state:")
        print(f"  N={N}, dx={dx}, p0={p0:.2e}, u0={u0}, T0={T0}")
        print(f"  rho1={rho1[0]:.4f}, rho2={rho2[0]:.4f}")
        print(f"  a1[0:3]={a1[0:3]}")
        print(f"  a1r1[0:3]={a1r1[0:3]}")

        # 기존 PASS 설정
        print("\nRunning: cfl=0.2, use_material_cfl=False (기존 PASS)")
        t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
            ph1, ph2, a1r1, a2r2, ru, rE, a1.copy(),
            dx=dx, t_end=1.0, cfl=0.2, use_material_cfl=False,
            bc_l='periodic', bc_r='periodic',
            max_steps=100, print_interval=99999,
            alpha_scheme='tvd', use_mmacm_ex=False, use_apec=False,
            use_compression=False, dissipation='none',
            primitive_recon='tvd', use_acid_face=True, acid_interface=True,
            iterative_im1=True, iterative_im1_max=5, iterative_im1_tol=1e-6
        )

        print(f"Result: t={t:.6e}, {100} steps")
        print(f"  Final: a1r1_f[0:3]={a1r1_f[0:3]}")
        print(f"  Final: a1_f[0:3]={a1_f[0:3]}")

        p_f, u_f, T_f, rho1_f, rho2_f, *_ = cons_to_prim(
            a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)

        err_p = float(np.max(np.abs(p_f - p0)) / p0)
        err_u = float(np.max(np.abs(u_f - u0)))

        print(f"\nErrors:")
        print(f"  err_p = {err_p:.6e}")
        print(f"  err_u = {err_u:.6e}")
        print(f"  Status: {'PASS' if (err_p < 1e-2 and err_u < 1e-2) else 'FAIL'}")
        print(f"  Wall time: {time.time()-start:.3f}s")

    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    run_baseline()
