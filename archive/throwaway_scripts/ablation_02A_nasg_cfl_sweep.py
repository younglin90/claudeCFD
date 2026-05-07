"""NASG CFL sweep — find where material CFL starts to fail.

IMEX acoustic implicit → material CFL (dt=cfl·dx/|u|) should work.
At u=1, dx=0.1:
  - mat CFL=0.01: dt=1e-3  (100 steps) acoustic_CFL=~15
  - mat CFL=0.05: dt=5e-3  acoustic_CFL=77
  - mat CFL=0.1:  dt=1e-2  acoustic_CFL=154
  - mat CFL=0.2:  dt=2e-2  acoustic_CFL=308
  - mat CFL=0.4:  dt=4e-2  acoustic_CFL=616
"""
import sys
import numpy as np
sys.path.insert(0, '/home/younglin90/work/claude_code/claudeCFD')
from solver.He2024.explicit_mmacm_ex import solve_IMEX, cons_to_prim
from solver.He2024.eos_general import IdealEOS, NASGEOS


def setup():
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5}
    ph2 = {'gamma': 1.187, 'pinf': 7.028e8, 'kv': 3610.0,
           'b': 6.61e-4, 'eta': -1.177788e6}
    eos1 = IdealEOS(gamma=1.4, kv=717.5)
    eos2 = NASGEOS(gamma=1.187, pinf=7.028e8, kv=3610.0,
                   b=6.61e-4, eta=-1.177788e6)
    N, L = 10, 1.0; dx = L/N
    x = np.linspace(dx/2, L-dx/2, N)
    p0, u0, T0 = 1.0e5, 1.0, 300.0
    a_water = ((x >= 0.4) & (x <= 0.6)).astype(float)
    a1 = (1 - a_water) * (1 - 1e-6) + a_water * 1e-6
    rho1 = eos1.density(np.full(N, p0), np.full(N, T0))
    rho2 = eos2.density(np.full(N, p0), np.full(N, T0))
    u = np.full(N, u0); p = np.full(N, p0)
    a1r1 = a1*rho1; a2r2 = (1-a1)*rho2
    rho = a1r1 + a2r2; ru = rho * u
    e1 = eos1.energy(rho1, p); e2 = eos2.energy(rho2, p)
    rE = a1r1*e1 + a2r2*e2 + 0.5*rho*u**2
    return ph1, ph2, a1r1, a2r2, ru, rE, a1, dx, p0, u0


def run(cfl, use_mat_cfl, t_end=1.0, **kw):
    ph1, ph2, a1r1, a2r2, ru, rE, a1, dx, p0, u0 = setup()
    c_water = 1540.0
    if use_mat_cfl:
        dt_est = cfl * dx / 1.0   # material CFL
        ac_cfl = c_water * dt_est / dx
        label = f'mat CFL={cfl}  (dt~{dt_est:.1e}, ac_CFL~{ac_cfl:.0f})'
    else:
        dt_est = cfl * dx / c_water
        ac_cfl = cfl
        label = f'acoustic CFL={cfl}  (dt~{dt_est:.1e})'
    import time; tic=time.time()
    try:
        ret = solve_IMEX(ph1, ph2, a1r1, a2r2, ru, rE, a1.copy(),
                         dx=dx, t_end=t_end, cfl=cfl,
                         use_material_cfl=use_mat_cfl,
                         bc_l='periodic', bc_r='periodic',
                         max_steps=200000, print_interval=99999,
                         alpha_scheme='tvd', use_mmacm_ex=False, use_apec=False,
                         use_compression=False, dissipation='none',
                         primitive_recon='tvd', use_acid_face=True,
                         acid_interface=True, **kw)
        t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = ret
        p_f, u_f, *_ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)
        err_p = float(np.max(np.abs(p_f-p0))/p0)
        err_u = float(np.max(np.abs(u_f-u0)))
        n_steps = int(t / dt_est) if dt_est > 0 else 0
        wall = time.time() - tic
        status = ('PASS' if err_p<1e-2 and err_u<1e-2 else
                  'NaN' if np.isnan(err_p) else 'FAIL')
        print(f'[{label:48s}] t={t:.2e} err_p={err_p:8.2e} err_u={err_u:8.2e} wall={wall:.1f}s → {status}')
    except Exception as e:
        print(f'[{label:48s}] EXC: {str(e)[:40]}')


print('=== NASG material CFL sweep (ACID face+interface, t_end=1.0) ===')
for c in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4]:
    run(c, True, t_end=1.0)

print('\n=== Acoustic CFL sweep for comparison ===')
for c in [0.1, 0.2, 0.4, 0.5]:
    run(c, False, t_end=1.0)
