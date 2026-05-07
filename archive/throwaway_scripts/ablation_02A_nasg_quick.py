"""Quick ablation: short t_end, key configs only."""
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
    a1r1 = a1 * rho1; a2r2 = (1 - a1) * rho2
    rho = a1r1 + a2r2; ru = rho * u
    e1 = eos1.energy(rho1, p); e2 = eos2.energy(rho2, p)
    rE = a1r1 * e1 + a2r2 * e2 + 0.5 * rho * u**2
    return ph1, ph2, a1r1, a2r2, ru, rE, a1, dx, p0, u0


def run(label, t_end=0.01, **kw):
    ph1, ph2, a1r1, a2r2, ru, rE, a1, dx, p0, u0 = setup()
    base = dict(dx=dx, bc_l='periodic', bc_r='periodic',
                print_interval=999999, t_end=t_end,
                cfl=0.2, use_material_cfl=False, max_steps=20000)
    base.update(kw)
    try:
        ret = solve_IMEX(ph1, ph2, a1r1, a2r2, ru, rE, a1.copy(), **base)
        t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = ret
        p_f, u_f, *_ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)
        err_p = float(np.max(np.abs(p_f-p0))/p0)
        err_u = float(np.max(np.abs(u_f-u0)))
        status = ('PASS' if err_p < 1e-2 and err_u < 1e-2 else
                  'NaN' if np.isnan(err_p) else 'FAIL')
        print(f'[{label:45s}] t={t:.2e} err_p={err_p:8.2e} err_u={err_u:8.2e} → {status}', flush=True)
    except Exception as e:
        print(f'[{label:45s}] EXC {type(e).__name__}: {str(e)[:50]}', flush=True)


t_end = 0.01  # short for speed
print(f'=== 02-A NASG quick ablation, t_end={t_end}, acoustic CFL=0.2 ===')

# Starting from PASS baseline + turn feature ON one at a time
base_kw = dict(alpha_scheme='tvd', use_mmacm_ex=False, use_apec=False,
               use_compression=False, dissipation='none', primitive_recon='tvd',
               use_acid_face=True, acid_interface=True)

run('base (ACID face+interface, all-off)', t_end=t_end, **base_kw)

run('+ APEC', t_end=t_end, **{**base_kw, 'use_apec': True})
run('+ MMACM-Ex', t_end=t_end, **{**base_kw, 'use_mmacm_ex': True})
run('+ THINC-BVD', t_end=t_end, **{**base_kw, 'alpha_scheme': 'thinc_bvd'})
run('+ dissipation hybrid', t_end=t_end, **{**base_kw, 'dissipation': 'hybrid'})
run('+ compression', t_end=t_end, **{**base_kw, 'use_compression': True})

# Paired combinations
run('+ APEC + MMACM', t_end=t_end,
    **{**base_kw, 'use_apec': True, 'use_mmacm_ex': True})
run('+ APEC + THINC-BVD', t_end=t_end,
    **{**base_kw, 'use_apec': True, 'alpha_scheme': 'thinc_bvd'})
run('+ APEC + MMACM + THINC', t_end=t_end,
    **{**base_kw, 'use_apec': True, 'use_mmacm_ex': True,
       'alpha_scheme': 'thinc_bvd'})
run('Full (all ON)', t_end=t_end,
    **{**base_kw, 'use_apec': True, 'use_mmacm_ex': True,
       'alpha_scheme': 'thinc_bvd', 'use_compression': True,
       'dissipation': 'hybrid'})

# Without ACID
print('\n=== Without ACID face ===')
noacid_kw = {**base_kw, 'use_acid_face': False, 'acid_interface': False}
run('No ACID base', t_end=t_end, **noacid_kw)
run('No ACID + APEC', t_end=t_end, **{**noacid_kw, 'use_apec': True})
run('No ACID + MMACM', t_end=t_end, **{**noacid_kw, 'use_mmacm_ex': True})
run('No ACID full', t_end=t_end,
    **{**noacid_kw, 'use_apec': True, 'use_mmacm_ex': True,
       'alpha_scheme': 'thinc_bvd', 'use_compression': True,
       'dissipation': 'hybrid'})
