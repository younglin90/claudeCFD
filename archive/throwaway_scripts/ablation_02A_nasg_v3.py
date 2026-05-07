"""02-A NASG v3 — feature ON progression, find max compatible config."""
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


def run(label, **kw):
    ph1, ph2, a1r1, a2r2, ru, rE, a1, dx, p0, u0 = setup()
    base = dict(dx=dx, bc_l='periodic', bc_r='periodic',
                print_interval=999999,
                t_end=1.0, cfl=0.2, use_material_cfl=False,
                max_steps=100000)
    base.update(kw)
    try:
        ret = solve_IMEX(ph1, ph2, a1r1, a2r2, ru, rE, a1.copy(), **base)
        t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = ret
        p_f, u_f, *_ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)
        err_p = float(np.max(np.abs(p_f-p0))/p0)
        err_u = float(np.max(np.abs(u_f-u0)))
        status = ('PASS' if err_p < 1e-2 and err_u < 1e-2 else
                  'NaN' if np.isnan(err_p) else 'FAIL')
        print(f'[{label:50s}] t={t:.2e} err_p={err_p:8.2e} err_u={err_u:8.2e} → {status}')
    except Exception as e:
        print(f'[{label:50s}] EXC {type(e).__name__}: {str(e)[:50]}')


print('=== Baseline PASS config: CFL=0.2 acoustic, ACID face+interface, all-off ===')
run('base (PASS from v2)',
    alpha_scheme='tvd', use_mmacm_ex=False, use_apec=False,
    use_compression=False, dissipation='none', primitive_recon='tvd',
    use_acid_face=True, acid_interface=True)

print('\n=== Features ON one-by-one ===')
run('+ APEC',
    alpha_scheme='tvd', use_mmacm_ex=False, use_apec=True,
    use_compression=False, dissipation='none', primitive_recon='tvd',
    use_acid_face=True, acid_interface=True)

run('+ MMACM-Ex',
    alpha_scheme='tvd', use_mmacm_ex=True, use_apec=False,
    use_compression=False, dissipation='none', primitive_recon='tvd',
    use_acid_face=True, acid_interface=True)

run('+ APEC + MMACM-Ex',
    alpha_scheme='tvd', use_mmacm_ex=True, use_apec=True,
    use_compression=False, dissipation='none', primitive_recon='tvd',
    use_acid_face=True, acid_interface=True)

run('+ THINC-BVD alpha',
    alpha_scheme='thinc_bvd', use_mmacm_ex=False, use_apec=False,
    use_compression=False, dissipation='none', primitive_recon='tvd',
    use_acid_face=True, acid_interface=True)

run('+ dissipation=hybrid',
    alpha_scheme='tvd', use_mmacm_ex=False, use_apec=False,
    use_compression=False, dissipation='hybrid', primitive_recon='tvd',
    use_acid_face=True, acid_interface=True)

run('+ compression',
    alpha_scheme='tvd', use_mmacm_ex=False, use_apec=False,
    use_compression=True, C_alpha=1.0,
    dissipation='none', primitive_recon='tvd',
    use_acid_face=True, acid_interface=True)

print('\n=== Combined: max features ON ===')
run('ALL ON (standard config)',
    alpha_scheme='thinc_bvd', use_mmacm_ex=True, use_apec=True,
    use_compression=True, C_alpha=1.0,
    dissipation='hybrid', primitive_recon='tvd',
    use_acid_face=True, acid_interface=True)

print('\n=== Without ACID face (baseline clean) ===')
run('No ACID, minimal',
    alpha_scheme='tvd', use_mmacm_ex=False, use_apec=False,
    use_compression=False, dissipation='none', primitive_recon='tvd',
    use_acid_face=False, acid_interface=False)

run('No ACID + APEC',
    alpha_scheme='tvd', use_mmacm_ex=False, use_apec=True,
    use_compression=False, dissipation='none', primitive_recon='tvd',
    use_acid_face=False, acid_interface=False)

run('No ACID + MMACM + APEC',
    alpha_scheme='thinc_bvd', use_mmacm_ex=True, use_apec=True,
    use_compression=False, dissipation='none', primitive_recon='tvd',
    use_acid_face=False, acid_interface=False)
