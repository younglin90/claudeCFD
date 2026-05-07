"""02-A NASG Ablation — Turn features OFF one-by-one to isolate failure cause.

Baseline: All default (MMACM-Ex + APEC + SLAU2 + Peluchon IM1 + compression).
Each row turns ONE feature off.
"""
import os
import sys
import numpy as np

sys.path.insert(0, '/home/younglin90/work/claude_code/claudeCFD')
from solver.He2024.explicit_mmacm_ex import solve_IMEX, cons_to_prim
from solver.He2024.eos_general import IdealEOS, NASGEOS


def run_case(label, **kwargs):
    """Run 02-A NASG with given kwargs, report result."""
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5}
    ph2 = {'gamma': 1.187, 'pinf': 7.028e8, 'kv': 3610.0,
           'b': 6.61e-4, 'eta': -1.177788e6}
    eos1 = IdealEOS(gamma=1.4, kv=717.5)
    eos2 = NASGEOS(gamma=1.187, pinf=7.028e8, kv=3610.0,
                   b=6.61e-4, eta=-1.177788e6)

    N, L = 10, 1.0
    dx = L/N
    x = np.linspace(dx/2, L-dx/2, N)
    p0, u0, T0 = 1.0e5, 1.0, 300.0
    a_water = ((x >= 0.4) & (x <= 0.6)).astype(float)
    a1 = (1 - a_water) * (1 - 1e-6) + a_water * 1e-6  # α_air
    rho1 = eos1.density(np.full(N, p0), np.full(N, T0))
    rho2 = eos2.density(np.full(N, p0), np.full(N, T0))
    u = np.full(N, u0); p = np.full(N, p0)
    a1r1 = a1 * rho1
    a2r2 = (1 - a1) * rho2
    rho = a1r1 + a2r2
    ru = rho * u
    e1 = eos1.energy(rho1, p); e2 = eos2.energy(rho2, p)
    rE = a1r1 * e1 + a2r2 * e2 + 0.5 * rho * u**2

    default_kw = dict(
        dx=dx, t_end=1.0, cfl=0.4, use_material_cfl=True,
        bc_l='periodic', bc_r='periodic',
        max_steps=300, print_interval=999999,
    )
    default_kw.update(kwargs)

    try:
        ret = solve_IMEX(ph1, ph2, a1r1, a2r2, ru, rE, a1.copy(), **default_kw)
        t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = ret
        p_f, u_f, *_ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)
        err_p = float(np.max(np.abs(p_f-p0))/p0)
        err_u = float(np.max(np.abs(u_f-u0)))
        if np.isnan(err_p) or np.isnan(err_u):
            status = 'NaN'
        elif err_p < 1e-2 and err_u < 1e-2:
            status = 'PASS'
        elif err_p < 0.1 and err_u < 0.1:
            status = 'marginal'
        else:
            status = 'FAIL'
        print(f'[{label:30s}] t={t:.2e}  err_p={err_p:.2e}  err_u={err_u:.2e}  → {status}')
        return status, err_p, err_u, t
    except Exception as e:
        msg = str(e)[:80]
        print(f'[{label:30s}] EXCEPTION: {type(e).__name__}: {msg} → EXC')
        return 'EXC', float('nan'), float('nan'), 0.0


if __name__ == '__main__':
    print('=' * 90)
    print('02-A NASG Ablation Study')
    print('=' * 90)

    # Baseline (all features on, default)
    run_case('Baseline (all ON)',
             alpha_scheme='thinc_bvd', use_mmacm_ex=True, use_apec=True,
             use_compression=False, dissipation='hybrid',
             acoustic_method='im1', primitive_recon='tvd')

    print('\n--- Individual feature OFF ---')
    run_case('MMACM-Ex OFF',
             alpha_scheme='thinc_bvd', use_mmacm_ex=False, use_apec=True,
             use_compression=False, dissipation='hybrid',
             acoustic_method='im1', primitive_recon='tvd')

    run_case('APEC OFF',
             alpha_scheme='thinc_bvd', use_mmacm_ex=True, use_apec=False,
             use_compression=False, dissipation='hybrid',
             acoustic_method='im1', primitive_recon='tvd')

    # SLAU2 is in _advective_rhs_imex. use_hllc_flux=True forces HLLC instead.
    run_case('SLAU2 OFF (HLLC forced)',
             alpha_scheme='thinc_bvd', use_mmacm_ex=True, use_apec=True,
             use_compression=False, dissipation='hybrid',
             acoustic_method='im1', primitive_recon='tvd',
             use_hllc_flux=True)

    run_case('Peluchon IM1 OFF (elliptic)',
             alpha_scheme='thinc_bvd', use_mmacm_ex=True, use_apec=True,
             use_compression=False, dissipation='hybrid',
             acoustic_method='elliptic', primitive_recon='tvd')

    run_case('dissipation none',
             alpha_scheme='thinc_bvd', use_mmacm_ex=True, use_apec=True,
             use_compression=False, dissipation='none',
             acoustic_method='im1', primitive_recon='tvd')

    print('\n--- Multi-feature OFF ---')
    run_case('MMACM+APEC OFF',
             alpha_scheme='thinc_bvd', use_mmacm_ex=False, use_apec=False,
             use_compression=False, dissipation='hybrid',
             acoustic_method='im1', primitive_recon='tvd')

    run_case('All advanced OFF (minimal)',
             alpha_scheme='tvd', use_mmacm_ex=False, use_apec=False,
             use_compression=False, dissipation='none',
             acoustic_method='im1', primitive_recon='tvd')

    print('\n--- ACID face density ---')
    run_case('ACID face (minimal)',
             alpha_scheme='tvd', use_mmacm_ex=False, use_apec=False,
             use_compression=False, dissipation='none',
             acoustic_method='im1', primitive_recon='tvd',
             use_acid_face=True)

    run_case('ACID face + full features',
             alpha_scheme='thinc_bvd', use_mmacm_ex=True, use_apec=True,
             use_compression=False, dissipation='hybrid',
             acoustic_method='im1', primitive_recon='tvd',
             use_acid_face=True)

    run_case('ACID face + ACID interface',
             alpha_scheme='thinc_bvd', use_mmacm_ex=True, use_apec=True,
             use_compression=False, dissipation='hybrid',
             acoustic_method='im1', primitive_recon='tvd',
             use_acid_face=True, acid_interface=True)

    print('\n' + '=' * 90)
