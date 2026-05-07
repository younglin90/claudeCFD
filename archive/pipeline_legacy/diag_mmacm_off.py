"""Test: disable MMACM-Ex for THINC-BVD. See if THINC-BVD alone is stable."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from solver.He2024.explicit_mmacm_ex import solve_IMEX, cons_to_prim, _advective_rhs_imex

# Monkey-patch to disable mmacm_ex
import solver.He2024.explicit_mmacm_ex as mod
orig_rhs = mod._advective_rhs_imex
def patched_rhs(*args, **kwargs):
    kwargs['use_mmacm_ex'] = False
    return orig_rhs(*args, **kwargs)

def run(scheme, label, patch=False):
    if patch:
        mod._advective_rhs_imex = patched_rhs
    else:
        mod._advective_rhs_imex = orig_rhs

    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
    ph2 = {'gamma': 4.4, 'pinf': 6e8, 'kv': 474.2, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
    N = 200; L = 1.0; dx = L / N
    x = np.linspace(dx/2, L - dx/2, N)
    xd = 0.7
    a_air = np.where(x < xd, 1e-6, 1.0 - 1e-6)
    p_init = np.where(x < xd, 1e9, 1e5)
    rho1 = 50.0 * np.ones(N); rho2 = 1000.0 * np.ones(N)
    a1r1 = a_air * rho1; a2r2 = (1 - a_air) * rho2
    ru = np.zeros(N)
    gm1, gm2 = 0.4, 3.4
    rho_e = a_air * p_init / gm1 + (1 - a_air) * (p_init + ph2['gamma'] * ph2['pinf']) / gm2
    rE = rho_e

    t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
        ph1, ph2, a1r1, a2r2, ru, rE, a_air,
        dx, t_end=2.29e-4, cfl=0.25, bc_l='transmissive', bc_r='transmissive',
        max_steps=100000, print_interval=10000,
        alpha_scheme=scheme, use_strang=True,
        use_defect_correction=False, use_material_cfl=False)

    p, u, _, _, _, _, _, _ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)
    u_max = np.max(np.abs(u))
    n_trans = int(np.sum((a1_f > 0.1) & (a1_f < 0.9)))
    print(f"  {label:24s}: u_max={u_max:.2f}, trans_cells={n_trans}")
    return u_max

if __name__ == '__main__':
    print("="*60)
    print("MMACM-Ex ON (default):")
    print("="*60)
    for s, l in [('tvd', 'TVD'), ('thinc_bvd', 'THINC-BVD'), ('cicsam', 'CICSAM'), ('mstacs', 'MSTACS')]:
        run(s, l, patch=False)

    print("\n" + "="*60)
    print("MMACM-Ex OFF:")
    print("="*60)
    for s, l in [('tvd', 'TVD'), ('thinc_bvd', 'THINC-BVD'), ('cicsam', 'CICSAM'), ('mstacs', 'MSTACS')]:
        run(s, l, patch=True)
