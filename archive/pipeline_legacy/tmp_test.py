#!/usr/bin/env python3
"""Phase 1 Abgrall test — 5-equation AD solver with BGS fallback."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from solver.denner_1d.main import run

# EOS parameters
ph_water = {
    'type': 'nasg',
    'gamma': 1.187,
    'p_inf': 7.028e8,
    'pinf': 7.028e8,
    'b': 6.61e-4,
    'b_covolume': 6.61e-4,
    'kappa_v': 3610.0,
    'eta': -1.177788e6,
}
ph_air = {
    'type': 'ideal',
    'gamma': 1.4,
    'p_inf': 0.0,
    'pinf': 0.0,
    'b': 0.0,
    'b_covolume': 0.0,
    'kappa_v': 717.5,
    'eta': 0.0,
}

N = 10
x_cells = np.linspace(0.05, 0.95, N)
psi_init = np.where((x_cells >= 0.4) & (x_cells <= 0.6), 1.0, 0.0)
p_init = np.full(N, 1e5)
u_init = np.full(N, 1.0)
T_init = np.full(N, 300.0)

case = {
    'ph1': ph_water,
    'ph2': ph_air,
    'x_cells': x_cells,
    'psi_init': psi_init,
    'p_init': p_init,
    'u_init': u_init,
    'T_init': T_init,
    't_end': 1.0,
    'CFL': 0.5,
    'bc_left': 'periodic',
    'bc_right': 'periodic',
    'max_iteration': 100,
    'five_eq_ad': True,
    'use_autograd': True,
    'max_newton': 50,
    'newton_tol': 1e-6,
    'verbose': True,
    'verbose_newton': True,
}

result = run(case)

# Evaluate
state = result['final_state']
p_err = np.max(np.abs((state['p'] - 1e5) / 1e5))
u_err = np.max(np.abs(state['u'] - 1.0))

from solver.denner_1d.eos.base import compute_mixture_props
psi_reg = np.clip(state['psi'], 0.01, 0.99)
props0 = compute_mixture_props(p_init, u_init, T_init, np.clip(psi_init, 0.01, 0.99), ph_water, ph_air)
props_f = compute_mixture_props(state['p'], state['u'], state['T'], psi_reg, ph_water, ph_air)
E0 = np.sum(props0['E_total'])
Ef = np.sum(props_f['E_total'])
E_err = abs((Ef - E0) / (E0 + 1e-300))

print(f"\n=== Phase 1 Abgrall Results ===")
print(f"  err_p = {p_err:.3e}  (limit: 1e-2)")
print(f"  err_u = {u_err:.3e}  (limit: 1e-2)")
print(f"  err_E = {E_err:.3e}  (limit: 1e-2)")
print(f"  diverged = {result['diverged']}")
print(f"  steps = {len(result['dt_history'])}")
print(f"  PASS = {not result['diverged'] and p_err < 0.01 and u_err < 0.01 and E_err < 0.01}")
