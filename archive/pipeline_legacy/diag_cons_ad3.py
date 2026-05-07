#!/usr/bin/env python3
"""Diagnose conservative AD Newton with hybrid: live p, lagged ACID phase."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from solver.denner_1d.eos.eos_class import create_eos
from solver.denner_1d.assembly_5eq_ad import (
    _get_eos_params, make_residual_cons_ad, compute_jacobian_ad,
    compute_K_ad, compute_face_velocity_ad, pack_Q, unpack_Q,
)

ph_water = {
    'type': 'nasg', 'gamma': 1.187, 'p_inf': 7.028e8, 'pinf': 7.028e8,
    'b': 6.61e-4, 'b_covolume': 6.61e-4, 'kappa_v': 3610.0, 'eta': -1.177788e6,
}
ph_air = {
    'type': 'ideal', 'gamma': 1.4, 'p_inf': 0.0, 'pinf': 0.0,
    'b': 0.0, 'b_covolume': 0.0, 'kappa_v': 717.5, 'eta': 0.0,
}

N = 10; dx = 0.1
ph1_params = _get_eos_params(ph_water)
ph2_params = _get_eos_params(ph_air)
eos1 = create_eos(ph_water); eos2 = create_eos(ph_air)

x_cells = np.linspace(0.05, 0.95, N)
a1 = np.clip(np.where((x_cells >= 0.4) & (x_cells <= 0.6), 1.0, 0.0), 1e-10, 1-1e-10)
a2 = 1.0 - a1
p = np.full(N, 1e5); u = np.full(N, 1.0); T = np.full(N, 300.0)

rho1 = eos1.rho(p, T); rho2 = eos2.rho(p, T); rho = a1*rho1 + a2*rho2
a1r1 = a1*rho1; a2r2 = a2*rho2; ru = rho*u
rE = a1*eos1.e_vol(p,T) + a2*eos2.e_vol(p,T) + 0.5*rho*u**2
Q_n = pack_Q(a1r1, a2r2, ru, rE, a1)

c1 = eos1.c(p,T); c2 = eos2.c(p,T)
c_wood_sq = 1.0/(a1/(rho1*c1**2) + a2/(rho2*c2**2) + 1e-300)/rho
c_max = np.max(np.abs(u) + np.sqrt(np.maximum(c_wood_sq, 0)))
dt = 0.5 * dx / (c_max + 1e-300)

theta, u_bar, d_hat, rho_star = compute_face_velocity_ad(u, p, rho, dx, dt, 'periodic', 'periodic')
K = compute_K_ad(a1, p, T, ph_water, ph_air)

res_func = make_residual_cons_ad(Q_n, N, dx, dt, ph1_params, ph2_params,
    'periodic', 'periodic', theta, K, p_lag=p, T_lag=T)

R0 = np.array(res_func(Q_n))
print(f"=== Residual at Q_n ===")
print(f"  |R| = {np.linalg.norm(R0):.3e}")
print(f"  R_en max = {np.max(np.abs(R0[3*N:4*N])):.3e}  (should be ~0 for PE)")

from autograd import jacobian as ad_jacobian
J = np.array(ad_jacobian(res_func)(Q_n))

print(f"\n=== Jacobian ===")
print(f"  max |entry|: {np.max(np.abs(J)):.3e}")
print(f"  cond: {np.linalg.cond(J):.3e}")
print(f"  rank: {np.linalg.matrix_rank(J)} / {J.shape[0]}")

diag = np.abs(np.diag(J))
row_sum = np.sum(np.abs(J), axis=1) - diag
print(f"  min diag/rowsum: {np.min(diag/(row_sum+1e-300)):.3e}")
print(f"  1/dt = {1/dt:.3e}")

# Block norms
names = ['a1r1','a2r2','ru','rE','a1']
for i in range(5):
    for j in range(5):
        block = J[i*N:(i+1)*N, j*N:(j+1)*N]
        norm = np.max(np.abs(block))
        if norm > 1e-6:
            print(f"  J[{names[i]},{names[j]}] max = {norm:.3e}")

# Newton
dQ = np.linalg.solve(J, -R0)
for omega in [1.0, 0.5, 0.1, 0.01]:
    Q_trial = Q_n + omega * dQ
    Q_trial[4*N:5*N] = np.clip(Q_trial[4*N:5*N], 1e-10, 1-1e-10)
    Q_trial[0*N:1*N] = np.maximum(Q_trial[0*N:1*N], 1e-20)
    Q_trial[1*N:2*N] = np.maximum(Q_trial[1*N:2*N], 1e-20)
    try:
        R_trial = np.array(res_func(Q_trial))
        Rn = np.linalg.norm(R_trial)
        ok = "✓" if Rn < np.linalg.norm(R0) else "✗"
        print(f"  ω={omega:.3f}: |R|={Rn:.3e} {ok}")
    except Exception as e:
        print(f"  ω={omega:.3f}: FAILED ({e})")
