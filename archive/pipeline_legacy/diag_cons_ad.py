#!/usr/bin/env python3
"""Diagnose conservative AD Newton — check Jacobian, residual, search direction."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from solver.denner_1d.eos.eos_class import create_eos
from solver.denner_1d.assembly_5eq_ad import (
    _get_eos_params, _mixture_p_from_Q_anp, _mixture_T_from_Q_anp,
    make_residual_cons_ad, compute_jacobian_ad, compute_K_ad,
    compute_face_velocity_ad, pack_Q, unpack_Q,
    _AUTOGRAD_AVAILABLE,
)

ph_water = {
    'type': 'nasg', 'gamma': 1.187,
    'p_inf': 7.028e8, 'pinf': 7.028e8,
    'b': 6.61e-4, 'b_covolume': 6.61e-4,
    'kappa_v': 3610.0, 'eta': -1.177788e6,
}
ph_air = {
    'type': 'ideal', 'gamma': 1.4,
    'p_inf': 0.0, 'pinf': 0.0,
    'b': 0.0, 'b_covolume': 0.0,
    'kappa_v': 717.5, 'eta': 0.0,
}

N = 10
dx = 0.1
ph1_params = _get_eos_params(ph_water)
ph2_params = _get_eos_params(ph_air)
eos1 = create_eos(ph_water)
eos2 = create_eos(ph_air)

# Initial state
x_cells = np.linspace(0.05, 0.95, N)
a1 = np.where((x_cells >= 0.4) & (x_cells <= 0.6), 1.0, 0.0)
a1 = np.clip(a1, 1e-10, 1.0 - 1e-10)
a2 = 1.0 - a1
p = np.full(N, 1e5)
u = np.full(N, 1.0)
T = np.full(N, 300.0)

rho1 = eos1.rho(p, T)
rho2 = eos2.rho(p, T)
rho = a1 * rho1 + a2 * rho2

a1r1 = a1 * rho1
a2r2 = a2 * rho2
ru = rho * u
rE = a1 * eos1.e_vol(p, T) + a2 * eos2.e_vol(p, T) + 0.5 * rho * u**2

Q_n = pack_Q(a1r1, a2r2, ru, rE, a1)

print("=== Initial State ===")
print(f"  rho1={rho1[0]:.4f}  rho2={rho2[0]:.4f}")
print(f"  a1r1 range: [{a1r1.min():.4e}, {a1r1.max():.4e}]")
print(f"  a2r2 range: [{a2r2.min():.4e}, {a2r2.max():.4e}]")
print(f"  rE range:   [{rE.min():.4e}, {rE.max():.4e}]")
print(f"  Q range:    [{Q_n.min():.4e}, {Q_n.max():.4e}]")

# Check algebraic p recovery
p_rec = np.array(_mixture_p_from_Q_anp(a1r1, a2r2, ru, rE, a1, ph1_params, ph2_params))
print(f"\n=== Algebraic p Recovery ===")
print(f"  p_original = {p[0]:.6e}")
print(f"  p_recovered = {p_rec[0]:.6e}")
print(f"  max |p_rec - p| = {np.max(np.abs(p_rec - p)):.3e}")

# Check algebraic T recovery
T_rec = np.array(_mixture_T_from_Q_anp(a1r1, a2r2, a1, p_rec, ph1_params, ph2_params))
print(f"\n=== Algebraic T Recovery ===")
print(f"  T_original = {T[0]:.6e}")
print(f"  T_recovered = {T_rec[0]:.6e}")
print(f"  max |T_rec - T| = {np.max(np.abs(T_rec - T)):.3e}")

# Compute dt (acoustic CFL)
c1 = eos1.c(p, T)
c2 = eos2.c(p, T)
c_wood_sq = 1.0 / (a1 / (rho1 * c1**2) + a2 / (rho2 * c2**2) + 1e-300) / rho
c_wood = np.sqrt(c_wood_sq)
c_max = np.max(np.abs(u) + np.sqrt(np.maximum(c_wood_sq, 0)))
dt = 0.5 * dx / (c_max + 1e-300)
print(f"\n=== Time Step ===")
print(f"  c_wood_max = {np.max(np.sqrt(np.maximum(c_wood_sq,0))):.2f}")
print(f"  dt = {dt:.6e}")

# Compute MWI face velocity
theta, u_bar, d_hat, rho_star = compute_face_velocity_ad(
    u, p, rho, dx, dt, 'periodic', 'periodic')
print(f"\n=== MWI Face Velocity ===")
print(f"  theta: [{theta.min():.6f}, {theta.max():.6f}]")
print(f"  should be ≈ 1.0 for uniform u=1")

# Compute K
K = compute_K_ad(a1, p, T, ph_water, ph_air)
print(f"\n=== K Coefficient ===")
print(f"  K range: [{K.min():.6e}, {K.max():.6e}]")

# Build residual and evaluate at Q_n
res_func = make_residual_cons_ad(
    Q_n, N, dx, dt, ph1_params, ph2_params,
    'periodic', 'periodic', theta, K)

R0 = np.array(res_func(Q_n))
print(f"\n=== Residual at Q_n ===")
print(f"  |R| = {np.linalg.norm(R0):.3e}")
print(f"  R_m1 = {R0[0*N:1*N]}")
print(f"  R_m2 = {R0[1*N:2*N]}")
print(f"  R_mom = {R0[2*N:3*N]}")
print(f"  R_en = {R0[3*N:4*N]}")
print(f"  R_a1 = {R0[4*N:5*N]}")

# Compute Jacobian
from autograd import jacobian as ad_jacobian
J_func = ad_jacobian(res_func)
J = np.array(J_func(Q_n))

print(f"\n=== Jacobian ===")
print(f"  J shape: {J.shape}")
print(f"  J max |entry|: {np.max(np.abs(J)):.3e}")
print(f"  J cond number: {np.linalg.cond(J):.3e}")
print(f"  J rank: {np.linalg.matrix_rank(J)} / {J.shape[0]}")

# Check diagonal dominance
diag = np.abs(np.diag(J))
row_sum = np.sum(np.abs(J), axis=1) - diag
print(f"  Diagonal dominance: min diag/row_sum = {np.min(diag / (row_sum + 1e-300)):.3e}")

# Solve J*dQ = -R
try:
    dQ = np.linalg.solve(J, -R0)
    print(f"\n=== Newton Direction ===")
    print(f"  |dQ| = {np.linalg.norm(dQ):.3e}")
    print(f"  dQ blocks:")
    print(f"    d(a1r1): [{dQ[0*N:1*N].min():.3e}, {dQ[0*N:1*N].max():.3e}]")
    print(f"    d(a2r2): [{dQ[1*N:2*N].min():.3e}, {dQ[1*N:2*N].max():.3e}]")
    print(f"    d(ru):   [{dQ[2*N:3*N].min():.3e}, {dQ[2*N:3*N].max():.3e}]")
    print(f"    d(rE):   [{dQ[3*N:4*N].min():.3e}, {dQ[3*N:4*N].max():.3e}]")
    print(f"    d(a1):   [{dQ[4*N:5*N].min():.3e}, {dQ[4*N:5*N].max():.3e}]")

    # Check if Newton direction reduces residual
    for omega in [1.0, 0.5, 0.25, 0.1, 0.01, 0.001]:
        Q_trial = Q_n + omega * dQ
        # Bounds
        a1_t = Q_trial[4*N:5*N]
        a1_t = np.clip(a1_t, 1e-10, 1.0 - 1e-10)
        Q_trial[4*N:5*N] = a1_t
        Q_trial[0*N:1*N] = np.maximum(Q_trial[0*N:1*N], 1e-20)
        Q_trial[1*N:2*N] = np.maximum(Q_trial[1*N:2*N], 1e-20)

        try:
            R_trial = np.array(res_func(Q_trial))
            R_trial_norm = np.linalg.norm(R_trial)
            improved = "✓" if R_trial_norm < np.linalg.norm(R0) else "✗"
            print(f"    ω={omega:.4f}: |R_trial|={R_trial_norm:.3e}  {improved}")
        except Exception as e:
            print(f"    ω={omega:.4f}: FAILED ({e})")

except np.linalg.LinAlgError as e:
    print(f"\n=== Solve failed: {e} ===")

# Also check: is J approximately I/dt on the diagonal (temporal dominance)?
print(f"\n=== Temporal Diagonal Check ===")
print(f"  1/dt = {1.0/dt:.3e}")
for k, name in enumerate(['a1r1', 'a2r2', 'ru', 'rE', 'a1']):
    diag_vals = [J[k*N+i, k*N+i] for i in range(min(N, 5))]
    print(f"  J[{name},{name}] diag: {[f'{v:.3e}' for v in diag_vals]}")
