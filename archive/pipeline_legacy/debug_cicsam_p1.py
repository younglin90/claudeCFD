"""Debug CICSAM Phase 1 divergence — compute S* manually."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from solver.He2024.explicit_mmacm_ex import (solve_IMEX, cons_to_prim,
    _advective_rhs_imex, _tvd_reconstruct, _nvd_face, _ghost,
    _sg_sound_speed_sq, _EPS)

ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
ph2 = {'gamma': 2.35, 'pinf': 1e9, 'kv': 943.8, 'b': 6.61e-4, 'eta': -1167e3, 'q': 0.0}
N = 10; L = 1.0; dx = L / N
x = np.linspace(dx/2, L - dx/2, N)
p0 = 1e5; u0 = 1.0; T0 = 300.0
a1 = np.where((x >= 0.4) & (x <= 0.6), 1e-6, 1.0 - 1e-6)
a2 = 1.0 - a1
rho1 = p0 / ((ph1['gamma'] - 1.0) * ph1['kv'] * T0)
rho2 = (p0 + ph2['pinf']) / ((ph2['gamma'] - 1.0) * ph2['kv'] * T0)
print(f"rho1 (air) = {rho1:.4f}, rho2 (water) = {rho2:.4f}")

a1r1 = a1 * rho1 * np.ones(N)
a2r2 = a2 * rho2 * np.ones(N)
rho = a1r1 + a2r2
ru = rho * u0
gm1, gm2 = ph1['gamma'] - 1.0, ph2['gamma'] - 1.0
rho_e = a1 * (p0 + ph1['gamma'] * ph1['pinf']) / gm1 + a2 * (p0 + ph2['gamma'] * ph2['pinf']) / gm2
rE = rho_e + 0.5 * rho * u0**2

# Compute primitives at t=0
p, u_vel, T, rho1_, rho2_, c1, c2, c_wood = cons_to_prim(a1r1, a2r2, ru, rE, a1, ph1, ph2)
print(f"\nInitial state (cell-center):")
print(f"  a1: {a1}")
print(f"  rho1: {rho1_}")
print(f"  rho2: {rho2_}")
print(f"  p: {p}")
print(f"  u: {u_vel}")
print(f"  c_wood: {c_wood}")

# CICSAM alpha_face
u_ext = _ghost(u_vel, 'periodic', 'periodic', ng=1)
u_face_est = 0.5 * (u_ext[:-1] + u_ext[1:])
dt_use = dx * 0.4 / np.max(np.abs(u_vel) + c_wood)
alpha_face = _nvd_face(a1, u_face_est, dt_use, dx, 'periodic', 'periodic', cds='hyper_c')
print(f"\nCICSAM alpha_face: {alpha_face}")
print(f"  min={alpha_face.min():.4e}, max={alpha_face.max():.4e}")

# Reconstruct rho1, rho2, u, p with TVD
rho1L, rho1R = _tvd_reconstruct(rho1_, 'periodic', 'periodic')
rho2L, rho2R = _tvd_reconstruct(rho2_, 'periodic', 'periodic')
uL, uR = _tvd_reconstruct(u_vel, 'periodic', 'periodic')
pL, pR = _tvd_reconstruct(p, 'periodic', 'periodic')

a1L = alpha_face.copy(); a1R = alpha_face.copy()
a2L = 1 - a1L; a2R = 1 - a1R

rho_fL = a1L * rho1L + a2L * rho2L
rho_fR = a1R * rho1R + a2R * rho2R
print(f"\nFace densities:")
print(f"  rho_fL: {rho_fL}")
print(f"  rho_fR: {rho_fR}")

c1_fL = np.sqrt(_sg_sound_speed_sq(pL, rho1L, ph1['gamma'], ph1['pinf']))
c2_fL = np.sqrt(_sg_sound_speed_sq(pL, rho2L, ph2['gamma'], ph2['pinf']))
c1_fR = np.sqrt(_sg_sound_speed_sq(pR, rho1R, ph1['gamma'], ph1['pinf']))
c2_fR = np.sqrt(_sg_sound_speed_sq(pR, rho2R, ph2['gamma'], ph2['pinf']))
c_fL = np.maximum(c1_fL, c2_fL)
c_fR = np.maximum(c1_fR, c2_fR)
print(f"\nFace sound speeds:")
print(f"  c1_fL: {c1_fL}")
print(f"  c2_fL: {c2_fL}")
print(f"  c_fL (max): {c_fL}")

S_L = np.minimum(uL - c_fL, uR - c_fR)
S_R = np.maximum(uL + c_fL, uR + c_fR)
num_Ss = (pR - pL + rho_fL * uL * (S_L - uL) - rho_fR * uR * (S_R - uR))
den_Ss = rho_fL * (S_L - uL) - rho_fR * (S_R - uR)
print(f"\nS_L: {S_L}")
print(f"S_R: {S_R}")
print(f"num_Ss: {num_Ss}")
print(f"den_Ss: {den_Ss}")
S_star = num_Ss / den_Ss
print(f"S_star: {S_star}")
