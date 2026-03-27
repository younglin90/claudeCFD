"""Diagnose APEC divergence — print epsilon values and correction terms."""
import sys
sys.path.insert(0, r'D:\work\claude_code\Abgrall_solve_1')
import numpy as np
from apec_1d import (initial_condition, epsilon_v, srk_p, srk_c2, prim,
                      _mix, srk_rhoe, _dpdT, _dpdr, _drhoedr, srk_Cv)

N   = 51
dx  = 1.0 / N
x   = np.linspace(dx/2, 1 - dx/2, N)

r1, r2, u, rhoE, T, p = initial_condition(x, 5e6)
rho = r1 + r2
rhoe = rhoE - 0.5*rho*u**2

print("\n--- Initial state at interface cells (cells 22-28 of 51) ---")
idx = slice(22, 28)
print(f"{'cell':>5} {'r1':>9} {'r2':>9} {'rho':>9} {'T':>8} {'p/1e6':>8} {'rhoe':>11}")
for i in range(22, 28):
    print(f"{i:5d} {r1[i]:9.2f} {r2[i]:9.2f} {rho[i]:9.2f} {T[i]:8.2f} {p[i]/1e6:8.4f} {rhoe[i]:11.1f}")

eps0 = epsilon_v(r1, r2, T, 0)
eps1 = epsilon_v(r1, r2, T, 1)

print("\n--- epsilon values at interface ---")
print(f"{'cell':>5} {'eps0':>12} {'eps1':>12}")
for i in range(22, 28):
    print(f"{i:5d} {eps0[i]:12.2f} {eps1[i]:12.2f}")

print("\n--- APEC correction at interface interfaces ---")
r1p = np.roll(r1, -1); r2p = np.roll(r2, -1)
dr1 = r1p - r1; dr2 = r2p - r2
eps0p = np.roll(eps0, -1); eps1p = np.roll(eps1, -1)

corr = 0.5*(eps0p - eps0)*0.5*dr1 + 0.5*(eps1p - eps1)*0.5*dr2
rhoe_h_std  = 0.5*(rhoe + np.roll(rhoe, -1))
rhoe_h_apec = rhoe_h_std - corr

print(f"{'intf':>5} {'rhoe_h_std':>12} {'corr':>12} {'rhoe_h_apec':>12}")
for i in range(22, 28):
    print(f"{i:5d} {rhoe_h_std[i]:12.1f} {corr[i]:12.1f} {rhoe_h_apec[i]:12.1f}")

print("\n--- PE-consistent dissipation vs standard ---")
u_h   = 0.5*(u + np.roll(u, -1))
rho_h = 0.5*(rho + np.roll(rho, -1))
du    = np.roll(u, -1) - u
eps0_h = 0.5*(eps0 + eps0p)
eps1_h = 0.5*(eps1 + eps1p)

drhoE_pep = eps0_h*dr1 + eps1_h*dr2 + rho_h*u_h*du
rhoE_arr  = rhoe + 0.5*rho*u**2
drhoE_std = np.roll(rhoE_arr, -1) - rhoE_arr

print(f"{'intf':>5} {'drhoE_std':>12} {'drhoE_pep':>12} {'ratio':>8}")
for i in range(22, 28):
    ratio = drhoE_pep[i] / (drhoE_std[i]+1e-10)
    print(f"{i:5d} {drhoE_std[i]:12.1f} {drhoE_pep[i]:12.1f} {ratio:8.3f}")

# Check if PE-consistent dissipation has same sign as standard
print("\n--- Sign check for dissipation stability ---")
same_sign = np.sign(drhoE_std) == np.sign(drhoE_pep)
print(f"Interfaces with same sign: {np.sum(same_sign)} / {N}")
print(f"Interfaces with OPPOSITE sign (destabilizing): {np.sum(~same_sign)} / {N}")
print(f"Indices with opposite sign: {np.where(~same_sign)[0]}")

# Check first-order (no MUSCL) APEC stability condition
c2 = srk_c2(r1, r2, T)
lam = np.abs(u) + np.sqrt(c2)
print(f"\n--- Wave speeds ---")
print(f"max |u|   = {np.max(np.abs(u)):.1f} m/s")
print(f"max c     = {np.max(np.sqrt(c2)):.1f} m/s")
print(f"max lambda= {np.max(lam):.1f} m/s")
