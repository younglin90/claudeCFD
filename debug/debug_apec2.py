"""Diagnose APEC at the ACTUAL interface cells."""
import sys
sys.path.insert(0, r'D:\work\claude_code\Abgrall_solve_1')
import numpy as np
from pressure_eq import (initial_condition, epsilon_v, srk_p, srk_c2, prim,
                      _mix, srk_rhoe, _dpdT, _dpdr, _drhoedr, srk_Cv,
                      interface_fluxes, rhs)

N   = 51
dx  = 1.0 / N
x   = np.linspace(dx/2, 1 - dx/2, N)

r1, r2, u, rhoE, T, p = initial_condition(x, 5e6)
rho = r1 + r2
rhoe = rhoE - 0.5*rho*u**2

print("\n--- Full density profile ---")
for i in range(N):
    print(f"{i:3d}  x={x[i]:.4f}  r1={r1[i]:7.2f}  r2={r2[i]:7.2f}  "
          f"rho={rho[i]:7.2f}  T={T[i]:7.2f}  p/1e6={p[i]/1e6:.6f}")

eps0 = epsilon_v(r1, r2, T, 0)
eps1 = epsilon_v(r1, r2, T, 1)

r1p = np.roll(r1, -1); r2p = np.roll(r2, -1)
dr1 = r1p - r1; dr2 = r2p - r2
eps0p = np.roll(eps0, -1); eps1p = np.roll(eps1, -1)

corr = 0.5*(eps0p - eps0)*0.5*dr1 + 0.5*(eps1p - eps1)*0.5*dr2
rhoe_h_std  = 0.5*(rhoe + np.roll(rhoe, -1))
rhoe_h_apec = rhoe_h_std - corr

print(f"\n--- APEC correction magnitude (all interfaces) ---")
for i in range(N):
    if abs(dr1[i]) > 1 or abs(dr2[i]) > 1:  # only show where densities change
        rel = abs(corr[i]) / (abs(rhoe_h_std[i]) + 1e-10)
        print(f"intf {i}-{i+1}: Ddr1={dr1[i]:8.2f} Ddr2={dr2[i]:8.2f}  "
              f"corr={corr[i]:12.1f}  rhoe_h={rhoe_h_std[i]:12.1f}  rel={rel:.2e}")

# Run one RHS call for APEC and FC
c2 = srk_c2(r1, r2, T)
lam_c = np.abs(u) + np.sqrt(c2)
rhoU = rho * u

eps_pair = (eps0, eps1)
F1a, F2a, FUa, FEa = interface_fluxes(r1, r2, u, rhoe, p, T, lam_c, 'APEC', eps_pair)
F1f, F2f, FUf, FEf = interface_fluxes(r1, r2, u, rhoe, p, T, lam_c, 'FC')

dEa = -(FEa - np.roll(FEa, 1)) / dx
dEf = -(FEf - np.roll(FEf, 1)) / dx

print(f"\n--- Energy RHS: APEC vs FC at interface cells ---")
for i in range(N):
    if abs(dEa[i] - dEf[i]) > 1e-3 * (abs(dEa[i]) + 1):
        print(f"cell {i:3d}: dE_APEC={dEa[i]:12.3e}  dE_FC={dEf[i]:12.3e}  "
              f"diff={dEa[i]-dEf[i]:12.3e}  r1={r1[i]:.1f} r2={r2[i]:.1f}")

print(f"\n--- Max |dE_APEC - dE_FC| = {np.max(np.abs(dEa-dEf)):.3e}")
print(f"--- Max |dE_FC| = {np.max(np.abs(dEf)):.3e}")
