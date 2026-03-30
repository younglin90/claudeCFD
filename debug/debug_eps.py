"""Check epsilon_v values at interface and compare QC vs FC vs APEC fluxes."""
import sys
sys.path.insert(0, r'D:\work\claude_code\Abgrall_solve_1')
import numpy as np
from pressure_eq import (initial_condition, srk_c2, epsilon_v, srk_rhoe, muscl_lr, _minmod)

N    = 501
dx   = 1.0 / N
x    = np.linspace(dx/2, 1 - dx/2, N)

r1, r2, u, rhoE, T, p = initial_condition(x, 5e6)
rho = r1 + r2
rhoe = rhoE - 0.5*rho*u**2

eps0 = epsilon_v(r1, r2, T, 0)
eps1 = epsilon_v(r1, r2, T, 1)

# Find interface cells (largest r1 gradient)
dr1 = np.abs(np.gradient(r1))
i_intf = np.argsort(dr1)[-10:]  # top 10 cells by gradient

print("Interface cell values (top 10 by |dr1/dx|):")
print(f"{'i':>5} {'x':>7} {'r1':>10} {'r2':>10} {'T':>8} {'p/1e6':>8} "
      f"{'eps0':>14} {'eps1':>12} {'rhoe':>14}")
for i in sorted(i_intf):
    print(f"{i:5d} {x[i]:7.4f} {r1[i]:10.2f} {r2[i]:10.2f} {T[i]:8.2f} {p[i]/1e6:8.4f} "
          f"{eps0[i]:14.2f} {eps1[i]:12.2f} {rhoe[i]:14.2f}")

# Check: eps0*r1 + eps1*r2 vs rhoe
print(f"\nCheck eps0*r1 + eps1*r2 vs rhoe at interface cells:")
print(f"{'i':>5} {'eps0*r1+eps1*r2':>20} {'rhoe':>14} {'ratio':>10}")
for i in sorted(i_intf):
    qc_rhoe = eps0[i]*r1[i] + eps1[i]*r2[i]
    print(f"{i:5d} {qc_rhoe:20.2f} {rhoe[i]:14.2f} {qc_rhoe/rhoe[i] if rhoe[i]!=0 else float('nan'):10.4f}")

# Compare QC flux vs FC at interface m=250
m = 250
mp = m + 1
eps0_h = 0.5*(eps0[m] + eps0[mp])
eps1_h = 0.5*(eps1[m] + eps1[mp])
rhoe_h = 0.5*(rhoe[m] + rhoe[mp])
rho_h = 0.5*(rho[m] + rho[mp])
u_h = 0.5*(u[m] + u[mp])
p_h = 0.5*(p[m] + p[mp])
r1_h = 0.5*(r1[m] + r1[mp])
r2_h = 0.5*(r2[m] + r2[mp])
rhoE_h = 0.5*(rhoE[m] + rhoE[mp])

# MUSCL reconstructed values
r1L, r1R = muscl_lr(r1)
r2L, r2R = muscl_lr(r2)
uL, uR = muscl_lr(u)
pL, pR = muscl_lr(p)
lam_cell = np.abs(u) + np.sqrt(srk_c2(r1, r2, T))
lam = np.maximum(lam_cell, np.roll(lam_cell, -1))

F1 = 0.5*(r1L*uL + r1R*uR) - 0.5*lam*(r1R - r1L)
F2 = 0.5*(r2L*uL + r2R*uR) - 0.5*lam*(r2R - r2L)

print(f"\nAt interface m={m} (x={x[m]:.4f} to x={x[mp]:.4f}):")
print(f"  eps0_h = {eps0_h:.2f}, eps1_h = {eps1_h:.2f}")
print(f"  rhoe_h (FC) = {rhoe_h:.2f}")
print(f"  eps0_h*r1_h + eps1_h*r2_h (QC) = {eps0_h*r1_h + eps1_h*r2_h:.2f}")
print(f"  F1[m] = {F1[m]:.4f}, F2[m] = {F2[m]:.4f}")

# FC energy flux centered
FE_FC_cen = (rhoe_h + 0.5*rho_h*u_h**2 + p_h)*u_h
# QC energy flux (centered part)
FE_QC_cen = (eps0_h*r1_h + eps1_h*r2_h + 0.5*rho_h*u_h**2 + p_h)*u_h

# Full QC
Frho = F1 + F2
FE_QC = eps0_h*F1[m] + eps1_h*F2[m] + 0.5*u_h**2*Frho[m] + p_h*u_h
FE_FC = FE_FC_cen - 0.5*lam[m]*(rhoE[mp] - rhoE[m])

print(f"\n  FE_FC = {FE_FC:.6e}")
print(f"  FE_QC = {FE_QC:.6e}")
print(f"  Ratio FE_QC/FE_FC = {FE_QC/FE_FC:.4f}")
print(f"  FE_QC - FE_FC = {FE_QC - FE_FC:.6e}")
print(f"  FE_FC_cen = {FE_FC_cen:.6e}")
print(f"  FE_QC_cen = {FE_QC_cen:.6e}")
print(f"  (QC_cen - FC_cen) = {FE_QC_cen - FE_FC_cen:.6e}")

# Also check corr
dr1 = r1[mp] - r1[m]
dr2 = r2[mp] - r2[m]
corr = 0.5*(eps0[mp] - eps0[m])*0.5*dr1 + 0.5*(eps1[mp] - eps1[m])*0.5*dr2
print(f"\n  APEC centered corr = {corr:.6e}")
print(f"  rhoe_h - corr = {rhoe_h - corr:.2f}")
print(f"  APEC centered flux = {(rhoe_h - corr + 0.5*rho_h*u_h**2 + p_h)*u_h:.6e}")
