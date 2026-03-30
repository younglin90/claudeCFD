"""Compare FC vs APEC PE error at steps 1-20 at N=501."""
import sys
sys.path.insert(0, r'D:\work\claude_code\Abgrall_solve_1')
import numpy as np
from pressure_eq import (initial_condition, rkstep, srk_c2, pe_err)

N    = 501
dx   = 1.0 / N
x    = np.linspace(dx/2, 1 - dx/2, N)
CFL  = 0.3

r1, r2, u, rhoE, T, p = initial_condition(x, 5e6)
rho  = r1 + r2
c2   = srk_c2(r1, r2, T)
lam  = float(np.max(np.abs(u) + np.sqrt(c2)))
dt   = CFL * dx / lam
print(f"N={N}  dt={dt:.3e} s  lambda_max={lam:.1f} m/s\n")

results = {}
for scheme in ['FC', 'APEC', 'PEqC']:
    U_init = [r1.copy(), r2.copy(), rho*u, rhoE.copy()]
    if scheme == 'PEqC':
        U_init = [r1.copy(), r2.copy(), rho*u, p.copy()]
    T_cur = T.copy()
    U = U_init

    pes = []
    for step in range(20):
        U, T_cur, p_cur = rkstep(U, scheme, dx, dt, T_cur)
        pes.append(pe_err(p_cur, 5e6))
    results[scheme] = pes
    print(f"{scheme}: step1={pes[0]:.3e}  step5={pes[4]:.3e}  step20={pes[19]:.3e}")

print(f"\nRatio FC/APEC  step1 = {results['FC'][0]/results['APEC'][0]:.2f}")
print(f"Ratio FC/APEC  step5 = {results['FC'][4]/results['APEC'][4]:.2f}")
print(f"Ratio FC/APEC  step20= {results['FC'][19]/results['APEC'][19]:.2f}")
print(f"PEqC  step1 = {results['PEqC'][0]:.3e}  (should be near 0)")
