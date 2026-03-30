"""Compare FC vs APEC at N=51 where interface is sharper (3-4 cells)."""
import sys
sys.path.insert(0, r'D:\work\claude_code\Abgrall_solve_1')
import numpy as np
from pressure_eq import (initial_condition, rkstep, srk_c2, pe_err, epsilon_v)

for N in [51, 101, 201, 501]:
    dx   = 1.0 / N
    x    = np.linspace(dx/2, 1 - dx/2, N)
    CFL  = 0.3

    r1, r2, u, rhoE, T, p = initial_condition(x, 5e6)
    rho  = r1 + r2
    c2   = srk_c2(r1, r2, T)
    lam  = float(np.max(np.abs(u) + np.sqrt(c2)))
    dt   = CFL * dx / lam

    # Compute epsilon and show max corr at initial condition
    eps0 = epsilon_v(r1, r2, T, 0)
    eps1 = epsilon_v(r1, r2, T, 1)
    dr1 = np.roll(r1, -1) - r1
    dr2 = np.roll(r2, -1) - r2
    corr = 0.5*(np.roll(eps0,-1) - eps0)*0.5*dr1 + 0.5*(np.roll(eps1,-1) - eps1)*0.5*dr2
    rhoe = rhoE - 0.5*rho*u**2
    rhoe_h = 0.5*(rhoe + np.roll(rhoe, -1))
    corr_frac = float(np.max(np.abs(corr)) / np.max(np.abs(rhoe_h)))

    pe_fc = []
    pe_apec = []
    for scheme, lst in [('FC', pe_fc), ('APEC', pe_apec)]:
        U = [r1.copy(), r2.copy(), rho*u, rhoE.copy()]
        T_cur = T.copy()
        for step in range(20):
            try:
                U, T_cur, p_cur = rkstep(U, scheme, dx, dt, T_cur)
                lst.append(pe_err(p_cur, 5e6))
            except:
                lst.append(float('nan'))
                break

    ratio = pe_fc[0]/pe_apec[0] if (pe_apec[0] and pe_apec[0]>0) else float('nan')
    print(f"N={N:3d}: corr/rhoe_max={corr_frac:.2e}  "
          f"FC_step1={pe_fc[0]:.3e}  APEC_step1={pe_apec[0]:.3e}  "
          f"ratio={ratio:.3f}  "
          f"FC_step20={pe_fc[-1]:.3e}  APEC_step20={pe_apec[-1]:.3e}")
