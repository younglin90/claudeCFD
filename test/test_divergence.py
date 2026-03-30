"""Track PE history for FC vs APEC to find when FC diverges."""
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
print(f"N={N}  dt={dt:.3e} s  u_init={float(u.mean()):.2f} m/s\n")

N_STEPS = 2000
PRINT_EVERY = 100

for scheme in ['FC', 'APEC']:
    U = [r1.copy(), r2.copy(), rho*u, rhoE.copy()]
    T_cur = T.copy()
    diverged = False
    print(f"--- {scheme} ---")
    for step in range(1, N_STEPS+1):
        try:
            U, T_cur, p_cur = rkstep(U, scheme, dx, dt, T_cur)
            pe = pe_err(p_cur, 5e6)
            if step % PRINT_EVERY == 0:
                print(f"  step {step:5d}  t={step*dt:.4e} s  PE={pe:.4e}")
            if pe > 5.0 or not np.isfinite(pe):
                print(f"  DIVERGED at step {step}  t={step*dt:.4e} s  PE={pe:.4e}")
                diverged = True
                break
        except Exception as e:
            print(f"  CRASHED at step {step}: {e}")
            diverged = True
            break
    if not diverged:
        print(f"  Survived {N_STEPS} steps  final PE={pe_err(p_cur, 5e6):.4e}")
    print()
