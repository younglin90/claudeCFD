"""Check PE error after 1, 5, 10 steps at N=501 to understand growth rate."""
import sys
sys.path.insert(0, r'D:\work\claude_code\Abgrall_solve_1')
import numpy as np
from pressure_eq import (initial_condition, rhs, rkstep, srk_c2,
                     pe_err, srk_p)

N    = 501
dx   = 1.0 / N
x    = np.linspace(dx/2, 1 - dx/2, N)
CFL  = 0.3

r1, r2, u, rhoE, T, p = initial_condition(x, 5e6)
rho = r1 + r2
U0  = [r1.copy(), r2.copy(), rho*u, rhoE.copy()]
T0  = T.copy()

print(f"\n--- Initial state ---")
print(f"PE_init  = {pe_err(p, 5e6):.3e}")
print(f"rho range: {r1.min():.1f} to {r1.max():.1f}  r2: {r2.min():.1f} to {r2.max():.1f}")
print(f"T   range: {T.min():.2f} to {T.max():.2f} K")
print(f"p   range: {p.min()/1e6:.6f} to {p.max()/1e6:.6f} MPa")

# Run FC scheme step by step and monitor PE
U = [arr.copy() for arr in U0]
T_cur = T0.copy()

c2  = srk_c2(r1, r2, T)
lam = float(np.max(np.abs(u) + np.sqrt(c2)))
dt  = CFL * dx / lam
print(f"\ndt = {dt:.3e} s  lambda_max = {lam:.1f} m/s")

print(f"\n--- PE after each step (FC, N={N}) ---")
for step in range(20):
    U, T_cur, p_cur = rkstep(U, 'FC', dx, dt, T_cur)
    pe = pe_err(p_cur, 5e6)
    p_max_dev = float(np.max(np.abs(p_cur - 5e6)))
    i_max = int(np.argmax(np.abs(p_cur - 5e6)))
    print(f"step {step+1:2d}: PE={pe:.4e}  max|dp|={p_max_dev:.2e} Pa  "
          f"at cell {i_max} (x={x[i_max]:.3f}  r1={U[0][i_max]:.1f}  r2={U[1][i_max]:.1f})")

# Also compare at N=51
print(f"\n--- Same at N=51 ---")
N2 = 51
dx2 = 1.0/N2
x2 = np.linspace(dx2/2, 1-dx2/2, N2)
r1b, r2b, ub, rhoEb, Tb, pb = initial_condition(x2, 5e6)
rhob = r1b + r2b
Ub = [r1b.copy(), r2b.copy(), rhob*ub, rhoEb.copy()]
Tc2 = Tb.copy()
c2b = srk_c2(r1b, r2b, Tb)
lamb = float(np.max(np.abs(ub) + np.sqrt(c2b)))
dtb = CFL * dx2 / lamb
print(f"dt={dtb:.3e} s  lambda_max={lamb:.1f} m/s")
for step in range(20):
    Ub, Tc2, pb_cur = rkstep(Ub, 'FC', dx2, dtb, Tc2)
    pe = pe_err(pb_cur, 5e6)
    print(f"step {step+1:2d}: PE={pe:.4e}")
