"""Time N=501 for 200 steps to estimate full run cost."""
import sys, time
sys.path.insert(0, r'D:\work\claude_code\Abgrall_solve_1')
import numpy as np
from apec_1d import (initial_condition, rhs, rkstep, srk_c2,
                     pe_err, energy_err)

N    = 501
dx   = 1.0 / N
x    = np.linspace(dx/2, 1 - dx/2, N)
CFL  = 0.3
p_inf = 5e6

r1, r2, u, rhoE, T, p = initial_condition(x, p_inf)
rho = r1 + r2
U   = [r1.copy(), r2.copy(), rho*u, rhoE.copy()]

t = 0.0
n_steps_target = 200
elapsed_per_step = []

for scheme in ['FC', 'APEC']:
    # Reset IC
    r1, r2, u, rhoE, T, p = initial_condition(x, p_inf)
    rho = r1 + r2
    U   = [r1.copy(), r2.copy(), rho*u, rhoE.copy()]
    T_cur = T.copy()
    t = 0.0

    t0 = time.time()
    for step in range(n_steps_target):
        r1_, r2_ = U[0], U[1]
        u_  = U[2] / np.maximum(r1_+r2_, 1e-30)
        c2_ = srk_c2(r1_, r2_, T_cur)
        lam = float(np.max(np.abs(u_) + np.sqrt(c2_)))
        dt  = CFL * dx / (lam + 1e-10)
        U, T_cur, p = rkstep(U, scheme, dx, dt, T_cur)
        t += dt
    elapsed = time.time() - t0

    dt_step = elapsed / n_steps_target
    # Estimate steps to t_end=0.2s
    lam_est = 1112.0  # m/s
    dt_est  = CFL * dx / lam_est
    n_steps_total = int(0.07 / dt_est)
    est_total_s = dt_step * n_steps_total
    print(f"{scheme}: {dt_step*1000:.1f} ms/step | "
          f"~{n_steps_total:,} steps for t=0.07s | "
          f"est {est_total_s/60:.1f} min total")

print("Done.")
