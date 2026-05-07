"""
Round 3 검증: 02-A Test A (NASG Abgrall advection)
"""
import sys
sys.path.insert(0, '/home/younglin90/work/claude_code/claudeCFD')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from solver.He2024.explicit_mmacm_ex import solve_IMEX
from solver.He2024.eos_general import NASGEOS, IdealEOS

# EOS setup
eos_water = NASGEOS(gamma=1.187, pinf=7.028e8, b=6.61e-4, kv=3610, eta=-1.177788e6)
eos_air = IdealEOS(gamma=1.4, kv=717.5)

# Grid
N = 10
L = 1.0
dx = L / N
x = np.arange(0.5*dx, L, dx)

# Initial condition
u0 = 1.0
p0 = 1e5
T0 = 300.0
rho1_0 = 998.0
rho2_0 = 1.225

# Water at [0.4, 0.6]
alpha1_0 = np.where((x >= 0.4) & (x < 0.6), 1.0, 0.0)

# Conservative variables
a1r1_0 = alpha1_0 * rho1_0
a2r2_0 = (1.0 - alpha1_0) * rho2_0
ru_0 = (alpha1_0 * rho1_0 + (1.0 - alpha1_0) * rho2_0) * u0
rE_0 = alpha1_0 * (eos_water.energy(rho1_0, p0) + 0.5*u0**2) * rho1_0 + \
       (1.0 - alpha1_0) * (eos_air.energy(rho2_0, p0) + 0.5*u0**2) * rho2_0
a1_0 = alpha1_0

# Time integration
dt = 0.01
t_end = 1.0
max_steps = int(t_end / dt)

print("="*70)
print("Validation 02-A Test A: NASG Abgrall Advection (Round 3)")
print("="*70)
print(f"Grid: N={N}, dx={dx}")
print(f"Time: t_end={t_end} s, dt={dt} s, max_steps={max_steps}")
print()

try:
    # Call solve_IMEX with correct signature
    result = solve_IMEX(
        ph1=eos_water, ph2=eos_air,
        a1r1_0=a1r1_0, a2r2_0=a2r2_0, ru_0=ru_0, rE_0=rE_0, a1_0=a1_0,
        dx=dx, t_end=t_end, cfl=0.4,
        bc_l='periodic', bc_r='periodic',
        max_steps=max_steps,
        max_newton=3,
        print_interval=50
    )
    
    print(f"\n✓ Solve completed")
    print(f"Final time: {result.get('time', 'unknown'):.6f} s")
    
    # Extract final state
    Q_final = result.get('Q_final', None)
    if Q_final is None:
        print("✗ ERROR: No Q_final in result")
    else:
        a1r1_f = Q_final['a1r1']
        a2r2_f = Q_final['a2r2']
        ru_f = Q_final['ru']
        rE_f = Q_final['rE']
        a1_f = Q_final['a1']
        
        # Diagnostics
        rho_f = a1r1_f + a2r2_f
        u_f = ru_f / np.maximum(rho_f, 1e-10)
        
        # Pressure from mixture EOS
        p_f = np.zeros(N)
        for i in range(N):
            if a1_f[i] > 1e-4:
                rho1_i = np.maximum(a1r1_f[i] / a1_f[i], 1e-10)
                e1_i = (rE_f[i] / a1_f[i] - 0.5*u_f[i]**2) / rho1_i
                p_f[i] = eos_water.pressure(rho1_i, e1_i)
            else:
                rho2_i = np.maximum(a2r2_f[i] / (1.0 - a1_f[i]), 1e-10)
                e2_i = (rE_f[i] / (1.0 - a1_f[i]) - 0.5*u_f[i]**2) / rho2_i
                p_f[i] = eos_air.pressure(rho2_i, e2_i)
        
        # Error metrics
        err_p = np.max(np.abs(p_f - p0)) / p0
        err_u = np.max(np.abs(u_f - u0))
        
        print(f"\nMetrics:")
        print(f"  err_p = {err_p:.6e} (target < 1e-2)")
        print(f"  err_u = {err_u:.6e} m/s (target < 1e-2)")
        
        pass_p = err_p < 1e-2
        pass_u = err_u < 1e-2
        print(f"\nResult: {' PASS' if (pass_p and pass_u) else ' FAIL'}")
        
except Exception as e:
    import traceback
    print(f"\n✗ ERROR:")
    traceback.print_exc()

print()
print("="*70)
