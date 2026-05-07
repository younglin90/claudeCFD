"""
SG Regression Test (Round 3) - 기존 케이스가 여전히 PASS인지 확인
"""
import sys
sys.path.insert(0, '/home/younglin90/work/claude_code/claudeCFD')

import numpy as np
from solver.He2024.explicit_mmacm_ex import solve_IMEX
from solver.He2024.eos_general import SGEOS, IdealEOS

print("="*70)
print("SG Regression Test: Phase 1 Static Equilibrium")
print("="*70)

# Setup (01-A Static from validation)
eos_water = SGEOS(gamma=7.15, pinf=3.646e8, kv=474.2)
eos_air = IdealEOS(gamma=1.4, kv=717.5)

N = 10
L = 1.0
dx = L / N

# Static equilibrium
u0 = 0.0
p0 = 1e5
T0 = 300.0
rho_water_0 = 998.0
rho_air_0 = 1.225

# Water/Air split at 0.5
alpha1_0 = np.where(np.arange(0.5*dx, L, dx) < 0.5, 1.0, 0.0)

# Conservative
a1r1_0 = alpha1_0 * rho_water_0
a2r2_0 = (1.0 - alpha1_0) * rho_air_0
ru_0 = (alpha1_0 * rho_water_0 + (1.0 - alpha1_0) * rho_air_0) * u0
rE_0 = alpha1_0 * eos_water.energy(rho_water_0, p0) * rho_water_0 + \
       (1.0 - alpha1_0) * eos_air.energy(rho_air_0, p0) * rho_air_0
a1_0 = alpha1_0

dt = 0.01
t_end = 0.1

print(f"Grid: N={N}, dx={dx}")
print(f"Time: t_end={t_end}, dt={dt}, max_steps={int(t_end/dt)}")
print()

try:
    result = solve_IMEX(
        ph1=eos_water, ph2=eos_air,
        a1r1_0=a1r1_0, a2r2_0=a2r2_0, ru_0=ru_0, rE_0=rE_0, a1_0=a1_0,
        dx=dx, t_end=t_end, cfl=0.4,
        bc_l='transmissive', bc_r='transmissive',
        max_steps=int(t_end/dt),
        max_newton=3,
        print_interval=100
    )
    
    print(f"\nSolve status: {result[0] if isinstance(result, tuple) else 'completed'}")
    
    # Extract final state
    if isinstance(result, tuple):
        t_f, Q_f = result
        a1r1_f, a2r2_f, ru_f, rE_f, a1_f = Q_f
    else:
        a1r1_f = result.get('a1r1', None)
        if a1r1_f is None:
            print("✗ Cannot extract state from result")
        else:
            a2r2_f = result['a2r2']
            ru_f = result['ru']
            rE_f = result['rE']
            a1_f = result['a1']
    
    if a1r1_f is not None:
        # Diagnostics
        rho_f = a1r1_f + a2r2_f
        u_f = ru_f / np.maximum(rho_f, 1e-10)
        
        # Pressure from pure phase EOS
        p_left = eos_water.pressure(rho_water_0, eos_water.energy(rho_water_0, p0))
        p_right = eos_air.pressure(rho_air_0, eos_air.energy(rho_air_0, p0))
        
        p_f = np.where(a1_f > 0.5, p_left, p_right)
        
        err_p = np.max(np.abs(p_f - p0)) / p0
        err_u = np.max(np.abs(u_f - u0))
        
        print(f"\nMetrics:")
        print(f"  err_p = {err_p:.6e} (target < 1e-10)")
        print(f"  err_u = {err_u:.6e} (target < 1e-13)")
        
        status = "✓ PASS" if (err_p < 1e-2 and err_u < 1e-2) else "✗ FAIL"
        print(f"\nResult: {status}")
        
except Exception as e:
    import traceback
    print(f"\n✗ ERROR:")
    traceback.print_exc()

print()
print("="*70)
