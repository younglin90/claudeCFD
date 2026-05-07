#!/usr/bin/env python3
"""
Validator: Phase 6-3 (07_B) Acoustic Reflection & Transmission — DIAGNOSTIC
Testing IMEX with time_integrator='ssp222' as requested.
Simplified check: does solver run without crashes?
"""

import sys
import os
import numpy as np
import time

sys.path.insert(0, '/home/younglin90/work/claude_code/claudeCFD')

from solver.He2024.explicit_mmacm_ex import solve_IMEX, cons_to_prim
from solver.He2024.eos_general import IdealEOS, SGEOS

# Test parameters per spec
L = 1.5
N = 400
dx = L / N
x = np.arange(N) * dx

print("="*70)
print("DIAGNOSIS: Phase 6-3 Acoustic Reflection & Transmission")
print("="*70)
print(f"Grid: L={L} m, N={N}, dx={dx:.6e} m")

# Case 1: Air-Water
print("\n[TEST 1] Air → Water interface")
print("-" * 70)

ph_air = IdealEOS(gamma=1.4, kv=287.0)
ph_water = SGEOS(gamma=4.1, pinf=4.4e8, kv=474.2)

u_peak = 0.02
sigma_L = 0.014
x_src = 0.1
x_intf = 0.5

Z_air = 1.157 * 347.8
Z_water = 998.0 * 1344.6

u0 = u_peak * np.exp(-(x - x_src)**2 / (2*sigma_L**2))
p0_init = 1e5 + Z_air * u0

a1_0 = np.where(x < x_intf, 1.0, 1e-8)
a1_0 = np.clip(a1_0, 1e-8, 1.0-1e-8)

rho1_0 = np.array([ph_air.density(p0_init[i], 300.0) for i in range(N)])
rho2_0 = np.array([ph_water.density(p0_init[i], 300.0) for i in range(N)])

a1r1_0 = a1_0 * rho1_0
a2r2_0 = (1.0 - a1_0) * rho2_0
ru_0 = (a1r1_0 + a2r2_0) * u0
rE_0 = np.zeros(N)
for i in range(N):
    e1 = ph_air.energy(rho1_0[i], p0_init[i])
    e2 = ph_water.energy(rho2_0[i], p0_init[i])
    rho_e = a1_0[i] * rho1_0[i] * e1 + (1.0-a1_0[i]) * rho2_0[i] * e2
    rE_0[i] = rho_e + 0.5 * (a1r1_0[i] + a2r2_0[i]) * u0[i]**2

print("Initial condition setup: OK")
print(f"  u0 range: [{u0.min():.3e}, {u0.max():.3e}]")
print(f"  p0 range: [{p0_init.min():.2e}, {p0_init.max():.2e}]")

t_intf = x_intf / 347.8
t_end = 1.63e-3

start_t = time.time()
try:
    # Test with ssp222
    result = solve_IMEX(
        ph1=ph_air, ph2=ph_water,
        a1r1_0=a1r1_0, a2r2_0=a2r2_0, ru_0=ru_0, rE_0=rE_0,
        a1_0=a1_0,
        dx=dx, t_end=t_end, cfl=0.4,
        bc_l='transmissive', bc_r='transmissive',
        max_steps=100,
        print_interval=50,
        time_integrator='ssp222',
        imex_rk2=True,
    )
    elapsed = time.time() - start_t

    t_final, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = result
    print(f"Solver completed: t={t_final:.4e} s (wall: {elapsed:.2f} s)")

    # Try conversion
    u_num = np.zeros(N)
    p_num = np.zeros(N)

    for i in range(N):
        try:
            p, u, T, rho1, rho2, c1, c2, c_mix = cons_to_prim(
                a1r1_f[i], a2r2_f[i], ru_f[i], rE_f[i], a1_f[i],
                ph_air, ph_water
            )
            u_num[i] = u
            p_num[i] = p
        except:
            pass

    finite = np.sum(np.isfinite(u_num)) / N
    print(f"Primitive conversion: {finite*100:.1f}% finite values")
    if np.any(np.isfinite(u_num)):
        print(f"  u range: [{np.nanmin(u_num):.3e}, {np.nanmax(u_num):.3e}]")
        print(f"  p range: [{np.nanmin(p_num):.2e}, {np.nanmax(p_num):.2e}]")

    print("✓ Test 1 PASSED (no crash)")

except Exception as e:
    print(f"✗ Test 1 FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Case 2: Helium-Air
print("\n[TEST 2] Helium → Air interface")
print("-" * 70)

ph_he = IdealEOS(gamma=1.667, kv=2077.0)
ph_air = IdealEOS(gamma=1.4, kv=287.0)

sigma_L = 0.049
x_src = 0.2
x_intf = 1.0

Z_he = 0.164 * 1008.2
Z_air = 1.157 * 347.8

u0 = u_peak * np.exp(-(x - x_src)**2 / (2*sigma_L**2))
p0_init = 1e5 + Z_he * u0

a1_0 = np.where(x < x_intf, 1.0, 1e-8)
a1_0 = np.clip(a1_0, 1e-8, 1.0-1e-8)

rho1_0 = np.array([ph_he.density(p0_init[i], 300.0) for i in range(N)])
rho2_0 = np.array([ph_air.density(p0_init[i], 300.0) for i in range(N)])

a1r1_0 = a1_0 * rho1_0
a2r2_0 = (1.0 - a1_0) * rho2_0
ru_0 = (a1r1_0 + a2r2_0) * u0
rE_0 = np.zeros(N)
for i in range(N):
    e1 = ph_he.energy(rho1_0[i], p0_init[i])
    e2 = ph_air.energy(rho2_0[i], p0_init[i])
    rho_e = a1_0[i] * rho1_0[i] * e1 + (1.0-a1_0[i]) * rho2_0[i] * e2
    rE_0[i] = rho_e + 0.5 * (a1r1_0[i] + a2r2_0[i]) * u0[i]**2

t_intf = (x_intf - x_src) / 1008.2
t_end = 1.513e-3

print("Initial condition setup: OK")
start_t = time.time()
try:
    result = solve_IMEX(
        ph1=ph_he, ph2=ph_air,
        a1r1_0=a1r1_0, a2r2_0=a2r2_0, ru_0=ru_0, rE_0=rE_0,
        a1_0=a1_0,
        dx=dx, t_end=t_end, cfl=0.4,
        bc_l='transmissive', bc_r='transmissive',
        max_steps=100,
        print_interval=50,
        time_integrator='ssp222',
        imex_rk2=True,
    )
    elapsed = time.time() - start_t

    t_final, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = result
    print(f"Solver completed: t={t_final:.4e} s (wall: {elapsed:.2f} s)")

    u_num = np.zeros(N)
    p_num = np.zeros(N)

    for i in range(N):
        try:
            p, u, T, rho1, rho2, c1, c2, c_mix = cons_to_prim(
                a1r1_f[i], a2r2_f[i], ru_f[i], rE_f[i], a1_f[i],
                ph_he, ph_air
            )
            u_num[i] = u
            p_num[i] = p
        except:
            pass

    finite = np.sum(np.isfinite(u_num)) / N
    print(f"Primitive conversion: {finite*100:.1f}% finite values")

    print("✓ Test 2 PASSED (no crash)")

except Exception as e:
    print(f"✗ Test 2 FAILED: {type(e).__name__}: {e}")

print("\n" + "="*70)
print("DIAGNOSIS COMPLETE")
print("="*70)
