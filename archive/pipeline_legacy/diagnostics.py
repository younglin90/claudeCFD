"""
Detailed diagnostics of solver run to identify where pressure deviations occur.
"""

import sys
import numpy as np
from pathlib import Path

solver_dir = Path("/home/younglin90/work/claude_code/claudeCFD/solver")
sys.path.insert(0, str(solver_dir.parent))

from solver.solver_1d import (
    run_solver_1d, build_water_air_ic, extract_primitive_profiles,
    _spatial_rhs, _backward_euler_step
)
from solver.flux_allspeed import cons_to_prim_allspeed
from solver.boundary import apply_bc_1d

# Build initial conditions
case = build_water_air_ic(
    N=10, x_lo=0.0, x_hi=1.0,
    u0=1.0, p0=1.0e5, T0=300.0
)

U_init = case['U_init'].copy()
N = U_init.shape[0]
Ns = len(case['eos_list'])
eos_list = case['eos_list']
dx = case['dx']

print("=" * 70)
print("INITIAL STATE")
print("=" * 70)
print()

# Verify initial primitives
T_guess_arr = np.full(N, 300.0)
prims = extract_primitive_profiles(U_init, eos_list, T_guess_arr)
print(f"Initial pressure range: {prims['p'].min():.4e} - {prims['p'].max():.4e} Pa")
print(f"Initial velocity range: {prims['u'].min():.6f} - {prims['u'].max():.6f} m/s")
print(f"Water location (Y_water > 0.5):")
for m in range(N):
    if prims['Yi'][m, 0] > 0.5:
        print(f"  Cell {m}: Y_water={prims['Yi'][m, 0]:.4f}")

print()
print("=" * 70)
print("SPATIAL RHS AT INITIAL STATE")
print("=" * 70)
print()

# Compute spatial RHS
rhs = _spatial_rhs(U_init, eos_list, dx, 'periodic', 'periodic', T_guess_arr)

print("RHS (dU/dt) for each cell (first 3 and last 3):")
for m in [0, 1, 2]:
    print(f"Cell {m}: d(rhoY_w)/dt={rhs[m, 0]:.4e}, d(rho*u)/dt={rhs[m, Ns]:.4e}, d(rho*E)/dt={rhs[m, Ns+1]:.4e}")

print()
for m in range(N-3, N):
    print(f"Cell {m}: d(rhoY_w)/dt={rhs[m, 0]:.4e}, d(rho*u)/dt={rhs[m, Ns]:.4e}, d(rho*E)/dt={rhs[m, Ns+1]:.4e}")

print()
print("=" * 70)
print("FIRST BACKWARD EULER STEP")
print("=" * 70)
print()

# Compute CFL dt
from solver.solver_1d import _compute_dt_cfl
dt = _compute_dt_cfl(U_init, eos_list, dx, 0.5, T_guess_arr)
print(f"CFL timestep: dt = {dt:.4e} s")

# Take one Backward Euler step
try:
    U_step1 = _backward_euler_step(
        U_init, eos_list, dx, dt,
        'periodic', 'periodic', T_guess_arr,
        max_newton_iter=20, newton_tol=1e-8
    )

    print(f"Step completed successfully")
    print()

    # Extract primitives after first step
    T_guess_arr_new = np.full(N, 300.0)
    for m in range(N):
        try:
            _, _, _, T_m, _ = cons_to_prim_allspeed(U_step1[m], eos_list, 300.0)
            T_guess_arr_new[m] = T_m
        except:
            pass

    prims_step1 = extract_primitive_profiles(U_step1, eos_list, T_guess_arr_new)

    print(f"After step 1:")
    print(f"  Pressure range: {prims_step1['p'].min():.4e} - {prims_step1['p'].max():.4e} Pa")
    print(f"  Velocity range: {prims_step1['u'].min():.6f} - {prims_step1['u'].max():.6f} m/s")

    print()
    print("Pressure differences (Pa):")
    for m in range(N):
        dp = prims_step1['p'][m] - case['p0']
        print(f"  Cell {m}: p={prims_step1['p'][m]:.4e}, Δp={dp:+.4e} Pa")

    print()
    print("Velocity differences (m/s):")
    for m in range(N):
        du = prims_step1['u'][m] - case['u0']
        print(f"  Cell {m}: u={prims_step1['u'][m]:.6f}, Δu={du:+.6e} m/s")

except Exception as e:
    print(f"Error in Backward Euler step: {e}")
    import traceback
    traceback.print_exc()

