"""
Debug: Check U_final directly after 100 iterations
"""

import sys
import numpy as np
from pathlib import Path

solver_dir = Path(__file__).parent.parent / "solver"
sys.path.insert(0, str(solver_dir.parent))

from solver.solver_1d import run_solver_1d, build_water_air_ic, extract_primitive_profiles
from solver.flux_allspeed import cons_to_prim_allspeed

# Build initial conditions
case = build_water_air_ic(
    N=10, x_lo=0.0, x_hi=1.0,
    u0=1.0, p0=1.0e5, T0=300.0
)

U_init = case['U_init'].copy()
N = U_init.shape[0]
Ns = len(case['eos_list'])

print("Initial U_init[0]:", U_init[0])
print("Initial U_init[4]:", U_init[4])
print()

# Run solver
result = run_solver_1d(case)

print(f"Result n_steps: {result['n_steps']}")
print(f"Result converged: {result['converged']}")
print()

U_final = result['U_final']
print("Final U_final[0]:", U_final[0])
print("Final U_final[4]:", U_final[4])
print()

# Check if U_final == U_init
print("U_final == U_init?", np.allclose(U_final, U_init))
print("Max difference:", np.max(np.abs(U_final - U_init)))
print()

# Extract final primitives
T_guess_arr = np.full(N, 300.0)
for m in range(N):
    try:
        _, _, _, T_m, _ = cons_to_prim_allspeed(U_final[m], case['eos_list'], 300.0)
        T_guess_arr[m] = T_m
    except:
        pass

prims_final = extract_primitive_profiles(U_final, case['eos_list'], T_guess_arr)

print("Final primitive profiles (first 3 cells, all 10 cells for center):")
for m in [0, 1, 2, 4, 5, 7, 8, 9]:
    print(f"  Cell {m}: p={prims_final['p'][m]:.6e}, u={prims_final['u'][m]:.8f}, "
          f"T={prims_final['T'][m]:.4f}")

print()
print("Pressure min/max:")
print(f"  min = {prims_final['p'].min():.6e}")
print(f"  max = {prims_final['p'].max():.6e}")

print()
print("Velocity min/max:")
print(f"  min = {prims_final['u'].min():.8f}")
print(f"  max = {prims_final['u'].max():.8f}")

print()
print("Temperature min/max:")
print(f"  min = {prims_final['T'].min():.4f}")
print(f"  max = {prims_final['T'].max():.4f}")
