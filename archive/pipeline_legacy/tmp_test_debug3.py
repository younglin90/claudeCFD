"""
Debug: Check conservative state details
"""

import sys
import numpy as np
from pathlib import Path

solver_dir = Path(__file__).parent.parent / "solver"
sys.path.insert(0, str(solver_dir.parent))

from solver.solver_1d import run_solver_1d, build_water_air_ic
from solver.flux_allspeed import cons_to_prim_allspeed

# Build and run
case = build_water_air_ic(N=10)
result = run_solver_1d(case)

U_final = result['U_final']
N = U_final.shape[0]
Ns = len(case['eos_list'])

print("=" * 80)
print("FINAL CONSERVATIVE STATE ANALYSIS")
print("=" * 80)
print()

for m in range(N):
    rhoY_water = U_final[m, 0]
    rhoY_air = U_final[m, 1]
    rho_u = U_final[m, 2]
    rho_E = U_final[m, 3]

    rho = rhoY_water + rhoY_air
    u = rho_u / rho if rho > 0 else 0
    e = (rho_E - 0.5 * rho_u ** 2 / rho) / rho if rho > 0 else 0

    print(f"Cell {m}:")
    print(f"  rhoY_water={rhoY_water:.6e}, rhoY_air={rhoY_air:.6e}")
    print(f"  rho*u={rho_u:.6e}, rho*E={rho_E:.6e}")
    print(f"  rho={rho:.6e}, u={u:.8f}, e={e:.6e}")

    # Try to get pressure
    try:
        T_guess = 300.0
        rho_m, u_m, p_m, T_m, rhoYi_m = cons_to_prim_allspeed(U_final[m], case['eos_list'], T_guess)
        print(f"  → p={p_m:.6e}, T={T_m:.4f}")
    except Exception as ex:
        print(f"  → Error converting to primitives: {ex}")
    print()

print("=" * 80)
print("PROBLEM: Cell 4 has negative pressure!")
print("This indicates a solver stability or convergence issue")
print("=" * 80)
