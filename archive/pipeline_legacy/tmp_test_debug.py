"""
Debug analysis for Phase 1 validation failure
"""

import sys
import numpy as np
from pathlib import Path

solver_dir = Path(__file__).parent.parent / "solver"
sys.path.insert(0, str(solver_dir.parent))

from solver.solver_1d import run_solver_1d, build_water_air_ic, extract_primitive_profiles
from solver.flux_allspeed import cons_to_prim_allspeed

def compute_energy(U_cells, eos_list, T_guess_arr):
    """Compute total energy E_total = sum of rho*E over all cells."""
    N = U_cells.shape[0]
    E_total = 0.0
    for m in range(N):
        E_total += U_cells[m, -1]  # rho*E is last component
    return E_total

# Build initial conditions
case = build_water_air_ic(
    N=10, x_lo=0.0, x_hi=1.0,
    u0=1.0, p0=1.0e5, T0=300.0
)

U_init = case['U_init'].copy()
N = U_init.shape[0]
Ns = len(case['eos_list'])

print("=" * 70)
print("INITIAL STATE DETAIL")
print("=" * 70)
print()
print("Initial conservative variables U:")
print(f"  Shape: {U_init.shape} (N={N}, Ns+2={Ns+2})")
for m in range(N):
    rhoY_water = U_init[m, 0]
    rhoY_air = U_init[m, 1]
    rho_u = U_init[m, 2]
    rho_E = U_init[m, 3]
    rho = rhoY_water + rhoY_air
    Y_water = rhoY_water / rho if rho > 0 else 0
    print(f"  Cell {m}: rhoY_w={rhoY_water:.4e}, rhoY_a={rhoY_air:.4e}, "
          f"rho={rho:.4e}, u={rho_u/rho:.4f}, E={rho_E/rho:.4e}, Y_w={Y_water:.4f}")

# Extract initial primitives
T_guess_arr = np.full(N, 300.0)
prims_init = extract_primitive_profiles(U_init, case['eos_list'], T_guess_arr)

print()
print("Initial primitive profiles:")
for m in range(N):
    print(f"  Cell {m}: p={prims_init['p'][m]:.4e}, u={prims_init['u'][m]:.6f}, "
          f"T={prims_init['T'][m]:.2f}, rho={prims_init['rho'][m]:.4e}, "
          f"Y_water={prims_init['Yi'][m,0]:.4f}")

print()
print("=" * 70)
print("SOLVER RUN WITH SNAPSHOTS")
print("=" * 70)
print()

# Modify case to capture snapshots at every step
case['output_times'] = np.linspace(0, case['t_end'], min(11, 101))
print(f"Capturing {len(case['output_times'])} snapshots")

result = run_solver_1d(case)

print(f"Solver completed: n_steps={result['n_steps']}, converged={result['converged']}")
print(f"  t_final = {result['t_final']:.6e} s")
print(f"  snapshots captured = {len(result['snapshots'])}")
print()

# Analyze final snapshot
if len(result['snapshots']) > 0:
    last_snap = result['snapshots'][-1]
    U_final = last_snap['U']
    print(f"Final snapshot (t={last_snap['t']:.6e}):")
    prims_final = extract_primitive_profiles(U_final, case['eos_list'], T_guess_arr)

    print()
    print("Final primitive profiles:")
    for m in range(N):
        print(f"  Cell {m}: p={prims_final['p'][m]:.4e}, u={prims_final['u'][m]:.6f}, "
              f"T={prims_final['T'][m]:.2f}, rho={prims_final['rho'][m]:.4e}, "
              f"Y_water={prims_final['Yi'][m,0]:.4f}")

    print()
    print("Pressure deviations from p0:")
    for m in range(N):
        dp = prims_final['p'][m] - case['p0']
        dp_rel = dp / case['p0']
        print(f"  Cell {m}: Δp={dp:.4e} Pa, Δp/p0={dp_rel:.6e}")

    print()
    print("Velocity deviations from u0:")
    for m in range(N):
        du = prims_final['u'][m] - case['u0']
        print(f"  Cell {m}: Δu={du:.6e} m/s")

    # Check species conservation
    print()
    print("Species checks (water):")
    for m in range(N):
        Y_w = prims_final['Yi'][m, 0]
        print(f"  Cell {m}: Y_water={Y_w:.6f}")

print()
print("=" * 70)
print("HYPOTHESIS CHECK: Pressure-velocity coupling issue?")
print("=" * 70)
print()
print("Looking at first few snapshots to see where deviations start:")
for snap_idx in range(min(3, len(result['snapshots']))):
    snap = result['snapshots'][snap_idx]
    U_snap = snap['U']
    prims = extract_primitive_profiles(U_snap, case['eos_list'], T_guess_arr)
    print(f"Snapshot {snap_idx} (t={snap['t']:.6e}):")
    print(f"  p: min={prims['p'].min():.6e}, max={prims['p'].max():.6e}, "
          f"mean={prims['p'].mean():.6e}")
    print(f"  u: min={prims['u'].min():.6f}, max={prims['u'].max():.6f}, "
          f"mean={prims['u'].mean():.6f}")
    print(f"  T: min={prims['T'].min():.2f}, max={prims['T'].max():.2f}, "
          f"mean={prims['T'].mean():.2f}")
