"""
Generate visualization and detailed report for Round 4 validation
"""

import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

solver_dir = Path(__file__).parent.parent / "solver"
sys.path.insert(0, str(solver_dir.parent))

from solver.solver_1d import run_solver_1d, build_water_air_ic, extract_primitive_profiles
from solver.flux_allspeed import cons_to_prim_allspeed

def compute_energy(U_cells, eos_list, T_guess_arr):
    N = U_cells.shape[0]
    E_total = 0.0
    for m in range(N):
        E_total += U_cells[m, -1]
    return E_total

# Run validation
case = build_water_air_ic(N=10)
U_init = case['U_init'].copy()
N = U_init.shape[0]
Ns = len(case['eos_list'])

# Get initial state
T_guess_arr = np.full(N, 300.0)
E_init = compute_energy(U_init, case['eos_list'], T_guess_arr)
prims_init = extract_primitive_profiles(U_init, case['eos_list'], T_guess_arr)

# Run solver
print("Running solver...")
result = run_solver_1d(case)

# Get final state
U_final = result['U_final']
T_guess_arr_final = np.full(N, 300.0)
for m in range(N):
    try:
        _, _, _, T_m, _ = cons_to_prim_allspeed(U_final[m], case['eos_list'], 300.0)
        T_guess_arr_final[m] = T_m
    except:
        pass

E_final = compute_energy(U_final, case['eos_list'], T_guess_arr_final)
prims_final = extract_primitive_profiles(U_final, case['eos_list'], T_guess_arr_final)

# Grid
x_cells = case['x_cells']

# Create figure with 5 subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Round 4: Phase 1 Water-Air Advection (100 iterations)', fontsize=14, fontweight='bold')

# 1. Pressure profile
ax = axes[0, 0]
ax.plot(x_cells, prims_init['p'], 'b-o', label='Initial', linewidth=2, markersize=6)
ax.plot(x_cells, prims_final['p'], 'r-s', label='Final (100 iter)', linewidth=2, markersize=6)
ax.axhline(case['p0'], color='k', linestyle='--', alpha=0.3, label=f'p0={case["p0"]:.2e}')
ax.set_xlabel('x [m]')
ax.set_ylabel('p [Pa]')
ax.set_title('Pressure Profile')
ax.grid(True, alpha=0.3)
ax.legend()

# 2. Velocity profile
ax = axes[0, 1]
ax.plot(x_cells, prims_init['u'], 'b-o', label='Initial', linewidth=2, markersize=6)
ax.plot(x_cells, prims_final['u'], 'r-s', label='Final (100 iter)', linewidth=2, markersize=6)
ax.axhline(case['u0'], color='k', linestyle='--', alpha=0.3, label=f'u0={case["u0"]:.3f} m/s')
ax.set_xlabel('x [m]')
ax.set_ylabel('u [m/s]')
ax.set_title('Velocity Profile')
ax.grid(True, alpha=0.3)
ax.legend()

# 3. Temperature profile
ax = axes[0, 2]
ax.plot(x_cells, prims_init['T'], 'b-o', label='Initial', linewidth=2, markersize=6)
ax.plot(x_cells, prims_final['T'], 'r-s', label='Final (100 iter)', linewidth=2, markersize=6)
ax.axhline(case['T0'], color='k', linestyle='--', alpha=0.3, label=f'T0={case["T0"]:.0f} K')
ax.set_xlabel('x [m]')
ax.set_ylabel('T [K]')
ax.set_title('Temperature Profile')
ax.grid(True, alpha=0.3)
ax.legend()

# 4. Species mass fraction (water)
ax = axes[1, 0]
ax.plot(x_cells, prims_init['Yi'][:, 0], 'b-o', label='Initial (t=0)', linewidth=2, markersize=6)
ax.plot(x_cells, prims_final['Yi'][:, 0], 'r-s', label='Final (100 iter)', linewidth=2, markersize=6)
ax.set_xlabel('x [m]')
ax.set_ylabel('Y_water')
ax.set_title('Water Mass Fraction')
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_ylim(-0.1, 1.1)

# 5. Density profile
ax = axes[1, 1]
ax.plot(x_cells, prims_init['rho'], 'b-o', label='Initial', linewidth=2, markersize=6)
ax.plot(x_cells, prims_final['rho'], 'r-s', label='Final (100 iter)', linewidth=2, markersize=6)
ax.set_xlabel('x [m]')
ax.set_ylabel('ρ [kg/m³]')
ax.set_title('Density Profile')
ax.grid(True, alpha=0.3)
ax.legend()

# 6. Convergence metrics
ax = axes[1, 2]
ax.axis('off')
metrics_text = f"""
VALIDATION RESULTS (100 iterations)

Convergence: {'PASS ✓' if result['converged'] else 'FAIL ✗'}
n_steps: {result['n_steps']}
t_final: {result['t_final']:.4e} s

Pressure deviation:
  max|(p-p0)/p0| = {np.max(np.abs(prims_final['p']-case['p0'])/case['p0']):.3e}
  Threshold: 1e-2 {'PASS ✓' if np.max(np.abs(prims_final['p']-case['p0'])/case['p0']) < 1e-2 else 'FAIL ✗'}

Velocity deviation:
  max|u-u0| = {np.max(np.abs(prims_final['u']-case['u0'])):.3e} m/s
  Threshold: 1e-2 {'PASS ✓' if np.max(np.abs(prims_final['u']-case['u0'])) < 1e-2 else 'FAIL ✗'}

Energy conservation:
  |(E-E0)/E0| = {np.abs(E_final-E_init)/np.abs(E_init):.3e}
  Threshold: 1e-2 {'PASS ✓' if np.abs(E_final-E_init)/np.abs(E_init) < 1e-2 else 'FAIL ✗'}

Species bounds:
  Yi_min = {prims_final['Yi'].min():.3e}
  Yi_max = {prims_final['Yi'].max():.3e}
  Valid: {(prims_final['Yi'].min() >= -1e-14) and (prims_final['Yi'].max() <= 1.0+1e-14)} {'✓' if (prims_final['Yi'].min() >= -1e-14) and (prims_final['Yi'].max() <= 1.0+1e-14) else '✗'}
"""
ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
results_dir = Path(__file__).parent.parent / "results" / "1D" / "phase1_abgrall_water_air"
results_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(results_dir / "all_metrics.png", dpi=150, bbox_inches='tight')
print(f"Saved figure to {results_dir}/all_metrics.png")
plt.close()

# Create a detail analysis figure (zoomed in on interface)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Round 4: Interface Detail (Cells 3-6)', fontsize=12, fontweight='bold')

cell_range = [3, 4, 5, 6]
x_detail = x_cells[cell_range]

# Pressure detail
ax = axes[0]
ax.plot(x_detail, prims_init['p'][cell_range], 'b-o', label='Initial', linewidth=2, markersize=8)
ax.plot(x_detail, prims_final['p'][cell_range], 'r-s', label='Final', linewidth=2, markersize=8)
ax.axhline(case['p0'], color='k', linestyle='--', alpha=0.3)
ax.set_xlabel('x [m]')
ax.set_ylabel('p [Pa]')
ax.set_title('Pressure at Interface')
ax.grid(True, alpha=0.3)
ax.legend()

# Velocity detail
ax = axes[1]
ax.plot(x_detail, prims_init['u'][cell_range], 'b-o', label='Initial', linewidth=2, markersize=8)
ax.plot(x_detail, prims_final['u'][cell_range], 'r-s', label='Final', linewidth=2, markersize=8)
ax.axhline(case['u0'], color='k', linestyle='--', alpha=0.3)
ax.set_xlabel('x [m]')
ax.set_ylabel('u [m/s]')
ax.set_title('Velocity at Interface')
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
fig.savefig(results_dir / "interface_detail.png", dpi=150, bbox_inches='tight')
print(f"Saved figure to {results_dir}/interface_detail.png")
plt.close()

print("Visualization complete.")
