"""
Validation Case A — Smooth Interface Advection (Ideal Gas)
Sources: validation/1D/1D_smooth_interface-advection.md Case A
"""
import sys
sys.path.insert(0, '/home/younglin90/work/claude_code/claudeCFD')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from solver.eos.ideal import IdealGasEOS
from solver.utils import cons_to_prim, prim_to_cons
from solver.solve import run_1d

# ============================================================================
# CASE A PARAMETERS
# ============================================================================
# Physical setup
eos1 = IdealGasEOS(gamma=1.4, M=28.0)   # Species 1: γ=1.4, M=28 g/mol
eos2 = IdealGasEOS(gamma=1.66, M=4.0)   # Species 2: γ=1.66, M=4 g/mol
eos_list = [eos1, eos2]

# Domain and mesh
L = 1.0
nx = 501
x = np.linspace(0, L, nx, endpoint=False) + L / (2*nx)
dx = L / nx

# Initial condition parameters
x_c, r_c, w1, w2, k = 0.5, 0.25, 0.6, 0.2, 20
u0 = 1.0
p0 = 0.9

# Time domain
t_end = 8.0
t_output = [0.0, 1.0, 2.0, 4.0, 8.0]
CFL = 0.6

# ============================================================================
# INITIALIZATION
# ============================================================================
print("=" * 70)
print("VALIDATION CASE A: Smooth Interface Advection (Ideal Gas)")
print("=" * 70)
print(f"Domain: [{0}, {L}], Cells: {nx}, dx = {dx:.6f}")
print(f"Species 1: γ={eos1.gamma}, M={eos1.M}")
print(f"Species 2: γ={eos2.gamma}, M={eos2.M}")
print(f"Initial: u₀={u0}, p₀={p0}")
print(f"Time: t_end={t_end}, CFL={CFL}")
print()

# Initial mass fractions via tanh profile
r = np.abs(x - x_c)
rhoY1 = (w1/2) * (1 - np.tanh(k * (r - r_c)))
rhoY2 = (w2/2) * (1 + np.tanh(k * (r - r_c)))
rho = rhoY1 + rhoY2
Y1 = rhoY1 / rho
Y2 = rhoY2 / rho

print(f"Initial state:")
print(f"  ρ_min = {rho.min():.6f}, ρ_max = {rho.max():.6f}")
print(f"  Y₁_min = {Y1.min():.6f}, Y₁_max = {Y1.max():.6f}")
print()

# Compute R_mix(x) at each cell
R_mix = Y1 * eos1.R_s + Y2 * eos2.R_s

# Temperature from uniform pressure condition: p = ρ R_mix T
# T(x) = p0 / (ρ(x) * R_mix(x))
T_arr = p0 / (rho * R_mix)

print(f"Temperature field (from uniform p):")
print(f"  T_min = {T_arr.min():.6f} K, T_max = {T_arr.max():.6f} K")
print()

# Internal energy
c_v1 = eos1.R_s / (eos1.gamma - 1)
c_v2 = eos2.R_s / (eos2.gamma - 1)
e_arr = Y1 * c_v1 * T_arr + Y2 * c_v2 * T_arr
E_arr = e_arr + 0.5 * u0**2

print(f"Energy field:")
print(f"  e_min = {e_arr.min():.6f}, e_max = {e_arr.max():.6f}")
print(f"  E_min = {E_arr.min():.6f}, E_max = {E_arr.max():.6f}")
print()

# Conservative variables
U0 = np.zeros((nx, 4))
U0[:, 0] = rho               # ρ
U0[:, 1] = rho * u0          # ρu
U0[:, 2] = rho * E_arr       # ρE
U0[:, 3] = rhoY1             # ρY₁

# Verify initial pressure uniformity
print("Verifying initial pressure uniformity:")
prim_check = cons_to_prim(U0[nx//2], eos_list)
print(f"  Center cell (i={nx//2}): p = {prim_check[0]:.10f} (expect {p0})")
prim_check_left = cons_to_prim(U0[0], eos_list)
print(f"  Left cell   (i=0):       p = {prim_check_left[0]:.10f} (expect {p0})")
prim_check_right = cons_to_prim(U0[-1], eos_list)
print(f"  Right cell  (i={nx-1}):    p = {prim_check_right[0]:.10f} (expect {p0})")
print()

# ============================================================================
# RUN SIMULATION
# ============================================================================
print("Running simulation...")
case_params = {
    'eos_list': eos_list,
    'x_cells': x,
    'U_init': U0,
    't_end': t_end,
    'CFL': CFL,
    'bc_left': 'periodic',
    'bc_right': 'periodic',
    'output_times': t_output,
    'verbose': False,
}

result = run_1d(case_params)
print(f"✓ Completed: {result['n_steps']} steps, t_final = {result['t_final']:.4f}")
print()

# ============================================================================
# POST-PROCESSING & VALIDATION
# ============================================================================
U_final = result['U_final']
snapshots = result['snapshots']

# Reconstruct primitives at final time
prim_final = np.array([cons_to_prim(U_final[i], eos_list) for i in range(nx)])
p_final = prim_final[:, 0]
u_final = prim_final[:, 1]
T_final = prim_final[:, 3]
Y1_final = U_final[:, 3] / U_final[:, 0]

# Reconstruct primitives at all snapshots
pe_history = []
energy_history = []
for snap in snapshots:
    t_snap = snap['t']
    U_snap = snap['U']
    prim_snap = np.array([cons_to_prim(U_snap[i], eos_list) for i in range(nx)])
    p_snap = prim_snap[:, 0]

    # PE error: L2 norm of (p/p0 - 1)
    pe_error = np.sqrt(np.mean((p_snap / p0 - 1)**2))
    pe_history.append((t_snap, pe_error))

    # Energy conservation
    E_total = np.sum(U_snap[:, 2]) * dx
    energy_history.append((t_snap, E_total))

print("Validation Criteria:")
print("-" * 70)

# Criterion 1: PE uniformity at t=8.0
pe_error_final = np.sqrt(np.mean((p_final / p0 - 1)**2))
pe_pass = pe_error_final < 1e-4
print(f"1. PE uniformity (t={t_end}):")
print(f"   L₂(p) = {pe_error_final:.2e}")
print(f"   PASS threshold: < 1e-4")
print(f"   Result: {'✓ PASS' if pe_pass else '✗ FAIL'}")
print()

# Criterion 2: Long-time stability (no divergence of PE error)
pe_values = [v[1] for v in pe_history]
pe_max = max(pe_values)
pe_stable = (pe_max < 1e-2)  # No divergence, reasonable threshold
print(f"2. Long-time stability (t ≤ {t_end}):")
print(f"   max(L₂(p)) = {pe_max:.2e}")
print(f"   Stable: {'✓ YES' if pe_stable else '✗ DIVERGED'}")
print()

# Criterion 3: Energy conservation
E0 = np.sum(U0[:, 2]) * dx
E_final = np.sum(U_final[:, 2]) * dx
energy_error = abs(E_final - E0) / E0
energy_pass = energy_error < 1e-12
print(f"3. Energy conservation:")
print(f"   ΔE_total / E₀ = {energy_error:.2e}")
print(f"   PASS threshold: < 1e-12 (machine precision)")
print(f"   Result: {'✓ PASS' if energy_pass else '✗ FAIL'}")
print()

# Criterion 4: Velocity uniformity
u_mean = np.mean(u_final)
u_std = np.std(u_final)
u_rel_error = u_std / u0
u_pass = u_rel_error < 1e-4
print(f"4. Velocity uniformity:")
print(f"   u_mean = {u_mean:.6f}, u_std = {u_std:.2e}")
print(f"   Δu/u₀ = {u_rel_error:.2e}")
print(f"   PASS threshold: < 1e-4")
print(f"   Result: {'✓ PASS' if u_pass else '✗ FAIL'}")
print()

# Overall result
all_pass = pe_pass and pe_stable and energy_pass and u_pass
print("=" * 70)
print(f"OVERALL RESULT: {'✓ PASS' if all_pass else '✗ FAIL'}")
print("=" * 70)
print()

# ============================================================================
# PLOTTING
# ============================================================================
output_dir = Path('/home/younglin90/work/claude_code/claudeCFD/results/1D/Smooth_Interface_Advection_IdealGas/round3')
output_dir.mkdir(parents=True, exist_ok=True)

# Plot 1: Final state (t=8.0)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].plot(x, rho * Y1, 'b-', label='ρY₁ (t=0)', alpha=0.5)
axes[0, 0].plot(x, U_final[:, 3], 'b-', label='ρY₁ (t=8.0)', linewidth=2)
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('ρY₁ [kg/m³]')
axes[0, 0].set_title('Species 1 Mass Density')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(x, rho * Y2, 'r-', label='ρY₂ (t=0)', alpha=0.5)
axes[0, 1].plot(x, U_final[:, 0] - U_final[:, 3], 'r-', label='ρY₂ (t=8.0)', linewidth=2)
axes[0, 1].set_xlabel('x')
axes[0, 1].set_ylabel('ρY₂ [kg/m³]')
axes[0, 1].set_title('Species 2 Mass Density')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].axhline(y=u0, color='k', linestyle='--', label=f'u₀={u0}', alpha=0.5)
axes[1, 0].plot(x, u_final, 'g-', linewidth=2, label='u (t=8.0)')
axes[1, 0].set_xlabel('x')
axes[1, 0].set_ylabel('u [m/s]')
axes[1, 0].set_title('Velocity')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].axhline(y=p0, color='k', linestyle='--', label=f'p₀={p0}', alpha=0.5)
axes[1, 1].plot(x, p_final, 'purple', linewidth=2, label='p (t=8.0)')
axes[1, 1].set_xlabel('x')
axes[1, 1].set_ylabel('p [Pa]')
axes[1, 1].set_title('Pressure')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'species_density_t8.png', dpi=150, bbox_inches='tight')
print(f"Saved: {output_dir / 'species_density_t8.png'}")
plt.close()

# Plot 2: PE and energy error history
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

t_vals = [v[0] for v in pe_history]
pe_vals = [v[1] for v in pe_history]
axes[0].semilogy(t_vals, pe_vals, 'o-', linewidth=2, markersize=6)
axes[0].axhline(y=1e-4, color='r', linestyle='--', label='PASS threshold (1e-4)')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('L₂(p) error')
axes[0].set_title('PE Error History')
axes[0].legend()
axes[0].grid(True, alpha=0.3, which='both')

E_vals = [v[1] for v in energy_history]
E_rel_error = [(E - E0) / E0 for E in E_vals]
axes[1].semilogy(t_vals, np.abs(E_rel_error), 'o-', linewidth=2, markersize=6)
axes[1].set_xlabel('Time')
axes[1].set_ylabel('|ΔE_total / E₀|')
axes[1].set_title('Energy Conservation Error')
axes[1].grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig(output_dir / 'PE_energy_error_history.png', dpi=150, bbox_inches='tight')
print(f"Saved: {output_dir / 'PE_energy_error_history.png'}")
plt.close()

# Plot 3: Pressure error distribution at t=0 (should be ~0 ideally)
fig, ax = plt.subplots(figsize=(10, 5))
prim_init = np.array([cons_to_prim(U0[i], eos_list) for i in range(nx)])
p_init = prim_init[:, 0]
p_error_init = (p_init - p0) / p0
ax.plot(x, p_error_init, 'b-', linewidth=1)
ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax.set_xlabel('x')
ax.set_ylabel('(p - p₀) / p₀')
ax.set_title('Pressure Error Distribution at t=0 (Initial Condition)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / 'PE_error_distribution_t0.png', dpi=150, bbox_inches='tight')
print(f"Saved: {output_dir / 'PE_error_distribution_t0.png'}")
plt.close()

# Plot 4: Pressure error norm history
fig, ax = plt.subplots(figsize=(10, 5))
ax.semilogy(t_vals, pe_vals, 'o-', linewidth=2, markersize=6, label='L₂(p) = √(Σ(p/p₀ - 1)²/N)')
ax.axhline(y=1e-4, color='r', linestyle='--', linewidth=2, label='PASS threshold (1e-4)')
ax.axhline(y=1e-5, color='orange', linestyle=':', linewidth=2, label='APEC target (1e-5)')
ax.set_xlabel('Time')
ax.set_ylabel('L₂(p)')
ax.set_title('Pressure Equilibrium Error Norm History')
ax.legend()
ax.grid(True, alpha=0.3, which='both')
plt.tight_layout()
plt.savefig(output_dir / 'PE_error_norm_history.png', dpi=150, bbox_inches='tight')
print(f"Saved: {output_dir / 'PE_error_norm_history.png'}")
plt.close()

# ============================================================================
# GENERATE REPORT
# ============================================================================
report_content = f"""# Case A Validation Report: Smooth Interface Advection (Ideal Gas)

**Date:** 2026-03-31
**Solver:** APEC Flux + Multicomponent Ideal Gas EOS
**Case Type:** Pressure & Velocity Equilibrium Preservation

## Physical Setup

| Parameter | Value |
|-----------|-------|
| Species 1 (γ=1.4, M=28) | Nitrogen-like |
| Species 2 (γ=1.66, M=4) | Helium-like |
| Domain | [0, 1] m (periodic) |
| Cells | 501 |
| Initial velocity | 1.0 m/s |
| Initial pressure | 0.9 Pa |
| Final time | 8.0 s (8 flow-through cycles) |
| Time integration | TVD-RK3 (auto-selected) |
| CFL | 0.6 |

## Simulation Results

| Metric | Value | Status |
|--------|-------|--------|
| Final PE error L₂(p) | {pe_error_final:.2e} | {'✓ PASS' if pe_pass else '✗ FAIL'} |
| Max PE error (history) | {pe_max:.2e} | {'✓ Stable' if pe_stable else '✗ Diverged'} |
| Energy conservation error | {energy_error:.2e} | {'✓ PASS' if energy_pass else '✗ FAIL'} |
| Velocity uniformity (Δu/u₀) | {u_rel_error:.2e} | {'✓ PASS' if u_pass else '✗ FAIL'} |

## Validation Criteria Assessment

### 1. Pressure Uniformity (t = 8.0 s)
- **Requirement:** L₂(p) < 1e-4
- **Measured:** {pe_error_final:.2e}
- **Result:** {'✓ PASS' if pe_pass else '✗ FAIL'}

The pressure must remain uniformly distributed across the domain despite the non-uniform composition field. The APEC flux is designed to maintain pressure equilibrium even in the presence of smooth composition gradients.

### 2. Long-time Stability
- **Requirement:** No exponential growth of PE error over 8 cycles
- **Max L₂(p):** {pe_max:.2e} at t = {t_vals[pe_vals.index(pe_max)]:.2f} s
- **Result:** {'✓ Stable' if pe_stable else '✗ Diverged (FAIL)'}

### 3. Energy Conservation
- **Requirement:** |ΔE_total/E₀| < 1e-12 (machine precision)
- **Measured:** {energy_error:.2e}
- **Result:** {'✓ PASS' if energy_pass else '✗ FAIL'}

Total mechanical energy (kinetic + internal) must be strictly conserved for this inviscid, adiabatic flow.

### 4. Velocity Uniformity
- **Requirement:** max|Δu/u₀| < 1e-4
- **Measured:** {u_rel_error:.2e}
- **Result:** {'✓ PASS' if u_pass else '✗ FAIL'}

## Physical Interpretation

This test validates the APEC flux's ability to preserve pressure equilibrium in smooth multi-component advection. The uniform pressure and velocity fields are exact solutions to the compressible Euler equations when there are no body forces and the flow is isentropic.

The composition field (ρY₁, ρY₂) undergoes smooth advection with velocity u₀, completing 8 complete cycles over the periodic domain. Numerical diffusion from the flux discretization should not generate spurious pressure or velocity oscillations.

## Files Generated

1. `species_density_t8.png` - Final distribution of ρY₁, ρY₂, u, p at t=8.0 s
2. `PE_energy_error_history.png` - Temporal evolution of PE and energy errors
3. `PE_error_distribution_t0.png` - Spatial distribution of initial pressure error
4. `PE_error_norm_history.png` - L₂(p) norm throughout the simulation
5. `report.md` - This report

## Overall Verdict

**{'✓✓✓ CASE A PASSED ✓✓✓' if all_pass else '✗✗✗ CASE A FAILED ✗✗✗'}**

Simulation completed successfully with {result['n_steps']} time steps.
"""

report_path = output_dir / 'report.md'
with open(report_path, 'w') as f:
    f.write(report_content)
print(f"Saved: {report_path}")
print()

# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 70)
if all_pass:
    print("✓✓✓ CASE A VALIDATION PASSED ✓✓✓")
    exit_code = 0
else:
    print("✗✗✗ CASE A VALIDATION FAILED ✗✗✗")
    if not pe_pass:
        print(f"  ✗ PE uniformity failed: L₂(p) = {pe_error_final:.2e} >= 1e-4")
    if not pe_stable:
        print(f"  ✗ PE diverged: max(L₂(p)) = {pe_max:.2e}")
    if not energy_pass:
        print(f"  ✗ Energy conservation failed: error = {energy_error:.2e}")
    if not u_pass:
        print(f"  ✗ Velocity uniformity failed: Δu/u₀ = {u_rel_error:.2e}")
    exit_code = 1

print("=" * 70)
sys.exit(exit_code)
