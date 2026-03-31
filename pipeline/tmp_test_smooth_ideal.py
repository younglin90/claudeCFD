#!/usr/bin/env python3
"""
Validation test: Smooth Interface Advection (Ideal Gas, Case A)
Based on: validation/1D/1D_smooth_interface-advection.md § Case A
"""

import sys
sys.path.insert(0, '/home/younglin90/work/claude_code/claudeCFD')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from solver.solve import run_1d
from solver.eos import ideal
import warnings
warnings.filterwarnings('ignore')

# Create output directory
output_dir = Path('/home/younglin90/work/claude_code/claudeCFD/results/1D/Smooth_Interface_Advection_IdealGas')
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("VALIDATION: Smooth Interface Advection (Ideal Gas, Case A)")
print("=" * 80)

# Case A parameters from validation spec
# Domain: [0, 1] periodic, 501 cells
# Initial condition: smooth tanh interface
# Runtime: t = 8.0 (8 flow-through time)

L = 1.0
nx = 501
x = np.linspace(0, L, nx, endpoint=False)
dx = L / nx

# Species 1: γ=1.4, M=28 g/mol
# Species 2: γ=1.66, M=4 g/mol
gamma1 = 1.4
gamma2 = 1.66
M1 = 28.0  # g/mol
M2 = 4.0   # g/mol

# Initial conditions parameters
x_c = 0.5
r_c = 0.25
w1 = 0.6
w2 = 0.2
k = 20
u0 = 1.0
p0 = 0.9

# Smooth tanh profile
r = np.abs(x - x_c)
rhoY1_0 = (w1 / 2) * (1 - np.tanh(k * (r - r_c)))
rhoY2_0 = (w2 / 2) * (1 + np.tanh(k * (r - r_c)))
rho_0 = rhoY1_0 + rhoY2_0

# Temperature and pressure (uniform)
T_0 = np.ones_like(x) * 300.0
u_0 = np.ones_like(x) * u0

# Setup EOS for each species
eos1 = ideal.IdealGasEOS(gamma=gamma1, M=M1)
eos2 = ideal.IdealGasEOS(gamma=gamma2, M=M2)

# Conservative variables: [rho, rho*u, rho*E, rho*Y1]
# Need to compute E from T
# For ideal gas: e = c_v * T
R_s1 = eos1.R_s  # J/(kg·K)
R_s2 = eos2.R_s
c_v1 = eos1.c_v
c_v2 = eos2.c_v

# Mixed c_v
Y1_0 = rhoY1_0 / rho_0
Y2_0 = rhoY2_0 / rho_0
c_v_mix = Y1_0 * c_v1 + Y2_0 * c_v2
e_0 = c_v_mix * T_0
E_0 = e_0 + 0.5 * u0**2

print(f"Domain: [0, {L}] m, nx={nx}, dx={dx:.6f}")
print(f"Initial density range: [{rho_0.min():.6f}, {rho_0.max():.6f}] kg/m³")
print(f"Initial energy: E_total(0) = {(rho_0 * E_0).sum() * dx:.6e} J")
print(f"Pressure (uniform): p = {p0} Pa")
print(f"Velocity (uniform): u = {u0} m/s")

# Conservative initial state
U0 = np.zeros((nx, 4))
U0[:, 0] = rho_0                    # rho
U0[:, 1] = rho_0 * u_0             # rho * u
U0[:, 2] = rho_0 * E_0             # rho * E
U0[:, 3] = rhoY1_0                 # rho * Y1

print(f"\nInitial state shape: {U0.shape}")
print(f"Initial state check: rho range [{U0[:, 0].min():.3e}, {U0[:, 0].max():.3e}]")

try:
    # Run simulation
    t_final = 8.0  # 8 flow-through times
    CFL = 0.6

    print(f"\nStarting simulation...")
    print(f"  t_final = {t_final} s")
    print(f"  CFL = {CFL}")

    # Cell coordinates
    x_cells = np.linspace(0, L, nx, endpoint=False) + dx / 2

    # Call solver: run_1d(case_params)
    # case_params contains eos_list, x_cells, U_init, t_end, CFL, bc_left, bc_right, etc.
    case_params = {
        'eos_list': [eos1, eos2],
        'x_cells': x_cells,
        'U_init': U0,
        't_end': t_final,
        'CFL': CFL,
        'bc_left': 'periodic',
        'bc_right': 'periodic',
        'verbose': True
    }

    results = run_1d(case_params)

    # Extract final state
    U_final = results['U_final']
    t_history = results.get('t_history', [])

    print(f"\nSimulation completed!")
    print(f"Final state shape: {U_final.shape}")
    print(f"Final rho range: [{U_final[:, 0].min():.3e}, {U_final[:, 0].max():.3e}]")

    # Compute pressure and check PE preservation
    rho_final = U_final[:, 0]
    u_final = U_final[:, 1] / rho_final
    E_final = U_final[:, 2] / rho_final
    rhoY1_final = U_final[:, 3]
    Y1_final = rhoY1_final / rho_final
    Y2_final = 1.0 - Y1_final

    # Internal energy
    e_final = E_final - 0.5 * u_final**2

    # Reconstruct temperature from internal energy
    # e = c_v(Y) * T => T = e / c_v(Y)
    c_v_final = Y1_final * c_v1 + Y2_final * c_v2
    T_final = e_final / c_v_final

    # Pressure from EOS (mixture rule)
    # For ideal gas: p = ρ_i * R_i * T * Y_i sum
    # p = rho * (Σ Y_i * R_i) * T
    R_mix_final = Y1_final * R_s1 + Y2_final * R_s2
    p_final = rho_final * R_mix_final * T_final

    # Check PE preservation (L2 norm)
    L2_p = np.sqrt(np.mean(((p_final - p0) / p0)**2))
    max_p_rel_error = np.max(np.abs((p_final - p0) / p0))
    max_u_error = np.max(np.abs(u_final - u0))

    print(f"\n" + "=" * 80)
    print("PRESSURE-EQUILIBRIUM CHECK")
    print("=" * 80)
    print(f"L2(p) norm:              {L2_p:.6e} (PASS criterion: < 1e-4)")
    print(f"max(|Δp/p₀|):           {max_p_rel_error:.6e}")
    print(f"max(|Δu|):              {max_u_error:.6e}")

    # Energy conservation check
    E_total_final = (rho_final * E_final).sum() * dx
    E_relative_error = np.abs((E_total_final - (rho_0 * E_0).sum() * dx) / ((rho_0 * E_0).sum() * dx))
    print(f"\nEnergy conservation:")
    print(f"E_total(0):    {(rho_0 * E_0).sum() * dx:.6e} J")
    print(f"E_total(T):    {E_total_final:.6e} J")
    print(f"|ΔE/E₀|:        {E_relative_error:.6e} (PASS: < 1e-12)")

    # Verdict
    pass_pe = L2_p < 1e-4
    pass_energy = E_relative_error < 1e-12
    pass_overall = pass_pe and pass_energy

    print(f"\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)
    print(f"PE preservation (L2 < 1e-4):  {'PASS' if pass_pe else 'FAIL'}")
    print(f"Energy conservation (< 1e-12): {'PASS' if pass_energy else 'FAIL'}")
    print(f"OVERALL:                       {'PASS' if pass_overall else 'FAIL'}")

    # Save plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.plot(x, rhoY1_final, 'b-', linewidth=1.5, label='Species 1 (ρY₁)')
    ax.plot(x, rhoY2_final, 'r-', linewidth=1.5, label='Species 2 (ρY₂)')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('ρY (kg/m³)')
    ax.set_title('Species Density Profile (t=8.0s)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(x, u_final, 'g-', linewidth=1.5, label='u(t)')
    ax.axhline(u0, color='k', linestyle='--', label=f'u₀ = {u0}')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('u (m/s)')
    ax.set_title('Velocity Profile')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(x, p_final, 'b-', linewidth=1.5, label='p(t)')
    ax.axhline(p0, color='k', linestyle='--', label=f'p₀ = {p0}')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('p (Pa)')
    ax.set_title('Pressure Profile')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.semilogy(x, np.abs((p_final - p0) / p0) + 1e-16, 'k-', linewidth=1)
    ax.axhline(1e-4, color='r', linestyle='--', label='PASS criterion')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('|Δp/p₀|')
    ax.set_title('Pressure Equilibrium Error')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'species_density_t8.png', dpi=100, bbox_inches='tight')
    print(f"\nSaved: species_density_t8.png")
    plt.close()

    # Save report
    with open(output_dir / 'report.md', 'w') as f:
        f.write("# Validation Report: Smooth Interface Advection (Ideal Gas, Case A)\n\n")
        f.write(f"**Test Date:** {np.datetime64('today')}\n\n")
        f.write("## Case Parameters\n\n")
        f.write(f"- Domain: [0, {L}] m (periodic boundary)\n")
        f.write(f"- Grid: {nx} cells, Δx = {dx:.6f} m\n")
        f.write(f"- Initial velocity: u₀ = {u0} m/s (uniform)\n")
        f.write(f"- Initial pressure: p₀ = {p0} Pa (uniform)\n")
        f.write(f"- CFL: {CFL}\n")
        f.write(f"- Total time: t = {t_final} s (8 flow-through times)\n\n")

        f.write("## Results\n\n")
        f.write("| Metric | Value | Criterion | Status |\n")
        f.write("|--------|-------|-----------|--------|\n")
        f.write(f"| L2(p) error | {L2_p:.6e} | < 1e-4 | {'PASS' if pass_pe else 'FAIL'} |\n")
        f.write(f"| Energy conservation | {E_relative_error:.6e} | < 1e-12 | {'PASS' if pass_energy else 'FAIL'} |\n")
        f.write(f"| max(\\|Δp/p₀\\|) | {max_p_rel_error:.6e} | - | - |\n")
        f.write(f"| max(\\|Δu\\|) | {max_u_error:.6e} | < 1e-10 | {'PASS' if max_u_error < 1e-10 else 'WARN'} |\n\n")

        f.write(f"## Verdict: **{'PASS' if pass_overall else 'FAIL'}**\n\n")
        if not pass_overall:
            if not pass_pe:
                f.write(f"❌ Pressure equilibrium criterion not met (L2 = {L2_p:.3e})\n")
            if not pass_energy:
                f.write(f"❌ Energy conservation criterion not met (|ΔE/E₀| = {E_relative_error:.3e})\n")

    print(f"Saved: report.md")
    print("\nTest completed successfully!")

except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
