#!/usr/bin/env python3
"""
Round 3 Validation — Case A: Smooth Interface Advection (Ideal Gas)

Case A-specific test:
- 2-species ideal gas mixture
- smooth interface advection
- 501 cells, t_end=8.0, CFL=0.6
- Periodic boundaries
- Check: PE preservation, energy conservation, no negative density
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

sys.path.insert(0, '/home/younglin90/work/claude_code/claudeCFD')

from solver.solve import run_1d
from solver.eos.ideal import IdealGasEOS

# ============================================================================
# Case A Configuration
# ============================================================================

def setup_case_A():
    """
    Case A: Ideal Gas, 2-species, smooth interface advection

    Species: Species 1 (γ=1.4, M=28) and Species 2 (γ=1.66, M=4)
    """
    # Species properties
    species = [
        {'name': 'Species_1', 'gamma': 1.4, 'M': 28.0},
        {'name': 'Species_2', 'gamma': 1.66, 'M': 4.0},
    ]

    # Compute cv for each species (for internal energy calculation)
    R_u = 8.314  # J/(mol·K)
    for sp in species:
        R_s = R_u / (sp['M'] * 1e-3)  # convert M to kg/mol, then J/(kg·K)
        sp['cv'] = R_s / (sp['gamma'] - 1)

    # Mesh
    n_cells = 501
    x_min, x_max = 0.0, 1.0
    dx = (x_max - x_min) / n_cells
    x_centers = np.linspace(x_min + dx/2, x_max - dx/2, n_cells)

    # Initial conditions
    p0 = 0.9
    T0 = 300.0
    u0 = 1.0

    # Smooth interface parameters
    x_c = 0.5
    r_c = 0.25
    k = 20.0
    w1 = 0.6  # rho_Y_1 at interface
    w2 = 0.2  # rho_Y_2 at interface

    # Compute rho_Y_i from the profile
    r = np.abs(x_centers - x_c)
    rho_Y1_profile = (w1 / 2.0) * (1.0 - np.tanh(k * (r - r_c)))
    rho_Y2_profile = (w2 / 2.0) * (1.0 + np.tanh(k * (r - r_c)))

    # Initial density and species mass fractions
    rho_0 = rho_Y1_profile + rho_Y2_profile  # total density
    Y1_0 = rho_Y1_profile / (rho_0 + 1e-15)
    Y2_0 = rho_Y2_profile / (rho_0 + 1e-15)

    # Build conservative variables U = [rho, rho*u, rho*E, rho*Y_1]
    # (rho*Y_2 is implicit: rho*Y_2 = rho - rho*Y_1)
    # E = e + u^2/2
    # e = sum(Y_i * e_i(T))

    # Internal energy per unit mass (ideal gas)
    e1 = species[0]['cv'] * T0
    e2 = species[1]['cv'] * T0
    e = Y1_0 * e1 + Y2_0 * e2  # mixture internal energy
    E = e + u0**2 / 2.0

    # U has shape (n_cells, n_vars) where n_vars = 2 + N_species = 2 + 2 = 4
    # [rho, rho*u, rho*E, rho*Y_1]
    U = np.zeros((n_cells, 4))
    U[:, 0] = rho_0
    U[:, 1] = rho_0 * u0
    U[:, 2] = rho_0 * E
    U[:, 3] = rho_Y1_profile

    # Initial state array for reference
    state = {
        'n_cells': n_cells,
        'x_min': x_min,
        'x_max': x_max,
        'dx': dx,
        'x': x_centers,
        'species': species,
        'p0': p0,
        'T0': T0,
        'u0': u0,
        'rho_Y1_0': rho_Y1_profile,
        'rho_Y2_0': rho_Y2_profile,
        'E0': E,
        'e0': e,
    }

    return U, state


# ============================================================================
# Run Simulation
# ============================================================================

def run_case_A(t_end=8.0, cfl=0.6, verbose=True):
    """
    Run Case A simulation and return results
    """
    U_init, state = setup_case_A()

    # Create EOS objects
    eos_list = [
        IdealGasEOS(gamma=sp['gamma'], M=sp['M'])
        for sp in state['species']
    ]

    print("=" * 70)
    print("CASE A: Ideal Gas Smooth Interface Advection")
    print("=" * 70)
    print(f"Grid: {state['n_cells']} cells, CFL={cfl}")
    print(f"Domain: [{state['x_min']}, {state['x_max']}]")
    print(f"t_end = {t_end}")
    print(f"Species: {[s['name'] for s in state['species']]}")
    print(f"p0={state['p0']}, T0={state['T0']}, u0={state['u0']}")
    print()

    try:
        # Prepare case parameters dictionary
        case_params = {
            'eos_list': eos_list,
            'x_cells': state['x'],
            'U_init': U_init,
            't_end': t_end,
            'CFL': cfl,
            'bc_left': 'periodic',
            'bc_right': 'periodic',
            'verbose': verbose,
        }

        result = run_1d(case_params)

        print(f"\n✓ Simulation completed successfully")
        return result, state

    except Exception as e:
        print(f"\n✗ Simulation failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None, state


# ============================================================================
# Validation Metrics
# ============================================================================

def compute_metrics(result, state, verbose=True):
    """
    Compute pressure/velocity/energy metrics from result
    """
    if result is None:
        return None

    U_final = result['U_final']
    t_final = result['t_final']
    history = result.get('history', {})

    rho_final = U_final[:, 0]

    # Recover primitive variables
    # Need to do proper cons-to-prim conversion for each cell
    from solver.utils import cons_to_prim

    eos_list = [
        IdealGasEOS(gamma=sp['gamma'], M=sp['M'])
        for sp in state['species']
    ]

    # Convert each cell's conservative state to primitive
    n_cells = U_final.shape[0]
    prim = np.zeros((n_cells, 4))  # [p, u, T, Y_1]
    for i in range(n_cells):
        prim[i, :] = cons_to_prim(U_final[i, :], eos_list)

    p_final = prim[:, 0]
    u_final = prim[:, 1]
    T_final = prim[:, 2]

    # ===== Metrics =====

    # 1. Pressure uniformity
    p_mean = np.mean(p_final)
    p_norm = np.linalg.norm(p_final - state['p0']) / np.sqrt(len(p_final))
    p_rel_norm = p_norm / state['p0']
    p_max_var = np.max(np.abs(p_final - state['p0'])) / state['p0']

    # 2. Velocity uniformity
    u_mean = np.mean(u_final)
    u_norm = np.linalg.norm(u_final - state['u0']) / np.sqrt(len(u_final))
    u_rel_norm = u_norm / state['u0']
    u_max_var = np.max(np.abs(u_final - state['u0'])) / state['u0']

    # 3. Density bounds
    rho_min = np.min(rho_final)
    rho_max = np.max(rho_final)

    # 4. Energy conservation
    e_final = U_final[2, :] / rho_final - u_final**2 / 2.0
    E_final_total = np.sum(U_final[2, :]) * state['dx']
    E_init_total = np.sum(state['E0']) * state['dx'] * np.mean(state['rho_Y1_0'] + state['rho_Y2_0'])
    energy_error = (E_final_total - E_init_total) / (np.abs(E_init_total) + 1e-16)

    metrics = {
        't_final': t_final,
        'p_mean': p_mean,
        'p_norm': p_norm,
        'p_rel_norm': p_rel_norm,
        'p_max_var': p_max_var,
        'u_mean': u_mean,
        'u_norm': u_norm,
        'u_rel_norm': u_rel_norm,
        'u_max_var': u_max_var,
        'rho_min': rho_min,
        'rho_max': rho_max,
        'energy_error': energy_error,
        'U_final': U_final,
        'p_final': p_final,
        'u_final': u_final,
        'T_final': T_final,
    }

    if verbose:
        print("\n" + "=" * 70)
        print("VALIDATION METRICS (t=8.0)")
        print("=" * 70)
        print(f"Pressure uniformity (L2 norm):")
        print(f"  |p - p0| (L2):        {p_norm:.6e}")
        print(f"  relative (L2):        {p_rel_norm:.6e}")
        print(f"  max relative:         {p_max_var:.6e}")
        print(f"  PASS criterion: < 1e-4")
        print()
        print(f"Velocity uniformity (L2 norm):")
        print(f"  |u - u0| (L2):        {u_norm:.6e}")
        print(f"  relative (L2):        {u_rel_norm:.6e}")
        print(f"  max relative:         {u_max_var:.6e}")
        print(f"  PASS criterion: < 1e-4")
        print()
        print(f"Density bounds:")
        print(f"  min: {rho_min:.6e}")
        print(f"  max: {rho_max:.6e}")
        print(f"  No negatives: {rho_min > 0}")
        print()
        print(f"Energy conservation:")
        print(f"  relative error: {energy_error:.6e}")
        print(f"  PASS criterion: < 1e-12")
        print()

    return metrics


# ============================================================================
# Plotting
# ============================================================================

def plot_results(result, state, metrics, output_dir):
    """
    Create validation plots
    """
    os.makedirs(output_dir, exist_ok=True)

    x = state['x']

    U_final = metrics['U_final']
    p_final = metrics['p_final']
    u_final = metrics['u_final']
    rho_final = U_final[0, :]
    rho_Y1_final = U_final[3, :]
    rho_Y2_final = U_final[4, :]

    # Figure 1: Species density and flow variables
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(x, rho_Y1_final, 'b-', label='ρY₁ (final)')
    axes[0, 0].plot(x, state['rho_Y1_0'], 'b--', alpha=0.5, label='ρY₁ (init)')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('ρY₁')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_title('Species 1 Mass Fraction Profile')

    axes[0, 1].plot(x, rho_Y2_final, 'r-', label='ρY₂ (final)')
    axes[0, 1].plot(x, state['rho_Y2_0'], 'r--', alpha=0.5, label='ρY₂ (init)')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('ρY₂')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_title('Species 2 Mass Fraction Profile')

    axes[1, 0].plot(x, u_final, 'g-', label='u (final)')
    axes[1, 0].axhline(state['u0'], color='g', linestyle='--', alpha=0.5, label=f'u₀ = {state["u0"]}')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('u')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_title('Velocity Profile')

    axes[1, 1].plot(x, p_final, 'k-', label='p (final)')
    axes[1, 1].axhline(state['p0'], color='k', linestyle='--', alpha=0.5, label=f'p₀ = {state["p0"]}')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('p')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_title('Pressure Profile')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'species_density_t8.png'), dpi=100)
    plt.close()

    print(f"✓ Saved: species_density_t8.png")


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    output_dir = '/home/younglin90/work/claude_code/claudeCFD/results/1D/Smooth_Interface_Advection_IdealGas'

    # Run simulation
    result, state = run_case_A(t_end=8.0, cfl=0.6, verbose=True)

    if result is not None:
        # Compute metrics
        metrics = compute_metrics(result, state, verbose=True)

        # Check PASS/FAIL
        passes = []
        if metrics['p_rel_norm'] < 1e-4:
            passes.append(('Pressure uniformity', True))
        else:
            passes.append(('Pressure uniformity', False))

        if metrics['u_rel_norm'] < 1e-4:
            passes.append(('Velocity uniformity', True))
        else:
            passes.append(('Velocity uniformity', False))

        if metrics['rho_min'] > 0:
            passes.append(('No negative density', True))
        else:
            passes.append(('No negative density', False))

        if abs(metrics['energy_error']) < 1e-12:
            passes.append(('Energy conservation', True))
        else:
            passes.append(('Energy conservation', False))

        print("\n" + "=" * 70)
        print("PASS/FAIL SUMMARY")
        print("=" * 70)
        for test_name, passed in passes:
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{status:8s} | {test_name}")

        all_pass = all(p[1] for p in passes)
        print()
        if all_pass:
            print("✓✓✓ CASE A: ALL TESTS PASS ✓✓✓")
        else:
            print("✗✗✗ CASE A: SOME TESTS FAILED ✗✗✗")

        # Create plots
        plot_results(result, state, metrics, output_dir)

        sys.exit(0 if all_pass else 1)
    else:
        print("\n✗ Case A validation failed (simulation error)")
        sys.exit(2)
