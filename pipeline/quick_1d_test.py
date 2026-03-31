#!/usr/bin/env python3
"""Quick 1D validation test with minimal cases"""
import sys
sys.path.insert(0, '/home/younglin90/work/claude_code/claudeCFD')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from scipy.optimize import fsolve

from solver.eos.ideal import IdealGasEOS
from solver.eos.nasg import NASGEOS
from solver.solve import run_1d
from solver.utils import cons_to_prim

def test_ideal_gas_simple():
    """Simple test: Ideal Gas smooth interface, t=0.5s instead of 8.0s"""
    print("\n" + "="*70)
    print("TEST: Ideal Gas Simple Interface (Short Time)")
    print("="*70)

    case_name = "Test_IdealGas_Short"
    output_dir = Path('/home/younglin90/work/claude_code/claudeCFD/results/1D') / case_name
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # EOS
        eos1 = IdealGasEOS(gamma=1.4, M=28.0)
        eos2 = IdealGasEOS(gamma=1.66, M=4.0)
        eos_list = [eos1, eos2]

        # Mesh
        n_cells = 101  # smaller for speed
        x_cells = np.linspace(0.5/n_cells, 1.0 - 0.5/n_cells, n_cells)

        # Initial condition
        x_c, r_c, k = 0.5, 0.25, 20
        w1, w2 = 0.6, 0.2
        u0, p0, T0 = 1.0, 0.9, 300.0

        rho1_0 = p0 / (eos1.R_s * T0)
        rho2_0 = p0 / (eos2.R_s * T0)
        e1 = eos1.internal_energy(T0)
        e2 = eos2.internal_energy(T0)

        U = np.zeros((n_cells, 4))
        for m, x in enumerate(x_cells):
            r = abs(x - x_c)
            rho_Y1 = (w1 / 2.0) * (1.0 - np.tanh(k * (r - r_c)))
            rho_Y2 = (w2 / 2.0) * (1.0 + np.tanh(k * (r - r_c)))
            rho = rho_Y1 + rho_Y2

            e = (rho_Y1 * e1 + rho_Y2 * e2) / rho if rho > 0 else 0.0
            E = e + 0.5 * u0**2

            U[m, 0] = rho
            U[m, 1] = rho * u0
            U[m, 2] = rho * E
            U[m, 3] = rho_Y1

        # Run
        params = {
            'eos_list': eos_list,
            'x_cells': x_cells,
            'U_init': U,
            't_end': 0.5,  # Short time
            'CFL': 0.6,
            'bc_left': 'periodic',
            'bc_right': 'periodic',
            'output_times': [0.5],
            'verbose': True,
            'time_scheme': 'tvd_rk3'
        }

        print(f"Running {n_cells}-cell ideal gas simulation...")
        result = run_1d(params)

        if not result:
            print("ERROR: Simulation failed")
            return 'FAIL'

        U_final = result['U_final']

        # PE error
        pe_errs = []
        for m in range(n_cells):
            try:
                W = cons_to_prim(U_final[m], eos_list)
                p_m = W[0]
                pe_errs.append(abs((p_m / p0) - 1.0))
            except:
                pass

        pe_max = np.max(pe_errs) if pe_errs else np.nan
        print(f"PE error max: {pe_max:.3e}")

        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        p_vals = np.array([cons_to_prim(U_final[m], eos_list)[0] for m in range(n_cells)])
        ax.plot(x_cells, p_vals, 'b-', label='Numerical')
        ax.axhline(p0, color='r', linestyle='--', label='Exact')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('Pressure (Pa)')
        ax.set_title(f'Ideal Gas PE: L∞ error = {pe_max:.3e}')
        ax.legend()
        ax.grid()
        fig.savefig(output_dir / 'pressure.png', dpi=100)
        plt.close(fig)

        status = 'PASS' if pe_max < 1e-4 else 'FAIL'
        print(f"Result: {status}")
        return status

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 'ERROR'


def test_water_air_simple():
    """Simple test: Water-Air zero velocity, t=0.2s"""
    print("\n" + "="*70)
    print("TEST: Water-Air Zero Velocity (Short Time)")
    print("="*70)

    case_name = "Test_WaterAir_Short"
    output_dir = Path('/home/younglin90/work/claude_code/claudeCFD/results/1D') / case_name
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # EOS
        eos_water = NASGEOS(gamma=1.19, p_inf=7.028e8, b=6.61e-4, c_v=3610.0, q=-1.177788e6)
        eos_air = IdealGasEOS(gamma=1.4, M=28.97)
        eos_list = [eos_water, eos_air]

        # Mesh
        n_cells = 50
        interface_idx = 25
        x_cells = np.linspace(0.5/n_cells, 1.0 - 0.5/n_cells, n_cells)

        # Initial
        p0, T0, u0 = 1e5, 300.0, 0.0
        rho_air = p0 / (eos_air.R_s * T0)

        # NASG rho from p, T
        def nasg_rho_eq(rho, p, T):
            return eos_water.pressure(rho, T) - p
        rho_water_init = 1000.0
        rho_water = fsolve(lambda rho: nasg_rho_eq(rho, p0, T0), rho_water_init)[0]

        e_water = eos_water.internal_energy(rho_water, T0)
        e_air = eos_air.internal_energy(T0)
        E_water = e_water + 0.5 * u0**2
        E_air = e_air + 0.5 * u0**2

        U = np.zeros((n_cells, 3))
        for m in range(n_cells):
            if m < interface_idx:
                U[m, 0] = rho_water
                U[m, 1] = rho_water * u0
                U[m, 2] = rho_water * E_water
            else:
                U[m, 0] = rho_air
                U[m, 1] = rho_air * u0
                U[m, 2] = rho_air * E_air

        # Run
        params = {
            'eos_list': eos_list,
            'x_cells': x_cells,
            'U_init': U,
            't_end': 0.2,  # Short time
            'CFL': 0.5,
            'bc_left': 'transmissive',
            'bc_right': 'transmissive',
            'output_times': [0.2],
            'verbose': True,
            'time_scheme': 'forward_euler'
        }

        print(f"Running {n_cells}-cell water-air simulation...")
        result = run_1d(params)

        if not result:
            print("ERROR: Simulation failed")
            return 'FAIL'

        U_final = result['U_final']

        # PE error
        pe_errs = []
        for m in range(n_cells):
            try:
                W = cons_to_prim(U_final[m], eos_list)
                p_m = W[0]
                pe_errs.append(abs((p_m / p0) - 1.0))
            except:
                pass

        pe_max = np.max(pe_errs) if pe_errs else np.nan
        print(f"PE error max: {pe_max:.3e}")

        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        p_vals = np.array([cons_to_prim(U_final[m], eos_list)[0] for m in range(n_cells)])
        ax.plot(x_cells, p_vals, 'b-', label='Numerical')
        ax.axhline(p0, color='r', linestyle='--', label='Exact')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('Pressure (Pa)')
        ax.set_title(f'Water-Air PE: L∞ error = {pe_max:.3e}')
        ax.legend()
        ax.grid()
        fig.savefig(output_dir / 'pressure.png', dpi=100)
        plt.close(fig)

        status = 'PASS' if pe_max < 1e-10 else 'FAIL'
        print(f"Result: {status}")
        return status

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 'ERROR'


if __name__ == '__main__':
    print(f"\n{'='*70}")
    print(f"Quick 1D Validation Tests - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    results = []

    try:
        r1 = test_ideal_gas_simple()
        results.append(('Ideal Gas', r1))
    except Exception as e:
        print(f"UNCAUGHT: {e}")
        results.append(('Ideal Gas', 'ERROR'))

    try:
        r2 = test_water_air_simple()
        results.append(('Water-Air', r2))
    except Exception as e:
        print(f"UNCAUGHT: {e}")
        results.append(('Water-Air', 'ERROR'))

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for case, status in results:
        print(f"{case}: {status}")

    pass_count = sum(1 for _, s in results if s == 'PASS')
    print(f"\nTotal: {pass_count}/{len(results)} PASS")
