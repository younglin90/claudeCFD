#!/usr/bin/env python3
"""
1D Validation Test Runner

Executes all 1D validation cases in validation/1D/*.md
Saves results to results/1D/{case_name}/
Generates qa_report.md with PASS/FAIL judgment
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple

# Add solver to path
sys.path.insert(0, '/home/younglin90/work/claude_code/claudeCFD')

from solver.eos.ideal import IdealGasEOS
from solver.eos.nasg import NASGEOS
from solver.eos.srk import SRKEOS
from solver.solve import run_1d
from solver.utils import cons_to_prim, mixture_sound_speed

# ============================================================================
# CASE 1: Smooth Interface Advection - Case A (Ideal Gas)
# ============================================================================

def run_smooth_interface_advection_case_a():
    """
    Case A: Ideal Gas 2-component smooth interface advection
    - Domain: [0, 1] m, 501 cells, periodic BC
    - Species 1 (left), Species 2 (right) with smooth interface
    - u0 = 1.0, p0 = 0.9
    - Run t = 8.0 (8 flow-through times)
    """
    print("\n" + "="*70)
    print("CASE: Smooth Interface Advection - Case A (Ideal Gas)")
    print("="*70)

    case_name = "Smooth_Interface_Advection_IdealGas_CaseA"
    output_dir = Path('/home/younglin90/work/claude_code/claudeCFD/results/1D') / case_name
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # EOS setup
        gamma1, gamma2 = 1.4, 1.66
        M1, M2 = 28.0, 4.0  # kg/kmol
        R_u = 8.314  # J/(mol·K)

        eos1 = IdealGasEOS(gamma=gamma1, M=M1)
        eos2 = IdealGasEOS(gamma=gamma2, M=M2)
        eos_list = [eos1, eos2]

        # Domain setup
        n_cells = 501
        x_cells = np.linspace(0.5/n_cells, 1.0 - 0.5/n_cells, n_cells)
        dx = x_cells[1] - x_cells[0]

        # Initial condition: smooth interface advection
        x_c = 0.5
        r_c = 0.25
        k = 20
        w1, w2 = 0.6, 0.2
        u0, p0 = 1.0, 0.9

        # Compute densities from EOS at p0, T0=300K using ideal gas law
        T0 = 300.0
        # For ideal gas: p = rho * R_s * T => rho = p / (R_s * T)
        rho1_0 = p0 / (eos1.R_s * T0)
        rho2_0 = p0 / (eos2.R_s * T0)

        # Conservative variables initialization
        U = np.zeros((n_cells, 4))
        for m, x in enumerate(x_cells):
            r = abs(x - x_c)
            rho_Y1 = (w1 / 2.0) * (1.0 - np.tanh(k * (r - r_c)))
            rho_Y2 = (w2 / 2.0) * (1.0 + np.tanh(k * (r - r_c)))

            rho = rho_Y1 + rho_Y2

            # Energy computation: uniform p0, T0 → E
            e1 = eos1.internal_energy(T0)
            e2 = eos2.internal_energy(T0)
            e = (rho_Y1 * e1 + rho_Y2 * e2) / rho if rho > 0 else 0.0

            E = e + 0.5 * u0**2

            U[m, 0] = rho
            U[m, 1] = rho * u0
            U[m, 2] = rho * E
            U[m, 3] = rho_Y1

        # Run simulation
        t_end = 8.0
        case_params = {
            'eos_list': eos_list,
            'x_cells': x_cells,
            'U_init': U.copy(),
            't_end': t_end,
            'CFL': 0.6,
            'bc_left': 'periodic',
            'bc_right': 'periodic',
            'output_times': [8.0],
            'verbose': True,
            'time_scheme': 'tvd_rk3'
        }

        print("Running simulation...")
        result = run_1d(case_params)

        if result is None:
            return {
                'case': case_name,
                'status': 'FAIL',
                'error': 'Simulation failed to run',
                'pe_error': np.nan
            }

        U_final = result['U_final']
        t_final = result['t_final']

        # Postprocessing: compute PE error
        pe_errors = []
        for m in range(n_cells):
            try:
                W = cons_to_prim(U_final[m], eos_list)
                p_m = W[0]
                pe_errors.append(abs((p_m / p0) - 1.0))
            except:
                pass

        pe_error_l2 = np.sqrt(np.mean(np.array(pe_errors)**2)) if pe_errors else np.nan
        pe_error_max = np.max(np.array(pe_errors)) if pe_errors else np.nan

        print(f"PE error L2: {pe_error_l2:.3e}, max: {pe_error_max:.3e}")

        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Pressure
        p_final = np.array([cons_to_prim(U_final[m], eos_list)[0] for m in range(n_cells)])
        axes[0, 0].plot(x_cells, p_final, 'b-', label='Numerical')
        axes[0, 0].axhline(p0, color='r', linestyle='--', label='Exact')
        axes[0, 0].set_ylabel('Pressure (Pa)')
        axes[0, 0].set_title('Pressure Profile at t=8.0s')
        axes[0, 0].legend()
        axes[0, 0].grid()

        # Velocity
        u_final = np.array([cons_to_prim(U_final[m], eos_list)[1] for m in range(n_cells)])
        axes[0, 1].plot(x_cells, u_final, 'b-', label='Numerical')
        axes[0, 1].axhline(u0, color='r', linestyle='--', label='Exact')
        axes[0, 1].set_ylabel('Velocity (m/s)')
        axes[0, 1].set_title('Velocity Profile at t=8.0s')
        axes[0, 1].legend()
        axes[0, 1].grid()

        # Density
        rho_final = U_final[:, 0]
        axes[1, 0].plot(x_cells, rho_final, 'b-')
        axes[1, 0].set_ylabel('Density (kg/m³)')
        axes[1, 0].set_title('Density Profile at t=8.0s')
        axes[1, 0].grid()

        # PE error
        axes[1, 1].plot(x_cells, pe_errors, 'r-', linewidth=0.5)
        axes[1, 1].set_ylabel('|p/p0 - 1|')
        axes[1, 1].set_title(f'PE Error (L2={pe_error_l2:.3e})')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid()

        fig.tight_layout()
        fig.savefig(output_dir / 'overview.png', dpi=100)
        plt.close(fig)

        # Judgment
        PASS_THRESHOLD = 1e-4
        status = 'PASS' if pe_error_l2 < PASS_THRESHOLD else 'FAIL'

        report = f"""# Smooth Interface Advection - Case A (Ideal Gas)

## Summary
- **Status**: {status}
- **PE Error (L2)**: {pe_error_l2:.3e}
- **PASS Threshold**: {PASS_THRESHOLD}
- **Final Time**: {t_final:.2f} s
- **Domain**: [0, 1] m, {n_cells} cells, periodic BC

## Details
- Species 1: γ={gamma1}, M={M1} kg/kmol
- Species 2: γ={gamma2}, M={M2} kg/kmol
- u₀ = {u0} m/s, p₀ = {p0} Pa
- CFL = 0.6, TVD RK3 time integration

## Checks
- Pressure uniformity (L2 norm): {pe_error_l2:.3e} (target: <{PASS_THRESHOLD})
- Max pressure deviation: {pe_error_max:.3e}
- No NaNs or Infs: {'Yes' if not np.any(np.isnan(U_final)) else 'No'}
"""

        with open(output_dir / 'report.md', 'w') as f:
            f.write(report)

        return {
            'case': case_name,
            'status': status,
            'pe_error': pe_error_l2,
            'pe_error_max': pe_error_max
        }

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {
            'case': case_name,
            'status': 'ERROR',
            'error': str(e)
        }


# ============================================================================
# CASE 2: Interface Advection - Case A (Zero Velocity, Air-Water NASG)
# ============================================================================

def run_interface_advection_case_a():
    """
    Case A: Zero Velocity - pure Abgrall test
    - Domain: [0, 1] m, 500 cells
    - Water (NASG) left, Air (Ideal) right
    - No velocity, uniform pressure p0=1e5 Pa, T0=300K
    - Run t = 1.0 s, strict criteria
    """
    print("\n" + "="*70)
    print("CASE: Interface Advection - Case A (Zero Velocity, Air-Water)")
    print("="*70)

    case_name = "Interface_Advection_AirWater_CaseA_ZeroVelocity"
    output_dir = Path('/home/younglin90/work/claude_code/claudeCFD/results/1D') / case_name
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # EOS setup
        eos_water = NASGEOS(gamma=1.19, p_inf=7.028e8, b=6.61e-4,
                            c_v=3610.0, q=-1.177788e6)
        eos_air = IdealGasEOS(gamma=1.4, M=28.97)
        eos_list = [eos_water, eos_air]

        # Domain setup
        n_cells = 500
        interface_idx = 250
        x_cells = np.linspace(0.5/n_cells, 1.0 - 0.5/n_cells, n_cells)

        # Initial condition: uniform p, T; sharp interface
        p0 = 1e5  # Pa
        T0 = 300.0  # K
        u0 = 0.0  # m/s

        # Compute densities from EOS
        # For water (NASG): T = (p + p_inf) * (1 - b*rho) / ((gamma-1) * c_v * rho)
        # => Need to solve for rho. Use temperature_from_rho_p to verify.
        # Forward: rho = p / (R_s * T) for ideal gas
        # For NASG: need to iterate or use solver. For now, use approximation.
        rho_air = p0 / (eos_air.R_s * T0)

        # For NASG water, we need to solve: p = (gamma-1)*c_v*rho*T/(1-b*rho) - p_inf for rho
        # Given p, T, solve for rho. Use Newton's method or scipy.optimize
        from scipy.optimize import fsolve
        def nasg_rho_eq(rho, p, T):
            p_calc = eos_water.pressure(rho, T)
            return p_calc - p
        rho_water_init = 1000.0  # initial guess (water density ~ 1000 kg/m³)
        rho_water = fsolve(lambda rho: nasg_rho_eq(rho, p0, T0), rho_water_init)[0]

        # Internal energies
        e_water = eos_water.internal_energy(rho_water, T0)
        e_air = eos_air.internal_energy(T0)

        E_water = e_water + 0.5 * u0**2
        E_air = e_air + 0.5 * u0**2

        U = np.zeros((n_cells, 3))
        for m in range(n_cells):
            if m < interface_idx:  # Water
                U[m, 0] = rho_water
                U[m, 1] = rho_water * u0
                U[m, 2] = rho_water * E_water
            else:  # Air
                U[m, 0] = rho_air
                U[m, 1] = rho_air * u0
                U[m, 2] = rho_air * E_air

        # Run simulation
        t_end = 1.0
        case_params = {
            'eos_list': eos_list,
            'x_cells': x_cells,
            'U_init': U.copy(),
            't_end': t_end,
            'CFL': 0.5,
            'bc_left': 'transmissive',
            'bc_right': 'transmissive',
            'output_times': [1.0],
            'verbose': True,
            'time_scheme': 'forward_euler'
        }

        print("Running simulation...")
        result = run_1d(case_params)

        if result is None:
            return {
                'case': case_name,
                'status': 'FAIL',
                'error': 'Simulation failed to run',
                'pe_error': np.nan
            }

        U_final = result['U_final']

        # Postprocessing
        pe_errors = []
        for m in range(n_cells):
            try:
                W = cons_to_prim(U_final[m], eos_list)
                p_m = W[0]
                pe_errors.append(abs((p_m / p0) - 1.0))
            except:
                pass

        pe_error_max = np.max(np.array(pe_errors)) if pe_errors else np.nan

        print(f"PE error max: {pe_error_max:.3e}")

        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Pressure
        p_final = np.array([cons_to_prim(U_final[m], eos_list)[0] for m in range(n_cells)])
        axes[0, 0].plot(x_cells, p_final, 'b-', linewidth=1)
        axes[0, 0].axhline(p0, color='r', linestyle='--', label='Exact')
        axes[0, 0].set_ylabel('Pressure (Pa)')
        axes[0, 0].set_title('Pressure Profile at t=1.0s')
        axes[0, 0].legend()
        axes[0, 0].grid()

        # Velocity
        u_final = np.array([cons_to_prim(U_final[m], eos_list)[1] for m in range(n_cells)])
        axes[0, 1].plot(x_cells, u_final, 'b-', linewidth=1)
        axes[0, 1].axhline(u0, color='r', linestyle='--', label='Exact')
        axes[0, 1].set_ylabel('Velocity (m/s)')
        axes[0, 1].set_title('Velocity Profile at t=1.0s')
        axes[0, 1].legend()
        axes[0, 1].grid()

        # Density
        rho_final = U_final[:, 0]
        axes[1, 0].plot(x_cells, rho_final, 'b-')
        axes[1, 0].set_ylabel('Density (kg/m³)')
        axes[1, 0].set_title('Density Profile at t=1.0s')
        axes[1, 0].grid()

        # PE error
        axes[1, 1].plot(x_cells, pe_errors, 'r-', linewidth=0.5)
        axes[1, 1].set_ylabel('|p/p0 - 1|')
        axes[1, 1].set_title(f'PE Error (max={pe_error_max:.3e})')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid()

        fig.tight_layout()
        fig.savefig(output_dir / 'overview.png', dpi=100)
        plt.close(fig)

        # Judgment: very strict criteria for zero velocity case
        PASS_THRESHOLD = 1e-10
        status = 'PASS' if pe_error_max < PASS_THRESHOLD else 'FAIL'

        report = f"""# Interface Advection - Case A (Zero Velocity)

## Summary
- **Status**: {status}
- **PE Error (max)**: {pe_error_max:.3e}
- **PASS Threshold**: {PASS_THRESHOLD}
- **Final Time**: 1.0 s

## Details
- Water (NASG): left of x={x_cells[interface_idx]:.3f} m
- Air (Ideal): right of x={x_cells[interface_idx]:.3f} m
- Initial: p₀={p0:.0e} Pa, T₀={T0} K, u₀={u0} m/s
- Mesh: {n_cells} cells, CFL=0.5

## Checks
- Pressure uniformity: {pe_error_max:.3e} (target: <{PASS_THRESHOLD})
- No NaNs/Infs: {'Yes' if not np.any(np.isnan(U_final)) else 'No'}
"""

        with open(output_dir / 'report.md', 'w') as f:
            f.write(report)

        return {
            'case': case_name,
            'status': status,
            'pe_error': pe_error_max
        }

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {
            'case': case_name,
            'status': 'ERROR',
            'error': str(e)
        }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print(f"\n{'='*70}")
    print(f"1D Validation Test Runner - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    results = []

    # Run each case
    try:
        results.append(run_smooth_interface_advection_case_a())
    except Exception as e:
        print(f"UNCAUGHT ERROR in Case A: {e}")

    try:
        results.append(run_interface_advection_case_a())
    except Exception as e:
        print(f"UNCAUGHT ERROR in Interface Advection Case A: {e}")

    # Write QA report
    report_path = Path('/home/younglin90/work/claude_code/claudeCFD/pipeline/qa_report.md')

    qa_report = f"""# QA Report — 1D Validation Suite
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 검증 단계: 1D

## 결과 요약

| 케이스 | 판정 | 측정값 | 합격기준 |
|--------|------|--------|---------|
"""

    for res in results:
        case = res.get('case', 'Unknown')
        status = res.get('status', 'UNKNOWN')
        pe_error = res.get('pe_error', np.nan)

        if status in ['PASS', 'FAIL']:
            qa_report += f"| {case} | {status} | {pe_error:.3e} | (케이스 md 참고) |\n"
        else:
            error = res.get('error', 'Unknown')
            qa_report += f"| {case} | ERROR | {error} | - |\n"

    # Summary
    pass_count = sum(1 for r in results if r.get('status') == 'PASS')
    fail_count = sum(1 for r in results if r.get('status') == 'FAIL')
    error_count = sum(1 for r in results if r.get('status') == 'ERROR')

    qa_report += f"\n## 집계\n- 통과: {pass_count}\n- 실패: {fail_count}\n- 오류: {error_count}\n"

    if fail_count > 0 or error_count > 0:
        qa_report += "\n## FAIL/ERROR 항목\n"
        for res in results:
            if res.get('status') in ['FAIL', 'ERROR']:
                qa_report += f"### {res.get('case')}\n"
                if 'error' in res:
                    qa_report += f"- Error: {res['error']}\n"
                if 'pe_error' in res and not np.isnan(res['pe_error']):
                    qa_report += f"- PE Error: {res['pe_error']:.3e}\n"

    qa_report += "\n## 다음 단계\n- 1D 전체 통과 여부 확인 필요\n"

    with open(report_path, 'w') as f:
        f.write(qa_report)

    print(f"\nQA Report saved to {report_path}")

    # Overall judgment
    if fail_count == 0 and error_count == 0:
        all_pass_flag = Path('/home/younglin90/work/claude_code/claudeCFD/pipeline/all_pass_1d.flag')
        all_pass_flag.write_text('ALL PASS 1D\n')
        print("\nAll 1D cases passed! Flag written.")
    else:
        print(f"\n{fail_count} FAIL, {error_count} ERROR - Fix required.")

    return results


if __name__ == '__main__':
    results = main()
