#!/usr/bin/env python3
"""
Test A Full Validation: Phase 1 NASG + Ideal Air (100 steps, 1.0 s)

Specification:
- Domain: [0, 1] m, periodic BC
- N=10 cells
- Water (NASG): x∈[0.4, 0.6], α=1
- Air (Ideal): x∉[0.4, 0.6], α=0
- u₀=1.0 m/s, p₀=1e5 Pa, T₀=300 K
- dt=0.01 s fixed, 100 steps, t_end=1.0 s
- CFL=0.4

PASS criteria:
- 100 steps complete
- err_p < 1e-2
- err_u < 1e-2
- ΔE/E < 1e-2
- 0 ≤ α ≤ 1
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/younglin90/work/claude_code/claudeCFD')

from solver.He2024.eos_general import NASGEOS, IdealEOS
from solver.He2024.explicit_mmacm_ex import solve_IMEX, cons_to_prim, prim_to_cons

def run_test_a_100step():
    """Run Test A for 100 steps (full revolution)."""
    print("\n" + "="*80)
    print("Test A: Phase 1 NASG Full 100-Step Validation")
    print("="*80)

    # Setup
    N = 10
    domain = [0, 1]
    dx = (domain[1] - domain[0]) / N
    x_cells = np.linspace(domain[0], domain[1], N, endpoint=False) + dx/2

    # EOS (same as unit test)
    nasg_water = NASGEOS(gamma=2.35, pinf=1e9, kv=943.8, b=6.61e-4, eta=-1167e3)
    ideal_air = IdealEOS(gamma=1.4, kv=717.5, pinf=0.0)

    # Initial condition
    p0 = 1e5
    u0 = 1.0  # m/s
    T0 = 300.0

    # α profile: water slug [0.4, 0.6]
    a1_init = np.where((x_cells >= 0.4) & (x_cells <= 0.6), 1.0 - 1e-6, 1e-6)

    # Phase densities from EOS
    rho1_init = nasg_water.density(p0, T0)
    rho2_init = ideal_air.density(p0, T0)

    # Convert to conservative variables
    a1r1_init, a2r2_init, ru_init, rE_init = prim_to_cons(
        rho1_init * np.ones(N), rho2_init * np.ones(N),
        u0 * np.ones(N), p0 * np.ones(N), a1_init,
        nasg_water, ideal_air
    )

    # Initial energy for conservation check
    E_init = np.sum(rE_init)

    print(f"\nConfiguration:")
    print(f"  N={N}, dx={dx:.4f}, periodic BC")
    print(f"  p₀={p0:.2e}, u₀={u0:.2f}, T₀={T0:.0f}")
    print(f"  Water slug: x∈[0.4, 0.6], α_max={np.max(a1_init):.2e}")
    print(f"  ρ₁(NASG)={rho1_init:.4f} kg/m³")
    print(f"  ρ₂(Ideal)={rho2_init:.4f} kg/m³")
    print(f"  Initial total energy: {E_init:.6e}")

    # Solver parameters
    dt_fixed = 0.01  # s
    t_end = 1.0  # s (100 steps)
    max_steps = int(t_end / dt_fixed)
    cfl = 0.4  # acoustic CFL

    print(f"\nSolver parameters:")
    print(f"  dt={dt_fixed}, t_end={t_end}, max_steps={max_steps}")
    print(f"  CFL={cfl}")
    print(f"  alpha_scheme='tvd', use_mmacm_ex=True, use_apec=True, use_compression=True")
    print(f"  use_material_cfl=False")

    # Run solver
    try:
        print(f"\nRunning solve_IMEX for {max_steps} steps...")
        sol = solve_IMEX(
            nasg_water, ideal_air,
            a1r1_init, a2r2_init, ru_init, rE_init, a1_init,
            dx, t_end, cfl=cfl,
            bc_l='periodic', bc_r='periodic',
            max_steps=max_steps,
            alpha_scheme='tvd',
            use_mmacm_ex=True,
            use_compression=True,
            use_apec=True,
            use_material_cfl=False,
            print_interval=10,
        )

        # Extract results
        time_final, a1r1_final, a2r2_final, ru_final, rE_final, a1_final = sol

        # Reconstruct primitives
        p_final, u_final, T_final, rho1_final, rho2_final, _, _, _ = cons_to_prim(
            a1r1_final, a2r2_final, ru_final, rE_final, a1_final,
            nasg_water, ideal_air
        )

        # Energy conservation
        E_final = np.sum(rE_final)
        dE_rel = np.abs(E_final - E_init) / E_init

        # Error analysis
        err_p = np.max(np.abs(p_final - p0) / p0)
        err_u = np.max(np.abs(u_final - u0))

        # α bounds
        a1_min, a1_max = np.min(a1_final), np.max(a1_final)
        a1_valid = (a1_min >= 0.0) and (a1_max <= 1.0)

        print(f"\n{'='*80}")
        print(f"✓ Solver completed successfully!")
        print(f"  Final time: {time_final:.6f} s (target {t_end} s)")
        print(f"  Time step convergence: {abs(time_final - t_end) / t_end * 100:.2e}% error")
        print(f"\nFinal state:")
        print(f"  p: min={np.min(p_final):.3e}, max={np.max(p_final):.3e}, range={np.max(p_final)-np.min(p_final):.3e}")
        print(f"  u: min={np.min(u_final):.3e}, max={np.max(u_final):.3e}, range={np.max(u_final)-np.min(u_final):.3e}")
        print(f"  α₁: min={a1_min:.3e}, max={a1_max:.3e}")
        print(f"\nConservation:")
        print(f"  ΔE/E = {dE_rel:.3e}")
        print(f"\nErrors vs initial:")
        print(f"  err_p = {err_p:.3e} (threshold 1e-2)")
        print(f"  err_u = {err_u:.3e} (threshold 1e-2)")
        print(f"\nValidation:")
        print(f"  0 ≤ α ≤ 1: {a1_valid} (min={a1_min:.3e}, max={a1_max:.3e})")

        # PASS/FAIL judgment
        pass_p = err_p < 1e-2
        pass_u = err_u < 1e-2
        pass_e = dE_rel < 1e-2
        pass_a = a1_valid
        pass_complete = abs(time_final - t_end) < 1e-6

        print(f"\nPASS Criteria:")
        print(f"  [{'✓' if pass_complete else '✗'}] 100 steps complete (t={time_final:.6f})")
        print(f"  [{'✓' if pass_p else '✗'}] err_p < 1e-2 ({err_p:.3e})")
        print(f"  [{'✓' if pass_u else '✗'}] err_u < 1e-2 ({err_u:.3e})")
        print(f"  [{'✓' if pass_e else '✗'}] ΔE/E < 1e-2 ({dE_rel:.3e})")
        print(f"  [{'✓' if pass_a else '✗'}] 0 ≤ α ≤ 1 (min={a1_min:.3e}, max={a1_max:.3e})")

        overall_pass = pass_complete and pass_p and pass_u and pass_e and pass_a
        print(f"\n{'='*80}")
        print(f"Result: {'PASS ✓' if overall_pass else 'FAIL ✗'}")
        print(f"{'='*80}\n")

        # Generate plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Test A: Phase 1 NASG (N={N}, 100 steps, t_end={t_end}s)', fontsize=14, fontweight='bold')

        # Pressure
        ax = axes[0, 0]
        ax.plot(x_cells, p_final, 'b-', linewidth=2, label='Final p')
        ax.axhline(p0, color='r', linestyle='--', alpha=0.7, label=f'p₀={p0:.2e}')
        ax.fill_between(x_cells, p0*(1-1e-2), p0*(1+1e-2), alpha=0.2, color='gray', label='±1% band')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('Pressure (Pa)')
        ax.set_title(f'Pressure (err_p={err_p:.3e})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Velocity
        ax = axes[0, 1]
        ax.plot(x_cells, u_final, 'g-', linewidth=2, label='Final u')
        ax.axhline(u0, color='r', linestyle='--', alpha=0.7, label=f'u₀={u0:.2f}')
        ax.fill_between(x_cells, u0-1e-2, u0+1e-2, alpha=0.2, color='gray', label='±0.01 m/s band')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('Velocity (m/s)')
        ax.set_title(f'Velocity (err_u={err_u:.3e})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Volume fraction
        ax = axes[1, 0]
        ax.plot(x_cells, a1_final, 'b-', linewidth=2, marker='o', label='Final α₁')
        ax.axhline(1.0, color='r', linestyle='--', alpha=0.5, label='max bound')
        ax.axhline(0.0, color='r', linestyle='--', alpha=0.5, label='min bound')
        ax.fill_between([0.4, 0.6], -0.1, 1.1, alpha=0.2, color='cyan', label='Water region')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('Volume Fraction')
        ax.set_title(f'Volume Fraction α₁ (min={a1_min:.3e}, max={a1_max:.3e})')
        ax.set_ylim([-0.1, 1.1])
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Density
        ax = axes[1, 1]
        ax.plot(x_cells, rho1_final, 'b-', linewidth=2, marker='s', label='ρ₁ (water)')
        ax.plot(x_cells, rho2_final, 'g-', linewidth=2, marker='^', label='ρ₂ (air)')
        ax.axhline(rho1_init, color='b', linestyle='--', alpha=0.5, label=f'ρ₁,init={rho1_init:.2f}')
        ax.axhline(rho2_init, color='g', linestyle='--', alpha=0.5, label=f'ρ₂,init={rho2_init:.3f}')
        ax.fill_between([0.4, 0.6], 0, 1000, alpha=0.2, color='cyan', label='Water region')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('Density (kg/m³)')
        ax.set_title(f'Phase Densities (ΔE/E={dE_rel:.3e})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        png_path = '/home/younglin90/work/claude_code/claudeCFD/results/cat_A_exact/02A_abgrall_nasg.png'
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {png_path}")
        plt.close()

        # Return results for report
        return {
            'pass': overall_pass,
            'err_p': err_p,
            'err_u': err_u,
            'dE_rel': dE_rel,
            'a1_min': a1_min,
            'a1_max': a1_max,
            'time_final': time_final,
            't_end': t_end,
            'max_steps': max_steps,
        }

    except Exception as ex:
        print(f"\n❌ Solver failed with exception:")
        print(f"  {type(ex).__name__}: {ex}")
        import traceback
        traceback.print_exc()
        return {'pass': False, 'error': str(ex)}

if __name__ == '__main__':
    result = run_test_a_100step()

    # Exit code
    if result.get('pass', False):
        print("\n✓ Test A PASSED")
        sys.exit(0)
    else:
        print("\n✗ Test A FAILED")
        sys.exit(1)
