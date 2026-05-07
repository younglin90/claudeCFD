#!/usr/bin/env python3
"""
Test A Full Validation v2: Phase 1 NASG + Ideal Air (t_end=1.0s)

명세서 엄격 해석:
- t_end = 1.0 s 도달 필수 (고정 dt=0.01s 가정, 100 iteration)
- N=10 cells
- err_p < 1e-2, err_u < 1e-2

현재 solve_IMEX는 CFL 기반 적응형 dt를 사용하므로,
max_steps를 매우 크게 설정하고 t_end=1.0s 도달을 추적한다.
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/younglin90/work/claude_code/claudeCFD')

from solver.He2024.eos_general import NASGEOS, IdealEOS
from solver.He2024.explicit_mmacm_ex import solve_IMEX, cons_to_prim, prim_to_cons

def run_test_a_1s():
    """Run Test A until t_end=1.0s."""
    print("\n" + "="*80)
    print("Test A: Phase 1 NASG — Full 1.0s Simulation")
    print("="*80)

    # Setup
    N = 10
    domain = [0, 1]
    dx = (domain[1] - domain[0]) / N
    x_cells = np.linspace(domain[0], domain[1], N, endpoint=False) + dx/2

    # EOS
    nasg_water = NASGEOS(gamma=2.35, pinf=1e9, kv=943.8, b=6.61e-4, eta=-1167e3)
    ideal_air = IdealEOS(gamma=1.4, kv=717.5, pinf=0.0)

    # Initial condition
    p0 = 1e5
    u0 = 1.0  # m/s
    T0 = 300.0

    # α profile
    a1_init = np.where((x_cells >= 0.4) & (x_cells <= 0.6), 1.0 - 1e-6, 1e-6)

    # Phase densities
    rho1_init = nasg_water.density(p0, T0)
    rho2_init = ideal_air.density(p0, T0)

    # Conservative variables
    a1r1_init, a2r2_init, ru_init, rE_init = prim_to_cons(
        rho1_init * np.ones(N), rho2_init * np.ones(N),
        u0 * np.ones(N), p0 * np.ones(N), a1_init,
        nasg_water, ideal_air
    )

    # Initial energy
    E_init = np.sum(rE_init)

    print(f"\nConfiguration:")
    print(f"  N={N}, dx={dx:.4f}, periodic BC")
    print(f"  p₀={p0:.2e}, u₀={u0:.2f}, T₀={T0:.0f}")
    print(f"  Water slug: x∈[0.4, 0.6]")
    print(f"  ρ₁(NASG)={rho1_init:.4f} kg/m³")
    print(f"  ρ₂(Ideal)={rho2_init:.4f} kg/m³")

    # Solver parameters
    t_end = 1.0  # s (full revolution at 1 m/s)
    cfl = 0.4    # acoustic CFL
    max_steps = 10000  # very large to ensure t_end reached

    print(f"\nSolver parameters:")
    print(f"  t_end={t_end} s (full revolution)")
    print(f"  CFL={cfl}, max_steps={max_steps}")
    print(f"  alpha_scheme='tvd', use_mmacm_ex=True, use_apec=True, use_compression=True")

    # Run solver
    try:
        print(f"\nRunning solve_IMEX...")
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
            print_interval=100,
        )

        # Extract results
        time_final, a1r1_final, a2r2_final, ru_final, rE_final, a1_final = sol

        # Reconstruct primitives
        p_final, u_final, T_final, rho1_final, rho2_final, _, _, _ = cons_to_prim(
            a1r1_final, a2r2_final, ru_final, rE_final, a1_final,
            nasg_water, ideal_air
        )

        # Metrics
        E_final = np.sum(rE_final)
        dE_rel = np.abs(E_final - E_init) / E_init
        err_p = np.max(np.abs(p_final - p0) / p0)
        err_u = np.max(np.abs(u_final - u0))
        a1_min, a1_max = np.min(a1_final), np.max(a1_final)
        a1_valid = (a1_min >= 0.0) and (a1_max <= 1.0)

        # Check if t_end reached
        t_reached = (time_final >= t_end * 0.999)  # within 0.1%

        print(f"\n{'='*80}")
        print(f"✓ Solver completed!")
        print(f"  Final time: {time_final:.6f} s (target {t_end} s)")
        print(f"  Time coverage: {100 * time_final / t_end:.1f}%")
        print(f"\nFinal state:")
        print(f"  p: min={np.min(p_final):.3e}, max={np.max(p_final):.3e}")
        print(f"  u: min={np.min(u_final):.3e}, max={np.max(u_final):.3e}")
        print(f"  α₁: min={a1_min:.3e}, max={a1_max:.3e}")
        print(f"\nConservation:")
        print(f"  ΔE/E = {dE_rel:.3e}")
        print(f"\nErrors vs initial:")
        print(f"  err_p = {err_p:.3e} (threshold 1e-2)")
        print(f"  err_u = {err_u:.3e} (threshold 1e-2)")
        print(f"\nPASS Criteria:")
        print(f"  [{'✓' if t_reached else '✗'}] t_end >= 0.99 × target ({time_final:.4f} / {t_end})")
        print(f"  [{'✓' if err_p < 1e-2 else '✗'}] err_p < 1e-2 ({err_p:.3e})")
        print(f"  [{'✓' if err_u < 1e-2 else '✗'}] err_u < 1e-2 ({err_u:.3e})")
        print(f"  [{'✓' if dE_rel < 1e-2 else '✗'}] ΔE/E < 1e-2 ({dE_rel:.3e})")
        print(f"  [{'✓' if a1_valid else '✗'}] 0 ≤ α ≤ 1 (min={a1_min:.3e}, max={a1_max:.3e})")

        overall_pass = t_reached and (err_p < 1e-2) and (err_u < 1e-2) and (dE_rel < 1e-2) and a1_valid

        print(f"\n{'='*80}")
        print(f"Result: {'PASS ✓' if overall_pass else 'FAIL ✗'}")
        print(f"{'='*80}\n")

        # Plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Test A: Phase 1 NASG (N={N}, t={time_final:.4f}s)', fontsize=14, fontweight='bold')

        ax = axes[0, 0]
        ax.plot(x_cells, p_final, 'b-', linewidth=2, label='Final p')
        ax.axhline(p0, color='r', linestyle='--', alpha=0.7, label=f'p₀')
        ax.fill_between(x_cells, p0*(1-1e-2), p0*(1+1e-2), alpha=0.2, color='gray', label='±1%')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('Pressure (Pa)')
        ax.set_title(f'Pressure (err_p={err_p:.3e})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ax.plot(x_cells, u_final, 'g-', linewidth=2, label='Final u')
        ax.axhline(u0, color='r', linestyle='--', alpha=0.7, label=f'u₀')
        ax.fill_between(x_cells, u0-1e-2, u0+1e-2, alpha=0.2, color='gray', label='±0.01')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('Velocity (m/s)')
        ax.set_title(f'Velocity (err_u={err_u:.3e})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        ax.plot(x_cells, a1_final, 'b-', linewidth=2, marker='o', label='Final α₁')
        ax.fill_between([0.4, 0.6], -0.1, 1.1, alpha=0.2, color='cyan', label='Water region')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('Volume Fraction')
        ax.set_title(f'Volume Fraction (min={a1_min:.3e}, max={a1_max:.3e})')
        ax.set_ylim([-0.1, 1.1])
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        ax.plot(x_cells, rho1_final, 'b-', linewidth=2, marker='s', label='ρ₁ (water)')
        ax.plot(x_cells, rho2_final, 'g-', linewidth=2, marker='^', label='ρ₂ (air)')
        ax.axhline(rho1_init, color='b', linestyle='--', alpha=0.5)
        ax.axhline(rho2_init, color='g', linestyle='--', alpha=0.5)
        ax.fill_between([0.4, 0.6], 0, 1000, alpha=0.2, color='cyan')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('Density (kg/m³)')
        ax.set_title(f'Phase Densities (ΔE/E={dE_rel:.3e})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        png_path = '/home/younglin90/work/claude_code/claudeCFD/results/cat_A_exact/02A_abgrall_nasg_1s.png'
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {png_path}")
        plt.close()

        return {
            'pass': overall_pass,
            'err_p': err_p,
            'err_u': err_u,
            'dE_rel': dE_rel,
            'time_final': time_final,
            't_end': t_end,
            't_reached': t_reached,
        }

    except Exception as ex:
        print(f"\n❌ Solver failed with exception:")
        print(f"  {type(ex).__name__}: {ex}")
        import traceback
        traceback.print_exc()
        return {'pass': False, 'error': str(ex)}

if __name__ == '__main__':
    result = run_test_a_1s()
    if result.get('pass', False):
        print("\n✓ Test A PASSED")
        sys.exit(0)
    else:
        print("\n✗ Test A FAILED")
        sys.exit(1)
