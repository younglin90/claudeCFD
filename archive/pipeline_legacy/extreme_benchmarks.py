"""Extreme Benchmarks validation — all 4 cases with exact Riemann comparison.

Cases:
  EB1. Shyue 2006 Gas-Water (pressure 1e9 vs 1e5, ρ 1:1000)
  EB2. Saurel-Abgrall 1999 Epoxy-Spinel analogue (SG extreme)
  EB3. Chang-Liou Underwater Explosion style
  EB4. Low-Mach pressure wave in water (Mach ~0.01, acoustic regime)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from solver.He2024.explicit_mmacm_ex import solve_IMEX, cons_to_prim
from pipeline.exact_riemann import exact_profile, exact_riemann_star


def l1_err(y_n, y_e):
    s = np.sum(np.abs(y_e))
    if s < 1e-30: return 0.0
    return np.sum(np.abs(y_n - y_e)) / s


# =========================================================
# EB1: Shyue 2006 — Gas (1 GPa) vs Water (1 atm), ρ 1:1000
# =========================================================
def eb1_shyue_gas_water():
    print("\n" + "="*70)
    print("EB1: Shyue 2006 Gas-Water Shock Tube (p_L=1e9, p_R=1e5)")
    print("="*70)
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
    ph2 = {'gamma': 7.15, 'pinf': 3.31e8, 'kv': 474.2, 'b': 0.0, 'eta': 0.0, 'q': 0.0}

    N = 400; L = 1.0; dx = L / N
    x = np.linspace(dx/2, L-dx/2, N)
    x0 = 0.5; t_end = 1.0e-4  # reduced so shock stays in domain
    # Shyue 2006 Test 2 adjusted to our framework
    p_L = 1e9; p_R = 1e5
    rho_L_gas = 1.0   # air
    rho_R_water = 1000.0

    p_star_e, u_star_e = exact_riemann_star(
        p_L, rho_L_gas, 0.0, 1.4, 0.0,
        p_R, rho_R_water, 0.0, 7.15, 3.31e8)
    print(f"  Exact: p*={p_star_e:.3e}, u*={u_star_e:.3f}")

    rho_e, u_e, p_e, _ = exact_profile(
        x, t_end, x0,
        pL=p_L, rhoL=rho_L_gas, uL=0.0, gammaL=1.4, pinfL=0.0,
        pR=p_R, rhoR=rho_R_water, uR=0.0, gammaR=7.15, pinfR=3.31e8)

    a_air = np.where(x < x0, 1.0-1e-6, 1e-6)
    p_init = np.where(x < x0, p_L, p_R)
    rho1 = np.where(x < x0, rho_L_gas, rho_L_gas)
    rho2 = np.where(x < x0, rho_R_water, rho_R_water)
    a1r1 = a_air * rho1; a2r2 = (1-a_air) * rho2
    rho = a1r1 + a2r2
    rho_e0 = a_air * p_init / 0.4 + (1-a_air) * (p_init + 7.15*3.31e8) / (7.15-1.0)
    rE = rho_e0

    t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
        ph1, ph2, a1r1, a2r2, np.zeros(N), rE, a_air,
        dx, t_end=t_end, cfl=0.25, bc_l='transmissive', bc_r='transmissive',
        max_steps=100000, print_interval=10000,
        alpha_scheme='tvd', use_strang=True,
        use_defect_correction=False, use_material_cfl=False)
    p_n, u_n, _, _, _, _, _, _ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)
    rho_n = a1r1_f + a2r2_f
    u_max_n = np.abs(u_n).max()
    err_u_star = abs(u_max_n - u_star_e) / max(abs(u_star_e), 1e-30) * 100
    passed = t >= t_end * 0.99 and np.all(np.isfinite(p_n)) and err_u_star < 10
    print(f"  Numerical: u_max={u_max_n:.2f}, err u*={err_u_star:.2f}%")
    print(f"  L1: p={l1_err(p_n, p_e)*100:.2f}%, u={l1_err(u_n, u_e)*100:.2f}%, rho={l1_err(rho_n, rho_e)*100:.2f}%")
    print(f"  >>> {'PASS' if passed else 'FAIL'}")
    return x, p_n, u_n, rho_n, p_e, u_e, rho_e, p_star_e, u_star_e, passed


# =========================================================
# EB2: Saurel-Abgrall 1999 analogue — Extreme SG (Pinf = 6e9)
# =========================================================
def eb2_strong_sg():
    print("\n" + "="*70)
    print("EB2: Strong SG Shock Tube (analogue: Pinf 10x increase)")
    print("="*70)
    # Use much stiffer water EOS to stress-test solver
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
    ph2 = {'gamma': 4.4, 'pinf': 6e9, 'kv': 474.2, 'b': 0.0, 'eta': 0.0, 'q': 0.0}

    N = 400; L = 1.0; dx = L / N
    x = np.linspace(dx/2, L-dx/2, N)
    x0 = 0.5; t_end = 5e-5
    p_L = 1e10; p_R = 1e6  # Very high pressure ratio 10000
    rho_L_gas = 10.0  # compressed gas
    rho_R_water = (p_R + 6e9) / (3.4 * 474.2 * 293.0)

    p_star_e, u_star_e = exact_riemann_star(
        p_L, rho_L_gas, 0.0, 1.4, 0.0,
        p_R, rho_R_water, 0.0, 4.4, 6e9)
    print(f"  Exact: p*={p_star_e:.3e}, u*={u_star_e:.3f}")

    rho_e, u_e, p_e, _ = exact_profile(
        x, t_end, x0,
        pL=p_L, rhoL=rho_L_gas, uL=0.0, gammaL=1.4, pinfL=0.0,
        pR=p_R, rhoR=rho_R_water, uR=0.0, gammaR=4.4, pinfR=6e9)

    a_air = np.where(x < x0, 1.0-1e-6, 1e-6)
    p_init = np.where(x < x0, p_L, p_R)
    rho1 = np.where(x < x0, rho_L_gas, 1.0)
    rho2 = np.where(x < x0, 3000.0, rho_R_water)
    a1r1 = a_air * rho1; a2r2 = (1-a_air) * rho2
    rho = a1r1 + a2r2
    rho_e0 = a_air * p_init / 0.4 + (1-a_air) * (p_init + 4.4*6e9) / 3.4
    rE = rho_e0

    t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
        ph1, ph2, a1r1, a2r2, np.zeros(N), rE, a_air,
        dx, t_end=t_end, cfl=0.25, bc_l='transmissive', bc_r='transmissive',
        max_steps=100000, print_interval=10000,
        alpha_scheme='tvd', use_strang=True,
        use_defect_correction=False, use_material_cfl=False)
    p_n, u_n, _, _, _, _, _, _ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)
    rho_n = a1r1_f + a2r2_f
    u_max_n = np.abs(u_n).max()
    err_u_star = abs(u_max_n - u_star_e) / max(abs(u_star_e), 1e-30) * 100
    passed = t >= t_end * 0.99 and np.all(np.isfinite(p_n)) and err_u_star < 20
    print(f"  Numerical: u_max={u_max_n:.2f}, err u*={err_u_star:.2f}%")
    print(f"  L1: p={l1_err(p_n, p_e)*100:.2f}%, rho={l1_err(rho_n, rho_e)*100:.2f}%")
    print(f"  >>> {'PASS' if passed else 'FAIL'}")
    return x, p_n, u_n, rho_n, p_e, u_e, rho_e, p_star_e, u_star_e, passed


# =========================================================
# EB3: Chang-Liou UNDEX analogue — detonated gas into water
# =========================================================
def eb3_undex():
    print("\n" + "="*70)
    print("EB3: Underwater Explosion analogue (detonation gas → water)")
    print("="*70)
    # Following typical UNDEX parameters: JWL-like gas mocked as ideal gas with γ=2
    ph1 = {'gamma': 2.0, 'pinf': 0.0, 'kv': 717.5, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
    ph2 = {'gamma': 4.4, 'pinf': 6e8, 'kv': 474.2, 'b': 0.0, 'eta': 0.0, 'q': 0.0}

    N = 400; L = 1.0; dx = L / N
    x = np.linspace(dx/2, L-dx/2, N)
    x0 = 0.4; t_end = 1e-4
    p_L = 1e9  # detonated gas
    p_R = 1e5  # water at atm
    rho_L = 1250.0  # explosion gas density
    rho_R = 1000.0  # water

    p_star_e, u_star_e = exact_riemann_star(
        p_L, rho_L, 0.0, 2.0, 0.0,
        p_R, rho_R, 0.0, 4.4, 6e8)
    print(f"  Exact: p*={p_star_e:.3e}, u*={u_star_e:.3f}")

    rho_e, u_e, p_e, _ = exact_profile(
        x, t_end, x0,
        pL=p_L, rhoL=rho_L, uL=0.0, gammaL=2.0, pinfL=0.0,
        pR=p_R, rhoR=rho_R, uR=0.0, gammaR=4.4, pinfR=6e8)

    a_air = np.where(x < x0, 1.0-1e-6, 1e-6)
    p_init = np.where(x < x0, p_L, p_R)
    rho1 = np.where(x < x0, rho_L, 1.0)
    rho2 = np.where(x < x0, rho_R, rho_R)
    a1r1 = a_air * rho1; a2r2 = (1-a_air) * rho2
    rho = a1r1 + a2r2
    rho_e0 = a_air * p_init / (2.0-1.0) + (1-a_air) * (p_init + 4.4*6e8) / 3.4
    rE = rho_e0

    t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
        ph1, ph2, a1r1, a2r2, np.zeros(N), rE, a_air,
        dx, t_end=t_end, cfl=0.25, bc_l='transmissive', bc_r='transmissive',
        max_steps=100000, print_interval=10000,
        alpha_scheme='tvd', use_strang=True,
        use_defect_correction=False, use_material_cfl=False)
    p_n, u_n, _, _, _, _, _, _ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)
    rho_n = a1r1_f + a2r2_f
    u_max_n = np.abs(u_n).max()
    err_u_star = abs(u_max_n - u_star_e) / max(abs(u_star_e), 1e-30) * 100
    passed = t >= t_end * 0.99 and np.all(np.isfinite(p_n)) and err_u_star < 15
    print(f"  Numerical: u_max={u_max_n:.2f}, err u*={err_u_star:.2f}%")
    print(f"  L1: p={l1_err(p_n, p_e)*100:.2f}%, rho={l1_err(rho_n, rho_e)*100:.2f}%")
    print(f"  >>> {'PASS' if passed else 'FAIL'}")
    return x, p_n, u_n, rho_n, p_e, u_e, rho_e, p_star_e, u_star_e, passed


# =========================================================
# EB4: Low-Mach Pressure Wave in Liquid (all-Mach검증)
# =========================================================
def eb4_low_mach_liquid():
    print("\n" + "="*70)
    print("EB4: Low-Mach Pressure Wave in Water (Mach~0.01, all-Mach regime)")
    print("="*70)
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
    ph2 = {'gamma': 4.4, 'pinf': 6e8, 'kv': 474.2, 'b': 0.0, 'eta': 0.0, 'q': 0.0}

    N = 200; L = 1.0; dx = L / N
    x = np.linspace(dx/2, L-dx/2, N)
    x0 = 0.5; t_end = 3e-4
    p0 = 1e5
    dp = 1.01 * p0  # 1% perturbation (low Mach)
    # Left: slightly higher pressure, Right: atm
    p_L = p0 + 0.01 * p0
    p_R = p0
    rho_L = (p_L + 6e8) / (3.4 * 474.2 * 293.0)
    rho_R = (p_R + 6e8) / (3.4 * 474.2 * 293.0)

    p_star_e, u_star_e = exact_riemann_star(
        p_L, rho_L, 0.0, 4.4, 6e8,
        p_R, rho_R, 0.0, 4.4, 6e8)
    print(f"  Exact: p*={p_star_e:.3e}, u*={u_star_e:.4f} (low-Mach)")
    c_water = np.sqrt(4.4 * (p0 + 6e8) / rho_R)
    print(f"  c_water={c_water:.0f} m/s → Mach={abs(u_star_e)/c_water:.1e}")

    rho_e, u_e, p_e, _ = exact_profile(
        x, t_end, x0,
        pL=p_L, rhoL=rho_L, uL=0.0, gammaL=4.4, pinfL=6e8,
        pR=p_R, rhoR=rho_R, uR=0.0, gammaR=4.4, pinfR=6e8)

    a_air = 1e-6 * np.ones(N)
    p_init = np.where(x < x0, p_L, p_R)
    rho1 = p_init / (0.4 * 717.5 * 293.0)
    rho2 = (p_init + 6e8) / (3.4 * 474.2 * 293.0)
    a1r1 = a_air * rho1; a2r2 = (1-a_air) * rho2
    rho = a1r1 + a2r2
    rho_e0 = a_air * p_init / 0.4 + (1-a_air) * (p_init + 4.4*6e8) / 3.4
    rE = rho_e0

    # Use material CFL for low-Mach efficiency
    t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
        ph1, ph2, a1r1, a2r2, np.zeros(N), rE, a_air,
        dx, t_end=t_end, cfl=0.4, bc_l='transmissive', bc_r='transmissive',
        max_steps=100000, print_interval=10000,
        alpha_scheme='tvd', use_strang=True,
        use_defect_correction=False, use_material_cfl=False)
    p_n, u_n, _, _, _, _, _, _ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)
    rho_n = a1r1_f + a2r2_f
    u_max_n = np.abs(u_n).max()
    err_u_star = abs(u_max_n - abs(u_star_e)) / max(abs(u_star_e), 1e-30) * 100
    passed = t >= t_end * 0.99 and np.all(np.isfinite(p_n)) and err_u_star < 30
    print(f"  Numerical: u_max={u_max_n:.4f}, err u*={err_u_star:.2f}%")
    print(f"  L1: p={l1_err(p_n, p_e)*100:.3f}%, rho={l1_err(rho_n, rho_e)*100:.3f}%")
    print(f"  >>> {'PASS' if passed else 'FAIL'}")
    return x, p_n, u_n, rho_n, p_e, u_e, rho_e, p_star_e, u_star_e, passed


# ===========================================================
if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)

    results = {
        'EB1 Shyue Gas-Water': eb1_shyue_gas_water(),
        'EB2 Strong SG (Pinf=6e9)': eb2_strong_sg(),
        'EB3 UNDEX analogue': eb3_undex(),
        'EB4 Low-Mach Water': eb4_low_mach_liquid(),
    }

    # Plot
    fig, axes = plt.subplots(4, 3, figsize=(18, 18))
    for i, (title, r) in enumerate(results.items()):
        x, p_n, u_n, rho_n, p_e, u_e, rho_e, p_s_e, u_s_e, passed = r
        axes[i,0].plot(x, p_e, 'k-', lw=2, label='Exact')
        axes[i,0].plot(x, p_n, 'b.-', markersize=3, label=f'IMEX ({"PASS" if passed else "FAIL"})')
        axes[i,0].set_title(f'{title}: p'); axes[i,0].legend(fontsize=8)
        axes[i,0].set_xlabel('x')

        axes[i,1].plot(x, u_e, 'k-', lw=2, label='Exact')
        axes[i,1].plot(x, u_n, 'b.-', markersize=3, label='IMEX')
        axes[i,1].set_title(f'u (u*={u_s_e:.2f})'); axes[i,1].legend(fontsize=8)
        axes[i,1].set_xlabel('x')

        axes[i,2].plot(x, rho_e, 'k-', lw=2, label='Exact')
        axes[i,2].plot(x, rho_n, 'b.-', markersize=3, label='IMEX')
        axes[i,2].set_title(f'rho'); axes[i,2].legend(fontsize=8)
        axes[i,2].set_xlabel('x')

    plt.suptitle('Extreme Benchmarks: IMEX vs Exact Riemann', fontsize=14)
    plt.tight_layout()
    plt.savefig('results/extreme_benchmarks.png', dpi=150)
    print(f"\nPlot saved: results/extreme_benchmarks.png")

    print("\n" + "="*70)
    print("EXTREME BENCHMARK SUMMARY")
    print("="*70)
    for title, r in results.items():
        passed = r[-1]
        print(f"  {title:30s}: {'PASS' if passed else 'FAIL'}")
