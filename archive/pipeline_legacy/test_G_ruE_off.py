"""Test: MMACM-Ex with G_ru=0, G_rE=0 (mass/alpha G only).
Compare TVD vs THINC-BVD with mmacm_G_ruE=False.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from solver.He2024.explicit_mmacm_ex import solve_IMEX, cons_to_prim

# ===== Phase 1: Abgrall advection =====
def run_phase1():
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
    ph2 = {'gamma': 2.35, 'pinf': 1e9, 'kv': 943.8, 'b': 6.61e-4, 'eta': -1167e3, 'q': 0.0}
    N = 10; L = 1.0; dx = L / N
    x = np.linspace(dx/2, L - dx/2, N)
    p0 = 1e5; u0 = 1.0; T0 = 300.0
    a1 = np.where((x >= 0.4) & (x <= 0.6), 1e-6, 1.0 - 1e-6)
    a2 = 1.0 - a1
    rho1 = p0 / (ph1['gamma'] - 1.0) / ph1['kv'] / T0
    rho2 = (p0 + ph2['pinf']) / (ph2['gamma'] - 1.0) / ph2['kv'] / T0
    a1r1 = a1 * rho1 * np.ones(N)
    a2r2 = a2 * rho2 * np.ones(N)
    rho = a1r1 + a2r2
    ru = rho * u0
    gm1, gm2 = ph1['gamma'] - 1.0, ph2['gamma'] - 1.0
    rho_e = a1 * (p0 + ph1['gamma'] * ph1['pinf']) / gm1 + a2 * (p0 + ph2['gamma'] * ph2['pinf']) / gm2
    rE = rho_e + 0.5 * rho * u0**2

    print("="*60)
    print("Phase 1: Abgrall advection (G_ruE=False, THINC-BVD)")
    print("="*60)
    t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
        ph1, ph2, a1r1, a2r2, ru, rE, a1,
        dx, t_end=1.0, cfl=0.4, bc_l='periodic', bc_r='periodic',
        max_steps=100, print_interval=20,
        alpha_scheme='thinc_bvd', use_strang=True,
        use_defect_correction=True, use_material_cfl=True,
        mmacm_G_ruE=False)

    p_f, u_f, _, _, _, _, _, _ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)
    err_p = np.max(np.abs(p_f / p0 - 1.0))
    err_u = np.max(np.abs(u_f / u0 - 1.0))
    print(f"\nPhase 1 Result: err_p={err_p:.2e}, err_u={err_u:.2e}")
    if err_p < 1e-2 and err_u < 1e-2:
        print(">>> Phase 1: PASS")
    else:
        print(">>> Phase 1: FAIL")
    return err_p, err_u

# ===== Phase 2-1: HP Air / LP Water =====
def run_phase2_1(alpha_scheme='tvd', label='TVD'):
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
    ph2 = {'gamma': 4.1, 'pinf': 4.4e8, 'kv': 474.2, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
    N = 200; L = 2.0; dx = L / N
    x = np.linspace(dx/2, L - dx/2, N)
    xd = 0.5
    p_L, p_R = 1e9, 1e4
    rho1_L = p_L / ((ph1['gamma'] - 1.0) * ph1['kv'] * 300.0)
    rho1_R = p_R / ((ph1['gamma'] - 1.0) * ph1['kv'] * 300.0)
    rho2_L = (p_L + ph2['pinf']) / ((ph2['gamma'] - 1.0) * ph2['kv'] * 300.0)
    rho2_R = (p_R + ph2['pinf']) / ((ph2['gamma'] - 1.0) * ph2['kv'] * 300.0)

    a_air = np.where(x < xd, 1.0 - 1e-6, 1e-6)
    a_water = 1.0 - a_air
    p_init = np.where(x < xd, p_L, p_R)
    rho1 = np.where(x < xd, rho1_L, rho1_R)
    rho2 = np.where(x < xd, rho2_L, rho2_R)

    a1r1 = a_air * rho1
    a2r2 = a_water * rho2
    rho = a1r1 + a2r2
    ru = np.zeros(N)
    gm1, gm2 = ph1['gamma'] - 1.0, ph2['gamma'] - 1.0
    rho_e = a_air * (p_init + ph1['gamma'] * ph1['pinf']) / gm1 + a_water * (p_init + ph2['gamma'] * ph2['pinf']) / gm2
    rE = rho_e + 0.0

    print(f"\n{'='*60}")
    print(f"Phase 2-1: HP Air / LP Water ({label}, G_ruE=False)")
    print(f"{'='*60}")
    t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
        ph1, ph2, a1r1, a2r2, ru, rE, a_air,
        dx, t_end=8e-4, cfl=0.4, bc_l='transmissive', bc_r='transmissive',
        max_steps=100000, print_interval=100,
        alpha_scheme=alpha_scheme, use_strang=True,
        use_defect_correction=False, use_material_cfl=False,
        mmacm_G_ruE=False)

    p_f, u_f, _, _, _, _, _, _ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)
    u_max = np.max(np.abs(u_f))
    print(f"\nPhase 2-1 ({label}): u_max={u_max:.1f} (ref ~228)")
    if t >= 8e-4 * 0.99 and u_max < 400:
        print(f">>> Phase 2-1 ({label}): PASS")
    else:
        print(f">>> Phase 2-1 ({label}): FAIL")
    return x, p_f, u_f, a1_f, a1r1_f, a2r2_f, rE_f, u_max

# ===== Phase 2-2: HP Water / LP Air (Yoo & Sung 2018) =====
def run_phase2_2(alpha_scheme='tvd', label='TVD'):
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
    ph2 = {'gamma': 4.4, 'pinf': 6e8, 'kv': 474.2, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
    N = 200; L = 1.0; dx = L / N
    x = np.linspace(dx/2, L - dx/2, N)
    xd = 0.7

    a_air = np.where(x < xd, 1e-6, 1.0 - 1e-6)
    a_water = 1.0 - a_air
    p_init = np.where(x < xd, 1e9, 1e5)
    rho1 = 50.0 * np.ones(N)
    rho2 = 1000.0 * np.ones(N)

    a1r1 = a_air * rho1
    a2r2 = a_water * rho2
    rho = a1r1 + a2r2
    ru = np.zeros(N)
    gm1, gm2 = ph1['gamma'] - 1.0, ph2['gamma'] - 1.0
    rho_e = a_air * (p_init + ph1['gamma'] * ph1['pinf']) / gm1 + a_water * (p_init + ph2['gamma'] * ph2['pinf']) / gm2
    rE = rho_e + 0.0

    print(f"\n{'='*60}")
    print(f"Phase 2-2: HP Water / LP Air ({label}, G_ruE=False)")
    print(f"{'='*60}")
    t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
        ph1, ph2, a1r1, a2r2, ru, rE, a_air,
        dx, t_end=2.29e-4, cfl=0.25, bc_l='transmissive', bc_r='transmissive',
        max_steps=100000, print_interval=100,
        alpha_scheme=alpha_scheme, use_strang=True,
        use_defect_correction=False, use_material_cfl=False,
        mmacm_G_ruE=False)

    p_f, u_f, _, _, _, _, _, _ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)
    u_max = np.max(np.abs(u_f))
    rho_f = a1r1_f + a2r2_f
    print(f"\nPhase 2-2 ({label}): u_max={u_max:.1f} (ref ~486)")
    if t >= 2.29e-4 * 0.99 and 400 < u_max < 600:
        print(f">>> Phase 2-2 ({label}): PASS")
    else:
        print(f">>> Phase 2-2 ({label}): FAIL")
    return x, p_f, u_f, a1_f, a1r1_f, a2r2_f, rE_f, rho_f, u_max

if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)

    # Phase 1
    err_p, err_u = run_phase1()

    # Phase 2-1: TVD vs THINC-BVD
    x21_t, p21_t, u21_t, a121_t, _, _, _, umax21_t = run_phase2_1('tvd', 'TVD')
    x21_b, p21_b, u21_b, a121_b, _, _, _, umax21_b = run_phase2_1('thinc_bvd', 'BVD')

    # Phase 2-2: TVD vs THINC-BVD
    x22_t, p22_t, u22_t, a122_t, _, _, _, rho22_t, umax22_t = run_phase2_2('tvd', 'TVD')
    x22_b, p22_b, u22_b, a122_b, _, _, _, rho22_b, umax22_b = run_phase2_2('thinc_bvd', 'BVD')

    # ===== Plot =====
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Phase 2-1
    axes[0, 0].plot(x21_t, p21_t, 'b-', label=f'TVD u={umax21_t:.0f}')
    axes[0, 0].plot(x21_b, p21_b, 'r--', label=f'BVD u={umax21_b:.0f}')
    axes[0, 0].set_title('Phase 2-1: Pressure'); axes[0, 0].legend()
    axes[0, 1].plot(x21_t, u21_t, 'b-'); axes[0, 1].plot(x21_b, u21_b, 'r--')
    axes[0, 1].set_title('Phase 2-1: Velocity')
    axes[0, 2].plot(x21_t, a121_t, 'b-'); axes[0, 2].plot(x21_b, a121_b, 'r--')
    axes[0, 2].set_title('Phase 2-1: α_air')
    axes[0, 3].text(0.1, 0.5, f'G_ruE=False\nPhase1: err_p={err_p:.1e}\n'
                    f'P2-1 TVD u={umax21_t:.0f}\nP2-1 BVD u={umax21_b:.0f}\n'
                    f'P2-2 TVD u={umax22_t:.0f}\nP2-2 BVD u={umax22_b:.0f}',
                    transform=axes[0, 3].transAxes, fontsize=14, va='center')
    axes[0, 3].set_title('Summary')

    # Phase 2-2
    axes[1, 0].plot(x22_t, p22_t, 'b-', label=f'TVD u={umax22_t:.0f}')
    axes[1, 0].plot(x22_b, p22_b, 'r--', label=f'BVD u={umax22_b:.0f}')
    axes[1, 0].set_title('Phase 2-2: Pressure'); axes[1, 0].legend()
    axes[1, 1].plot(x22_t, u22_t, 'b-'); axes[1, 1].plot(x22_b, u22_b, 'r--')
    axes[1, 1].set_title('Phase 2-2: Velocity')
    axes[1, 2].plot(x22_t, a122_t, 'b-'); axes[1, 2].plot(x22_b, a122_b, 'r--')
    axes[1, 2].set_title('Phase 2-2: α_air')
    axes[1, 3].plot(x22_t, rho22_t, 'b-', label='TVD')
    axes[1, 3].plot(x22_b, rho22_b, 'r--', label='BVD')
    axes[1, 3].set_title('Phase 2-2: Density'); axes[1, 3].legend()

    plt.suptitle('MMACM-Ex G_ru=0, G_rE=0 (mass/alpha G only)', fontsize=16)
    plt.tight_layout()
    plt.savefig('results/test_G_ruE_off.png', dpi=150)
    print(f"\nPlot saved: results/test_G_ruE_off.png")
