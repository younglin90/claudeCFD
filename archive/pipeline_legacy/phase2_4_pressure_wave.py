"""Phase 2-4: Pressure Wave Propagation in Liquid (single-phase water)

Setup:
- Domain [0, 1], N=100
- Left: p = 1 GPa (10e8 Pa), Right: p = 0.1 GPa (1e8 Pa)
- T=293K, u=0 everywhere
- Single phase water only (alpha_water ≈ 1)

Expected: symmetric shock (right) + rarefaction (left)
Star region pressure: (1e9 + 1e8) / 2 = 5.5e8 Pa
PASS: t_end completion, star pressure ~5.5e8
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from solver.He2024.explicit_mmacm_ex import solve_IMEX, cons_to_prim

def run_pressure_wave(alpha_scheme='tvd', label='TVD', t_end=1.5e-4):
    # Water NASG (Denner 2018-style)
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5, 'b': 0.0, 'eta': 0.0, 'q': 0.0}    # air (dummy)
    ph2 = {'gamma': 4.4, 'pinf': 6e8, 'kv': 474.2, 'b': 0.0, 'eta': 0.0, 'q': 0.0}    # water SG

    N = 100; L = 1.0; dx = L / N
    x = np.linspace(dx/2, L - dx/2, N)
    xd = 0.5

    # Near-pure water (alpha_air = 1e-6)
    a_air = 1e-6 * np.ones(N)
    a_water = 1.0 - a_air

    p_L = 1e9; p_R = 1e8
    p_init = np.where(x < xd, p_L, p_R)
    T_init = 293.0

    rho1 = p_init / ((ph1['gamma'] - 1.0) * ph1['kv'] * T_init)  # air
    rho2 = (p_init + ph2['pinf']) / ((ph2['gamma'] - 1.0) * ph2['kv'] * T_init)  # water
    u_init = 0.0

    a1r1 = a_air * rho1
    a2r2 = a_water * rho2
    rho = a1r1 + a2r2
    ru = rho * u_init
    gm1, gm2 = ph1['gamma'] - 1.0, ph2['gamma'] - 1.0
    rho_e = a_air * (p_init + ph1['gamma'] * ph1['pinf']) / gm1 + a_water * (p_init + ph2['gamma'] * ph2['pinf']) / gm2
    rE = rho_e + 0.5 * rho * u_init**2

    print(f"Phase 2-4 Pressure Wave in Liquid ({label}):")
    print(f"  rho_water_L={rho2[0]:.1f}, rho_water_R={rho2[-1]:.1f}")

    t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
        ph1, ph2, a1r1, a2r2, ru, rE, a_air,
        dx, t_end=t_end, cfl=0.25, bc_l='transmissive', bc_r='transmissive',
        max_steps=100000, print_interval=500,
        alpha_scheme=alpha_scheme, use_strang=True,
        use_defect_correction=False, use_material_cfl=False)

    p_f, u_f, _, _, _, _, _, _ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)
    rho_f = a1r1_f + a2r2_f

    # Expected star region pressure ~ 5.5e8
    # Check: star region should exist and be ~5.5e8
    # Find region with roughly uniform p between left/right waves
    p_star_expected = 5.5e8
    center_region = (x > 0.45) & (x < 0.55)
    p_center_mean = p_f[center_region].mean()
    p_star_error = abs(p_center_mean - p_star_expected) / p_star_expected

    completed = t >= t_end * 0.99
    p_in_range = 1e8 <= p_f.min() and p_f.max() <= 1e9 * 1.1
    star_match = p_star_error < 0.15  # within 15%

    passed = completed and p_in_range and star_match
    status = "PASS" if passed else "FAIL"

    print(f"  t_final={t:.3e}")
    print(f"  p_range=[{p_f.min():.3e}, {p_f.max():.3e}]")
    print(f"  p_center (avg at x~0.5): {p_center_mean:.3e} (expected 5.5e8)")
    print(f"  u_range=[{u_f.min():.2f}, {u_f.max():.2f}]")
    print(f"  star_error={p_star_error*100:.1f}% {'<15%' if star_match else '>=15%'}")
    print(f"  >>> {status}\n")

    return x, p_f, u_f, a1_f, rho_f, passed, p_center_mean

if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)

    schemes = [('tvd', 'TVD'), ('thinc_bvd', 'THINC-BVD'),
               ('cicsam', 'CICSAM'), ('mstacs', 'MSTACS')]
    results = {}
    for s, l in schemes:
        results[s] = run_pressure_wave(s, l)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = {'tvd': 'blue', 'thinc_bvd': 'red', 'cicsam': 'green', 'mstacs': 'purple'}
    for s, l in schemes:
        x, p, u, a1, rho, _, pc = results[s]
        c = colors[s]
        axes[0,0].plot(x, p/1e8, color=c, label=f'{l} p*={pc/1e8:.2f}')
        axes[0,1].plot(x, u, color=c, label=l)
        axes[1,0].plot(x, rho, color=c, label=l)
        axes[1,1].plot(x, a1, color=c, label=l)

    axes[0,0].axhline(5.5, ls='--', color='gray', label='p*=5.5e8')
    axes[0,0].set_title('Pressure / 1e8 Pa'); axes[0,0].legend(fontsize=8); axes[0,0].set_xlabel('x')
    axes[0,1].set_title('Velocity'); axes[0,1].legend(); axes[0,1].set_xlabel('x')
    axes[1,0].set_title('Density (water)'); axes[1,0].legend(); axes[1,0].set_xlabel('x')
    axes[1,1].set_title('alpha_air'); axes[1,1].legend(); axes[1,1].set_xlabel('x')

    plt.suptitle('Phase 2-4: 1D Pressure Wave in Liquid (1 GPa / 0.1 GPa)', fontsize=14)
    plt.tight_layout()
    plt.savefig('results/phase2_4_pressure_wave.png', dpi=150)
    print("Plot saved: results/phase2_4_pressure_wave.png")

    print("="*50); print("SUMMARY")
    for s, l in schemes:
        _, _, _, _, _, p, _ = results[s]
        print(f"  {l:12s}: {'PASS' if p else 'FAIL'}")
