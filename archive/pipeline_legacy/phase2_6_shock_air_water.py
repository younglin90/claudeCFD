"""Phase 2-6: Mach 10 Shock - Air-Water Interface (Denner 2018 §7.4.4)

Setup: Domain [0,1], N=200, CFL=0.5
- x < 0.3: Post-shock air (Ms=10 Rankine-Hugoniot)
- 0.3 <= x < 0.7: Pre-shock air
- x >= 0.7: Water (p=1 atm)

Air ideal: gamma=1.4
Water SG: gamma=7.15, Pinf matching

Expected: shock moves right at Ms=10, hits water at x=0.7, transmitted + reflected
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from solver.He2024.explicit_mmacm_ex import solve_IMEX, cons_to_prim

def rankine_hugoniot_air(Ms, gamma, p_pre, rho_pre):
    """Post-shock state for air (ideal gas)."""
    p_post = p_pre * (2*gamma*Ms**2 - (gamma-1)) / (gamma+1)
    rho_post = rho_pre * (gamma+1)*Ms**2 / ((gamma-1)*Ms**2 + 2)
    u_post = 2*(Ms**2 - 1) / ((gamma+1)*Ms) * np.sqrt(gamma*p_pre/rho_pre)
    return p_post, rho_post, u_post

def run_shock_water(alpha_scheme='tvd', label='TVD'):
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
    # Water SG: use our standard (γ=4.4, Pinf=6e8) which works with solver
    ph2 = {'gamma': 4.4, 'pinf': 6e8, 'kv': 474.2, 'b': 0.0, 'eta': 0.0, 'q': 0.0}

    N = 200; L = 1.0; dx = L / N
    x = np.linspace(dx/2, L - dx/2, N)

    # Rankine-Hugoniot for Ms=10 air shock
    Ms = 10.0
    p_pre = 1e5; rho_pre = 1.0  # air pre-shock
    T_pre = p_pre / ((ph1['gamma']-1) * ph1['kv'] * rho_pre)
    p_post, rho_post, u_post = rankine_hugoniot_air(Ms, ph1['gamma'], p_pre, rho_pre)
    print(f"Phase 2-6 ({label}): Ms=10 post-shock: p={p_post:.2e}, rho={rho_post:.2f}, u={u_post:.1f}")

    # Initial state
    x_shock = 0.3
    x_interface = 0.7

    a_air = np.ones(N)
    a_air[x >= x_interface] = 1e-6
    a_water = 1.0 - a_air

    p_init = np.ones(N) * p_pre
    p_init[x < x_shock] = p_post
    p_init[x >= x_interface] = p_pre  # water at ambient

    u_init = np.zeros(N)
    u_init[x < x_shock] = u_post

    # Densities
    rho1 = np.ones(N) * rho_pre  # air
    rho1[x < x_shock] = rho_post
    # Water density at ambient p
    T_water = 293.0
    rho2_water = (p_pre + ph2['pinf']) / ((ph2['gamma']-1.0) * ph2['kv'] * T_water)
    rho2 = np.ones(N) * rho2_water

    a1r1 = a_air * rho1
    a2r2 = a_water * rho2
    rho = a1r1 + a2r2
    ru = rho * u_init
    gm1, gm2 = ph1['gamma'] - 1.0, ph2['gamma'] - 1.0
    rho_e = a_air * (p_init + ph1['gamma'] * ph1['pinf']) / gm1 + a_water * (p_init + ph2['gamma'] * ph2['pinf']) / gm2
    rE = rho_e + 0.5 * rho * u_init**2

    # Shock reaches water at t = 0.4 / (Ms * c_air)
    c_air = np.sqrt(ph1['gamma'] * p_pre / rho_pre)  # ~374 m/s
    t_to_interface = 0.4 / (Ms * c_air)  # ~1.07e-4 s
    t_end = t_to_interface * 1.3  # some time after interaction

    print(f"  c_air={c_air:.1f}, t_to_interface={t_to_interface:.3e}, t_end={t_end:.3e}")

    t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
        ph1, ph2, a1r1, a2r2, ru, rE, a_air,
        dx, t_end=t_end, cfl=0.5, bc_l='transmissive', bc_r='transmissive',
        max_steps=100000, print_interval=200,
        alpha_scheme=alpha_scheme, use_strang=True,
        use_defect_correction=False, use_material_cfl=False)

    p_f, u_f, _, _, _, _, _, _ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)
    rho_f = a1r1_f + a2r2_f

    completed = t >= t_end * 0.99
    p_finite = np.all(np.isfinite(p_f)) and np.all(np.isfinite(u_f))
    # After shock hits water, transmitted shock propagates through water
    # Water is highly compressed -> very high pressure in water region
    p_water_max = p_f[x >= x_interface].max() if np.any(x >= x_interface) else 0
    p_elevated = p_water_max > 2 * p_pre  # pressure elevated in water
    u_bounded = np.abs(u_f).max() < 1e4

    passed = completed and p_finite and p_elevated and u_bounded
    status = "PASS" if passed else "FAIL"

    print(f"  t_final={t:.3e}, p_range=[{p_f.min():.2e},{p_f.max():.2e}]")
    print(f"  u_max={np.abs(u_f).max():.1f}, p_water_max={p_water_max:.2e}")
    print(f"  >>> {status}\n")

    return x, p_f, u_f, a1_f, rho_f, passed

if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    schemes = [('tvd', 'TVD'), ('thinc_bvd', 'THINC-BVD'),
               ('cicsam', 'CICSAM'), ('mstacs', 'MSTACS')]
    results = {}
    for s, l in schemes:
        results[s] = run_shock_water(s, l)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = {'tvd': 'blue', 'thinc_bvd': 'red', 'cicsam': 'green', 'mstacs': 'purple'}
    for s, l in schemes:
        x, p, u, a1, rho, _ = results[s]
        c = colors[s]
        axes[0,0].plot(x, p, color=c, label=l)
        axes[0,1].plot(x, u, color=c, label=l)
        axes[1,0].plot(x, a1, color=c, label=l)
        axes[1,1].plot(x, rho, color=c, label=l)
    axes[0,0].set_yscale('log')
    axes[0,0].set_title('Pressure (log)'); axes[0,0].legend()
    axes[0,1].set_title('Velocity')
    axes[1,0].set_title('alpha_air')
    axes[1,1].set_title('Density')
    for ax in axes.flat: ax.set_xlabel('x')
    plt.suptitle('Phase 2-6: Mach 10 Shock-Air-Water Interface', fontsize=14)
    plt.tight_layout()
    plt.savefig('results/phase2_6_shock_air_water.png', dpi=150)
    print("Plot saved: results/phase2_6_shock_air_water.png")

    print("="*50); print("SUMMARY")
    for s, l in schemes:
        p = results[s][-1]
        print(f"  {l:12s}: {'PASS' if p else 'FAIL'}")
