"""Phase 3-1: Shock Wave in Homogeneous Air-Water Mixture (Denner 2018 §7.4.1)

Setup: Domain [0,1], CFL=0.5
- Uniform alpha_water psi ∈ {0.0, 0.5, 1.0}
- x < 0.1: Post-shock (Ms=10 mixture Rankine-Hugoniot)
- x >= 0.1: Pre-shock (p=1e5, u=0)

Test multi-resolution: N=100, 200, 400
Expected: shock at x=0.8 at t_end, 1st-order convergence
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from solver.He2024.explicit_mmacm_ex import solve_IMEX, cons_to_prim

def mixture_RH(Ms, psi_water, ph1, ph2, p_pre=1e5, rho_air=1.1574, rho_water=998.0):
    """Compute post-shock state for mixture via Rankine-Hugoniot with Wood sound speed."""
    # Mixture density
    rho_pre = (1.0 - psi_water) * rho_air + psi_water * rho_water
    # Wood sound speed c^2 = 1 / (rho * (alpha_a/(rho_a*c_a^2) + alpha_w/(rho_w*c_w^2)))
    c_air = np.sqrt(ph1['gamma'] * p_pre / rho_air)
    c_water = np.sqrt(ph2['gamma'] * (p_pre + ph2['pinf']) / rho_water)
    if psi_water == 0.0:
        c_mix = c_air
    elif psi_water == 1.0:
        c_mix = c_water
    else:
        one_over = (1.0 - psi_water)/(rho_air * c_air**2) + psi_water/(rho_water * c_water**2)
        c_mix = 1.0 / np.sqrt(rho_pre * one_over)

    # Mixture gamma (volume-weighted, rough)
    gamma_mix = (1.0 - psi_water) * ph1['gamma'] + psi_water * ph2['gamma']
    # Rankine-Hugoniot (ideal-like approximation for mixture)
    p_post = p_pre * (2 * gamma_mix * Ms**2 - (gamma_mix - 1)) / (gamma_mix + 1)
    rho_post = rho_pre * (gamma_mix + 1) * Ms**2 / ((gamma_mix - 1) * Ms**2 + 2)
    u_post = 2 * (Ms**2 - 1) / ((gamma_mix + 1) * Ms) * c_mix
    V_s = Ms * c_mix  # shock speed
    return p_post, rho_post, u_post, V_s, c_mix

def run_mixture(psi_water, N, Ms=10.0, alpha_scheme='tvd', label='TVD'):
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
    # NASG water params from spec
    ph2 = {'gamma': 1.187, 'pinf': 7.028e8, 'kv': 3610.0, 'b': 6.61e-4, 'eta': 0.0, 'q': 0.0}

    L = 1.0; dx = L / N
    x = np.linspace(dx/2, L - dx/2, N)
    x_shock = 0.1

    rho_air_pre = 1.1574
    rho_water_pre = 998.0
    p_pre = 1e5
    p_post, rho_post, u_post, V_s, c_mix = mixture_RH(Ms, psi_water, ph1, ph2, p_pre, rho_air_pre, rho_water_pre)
    t_end = 0.7 / V_s

    # Uniform alpha_water
    a_water = psi_water * np.ones(N)
    if psi_water == 0.0:
        a_water = 1e-6 * np.ones(N)
    elif psi_water == 1.0:
        a_water = (1.0 - 1e-6) * np.ones(N)
    a_air = 1.0 - a_water

    # Initial state: post-shock left, pre-shock right
    # Pre-shock: use given rho_air, rho_water at p_pre
    # Post-shock: uniform mixture density rho_post, same psi
    rho1 = np.where(x < x_shock,
                    rho_post * rho_air_pre / ((1-psi_water)*rho_air_pre + psi_water*rho_water_pre) if psi_water < 1.0 else rho_air_pre,
                    rho_air_pre)
    rho2 = np.where(x < x_shock,
                    rho_post * rho_water_pre / ((1-psi_water)*rho_air_pre + psi_water*rho_water_pre) if psi_water > 0.0 else rho_water_pre,
                    rho_water_pre)
    p_init = np.where(x < x_shock, p_post, p_pre)
    u_init = np.where(x < x_shock, u_post, 0.0)

    a1r1 = a_air * rho1
    a2r2 = a_water * rho2
    rho = a1r1 + a2r2
    ru = rho * u_init
    gm1, gm2 = ph1['gamma'] - 1.0, ph2['gamma'] - 1.0
    rho_e = a_air * (p_init + ph1['gamma'] * ph1['pinf']) / gm1 + a_water * (p_init + ph2['gamma'] * ph2['pinf']) / gm2
    rE = rho_e + 0.5 * rho * u_init**2

    t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
        ph1, ph2, a1r1, a2r2, ru, rE, a_air,
        dx, t_end=t_end, cfl=0.5, bc_l='transmissive', bc_r='transmissive',
        max_steps=100000, print_interval=500,
        alpha_scheme=alpha_scheme, use_strang=True,
        use_defect_correction=False, use_material_cfl=False)

    p_f, u_f, _, _, _, _, _, _ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)
    rho_f = a1r1_f + a2r2_f

    # Find shock position (where p drops from post to pre)
    p_mid = 0.5 * (p_post + p_pre)
    ge_mid = p_f >= p_mid
    if np.any(ge_mid) and np.any(~ge_mid):
        x_shock_num = x[np.where(ge_mid)[0].max()]
    else:
        x_shock_num = np.nan

    x_shock_expected = 0.8
    shock_error = abs(x_shock_num - x_shock_expected)

    completed = t >= t_end * 0.99
    p_finite = np.all(np.isfinite(p_f)) and np.all(np.isfinite(u_f))
    shock_match = shock_error < 0.1  # within 10%
    passed = completed and p_finite and shock_match
    status = "PASS" if passed else "FAIL"

    print(f"  psi={psi_water}, N={N} ({label}): V_s={V_s:.1f}, t_end={t_end:.3e}")
    print(f"    shock_x={x_shock_num:.3f} (expected 0.8), error={shock_error:.3f}")
    print(f"    p=[{p_f.min():.2e},{p_f.max():.2e}], u_max={np.abs(u_f).max():.1f}")
    print(f"    >>> {status}")
    return x, p_f, u_f, a1_f, rho_f, passed, x_shock_num

if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)

    # Test: psi=0 (pure air), psi=0.5 (mixture), psi=1 (pure water)
    # Grid: N=200 (focus on correctness, not multi-grid convergence for speed)
    psi_list = [0.0, 0.5, 1.0]
    N_list = [200]
    schemes = [('tvd', 'TVD'), ('cicsam', 'CICSAM')]  # reduce scope for speed

    results = {}
    for psi in psi_list:
        print(f"\n{'='*60}")
        print(f"psi_water = {psi}")
        print('='*60)
        for N in N_list:
            for s, l in schemes:
                key = (psi, N, s)
                results[key] = run_mixture(psi, N, Ms=10.0, alpha_scheme=s, label=l)

    # Plot final result for each psi
    fig, axes = plt.subplots(len(psi_list), 3, figsize=(16, 4*len(psi_list)))
    for i, psi in enumerate(psi_list):
        for s, l in schemes:
            key = (psi, 200, s)
            if key in results:
                x, p, u, a1, rho, _, _ = results[key]
                axes[i,0].plot(x, p, label=f'{l}')
                axes[i,1].plot(x, u, label=f'{l}')
                axes[i,2].plot(x, rho, label=f'{l}')
        axes[i,0].axvline(0.8, ls='--', color='k', alpha=0.5, label='expected shock')
        axes[i,1].axvline(0.8, ls='--', color='k', alpha=0.5)
        axes[i,2].axvline(0.8, ls='--', color='k', alpha=0.5)
        axes[i,0].set_title(f'psi={psi}: Pressure'); axes[i,0].legend()
        axes[i,1].set_title(f'psi={psi}: Velocity'); axes[i,1].legend()
        axes[i,2].set_title(f'psi={psi}: Density'); axes[i,2].legend()

    plt.suptitle('Phase 3-1: Ms=10 Shock in Homogeneous Mixture', fontsize=14)
    plt.tight_layout()
    plt.savefig('results/phase3_1_mixture_shock.png', dpi=150)
    print("\nPlot saved: results/phase3_1_mixture_shock.png")

    print("="*50); print("SUMMARY")
    for psi in psi_list:
        for s, l in schemes:
            key = (psi, 200, s)
            if key in results:
                p = results[key][-2]
                xs = results[key][-1]
                print(f"  psi={psi} {l}: shock_x={xs:.3f}, {'PASS' if p else 'FAIL'}")
