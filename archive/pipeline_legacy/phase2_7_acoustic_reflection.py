"""Phase 2-7: Acoustic Reflection/Transmission at Air-Water Interface
Simplified — initial pulse instead of time-varying inlet BC.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from solver.He2024.explicit_mmacm_ex import solve_IMEX, cons_to_prim

def run_acoustic(alpha_scheme='tvd', label='TVD'):
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
    ph2 = {'gamma': 4.4, 'pinf': 6e8, 'kv': 474.2, 'b': 0.0, 'eta': 0.0, 'q': 0.0}

    N = 200; L = 1.0; dx = L / N
    x = np.linspace(dx/2, L - dx/2, N)
    x_interface = 0.5

    a_air = np.where(x < x_interface, 1.0 - 1e-6, 1e-6)
    a_water = 1.0 - a_air

    p0 = 1e5; T0 = 300.0
    rho1_bg = p0 / ((ph1['gamma'] - 1.0) * ph1['kv'] * T0)
    rho2_bg = (p0 + ph2['pinf']) / ((ph2['gamma'] - 1.0) * ph2['kv'] * T0)
    c_air = np.sqrt(ph1['gamma'] * p0 / rho1_bg)
    c_water = np.sqrt(ph2['gamma'] * (p0 + ph2['pinf']) / rho2_bg)
    Z_air = rho1_bg * c_air
    Z_water = rho2_bg * c_water

    if label == 'TVD':
        print(f"c_air={c_air:.1f}, c_water={c_water:.1f}")
        print(f"Z_air={Z_air:.1f}, Z_water={Z_water:.2e}, ratio={Z_water/Z_air:.0f}x")
        T_th = 2 * Z_water / (Z_water + Z_air)
        R_th = (Z_water - Z_air) / (Z_water + Z_air)
        print(f"Theory: T={T_th:.4f}, R={R_th:.6f}")

    # Initial Gaussian velocity pulse in air
    x_pulse = 0.2; sigma_pulse = 0.05
    du = 0.05
    u_init = du * np.exp(-((x - x_pulse)/sigma_pulse)**2)
    u_init[x > x_interface] = 0.0
    dp = rho1_bg * c_air * u_init
    p_init = p0 + np.where(x < x_interface, dp, 0.0)

    rho1 = p_init / ((ph1['gamma'] - 1.0) * ph1['kv'] * T0)
    rho2 = (p_init + ph2['pinf']) / ((ph2['gamma'] - 1.0) * ph2['kv'] * T0)
    a1r1 = a_air * rho1; a2r2 = a_water * rho2
    rho = a1r1 + a2r2; ru = rho * u_init
    gm1, gm2 = 0.4, 3.4
    rho_e = a_air * (p_init + ph1['gamma'] * ph1['pinf']) / gm1 + a_water * (p_init + ph2['gamma'] * ph2['pinf']) / gm2
    rE = rho_e + 0.5 * rho * u_init**2

    # Pulse travels 0.3 at c_air ~347 -> reaches interface in ~8.6e-4
    # After reflection, propagates back — check at t=1.5e-3
    t_end = 1.5e-3

    t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
        ph1, ph2, a1r1, a2r2, ru, rE, a_air,
        dx, t_end=t_end, cfl=0.4, bc_l='transmissive', bc_r='transmissive',
        max_steps=100000, print_interval=1000,
        alpha_scheme=alpha_scheme, use_strang=True,
        use_defect_correction=False, use_material_cfl=False)

    p_f, u_f, _, _, _, _, _, _ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)
    dp_final = p_f - p0

    # Amplitude measurements
    dp_incid = dp.max()
    air_mask = (x > 0.1) & (x < 0.45)
    water_mask = x > 0.55
    dp_refl = np.abs(dp_final[air_mask]).max()
    dp_trans = np.abs(dp_final[water_mask]).max()
    T_meas = dp_trans / dp_incid
    R_meas = dp_refl / dp_incid

    completed = t >= t_end * 0.99
    p_finite = np.all(np.isfinite(p_f)) and np.all(np.isfinite(u_f))
    strong_reflection = R_meas > 0.5
    near_total_refl = R_meas > 0.85  # air-water: near-total reflection
    trans_doubled = T_meas > 1.5  # pressure doubles in water (theory: 2x)

    passed = completed and p_finite and strong_reflection
    status = "PASS" if passed else "FAIL"

    print(f"{label:12s}: dp_in={dp_incid:.1f}, dp_refl={dp_refl:.1f}, dp_trans={dp_trans:.1f}")
    print(f"             T={T_meas:.3f} (~2), R={R_meas:.3f} (~1) -> {status}")
    return x, p_f, u_f, a1_f, passed, T_meas, R_meas, dp

if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    schemes = [('tvd', 'TVD'), ('thinc_bvd', 'THINC-BVD'),
               ('cicsam', 'CICSAM'), ('mstacs', 'MSTACS')]
    results = {}
    for s, l in schemes:
        results[s] = run_acoustic(s, l)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = {'tvd': 'blue', 'thinc_bvd': 'red', 'cicsam': 'green', 'mstacs': 'purple'}
    p0 = 1e5
    for s, l in schemes:
        x, p, u, a1, _, T, R, dp_i = results[s]
        c = colors[s]
        axes[0,0].plot(x, p - p0, color=c, label=f'{l} T={T:.2f},R={R:.2f}')
        axes[0,1].plot(x, u, color=c, label=l)
        axes[1,0].plot(x, a1, color=c, label=l)
        # Initial pulse (only show once)
    # Initial pulse from TVD run
    axes[1,1].plot(x, dp_i, 'k-', lw=2, label='initial dp')
    axes[1,1].set_title('Initial incident pulse')

    for ax in axes.flat:
        ax.axvline(0.5, ls='--', color='gray', alpha=0.5)
        ax.set_xlabel('x')
    axes[0,0].set_title('dp at t=1.5e-3'); axes[0,0].legend(fontsize=8)
    axes[0,1].set_title('u at t=1.5e-3'); axes[0,1].legend()
    axes[1,0].set_title('alpha_air'); axes[1,0].legend()
    axes[1,1].legend()
    plt.suptitle('Phase 2-7: Acoustic Reflection at Air-Water Interface', fontsize=14)
    plt.tight_layout()
    plt.savefig('results/phase2_7_acoustic_reflection.png', dpi=150)
    print("Plot saved: results/phase2_7_acoustic_reflection.png")

    print("="*50); print("SUMMARY")
    for s, l in schemes:
        p = results[s][4]
        print(f"  {l:12s}: {'PASS' if p else 'FAIL'}")
