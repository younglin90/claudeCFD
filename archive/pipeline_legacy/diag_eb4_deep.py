"""EB4 심층 분석: 진동 원인 규명
1. 시간 진화 관찰 (t=0, dt, 2dt, ..., t_end)
2. 파장 FFT 분석
3. Smooth IC vs Heaviside IC
4. IM1 기여 vs advective 기여 분리
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from solver.He2024.explicit_mmacm_ex import solve_IMEX, cons_to_prim


def setup_eb4(N, smooth_width=0.0):
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
    ph2 = {'gamma': 4.4, 'pinf': 6e8, 'kv': 474.2, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
    L = 1.0; dx = L/N
    x = np.linspace(dx/2, L-dx/2, N)
    x0 = 0.5
    p0 = 1e5
    p_L = p0 * 1.01
    p_R = p0

    if smooth_width > 0:
        p_init = p0 + 0.005*p0*(1 - np.tanh((x - x0)/smooth_width))
    else:
        p_init = np.where(x < x0, p_L, p_R)

    rho1 = p_init / (0.4 * 717.5 * 293.0)
    rho2 = (p_init + 6e8) / (3.4 * 474.2 * 293.0)
    a_air = 1e-6 * np.ones(N)
    a1r1 = a_air * rho1; a2r2 = (1-a_air) * rho2
    rho_e0 = a_air * p_init / 0.4 + (1-a_air) * (p_init + 4.4*6e8) / 3.4
    return ph1, ph2, dx, x, a1r1, a2r2, rho_e0, a_air, p0


def run(ph1, ph2, dx, a1r1, a2r2, rho_e0, a_air, t_end, N):
    t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
        ph1, ph2, a1r1.copy(), a2r2.copy(), np.zeros(N), rho_e0.copy(), a_air.copy(),
        dx, t_end=t_end, cfl=0.4, bc_l='transmissive', bc_r='transmissive',
        max_steps=200000, print_interval=100000,
        alpha_scheme='tvd', use_strang=True,
        use_defect_correction=False, use_material_cfl=False,
        use_mmacm_ex=True)
    p_n, u_n, _, _, _, _, _, _ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)
    return p_n, u_n


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    N = 200
    p0 = 1e5

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))

    # ========= Test 1: Time evolution =========
    print("="*60)
    print("Test 1: Time evolution of oscillation")
    print("="*60)
    ph1, ph2, dx, x, a1r1, a2r2, rho_e0, a_air, _ = setup_eb4(N=200, smooth_width=0)

    times = [1e-5, 5e-5, 1e-4, 2e-4, 3e-4]
    for t_end in times:
        p_n, u_n = run(ph1, ph2, dx, a1r1, a2r2, rho_e0, a_air, t_end, N)
        p_osc = (p_n - p_n.mean()).std() / p0
        print(f"  t={t_end:.1e}: p osc std={p_osc:.3e}, p range=[{(p_n.min()-p0)/p0*100:.3f}%, {(p_n.max()-p0)/p0*100:.3f}%]")
        axes[0,0].plot(x, (p_n - p0)/p0 * 100, label=f't={t_end:.1e}')
        axes[0,1].plot(x, u_n*1000, label=f't={t_end:.1e}')
    axes[0,0].set_title('Time evolution: p perturbation (%)'); axes[0,0].legend(fontsize=8); axes[0,0].set_xlabel('x')
    axes[0,1].set_title('Time evolution: u (mm/s)'); axes[0,1].legend(fontsize=8); axes[0,1].set_xlabel('x')

    # FFT at t=3e-4
    p_3e4, _ = run(ph1, ph2, dx, a1r1, a2r2, rho_e0, a_air, 3e-4, N)
    signal = (p_3e4 - p_3e4.mean()) / p0
    freq = np.fft.fftfreq(N, d=dx)
    fft_mag = np.abs(np.fft.fft(signal))
    axes[0,2].plot(freq[:N//2], fft_mag[:N//2])
    axes[0,2].set_xlabel('Wavenumber (1/m)')
    axes[0,2].set_ylabel('|FFT(p)|')
    axes[0,2].set_title('FFT of p oscillation (t=3e-4)')
    axes[0,2].set_yscale('log')
    nyquist_k = 1.0 / (2*dx)
    axes[0,2].axvline(nyquist_k, ls='--', color='r', label=f'Nyquist={nyquist_k:.0f}')
    axes[0,2].legend()

    # ========= Test 2: Smooth vs Heaviside IC =========
    print("\nTest 2: Smooth IC vs Heaviside")
    for smooth_w, label in [(0.0, 'Heaviside'), (0.001, 'tanh_w=0.001'),
                             (0.005, 'tanh_w=0.005'), (0.02, 'tanh_w=0.02')]:
        ph1, ph2, dx, x, a1r1, a2r2, rho_e0, a_air, _ = setup_eb4(N=200, smooth_width=smooth_w)
        p_n, u_n = run(ph1, ph2, dx, a1r1, a2r2, rho_e0, a_air, 3e-4, N)
        p_osc = (p_n - p_n.mean()).std() / p0
        p_range = (p_n.max() - p_n.min()) / p0
        print(f"  {label}: p osc std={p_osc:.3e}, range={p_range*100:.4f}%")
        axes[1,0].plot(x, (p_n - p0)/p0 * 100, label=label)
        axes[1,1].plot(x, u_n*1000, label=label)
    axes[1,0].set_title('IC smoothness: p perturbation (%)'); axes[1,0].legend(fontsize=8); axes[1,0].set_xlabel('x')
    axes[1,1].set_title('IC smoothness: u (mm/s)'); axes[1,1].legend(fontsize=8); axes[1,1].set_xlabel('x')

    # Zoom at interface for smooth_w=0.005
    ph1, ph2, dx, x, a1r1, a2r2, rho_e0, a_air, _ = setup_eb4(N=200, smooth_width=0.005)
    p_n, _ = run(ph1, ph2, dx, a1r1, a2r2, rho_e0, a_air, 3e-4, N)
    mask = (x > 0.3) & (x < 0.7)
    axes[1,2].plot(x[mask], (p_n[mask] - p0)/p0 * 100, 'o-', label='tanh_w=0.005')
    ph1, ph2, dx, x, a1r1, a2r2, rho_e0, a_air, _ = setup_eb4(N=200, smooth_width=0.0)
    p_n_h, _ = run(ph1, ph2, dx, a1r1, a2r2, rho_e0, a_air, 3e-4, N)
    axes[1,2].plot(x[mask], (p_n_h[mask] - p0)/p0 * 100, 's-', label='Heaviside')
    axes[1,2].set_title('Interface zoom (x in [0.3, 0.7])'); axes[1,2].legend(fontsize=8); axes[1,2].set_xlabel('x')

    # ========= Test 3: Smaller dt (force more acoustic steps) =========
    print("\nTest 3: Very small CFL")
    ph1, ph2, dx, x, a1r1, a2r2, rho_e0, a_air, _ = setup_eb4(N=200, smooth_width=0)
    for cfl_test in [0.4, 0.05, 0.01]:
        t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
            ph1, ph2, a1r1.copy(), a2r2.copy(), np.zeros(N),
            rho_e0.copy(), a_air.copy(), dx, t_end=3e-4, cfl=cfl_test,
            bc_l='transmissive', bc_r='transmissive',
            max_steps=1000000, print_interval=1000000,
            alpha_scheme='tvd', use_strang=True,
            use_defect_correction=False, use_material_cfl=False,
            use_mmacm_ex=True)
        p_n, u_n, _, _, _, _, _, _ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)
        p_osc = (p_n - p_n.mean()).std() / p0
        print(f"  CFL={cfl_test}: p osc std={p_osc:.3e}")
        axes[2,0].plot(x, (p_n - p0)/p0 * 100, label=f'CFL={cfl_test}')
        axes[2,1].plot(x, u_n*1000, label=f'CFL={cfl_test}')
    axes[2,0].set_title('Very small CFL: p perturbation (%)'); axes[2,0].legend(fontsize=8); axes[2,0].set_xlabel('x')
    axes[2,1].set_title('Very small CFL: u (mm/s)'); axes[2,1].legend(fontsize=8); axes[2,1].set_xlabel('x')
    axes[2,2].axis('off')

    plt.suptitle('EB4 Deep Diagnosis: Oscillation Source', fontsize=14)
    plt.tight_layout()
    plt.savefig('results/diag_eb4_deep.png', dpi=150)
    print(f"\nPlot saved: results/diag_eb4_deep.png")
