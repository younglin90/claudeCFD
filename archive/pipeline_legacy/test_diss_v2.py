"""Better metric: FFT amplitude at Nyquist (pure 2Δx mode)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from solver.He2024.explicit_mmacm_ex import solve_IMEX, cons_to_prim


def run(diss, coef):
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
    ph2 = {'gamma': 4.4, 'pinf': 6e8, 'kv': 474.2, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
    N = 200; L = 1.0; dx = L/N
    x = np.linspace(dx/2, L-dx/2, N)
    x0 = 0.5
    p0 = 1e5; p_L = p0*1.01; p_R = p0
    p_init = np.where(x < x0, p_L, p_R)
    rho1 = p_init / (0.4 * 717.5 * 293.0)
    rho2 = (p_init + 6e8) / (3.4 * 474.2 * 293.0)
    a_air = 1e-6 * np.ones(N)
    a1r1 = a_air * rho1; a2r2 = (1-a_air) * rho2
    rho_e0 = a_air * p_init / 0.4 + (1-a_air) * (p_init + 4.4*6e8) / 3.4

    t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
        ph1, ph2, a1r1, a2r2, np.zeros(N), rho_e0, a_air,
        dx, t_end=3e-4, cfl=0.4, bc_l='transmissive', bc_r='transmissive',
        max_steps=500, print_interval=1000,
        alpha_scheme='tvd', use_strang=True,
        use_defect_correction=False, use_material_cfl=False,
        dissipation=diss, diss_coef=coef)
    p_n, u_n, _, _, _, _, _, _ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)
    return x, p_n, u_n


def metrics(p_n, p0):
    # Extract Nyquist component by inner product with alternating sign
    N = len(p_n)
    sign_alt = (-1.0) ** np.arange(N)
    nyquist_amp = np.abs(np.mean(p_n * sign_alt)) / p0
    # Also 2nd-difference (detects high-freq)
    d2 = p_n[2:] - 2*p_n[1:-1] + p_n[:-2]
    d2_rms = np.sqrt(np.mean(d2**2)) / p0
    return nyquist_amp, d2_rms


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    p0 = 1e5
    configs = [
        ('none', 0.0, 'Baseline'),
        ('shapiro', 0.1, 'Shapiro 0.1'),
        ('shapiro', 0.3, 'Shapiro 0.3'),
        ('shapiro', 0.5, 'Shapiro 0.5'),
        ('shapiro', 1.0, 'Shapiro 1.0'),
        ('mwi', 0.3, 'MWI 0.3'),
        ('mwi', 0.5, 'MWI 0.5'),
    ]

    print(f"{'Config':<15s} {'Nyquist amp (rel)':<20s} {'d2_rms (rel)':<18s}")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, len(configs)))

    results_store = []
    for i, (diss, coef, label) in enumerate(configs):
        try:
            x, p_n, u_n = run(diss, coef)
            nyq, d2 = metrics(p_n, p0)
            print(f"{label:<15s} {nyq:<20.3e} {d2:<18.3e}")
            if np.isfinite(nyq):
                axes[0,0].plot(x, (p_n - p0)/p0 * 100, color=colors[i], label=f'{label} Nyq={nyq:.1e}')
                mask = (x > 0.35) & (x < 0.65)
                axes[0,1].plot(x[mask], (p_n[mask] - p0)/p0 * 100, 'o-', color=colors[i],
                               markersize=3, label=label)
                axes[1,0].plot(x, u_n*1000, color=colors[i], label=label)
                axes[1,1].plot(x[mask], u_n[mask]*1000, 'o-', color=colors[i],
                               markersize=3, label=label)
                results_store.append((label, x, p_n, u_n, nyq, d2))
        except Exception as e:
            print(f"{label:<15s} FAILED: {e}")

    axes[0,0].set_title('Full p perturbation (%)')
    axes[0,0].legend(fontsize=7); axes[0,0].set_xlabel('x')
    axes[0,1].set_title('Center zoom: p (%)')
    axes[0,1].legend(fontsize=7); axes[0,1].set_xlabel('x')
    axes[1,0].set_title('Full u (mm/s)')
    axes[1,0].legend(fontsize=7); axes[1,0].set_xlabel('x')
    axes[1,1].set_title('Center zoom: u (mm/s)')
    axes[1,1].legend(fontsize=7); axes[1,1].set_xlabel('x')

    plt.suptitle('EB4 Dissipation Comparison (Nyquist amp measured)', fontsize=14)
    plt.tight_layout()
    plt.savefig('results/dissipation_v2.png', dpi=150)
    print(f"\nPlot: results/dissipation_v2.png")
