"""EB4 진동 비교: solve_IMEX vs solve() (explicit HLLC)"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from solver.He2024.explicit_mmacm_ex import (solve_IMEX, solve, cons_to_prim)
from pipeline.exact_riemann import exact_profile

ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
ph2 = {'gamma': 4.4, 'pinf': 6e8, 'kv': 474.2, 'b': 0.0, 'eta': 0.0, 'q': 0.0}
N = 200; L = 1.0; dx = L/N
x = np.linspace(dx/2, L-dx/2, N)
x0 = 0.5; t_end = 3e-4
p0 = 1e5
p_L = p0 * 1.01
p_R = p0

p_init = np.where(x < x0, p_L, p_R)
rho1 = p_init / (0.4 * 717.5 * 293.0)
rho2 = (p_init + 6e8) / (3.4 * 474.2 * 293.0)
a_air = 1e-6 * np.ones(N)
a1r1 = a_air * rho1; a2r2 = (1-a_air) * rho2
rho_e0 = a_air * p_init / 0.4 + (1-a_air) * (p_init + 4.4*6e8) / 3.4

# Exact (single-phase water)
rho_L_w = (p_L + 6e8) / (3.4 * 474.2 * 293.0)
rho_R_w = (p_R + 6e8) / (3.4 * 474.2 * 293.0)
rho_e, u_e, p_e, _ = exact_profile(
    x, t_end, x0,
    pL=p_L, rhoL=rho_L_w, uL=0.0, gammaL=4.4, pinfL=6e8,
    pR=p_R, rhoR=rho_R_w, uR=0.0, gammaR=4.4, pinfR=6e8)

# ============ 1. IMEX ============
t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
    ph1, ph2, a1r1.copy(), a2r2.copy(), np.zeros(N),
    rho_e0.copy(), a_air.copy(), dx, t_end=t_end, cfl=0.4,
    bc_l='transmissive', bc_r='transmissive',
    max_steps=200000, print_interval=100000,
    alpha_scheme='tvd', use_strang=True,
    use_defect_correction=False, use_material_cfl=False)
p_imex, u_imex, _, _, _, _, _, _ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)

# ============ 2. Explicit (solve) ============
# Explicit uses acoustic CFL so dt much smaller - many more steps
t2, a1r1_f2, a2r2_f2, ru_f2, rE_f2, a1_f2 = solve(
    ph1, ph2, a1r1.copy(), a2r2.copy(), np.zeros(N),
    rho_e0.copy(), a_air.copy(), dx, t_end=t_end, cfl=0.4,
    bc_l='transmissive', bc_r='transmissive',
    max_steps=200000, print_interval=100000,
    alpha_recon='tvd')
p_expl, u_expl, _, _, _, _, _, _ = cons_to_prim(a1r1_f2, a2r2_f2, ru_f2, rE_f2, a1_f2, ph1, ph2)

# ============ Analysis ============
def osc_metric(p_n):
    return (p_n - p_n.mean()).std() / p0

print(f"IMEX:     t_final={t:.2e}, p osc std={osc_metric(p_imex):.3e}")
print(f"Explicit: t_final={t2:.2e}, p osc std={osc_metric(p_expl):.3e}")
print(f"IMEX dt ratio vs Explicit: {t2/t}x... wait, t_final same")
print(f"Number of steps (from output): check above")

# FFT comparison
fft_imex = np.abs(np.fft.fft((p_imex - p_imex.mean())/p0))[:N//2]
fft_expl = np.abs(np.fft.fft((p_expl - p_expl.mean())/p0))[:N//2]
freq = np.fft.fftfreq(N, d=dx)[:N//2]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

axes[0,0].plot(x, (p_e - p0)/p0 * 100, 'k-', lw=2, label='Exact')
axes[0,0].plot(x, (p_imex - p0)/p0 * 100, 'b.-', markersize=3, label=f'IMEX (osc={osc_metric(p_imex):.1e})')
axes[0,0].plot(x, (p_expl - p0)/p0 * 100, 'r.-', markersize=3, label=f'Explicit (osc={osc_metric(p_expl):.1e})')
axes[0,0].set_title('p perturbation (%)')
axes[0,0].legend(); axes[0,0].set_xlabel('x')

axes[0,1].plot(x, u_e*1000, 'k-', lw=2, label='Exact')
axes[0,1].plot(x, u_imex*1000, 'b.-', markersize=3, label='IMEX')
axes[0,1].plot(x, u_expl*1000, 'r.-', markersize=3, label='Explicit')
axes[0,1].set_title('u (mm/s)')
axes[0,1].legend(); axes[0,1].set_xlabel('x')

# Zoom on center
mask = (x > 0.35) & (x < 0.65)
axes[0,2].plot(x[mask], (p_imex[mask] - p0)/p0 * 100, 'bo-', markersize=5, label='IMEX')
axes[0,2].plot(x[mask], (p_expl[mask] - p0)/p0 * 100, 'rs-', markersize=5, label='Explicit')
axes[0,2].plot(x[mask], (p_e[mask] - p0)/p0 * 100, 'k--', lw=2, label='Exact')
axes[0,2].set_title('Center zoom: p perturbation (%)'); axes[0,2].legend(); axes[0,2].set_xlabel('x')

# FFT comparison
axes[1,0].semilogy(freq, fft_imex, 'b-', label='IMEX')
axes[1,0].semilogy(freq, fft_expl, 'r-', label='Explicit')
axes[1,0].axvline(1.0/(2*dx), ls='--', color='k', alpha=0.5, label=f'Nyquist=1/(2dx)={1.0/(2*dx):.0f}')
axes[1,0].set_xlabel('Wavenumber (1/m)'); axes[1,0].set_ylabel('|FFT(p)|')
axes[1,0].set_title('FFT magnitude of p')
axes[1,0].legend()

# Difference from exact
axes[1,1].plot(x, (p_imex - p_e)/p0 * 100, 'b-', label='IMEX')
axes[1,1].plot(x, (p_expl - p_e)/p0 * 100, 'r-', label='Explicit')
axes[1,1].axhline(0, ls='--', color='k', alpha=0.5)
axes[1,1].set_title('Error from exact: (p_num - p_exact)/p0 (%)')
axes[1,1].legend(); axes[1,1].set_xlabel('x')

# u zoom
axes[1,2].plot(x[mask], u_imex[mask]*1000, 'bo-', markersize=5, label='IMEX')
axes[1,2].plot(x[mask], u_expl[mask]*1000, 'rs-', markersize=5, label='Explicit')
axes[1,2].plot(x[mask], u_e[mask]*1000, 'k--', lw=2, label='Exact')
axes[1,2].set_title('Center zoom: u (mm/s)'); axes[1,2].legend(); axes[1,2].set_xlabel('x')

plt.suptitle('EB4 Low-Mach: IMEX vs Explicit (HLLC)', fontsize=14)
plt.tight_layout()
plt.savefig('results/diag_eb4_explicit.png', dpi=150)
print("Plot saved: results/diag_eb4_explicit.png")
