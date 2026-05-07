"""Validation 05-B / 07-B / 08-B — re-run aligned with spec files.

- 05-B: Ultra-low Mach pulse in water (internal custom test, SG water γ=4.4, P∞=6e8)
- 07-B: Sinusoidal acoustic in air f=2000 Hz (Denner 2018 Fig. 9)
- 08-B: Sinusoidal acoustic in water f=6000 Hz (Denner 2018 Fig. 10)

Outputs go to results/cat_B_exact/ (overwrite).
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/younglin90/work/claude_code/claudeCFD')
from solver.He2024.explicit_mmacm_ex import solve_IMEX, cons_to_prim
from solver.He2024.eos_general import IdealEOS, SGEOS

R = '/home/younglin90/work/claude_code/claudeCFD/results'
OUT = f'{R}/cat_B_exact'
os.makedirs(OUT, exist_ok=True)


def _plot5(x, a1, rho1, rho2, u, p, title, out, exact=None):
    fig, ax = plt.subplots(1, 5, figsize=(22, 4))
    data = [(a1, r'$\alpha_1$'), (rho1, r'$\rho_1$'), (rho2, r'$\rho_2$'),
            (u, 'u'), (p, 'p')]
    keys = ['a1_exact', 'rho1_exact', 'rho2_exact', 'u_exact', 'p_exact']
    for a, (y, lbl), k in zip(ax, data, keys):
        a.plot(x, y, 'b-', lw=1.3, label='numerical')
        if exact is not None and k in exact:
            a.plot(x, exact[k], 'r--', lw=1.1, label='exact')
            a.legend(fontsize=8)
        a.set_xlabel('x'); a.set_ylabel(lbl); a.grid(alpha=0.3)
    fig.suptitle(f'{title}  (blue=num, red-dashed=exact)')
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f'  saved: {out}', flush=True)


# ---------------------------------------------------------------------------
# 05-B: Ultra-low Mach pressure pulse in water (internal custom test)
# ---------------------------------------------------------------------------
def run_05B():
    print('\n[05-B: Ultra-low Mach pulse in water]', flush=True)
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5}
    ph2 = {'gamma': 4.4, 'pinf': 6.0e8, 'kv': 474.2}
    eos1 = IdealEOS(gamma=1.4, kv=717.5)
    eos2 = SGEOS(gamma=4.4, pinf=6.0e8, kv=474.2)

    N, L = 200, 1.0
    dx = L / N
    x = np.linspace(dx/2, L-dx/2, N)
    p0, T0, u0 = 1.0e5, 293.0, 0.0
    dp = 1.0

    a1 = np.full(N, 1e-6)  # pure water (α_air minority)
    p_init = np.where(np.abs(x - 0.5) < 0.1, p0 + dp, p0)
    rho2 = eos2.density(p_init, np.full(N, T0))
    rho1 = eos1.density(np.full(N, p0), np.full(N, T0))
    u = np.full(N, u0)

    a1r1 = a1 * rho1
    a2r2 = (1 - a1) * rho2
    rho = a1r1 + a2r2
    ru = rho * u
    e1 = eos1.energy(rho1, p_init)
    e2 = eos2.energy(rho2, p_init)
    rE = a1r1 * e1 + a2r2 * e2 + 0.5 * rho * u**2

    t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
        ph1, ph2, a1r1, a2r2, ru, rE, a1.copy(),
        dx=dx, t_end=3.0e-4, cfl=0.4,
        bc_l='transmissive', bc_r='transmissive',
        max_steps=5000, print_interval=999999,
        alpha_scheme='tvd', use_mmacm_ex=True,
        primitive_recon='tvd', use_acid_face=True,
        dissipation='none')
    p_f, u_f, T_f, rho1_f, rho2_f, *_ = cons_to_prim(
        a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)

    # d'Alembert exact (half amplitude, left- and right-moving)
    rho2_mean = float(np.mean(rho2))
    c_water = float(np.sqrt(4.4 * (p0 + 6e8) / rho2_mean))
    left_c = 0.4 - c_water * t
    right_c = 0.6 + c_water * t
    p_ex = np.full(N, p0)
    mask_l = np.abs(x - left_c) < 0.1
    mask_r = np.abs(x - right_c) < 0.1
    p_ex = p_ex + np.where(mask_l, dp/2, 0.0) + np.where(mask_r, dp/2, 0.0)
    u_ex = np.zeros(N)
    u_ex[mask_r] = +(dp/2) / (rho2_mean * c_water)
    u_ex[mask_l] = -(dp/2) / (rho2_mean * c_water)

    exact = {'a1_exact': a1, 'rho1_exact': rho1, 'rho2_exact': rho2,
             'u_exact': u_ex, 'p_exact': p_ex}

    d2 = p_f[2:] - 2*p_f[1:-1] + p_f[:-2]
    d2_rms = float(np.sqrt(np.mean(d2**2)) / p0)
    dp_max_num = float(np.max(p_f) - p0)
    dp_peak_ex = 0.5 * dp

    err_pct = abs(dp_max_num - dp_peak_ex) / dp_peak_ex * 100.0
    print(f'  c_water={c_water:.2f} m/s,  t={t:.3e} s', flush=True)
    print(f'  dp_peak exact={dp_peak_ex:.4f} Pa, numerical={dp_max_num:.4f} Pa, err={err_pct:.2f}%', flush=True)
    print(f'  d2_rms(p)/p0={d2_rms:.3e}', flush=True)

    title = (f'05-B Ultra-low Mach pulse (water)  '
             f'dp_peak: exact=0.5, num={dp_max_num:.3f}, err={err_pct:.1f}%, '
             f'd2={d2_rms:.2e}')
    _plot5(x, a1_f, rho1_f, rho2_f, u_f, p_f, title,
           f'{OUT}/05_ultra_low_mach.png', exact=exact)


# ---------------------------------------------------------------------------
# 07-B: Air sinusoidal acoustic  (Denner 2018 Fig. 9, f=2000 Hz)
# ---------------------------------------------------------------------------
def run_07B():
    print('\n[07-B: Air sinusoidal f=2000 Hz]', flush=True)
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5}
    ph2 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5}
    eos1 = IdealEOS(gamma=1.4, kv=717.5)
    eos2 = IdealEOS(gamma=1.4, kv=717.5)

    N, L = 500, 1.0
    dx = L / N
    x = np.linspace(dx/2, L-dx/2, N)
    p0, T0, u0, f = 1.0e5, 300.0, 1.0, 2000.0
    du = 0.01 * u0

    a1 = np.full(N, 1 - 1e-6)
    rho1 = eos1.density(np.full(N, p0), np.full(N, T0))
    rho2 = rho1.copy()
    a0 = float(np.sqrt(1.4 * p0 / rho1[0]))

    u = np.full(N, u0)
    p = np.full(N, p0)
    a1r1 = a1 * rho1
    a2r2 = (1 - a1) * rho2
    rho = a1r1 + a2r2
    ru = rho * u
    e1 = eos1.energy(rho1, p)
    e2 = eos2.energy(rho2, p)
    rE = a1r1 * e1 + a2r2 * e2 + 0.5 * rho * u**2

    def u_in(t_):
        return u0 + du * np.sin(2.0 * np.pi * f * t_)

    t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
        ph1, ph2, a1r1, a2r2, ru, rE, a1.copy(),
        dx=dx, t_end=2.3e-3, cfl=0.48,
        bc_l='inlet', bc_r='transmissive', u_inlet_func=u_in,
        max_steps=20000, print_interval=999999,
        alpha_scheme='tvd', use_mmacm_ex=False,
        primitive_recon='tvd', use_acid_face=True,
        dissipation='none')
    p_f, u_f, T_f, rho1_f, rho2_f, *_ = cons_to_prim(
        a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)

    # Linear acoustic exact, valid for x<a0*t (wave reached region)
    t_ret = t - x/a0
    mask = t_ret > 0
    u_ex = np.full(N, u0)
    p_ex = np.full(N, p0)
    u_ex[mask] = u0 + du * np.sin(2*np.pi*f*t_ret[mask])
    p_ex[mask] = p0 + rho1[0]*a0*du * np.sin(2*np.pi*f*t_ret[mask])
    rho_ex = rho1 + (p_ex - p0) / a0**2

    # Measurement band (exclude inlet & outlet) – 1 wavelength each side
    lam0 = a0 / f
    band = (x > lam0) & (x < L - lam0) & mask
    dp_num_band = p_f[band] - p0
    drho_num_band = rho1_f[band] - rho1[0]
    dp_amp = 0.5 * float(np.max(dp_num_band) - np.min(dp_num_band))
    drho_amp = 0.5 * float(np.max(drho_num_band) - np.min(drho_num_band))
    dp0 = rho1[0] * a0 * du
    drho0 = rho1[0] * du / a0
    err_p = abs(dp_amp - dp0) / dp0 * 100.0
    err_rho = abs(drho_amp - drho0) / drho0 * 100.0

    # Wavelength via zero crossings of dp on the band
    dp_sig = p_f[band] - p0
    x_band = x[band]
    zc = np.where(np.sign(dp_sig[:-1]) != np.sign(dp_sig[1:]))[0]
    if len(zc) >= 3:
        lam_meas = 2.0 * float(np.mean(np.diff(x_band[zc])))
    else:
        lam_meas = float('nan')
    err_lam = abs(lam_meas - lam0) / lam0 * 100.0 if not np.isnan(lam_meas) else float('nan')

    print(f'  a0={a0:.2f}, lam0={lam0:.4f}, t={t:.3e}', flush=True)
    print(f'  dp0={dp0:.3f} Pa  dp_num={dp_amp:.3f} Pa  err={err_p:.2f}%', flush=True)
    print(f'  drho0={drho0:.3e}  drho_num={drho_amp:.3e}  err={err_rho:.2f}%', flush=True)
    print(f'  lam0={lam0:.4f}  lam_num={lam_meas:.4f}  err={err_lam:.2f}%', flush=True)

    exact = {'a1_exact': a1, 'rho1_exact': rho_ex, 'rho2_exact': rho2,
             'u_exact': u_ex, 'p_exact': p_ex}
    title = (f'07-B Air Acoustic f=2000 Hz  '
             f'dp={dp_amp:.2f}/{dp0:.2f} Pa ({err_p:.1f}%), '
             f'drho={drho_amp:.2e}/{drho0:.2e} kg/m³ ({err_rho:.1f}%)')
    _plot5(x, a1_f, rho1_f, rho2_f, u_f, p_f, title,
           f'{OUT}/07_air_acoustic.png', exact=exact)


# ---------------------------------------------------------------------------
# 08-B: Water sinusoidal acoustic (Denner 2018 Fig. 10, f=6000 Hz)
# ---------------------------------------------------------------------------
def run_08B():
    print('\n[08-B: Water sinusoidal f=6000 Hz]', flush=True)
    # Denner 2018 Table 1 water SG EOS (γ=4.1, P∞=4.4e8, ρ0=998, a0=1344.6)
    gamma_w, pinf_w = 4.1, 4.4e8
    rho0_w = 998.0
    p0, T0, u0, f = 1.0e5, 300.0, 1.0, 6000.0
    # kv to keep ρ(p0,T0) ≈ 998 via SG: ρ=(p+P∞)/((γ-1)cv T)
    kv_w = (p0 + pinf_w) / ((gamma_w - 1.0) * rho0_w * T0)

    ph1 = {'gamma': gamma_w, 'pinf': pinf_w, 'kv': kv_w}
    ph2 = {'gamma': gamma_w, 'pinf': pinf_w, 'kv': kv_w}
    eos1 = SGEOS(gamma=gamma_w, pinf=pinf_w, kv=kv_w)
    eos2 = SGEOS(gamma=gamma_w, pinf=pinf_w, kv=kv_w)

    N, L = 500, 1.0
    dx = L / N
    x = np.linspace(dx/2, L-dx/2, N)
    du = 0.01 * u0

    a1 = np.full(N, 1 - 1e-6)
    rho1 = eos1.density(np.full(N, p0), np.full(N, T0))
    rho2 = rho1.copy()
    a0 = float(np.sqrt(gamma_w * (p0 + pinf_w) / rho1[0]))

    u = np.full(N, u0)
    p = np.full(N, p0)
    a1r1 = a1 * rho1
    a2r2 = (1 - a1) * rho2
    rho = a1r1 + a2r2
    ru = rho * u
    e1 = eos1.energy(rho1, p)
    e2 = eos2.energy(rho2, p)
    rE = a1r1 * e1 + a2r2 * e2 + 0.5 * rho * u**2

    def u_in(t_):
        return u0 + du * np.sin(2.0 * np.pi * f * t_)

    t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
        ph1, ph2, a1r1, a2r2, ru, rE, a1.copy(),
        dx=dx, t_end=6.5e-4, cfl=0.48,
        bc_l='inlet', bc_r='transmissive', u_inlet_func=u_in,
        max_steps=20000, print_interval=999999,
        alpha_scheme='tvd', use_mmacm_ex=False,
        primitive_recon='tvd', use_acid_face=True,
        dissipation='none')
    p_f, u_f, T_f, rho1_f, rho2_f, *_ = cons_to_prim(
        a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)

    t_ret = t - x/a0
    mask = t_ret > 0
    u_ex = np.full(N, u0)
    p_ex = np.full(N, p0)
    u_ex[mask] = u0 + du * np.sin(2*np.pi*f*t_ret[mask])
    p_ex[mask] = p0 + rho1[0]*a0*du * np.sin(2*np.pi*f*t_ret[mask])
    rho_ex = rho1 + (p_ex - p0) / a0**2

    lam0 = a0 / f
    band = (x > lam0) & (x < L - lam0) & mask
    dp_num_band = p_f[band] - p0
    drho_num_band = rho1_f[band] - rho1[0]
    dp_amp = 0.5 * float(np.max(dp_num_band) - np.min(dp_num_band))
    drho_amp = 0.5 * float(np.max(drho_num_band) - np.min(drho_num_band))
    dp0 = rho1[0] * a0 * du
    drho0 = rho1[0] * du / a0
    err_p = abs(dp_amp - dp0) / dp0 * 100.0
    err_rho = abs(drho_amp - drho0) / drho0 * 100.0

    dp_sig = p_f[band] - p0
    x_band = x[band]
    zc = np.where(np.sign(dp_sig[:-1]) != np.sign(dp_sig[1:]))[0]
    if len(zc) >= 3:
        lam_meas = 2.0 * float(np.mean(np.diff(x_band[zc])))
    else:
        lam_meas = float('nan')
    err_lam = abs(lam_meas - lam0) / lam0 * 100.0 if not np.isnan(lam_meas) else float('nan')

    print(f'  a0={a0:.2f} m/s (ref 1344.6), lam0={lam0:.4f}, t={t:.3e}', flush=True)
    print(f'  dp0={dp0:.1f} Pa  dp_num={dp_amp:.1f} Pa  err={err_p:.2f}%', flush=True)
    print(f'  drho0={drho0:.3e}  drho_num={drho_amp:.3e}  err={err_rho:.2f}%', flush=True)
    print(f'  lam0={lam0:.4f}  lam_num={lam_meas:.4f}  err={err_lam:.2f}%', flush=True)

    exact = {'a1_exact': a1, 'rho1_exact': rho_ex, 'rho2_exact': rho2,
             'u_exact': u_ex, 'p_exact': p_ex}
    title = (f'08-B Water Acoustic f=6000 Hz  '
             f'dp={dp_amp:.0f}/{dp0:.0f} Pa ({err_p:.1f}%), '
             f'drho={drho_amp:.2e}/{drho0:.2e} ({err_rho:.1f}%)')
    _plot5(x, a1_f, rho1_f, rho2_f, u_f, p_f, title,
           f'{OUT}/08_water_acoustic.png', exact=exact)


if __name__ == '__main__':
    run_05B()
    run_07B()
    run_08B()
    print('\nAll done.')
