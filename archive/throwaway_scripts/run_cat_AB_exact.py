"""Category A (PE Preservation) + B (Low-Mach Acoustics) validation with exact overlay.

각 case 재실행 + 5-panel plot (α₁, ρ₁, ρ₂, u, p) with numerical + exact overlay.
"""
import os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/younglin90/work/claude_code/claudeCFD')
from solver.He2024.explicit_mmacm_ex import solve_IMEX, cons_to_prim
from solver.He2024.eos_general import SGEOS, IdealEOS, NASGEOS

R = '/home/younglin90/work/claude_code/claudeCFD/results'
os.makedirs(f'{R}/cat_A_exact', exist_ok=True)
os.makedirs(f'{R}/cat_B_exact', exist_ok=True)


def _plot(x, a1, rho1, rho2, u, p, title, out, exact=None):
    fig, ax = plt.subplots(1, 5, figsize=(22, 4))
    for a, y, lbl in zip(ax, [a1, rho1, rho2, u, p], ['alpha_1', 'rho_1', 'rho_2', 'u', 'p']):
        a.plot(x, y, 'b-', lw=1.3, label='numerical')
        a.set_xlabel('x'); a.set_ylabel(lbl); a.grid(alpha=0.3)
    if exact:
        if 'a1_exact' in exact:
            ax[0].plot(x, exact['a1_exact'], 'r--', lw=1.0, label='exact'); ax[0].legend(fontsize=7)
        if 'rho1_exact' in exact:
            ax[1].plot(x, exact['rho1_exact'], 'r--', lw=1.0, label='exact'); ax[1].legend(fontsize=7)
        if 'rho2_exact' in exact:
            ax[2].plot(x, exact['rho2_exact'], 'r--', lw=1.0, label='exact'); ax[2].legend(fontsize=7)
        if 'u_exact' in exact:
            ax[3].plot(x, exact['u_exact'], 'r--', lw=1.0, label='exact'); ax[3].legend(fontsize=7)
        if 'p_exact' in exact:
            ax[4].plot(x, exact['p_exact'], 'r--', lw=1.0, label='exact'); ax[4].legend(fontsize=7)
    fig.suptitle(f'{title}  (blue=num, red-dashed=exact)')
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f'  saved: {out}', flush=True)


# =====================================================================
# CATEGORY A: PE Preservation
# =====================================================================

def run_01A_abgrall_2phase():
    """Test A: 2-phase water(NASG)-air(ideal) advection, u=1, p=1e5, N=10, periodic."""
    print('[01A: 2-phase Abgrall]', flush=True)
    ph1 = {'gamma': 1.187, 'pinf': 7.028e8, 'kv': 3610, 'b': 6.61e-4, 'eta': -1.177788e6}
    ph2 = {'gamma': 1.4, 'pinf': 0, 'kv': 717.5}
    eos1 = NASGEOS(gamma=1.187, pinf=7.028e8, kv=3610, b=6.61e-4, eta=-1.177788e6)
    eos2 = IdealEOS(gamma=1.4, kv=717.5)
    N, L = 10, 1.0; dx = L/N
    x = np.linspace(dx/2, L-dx/2, N)
    p0, u0, T0 = 1e5, 1.0, 300.0
    a_water = ((x >= 0.4) & (x <= 0.6)).astype(float)
    a1 = a_water * (1 - 1e-6) + (1 - a_water) * 1e-6
    # densities from EOS(p, T)
    rho1 = eos1.density(np.full(N, p0), np.full(N, T0))
    rho2 = eos2.density(np.full(N, p0), np.full(N, T0))
    u = np.full(N, u0); p = np.full(N, p0)
    a1r1 = a1 * rho1; a2r2 = (1 - a1) * rho2
    rho = a1r1 + a2r2; ru = rho * u
    e1 = eos1.energy(rho1, p); e2 = eos2.energy(rho2, p)
    rE = a1r1 * e1 + a2r2 * e2 + 0.5 * rho * u**2
    t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
        ph1, ph2, a1r1, a2r2, ru, rE, a1.copy(),
        dx=dx, t_end=1.0, cfl=10.0,  # implicit; large CFL ok
        bc_l='periodic', bc_r='periodic',
        max_steps=200, print_interval=100,
        alpha_scheme='thinc_bvd', use_mmacm_ex=True)
    p_f, u_f, T_f, rho1_f, rho2_f, *_ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)
    # Exact after 1 full revolution (u0=1, t=1.0): same as initial
    exact = {'a1_exact': a1, 'rho1_exact': rho1, 'rho2_exact': rho2,
             'u_exact': np.full(N, u0), 'p_exact': np.full(N, p0)}
    err_p = np.max(np.abs(p_f - p0) / p0); err_u = np.max(np.abs(u_f - u0))
    print(f'  err_p={err_p:.3e} err_u={err_u:.3e}', flush=True)
    _plot(x, a1_f, rho1_f, rho2_f, u_f, p_f,
          f'01A 2-phase Abgrall  err_p={err_p:.2e}, err_u={err_u:.2e}',
          f'{R}/cat_A_exact/01A_2phase.png', exact=exact)


def run_02_static_interface():
    """Static air-water, u=0, p=1e5 uniform, long-time equilibrium."""
    print('[02: Static interface]', flush=True)
    ph1 = {'gamma': 1.4, 'pinf': 0, 'kv': 717.5}
    ph2 = {'gamma': 4.4, 'pinf': 6e8, 'kv': 474.2}
    eos1 = IdealEOS(gamma=1.4, kv=717.5)
    eos2 = SGEOS(gamma=4.4, pinf=6e8, kv=474.2)
    N, L = 100, 1.0; dx = L/N
    x = np.linspace(dx/2, L-dx/2, N)
    p0, u0, T0 = 1e5, 0.0, 293.0
    a1 = np.where(x < 0.5, 1 - 1e-6, 1e-6)  # air left, water right
    rho1 = eos1.density(np.full(N, p0), np.full(N, T0))
    rho2 = eos2.density(np.full(N, p0), np.full(N, T0))
    u = np.zeros(N); p = np.full(N, p0)
    a1r1 = a1 * rho1; a2r2 = (1 - a1) * rho2
    rho = a1r1 + a2r2; ru = rho * u
    e1 = eos1.energy(rho1, p); e2 = eos2.energy(rho2, p)
    rE = a1r1 * e1 + a2r2 * e2 + 0.5 * rho * u**2
    t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
        ph1, ph2, a1r1, a2r2, ru, rE, a1.copy(),
        dx=dx, t_end=1e-3, cfl=0.4,
        bc_l='transmissive', bc_r='transmissive',
        max_steps=10000, print_interval=2000,
        alpha_scheme='thinc_bvd', use_mmacm_ex=True)
    p_f, u_f, T_f, rho1_f, rho2_f, *_ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)
    # Exact: stays at initial (PE preservation)
    exact = {'a1_exact': a1, 'rho1_exact': rho1, 'rho2_exact': rho2,
             'u_exact': np.zeros(N), 'p_exact': np.full(N, p0)}
    err_p = np.max(np.abs(p_f - p0) / p0); err_u = np.max(np.abs(u_f))
    print(f'  err_p={err_p:.3e} err_u={err_u:.3e}', flush=True)
    _plot(x, a1_f, rho1_f, rho2_f, u_f, p_f,
          f'02 Static interface  err_p={err_p:.2e}, err_u={err_u:.2e}',
          f'{R}/cat_A_exact/02_static.png', exact=exact)


def run_03_moving_contact():
    """Moving contact u=100 uniform, periodic BC. TVD + MMACM=off (optimal)."""
    print('[03: Moving contact u=100]', flush=True)
    ph1 = {'gamma': 1.4, 'pinf': 0, 'kv': 717.5}
    ph2 = {'gamma': 1.4, 'pinf': 0, 'kv': 717.5}
    eos1 = IdealEOS(gamma=1.4, kv=717.5)
    eos2 = IdealEOS(gamma=1.4, kv=717.5)
    N, L = 200, 1.0; dx = L/N
    x = np.linspace(dx/2, L-dx/2, N)
    p0, u0, T0 = 1e5, 100.0, 300.0
    a1 = np.where((x >= 0.4) & (x <= 0.6), 1e-6, 1-1e-6)
    rho1 = eos1.density(np.full(N, p0), np.full(N, T0))
    rho2 = eos2.density(np.full(N, p0), np.full(N, T0))
    u = np.full(N, u0); p = np.full(N, p0)
    a1r1 = a1 * rho1; a2r2 = (1 - a1) * rho2
    rho = a1r1 + a2r2; ru = rho * u
    e1 = eos1.energy(rho1, p); e2 = eos2.energy(rho2, p)
    rE = a1r1 * e1 + a2r2 * e2 + 0.5 * rho * u**2
    t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
        ph1, ph2, a1r1, a2r2, ru, rE, a1.copy(),
        dx=dx, t_end=0.01, cfl=0.4,
        bc_l='periodic', bc_r='periodic',
        max_steps=20000, print_interval=5000,
        alpha_scheme='tvd', use_mmacm_ex=False)  # Optimal: TVD + MMACM off for smooth advection
    p_f, u_f, T_f, rho1_f, rho2_f, *_ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)
    # Exact after u0*t = 1.0 (one full rev): α back to initial, p/u uniform
    exact = {'a1_exact': a1, 'rho1_exact': rho1, 'rho2_exact': rho2,
             'u_exact': np.full(N, u0), 'p_exact': np.full(N, p0)}
    err_p = np.max(np.abs(p_f - p0) / p0); err_u = np.max(np.abs(u_f - u0))
    print(f'  err_p={err_p:.3e} err_u={err_u:.3e}', flush=True)
    _plot(x, a1_f, rho1_f, rho2_f, u_f, p_f,
          f'03 Moving contact u=100  err_p={err_p:.2e}, err_u={err_u:.2e}',
          f'{R}/cat_A_exact/03_moving_contact.png', exact=exact)


# =====================================================================
# CATEGORY B: Low-Mach Acoustics
# =====================================================================

def run_05_ultralow_mach_pulse():
    """Ultra-low Mach pressure pulse in water, dp=1 Pa."""
    print('[05: Ultra-low Mach pulse]', flush=True)
    ph1 = {'gamma': 1.4, 'pinf': 0, 'kv': 717.5}
    ph2 = {'gamma': 4.4, 'pinf': 6e8, 'kv': 474.2}
    eos1 = IdealEOS(gamma=1.4, kv=717.5)
    eos2 = SGEOS(gamma=4.4, pinf=6e8, kv=474.2)
    N, L = 200, 1.0; dx = L/N
    x = np.linspace(dx/2, L-dx/2, N)
    p0, u0, T0 = 1e5, 0.0, 293.0
    a1 = np.full(N, 1e-6)  # pure water
    # pulse: dp = 1 Pa in |x-0.5|<0.1
    dp = 1.0
    p = np.where(np.abs(x - 0.5) < 0.1, p0 + dp, p0)
    rho2 = eos2.density(p, np.full(N, T0))
    rho1 = eos1.density(np.full(N, p0), np.full(N, T0))
    u = np.zeros(N)
    a1r1 = a1 * rho1; a2r2 = (1 - a1) * rho2
    rho = a1r1 + a2r2; ru = rho * u
    e1 = eos1.energy(rho1, p); e2 = eos2.energy(rho2, p)
    rE = a1r1 * e1 + a2r2 * e2 + 0.5 * rho * u**2
    p_init = p.copy()
    t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
        ph1, ph2, a1r1, a2r2, ru, rE, a1.copy(),
        dx=dx, t_end=3e-4, cfl=0.4,
        bc_l='transmissive', bc_r='transmissive',
        max_steps=5000, print_interval=1000,
        alpha_scheme='thinc_bvd', use_mmacm_ex=True)
    p_f, u_f, T_f, rho1_f, rho2_f, *_ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)
    # Linear acoustic exact: pulse splits into left + right moving half-amplitude
    # At t: left pulse centered at 0.4 - c*t, right pulse at 0.6 + c*t
    c_water = np.sqrt(4.4 * (p0 + 6e8) / np.mean(rho2))
    left_pulse_center = 0.4 - c_water * t
    right_pulse_center = 0.6 + c_water * t
    p_exact = np.full(N, p0)
    # left-moving half
    mask_l = np.abs(x - left_pulse_center) < 0.1
    p_exact[mask_l] += dp / 2
    # right-moving half
    mask_r = np.abs(x - right_pulse_center) < 0.1
    p_exact[mask_r] += dp / 2
    u_exact = np.zeros(N)
    u_exact[mask_r] = (dp / 2) / (np.mean(rho2) * c_water)
    u_exact[mask_l] = -(dp / 2) / (np.mean(rho2) * c_water)
    exact = {'p_exact': p_exact, 'u_exact': u_exact,
             'rho1_exact': rho1, 'rho2_exact': rho2, 'a1_exact': a1}
    # 2Δx indicator
    d2 = p_f[2:] - 2*p_f[1:-1] + p_f[:-2]
    d2_rms = np.sqrt(np.mean(d2**2)) / p0
    print(f'  d2_rms={d2_rms:.3e}, dp_max={np.max(p_f)-p0:.3e}', flush=True)
    _plot(x, a1_f, rho1_f, rho2_f, u_f, p_f,
          f'05 Ultra-low Mach pulse  d2={d2_rms:.2e}, dp_max={np.max(p_f)-p0:.2f}',
          f'{R}/cat_B_exact/05_ultra_low_mach.png', exact=exact)


def run_07_air_acoustic_2000Hz():
    """Sinusoidal acoustic wave in air, inlet BC, f=2000Hz."""
    print('[07: Air acoustic f=2000Hz]', flush=True)
    ph1 = {'gamma': 1.4, 'pinf': 0, 'kv': 717.5}
    ph2 = {'gamma': 1.4, 'pinf': 0, 'kv': 717.5}
    eos1 = IdealEOS(gamma=1.4, kv=717.5); eos2 = IdealEOS(gamma=1.4, kv=717.5)
    N, L = 500, 1.0; dx = L/N
    x = np.linspace(dx/2, L-dx/2, N)
    p0, u0, T0, f = 1e5, 1.0, 300.0, 2000.0
    du = 0.01 * u0
    a1 = np.full(N, 1 - 1e-6)
    rho1 = eos1.density(np.full(N, p0), np.full(N, T0))
    rho2 = rho1.copy()
    a0 = np.sqrt(1.4 * p0 / rho1[0])
    u = np.full(N, u0); p = np.full(N, p0)
    a1r1 = a1 * rho1; a2r2 = (1 - a1) * rho2
    rho = a1r1 + a2r2; ru = rho * u
    e1 = eos1.energy(rho1, p); e2 = eos2.energy(rho2, p)
    rE = a1r1 * e1 + a2r2 * e2 + 0.5 * rho * u**2
    def u_in(t_): return u0 + du * np.sin(2*np.pi*f*t_)
    t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
        ph1, ph2, a1r1, a2r2, ru, rE, a1.copy(),
        dx=dx, t_end=2.3e-3, cfl=0.4,
        bc_l='inlet', bc_r='transmissive', u_inlet_func=u_in,
        max_steps=10000, print_interval=5000,
        alpha_scheme='thinc_bvd', use_mmacm_ex=True)
    p_f, u_f, T_f, rho1_f, rho2_f, *_ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)
    # Linear acoustic: p = p0 + rho*a*du*sin(2πf(t-x/a0)) for x < a0*t
    t_ret = t - x/a0
    mask = t_ret > 0
    u_exact = np.full(N, u0); p_exact = np.full(N, p0)
    u_exact[mask] = u0 + du * np.sin(2*np.pi*f*t_ret[mask])
    p_exact[mask] = p0 + rho1[0]*a0*du * np.sin(2*np.pi*f*t_ret[mask])
    exact = {'p_exact': p_exact, 'u_exact': u_exact,
             'rho1_exact': rho1, 'rho2_exact': rho2, 'a1_exact': a1}
    dp_exp = rho1[0]*a0*du
    dp_meas = (np.max(p_f) - np.min(p_f)) / 2
    print(f'  dp_exp={dp_exp:.2f}, dp_meas={dp_meas:.2f}', flush=True)
    _plot(x, a1_f, rho1_f, rho2_f, u_f, p_f,
          f'07 Air acoustic f=2000Hz  dp_exp={dp_exp:.1f}, dp_meas={dp_meas:.1f}',
          f'{R}/cat_B_exact/07_air_acoustic.png', exact=exact)


def run_09_impedance_matching():
    """Acoustic impedance matching, two gases with Z_L = Z_R, no reflection."""
    print('[09: Impedance matching]', flush=True)
    ph1 = {'gamma': 1.4, 'pinf': 0, 'kv': 717.5}
    ph2 = {'gamma': 1.01, 'pinf': 0, 'kv': 3000}
    eos1 = IdealEOS(gamma=1.4, kv=717.5)
    eos2 = IdealEOS(gamma=1.01, kv=3000)
    N, L = 500, 1.0; dx = L/N
    x = np.linspace(dx/2, L-dx/2, N)
    p0, u0 = 1e5, 0.30886
    rho1_ref = 1.265; rho2_ref = 1.7537
    f, du = 2000.0, 0.01 * u0
    a1 = np.where(x < 0.5, 1 - 1e-6, 1e-6)
    rho1 = np.full(N, rho1_ref); rho2 = np.full(N, rho2_ref)
    u = np.full(N, u0); p = np.full(N, p0)
    a1r1 = a1 * rho1; a2r2 = (1 - a1) * rho2
    rho = a1r1 + a2r2; ru = rho * u
    e1 = eos1.energy(rho1, p); e2 = eos2.energy(rho2, p)
    rE = a1r1 * e1 + a2r2 * e2 + 0.5 * rho * u**2
    def u_in(t_): return u0 + du * np.sin(2*np.pi*f*t_)
    t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
        ph1, ph2, a1r1, a2r2, ru, rE, a1.copy(),
        dx=dx, t_end=1.5e-3, cfl=0.4,
        bc_l='inlet', bc_r='transmissive', u_inlet_func=u_in,
        max_steps=10000, print_interval=5000,
        alpha_scheme='thinc_bvd', use_mmacm_ex=True)
    p_f, u_f, T_f, rho1_f, rho2_f, *_ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)
    # Exact: impedance-matched → wave passes through; no reflection. Amplitude continues.
    c1 = np.sqrt(1.4 * p0 / rho1_ref); c2 = np.sqrt(1.01 * p0 / rho2_ref)
    Z1 = rho1_ref * c1; Z2 = rho2_ref * c2
    # Simplified exact: p uniform at p0 + dp_amp, u uniform at u0 + du (wave not yet exited)
    u_exact = np.full(N, u0); p_exact = np.full(N, p0)
    # both halves carry same wave (impedance matched)
    t_ret_L = t - x/c1
    t_ret_R = t - 0.5/c1 - (x - 0.5)/c2  # cross interface at x=0.5
    mask_L = (x < 0.5) & (t_ret_L > 0)
    mask_R = (x >= 0.5) & (t_ret_R > 0)
    u_exact[mask_L] = u0 + du * np.sin(2*np.pi*f*t_ret_L[mask_L])
    p_exact[mask_L] = p0 + Z1*du * np.sin(2*np.pi*f*t_ret_L[mask_L])
    u_exact[mask_R] = u0 + du * np.sin(2*np.pi*f*t_ret_R[mask_R])
    p_exact[mask_R] = p0 + Z2*du * np.sin(2*np.pi*f*t_ret_R[mask_R])
    exact = {'p_exact': p_exact, 'u_exact': u_exact,
             'rho1_exact': rho1, 'rho2_exact': rho2, 'a1_exact': a1}
    dp_L = np.max(p_f[x < 0.4]) - np.min(p_f[x < 0.4])
    dp_R = np.max(p_f[x > 0.6]) - np.min(p_f[x > 0.6])
    ratio = dp_R / dp_L if dp_L > 0 else 0
    print(f'  Z1={Z1:.2f}, Z2={Z2:.2f}, dp_R/dp_L={ratio:.3f}', flush=True)
    _plot(x, a1_f, rho1_f, rho2_f, u_f, p_f,
          f'09 Impedance match Z1={Z1:.0f} Z2={Z2:.0f}, dp_R/dp_L={ratio:.2f}',
          f'{R}/cat_B_exact/09_impedance_matching.png', exact=exact)


def run_01A_v2():
    """Fixed: CFL=0.4, shorter t_end=0.1 (10 periods at 10 cells)."""
    print('[01A_v2: 2-phase Abgrall, CFL=0.4, t=0.1]', flush=True)
    ph1 = {'gamma': 1.187, 'pinf': 7.028e8, 'kv': 3610, 'b': 6.61e-4, 'eta': -1.177788e6}
    ph2 = {'gamma': 1.4, 'pinf': 0, 'kv': 717.5}
    eos1 = NASGEOS(gamma=1.187, pinf=7.028e8, kv=3610, b=6.61e-4, eta=-1.177788e6)
    eos2 = IdealEOS(gamma=1.4, kv=717.5)
    N, L = 10, 1.0; dx = L/N
    x = np.linspace(dx/2, L-dx/2, N)
    p0, u0, T0 = 1e5, 1.0, 300.0
    a_water = ((x >= 0.4) & (x <= 0.6)).astype(float)
    a1 = a_water * (1 - 1e-6) + (1 - a_water) * 1e-6
    rho1 = eos1.density(np.full(N, p0), np.full(N, T0))
    rho2 = eos2.density(np.full(N, p0), np.full(N, T0))
    u = np.full(N, u0); p = np.full(N, p0)
    a1r1 = a1 * rho1; a2r2 = (1 - a1) * rho2
    rho = a1r1 + a2r2; ru = rho * u
    e1 = eos1.energy(rho1, p); e2 = eos2.energy(rho2, p)
    rE = a1r1 * e1 + a2r2 * e2 + 0.5 * rho * u**2
    t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
        ph1, ph2, a1r1, a2r2, ru, rE, a1.copy(),
        dx=dx, t_end=0.1, cfl=0.4,
        bc_l='periodic', bc_r='periodic',
        max_steps=20000, print_interval=2000,
        alpha_scheme='tvd', use_mmacm_ex=True)  # Optimal for NASG: TVD + MMACM on
    p_f, u_f, T_f, rho1_f, rho2_f, *_ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)
    # Exact: after t=0.1, u=1 uniform, p=1e5 uniform. α shifted by u*t = 0.1
    shift = (u0 * t) % L
    a1_exact_shift = np.interp((x - shift) % L, x, a1, period=L)
    exact = {'a1_exact': a1_exact_shift, 'rho1_exact': rho1, 'rho2_exact': rho2,
             'u_exact': np.full(N, u0), 'p_exact': np.full(N, p0)}
    err_p = np.max(np.abs(p_f - p0) / p0); err_u = np.max(np.abs(u_f - u0))
    print(f'  err_p={err_p:.3e} err_u={err_u:.3e}', flush=True)
    _plot(x, a1_f, rho1_f, rho2_f, u_f, p_f,
          f'01A 2-phase Abgrall v2 err_p={err_p:.2e} err_u={err_u:.2e}',
          f'{R}/cat_A_exact/01A_2phase_v2.png', exact=exact)


def run_03_v2():
    """Fixed: max_steps=20000."""
    print('[03_v2: Moving contact, max_steps=20000]', flush=True)
    ph1 = {'gamma': 1.4, 'pinf': 0, 'kv': 717.5}
    ph2 = {'gamma': 1.4, 'pinf': 0, 'kv': 717.5}
    eos1 = IdealEOS(gamma=1.4, kv=717.5); eos2 = IdealEOS(gamma=1.4, kv=717.5)
    N, L = 200, 1.0; dx = L/N
    x = np.linspace(dx/2, L-dx/2, N)
    p0, u0, T0 = 1e5, 100.0, 300.0
    a1 = np.where((x >= 0.4) & (x <= 0.6), 1e-6, 1-1e-6)
    rho1 = eos1.density(np.full(N, p0), np.full(N, T0))
    rho2 = eos2.density(np.full(N, p0), np.full(N, T0))
    u = np.full(N, u0); p = np.full(N, p0)
    a1r1 = a1 * rho1; a2r2 = (1 - a1) * rho2
    rho = a1r1 + a2r2; ru = rho * u
    e1 = eos1.energy(rho1, p); e2 = eos2.energy(rho2, p)
    rE = a1r1 * e1 + a2r2 * e2 + 0.5 * rho * u**2
    t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
        ph1, ph2, a1r1, a2r2, ru, rE, a1.copy(),
        dx=dx, t_end=0.01, cfl=0.4,
        bc_l='periodic', bc_r='periodic',
        max_steps=20000, print_interval=5000,
        alpha_scheme='thinc_bvd', use_mmacm_ex=True)
    p_f, u_f, T_f, rho1_f, rho2_f, *_ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)
    exact = {'a1_exact': a1, 'rho1_exact': rho1, 'rho2_exact': rho2,
             'u_exact': np.full(N, u0), 'p_exact': np.full(N, p0)}
    err_p = np.max(np.abs(p_f - p0) / p0); err_u = np.max(np.abs(u_f - u0))
    print(f'  err_p={err_p:.3e} err_u={err_u:.3e}', flush=True)
    _plot(x, a1_f, rho1_f, rho2_f, u_f, p_f,
          f'03 Moving contact v2 err_p={err_p:.2e} err_u={err_u:.2e}',
          f'{R}/cat_A_exact/03_moving_contact_v2.png', exact=exact)


def run_09_v2():
    """Fixed: larger N + longer time, better amplitude measurement."""
    print('[09_v2: Impedance matching, better measurement]', flush=True)
    ph1 = {'gamma': 1.4, 'pinf': 0, 'kv': 717.5}
    ph2 = {'gamma': 1.01, 'pinf': 0, 'kv': 3000}
    eos1 = IdealEOS(gamma=1.4, kv=717.5); eos2 = IdealEOS(gamma=1.01, kv=3000)
    N, L = 1000, 1.0; dx = L/N
    x = np.linspace(dx/2, L-dx/2, N)
    p0, u0 = 1e5, 0.30886
    rho1_ref = 1.265; rho2_ref = 1.7537
    f, du = 2000.0, 0.01 * u0
    a1 = np.where(x < 0.5, 1 - 1e-6, 1e-6)
    rho1 = np.full(N, rho1_ref); rho2 = np.full(N, rho2_ref)
    u = np.full(N, u0); p = np.full(N, p0)
    a1r1 = a1 * rho1; a2r2 = (1 - a1) * rho2
    rho = a1r1 + a2r2; ru = rho * u
    e1 = eos1.energy(rho1, p); e2 = eos2.energy(rho2, p)
    rE = a1r1 * e1 + a2r2 * e2 + 0.5 * rho * u**2
    def u_in(t_): return u0 + du * np.sin(2*np.pi*f*t_)
    t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
        ph1, ph2, a1r1, a2r2, ru, rE, a1.copy(),
        dx=dx, t_end=2.0e-3, cfl=0.4,
        bc_l='inlet', bc_r='transmissive', u_inlet_func=u_in,
        max_steps=20000, print_interval=5000,
        alpha_scheme='thinc_bvd', use_mmacm_ex=True)
    p_f, u_f, T_f, rho1_f, rho2_f, *_ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)
    c1 = np.sqrt(1.4 * p0 / rho1_ref); c2 = np.sqrt(1.01 * p0 / rho2_ref)
    Z1 = rho1_ref * c1; Z2 = rho2_ref * c2
    u_exact = np.full(N, u0); p_exact = np.full(N, p0)
    t_ret_L = t - x/c1
    t_ret_R = t - 0.5/c1 - (x - 0.5)/c2
    mask_L = (x < 0.5) & (t_ret_L > 0)
    mask_R = (x >= 0.5) & (t_ret_R > 0)
    u_exact[mask_L] = u0 + du * np.sin(2*np.pi*f*t_ret_L[mask_L])
    p_exact[mask_L] = p0 + Z1*du * np.sin(2*np.pi*f*t_ret_L[mask_L])
    u_exact[mask_R] = u0 + du * np.sin(2*np.pi*f*t_ret_R[mask_R])
    p_exact[mask_R] = p0 + Z2*du * np.sin(2*np.pi*f*t_ret_R[mask_R])
    exact = {'p_exact': p_exact, 'u_exact': u_exact,
             'rho1_exact': rho1, 'rho2_exact': rho2, 'a1_exact': a1}
    # Measure full amplitude (peak-to-peak) in each half excluding interface zone
    idx_L = (x > 0.2) & (x < 0.45)
    idx_R = (x > 0.55) & (x < 0.8)
    dp_L_pp = np.max(p_f[idx_L]) - np.min(p_f[idx_L])
    dp_R_pp = np.max(p_f[idx_R]) - np.min(p_f[idx_R])
    ratio = dp_R_pp / dp_L_pp if dp_L_pp > 0 else 0
    print(f'  Z1={Z1:.2f} Z2={Z2:.2f} dp_L_pp={dp_L_pp:.3f} dp_R_pp={dp_R_pp:.3f} ratio={ratio:.3f}', flush=True)
    _plot(x, a1_f, rho1_f, rho2_f, u_f, p_f,
          f'09 Impedance match v2 Z1={Z1:.0f} Z2={Z2:.0f} dp_R/dp_L={ratio:.2f}',
          f'{R}/cat_B_exact/09_impedance_matching_v2.png', exact=exact)


def run_06_wood_sound_speed():
    """Wood mixture sound speed: air-helium at α=0.5, measure wavelength from inlet wave."""
    print('[06: Wood mixture sound speed at α=0.5 (air-He)]', flush=True)
    ph1 = {'gamma': 1.4, 'pinf': 0, 'kv': 717.5}
    ph2 = {'gamma': 1.667, 'pinf': 0, 'kv': 3116}
    eos1 = IdealEOS(gamma=1.4, kv=717.5); eos2 = IdealEOS(gamma=1.667, kv=3116)
    N, L = 500, 1.0; dx = L/N
    x = np.linspace(dx/2, L-dx/2, N)
    p0, u0, T0, f = 1e5, 1.0, 300.0, 5000.0
    du = 0.01 * u0
    a1 = np.full(N, 0.5)  # uniform mixture
    rho1 = eos1.density(np.full(N, p0), np.full(N, T0))
    rho2 = eos2.density(np.full(N, p0), np.full(N, T0))
    u = np.full(N, u0); p = np.full(N, p0)
    a1r1 = a1 * rho1; a2r2 = (1 - a1) * rho2
    rho = a1r1 + a2r2; ru = rho * u
    e1 = eos1.energy(rho1, p); e2 = eos2.energy(rho2, p)
    rE = a1r1 * e1 + a2r2 * e2 + 0.5 * rho * u**2
    # Wood formula
    c1 = np.sqrt(1.4 * p0 / rho1[0]); c2 = np.sqrt(1.667 * p0 / rho2[0])
    rho_mix = 0.5 * rho1[0] + 0.5 * rho2[0]
    c_mix_wood = 1.0 / np.sqrt(rho_mix * (0.5/(rho1[0]*c1**2) + 0.5/(rho2[0]*c2**2)))
    print(f'  c_air={c1:.2f} c_He={c2:.2f} c_mix_Wood={c_mix_wood:.2f}', flush=True)
    def u_in(t_): return u0 + du * np.sin(2*np.pi*f*t_)
    t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
        ph1, ph2, a1r1, a2r2, ru, rE, a1.copy(),
        dx=dx, t_end=1.5e-3, cfl=0.4,
        bc_l='inlet', bc_r='transmissive', u_inlet_func=u_in,
        max_steps=10000, print_interval=5000,
        alpha_scheme='thinc_bvd', use_mmacm_ex=True)
    p_f, u_f, T_f, rho1_f, rho2_f, *_ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)
    # Exact with c_mix_wood
    t_ret = t - x/c_mix_wood; mask = t_ret > 0
    u_exact = np.full(N, u0); p_exact = np.full(N, p0)
    u_exact[mask] = u0 + du * np.sin(2*np.pi*f*t_ret[mask])
    p_exact[mask] = p0 + rho_mix*c_mix_wood*du * np.sin(2*np.pi*f*t_ret[mask])
    exact = {'p_exact': p_exact, 'u_exact': u_exact,
             'rho1_exact': rho1, 'rho2_exact': rho2, 'a1_exact': a1}
    # Measure wavelength from last few peaks
    p_wave = p_f - p0
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(p_wave, height=rho_mix*c_mix_wood*du*0.5)
    if len(peaks) >= 2:
        lam_meas = np.mean(np.diff(x[peaks]))
        c_mix_meas = lam_meas * f
    else:
        c_mix_meas = -1
    err_c = abs(c_mix_meas - c_mix_wood) / c_mix_wood if c_mix_meas > 0 else 1
    print(f'  c_mix_measured={c_mix_meas:.2f} err={err_c:.3e}', flush=True)
    _plot(x, a1_f, rho1_f, rho2_f, u_f, p_f,
          f'06 Wood c_mix  Wood={c_mix_wood:.1f}, meas={c_mix_meas:.1f}, err={err_c*100:.1f}%',
          f'{R}/cat_B_exact/06_wood_sound_speed.png', exact=exact)


def run_10_reflection_transmission():
    """Air-water interface pulse reflection/transmission."""
    print('[10: Reflection/transmission air-water]', flush=True)
    ph1 = {'gamma': 1.4, 'pinf': 0, 'kv': 717.5}
    ph2 = {'gamma': 4.4, 'pinf': 6e8, 'kv': 474.2}
    eos1 = IdealEOS(gamma=1.4, kv=717.5); eos2 = SGEOS(gamma=4.4, pinf=6e8, kv=474.2)
    N, L = 500, 1.0; dx = L/N
    x = np.linspace(dx/2, L-dx/2, N)
    p0, u0, T0, f = 1e5, 1.0, 300.0, 5000.0
    du = 0.02 * u0
    # air left, water right; interface at x=0.5
    a1 = np.where(x < 0.5, 1 - 1e-6, 1e-6)
    rho1 = eos1.density(np.full(N, p0), np.full(N, T0))
    rho2 = eos2.density(np.full(N, p0), np.full(N, T0))
    u = np.full(N, u0); p = np.full(N, p0)
    a1r1 = a1 * rho1; a2r2 = (1 - a1) * rho2
    rho = a1r1 + a2r2; ru = rho * u
    e1 = eos1.energy(rho1, p); e2 = eos2.energy(rho2, p)
    rE = a1r1 * e1 + a2r2 * e2 + 0.5 * rho * u**2
    # Single pulse: one period of sin then uniform
    T_period = 1.0/f
    def u_in(t_):
        if t_ < T_period:
            return u0 + du * np.sin(2*np.pi*f*t_ + 3*np.pi/2)
        else:
            return u0 - du
    t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
        ph1, ph2, a1r1, a2r2, ru, rE, a1.copy(),
        dx=dx, t_end=1.2e-3, cfl=0.4,
        bc_l='inlet', bc_r='transmissive', u_inlet_func=u_in,
        max_steps=20000, print_interval=5000,
        alpha_scheme='thinc_bvd', use_mmacm_ex=True)
    p_f, u_f, T_f, rho1_f, rho2_f, *_ = cons_to_prim(a1r1_f, a2r2_f, ru_f, rE_f, a1_f, ph1, ph2)
    # Acoustic impedances
    c_air = np.sqrt(1.4 * p0 / rho1[0])
    c_water = np.sqrt(4.4 * (p0 + 6e8) / rho2[0])
    Z_air = rho1[0] * c_air; Z_water = rho2[0] * c_water
    # Theory: transmission coef T = 2*Z_water / (Z_air + Z_water) ≈ 2
    T_theory = 2 * Z_water / (Z_air + Z_water)
    R_theory = (Z_water - Z_air) / (Z_air + Z_water)  # ≈ 1
    print(f'  Z_air={Z_air:.2f} Z_water={Z_water:.2e} T_theory={T_theory:.3f} R_theory={R_theory:.4f}', flush=True)
    # Measured: peak pressure amplitudes in air (reflected) and water (transmitted)
    dp_air = np.max(np.abs(p_f[x < 0.45] - p0))
    dp_water = np.max(np.abs(p_f[x > 0.55] - p0))
    dp_incident_theory = Z_air * du
    print(f'  dp_air_refl={dp_air:.2f} dp_water_trans={dp_water:.2f}', flush=True)
    # No exact overlay (complex time-dependent)
    exact = None
    _plot(x, a1_f, rho1_f, rho2_f, u_f, p_f,
          f'10 Refl/Trans dp_air={dp_air:.1f} dp_water={dp_water:.1f} (Z_air={Z_air:.0f}, Z_water={Z_water:.0e})',
          f'{R}/cat_B_exact/10_reflection_transmission.png', exact=exact)


CASES = {
    '01A': run_01A_abgrall_2phase,
    '01A_v2': run_01A_v2,
    '02': run_02_static_interface,
    '03': run_03_moving_contact,
    '03_v2': run_03_v2,
    '05': run_05_ultralow_mach_pulse,
    '06': run_06_wood_sound_speed,
    '07': run_07_air_acoustic_2000Hz,
    '09': run_09_impedance_matching,
    '09_v2': run_09_v2,
    '10': run_10_reflection_transmission,
}

if __name__ == '__main__':
    key = sys.argv[1]
    CASES[key]()
