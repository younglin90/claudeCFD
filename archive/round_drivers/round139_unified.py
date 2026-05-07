"""Round 139: Tallois 2022 §3.2 θ-stage velocity post-correction.

Wraps round132 cases with a theta_post sweep for the argon-air 07 case.
Target: argon-air Liu 0.598 → < 0.5 (FAIL → PASS).
02-A regression guard: err_p must remain ~2.897e-13 at theta_post=0.0.

Ref: Tallois, Peluchon, Villedieu 2022 C&F 244 §3.2, Eq. 26.
"""
import sys, os, time, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from solver.He2024.explicit_mmacm_ex import solve_IMEX, cons_to_prim

# R139 default theta_post value (Tallois recommended starting point)
THETA_POST_DEFAULT = 0.2

UNIFIED_BASE = dict(
    cfl=0.9,
    time_integrator='auto',
    acoustic_method='auto',
    primitive_recon='auto',
    alpha_scheme='thinc_bvd',
    acid_interface=False,
    dissipation='none',
    strang_richardson=False,
    im1_theta=0.5,
    advective_flux='slau2',
)


def gauss(xx, x0, s):
    return np.exp(-(xx - x0) ** 2 / (2 * s ** 2))


# ===================== 02-A NASG (regression guard) =====================
def run_02A(theta_post=0.0):
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5}
    ph2 = {'gamma': 1.187, 'pinf': 7.028e8, 'kv': 3610.0,
           'b': 6.61e-4, 'eta': -1.177788e6}
    N = 10; L = 1.0; dx = L / N
    x = (np.arange(N) + 0.5) * dx
    p0 = 1e5; u0 = 1.0; T0 = 300.0
    a1_0 = np.where((x >= 0.4) & (x < 0.6), 1e-6, 1.0 - 1e-6)
    inv_rho2 = (ph2['gamma'] - 1) * ph2['kv'] * T0 / (p0 + ph2['pinf']) + ph2['b']
    rho2 = 1.0 / inv_rho2
    rho1 = p0 / ((ph1['gamma'] - 1) * ph1['kv'] * T0)
    a1r1 = a1_0 * rho1; a2r2 = (1 - a1_0) * rho2
    rho_init = a1r1 + a2r2; ru = rho_init * u0
    e1v = (p0 + ph1['gamma'] * ph1['pinf']) / (ph1['gamma'] - 1)
    br2 = ph2['b'] * rho2
    e2v = (p0 + ph2['gamma'] * ph2['pinf']) * (1 - br2) / (ph2['gamma'] - 1) + rho2 * ph2['eta']
    rE = a1_0 * e1v + (1 - a1_0) * e2v + 0.5 * rho_init * u0 * u0
    t0 = time.time()
    out = solve_IMEX(ph1, ph2, a1r1, a2r2, ru, rE, a1_0,
                     dx=dx, t_end=1.0, dt_fixed=0.01, use_material_cfl=False,
                     max_steps=200, bc_l='periodic', bc_r='periodic',
                     print_interval=99999, theta_post=theta_post,
                     **UNIFIED_BASE)
    wall = time.time() - t0
    tf, a1r1f, a2r2f, ruf, rEf, a1f = out
    pf, uf, *_ = cons_to_prim(a1r1f, a2r2f, ruf, rEf, a1f, ph1, ph2)
    ep = float(np.max(np.abs((pf - p0) / p0)))
    eu = float(np.max(np.abs(uf - u0)))
    fin = bool(np.all(np.isfinite(pf)))
    PASS = (ep < 1e-2 and eu < 1e-2 and fin)
    msg = (f'02-A θ={theta_post:.1f}: t={float(tf):.4f} ep={ep:.3e} eu={eu:.3e}'
           f' fin={fin} {"PASS" if PASS else "FAIL"} wall={wall:.2f}s')
    return msg, PASS, ep


# ===================== 07-B acoustic wave (LP-Strang) =====================
def run_07B(name, ph1c, ph2c, x_intf, sigma_L, x_src, t_end, theta_post=0.0):
    L = 1.5; N = 200; dx = L / N
    x = (np.arange(N) + 0.5) * dx
    u_peak = 0.02; p0 = 1e5
    a1 = np.where(x < x_intf, 1.0 - 1e-6, 1e-6)
    rho = a1 * ph1c['rho'] + (1 - a1) * ph2c['rho']
    u_init = u_peak * np.exp(-(x - x_src) ** 2 / (2 * sigma_L ** 2))
    a1r1 = a1 * ph1c['rho']; a2r2 = (1 - a1) * ph2c['rho']
    ru = rho * u_init
    e1 = (np.full_like(x, p0) + ph1c['gamma'] * ph1c['pinf']) / (ph1c['gamma'] - 1)
    e2 = (np.full_like(x, p0) + ph2c['gamma'] * ph2c['pinf']) / (ph2c['gamma'] - 1)
    rE = a1 * e1 + (1 - a1) * e2 + 0.5 * rho * u_init ** 2
    t0 = time.time()
    out = solve_IMEX(ph1c, ph2c, a1r1, a2r2, ru, rE, a1, dx=dx, t_end=t_end,
                     use_material_cfl=False,
                     max_steps=20000, bc_l='reflective', bc_r='transmissive',
                     print_interval=99999, theta_post=theta_post,
                     **UNIFIED_BASE)
    wall = time.time() - t0
    tf, a1r1f, a2r2f, ruf, rEf, a1f = out
    pf, uf, *_ = cons_to_prim(a1r1f, a2r2f, ruf, rEf, a1f, ph1c, ph2c)
    fin = bool(np.all(np.isfinite(pf)))
    cL = ph1c['c']; ZL = ph1c['rho'] * cL
    cR = ph2c['c']; ZR = ph2c['rho'] * cR
    R = (ZR - ZL) / (ZR + ZL); Tu = 2 * ZL / (ZL + ZR); Tp = 2 * ZR / (ZL + ZR)
    sigR = sigma_L * (cR / cL)
    t_intf_t = (x_intf - x_src) / cL
    left = x < x_intf
    u_in = u_peak * gauss(x, x_src + cL * float(tf), sigma_L) * left
    u_ref = u_peak * gauss(x, 2 * x_intf - x_src - cL * float(tf), sigma_L) * left
    u_tr = (u_peak * gauss(x, x_intf + cR * (float(tf) - t_intf_t), sigR) * (~left)
            if float(tf) > t_intf_t else np.zeros_like(x))
    dp = ZL * u_in + R * ZL * u_ref + Tp * ZL * u_tr
    u_ex = u_in - R * u_ref + Tu * u_tr
    p_ex = p0 + dp
    dp_w = ZL * u_peak
    L2p = float(np.sqrt(np.mean((pf - p_ex) ** 2)) / dp_w) if fin else float('inf')
    Lip = float(np.max(np.abs(pf - p_ex)) / dp_w) if fin else float('inf')
    L2u = float(np.sqrt(np.mean((uf - u_ex) ** 2)) / u_peak) if fin else float('inf')
    Liu = float(np.max(np.abs(uf - u_ex)) / u_peak) if fin else float('inf')
    PASS = (L2p < 0.30 and Lip < 0.50 and L2u < 0.30 and Liu < 0.50 and fin)
    msg = (f'07 {name:11s} θ={theta_post:.2f}: L2p={L2p:.3f} Lip={Lip:.3f}'
           f' L2u={L2u:.3f} Liu={Liu:.3f} fin={fin} {"PASS" if PASS else "FAIL"}'
           f' wall={wall:.1f}s')
    return msg, PASS, Lip, Liu


if __name__ == '__main__':
    EOS = {
        'air':   {'gamma': 1.4,   'pinf': 0.0,   'kv': 717.5,  'rho': 1.157,  'c': 347.8},
        'helium':{'gamma': 1.667, 'pinf': 0.0,   'kv': 2077.0, 'rho': 0.164,  'c': 1008.2},
        'argon': {'gamma': 1.66,  'pinf': 0.0,   'kv': 312.2,  'rho': 1.748,  'c': 308.2},
        'water': {'gamma': 4.1,   'pinf': 4.4e8, 'kv': 474.2,  'rho': 998.0,  'c': 1344.6},
    }
    os.makedirs('results', exist_ok=True)
    results = []

    # --- 02-A regression guard at theta_post=0 ---
    msg, p02, ep02 = run_02A(theta_post=0.0)
    results.append(msg); print(msg, flush=True)
    print(f'  [R139 guard] 02-A err_p at theta=0: {ep02:.6e} (expect ~2.897e-13)', flush=True)

    if not p02:
        print('02-A FAIL at theta=0 — ABORT', flush=True)
        sys.exit(1)

    # --- 07 baseline (theta_post=0) for all sub-cases ---
    print('\n--- 07 baseline (theta_post=0.0) ---', flush=True)
    for name, l, r, xi, sl, xs, te in [
        ('air-water',  'air',    'water', 0.5, 0.014, 0.1, 1.63e-3),
        ('helium-air', 'helium', 'air',   1.0, 0.049, 0.2, 1.513e-3),
        ('argon-air',  'argon',  'air',   0.5, 0.038, 0.1, 2.02e-3),
    ]:
        m, _, lip, liu = run_07B(name, EOS[l], EOS[r], xi, sl, xs, te, theta_post=0.0)
        results.append(m); print(m, flush=True)

    # --- argon-air theta_post sweep: {0.1, 0.2, 0.3, 0.4, 0.5} ---
    print('\n--- R139: argon-air θ-stage sweep ---', flush=True)
    sweep_rows = []
    for theta in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        m, pss, lip, liu = run_07B(
            'argon-air', EOS['argon'], EOS['air'],
            0.5, 0.038, 0.1, 2.02e-3,
            theta_post=theta)
        results.append(m); print(m, flush=True)
        sweep_rows.append((theta, lip, liu, pss))

    # --- Summary table ---
    print('\n=== R139 argon-air θ sweep summary ===', flush=True)
    print(f"{'theta':>6} {'Lip':>8} {'Liu':>8} {'PASS':>6}", flush=True)
    for theta, lip, liu, pss in sweep_rows:
        flag = 'PASS' if pss else 'FAIL'
        print(f"{theta:>6.2f} {lip:>8.4f} {liu:>8.4f} {flag:>6}", flush=True)

    # --- Plot: Lip and Liu vs theta ---
    thetas = [r[0] for r in sweep_rows]
    lips   = [r[1] for r in sweep_rows]
    lius   = [r[2] for r in sweep_rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thetas, lips, 'b-o', label='Lip (pressure)')
    ax.plot(thetas, lius, 'r-s', label='Liu (velocity)')
    ax.axhline(0.5, color='k', linestyle='--', label='PASS threshold 0.5')
    ax.set_xlabel('theta_post'); ax.set_ylabel('Error norm')
    ax.set_title('R139: argon-air θ-stage sweep — Lip & Liu vs theta_post')
    ax.legend(); ax.grid()
    plt.tight_layout()
    plt.savefig('results/round139_argon_theta_sweep.png', dpi=150)
    plt.close()
    print('Plot saved: results/round139_argon_theta_sweep.png', flush=True)

    with open('results/round139_results.txt', 'w') as f:
        f.write('\n'.join(results) + '\n')
    print('\nAll results written to results/round139_results.txt', flush=True)
