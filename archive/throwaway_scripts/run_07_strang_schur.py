"""Case 07 comparison: im1 vs imex_5n_strang vs schur_5n.

R13 implementation validation driver.
Runs Case 07 (Air-Water / Helium-Air / Argon-Air) with three acoustic methods
and compares performance: profile quality, error metrics, wall time.

Output: results/run_07_strang_schur_compare.png + summary table.
"""
import os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/younglin90/work/claude_code/claudeCFD')
from solver.He2024.explicit_mmacm_ex import solve_IMEX, cons_to_prim
from solver.He2024.eos_general import IdealEOS, SGEOS

R = '/home/younglin90/work/claude_code/claudeCFD/results'
OUT = R
os.makedirs(OUT, exist_ok=True)

# ============================================================
# Case 07 EOS and sub-case definitions (mirrors run_01_07_validated.py)
# ============================================================
_CASE_07_EOS = {
    'air':    {'gamma': 1.4,   'pinf': 0.0,    'kv': 717.5,  'rho': 1.157, 'c': 347.8},
    'helium': {'gamma': 1.667, 'pinf': 0.0,    'kv': 2077.0, 'rho': 0.164, 'c': 1008.2},
    'argon':  {'gamma': 1.66,  'pinf': 0.0,    'kv': 208.0,  'rho': 1.748, 'c': 308.2},
    'water':  {'gamma': 4.1,   'pinf': 4.4e8,  'kv': 474.2,  'rho': 998.0, 'c': 1344.6},
}
_CASE_07_CASES = {
    1: dict(name='Air-Water',  left='air',    right='water',
            x_intf=0.5, x_src=0.1, sigma=0.014, t_end=1.63e-3),
    2: dict(name='Helium-Air', left='helium', right='air',
            x_intf=1.0, x_src=0.2, sigma=0.049, t_end=1.513e-3),
    3: dict(name='Argon-Air',  left='argon',  right='air',
            x_intf=0.5, x_src=0.1, sigma=0.038, t_end=2.02e-3),
}


def _build_eos_07(name):
    p = _CASE_07_EOS[name]
    if p['pinf'] > 0.0:
        return SGEOS(gamma=p['gamma'], pinf=p['pinf'], kv=p['kv'])
    return IdealEOS(gamma=p['gamma'], kv=p['kv'])


def _find_peak_pos(x_arr, p_arr, negative=False):
    if len(x_arr) == 0:
        return -999.0
    idx = np.argmin(p_arr) if negative else np.argmax(np.abs(p_arr))
    return float(x_arr[idx])


def _run_subcase(cid, acoustic_method, N=100, L=1.5, p0=1e5, u_peak=0.02):
    """Run one Case 07 sub-case with given acoustic_method. Returns metrics dict."""
    c = _CASE_07_CASES[cid]
    left, right = c['left'], c['right']
    eL, eR = _CASE_07_EOS[left], _CASE_07_EOS[right]
    rho_L, c_L = eL['rho'], eL['c']
    rho_R, c_R = eR['rho'], eR['c']
    Z_L = rho_L * c_L; Z_R = rho_R * c_R
    R_coef = (Z_R - Z_L) / (Z_R + Z_L)
    T_coef = 2.0 * Z_R / (Z_R + Z_L)
    T_pressure = 2.0 * Z_R / (Z_R + Z_L)

    dx = L / N
    x = np.linspace(dx/2, L - dx/2, N)
    x_intf = c['x_intf']; x_src = c['x_src']; sigma = c['sigma']
    t_end = c['t_end']

    ph1 = _CASE_07_EOS[left]; ph2 = _CASE_07_EOS[right]
    eos1 = _build_eos_07(left); eos2 = _build_eos_07(right)

    mask_L = x < x_intf
    u_pulse = u_peak * np.exp(-((x - x_src)**2) / (2*sigma**2))
    u_init = np.where(mask_L, u_pulse, 0.0)
    p_init = p0 + Z_L * u_init
    p_init = np.maximum(p_init, 1.0)
    eps_pure = 1e-6
    a1_init = np.where(mask_L, 1.0 - eps_pure, eps_pure)
    rho1 = np.full(N, rho_L); rho2 = np.full(N, rho_R)
    a1r1 = a1_init * rho1; a2r2 = (1.0 - a1_init) * rho2
    rho_mix = a1r1 + a2r2
    ru = rho_mix * u_init
    e1 = eos1.energy(rho1, p_init); e2 = eos2.energy(rho2, p_init)
    rE = a1r1*e1 + a2r2*e2 + 0.5*rho_mix*u_init**2

    t0 = time.time()
    t_f, ar1, ar2, ru_f, rE_f, a1_f = solve_IMEX(
        ph1, ph2, a1r1, a2r2, ru, rE, a1_init.copy(),
        dx=dx, t_end=t_end, cfl=0.4, use_material_cfl=False,
        bc_l='reflective', bc_r='transmissive',
        max_steps=50000, print_interval=999999,
        acoustic_method=acoustic_method, imex_rk2=False,
        time_integrator='strang')
    wall = time.time() - t0

    p_f, u_f, *_ = cons_to_prim(ar1, ar2, ru_f, rE_f, a1_f, ph1, ph2)
    rho_f = ar1 + ar2

    # Exact solution (d'Alembert)
    sigma_R = sigma * (c_R / c_L)
    t_intf_time = (x_intf - x_src) / c_L
    t_margin = t_f - t_intf_time
    x_peak_trans = x_intf + c_R * t_margin if t_margin > 0 else -1
    p_exact = np.full(N, p0); u_exact = np.zeros(N)
    if 0 < x_peak_trans < L:
        m_t = x >= x_intf
        p_exact[m_t] += T_coef * Z_L * u_peak * np.exp(
            -((x[m_t] - x_peak_trans)**2) / (2*sigma_R**2))
    exact = {'a1': a1_init, 'u': u_exact, 'p': p_exact,
             'rho': a1_init*rho1 + (1-a1_init)*rho2}

    ep = float(np.max(np.abs(p_f - p_exact))) / p0
    eu = float(np.max(np.abs(u_f - u_exact)))
    finite = bool(np.all(np.isfinite(p_f)))

    # Transmitted amplitude ratio check (±30%)
    x_trans_mask = x >= x_intf
    amp_ok = False
    if finite and t_f > t_intf_time and x_trans_mask.any():
        p_dev = p_f - p0
        dp_trans_num = np.max(np.abs(p_dev[x_trans_mask]))
        dp_incid = Z_L * u_peak
        amp_ratio = dp_trans_num / max(dp_incid, 1e-30)
        amp_ok = abs(amp_ratio - T_pressure) / max(abs(T_pressure), 0.1) < 0.30

    status = 'PASS' if (finite and ep < 2.0 and amp_ok) else 'FAIL'

    return dict(
        cid=cid, name=c['name'], method=acoustic_method,
        status=status, ep=ep, eu=eu, wall=wall,
        finite=finite, amp_ok=amp_ok,
        x=x, a1=a1_f, u=u_f, p=p_f, rho=rho_f,
        p_exact=p_exact, u_exact=u_exact, a1_exact=a1_init,
        rho_exact=exact['rho'])


def run_all():
    """Run all 3 sub-cases x 3 methods and produce comparison plot."""
    methods = ['im1', 'imex_5n_strang', 'schur_5n']
    method_labels = {
        'im1':             'IM1 (Peluchon)',
        'imex_5n_strang':  'imex_5n_strang (R13-A)',
        'schur_5n':        'schur_5n (R13-B)',
    }
    colors = {'im1': 'blue', 'imex_5n_strang': 'green', 'schur_5n': 'red'}

    all_results = []
    for method in methods:
        print(f'\n=== Method: {method} ===', flush=True)
        for cid in [1, 2, 3]:
            print(f'  sub-case {cid}: {_CASE_07_CASES[cid]["name"]}', flush=True)
            try:
                r = _run_subcase(cid, method)
                all_results.append(r)
                print(f'    {r["status"]} ep={r["ep"]:.3e} eu={r["eu"]:.3e} '
                      f'wall={r["wall"]:.2f}s amp_ok={r["amp_ok"]}', flush=True)
            except Exception as e:
                import traceback
                print(f'    ERROR: {e}', flush=True)
                traceback.print_exc()

    # ---- Comparison plot: 3 sub-cases × 3 methods, pressure profile ----
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle('Case 07: Reflection/Transmission — im1 vs imex_5n_strang vs schur_5n (R13)',
                 fontsize=13)

    for i_cid, cid in enumerate([1, 2, 3]):
        ax_row = axes[i_cid]
        for i_m, method in enumerate(methods):
            ax = ax_row[i_m]
            rr = [r for r in all_results if r['cid'] == cid and r['method'] == method]
            if not rr:
                ax.text(0.5, 0.5, 'ERROR', ha='center', transform=ax.transAxes)
                continue
            r = rr[0]
            ax.plot(r['x'], r['p'], '-', color=colors[method], lw=1.5, label='numerical')
            ax.plot(r['x'], r['p_exact'], 'k--', lw=1.0, label='exact (d\'Alembert)')
            ax.set_title(
                f'{r["name"]} | {method_labels[method]}\n'
                f'[{r["status"]}] ep={r["ep"]:.2e} wall={r["wall"]:.1f}s',
                fontsize=9)
            ax.set_xlabel('x [m]'); ax.set_ylabel('p [Pa]')
            ax.legend(fontsize=7); ax.grid(alpha=0.3)

    fig.tight_layout()
    out = f'{OUT}/run_07_strang_schur_compare.png'
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f'\nPlot saved: {out}', flush=True)

    # ---- Summary table ----
    print('\n' + '='*90)
    print(f'{"Method":<22} {"Sub-case":<14} {"Status":<8} '
          f'{"ep":>10} {"eu":>10} {"wall":>8} {"amp_ok":>8}')
    print('-'*90)
    for r in all_results:
        print(f'{r["method"]:<22} {r["name"]:<14} {r["status"]:<8} '
              f'{r["ep"]:>10.3e} {r["eu"]:>10.3e} {r["wall"]:>8.2f}s '
              f'{str(r["amp_ok"]):>8}')
    print('='*90)

    # Check regression: im1 PASS count should be >= 2/3
    im1_pass = sum(1 for r in all_results
                   if r['method'] == 'im1' and r['status'] == 'PASS')
    strang_pass = sum(1 for r in all_results
                      if r['method'] == 'imex_5n_strang' and r['status'] == 'PASS')
    schur_pass = sum(1 for r in all_results
                     if r['method'] == 'schur_5n' and r['status'] == 'PASS')
    print(f'\nim1:            {im1_pass}/3 PASS')
    print(f'imex_5n_strang: {strang_pass}/3 PASS  (R13-A)')
    print(f'schur_5n:       {schur_pass}/3 PASS  (R13-B)')

    overall = ('PASS' if strang_pass >= 2 and schur_pass >= 2
               else 'PARTIAL' if strang_pass >= 1 or schur_pass >= 1
               else 'FAIL')
    print(f'\nR13 overall: {overall}')
    return overall


if __name__ == '__main__':
    run_all()
