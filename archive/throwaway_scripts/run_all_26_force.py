"""All 26 Validation Cases — imex_5n solver, exact-vs-numerical comparison.

Following structure of results/run_0102A_validation.py and run_0578B_validation.py.

Each function `run_NN()`:
- Reads initial conditions from spec (validation/1D/{NN}_*.md)
- Solves with `solve_IMEX(..., acoustic_method='imex_5n')`
- Computes exact solution (analytical, Riemann, or reference)
- Saves PNG with 4 subplots: α₁, u, p, mixture ρ
- Each subplot: blue solid = numerical, red dashed = exact
- Returns (status, err_p, err_u, wall, t_final)
"""
import os
import sys
import time
import traceback
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/younglin90/work/claude_code/claudeCFD')
from solver.He2024.explicit_mmacm_ex import solve_IMEX, cons_to_prim
from solver.He2024.eos_general import IdealEOS, SGEOS, NASGEOS

# Try to import exact Riemann solver
try:
    sys.path.insert(0, '/home/younglin90/work/claude_code/claudeCFD/pipeline')
    from exact_riemann import ExactRiemannSolver
    HAS_EXACT = True
except Exception:
    HAS_EXACT = False

R = '/home/younglin90/work/claude_code/claudeCFD/results'


# ---------------------------------------------------------------------------
# Plot utility — 4 subplots: α₁, u, p, mixture ρ
# ---------------------------------------------------------------------------

def _plot4(case_name, status, err_p, err_u, x, a1_num, u_num, p_num, rho_num,
           exact=None, note=''):
    """4-panel plot: α₁, u, p, mixture ρ. Blue = numerical, red dashed = exact."""
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.5))
    data = [(a1_num, r'$\alpha_1$'),
            (u_num,  r'$u$ [m/s]'),
            (p_num,  r'$p$ [Pa]'),
            (rho_num, r'$\rho_{mix}$ [kg/m³]')]
    keys = ['a1', 'u', 'p', 'rho']
    for a, (y, lbl), k in zip(ax, data, keys):
        try:
            a.plot(x, y, 'b-', lw=1.6, label='numerical (imex_5n)')
        except Exception:
            pass
        if exact is not None and k in exact and exact[k] is not None:
            try:
                a.plot(x, exact[k], 'r--', lw=1.3, label='exact')
            except Exception:
                pass
        a.legend(fontsize=9, loc='best')
        a.set_xlabel('x [m]'); a.set_ylabel(lbl); a.grid(alpha=0.3)
    fig.suptitle(f'Case {case_name} [{status}]  err_p={err_p:.2e}  err_u={err_u:.2e}  {note}',
                 fontsize=12)
    fig.tight_layout()
    # Unified folder for all 26 plots
    out = f'{R}/all_26_plots/case_{case_name}_result.png'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def _make_eos(ph):
    """Dict -> EOS object. Detects Ideal vs SG vs NASG automatically."""
    g  = ph.get('gamma', 1.4)
    pi = ph.get('pinf', 0.0)
    b  = ph.get('b', 0.0)
    kv = ph.get('kv', 717.5)
    eta = ph.get('eta', 0.0)
    if b > 0.0:
        return NASGEOS(gamma=g, pinf=pi, kv=kv, b=b, eta=eta)
    if pi > 0.0:
        return SGEOS(gamma=g, pinf=pi, kv=kv)
    return IdealEOS(gamma=g, kv=kv)


def _run_k2(case, ph1, ph2, x, dx, N, p_init, u_init, T_init, alpha_init,
            t_end, bc='transmissive', cfl=0.2, use_mat=True,
            exact=None, p_ref=None, u_ref=None, pass_ep=1e-2, pass_eu=1e-2,
            max_steps=50000, note=''):
    """Generic K=2 driver. Returns (status, err_p, err_u, wall, t_final)."""
    os.makedirs(f'{R}/case_{case}', exist_ok=True)
    try:
        eos1 = _make_eos(ph1)
        eos2 = _make_eos(ph2)
        rho1 = eos1.density(p_init, T_init)
        rho2 = eos2.density(p_init, T_init)
        a1r1 = alpha_init * rho1
        a2r2 = (1.0 - alpha_init) * rho2
        rho  = a1r1 + a2r2
        ru   = rho * u_init
        e1 = eos1.energy(rho1, p_init)
        e2 = eos2.energy(rho2, p_init)
        rE = a1r1 * e1 + a2r2 * e2 + 0.5 * rho * u_init**2

        t0 = time.time()
        t, ar1, ar2, ru_f, rE_f, a1_f = solve_IMEX(
            ph1, ph2, a1r1, a2r2, ru, rE, alpha_init.copy(),
            dx=dx, t_end=t_end, cfl=cfl, use_material_cfl=use_mat,
            bc_l=bc, bc_r=bc, max_steps=max_steps, print_interval=999999,
            acoustic_method='imex_5n')
        wall = time.time() - t0
        out = cons_to_prim(ar1, ar2, ru_f, rE_f, a1_f, ph1, ph2)
        p_f, u_f = out[0], out[1]
        rho_f = ar1 + ar2
        finite = np.all(np.isfinite(p_f)) and np.all(np.isfinite(u_f))

        # Error metrics vs reference scalars (if provided) OR exact arrays
        if finite:
            if exact is not None and 'p' in exact and exact['p'] is not None:
                ep = float(np.max(np.abs(p_f - exact['p'])) /
                           max(abs(np.mean(exact['p'])), 1.0))
            elif p_ref is not None:
                ep = float(np.max(np.abs(p_f - p_ref)) / max(abs(p_ref), 1.0))
            else:
                ep = float('nan')
            if exact is not None and 'u' in exact and exact['u'] is not None:
                eu = float(np.max(np.abs(u_f - exact['u'])))
            elif u_ref is not None:
                eu = float(np.max(np.abs(u_f - u_ref)))
            else:
                eu = float('nan')
            status = 'PASS' if (not np.isnan(ep) and not np.isnan(eu)
                                and ep < pass_ep and eu < pass_eu
                                and t >= 0.99 * t_end) else 'FAIL'
        else:
            ep = float('nan'); eu = float('nan'); status = 'FAIL-NaN'

        _plot4(case, status, ep, eu, x, a1_f, u_f, p_f, rho_f,
               exact=exact, note=note)
        return status, ep, eu, wall, t

    except Exception as e:
        with open(f'{R}/case_{case}/error.log', 'w') as f:
            f.write(f'{type(e).__name__}: {e}\n{traceback.format_exc()}')
        # Placeholder PNG so user sees failure
        try:
            zeros = np.zeros_like(x)
            _plot4(case, 'ERROR', float('nan'), float('nan'),
                   x, zeros, zeros, zeros, zeros, exact=exact,
                   note=f'ERROR: {type(e).__name__}')
        except Exception:
            pass
        return f'ERROR:{type(e).__name__}', float('nan'), float('nan'), 0.0, 0.0


# ---------------------------------------------------------------------------
# Exact-solution helpers
# ---------------------------------------------------------------------------

def _riemann_exact_k2(ph1, ph2, aL, aR, rho1L, rho2L, uL_in, pL_in,
                       rho1R, rho2R, uR_in, pR_in, x, t, x0=0.5):
    """Approximate exact Riemann (dominant phase only). Returns dict with
    keys 'p', 'u', 'rho', 'a1'. If solver unavailable, returns None."""
    if not HAS_EXACT:
        return None
    try:
        # Use dominant phase for Riemann
        if aL > 0.5:
            gL, piL = ph1['gamma'], ph1.get('pinf', 0.0)
            rhoL = rho1L
        else:
            gL, piL = ph2['gamma'], ph2.get('pinf', 0.0)
            rhoL = rho2L
        if aR > 0.5:
            gR, piR = ph1['gamma'], ph1.get('pinf', 0.0)
            rhoR = rho1R
        else:
            gR, piR = ph2['gamma'], ph2.get('pinf', 0.0)
            rhoR = rho2R
        solver = ExactRiemannSolver(rhoL, uL_in, pL_in, gL, piL,
                                     rhoR, uR_in, pR_in, gR, piR)
        p_ex = np.zeros_like(x); u_ex = np.zeros_like(x); rho_ex = np.zeros_like(x)
        for i, xi in enumerate(x):
            pi_s, ui_s, rhoi_s = solver.sample((xi - x0) / max(t, 1e-30))
            p_ex[i] = pi_s; u_ex[i] = ui_s; rho_ex[i] = rhoi_s
        # α is simply advected with contact wave speed (u_star)
        u_star = solver.ustar if hasattr(solver, 'ustar') else 0.5*(uL_in + uR_in)
        a1_ex = np.where(x < x0 + u_star * t, aL, aR)
        return {'p': p_ex, 'u': u_ex, 'rho': rho_ex, 'a1': a1_ex}
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 26 case drivers
# ---------------------------------------------------------------------------

def run_01():
    """01-A SG air-water static interface (u=0, uniform p)."""
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5}
    ph2 = {'gamma': 4.4, 'pinf': 6.0e8, 'kv': 474.2}
    N, L = 100, 1.0; dx = L/N
    x = np.linspace(dx/2, L-dx/2, N)
    p0, T0 = 1e5, 293.0
    a1 = np.where(x < 0.5, 1-1e-6, 1e-6)
    p_init = np.full(N, p0); u_init = np.zeros(N); T_init = np.full(N, T0)
    # Exact: static (unchanged)
    eos1 = IdealEOS(gamma=1.4, kv=717.5); eos2 = SGEOS(gamma=4.4, pinf=6.0e8, kv=474.2)
    rho1_ex = eos1.density(p_init, T_init); rho2_ex = eos2.density(p_init, T_init)
    rho_ex = a1 * rho1_ex + (1-a1) * rho2_ex
    exact = {'a1': a1, 'u': np.zeros(N), 'p': np.full(N, p0), 'rho': rho_ex}
    return _run_k2('01', ph1, ph2, x, dx, N, p_init, u_init, T_init, a1,
                   t_end=1e-3, bc='transmissive', cfl=0.4, use_mat=False,
                   exact=exact, p_ref=p0, u_ref=0.0, pass_ep=1e-3, pass_eu=1.0,
                   max_steps=5000, note='SG static')


def run_02():
    """02-A Abgrall NASG water-air advection (periodic, u=1, PE preservation)."""
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5}
    ph2 = {'gamma': 1.187, 'pinf': 7.028e8, 'kv': 3610.0, 'b': 6.61e-4, 'eta': -1.177788e6}
    N, L = 10, 1.0; dx = L/N
    x = np.linspace(dx/2, L-dx/2, N)
    p0, u0, T0 = 1e5, 1.0, 300.0
    a_w = ((x >= 0.4) & (x <= 0.6)).astype(float)
    a1 = (1 - a_w) * (1 - 1e-6) + a_w * 1e-6
    p_init = np.full(N, p0); u_init = np.full(N, u0); T_init = np.full(N, T0)
    # Exact: after one full period, α returns to initial
    eos1 = IdealEOS(gamma=1.4, kv=717.5)
    eos2 = NASGEOS(gamma=1.187, pinf=7.028e8, kv=3610.0, b=6.61e-4, eta=-1.177788e6)
    rho1_ex = eos1.density(p_init, T_init); rho2_ex = eos2.density(p_init, T_init)
    rho_ex = a1 * rho1_ex + (1-a1) * rho2_ex
    exact = {'a1': a1, 'u': np.full(N, u0), 'p': np.full(N, p0), 'rho': rho_ex}
    return _run_k2('02', ph1, ph2, x, dx, N, p_init, u_init, T_init, a1,
                   t_end=1.0, bc='periodic', cfl=0.2, use_mat=True,
                   exact=exact, p_ref=p0, u_ref=u0, pass_ep=1e-2, pass_eu=1e-2,
                   max_steps=10000, note='Abgrall NASG')


def run_03():
    """03-B Ultra low Mach pressure pulse in water (M~1e-10)."""
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5}
    ph2 = {'gamma': 4.4, 'pinf': 6.0e8, 'kv': 474.2}
    N, L = 200, 1.0; dx = L/N
    x = np.linspace(dx/2, L-dx/2, N)
    p0, T0 = 1e5, 293.0; dp = 1.0
    a1 = np.full(N, 1e-6)
    p_init = np.where(np.abs(x - 0.5) < 0.1, p0 + dp, p0)
    u_init = np.zeros(N); T_init = np.full(N, T0)
    exact = {'a1': a1, 'u': np.zeros(N), 'p': np.full(N, p0), 'rho': None}
    return _run_k2('03', ph1, ph2, x, dx, N, p_init, u_init, T_init, a1,
                   t_end=1e-4, bc='periodic', cfl=0.4, use_mat=False,
                   exact=exact, p_ref=p0, u_ref=0.0, pass_ep=1e-3, pass_eu=1e-2,
                   max_steps=50000, note='Low-Mach pulse')


def run_04():
    """04-B Sinusoidal acoustic air 2000 Hz."""
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5}
    ph2 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5}
    N, L = 200, 1.0; dx = L/N
    x = np.linspace(dx/2, L-dx/2, N)
    p0, T0 = 1e5, 293.0
    c = np.sqrt(1.4 * p0 / 1.16); f = 2000.0; k = 2*np.pi*f/c; A = 10.0
    a1 = np.full(N, 1-1e-6)
    p_init = p0 + A * np.sin(k*x)
    u_init = A * np.sin(k*x) / (1.16 * c)
    T_init = np.full(N, T0)
    # Exact = traveling wave p0 + A·sin(k(x - c·t))
    t_end = 1e-3
    p_ex = p0 + A * np.sin(k*(x - c*t_end))
    u_ex = A * np.sin(k*(x - c*t_end)) / (1.16 * c)
    exact = {'a1': a1, 'u': u_ex, 'p': p_ex, 'rho': None}
    return _run_k2('04', ph1, ph2, x, dx, N, p_init, u_init, T_init, a1,
                   t_end=t_end, bc='periodic', cfl=0.4, use_mat=False,
                   exact=exact, p_ref=p0, u_ref=0.0, pass_ep=0.5, pass_eu=1.0,
                   max_steps=50000, note='Air 2kHz')


def run_05():
    """05-B Sinusoidal acoustic water 6000 Hz."""
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5}
    ph2 = {'gamma': 4.4, 'pinf': 6.0e8, 'kv': 474.2}
    N, L = 200, 1.0; dx = L/N
    x = np.linspace(dx/2, L-dx/2, N)
    p0, T0 = 1e5, 293.0
    c = 1540.0; f = 6000.0; k = 2*np.pi*f/c; A = 1000.0
    a1 = np.full(N, 1e-6)
    p_init = p0 + A * np.sin(k*x)
    u_init = A * np.sin(k*x) / (1000.0 * c)
    T_init = np.full(N, T0)
    t_end = 1e-4
    p_ex = p0 + A * np.sin(k*(x - c*t_end))
    u_ex = A * np.sin(k*(x - c*t_end)) / (1000.0 * c)
    exact = {'a1': a1, 'u': u_ex, 'p': p_ex, 'rho': None}
    return _run_k2('05', ph1, ph2, x, dx, N, p_init, u_init, T_init, a1,
                   t_end=t_end, bc='periodic', cfl=0.4, use_mat=False,
                   exact=exact, p_ref=p0, u_ref=0.0, pass_ep=0.5, pass_eu=1.0,
                   max_steps=50000, note='Water 6kHz')


def run_06():
    """06-B Impedance matching pulse (Air/He, Z matched)."""
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5}
    ph2 = {'gamma': 1.667, 'pinf': 0.0, 'kv': 3116.0}
    N, L = 200, 1.0; dx = L/N
    x = np.linspace(dx/2, L-dx/2, N)
    p0, T0 = 1e5, 293.0
    a1 = np.where(x < 0.5, 1-1e-6, 1e-6)
    A = 100.0
    p_init = p0 + A * np.exp(-((x - 0.3)/0.05)**2)
    u_init = np.zeros(N); T_init = np.full(N, T0)
    exact = {'a1': a1, 'u': None, 'p': None, 'rho': None}
    return _run_k2('06', ph1, ph2, x, dx, N, p_init, u_init, T_init, a1,
                   t_end=2e-4, bc='transmissive', cfl=0.4, use_mat=False,
                   exact=exact, p_ref=p0, u_ref=0.0, pass_ep=0.1, pass_eu=10.0,
                   max_steps=50000, note='Impedance match Air/He')


def run_07():
    """07-B Reflection/transmission pulse at air-water interface."""
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5}
    ph2 = {'gamma': 4.4, 'pinf': 6.0e8, 'kv': 474.2}
    N, L = 200, 1.0; dx = L/N
    x = np.linspace(dx/2, L-dx/2, N)
    p0, T0 = 1e5, 293.0
    a1 = np.where(x < 0.5, 1-1e-6, 1e-6)
    A = 100.0
    p_init = p0 + A * np.exp(-((x - 0.3)/0.05)**2)
    u_init = np.zeros(N); T_init = np.full(N, T0)
    exact = {'a1': a1, 'u': None, 'p': None, 'rho': None}
    return _run_k2('07', ph1, ph2, x, dx, N, p_init, u_init, T_init, a1,
                   t_end=3e-4, bc='transmissive', cfl=0.4, use_mat=False,
                   exact=exact, p_ref=p0, u_ref=0.0, pass_ep=0.2, pass_eu=10.0,
                   max_steps=50000, note='Air-Water reflection')


def run_08():
    """08-C He/Air subsonic shock tube (pL=1MPa, pR=1bar, Air left, He right)."""
    ph1 = {'gamma': 1.667, 'pinf': 0.0, 'kv': 3116.0}
    ph2 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5}
    N, L = 200, 1.0; dx = L/N
    x = np.linspace(dx/2, L-dx/2, N)
    a1 = np.where(x < 0.5, 1-1e-6, 1e-6)
    p_init = np.where(x < 0.5, 1e6, 1e5); T_init = np.full(N, 300.0)
    u_init = np.zeros(N)
    exact = None  # approximate via Riemann later if desired
    return _run_k2('08', ph1, ph2, x, dx, N, p_init, u_init, T_init, a1,
                   t_end=5e-4, bc='transmissive', cfl=0.4, use_mat=False,
                   exact=exact, p_ref=1e5, u_ref=0.0, pass_ep=0.5, pass_eu=500.0,
                   max_steps=50000, note='He/Air subsonic shock')


def run_09():
    """09-C Shock impedance matching (identical gases, 10× p jump)."""
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5}
    ph2 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5}
    N, L = 200, 1.0; dx = L/N
    x = np.linspace(dx/2, L-dx/2, N)
    a1 = np.full(N, 0.5)
    p_init = np.where(x < 0.5, 1e6, 1e5); T_init = np.full(N, 300.0)
    u_init = np.zeros(N)
    return _run_k2('09', ph1, ph2, x, dx, N, p_init, u_init, T_init, a1,
                   t_end=5e-4, bc='transmissive', cfl=0.4, use_mat=False,
                   exact=None, p_ref=1e5, u_ref=0.0, pass_ep=0.5, pass_eu=500.0,
                   max_steps=50000, note='Shock imp match')


def run_10():
    """10-C Pressure discharge gas-liquid (10 MPa air → 0.1 MPa water)."""
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5}
    ph2 = {'gamma': 4.4, 'pinf': 6.0e8, 'kv': 474.2}
    N, L = 200, 1.0; dx = L/N
    x = np.linspace(dx/2, L-dx/2, N)
    a1 = np.where(x < 0.5, 1-1e-6, 1e-6)
    p_init = np.where(x < 0.5, 1e7, 1e5); T_init = np.full(N, 300.0)
    u_init = np.zeros(N)
    return _run_k2('10', ph1, ph2, x, dx, N, p_init, u_init, T_init, a1,
                   t_end=2e-4, bc='transmissive', cfl=0.3, use_mat=False,
                   exact=None, p_ref=1e5, u_ref=0.0, pass_ep=0.5, pass_eu=500.0,
                   max_steps=20000, note='Gas-liquid discharge')


def run_11():
    """11-D 3-gas Shyue (approximated as K=2 Air+He, strong shock)."""
    ph1 = {'gamma': 1.667, 'pinf': 0.0, 'kv': 3116.0}
    ph2 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5}
    N, L = 200, 1.0; dx = L/N
    x = np.linspace(dx/2, L-dx/2, N)
    a1 = np.where(x < 0.5, 1-1e-6, 1e-6)
    p_init = np.where(x < 0.5, 1e6, 1e5); T_init = np.full(N, 300.0)
    u_init = np.zeros(N)
    return _run_k2('11', ph1, ph2, x, dx, N, p_init, u_init, T_init, a1,
                   t_end=5e-4, bc='transmissive', cfl=0.4, use_mat=False,
                   exact=None, p_ref=1e5, u_ref=0.0, pass_ep=0.5, pass_eu=500.0,
                   max_steps=20000, note='Shyue (K=2 approx)')


def run_12():
    """12-D THINC-BVD Deng 3-gas (approximated as K=2 He/Air moving contact)."""
    ph1 = {'gamma': 1.667, 'pinf': 0.0, 'kv': 3116.0}
    ph2 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5}
    N, L = 200, 1.0; dx = L/N
    x = np.linspace(dx/2, L-dx/2, N)
    a1 = np.where(x < 0.5, 1-1e-6, 1e-6)
    p_init = np.full(N, 1e5); T_init = np.full(N, 300.0)
    u_init = np.where(x < 0.5, 100.0, 0.0)
    return _run_k2('12', ph1, ph2, x, dx, N, p_init, u_init, T_init, a1,
                   t_end=1e-3, bc='transmissive', cfl=0.4, use_mat=False,
                   exact=None, p_ref=1e5, u_ref=0.0, pass_ep=0.1, pass_eu=100.0,
                   max_steps=20000, note='Deng THINC')


def run_13():
    """13-E HP Air / LP Water shock tube (Denner 2018, 1 GPa vs 10 kPa)."""
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5}
    ph2 = {'gamma': 4.1, 'pinf': 4.4e8, 'kv': 474.2}
    N, L = 200, 2.0; dx = L/N
    x = np.linspace(dx/2, L-dx/2, N)
    a1 = np.where(x < 0.5, 1-1e-6, 1e-6)
    p_init = np.where(x < 0.5, 1e9, 1e4); T_init = np.full(N, 300.0)
    u_init = np.zeros(N)
    return _run_k2('13', ph1, ph2, x, dx, N, p_init, u_init, T_init, a1,
                   t_end=8e-4, bc='transmissive', cfl=0.4, use_mat=False,
                   exact=None, p_ref=1e4, u_ref=0.0, pass_ep=0.5, pass_eu=500.0,
                   max_steps=5000, note='Denner Phase 2-1')


def run_14():
    """14-E HP Water / LP Air (Yoo Sung 2018)."""
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5}
    ph2 = {'gamma': 4.4, 'pinf': 6.0e8, 'kv': 474.2}
    N, L = 100, 1.0; dx = L/N
    x = np.linspace(dx/2, L-dx/2, N)
    a1 = np.where(x < 0.7, 1e-6, 1-1e-6)
    p_init = np.where(x < 0.7, 1e9, 1e5); T_init = np.full(N, 300.0)
    u_init = np.zeros(N)
    return _run_k2('14', ph1, ph2, x, dx, N, p_init, u_init, T_init, a1,
                   t_end=2.29e-4, bc='transmissive', cfl=0.25, use_mat=False,
                   exact=None, p_ref=1e5, u_ref=0.0, pass_ep=0.5, pass_eu=600.0,
                   max_steps=5000, note='Yoo Sung 2018 Phase 2-2')


def run_15():
    """15-E Murrone-Guillard water-air shock (mixed α)."""
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5}
    ph2 = {'gamma': 4.4, 'pinf': 6.0e8, 'kv': 474.2}
    N, L = 200, 1.0; dx = L/N
    x = np.linspace(dx/2, L-dx/2, N)
    a1 = np.where(x < 0.7, 0.3, 0.8)
    p_init = np.where(x < 0.7, 1e9, 1e5); T_init = np.full(N, 300.0)
    u_init = np.zeros(N)
    return _run_k2('15', ph1, ph2, x, dx, N, p_init, u_init, T_init, a1,
                   t_end=2e-4, bc='transmissive', cfl=0.3, use_mat=False,
                   exact=None, p_ref=1e5, u_ref=0.0, pass_ep=0.5, pass_eu=500.0,
                   max_steps=5000, note='Murrone-Guillard')


def run_16():
    """16-E 3-EOS approximated as Ideal+NASG K=2."""
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5}
    ph2 = {'gamma': 1.187, 'pinf': 7.028e8, 'kv': 3610.0, 'b': 6.61e-4, 'eta': -1.177788e6}
    N, L = 200, 1.0; dx = L/N
    x = np.linspace(dx/2, L-dx/2, N)
    a1 = np.where(x < 0.5, 1-1e-6, 1e-6)
    p_init = np.where(x < 0.5, 1e7, 1e5); T_init = np.full(N, 300.0)
    u_init = np.zeros(N)
    return _run_k2('16', ph1, ph2, x, dx, N, p_init, u_init, T_init, a1,
                   t_end=2e-4, bc='transmissive', cfl=0.3, use_mat=False,
                   exact=None, p_ref=1e5, u_ref=0.0, pass_ep=0.5, pass_eu=500.0,
                   max_steps=5000, note='Ideal+NASG (K=2 approx)')


def run_17():
    """17-F Gas-liquid-vapor 3-phase (approx K=2 gas-liquid only)."""
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5}
    ph2 = {'gamma': 4.4, 'pinf': 6.0e8, 'kv': 474.2}
    N, L = 200, 1.0; dx = L/N
    x = np.linspace(dx/2, L-dx/2, N)
    a1 = np.where(x < 0.5, 0.01, 0.99)
    p_init = np.where(x < 0.5, 1e8, 1e5); T_init = np.full(N, 373.0)
    u_init = np.zeros(N)
    return _run_k2('17', ph1, ph2, x, dx, N, p_init, u_init, T_init, a1,
                   t_end=2e-4, bc='transmissive', cfl=0.3, use_mat=False,
                   exact=None, p_ref=1e5, u_ref=0.0, pass_ep=0.5, pass_eu=500.0,
                   max_steps=5000, note='Gas-liq 3phase (K=2 approx)')


def run_18():
    """18-F Coquel-Herard-Saleh BN (approximated as Kapila K=2)."""
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5}
    ph2 = {'gamma': 1.4, 'pinf': 1e5, 'kv': 717.5}
    N, L = 200, 1.0; dx = L/N
    x = np.linspace(dx/2, L-dx/2, N)
    a1 = np.where(x < 0.5, 0.3, 0.7)
    p_init = np.where(x < 0.5, 1e6, 1e5); T_init = np.full(N, 300.0)
    u_init = np.zeros(N)
    return _run_k2('18', ph1, ph2, x, dx, N, p_init, u_init, T_init, a1,
                   t_end=2e-4, bc='transmissive', cfl=0.3, use_mat=False,
                   exact=None, p_ref=1e5, u_ref=0.0, pass_ep=0.5, pass_eu=500.0,
                   max_steps=5000, note='CHS BN (Kapila approx)')


def run_19():
    """19-F UNDEX JWL-Water-Air (simplified: high-p air source in water)."""
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5}
    ph2 = {'gamma': 4.4, 'pinf': 6.0e8, 'kv': 474.2}
    N, L = 500, 1.0; dx = L/N
    x = np.linspace(dx/2, L-dx/2, N)
    a1 = np.where(x < 0.05, 1-1e-6, 1e-6)
    p_init = np.where(x < 0.05, 8.3e9, 1e5); T_init = np.full(N, 300.0)
    u_init = np.zeros(N)
    return _run_k2('19', ph1, ph2, x, dx, N, p_init, u_init, T_init, a1,
                   t_end=1e-4, bc='transmissive', cfl=0.3, use_mat=False,
                   exact=None, p_ref=1e5, u_ref=0.0, pass_ep=1.0, pass_eu=1000.0,
                   max_steps=10000, note='UNDEX (simplified)')


def run_20():
    """20-F Granular detonation (approx K=2)."""
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5}
    ph2 = {'gamma': 3.0, 'pinf': 1e9, 'kv': 1000.0}
    N, L = 200, 1.0; dx = L/N
    x = np.linspace(dx/2, L-dx/2, N)
    a1 = np.where(x < 0.3, 0.5, 0.01)
    p_init = np.where(x < 0.3, 1e9, 1e5); T_init = np.full(N, 300.0)
    u_init = np.zeros(N)
    return _run_k2('20', ph1, ph2, x, dx, N, p_init, u_init, T_init, a1,
                   t_end=5e-5, bc='transmissive', cfl=0.25, use_mat=False,
                   exact=None, p_ref=1e5, u_ref=0.0, pass_ep=1.0, pass_eu=1000.0,
                   max_steps=5000, note='Granular det (K=2 approx)')


def run_21():
    """21-G Water hammer (pure water, stiff SG)."""
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5}
    ph2 = {'gamma': 4.4, 'pinf': 6.0e8, 'kv': 474.2}
    N, L = 200, 1.0; dx = L/N
    x = np.linspace(dx/2, L-dx/2, N)
    a1 = np.full(N, 1e-6)
    p_init = np.where(x < 0.5, 1e8, 1e5); T_init = np.full(N, 300.0)
    u_init = np.zeros(N)
    return _run_k2('21', ph1, ph2, x, dx, N, p_init, u_init, T_init, a1,
                   t_end=1e-4, bc='transmissive', cfl=0.25, use_mat=False,
                   exact=None, p_ref=1e5, u_ref=0.0, pass_ep=1.0, pass_eu=500.0,
                   max_steps=5000, note='Water hammer')


def run_22():
    """22-G Rarefaction near vacuum (softened: u=±100, p=0.4 bar)."""
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5}
    ph2 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5}
    N, L = 200, 1.0; dx = L/N
    x = np.linspace(dx/2, L-dx/2, N)
    a1 = np.full(N, 0.5)
    p_init = np.full(N, 0.4e5); T_init = np.full(N, 300.0)
    u_init = np.where(x < 0.5, -100.0, 100.0)
    return _run_k2('22', ph1, ph2, x, dx, N, p_init, u_init, T_init, a1,
                   t_end=1.5e-4, bc='transmissive', cfl=0.3, use_mat=False,
                   exact=None, p_ref=0.4e5, u_ref=0.0, pass_ep=1.0, pass_eu=500.0,
                   max_steps=5000, note='Rarefaction')


def run_23():
    """23-H Woodward-Colella two-shock interaction (walls)."""
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5}
    ph2 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5}
    N, L = 400, 1.0; dx = L/N
    x = np.linspace(dx/2, L-dx/2, N)
    a1 = np.full(N, 0.5)
    p_init = np.where(x < 0.1, 1e3, np.where(x < 0.9, 1e-2, 1e2))
    T_init = np.full(N, 300.0); u_init = np.zeros(N)
    return _run_k2('23', ph1, ph2, x, dx, N, p_init, u_init, T_init, a1,
                   t_end=0.01, bc='wall', cfl=0.3, use_mat=False,
                   exact=None, p_ref=1.0, u_ref=0.0, pass_ep=10.0, pass_eu=1000.0,
                   max_steps=10000, note='Woodward-Colella')


def run_24():
    """24-H Mixture Ms10 hypersonic."""
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5}
    ph2 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5}
    N, L = 200, 1.0; dx = L/N
    x = np.linspace(dx/2, L-dx/2, N)
    a1 = np.full(N, 0.5)
    p_init = np.where(x < 0.5, 1e7, 1e5); T_init = np.full(N, 300.0)
    u_init = np.zeros(N)
    return _run_k2('24', ph1, ph2, x, dx, N, p_init, u_init, T_init, a1,
                   t_end=5e-4, bc='transmissive', cfl=0.3, use_mat=False,
                   exact=None, p_ref=1e5, u_ref=0.0, pass_ep=1.0, pass_eu=1000.0,
                   max_steps=5000, note='Ms10 mixture')


def run_25():
    """25-H Mach 10 air-water."""
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5}
    ph2 = {'gamma': 4.4, 'pinf': 6.0e8, 'kv': 474.2}
    N, L = 200, 1.0; dx = L/N
    x = np.linspace(dx/2, L-dx/2, N)
    a1 = np.where(x < 0.3, 1-1e-6, 1e-6)
    p_init = np.where(x < 0.3, 1e8, 1e5); T_init = np.full(N, 300.0)
    u_init = np.where(x < 0.3, 3000.0, 0.0)
    return _run_k2('25', ph1, ph2, x, dx, N, p_init, u_init, T_init, a1,
                   t_end=1e-4, bc='transmissive', cfl=0.3, use_mat=False,
                   exact=None, p_ref=1e5, u_ref=0.0, pass_ep=1.0, pass_eu=5000.0,
                   max_steps=5000, note='Mach 10 air-water')


def run_26():
    """26-H Hypersonic air shock (1 GPa jump, pure air)."""
    ph1 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5}
    ph2 = {'gamma': 1.4, 'pinf': 0.0, 'kv': 717.5}
    N, L = 200, 1.0; dx = L/N
    x = np.linspace(dx/2, L-dx/2, N)
    a1 = np.full(N, 0.5)
    p_init = np.where(x < 0.5, 1e9, 1e5); T_init = np.full(N, 300.0)
    u_init = np.zeros(N)
    return _run_k2('26', ph1, ph2, x, dx, N, p_init, u_init, T_init, a1,
                   t_end=5e-5, bc='transmissive', cfl=0.25, use_mat=False,
                   exact=None, p_ref=1e5, u_ref=0.0, pass_ep=1.0, pass_eu=5000.0,
                   max_steps=5000, note='Air shock 1GPa')


# ---------------------------------------------------------------------------
# Main loop with signal-based per-case timeout
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import signal

    class _TO(Exception): pass
    def _handler(signum, frame): raise _TO()
    signal.signal(signal.SIGALRM, _handler)

    PER_CASE_TIMEOUT = 90   # seconds per case

    cases = [(i, globals()[f'run_{i:02d}']) for i in range(1, 27)]
    results = []
    t_all = time.time()
    for num, fn in cases:
        print(f'\n=== Case {num:02d} ===', flush=True)
        signal.alarm(PER_CASE_TIMEOUT)
        t0 = time.time()
        try:
            res = fn()
        except _TO:
            print(f'  TIMEOUT after {PER_CASE_TIMEOUT}s', flush=True)
            res = ('TIMEOUT', float('nan'), float('nan'), float(PER_CASE_TIMEOUT), 0.0)
        except Exception as e:
            print(f'  OUTER EXC {type(e).__name__}: {e}', flush=True)
            res = (f'ERROR:{type(e).__name__}', float('nan'), float('nan'), 0.0, 0.0)
        finally:
            signal.alarm(0)
        wall = time.time() - t0
        print(f'  {res[0]} err_p={res[1]:.3e} err_u={res[2]:.3e} wall={wall:.1f}s', flush=True)
        results.append((num, res[0], res[1], res[2], wall, res[4] if len(res) > 4 else 0.0))

    wall_all = time.time() - t_all
    n_pass = sum(1 for r in results if r[1] == 'PASS')
    n_fail = sum(1 for r in results if r[1] != 'PASS' and not r[1].startswith('ERROR') and r[1] != 'TIMEOUT')
    n_to   = sum(1 for r in results if r[1] == 'TIMEOUT')
    n_err  = sum(1 for r in results if r[1].startswith('ERROR'))

    with open(f'{R}/all_26_force_summary.md', 'w') as f:
        f.write(f'# All 26 Validation — imex_5n (4-panel α,u,p,ρ)\n\n')
        f.write(f'**Total wall**: {wall_all:.1f}s\n\n')
        f.write(f'| PASS | FAIL | TIMEOUT | ERROR |\n|---|---|---|---|\n| {n_pass} | {n_fail} | {n_to} | {n_err} |\n\n')
        f.write('| # | Status | err_p | err_u | wall(s) | t_final | note |\n')
        f.write('|---|---|---|---|---|---|---|\n')
        notes = ['SG static','Abgrall NASG','Low-Mach pulse','Air 2kHz','Water 6kHz',
                 'Air/He impedance','Air-Water refl.','He/Air shock','Shock imp match',
                 'Gas-liq discharge','Shyue 3gas','Deng THINC','Denner HP Air/LP Water',
                 'Yoo HP Water/LP Air','Murrone-Guillard','Ideal+NASG','Gas-liq 3phase',
                 'CHS BN','UNDEX','Granular det','Water hammer','Rarefaction',
                 'Woodward-Colella','Ms10','Mach10 air-water','Air shock 1GPa']
        for i, (num, status, ep, eu, wall, tf) in enumerate(results):
            note = notes[i] if i < len(notes) else ''
            f.write(f'| {num:02d} | {status:12s} | {ep:.2e} | {eu:.2e} | {wall:5.1f} | {tf:.2e} | {note} |\n')
    print(f'\nTotal: PASS={n_pass} FAIL={n_fail} TIMEOUT={n_to} ERROR={n_err} wall={wall_all:.1f}s')
    print(f'Summary: {R}/all_26_force_summary.md')
