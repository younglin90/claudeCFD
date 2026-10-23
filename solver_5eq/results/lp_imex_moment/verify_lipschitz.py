"""Section 83: empirical Lipschitz test of the one-step map of the real scheme.  Random node-velocity perturbations of
amplitude eps on three backgrounds (uniform, acoustic pulse on the interface, shock-tube state at t = 0.004); the ratio
||Delta_n|| / ||Delta_0|| (energy norm of the difference, matched by grid index / interface rank) over 25 steps is the
amplification of a perturbation.  A Lipschitz (differentiable-like) map gives ratios independent of eps; knife-edge
branch switches give ratios that grow as eps -> 0.  Compared for X_TOL = 0 (earlier code) and 1e-12 (fixed)."""
import importlib.util
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def load(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def norm_diff(tm, fs, PH, N, sa, sb):
    ka, kb = tm.keyed(sa, N), tm.keyed(sb, N)
    if set(ka) != set(kb):
        return np.nan
    m = np.array([z[3] for z in sa.zones])
    M = 0.5 * (m + np.roll(m, 1))
    e2 = 0.0
    for key, ia in ka.items():
        ib = kb[key]
        e2 += 0.5 * M[ia] * (sa.w[ia] - sb.w[ib]) ** 2
        za, zb = sa.zones[ia], sb.zones[ib]
        if za[0] != zb[0]:
            return np.nan
        ph = PH[za[0]]
        pf, Ef, l0 = fs._isentrope(za, ph)
        K = -l0 * (pf(l0 * (1 + 1e-7)) - pf(l0 * (1 - 1e-7))) / (2e-7 * l0)
        e2 += 0.5 * (za[2] - za[1]) / K * (fs.zone_p(za, ph) - fs.zone_p(zb, ph)) ** 2
    return np.sqrt(e2)


def job(args):
    bg, xtol, eps, seed = args
    vp = load("vp", "verify_pulse_interface.py")
    tm = load("tm", "verify_trapped_mode.py")
    fs = vp.fs
    fs.X_TOL = xtol
    PH = vp.PH
    N = 100
    h = 1.0 / N
    tau = 0.5 * h / 8.4
    X1, X2 = 0.3 + 0.5 * h, 0.7 + 0.5 * h
    vp.X1, vp.X2 = X1, X2
    if bg == "uniform":
        vp.U0, vp.EPS = 1.0, 0.0
        mk = lambda: fs.build(N, PH, vp.mat_of, vp.rho_of, vp.p_of, vp.u_of, [X1, X2])
    elif bg == "pulse":
        vp.U0, vp.EPS = 1.0, 0.08
        vp.bump = lambda x: np.where(np.abs((x - X1) / 0.08) < 1, np.cos(0.5 * np.pi * (x - X1) / 0.08) ** 6, 0.0)
        mk = lambda: fs.build(N, PH, vp.mat_of, vp.rho_of, vp.p_of, vp.u_of, [X1, X2])
    else:
        sc = load("sc", "verify_shock_capture.py")
        def mk():
            st = sc.setup(N, 40.0, 0.0, 0.0)[2]
            for _ in range(int(0.004 / tau)):
                st, _ = fs.step(st, tau, PH, N)
            return st
    a = mk()
    b = mk()
    rng = np.random.default_rng(seed)
    pert = rng.standard_normal(len(b.w)) * eps
    b.w = b.w + pert
    d0 = norm_diff(tm, fs, PH, N, a, b)
    amp = []
    for n in range(25):
        a, _ = fs.step(a, tau, PH, N)
        b, _ = fs.step(b, tau, PH, N)
        amp.append(norm_diff(tm, fs, PH, N, a, b) / d0)
    return args, float(np.nanmax(amp))


def main():
    jobs = [(bg, xt, eps, seed) for bg in ("uniform", "pulse", "shock") for xt in (0.0, 1e-12)
            for eps in (1e-9, 1e-7, 1e-5, 1e-3) for seed in (0, 1, 2)]
    with Pool(32) as pool:
        res = pool.map(job, jobs)
    tab = {}
    for (bg, xt, eps, seed), amp in res:
        tab.setdefault((bg, xt, eps), []).append(amp)
    print("max over 25 steps of ||Delta_n||/||Delta_0|| (median / max over 3 random perturbations)")
    for bg in ("uniform", "pulse", "shock"):
        for xt in (0.0, 1e-12):
            print(f"  {bg:8s} X_TOL={xt:g}: " + "  ".join(f"eps={eps:.0e}: {np.median(tab[(bg, xt, eps)]):.3g}/{np.max(tab[(bg, xt, eps)]):.3g}"
                                                          for eps in (1e-9, 1e-7, 1e-5, 1e-3)))


if __name__ == "__main__":
    os.chdir(os.path.join(HERE, "..", ".."))
    main()
