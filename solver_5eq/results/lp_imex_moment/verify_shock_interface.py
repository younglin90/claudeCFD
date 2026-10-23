"""Section 88: a shock travelling in the gas hits the gas | water interface (the interaction left untested in section 83).
Periodic unit interval: shocked gas state 2 on [0, 0.45), gas at rest (state 1) on [0.45, X1), water on [X1, X2), gas at rest
beyond.  The incident shock is the Rankine-Hugoniot shock of state 1 -> state 2 (pressure p2).  Exact solution after the hit
(t > t_hit = (X1 - 0.45)/s): the interface-centred two-material Riemann solution with left state 2 (rho2, u2, p2) and right state
water at rest, similarity origin (X1, t_hit)  -- valid near the interface until the rarefaction from the slab edge arrives;
L1 errors are taken on a window around the interface (reflected wave + transmitted shock).
Usage: verify_shock_interface.py [p2] [cq] [cl] [N...]"""
import importlib.util
import os
import sys
from multiprocessing import Pool

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PNG = "results/lp_imex_moment/shock_interface.png"
RW, P_W, R1, P1 = 1.0, 10.0, 0.2, 10.0
XS0, XI = 0.45, 0.60
POST = 0.008


def states(p2):
    tp = load("tp", "verify_two_phase.py")
    fs = load("fs0", "fitted_scheme.py")
    PH = [fs.Phase(1.0, 3, 1.5), fs.Phase(0.5, 2, 0.4)]
    c_isen, p_isen, wave = tp.side_funcs(PH[1])
    du, r2 = wave(p2, R1, P1)           # shock (p2 > P1): velocity jump and post-shock density
    u2 = du
    s = r2 * u2 / (r2 - R1)
    return tp, PH, r2, u2, s


def run(args):
    N, p2, cq, cl = args
    tp, PH0, r2, u2, s = states(p2)
    fs = load("fs", "fitted_scheme.py")
    fs.QVISC[0], fs.QVISC[1] = cq, cl
    PH = [fs.Phase(1.0, 3, 1.5), fs.Phase(0.5, 2, 0.4)]
    h = 1.0 / N
    X1, X2 = XI + 0.37 * h, 0.95 + 0.37 * h
    t_hit = (X1 - XS0) / s
    T = t_hit + POST
    st = fs.build(N, PH, lambda x: 0 if X1 <= x < X2 else 1,
                  lambda k, x: (RW + 0 * x) if k == 0 else np.where(x < XS0, r2, R1),
                  lambda x: np.where((x >= X1) & (x < X2), P_W, np.where(x < XS0, p2, P1)),
                  lambda x: np.where((x < XS0) & ~((x >= X1) & (x < X2)), u2, 0.0), [X1, X2])
    tau0 = 0.5 * h / 16.0
    nst = int(np.ceil(T / tau0))
    tau = T / nst
    t0 = fs.totals(st)
    for n in range(nst):
        st, _ = fs.step(st, tau, PH, N)
    t1 = fs.totals(st)
    rho, pc, wf = np.zeros(N), np.zeros(N), np.zeros(N)
    for z in st.zones:
        c = int(np.floor((0.5 * (z[1] + z[2])) / h + 1e-12)) % N
        rho[c] += z[3] / h
        pc[c] += fs.zone_p(z, PH[z[0]]) * (z[2] - z[1]) / h
    for b, z in enumerate(st.zones):
        if st.isface[b]:
            wf[int(round((z[1] % 1.0) / h)) % N] = st.w[b]
    return N, rho, pc, wf, (t1[0] - t0[0], t1[1] - t0[1], (t1[2] - t0[2]) / t0[2]), X1, T, t_hit


def exact(N, p2, X1, T, t_hit):
    tp, PH, r2, u2, s = states(p2)
    h = 1.0 / N
    ns = 6
    xs = np.array([(c + (k + 0.5) / ns) * h for c in range(N) for k in range(ns)])
    sol, ps, us = tp.riemann2(PH[1], PH[0], (r2, u2, p2), (RW, 0.0, P_W), xs, T - t_hit, x0=X1)
    xf = np.arange(N) * h
    sf, _, _ = tp.riemann2(PH[1], PH[0], (r2, u2, p2), (RW, 0.0, P_W), xf, T - t_hit, x0=X1)
    return sol[:, 0].reshape(N, ns).mean(axis=1), sol[:, 2].reshape(N, ns).mean(axis=1), sf[:, 1], ps, us


def main():
    p2 = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
    cq = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5
    cl = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
    Ns = [int(a) for a in sys.argv[4:]] or [200, 400, 800, 1600]
    tp, PH, r2, u2, s = states(p2)
    print(f"incident shock: p1={P1} -> p2={p2}, rho2={r2:.4f}, u2={u2:.4f}, speed s={s:.3f}")
    jobs = [(N, p2, c, l) for N in Ns for (c, l) in ((0.0, 0.0), (cq, cl))]
    with Pool(len(jobs)) as pool:
        res = pool.map(run, jobs)
        ex = pool.starmap(exact, [(r[0], p2, r[5], r[6], r[7]) for r in res if r[0] in Ns and True][::2] if False else [(N, p2, XI + 0.37 / N, ((XI + 0.37 / N) - XS0) / s + POST, ((XI + 0.37 / N) - XS0) / s) for N in Ns])
    print(f"exact interface state after the hit: p*={ex[0][3]:.4f}, u*={ex[0][4]:.4f}")
    err = {}
    for r, (c, l) in zip(res, [(c, l) for N in Ns for (c, l) in ((0.0, 0.0), (cq, cl))]):
        N = r[0]
        k = Ns.index(N)
        er, ep, eu, _, _ = ex[k]
        h = 1.0 / N
        xc = (np.arange(N) + 0.5) / N
        win = (xc > 0.55) & (xc < 0.68)
        xf = np.arange(N) / N
        winf = (xf > 0.55) & (xf < 0.68)
        e = (h * np.sum(np.abs(r[1] - er)[win]), h * np.sum(np.abs(r[2] - ep)[win]), h * np.sum(np.abs(r[3] - eu)[winf]))
        err[(N, c, l)] = e
        print(f"  N={N} (cq,cl)=({c},{l}): window L1 rho={e[0]:.3e} p={e[1]:.3e} u={e[2]:.3e} | dm={r[4][0]:.1e} dP={r[4][1]:.1e} dE/E={r[4][2]:.1e}", flush=True)
    for (c, l) in ((0.0, 0.0), (cq, cl)):
        o = [np.log2(np.array(err[(a, c, l)]) / np.array(err[(b, c, l)])) for a, b in zip(Ns[:-1], Ns[1:])]
        print(f"  (cq,cl)=({c},{l}) orders (rho,p,u):", [np.round(x, 2).tolist() for x in o],
              " mean:", np.round(np.mean(o, axis=0), 2).tolist())
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    N = Ns[-1]
    xc = (np.arange(N) + 0.5) / N
    er, ep, eu, ps, us = ex[-1]
    sel = {(c, l): r for r, (c, l) in zip(res, [(c, l) for N_ in Ns for (c, l) in ((0.0, 0.0), (cq, cl))]) if r[0] == N}
    for (c, l), r in sel.items():
        lab = "no viscosity" if c == 0 else f"viscosity cq={c}, cl={l}"
        for a, y in zip(ax, (r[1], r[2], r[3])):
            a.plot(xc if y is not r[3] else np.arange(N) / N, y, ".", ms=2, label=lab)
    ax[0].plot(xc, er, "k-", lw=0.8, label="exact (interface Riemann)")
    ax[1].plot(xc, ep, "k-", lw=0.8)
    ax[2].plot(np.arange(N) / N, eu, "k-", lw=0.8)
    for a, t in zip(ax, ("density", "pressure", "velocity")):
        a.set_xlim(0.45, 0.75)
        a.set_title(f"{t}, N={N}, shortly after the shock hits the interface")
        a.set_xlabel("x")
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(PNG, dpi=110)
    print(f"Plot saved: {PNG}")


if __name__ == "__main__":
    os.chdir(os.path.join(HERE, "..", ".."))
    main()
