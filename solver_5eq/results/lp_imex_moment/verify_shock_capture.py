"""Section 83: shock capturing of the interface-fitted LP-IMEX scheme.
Gas | water | gas on a periodic unit interval (water in [X1, 1-X1), mirror-symmetric), initial gas pressure P_G >> water
pressure: at X1 a shock runs into the water and a rarefaction into the gas (mirror at 1-X1).  Exact: two-material
Riemann solution (verify_two_phase.riemann2: Rankine-Hugoniot with the Mie-Gruneisen energy), valid until waves from the
two interfaces meet.  Without artificial viscosity the Lagrange step is isentropic per zone (no entropy production), with
von Neumann-Richtmyer viscosity (energy returned as heat, total energy exact) shocks are captured.
Usage: verify_shock_capture.py [PG] [cq] [cl] [Nlist...]"""
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


PNG = "results/lp_imex_moment/shock_capture.png"
RW, RG, P_W = 1.0, 0.2, 10.0
T_END = float(os.environ.get("T_END", 0.012))


def setup(N, PG, cq, cl):
    fs = load("fs", "fitted_scheme.py")
    fs.QVISC[0], fs.QVISC[1] = cq, cl
    PH = [fs.Phase(1.0, 3, 1.5), fs.Phase(0.5, 2, 0.4)]
    h = 1.0 / N
    X1 = 0.25 + 0.37 * h
    X2 = 1.0 - X1
    st = fs.build(N, PH, lambda x: 0 if X1 <= x < X2 else 1,
                  lambda k, x: (RW + 0 * x) if k == 0 else (RG + 0 * x),
                  lambda x: np.where((x >= X1) & (x < X2), P_W, PG), lambda x: 0 * x, [X1, X2])
    return fs, PH, st, X1, X2


def run(args):
    N, PG, cq, cl = args
    fs, PH, st, X1, X2 = setup(N, PG, cq, cl)
    h = 1.0 / N
    cmax = 12.0
    tau0 = 0.5 * h / cmax
    nst = int(np.ceil(T_END / tau0))
    tau = T_END / nst
    t0 = fs.totals(st)
    its = 0
    for n in range(nst):
        st, it = fs.step(st, tau, PH, N)
        its = max(its, it)
    t1 = fs.totals(st)
    rho, pc, hv = np.zeros(N), np.zeros(N), np.zeros(N)
    for z in st.zones:
        c = int(np.floor((0.5 * (z[1] + z[2])) / h + 1e-12)) % N
        rho[c] += z[3] / h
        pc[c] += fs.zone_p(z, PH[z[0]]) * (z[2] - z[1]) / h
    mom = np.zeros(N)
    wf = np.zeros(N)
    for b, z in enumerate(st.zones):
        if st.isface[b]:
            wf[int(round((z[1] % 1.0) / h)) % N] = st.w[b]
    cons = (t1[0] - t0[0], t1[1] - t0[1], (t1[2] - t0[2]) / t0[2])
    return N, rho, pc, wf, cons, its, X1


def exact(N, PG, X1):
    tp = load("tp", "verify_two_phase.py")
    fs = load("fs0", "fitted_scheme.py")
    PH = [fs.Phase(1.0, 3, 1.5), fs.Phase(0.5, 2, 0.4)]
    h = 1.0 / N
    ns = 6
    xs = np.array([(c + (k + 0.5) / ns) * h for c in range(N) for k in range(ns)])
    left = xs <= 0.5
    xq = np.where(left, xs, 1.0 - xs)
    sol, ps, us = tp.riemann2(PH[1], PH[0], (RG, 0.0, PG), (RW, 0.0, P_W), xq, T_END, x0=X1)
    rho = sol[:, 0].reshape(N, ns).mean(axis=1)
    p = sol[:, 2].reshape(N, ns).mean(axis=1)
    u_face = None
    xf = np.arange(N) * h
    xfq = np.where(xf <= 0.5, xf, 1.0 - xf)
    sf, _, _ = tp.riemann2(PH[1], PH[0], (RG, 0.0, PG), (RW, 0.0, P_W), xfq, T_END, x0=X1)
    u = np.where(xf <= 0.5, sf[:, 1], -sf[:, 1])
    return rho, p, u, ps, us


def main():
    PG = float(sys.argv[1]) if len(sys.argv) > 1 else 40.0
    cq = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5
    cl = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
    Ns = [int(a) for a in sys.argv[4:]] or [100, 200, 400, 800]
    jobs = [(N, PG, c, l) for N in Ns for (c, l) in ((0.0, 0.0), (cq, cl))]
    with Pool(len(jobs)) as pool:
        res = pool.map(run, jobs)
        ex = pool.starmap(exact, [(N, PG, 0.25 + 0.37 / N) for N in Ns])
    print(f"gas pressure {PG}, water pressure {P_W}; exact interface state p*={ex[0][3]:.4f}, u*={ex[0][4]:.4f}")
    err = {}
    for (N, rho, pc, wf, cons, its, X1), (c, l) in zip(res, [(c, l) for N in Ns for (c, l) in ((0.0, 0.0), (cq, cl))]):
        k = Ns.index(N)
        er, ep, eu, _, _ = ex[k]
        h = 1.0 / N
        # exclude 3 cells around the discontinuities? no: plain L1 over the periodic cell range
        e = (h * np.sum(np.abs(rho - er)), h * np.sum(np.abs(pc - ep)), h * np.sum(np.abs(wf - eu)))
        err[(N, c, l)] = e
        print(f"  N={N} (cq,cl)=({c},{l}): L1 rho={e[0]:.3e} p={e[1]:.3e} u={e[2]:.3e} | dm={cons[0]:.1e} dP={cons[1]:.1e} dE/E={cons[2]:.1e} newton its {its}", flush=True)
    for (c, l) in ((0.0, 0.0), (cq, cl)):
        o = []
        for a, b in zip(Ns[:-1], Ns[1:]):
            o.append(np.log2(np.array(err[(a, c, l)]) / np.array(err[(b, c, l)])))
        print(f"  (cq,cl)=({c},{l}) orders (rho,p,u) per refinement:", [np.round(x, 2).tolist() for x in o])
    # figure at the finest grid
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    k = len(Ns) - 1
    N = Ns[k]
    xc = (np.arange(N) + 0.5) / N
    er, ep, eu, ps, us = ex[k]
    sel = {(c, l): r for r, (c, l) in zip(res, [(c, l) for N_ in Ns for (c, l) in ((0.0, 0.0), (cq, cl))]) if r[0] == N}
    for (c, l), r in sel.items():
        lab = "no viscosity" if c == 0 else f"viscosity cq={c}, cl={l}"
        ax[0].plot(xc, r[1], ".", ms=2, label=lab)
        ax[1].plot(xc, r[2], ".", ms=2, label=lab)
        ax[2].plot(np.arange(N) / N, r[3], ".", ms=2, label=lab)
    ax[0].plot(xc, er, "k-", lw=0.8, label="exact")
    ax[1].plot(xc, ep, "k-", lw=0.8, label="exact")
    ax[2].plot(np.arange(N) / N, eu, "k-", lw=0.8, label="exact")
    for a, t in zip(ax, ("density", "pressure", "velocity")):
        a.set_title(f"{t}, N={N}, t={T_END}")
        a.set_xlabel("x")
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(PNG, dpi=110)
    print(f"Plot saved: {PNG}")


if __name__ == "__main__":
    os.chdir(os.path.join(HERE, "..", ".."))
    main()
