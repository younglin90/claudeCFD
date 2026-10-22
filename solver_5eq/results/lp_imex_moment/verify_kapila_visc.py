"""(1) Kapila vs Allaire mixture isentrope for the H_c counterexample.
(2) Compression-only implicit artificial viscosity inside the AVF Lagrange stage
    (energy-exact, heat >= 0): effect on post-shock oscillation, single- and two-phase
    shock tubes, against exact Riemann solutions.
"""
import importlib.util
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


va = load("va", "verify_acoustic.py")
vt = load("vt", "verify_two_phase.py")
PNG = "results/lp_imex_moment/kapila_visc_theory_vs_numeric.png"


def kapila_p_of_J(phs, masses, rho0, p0, J, L0=1.0):
    """Each phase on its own isentrope at a common pressure: sum_k m_k/rho_k(p) = J L0."""
    def rho_of_p(ph, r0, p):
        f = lambda r: ph.A * r ** ph.n + (p0 - ph.A * r0 ** ph.n) * (r / r0) ** (1 + ph.G) - p
        return brentq(f, 1e-12 * r0, 1e6 * r0)
    g = lambda p: sum(m / rho_of_p(ph, r0, p) for ph, m, r0 in zip(phs, masses, rho0)) - J * L0
    return brentq(g, 1e-9, 1e9)


def shock_metrics(xs, h, rho, p, exact, win):
    return (np.sum(np.abs(rho - exact[:, 0])[win]) * h, np.sum(np.abs(p - exact[:, 2])[win]) * h,
            np.max((rho - exact[:, 0])[win]))


def main():
    # ---------- (1) Kapila vs Allaire ----------
    phD = [va.Phase(1.0, 3, 3.0), va.Phase(0.5, 2, 0.4)]
    pcs = [[0, 0.0, 0.5, 0.5, 0.5], [1, 0.5, 1.0, 0.1, 0.02]]
    pD = 1.5
    pA, _ = va.isentrope(1.0, pcs, phD, pD)
    Js = np.linspace(1.0, 0.02, 200)
    alla = np.array([pA(J) for J in Js])
    kap = np.array([kapila_p_of_J(phD, [0.5, 0.1], [1.0, 0.2], pD, J) for J in Js])
    Jst = brentq(pA, 0.02, 1.0)
    print(f"== Kapila vs Allaire (H_c counterexample): Allaire p=0 at J={Jst:.4f}; Kapila min p on [0.02,1] = {kap.min():.4f} (> 0)")
    # Kapila thermal positivity check per phase: p - A rho^n
    th = []
    for J, p in zip(Js, kap):
        for ph, r0 in zip(phD, [1.0, 0.2]):
            r = brentq(lambda r: ph.A * r ** ph.n + (pD - ph.A * r0 ** ph.n) * (r / r0) ** (1 + ph.G) - p, 1e-12, 1e6)
            th.append((p - ph.A * r ** ph.n) / p)
    print(f"   Kapila min thermal fraction (p - A rho^n)/p = {min(th):.4f}")

    # ---------- (2) artificial viscosity ----------
    water, gas = va.Phase(1.0, 3, 1.5), va.Phase(0.5, 2, 0.4)
    phases = [water, gas]
    N = 200
    h = 1.0 / N
    xs = (np.arange(N) + 0.5) * h
    cases = {
        "single": dict(WL=(1.0, 0.0, 20.0), WR=(1.0, 0.0, 10.0), T=0.04, ph=(water, water),
                       mat=lambda x: 0, cref=np.sqrt(3 + 2.5 * 19), win=(0.29, 0.75)),
        "gas|water": dict(WL=(0.2, 0.0, 20.0), WR=(1.0, 0.0, 10.0), T=0.02, ph=(gas, water),
                          mat=lambda x: 1 if x < 0.5 else 0,
                          cref=np.sqrt(2 * 0.5 * 0.2 + 1.4 * (20 - 0.02) / 0.2), win=(0.28, 0.85)),
    }
    visc_opts = {"no viscosity": None, "q: c1=0.5, c2=1.0": (0.5, 1.0)}
    store = {}
    for cname, cs in cases.items():
        exact, ps, us = vt.riemann2(cs["ph"][0], cs["ph"][1], cs["WL"], cs["WR"], xs, cs["T"])
        win = (xs > cs["win"][0]) & (xs < cs["win"][1])
        for lam in (0.8, 3.0):
            for vname, vis in visc_opts.items():
                tau = lam * h / cs["cref"]
                nst = int(np.ceil(cs["T"] / tau))
                tau = cs["T"] / nst
                st = va.init_state(N, phases, lambda x: cs["WL"][0] if x < 0.5 else cs["WR"][0],
                                   lambda x: cs["WL"][2] if x < 0.5 else cs["WR"][2], lambda x: 0.0, mat_fun=cs["mat"])
                t0 = va.totals(st, phases)
                thm = [np.inf]
                st, Qmin, Qs, itm = va.run(st, phases, tau, nst, visc=vis,
                                           hook=lambda n, s: thm.__setitem__(0, min(thm[0], vt.min_theta(s, phases))))
                t1 = va.totals(st, phases)
                rho = np.array([sum(q[3] for q in pc) for pc in st["cells"]]) / h
                p = va.pressures(st, phases)
                l1r, l1p, over = shock_metrics(xs, h, rho, p, exact, win)
                ifc = [x for x in vt.interface_pos(st) if 0.3 < x < 0.8]
                store[(cname, lam, vname)] = (rho, p, exact)
                print(f"{cname:9s} lambda={lam} {vname:18s}: L1 rho={l1r:.3e} L1 p={l1p:.3e} max overshoot rho={over:.3e} "
                      f"dE/E={abs(t1[2]-t0[2])/t0[2]:.1e} min theta={thm[0]:.3f} newton<={itm}"
                      + (f" interface err={abs(ifc[0]-(0.5+us*cs['T'])):.1e}" if ifc else ""))

    fig, ax = plt.subplots(2, 3, figsize=(16, 8.5))
    ax[0, 0].plot(Js, alla, "r-", label="Allaire (frozen alpha) isentrope")
    ax[0, 0].plot(Js, kap, "b-", label="Kapila (each phase isentropic, common p)")
    ax[0, 0].axhline(0, color="k", lw=0.8)
    ax[0, 0].axvline(Jst, color="r", ls=":", label=f"Allaire p=0 at J={Jst:.3f}")
    ax[0, 0].set_yscale("symlog", linthresh=1)
    ax[0, 0].set_xlabel("J = L/L0")
    ax[0, 0].set_ylabel("mixture pressure")
    ax[0, 0].set_title("H_c counterexample: (n=3,G=3)+(n=2,G=0.4)")
    ax[0, 0].legend(fontsize=7)
    panels = [(0, 1, "single", 0, "single phase: density"), (0, 2, "single", 1, "single phase: pressure"),
              (1, 0, "gas|water", 0, "gas|water: density"), (1, 1, "gas|water", 1, "gas|water: pressure")]
    for r_, c_, cname, j, title in panels:
        a_ = ax[r_, c_]
        exact = store[(cname, 3.0, "no viscosity")][2]
        a_.plot(xs, exact[:, 0 if j == 0 else 2], "k-", lw=1.5, label="exact Riemann")
        for vname, mk in (("no viscosity", "x"), ("q: c1=0.5, c2=1.0", "o")):
            a_.plot(xs, store[(cname, 3.0, vname)][j], mk, ms=3.5, mfc="none", label=f"lambda=3, {vname}")
        a_.set_xlim(*cases[cname]["win"])
        a_.set_title(title + " (T end)")
        a_.legend(fontsize=7)
    labels, l1_nov, l1_v = [], [], []
    for cname in cases:
        for lam in (0.8, 3.0):
            exact = store[(cname, lam, "no viscosity")][2]
            win = (xs > cases[cname]["win"][0]) & (xs < cases[cname]["win"][1])
            labels.append(f"{cname}\nlam={lam}")
            l1_nov.append(np.sum(np.abs(store[(cname, lam, "no viscosity")][1] - exact[:, 2])[win]) * h)
            l1_v.append(np.sum(np.abs(store[(cname, lam, "q: c1=0.5, c2=1.0")][1] - exact[:, 2])[win]) * h)
    xx = np.arange(len(labels))
    ax[1, 2].bar(xx - 0.18, l1_nov, 0.36, label="no viscosity")
    ax[1, 2].bar(xx + 0.18, l1_v, 0.36, label="with q")
    ax[1, 2].set_xticks(xx, labels, fontsize=8)
    ax[1, 2].set_ylabel("L1 pressure error vs exact")
    ax[1, 2].set_title("pressure L1 error vs exact Riemann")
    ax[1, 2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(PNG, dpi=115)
    print(f"Plot saved: {PNG}")


if __name__ == "__main__":
    main()
