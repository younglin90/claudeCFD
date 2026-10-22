"""Two-phase shock tube for the staggered LP-IMEX oracle (verify_acoustic.py):
high-pressure gas-like phase | water-like phase, interface = contact.
Theory: exact two-material Riemann solution (pure phases on each side).
Checks: plateau states, contact (interface) position, pressure/velocity at the
contact, per-material mass, momentum, total energy, piece thermal positivity, Q >= 0.
"""
import importlib.util
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("va", os.path.join(HERE, "verify_acoustic.py"))
va = importlib.util.module_from_spec(spec)
spec.loader.exec_module(va)
PNG = "results/lp_imex_moment/two_phase_theory_vs_numeric.png"


def side_funcs(ph):
    A, n, G, C = ph.A, ph.n, ph.G, ph.C

    def e(r, p):
        return p / (G * r) + C * r ** (n - 1)

    def c_isen(r, ra, pa):
        return np.sqrt(n * A * r ** (n - 1) + (1 + G) * (pa - A * ra ** n) * r ** G / ra ** (1 + G))

    def p_isen(r, ra, pa):
        return A * r ** n + (pa - A * ra ** n) * (r / ra) ** (1 + G)

    def wave(ps, rK, pK):
        if ps > pK:
            rs = brentq(lambda r: e(r, ps) - e(rK, pK) - 0.5 * (pK + ps) * (1 / rK - 1 / r), rK, 50 * rK)
            return np.sqrt((ps - pK) * (1 / rK - 1 / rs)), rs
        rs = brentq(lambda r: p_isen(r, rK, pK) - ps, 1e-9 * rK, rK)
        return -quad(lambda r: c_isen(r, rK, pK) / r, rs, rK)[0], rs

    return c_isen, p_isen, wave


def riemann2(phL, phR, WL, WR, xs, t, x0=0.5):
    cL_, pL_, waveL = side_funcs(phL)
    cR_, pR_, waveR = side_funcs(phR)
    rL, uL, pL = WL
    rR, uR, pR = WR
    ps = brentq(lambda p: waveL(p, rL, pL)[0] + waveR(p, rR, pR)[0] + uR - uL,
                0.3 * min(pL, pR), 5 * max(pL, pR))
    fL, rsL = waveL(ps, rL, pL)
    fR, rsR = waveR(ps, rR, pR)
    us = 0.5 * (uL + uR) + 0.5 * (fR - fL)
    out = np.zeros((len(xs), 3))
    for i, x in enumerate(xs):
        xi = (x - x0) / t
        if xi < us:
            if ps <= pL:
                head, tail = uL - cL_(rL, rL, pL), us - cL_(rsL, rL, pL)
                if xi < head:
                    out[i] = (rL, uL, pL)
                elif xi > tail:
                    out[i] = (rsL, us, ps)
                else:
                    g = lambda r: uL + quad(lambda s: cL_(s, rL, pL) / s, r, rL)[0] - cL_(r, rL, pL) - xi
                    r = brentq(g, rsL, rL)
                    out[i] = (r, xi + cL_(r, rL, pL), pL_(r, rL, pL))
            else:
                s = uL - np.sqrt((ps - pL) / (1 / rL - 1 / rsL)) / rL
                out[i] = (rL, uL, pL) if xi < s else (rsL, us, ps)
        else:
            if ps > pR:
                s = uR + np.sqrt((ps - pR) / (1 / rR - 1 / rsR)) / rR
                out[i] = (rsR, us, ps) if xi < s else (rR, uR, pR)
            else:
                head, tail = uR + cR_(rR, rR, pR), us + cR_(rsR, rR, pR)
                if xi > head:
                    out[i] = (rR, uR, pR)
                elif xi < tail:
                    out[i] = (rsR, us, ps)
                else:
                    g = lambda r: uR - quad(lambda s: cR_(s, rR, pR) / s, r, rR)[0] + cR_(r, rR, pR) - xi
                    r = brentq(g, rsR, rR)
                    out[i] = (r, xi - cR_(r, rR, pR), pR_(r, rR, pR))
    return out, ps, us


def material_totals(state):
    tot = np.zeros(2)
    for pcs in state["cells"]:
        for q in pcs:
            tot[q[0]] += q[3]
    return tot


def min_theta(state, phases):
    worst = np.inf
    for pcs, I in zip(state["cells"], state["I"]):
        p = va.decode(pcs, I, phases)
        for q in pcs:
            L = q[2] - q[1]
            if L <= 0:
                continue
            ph = phases[q[0]]
            worst = min(worst, (p * L - ph.A * q[4]) / (ph.G * L * p))
    return worst


def interface_pos(state):
    pos = []
    for pcs in state["cells"]:
        for a, b in zip(pcs[:-1], pcs[1:]):
            if a[0] != b[0]:
                pos.append(a[2])
    return pos


def main():
    water = va.Phase(1.0, 3, 1.5)
    gas = va.Phase(0.5, 2, 0.4)
    phases = [water, gas]
    WL, WR = (0.2, 0.0, 20.0), (1.0, 0.0, 10.0)     # gas | water
    N = 200
    h = 1.0 / N
    Tend = 0.02
    cgas = np.sqrt(2 * gas.A * WL[0] + (1 + gas.G) * (WL[2] - gas.A * WL[0] ** 2) / WL[0])
    xs = (np.arange(N) + 0.5) * h
    exact, ps, us = riemann2(gas, water, WL, WR, xs, Tend)
    print(f"exact: p*={ps:.5f} u*={us:.5f} contact at {0.5 + us * Tend:.5f}, c_gas={cgas:.3f}")
    win = (xs > 0.28) & (xs < 0.85)
    res = {}
    for lam in (0.8, 3.0):
        tau = lam * h / cgas
        nst = int(np.ceil(Tend / tau))
        tau = Tend / nst
        st = va.init_state(N, phases, lambda x: WL[0] if x < 0.5 else WR[0],
                           lambda x: WL[2] if x < 0.5 else WR[2], lambda x: 0.0,
                           mat_fun=lambda x: 1 if x < 0.5 else 0)
        m0, t0 = material_totals(st), va.totals(st, phases)
        th = [np.inf]
        st, Qmin, Qs, itm = va.run(st, phases, tau, nst,
                                   hook=lambda n, s: th.__setitem__(0, min(th[0], min_theta(s, phases))))
        m1, t1 = material_totals(st), va.totals(st, phases)
        rho = np.array([sum(q[3] for q in pcs) for pcs in st["cells"]]) / h
        p = va.pressures(st, phases)
        ifc = [x for x in interface_pos(st) if 0.3 < x < 0.8]
        near = np.abs(xs - (0.5 + us * Tend)) < 6 * h
        res[lam] = dict(st=st, rho=rho, p=p, ifc=ifc)
        print(f"lambda={lam}: steps={nst} newton<={itm} dmass_k={np.abs(m1-m0).max():.1e} dmom={abs(t1[1]-t0[1]):.1e} "
              f"dE/E={abs(t1[2]-t0[2])/t0[2]:.1e} minQ/maxQ={Qmin/max(Qs,1e-300):.1e} min theta/(pL)={th[0]:.3f}")
        print(f"   interface numeric={ifc}  exact={0.5 + us * Tend:.6f}")
        print(f"   contact window |p-p*|/p* max={np.max(np.abs(p[near]-ps))/ps:.3e}  "
              f"L1(rho) window={np.sum(np.abs(rho-exact[:,0])[win])*h:.3e} L1(p)={np.sum(np.abs(p-exact[:,2])[win])*h:.3e}")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for lam, mk in ((0.8, "."), (3.0, "x")):
        r = res[lam]
        ax[0].plot(xs, r["rho"], mk, ms=4, label=f"numerical lambda={lam}")
        ax[1].plot(xs, r["p"], mk, ms=4, label=f"numerical lambda={lam}")
        wn = r["st"]["w"]
        ax[2].plot((np.arange(N) + 1) * h, wn, mk, ms=4, label=f"numerical lambda={lam} (nodes)")
    for a_, j, t_ in ((ax[0], 0, "density (cell mass / h)"), (ax[1], 2, "pressure"), (ax[2], 1, "velocity")):
        a_.plot(xs, exact[:, j], "k-", lw=1.5, label="exact two-material Riemann")
        a_.axvline(0.5 + us * Tend, color="gray", ls=":", lw=0.8)
        a_.set_xlim(0.28, 0.85)
        a_.set_title(f"gas|water shock tube, T={Tend}: {t_}")
        a_.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(PNG, dpi=115)
    print(f"Plot saved: {PNG}")


if __name__ == "__main__":
    main()
