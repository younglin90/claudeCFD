"""Nonlinear (finite-amplitude) stability of the piecewise-Kapila LP-IMEX step:
two solutions U, V of a smooth finite-amplitude two-material flow differ by a small
pressure perturbation of size sigma.  Measured: the growth of their distance
    d_n = sqrt( sum_f M_f |w_U - w_V|^2 + h sum_i |p_U - p_V|^2 / K_i )
over time, while the minimum cell occupancy of the material interfaces is driven
down to 1e-6 by shifting the interfaces relative to the grid.
Theory target (section 36): d_n <= exp(C t_n) d_0 with C independent of the occupancy.
"""
import importlib.util
import os

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


va = load("va", "verify_acoustic.py")
vk = load("vk", "verify_kapila_scheme.py")
PNG = "results/lp_imex_moment/nonlinear_lipschitz_occupancy.png"
PH = [va.Phase(1.0, 3, 1.5), va.Phase(0.5, 2, 0.4)]
AMP = [0.3]


def make_state(N, shift, sigma):
    h = 1.0 / N
    x1, x2 = 0.3 + shift * h, 0.7 + shift * h
    cells, I = [], np.zeros(N)
    for c in range(N):
        a, b = c * h, (c + 1) * h
        cuts = sorted({a, b} | {x for x in (x1, x2) if a < x < b})
        pcs = []
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            k = 0 if x1 <= 0.5 * (lo + hi) < x2 else 1
            rho = 1.0 if k == 0 else 0.2
            pcs.append([k, lo, hi, rho * (hi - lo), rho ** PH[k].n * (hi - lo)])
        xc = (c + 0.5) * h
        p = 10.0 * (1 + 0.05 * np.cos(2 * np.pi * xc)) + sigma * 10.0 * np.sin(4 * np.pi * xc)
        I[c] = p * sum((q[2] - q[1]) / PH[q[0]].G for q in pcs) + sum(PH[q[0]].C * q[4] for q in pcs)
        cells.append(pcs)
    w = np.array([0.5 + AMP[0] * np.sin(2 * np.pi * (j + 1) * h) for j in range(N)])
    st = dict(cells=cells, I=I, w=w)
    return dict(cells=[vk.piece_energy_default(p, I_, PH) for p, I_ in zip(cells, I)], I=I, w=w)


def pressures_relaxed(st, h):
    out = []
    for c, pcs in enumerate(st["cells"]):
        pc, I = vk.relax_cell(pcs, PH, h, c * h)
        out.append(va.decode(pc, I, PH))
    return np.array(out)


def distance(U, V, h, K):
    mU = np.array([sum(q[3] for q in pcs) for pcs in U["cells"]])
    M = 0.5 * (mU + np.roll(mU, -1))
    dp = pressures_relaxed(U, h) - pressures_relaxed(V, h)
    return np.sqrt(np.sum(M * (U["w"] - V["w"]) ** 2) + h * np.sum(dp ** 2 / K))


def min_occupancy(st, h):
    occ = 1.0
    for pcs in st["cells"]:
        if len({q[0] for q in pcs}) > 1:
            for k in (0, 1):
                v = sum(q[2] - q[1] for q in pcs if q[0] == k) / h
                if v > 0:
                    occ = min(occ, v)
    return occ


def run(shift, sigma, N=100, lam=2.0, T=0.06):
    h = 1.0 / N
    U = make_state(N, shift, sigma)
    V = make_state(N, shift, 0.0)
    cmax = np.sqrt((2 * 0.5 * 0.2 + 1.4 * (10.5 - 0.02)) / 0.2)
    tau = min(lam * h / cmax, 0.5 * h / 0.8)
    nst = int(np.ceil(T / tau))
    tau = T / nst
    K = np.full(N, 1.0)
    pcs0 = V["cells"]
    for c, pcs in enumerate(pcs0):
        pc, I = vk.relax_cell(pcs, PH, h, c * h)
        pf, _, _ = va.isentrope_kapila(h, pc, PH, va.decode(pc, I, PH))
        e = 1e-6 * h
        K[c] = -h * (pf(h + e) - pf(h - e)) / (2 * e)
    d0 = distance(U, V, h, K)
    d = [1.0]
    occ = min_occupancy(V, h)
    for n in range(nst):
        U, _, _ = vk.step_kapila(U, PH, tau)
        V, _, _ = vk.step_kapila(V, PH, tau)
        d.append(distance(U, V, h, K) / d0)
        occ = min(occ, min_occupancy(V, h))
    return np.array(d), occ, tau * np.arange(nst + 1)


def main():
    res = {}
    for sigma in (1e-6, 1e-3):
        for shift in (0.5, 1e-2, 1e-4, 1e-6):
            d, occ, t = run(shift, sigma)
            res[(sigma, shift)] = (d, occ, t)
            C = np.log(d.max()) / t[-1] if d.max() > 1 else 0.0
            print(f"sigma={sigma:.0e} shift={shift:.0e}: min occupancy={occ:.2e}  max d_n/d_0={d.max():.5f}  "
                  f"final={d[-1]:.5f}  implied C=log(max)/T={C:.3f}")
    print("== C vs background velocity-gradient amplitude (theory: C grows with ||w_x||, independent of occupancy) ==")
    camp = {}
    for amp in (0.0, 0.15, 0.3):
        AMP[0] = amp
        d, occ, t = run(1e-4, 1e-6)
        camp[amp] = np.log(d.max()) / t[-1] if d.max() > 1 else 0.0
        print(f" amp={amp}: ||w_x||_inf={2*np.pi*amp:.3f}  max d/d0={d.max():.5f}  implied C={camp[amp]:.3f}")
    AMP[0] = 0.3
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    for sigma, a_ in zip((1e-6, 1e-3), ax):
        for shift in (0.5, 1e-2, 1e-4, 1e-6):
            d, occ, t = res[(sigma, shift)]
            a_.plot(t, d, label=f"min occupancy {occ:.0e}")
        a_.set_xlabel("time")
        a_.set_ylabel("d_n / d_0")
        a_.set_title(f"two finite-amplitude solutions, perturbation sigma={sigma:.0e}")
        a_.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(PNG, dpi=115)
    print(f"Plot saved: {PNG}")


if __name__ == "__main__":
    main()
