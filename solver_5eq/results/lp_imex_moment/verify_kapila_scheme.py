"""Piecewise-Kapila LP-IMEX with per-piece internal energy and in-cell pressure relaxation.

Per step: (1) relax each cell's pieces to one pressure at fixed cell volume and energy
(E_r' = E_r - p_hat (V_r' - V_r), sum V_r' = h; Saurel-type, energy exact),
(2) Kapila AVF Lagrange stage (every piece on its own single-material isentrope, common p),
(3) remap carrying per-piece energies (fragments proportional, union additive).
Compared with the Allaire-closure scheme on strong compression of a lamellar mixture with
the H_c counterexample EOS, and regression on uniform (u0, p0) and the gas|water shock tube.
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
PNG = "results/lp_imex_moment/kapila_scheme_vs_allaire.png"


def piece_energy_default(pcs, I, phases):
    p = va.decode(pcs, I, phases)
    return [q[:5] + [p * (q[2] - q[1]) / phases[q[0]].G + phases[q[0]].C * q[4]] for q in pcs]


def relax_cell(pcs, phases, h, x0):
    """Pieces [k,a,b,m,Z,E] -> common pressure; returns relaxed 5-field pieces and I."""
    if len(pcs) == 1:
        q = pcs[0]
        return [q[:5]], q[5]
    E0 = sum(q[5] for q in pcs)

    def V_of(q, ph_hat):
        ph = phases[q[0]]
        V0, Z0, Ei = q[2] - q[1], q[4], q[5]
        f = lambda y: (ph_hat * np.exp(y) * (1 + ph.G) + ph.G * ph.C * Z0 * (np.exp(y) / V0) ** (1 - ph.n)
                       - ph.G * (Ei + ph_hat * V0))
        y0 = np.log(V0)
        f0 = f(y0)
        if f0 == 0.0:
            return V0
        step = 0.05 if f0 < 0 else -0.05       # physical root: nearest to V0 on the correct side
        y1 = y0
        for _ in range(2000):
            y1 += step
            if np.sign(f(y1)) != np.sign(f0):
                return np.exp(brentq(f, min(y0, y1), max(y0, y1), xtol=1e-15))
            y0 = y1
        raise RuntimeError("relaxation infeasible")

    g = lambda y: sum(V_of(q, np.exp(y)) for q in pcs) - h
    ps = [phases[q[0]].G * (q[5] - phases[q[0]].C * q[4]) / (q[2] - q[1]) for q in pcs]
    lo, hi = np.log(min(ps)), np.log(max(ps))
    if hi - lo < 1e-9 or g(lo) * g(hi) > 0:
        # pieces already (to roundoff) at one pressure: isobaric value of the cell energy
        S = sum((q[2] - q[1]) / phases[q[0]].G for q in pcs)
        pr = (E0 - sum(phases[q[0]].C * q[4] for q in pcs)) / S
    else:
        pr = np.exp(brentq(g, lo, hi, xtol=1e-15))
    out, pos = [], x0
    for q in pcs:
        V = V_of(q, pr)
        ph = phases[q[0]]
        Z = q[4] * (V / (q[2] - q[1])) ** (1 - ph.n)
        out.append([q[0], pos, pos + V, q[3], Z])
        pos += V
    out[-1][2] = x0 + h
    I = sum(pr * (o[2] - o[1]) / phases[o[0]].G + phases[o[0]].C * o[4] for o in out)
    return out, I


def step_kapila(st, phases, tau, visc=None):
    N = len(st["cells"])
    h = 1.0 / N
    cells, I = [], np.zeros(N)
    for c, pcs in enumerate(st["cells"]):
        pc, I[c] = relax_cell(pcs, phases, h, c * h)
        cells.append(pc)
    rel = dict(cells=cells, I=I, w=st["w"])
    va.RELAX_HOOK[0] = lambda cs: relaxed_config(cs, phases)
    lag = va.lagrange(rel, tau, phases, visc, model="kapila")
    new, Q = va.remap(rel, lag, phases, keep_piece_energy=True)
    return new, Q, rel


def min_theta_pieces(st, phases):
    worst = np.inf
    for pcs in st["cells"]:
        for q in pcs:
            ph, L = phases[q[0]], q[2] - q[1]
            if L <= 0:
                continue
            th = (q[5] - ph.C * q[4]) * ph.G / L         # piece pressure
            worst = min(worst, (th * L - ph.A * q[4]) / (th * L))
    return worst


def min_theta_isobaric(st, phases):
    """Allaire closure: all pieces of a cell at the decoded common pressure."""
    worst = np.inf
    for pcs, I in zip(st["cells"], st["I"]):
        p = va.decode(pcs, I, phases)
        for q in pcs:
            L = q[2] - q[1]
            if L > 0:
                worst = min(worst, (p * L - phases[q[0]].A * q[4]) / (p * L))
    return worst


def relaxed_config(cells, phases):
    N = len(cells)
    h = 1.0 / N
    return [relax_cell(pcs, phases, h, c * h)[0] if len(pcs[0]) > 5 else pcs for c, pcs in enumerate(cells)]


def totals6(st, phases):
    m = np.array([sum(q[3] for q in pcs) for pcs in st["cells"]])
    M = va.node_masses(relaxed_config(st["cells"], phases), len(st["cells"]))
    E = sum(q[5] for pcs in st["cells"] for q in pcs)
    return m.sum(), np.sum(M * st["w"]), E + np.sum(0.5 * M * st["w"] ** 2)


def lamella_state(N, phases, U, p0=1.5):
    h = 1.0 / N
    cells, I = [], np.zeros(N)
    for c in range(N):
        a, b, mid = c * h, (c + 1) * h, c * h + 0.5 * h
        if 0.3 <= a < 0.7:
            pcs = [[0, a, mid, 0.5 * h, 0.5 * h], [1, mid, b, 0.1 * h, 0.02 * h]]
        else:
            pcs = [[1, a, b, 0.2 * h, 0.04 * h]]
        cells.append(pcs)
        I[c] = p0 * sum((q[2] - q[1]) / phases[q[0]].G for q in pcs) + sum(phases[q[0]].C * q[4] for q in pcs)
    w = np.array([U * np.sin(2 * np.pi * (j + 1) * h) for j in range(N)])
    return dict(cells=cells, I=I, w=w)


def run_compression(model, U, phases, N=40, nsteps=400):
    h = 1.0 / N
    st = lamella_state(N, phases, U)
    tau = 0.2 * h / max(U, 1.0)
    if model == "kapila":
        st = dict(cells=[piece_energy_default(p, I, phases) for p, I in zip(st["cells"], st["I"])], I=st["I"], w=st["w"])
        E0 = totals6(st, phases)[2]
    else:
        E0 = va.totals(st, phases)[2]
    hist = dict(t=[], pmin=[], theta=[], rhomax=[])
    status = "completed"
    for n in range(nsteps):
        try:
            if model == "kapila":
                st, Q, _ = step_kapila(st, phases, tau)
                th = min_theta_pieces(st, phases)
                p = np.array([va.decode([q[:5] for q in pcs], sum(q[5] for q in pcs), phases) for pcs in st["cells"]])
            else:
                lag = va.lagrange(st, tau, phases)
                st, Q = va.remap(st, lag, phases)
                th = min_theta_isobaric(st, phases)
                p = va.pressures(st, phases)
        except Exception as e:
            status = f"failed at step {n}: {type(e).__name__}"
            break
        hist["t"].append((n + 1) * tau)
        hist["pmin"].append(p.min())
        hist["theta"].append(th)
        hist["rhomax"].append(max(sum(q[3] for q in pcs) / h for pcs in st["cells"]))
        if th < 0 and status == "completed":
            status = f"theta<0 at step {n}"
    E1 = totals6(st, phases)[2] if model == "kapila" else va.totals(st, phases)[2]
    return hist, status, abs(E1 - E0) / E0


def main():
    phases = [va.Phase(1.0, 3, 3.0), va.Phase(0.5, 2, 0.4)]   # H_c counterexample pair
    res = {}
    for U in (2.0, 4.0, 6.0):
        for model in ("allaire", "kapila"):
            hist, status, dE = run_compression(model, U, phases)
            res[(model, U)] = hist
            print(f"{model:8s} U={U}: {status:28s} min p={min(hist['pmin']):+.4f} min theta={min(hist['theta']):+.4f} "
                  f"max rho={max(hist['rhomax']):.3f} dE/E={dE:.1e}")

    # regression 1: uniform (u0, p0), two phases with density variation, kapila scheme
    ph2 = [va.Phase(1.0, 3, 1.5), va.Phase(0.5, 2, 0.4)]
    N = 50
    h = 1.0 / N
    st = va.init_state(N, ph2, lambda x: 1.0 + 0.3 * np.sin(2 * np.pi * x) if 0.2 <= x < 0.6 else 0.2 + 0.05 * np.cos(2 * np.pi * x),
                       lambda x: 10.0, lambda x: 0.7, mat_fun=lambda x: 0 if 0.2 <= x < 0.6 else 1)
    st = dict(cells=[piece_energy_default(p, I, ph2) for p, I in zip(st["cells"], st["I"])], I=st["I"], w=st["w"])
    worst = 0.0
    for n in range(100):
        st, Q, rel = step_kapila(st, ph2, 0.4 * h / 0.7)
        p = va.pressures(rel, ph2)
        worst = max(worst, np.max(np.abs(p - 10.0)) / 10.0)
    print(f"kapila scheme, uniform u0=0.7 p0=10, 100 steps: max|dp|/p0={worst:.2e} max|w-u0|={np.max(np.abs(st['w']-0.7)):.1e}")

    # regression 2: gas|water shock tube with the kapila scheme
    water, gas = va.Phase(1.0, 3, 1.5), va.Phase(0.5, 2, 0.4)
    ph3 = [water, gas]
    N = 200
    h = 1.0 / N
    WL, WR, T = (0.2, 0.0, 20.0), (1.0, 0.0, 10.0), 0.02
    xs = (np.arange(N) + 0.5) * h
    exact, ps, us = vt.riemann2(gas, water, WL, WR, xs, T)
    cg = np.sqrt(2 * 0.5 * 0.2 + 1.4 * (20 - 0.02) / 0.2)
    tau = 0.8 * h / cg
    nst = int(np.ceil(T / tau))
    tau = T / nst
    st = va.init_state(N, ph3, lambda x: WL[0] if x < 0.5 else WR[0], lambda x: WL[2] if x < 0.5 else WR[2],
                       lambda x: 0.0, mat_fun=lambda x: 1 if x < 0.5 else 0)
    st = dict(cells=[piece_energy_default(p, I, ph3) for p, I in zip(st["cells"], st["I"])], I=st["I"], w=st["w"])
    E0 = totals6(st, ph3)[2]
    for n in range(nst):
        st, Q, rel = step_kapila(st, ph3, tau)
    rel = dict(cells=[relax_cell(pcs, ph3, h, c * h)[0] for c, pcs in enumerate(st["cells"])],
               I=np.array([relax_cell(pcs, ph3, h, c * h)[1] for c, pcs in enumerate(st["cells"])]), w=st["w"])
    p = va.pressures(rel, ph3)
    rho = np.array([sum(q[3] for q in pcs) for pcs in st["cells"]]) / h
    ifc = [x for x in vt.interface_pos(rel) if 0.3 < x < 0.8]
    win = (xs > 0.28) & (xs < 0.85)
    print(f"kapila scheme gas|water: dE/E={abs(totals6(st, ph3)[2]-E0)/E0:.1e} interface err={abs(ifc[0]-(0.5+us*T)):.1e} "
          f"L1 rho={np.sum(np.abs(rho-exact[:,0])[win])*h:.3e} L1 p={np.sum(np.abs(p-exact[:,2])[win])*h:.3e} "
          f"min theta={min_theta_pieces(st, ph3):.3f}")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for (model, U), hist in res.items():
        ls = "-" if model == "kapila" else "--"
        ax[0].plot(hist["t"], hist["pmin"], ls, label=f"{model}, U={U}")
        ax[1].plot(hist["t"], hist["theta"], ls, label=f"{model}, U={U}")
    ax[0].set_yscale("symlog", linthresh=0.1)
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_title("min cell pressure (theory: > 0)")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_ylim(-0.5, 1.0)
    ax[1].set_title("min piece thermal fraction (p - A zbar)/p (theory: > 0)")
    for a_ in ax[:2]:
        a_.set_xlabel("time")
        a_.legend(fontsize=7)
    ax[2].plot(xs, exact[:, 2], "k-", label="exact two-material Riemann")
    ax[2].plot(xs, p, "o", ms=3, mfc="none", label="kapila scheme, lambda=0.8")
    ax[2].set_xlim(0.28, 0.85)
    ax[2].set_title("gas|water shock tube pressure, kapila scheme")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(PNG, dpi=115)
    print(f"Plot saved: {PNG}")


if __name__ == "__main__":
    main()
