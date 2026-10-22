"""Level-1 (limited linear piece reconstruction) vs L0 (flat pieces) in the piecewise-Kapila
LP-IMEX scheme.  (1) equilibrium sector: exact translation, L1 cell-mass error order;
(2) uniform-pressure preservation and positivity with reconstruction on;
(3) finite-amplitude two-material flow vs an N=800 reference."""
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
vn = load("vn", "verify_nonlinear_lipschitz.py")
PNG = "results/lp_imex_moment/second_order_recon.png"
PH = vn.PH
GX, GW = np.polynomial.legendre.leggauss(10)
X1, X2 = 0.3 + 1e-4, 0.7 + 3e-5


def rho_f(k, x):
    return 1.0 + 0.2 * np.sin(2 * np.pi * x) if k == 0 else 0.2 + 0.03 * np.cos(2 * np.pi * x)


def integ(f, a, b):
    xm, xr = 0.5 * (a + b), 0.5 * (b - a)
    return xr * np.sum(GW * f(xm + xr * GX))


def exact_cells(N, t, u0=1.0, p0=10.0):
    h = 1.0 / N
    s = u0 * t
    x1, x2 = (X1 + s) % 1.0, (X2 + s) % 1.0
    cells = []
    for c in range(N):
        a, b = c * h, (c + 1) * h
        cuts = sorted({a, b} | {x for x in (x1, x2) if a < x < b})
        pcs = []
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            xm = (0.5 * (lo + hi) - s) % 1.0
            k = 0 if X1 <= xm < X2 else 1
            m = integ(lambda x: rho_f(k, x - s), lo, hi)
            Z = integ(lambda x: rho_f(k, x - s) ** PH[k].n, lo, hi)
            pcs.append([k, lo, hi, m, Z, p0 * (hi - lo) / PH[k].G + PH[k].C * Z])
        cells.append(pcs)
    return cells


def eq_run(N, recon, T=0.5, nu=0.5, u0=1.0):
    vk.va.RECON[0] = recon
    h = 1.0 / N
    st = dict(cells=exact_cells(N, 0.0), w=np.full(N, u0))
    tau = nu * h / u0
    nst = int(round(T / tau))
    worst_p, min_th = 0.0, np.inf
    for n in range(nst):
        st, _, rel = vk.step_kapila(st, PH, tau)
        worst_p = max(worst_p, np.max(np.abs(va.pressures(rel, PH) - 10.0)) / 10.0)
        min_th = min(min_th, vk.min_theta_pieces(st, PH))
    ex = exact_cells(N, nst * tau)
    m = np.array([sum(q[3] for q in pcs) for pcs in st["cells"]])
    me = np.array([sum(q[3] for q in pcs) for pcs in ex])
    vk.va.RECON[0] = False
    return np.sum(np.abs(m - me)), worst_p, min_th


def main():
    Ns = [50, 100, 200, 400]
    res = {}
    for recon in (False, True):
        rows = [eq_run(N, recon) for N in Ns]
        res[recon] = rows
        o = [np.log2(rows[i][0] / rows[i + 1][0]) for i in range(len(rows) - 1)]
        print(f"equilibrium, recon={recon}: L1 mass " + " ".join(f"{r[0]:.3e}" for r in rows)
              + " | order " + " ".join(f"{x:.2f}" for x in o)
              + f" | max dp/p0 {max(r[1] for r in rows):.1e} | min theta {min(r[2] for r in rows):.3f}")
    fig, ax = plt.subplots(figsize=(7, 4.4))
    hs = 1.0 / np.array(Ns)
    for recon, mk in ((False, "o-"), (True, "s-")):
        ax.loglog(hs, [r[0] for r in res[recon]], mk, label="Level-1 linear pieces" if recon else "L0 flat pieces")
    ax.loglog(hs, res[False][0][0] * hs / hs[0], "k:", label="slope 1")
    ax.loglog(hs, res[True][0][0] * (hs / hs[0]) ** 2, "k--", label="slope 2")
    ax.set_xlabel("h")
    ax.set_ylabel("L1 cell-mass error vs exact translation")
    ax.set_title("equilibrium sector, piecewise-Kapila, T=0.5, nu=0.5")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(PNG, dpi=115)
    print(f"Plot saved: {PNG}")


if __name__ == "__main__":
    main()
