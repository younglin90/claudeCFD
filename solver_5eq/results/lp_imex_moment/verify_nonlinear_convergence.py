"""Theorem NL check (section 36): finite-amplitude smooth two-material flow (no shock),
piecewise-Kapila LP-IMEX, errors against a fine-grid reference (N=800), with the interfaces
placed to create tiny occupancies.  Expected: e(h) = O(h) for the L0 closure, constant
independent of the occupancy.
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
vn = load("vn", "verify_nonlinear_lipschitz.py")
PNG = "results/lp_imex_moment/nonlinear_convergence.png"
T = 0.05
NREF = 800


def solve(N, shift_cells):
    h = 1.0 / N
    st = vn.make_state(N, shift_cells, 0.0)
    cmax = np.sqrt((2 * 0.5 * 0.2 + 1.4 * (10.5 - 0.02)) / 0.2)
    tau = min(2.0 * (1.0 / NREF) / cmax * (NREF / N), 0.5 * h / 0.8)
    nst = int(np.ceil(T / tau))
    tau = T / nst
    for n in range(nst):
        st, _, _ = vk.step_kapila(st, vn.PH, tau)
    m = np.array([sum(q[3] for q in pcs) for pcs in st["cells"]])
    p = vn.pressures_relaxed(st, h)
    return m, p, st["w"]


def coarsen(fine, r, kind):
    if kind == "sum":
        return fine.reshape(-1, r).sum(axis=1)
    if kind == "mean":
        return fine.reshape(-1, r).mean(axis=1)
    return fine[r - 1::r]          # nodes: right face of each coarse cell


def main():
    out = {}
    for shift in (0.5, 1e-5):
        # interface offsets are given in units of the REFERENCE cell so all grids see the same geometry
        mR, pR, wR = solve(NREF, shift)
        rows = []
        for N in (50, 100, 200, 400):
            r = NREF // N
            st_shift = shift / r
            m, p, w = solve(N, st_shift)
            h = 1.0 / N
            em = np.sum(np.abs(m - coarsen(mR, r, "sum")))
            ep = h * np.sum(np.abs(p - coarsen(pR, r, "mean")))
            ew = h * np.sum(np.abs(w - coarsen(wR, r, "node")))
            rows.append((N, em, ep, ew))
            print(f"shift(ref cells)={shift:.0e} N={N}: L1 mass={em:.3e} L1 p={ep:.3e} L1 w={ew:.3e}")
        for j, name in ((1, "mass"), (2, "p"), (3, "w")):
            o = [np.log2(rows[i][j] / rows[i + 1][j]) for i in range(len(rows) - 1)]
            print(f"   order {name}: " + " ".join(f"{x:.2f}" for x in o))
        out[shift] = rows
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for shift, mk in ((0.5, "o-"), (1e-5, "s--")):
        rows = out[shift]
        hs = [1.0 / r[0] for r in rows]
        for j, name in ((1, "mass"), (2, "pressure"), (3, "velocity")):
            ax.loglog(hs, [r[j] for r in rows], mk, label=f"{name}, interface offset {shift:g} ref-cells")
    hs = np.array([1.0 / r[0] for r in out[0.5]])
    ax.loglog(hs, out[0.5][0][2] * hs / hs[0], "k:", label="slope 1")
    ax.set_xlabel("h")
    ax.set_ylabel("L1 error vs N=800 reference")
    ax.set_title("finite-amplitude two-material flow, piecewise-Kapila")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(PNG, dpi=115)
    print(f"Plot saved: {PNG}")


if __name__ == "__main__":
    main()
