"""Section 44 check: L2 convergence order of the finite-amplitude two-material flow with the
Level-1 reconstruction (needed > 3/2 for the inverse-inequality bootstrap), vs an N=800
reference computed with the same scheme; L0 shown for comparison."""
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


nc = load("nc", "verify_nonlinear_convergence.py")
PNG = "results/lp_imex_moment/nonlinear_l2_recon.png"


def main():
    out = {}
    for recon in (False, True):
        nc.vk.va.RECON[0] = recon
        mR, pR, wR = nc.solve(nc.NREF, 1e-5)
        rows = []
        for N in (50, 100, 200, 400):
            r = nc.NREF // N
            m, p, w = nc.solve(N, 1e-5 / r)
            h = 1.0 / N
            e = np.sqrt(h * np.sum((m / h - nc.coarsen(mR, r, "sum") / h) ** 2)
                        + h * np.sum((p - nc.coarsen(pR, r, "mean")) ** 2) / 100.0
                        + h * np.sum((w - nc.coarsen(wR, r, "node")) ** 2))
            rows.append((N, e))
            print(f"recon={recon} N={N}: L2 error (rho, p/10, w) = {e:.4e}")
        o = [np.log2(rows[i][1] / rows[i + 1][1]) for i in range(len(rows) - 1)]
        print(f"   recon={recon} L2 order: " + " ".join(f"{x:.2f}" for x in o))
        out[recon] = rows
    nc.vk.va.RECON[0] = False
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for recon, mk in ((False, "o-"), (True, "s-")):
        hs = [1.0 / r[0] for r in out[recon]]
        ax.loglog(hs, [r[1] for r in out[recon]], mk, label="Level-1" if recon else "L0")
    hs = np.array([1.0 / r[0] for r in out[True]])
    ax.loglog(hs, out[True][0][1] * (hs / hs[0]) ** 1.5, "k:", label="slope 3/2 (bootstrap threshold)")
    ax.loglog(hs, out[True][0][1] * (hs / hs[0]) ** 2, "k--", label="slope 2")
    ax.set_xlabel("h")
    ax.set_ylabel("L2 error vs N=800 reference")
    ax.set_title("finite-amplitude two-material flow, tiny occupancy")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(PNG, dpi=115)
    print(f"Plot saved: {PNG}")


if __name__ == "__main__":
    main()
