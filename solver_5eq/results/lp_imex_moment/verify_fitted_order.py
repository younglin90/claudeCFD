"""Interface-fitted LP-IMEX (fitted_scheme.py): convergence of a finite-amplitude smooth
two-material flow (water-like | gas-like, tiny occupancy offsets) against an N=400 reference
of the same scheme, compared with the mixed-cell piecewise-Kapila scheme (section 45)."""
import importlib.util
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("fs", os.path.join(HERE, "fitted_scheme.py"))
fs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fs)
PNG = "results/lp_imex_moment/fitted_order.png"
PH = [fs.Phase(1.0, 3, 1.5), fs.Phase(0.5, 2, 0.4)]
X1, X2 = 0.3 + 1e-5 / 400, 0.7 + 1e-5 / 400
T = 0.05


def solve(N):
    h = 1.0 / N
    st = fs.build(N, PH, lambda x: 0 if X1 <= x < X2 else 1, lambda k, x: (1.0 + 0 * x) if k == 0 else 0.2 + 0 * x,
                  lambda x: 10 * (1 + 0.05 * np.cos(2 * np.pi * x)), lambda x: 0.5 + 0.3 * np.sin(2 * np.pi * x), [X1, X2])
    cmax = np.sqrt((2 * 0.5 * 0.2 + 1.4 * (10.5 - 0.02)) / 0.2)
    tau = min(2.0 * h / cmax, 0.5 * h / 0.8)
    nst = int(np.ceil(T / tau))
    tau = T / nst
    t0 = fs.totals(st)
    for n in range(nst):
        st, it = fs.step(st, tau, PH, N)
    t1 = fs.totals(st)
    # cell averages on the Eulerian grid: density (mass/h), pressure (volume average), face velocity
    rho, pc = np.zeros(N), np.zeros(N)
    for z in st.zones:
        c = int(np.floor((0.5 * (z[1] + z[2])) / h + 1e-12)) % N
        rho[c] += z[3] / h
        pc[c] += fs.zone_p(z, PH[z[0]]) * (z[2] - z[1]) / h
    wf = np.zeros(N)
    for b, z in enumerate(st.zones):
        if st.isface[b]:
            j = int(round((z[1] % 1.0) / h)) % N
            wf[j] = st.w[b]
    return rho, pc, wf, (t1[0] - t0[0], t1[1] - t0[1], (t1[2] - t0[2]) / t0[2]), fs.min_theta(st, PH)


def main():
    R = solve(400)
    prev = None
    rows = []
    for N in (50, 100, 200):
        r = 400 // N
        rho, p, w, cons, th = solve(N)
        h = 1.0 / N
        er = np.sqrt(h * np.sum((rho - R[0].reshape(-1, r).mean(axis=1)) ** 2))
        ep = np.sqrt(h * np.sum((p - R[1].reshape(-1, r).mean(axis=1)) ** 2))
        ew = np.sqrt(h * np.sum((w - R[2][::r]) ** 2))
        cur = (er, ep, ew)
        rows.append((N,) + cur)
        print(f"fitted N={N}: L2 rho={er:.3e} p={ep:.3e} w={ew:.3e} "
              + ("" if prev is None else "orders " + " ".join(f"{np.log2(a / b):.2f}" for a, b in zip(prev, cur)))
              + f"  | dm={cons[0]:.1e} dP={cons[1]:.1e} dE/E={cons[2]:.1e} min theta={th:.3f}", flush=True)
        prev = cur
    fig, ax = plt.subplots(figsize=(7, 4.4))
    hs = np.array([1.0 / r[0] for r in rows])
    for j, name in ((1, "density"), (2, "pressure"), (3, "velocity")):
        ax.loglog(hs, [r[j] for r in rows], "o-", label=f"interface-fitted, {name}")
    ax.loglog(hs, rows[0][2] * (hs / hs[0]) ** 2, "k--", label="slope 2")
    ax.loglog(hs, rows[0][2] * (hs / hs[0]), "k:", label="slope 1")
    ax.set_xlabel("h")
    ax.set_ylabel("L2 error vs N=400 reference")
    ax.set_title("two-material finite-amplitude flow, interface-fitted LP-IMEX")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(PNG, dpi=115)
    print(f"Plot saved: {PNG}")


if __name__ == "__main__":
    main()
