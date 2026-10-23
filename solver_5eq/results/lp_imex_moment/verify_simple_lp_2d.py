"""Section 89: 2D robustness demo of simple_lp_nd.py -- a planar gas shock (p 10 -> 30) hits a circular water column
(Schlieren-type picture of |grad rho|), plus positivity / conservation diagnostics.  PE of a translating circle was
checked separately (machine precision, section 89)."""
import importlib.util
import os
import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import simple_lp_nd as nd
import simple_lp as sl

PNG = "results/lp_imex_moment/simple_lp_2d.png"


def main():
    spec = importlib.util.spec_from_file_location("si", os.path.join(HERE, "verify_shock_interface.py"))
    si = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(si)
    tp, PH0, r2, u2, s = si.states(30.0)
    n = 256
    dx = [1.0 / n] * 2
    xx = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(xx, xx, indexing="ij")
    water = ((X - 0.55) ** 2 + (Y - 0.5) ** 2) < 0.12 ** 2
    slab = X < 0.30
    a = water.astype(float)
    r0 = np.full((n, n), 1.0)
    r1 = np.where(slab, r2, 0.2)
    u = np.where(slab & ~water, u2, 0.0)
    v = np.zeros((n, n))
    p = np.where(water, 10.0, np.where(slab, 30.0, 10.0))
    U = nd.cons(a, r0, r1, (u, v), p)
    t0 = nd.totals(U, dx)
    tt = time.time()
    U, ns = nd.run(U, dx, 0.030, order=2)
    wall = time.time() - tt
    W = nd.prim(U)
    t1 = nd.totals(U, dx)
    rho = U[0] + U[1]
    print(f"2D {n}^2, order 2 (+THINC): {ns} steps, {wall:.0f}s; alpha in [{W[0].min():.2e}, {W[0].max():.6f}], min p = {W[-1].min():.3f}, max p = {W[-1].max():.2f}, "
          f"min rho_0 = {W[1].min():.3f}, min rho_1 = {W[2][W[0] < 0.5].min():.4f}")
    print("conservation drift (m0, m1, qx, qy, E) relative to |initial| (qy initial 0 -> absolute):",
          [float(f"{(b - a_) / (abs(a_) if abs(a_) > 1e-12 else 1.0):.2e}") for a_, b in zip(t0, t1)])
    gx, gy = np.gradient(rho, dx[0], dx[1])
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    ax[0].imshow(np.log1p(np.hypot(gx, gy)).T, origin="lower", extent=(0, 1, 0, 1), cmap="gray_r")
    ax[0].set_title("|grad rho| (log), t = 0.030")
    ax[1].contourf(xx, xx, W[-1].T, 30, cmap="viridis")
    ax[1].contour(xx, xx, W[0].T, [0.5], colors="w", linewidths=0.8)
    ax[1].set_title("pressure, alpha = 0.5 (white)")
    for a_ in ax:
        a_.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(PNG, dpi=110)
    print(f"Plot saved: {PNG}")


if __name__ == "__main__":
    os.chdir(os.path.join(HERE, "..", ".."))
    main()
