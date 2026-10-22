"""Theory vs numerical comparison plots for the piecewise-moment oracle (verify_cv_t3.py)."""
import importlib.util
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("v", os.path.join(HERE, "verify_cv_t3.py"))
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)
PNG = "results/lp_imex_moment/cv_t3_theory_vs_numeric.png"

eos, nu, T, N = v.EOS_NEG, 0.5, 0.5, 50
h = 1.0 / N
nsteps = int(round(T / (nu * h)))
cells = v.project(N, eos, 0.0)
for _ in range(nsteps):
    cells = v.step(cells, N, nu)
Tn = nsteps * nu * h

fig, ax = plt.subplots(2, 2, figsize=(12, 8))
x = np.linspace(0, 1, 2000)
mat = np.array([v.mat_of(xx - v.U0 * Tn) for xx in x])
for k, col in ((0, "C0"), (1, "C1")):
    n = eos[k][1]
    ex = np.where(mat == k, v.rho_field(k, x - v.U0 * Tn), np.nan)
    exz = np.where(mat == k, v.rho_field(k, x - v.U0 * Tn) ** n, np.nan)
    ax[0, 0].plot(x, ex, "-", color=col, lw=1.2, label=f"exact rho_{k+1}")
    ax[0, 1].plot(x, exz, "-", color=col, lw=1.2, label=f"exact rho_{k+1}^{n}")
    first = True
    for pcs in cells:
        for q in pcs:
            if q[0] != k:
                continue
            L = q[2] - q[1]
            ax[0, 0].plot([q[1], q[2]], [q[3] / L] * 2, color=col, lw=2.2, alpha=0.6,
                          label=f"numerical m/l, mat {k+1}" if first else None)
            ax[0, 1].plot([q[1], q[2]], [q[4] / L] * 2, color=col, lw=2.2, alpha=0.6,
                          label=f"numerical Z/l, mat {k+1}" if first else None)
            first = False
for a in ax[0]:
    for s in v.MAT1:
        a.axvline((s + v.U0 * Tn) % 1.0, color="k", ls=":", lw=0.8)
ax[0, 0].set_title(f"density: exact rho(x-uT) vs numerical (N={N}, T={Tn})")
ax[0, 1].set_title("EOS moment: exact rho^n vs numerical Z/l")
ax[0, 0].legend(fontsize=7)
ax[0, 1].legend(fontsize=7)
ax[0, 0].set_xlabel("x  (dotted: exact interface positions)")
ax[0, 1].set_xlabel("x")

pz = v.decode(cells, eos, True)
pn = v.decode(cells, eos, False)
xc = (np.arange(N) + 0.5) * h
ax[1, 0].plot(xc, np.full(N, v.P0), "k-", lw=2, label="theory p = p0 = 10")
ax[1, 0].plot(xc, pz, "o", ms=4, label="numerical, independent Z")
ax[1, 0].plot(xc, pn, "x", ms=5, label="numerical, Z := V*rho_bar^n")
ax[1, 0].set_title("pressure per cell at T")
ax[1, 0].set_xlabel("x")
ax[1, 0].legend(fontsize=8)

Ns = [25, 50, 100, 200, 400]
rows = v.run_cv(eos, nu, T, Ns)
hs = np.array([1.0 / r[0] for r in rows])
err = np.array([r[1] + r[2] for r in rows])
xs = np.linspace(0, 1, 40001)
own = np.array([v.mat_of(xx) for xx in xs])  # material support in the co-moving frame
lead = 0.0
for k in (0, 1):
    n = eos[k][1]
    for g in (v.rho_field(k, xs), v.rho_field(k, xs) ** n):
        d2 = np.gradient(np.gradient(g, xs), xs)
        lead += np.trapezoid(np.abs(d2) * (own == k), xs)
pred = 0.5 * T * v.U0 * (1 - nu) * hs * lead
ax[1, 1].loglog(hs, err, "o-", label="measured L1 (m+Z)")
ax[1, 1].loglog(hs, pred, "s--", label="theory leading term 1/2 T u (1-nu) h ||g''||_1 (own material support)")
ax[1, 1].loglog(hs, [r[3] for r in rows], "k:", label="Theorem CV upper bound")
for hh, e, p in zip(hs, err, pred):
    ax[1, 1].annotate(f"{e/p:.2f}", (hh, e), textcoords="offset points", xytext=(4, 6), fontsize=7)
ax[1, 1].set_xlabel("h  (labels: measured / theory leading term)")
ax[1, 1].set_title("L1 error: measured vs theory")
ax[1, 1].legend(fontsize=7)
fig.tight_layout()
fig.savefig(PNG, dpi=120)
print("ratio measured/leading:", " ".join(f"{e/p:.3f}" for e, p in zip(err, pred)))
print("max|p-p0| with Z:", np.max(np.abs(pz - v.P0)), " without Z:", np.max(np.abs(pn - v.P0)))
print(f"Plot saved: {PNG}")
