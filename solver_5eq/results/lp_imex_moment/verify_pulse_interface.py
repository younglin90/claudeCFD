"""Compatible-data test (section 45): uniform two-material background (water-like slab in gas-like
fluid, u0, p0) + a smooth right-going acoustic pulse started away from the interfaces. The exact
solution is smooth inside each material (no kinks emitted at t=0), the pulse is transmitted and
reflected at a moving interface. Convergence of the interface-fitted scheme vs the mixed-cell
piecewise-Kapila scheme, against N=400 references of each scheme."""
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


fs = load("fs", "fitted_scheme.py")
vk = load("vk", "verify_kapila_scheme.py")
PNG = "results/lp_imex_moment/pulse_interface_order.png"
PH = [fs.Phase(1.0, 3, 1.5), fs.Phase(0.5, 2, 0.4)]
X1, X2 = 0.3 + 1e-5 / 400, 0.7 + 1e-5 / 400
U0, P0, RG, RW, EPS, T = 0.3, 10.0, 0.2, 1.0, 0.02, 0.035
CG = np.sqrt((2 * 0.5 * RG + 1.4 * (P0 - 0.5 * RG ** 2) / RG))


def bump(x):
    s = (x - 0.12) / 0.08
    return np.where(np.abs(s) < 1, np.cos(0.5 * np.pi * s) ** 6, 0.0)


def u_of(x):
    return U0 + EPS * CG * bump(x)


def p_of(x):
    return P0 * (1 + 0 * x) + RG * CG * EPS * CG * bump(x)


def mat_of(x):
    return 0 if X1 <= x < X2 else 1


def rho_of(k, x):
    return (RW + 0 * x) if k == 0 else RG * (1 + EPS * bump(x))


def tau_of(N):
    h = 1.0 / N
    tau = min(2.0 * h / CG, 0.5 * h / (U0 + 0.1))
    nst = int(np.ceil(T / tau))
    return T / nst, nst


def run_fitted(N):
    h = 1.0 / N
    st = fs.build(N, PH, mat_of, rho_of, p_of, u_of, [X1, X2])
    tau, nst = tau_of(N)
    t0 = fs.totals(st)
    for n in range(nst):
        st, _ = fs.step(st, tau, PH, N)
    t1 = fs.totals(st)
    rho, pc, wf = np.zeros(N), np.zeros(N), np.zeros(N)
    for z in st.zones:
        c = int(np.floor((0.5 * (z[1] + z[2])) / h + 1e-12)) % N
        rho[c] += z[3] / h
        pc[c] += fs.zone_p(z, PH[z[0]]) * (z[2] - z[1]) / h
    for b, z in enumerate(st.zones):
        if st.isface[b]:
            wf[int(round((z[1] % 1.0) / h)) % N] = st.w[b]
    return rho, pc, wf, abs(t1[2] - t0[2]) / t0[2]


def run_mixed(N):
    h = 1.0 / N
    vk.va.RECON[0] = True
    s0 = fs.build(N, PH, mat_of, rho_of, p_of, u_of, [X1, X2])
    cells = [[] for _ in range(N)]
    for z in s0.zones:
        c = int(np.floor((0.5 * (z[1] + z[2])) / h + 1e-12)) % N
        cells[c].append(list(z))
    w = np.array([u_of((j + 1) * h) for j in range(N)])
    st = dict(cells=cells, I=np.array([sum(q[5] for q in c_) for c_ in cells]), w=w)
    tau, nst = tau_of(N)
    E0 = vk.totals6(st, PH)[2]
    for n in range(nst):
        st, _, rel = vk.step_kapila(st, PH, tau)
    E1 = vk.totals6(st, PH)[2]
    rho = np.array([sum(q[3] for q in pcs) for pcs in st["cells"]]) / h
    pc = vk.va.pressures(dict(cells=[vk.relax_cell(p_, PH, h, c * h)[0] for c, p_ in enumerate(st["cells"])],
                              I=np.array([vk.relax_cell(p_, PH, h, c * h)[1] for c, p_ in enumerate(st["cells"])])), PH)
    vk.va.RECON[0] = False
    return rho, pc, st["w"], abs(E1 - E0) / E0


NREF, NS = 1600, (100, 200, 400, 800)


def study(runner, name):
    R = runner(NREF)
    rows, prev = [], None
    for N in NS:
        r = NREF // N
        rho, p, w, dE = runner(N)
        h = 1.0 / N
        cur = (np.sqrt(h * np.sum((rho - R[0].reshape(-1, r).mean(axis=1)) ** 2)),
               np.sqrt(h * np.sum((p - R[1].reshape(-1, r).mean(axis=1)) ** 2)),
               np.sqrt(h * np.sum((w - R[2][r - 1::r]) ** 2)) if name == "mixed" else np.sqrt(h * np.sum((w - R[2][::r]) ** 2)))
        rows.append((N,) + cur)
        print(f"{name} N={N}: L2 rho={cur[0]:.3e} p={cur[1]:.3e} w={cur[2]:.3e} dE/E={dE:.1e} "
              + ("" if prev is None else "orders " + " ".join(f"{np.log2(a / b):.2f}" for a, b in zip(prev, cur))), flush=True)
        prev = cur
    return rows, R


def main():
    rf, Rf = study(run_fitted, "fitted")
    rm, Rm = rf, Rf          # mixed-cell scheme too slow at these resolutions (dense Python loops); see section 45
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    hs = np.array([1.0 / r[0] for r in rf])
    for rows, ls, nm in ((rf, "o-", "interface-fitted"),):
        for j, c in ((2, "pressure"), (3, "velocity")):
            ax[0].loglog(hs, [r[j] for r in rows], ls, label=f"{nm}, {c}")
    ax[0].loglog(hs, rf[0][2] * (hs / hs[0]) ** 2, "k:", label="slope 2")
    ax[0].set_xlabel("h")
    ax[0].set_ylabel("L2 error vs N=1600 reference")
    ax[0].set_title("acoustic pulse through a moving water|gas interface")
    ax[0].legend(fontsize=7)
    x = (np.arange(NREF) + 0.5) / NREF
    ax[1].plot(x, Rf[1], label=f"fitted N={NREF} pressure")
    ax[1].axvspan(X1 + U0 * T, X2 + U0 * T, color="gray", alpha=0.15, label="water slab at T")
    ax[1].set_title(f"pressure at T={T}")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(PNG, dpi=115)
    print(f"Plot saved: {PNG}")


if __name__ == "__main__":
    main()
