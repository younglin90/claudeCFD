"""Section 74: does the remap damp interface-localised grid-scale perturbations of the real scheme?
Uniform two-material background (exactly preserved), grid-scale sawtooth in the node velocities within 3 cells
of the left interface, then free evolution.  Reported: perturbation energy (kinetic + l/K dp^2) relative to its
initial value, for u0 = 0 (remap inactive) and u0 > 0, and for two amplitudes (linearity check)."""
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
PNG = "results/lp_imex_moment/trapped_mode.png"
PH = [fs.Phase(1.0, 3, 1.5), fs.Phase(0.5, 2, 0.4)]
P0, RG, RW = 10.0, 0.2, 1.0
CG = np.sqrt((2 * 0.5 * RG + 1.4 * (P0 - 0.5 * RG ** 2) / RG))


def make(N, u0, theta, eps):
    h = 1.0 / N
    X1, X2 = 0.3 + theta * h, 0.7 + theta * h
    st = fs.build(N, PH, lambda x: 0 if X1 <= x < X2 else 1, lambda k, x: (RW + 0 * x) if k == 0 else RG + 0 * x,
                  lambda x: P0 + 0 * x, lambda x: u0 + 0 * x, [X1, X2])
    if eps:
        bI = [b for b in range(len(st.zones)) if not st.isface[b] and abs(st.zones[b][1] - X1) < 1e-12][0]
        for j in range(-3, 4):
            st.w[(bI + j) % len(st.w)] += eps * (-1.0) ** j
    return st


def keyed(st, N):
    """boundary key: grid index for face nodes, ('I', k) for interface nodes (list rotation / wrap independent)."""
    h = 1.0 / N
    inter = sorted([b for b in range(len(st.zones)) if not st.isface[b]], key=lambda b: st.zones[b][1] % 1.0)
    rank = {b: k for k, b in enumerate(inter)}
    return {(int(round((st.zones[b][1] % 1.0) / h)) % N if st.isface[b] else ("I", rank[b])): b
            for b in range(len(st.zones))}


def energy(sa, sb, N, local=None):
    ka, kb = keyed(sa, N), keyed(sb, N)
    if set(ka) != set(kb):
        return np.nan
    m = np.array([z[3] for z in sa.zones])
    M = 0.5 * (m + np.roll(m, 1))
    e2 = 0.0
    for key, ia in ka.items():
        ib = kb[key]
        if local is not None:
            xk = sa.zones[ia][1] % 1.0
            if abs(xk - local) > 6.5 / N:
                continue
        e2 += 0.5 * M[ia] * (sa.w[ia] - sb.w[ib]) ** 2
        za, zb = sa.zones[ia], sb.zones[ib]
        if za[0] != zb[0]:
            return np.nan
        ph = PH[za[0]]
        pf, Ef, l0 = fs._isentrope(za, ph)
        K = -l0 * (pf(l0 * (1 + 1e-7)) - pf(l0 * (1 - 1e-7))) / (2e-7 * l0)
        e2 += 0.5 * (za[2] - za[1]) / K * (fs.zone_p(za, ph) - fs.zone_p(zb, ph)) ** 2
    return np.sqrt(e2)


def run(N, u0, theta, eps, nst, local=False):
    h = 1.0 / N
    tau = 2.0 * h / CG
    a, b = make(N, u0, theta, 0), make(N, u0, theta, eps)
    X1 = 0.3 + theta / N
    lc = X1 if local else None
    e0 = energy(a, b, N)
    e0l = energy(a, b, N, lc) if local else e0
    out = [1.0]
    for n in range(nst):
        a, _ = fs.step(a, tau, PH, N)
        b, _ = fs.step(b, tau, PH, N)
        out.append(energy(a, b, N, lc) / e0 if local else energy(a, b, N) / e0)
    return np.array(out), u0 * tau / h


def main():
    for th in (0.5, 0.05):
        for u0 in (0.0, 0.05, 0.3):
            r, nu = run(100, u0, th, 1e-3, 30, local=True)
            print(f"LOCAL energy within +-6 cells of interface 1, N=100 theta={th} u0={u0} (nu={nu:.3f}): "
                  f"steps 0/5/10/15/20/30: {np.round(r[[0, 5, 10, 15, 20, 30]], 4)}", flush=True)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for N, u0, th in ((100, 0.0, 0.5), (100, 0.3, 0.5), (100, 1.0, 0.5), (200, 0.3, 0.5), (100, 0.3, 0.05)):
        r1, nu = run(N, u0, th, 1e-3, 120)
        r2, _ = run(N, u0, th, 1e-4, 120)
        print(f"N={N} u0={u0} nu={nu:.3f} theta={th}: energy ratio after 10/40/120 steps: "
              f"{r1[10]:.3e} {r1[40]:.3e} {r1[120]:.3e}  (amplitude 1e-4: {r2[10]:.3e} {r2[40]:.3e} {r2[120]:.3e})",
              flush=True)
        ax[0].semilogy(r1, label=f"N={N}, u0={u0} (nu={nu:.2f}), theta={th}")
    ax[0].set_xlabel("step")
    ax[0].set_ylabel("perturbation energy / initial")
    ax[0].set_title("interface-localised grid-scale perturbation, real scheme")
    ax[0].legend(fontsize=7)
    for u0 in (0.05, 0.1, 0.3, 1.0):
        r, nu = run(100, u0, 0.5, 1e-3, 60)
        ax[1].semilogy(r, label=f"u0={u0} (nu={nu:.3f})")
    ax[1].set_xlabel("step")
    ax[1].set_ylabel("perturbation energy / initial")
    ax[1].set_title("decay vs interface-grid speed (N=100)")
    ax[1].legend(fontsize=7)
    for u0 in (0.0, 0.05, 0.3):
        r, nu = run(100, u0, 0.5, 1e-3, 30, local=True)
        ax[2].semilogy(r, label=f"u0={u0} (nu={nu:.3f})")
    ax[2].set_xlabel("step")
    ax[2].set_ylabel("local perturbation energy (+-6 cells) / initial total")
    ax[2].set_title("energy near the interface (trapping test), theta=0.5")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(PNG, dpi=110)
    print(f"Plot saved: {PNG}")


if __name__ == "__main__":
    os.chdir(os.path.join(HERE, "..", ".."))
    main()
