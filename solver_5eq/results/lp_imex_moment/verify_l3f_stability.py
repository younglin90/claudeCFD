"""Theorem L3-F trace (section 62): growth of a grid-scale velocity perturbation (1e-9 sawtooth) in the
interface-fitted scheme, small-amplitude acoustic background, N=50, acoustic CFL 2.
 (a) Fourier model of the force correction: lagged (explicit) vs secant (CORR_IMPLICIT) amplification
 (b) background at rest (u0=0, no remap damping): correction off / lagged / secant
 (c) fullsym secant correction, interface occupancy 1e-1 ... 1e-7 (u0=0): perturbation stays bounded, same level
"""
import importlib.util
import os
import sys
from multiprocessing import Pool

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PNG = "results/lp_imex_moment/l3f_stability.png"
PH_ARGS = [(1.0, 3, 1.5), (0.5, 2, 0.4)]
NST = 300


def run(args):
    mode, u0, occ = args
    spec = importlib.util.spec_from_file_location("fs", os.path.join(HERE, "fitted_scheme.py"))
    fs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fs)
    PH = [fs.Phase(*a) for a in PH_ARGS]
    fs.PARABOLIC[0] = True
    fs.FORCE_CORR[0] = mode != "off"
    fs.CORR_ALL[0] = mode != "off"
    fs.CORR_IMPLICIT[0] = mode in ("secant", "fullsym")
    fs.CORR_STENCIL[0] = "fullsym" if mode == "fullsym" else "raw"
    N = 50
    h = 1.0 / N
    X1, X2 = 0.3 + occ * h, 0.7 + occ * h

    def mk(eps):
        st = fs.build(N, PH, lambda x: 0 if X1 <= x < X2 else 1, lambda k, x: (1.0 + 0 * x) if k == 0 else 0.2 + 0 * x,
                      lambda x: 10 * (1 + 0.002 * np.cos(2 * np.pi * x)), lambda x: u0 + 0.01 * np.sin(2 * np.pi * x),
                      [X1, X2])
        if eps:
            st.w = st.w + eps * (-1.0) ** np.arange(len(st.w))
        return st
    cmax = np.sqrt((2 * 0.5 * 0.2 + 1.4 * (10.5 - 0.02)) / 0.2)
    tau = 2.0 * h / cmax
    a, b = mk(0), mk(1e-9)
    def faces(st):
        # Eulerian face velocities keyed by face index (the interface may sit on either side of a face
        # in the two runs, so boundary indices are not comparable)
        return {int(round((z[1] % 1.0) / h)) % N: st.w[b] for b, z in enumerate(st.zones) if st.isface[b]}
    hist = []
    for n in range(NST):
        a, _ = fs.step(a, tau, PH, N)
        b, _ = fs.step(b, tau, PH, N)
        fa, fb = faces(a), faces(b)
        hist.append(max(abs(fa[j] - fb[j]) for j in fa if j in fb))
    return mode, u0, occ, np.array(hist)


def fourier():
    ks = np.linspace(0.01, np.pi, 300)
    out = {}
    for lam in (0.5, 2.0, 9.6):
        for impl in (False, True):
            g = []
            for k in ks:
                th, sg = 2 * lam * np.sin(k / 2), np.sin(k / 2) ** 2 / 6
                if impl:
                    A = np.array([[1, 1j * th * (1 + sg) / 2], [1j * th / 2, 1]])
                    B = np.array([[1, -1j * th * (1 + sg) / 2], [-1j * th / 2, 1]])
                else:
                    A = np.array([[1, 1j * th / 2], [1j * th / 2, 1]])
                    B = np.array([[1, -1j * th / 2 - 1j * th * sg], [-1j * th / 2, 1]])
                g.append(max(abs(np.linalg.eigvals(np.linalg.solve(A, B)))))
            out[(lam, impl)] = np.array(g)
    return ks, out


def main():
    jobs = [(m, 0.0, 1e-5) for m in ("off", "lagged", "secant", "fullsym")]
    jobs += [("fullsym", 0.0, o) for o in (1e-1, 1e-3, 1e-7)]
    with Pool(min(7, os.cpu_count() or 1)) as pool:
        res = pool.map(run, jobs)
    for mode, u0, occ, hst in res:
        print(f"{mode:7s} u0={u0} occ={occ:.0e}: max|dw| at steps 50/150/300 = "
              f"{hst[49]:.2e} {hst[149]:.2e} {hst[-1]:.2e}", flush=True)
    ks, F = fourier()
    for (lam, impl), g in F.items():
        print(f"Fourier lam={lam} {'secant' if impl else 'lagged'}: max|G| = {g.max():.6f}")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    for (lam, impl), g in F.items():
        ax[0].plot(ks / np.pi, g, "-" if impl else "--", label=f"{'secant' if impl else 'lagged'}, CFL {lam}")
    ax[0].set_xlabel("k h / pi")
    ax[0].set_ylabel("|G| (Fourier, rest state)")
    ax[0].set_title("(a) force correction: lagged vs secant")
    ax[0].legend(fontsize=7)
    steps = np.arange(1, NST + 1)
    for mode, u0, occ, hst in res:
        if occ == 1e-5:
            ax[1].semilogy(steps, hst, label=f"correction {mode}")
        if mode == "fullsym":
            ax[2].semilogy(steps, hst, label=f"fullsym, occupancy {occ:.0e}")
    ax[1].set_title("(b) u0=0: grid-noise 1e-9, max|dw|")
    ax[2].set_title("(c) fullsym secant correction vs occupancy (u0=0)")
    for a in ax[1:]:
        a.set_xlabel("step")
        a.set_ylabel("max |w_perturbed - w|")
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(PNG, dpi=110)
    print(f"Plot saved: {PNG}")


if __name__ == "__main__":
    os.chdir(os.path.join(HERE, "..", ".."))
    main()
