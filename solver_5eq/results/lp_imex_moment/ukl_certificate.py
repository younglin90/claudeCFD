"""Section 84: numerical UKL (strong-stability) certificate of the reduced linear model in the GKS form.
Kernel T(n; theta0)_{ij}: response (channel i, after n steps) to a unit-energy impulse (channel j) at the interface,
channels = (interface-node velocity, left cut-piece pressure, right cut-piece pressure), energy-normalised.
Row bound   S1 = sup_theta0 max_ij sum_n |T_ij|        (impulse at fixed phase, summed over time)
Column bound S2 = sup_theta_out max_ij sum_k |T_ij(k; theta_out - k nu)|  (fixed output phase, summed over input times)
Schur test: the input->trace operator has l2 -> l2 norm <= sqrt(S1 S2) (entrywise; 3x3 block bound by the matrix 2-norm of
the entry bounds).  This is the discrete counterpart of the GKS trace estimate  sum |trace|^2 <= K sum |g|^2.
The phase grid is interpolated linearly in theta0."""
import os
import sys
from multiprocessing import Pool

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reduced_model_sparse as rms

PNG = "results/lp_imex_moment/ukl_certificate.png"
THG = [(k + 0.5) / 40 for k in range(40)]


def job(args):
    nu, th, order = args
    W = int(min(900, max(200, 12.0 / nu)))
    return nu, th, order, rms.run_channels(nu, th, W, order=order)


def main():
    nus = [0.02, 0.05, 0.1, 0.2]
    jobs = [(nu, th, o) for nu in nus for th in THG for o in (1, 2)]
    with Pool(32) as pool:
        res = pool.map(job, jobs)
    tab = {}
    for nu, th, o, T in res:
        tab[(nu, o, th)] = T
    out = {}
    for o in (1, 2):
        for nu in nus:
            Wm = min(t.shape[0] for (n_, o_, th), t in tab.items() if n_ == nu and o_ == o)
            Tall = np.array([tab[(nu, o, th)][:Wm] for th in THG])        # (theta0, n, i, j)
            S1m = np.abs(Tall).sum(axis=1).max(axis=0)                   # (i,j) sup over theta0 of sum_n
            # column bound: fixed output phase, lag k, input phase theta_out - k nu
            S2m = np.zeros((3, 3))
            thg = np.array(THG)
            for tho in np.linspace(0, 1, 80, endpoint=False):
                acc = np.zeros((3, 3))
                for k in range(Wm):
                    ph = (tho - (k + 1) * nu) % 1.0
                    j0 = int(np.floor(ph * 40 - 0.5)) % 40
                    j1 = (j0 + 1) % 40
                    fr = (ph * 40 - 0.5) - np.floor(ph * 40 - 0.5)
                    acc += (1 - fr) * np.abs(Tall[j0, k]) + fr * np.abs(Tall[j1, k])
                S2m = np.maximum(S2m, acc)
            Sch = np.sqrt(np.linalg.norm(S1m, 2) * np.linalg.norm(S2m, 2))
            out[(o, nu)] = (S1m, S2m, Sch)
            print(f"order {o} nu={nu}: max S1_ij={S1m.max():.2f} max S2_ij={S2m.max():.2f} "
                  f"l2 gain bound sqrt(||S1||2||S2||2) = {Sch:.2f}   (gain*nu = {Sch * nu:.3f})", flush=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for o in (1, 2):
        ax.loglog(nus, [out[(o, nu)][2] for nu in nus], "o-", label=f"remap order {o}")
    ax.loglog(nus, [0.8 / nu for nu in nus], "k:", label="0.8/nu")
    ax.set_xlabel("nu")
    ax.set_ylabel("l2 trace-gain bound sqrt(S1 S2)")
    ax.set_title("GKS-form trace estimate constant of the reduced model")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PNG, dpi=110)
    print(f"Plot saved: {PNG}")


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    main()
