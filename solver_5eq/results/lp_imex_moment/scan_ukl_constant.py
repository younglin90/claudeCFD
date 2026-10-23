"""Section 78: UKL (trace) constant of the reduced linear model, K(nu) = sup over the impulse phase theta0 of
S1_inf = sum_n |interface trace_n| (normalised so that the continuum value is 1), for several material pairs and
Courant numbers.  Uses the sparse reduced model (reduced_model_sparse.py); N-independent."""
import os
from multiprocessing import Pool

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import reduced_model_sparse as rms

PNG = "results/lp_imex_moment/ukl_constant.png"
CG = 8.4183
NUS = [0.01, 0.02, 0.035, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3]
THS = [0.05 + 0.1 * k for k in range(10)]
PAIRS = {
    "gas | water (base), CFL_gas 1": (rms.DEFAULT, 1.0 / CG),
    "water | gas (swapped)": (dict(rho_L=1.0, K_L=25.5, rho_R=0.2, K_R=14.012, lam_L=0.6), 1.0 / CG),
    "Z ratio 10 | CFL 1": (dict(rho_L=1.0, K_L=1.0, rho_R=10.0, K_R=10.0, lam_L=1.0), 1.0),
    "Z ratio 0.1 | CFL 1": (dict(rho_L=10.0, K_L=10.0, rho_R=1.0, K_R=1.0, lam_L=1.0), 1.0),
    "gas | water, CFL 0.5": (rms.DEFAULT, 0.5 / CG),
    "gas | water, CFL 2": (rms.DEFAULT, 2.0 / CG),
}


def job(args):
    name, nu, th = args
    P, tau_h = PAIRS[name]
    W = int(min(1500, max(250, 14.0 / nu)))
    tr = rms.run(nu, th, W, P=P, tau_over_h=tau_h)
    return name, nu, th, float(np.abs(tr).sum())


def main():
    jobs = [(n, nu, th) for n in PAIRS for nu in NUS for th in THS]
    with Pool(32) as pool:
        res = pool.map(job, jobs)
    tab = {}
    for name, nu, th, s1 in res:
        tab.setdefault(name, {}).setdefault(nu, []).append(s1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for name in PAIRS:
        mx = [max(tab[name][nu]) for nu in NUS]
        mn = [np.mean(tab[name][nu]) for nu in NUS]
        print(f"{name}: K(nu)=sup_theta S1  " + " ".join(f"{nu}:{m:.2f}" for nu, m in zip(NUS, mx)))
        print(f"   mean_theta S1 " + " ".join(f"{nu}:{m:.2f}" for nu, m in zip(NUS, mn)) +
              "   | sup*nu: " + " ".join(f"{m * nu:.2f}" for nu, m in zip(NUS, mx)))
        ax[0].loglog(NUS, mx, "o-", label=name)
        ax[1].semilogx(NUS, [m * nu for nu, m in zip(NUS, mx)], "o-", label=name)
    ax[0].loglog(NUS, [0.8 / nu for nu in NUS], "k:", label="0.8/nu")
    ax[0].set_xlabel("nu")
    ax[0].set_ylabel("K(nu) = sup_theta0 S1_inf")
    ax[0].set_title("UKL constant of the reduced linear model")
    ax[0].legend(fontsize=6)
    ax[1].set_xlabel("nu")
    ax[1].set_ylabel("K(nu) * nu")
    ax[1].set_title("K nu (flat = 1/nu law)")
    fig.tight_layout()
    fig.savefig(PNG, dpi=110)
    print(f"Plot saved: {PNG}")


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
    main()
