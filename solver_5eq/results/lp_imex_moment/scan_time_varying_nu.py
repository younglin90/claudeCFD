"""Section 80: time-dependent interface Courant number nu(t) = nu0 cos(2 pi n / P + phase), including sign changes
(interface reversing, |u_I| not bounded below: the assumption (U) of section 66 is violated).  S1 = sum |trace| of the
reduced linear model for an impulse at step 0; sup over the start phase and theta0 compared with the constant-nu
constants K(nu) of section 78."""
import os
import sys
from multiprocessing import Pool

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reduced_model_sparse as rms

PNG = "results/lp_imex_moment/time_varying_nu.png"


def job(args):
    nu0, P, ph, th0, W = args
    n = np.arange(W)
    nu = nu0 * np.cos(2 * np.pi * n / P + ph)
    tr = rms.run(nu, th0, W)
    return nu0, P, ph, th0, float(np.abs(tr).sum()), float(np.abs(tr[:50]).sum())


def main():
    W = 700
    phases = [2 * np.pi * k / 12 for k in range(12)]
    ths = [0.1, 0.37, 0.7]
    jobs = [(nu0, P, ph, th, W) for nu0 in (0.1, 0.3) for P in (100, 400) for ph in phases for th in ths]
    with Pool(32) as pool:
        res = pool.map(job, jobs)
    out = {}
    for nu0, P, ph, th, s1, s50 in res:
        out.setdefault((nu0, P), []).append((ph, th, s1))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for (nu0, P), v in out.items():
        arr = np.array(v)
        mx, mean = arr[:, 2].max(), arr[:, 2].mean()
        j = int(np.argmax(arr[:, 2]))
        print(f"nu(t)={nu0} cos(2 pi n/{P}+phase): S1 over (phase, theta0): max {mx:.1f} (phase {arr[j, 0] / np.pi:.2f} pi, theta0 {arr[j, 1]}), mean {mean:.1f}, min {arr[:, 2].min():.1f}")
        ph_axis = sorted(set(arr[:, 0]))
        ax[0].plot(np.array(ph_axis) / np.pi, [arr[arr[:, 0] == p, 2].max() for p in ph_axis], "o-", label=f"nu0={nu0}, period {P}")
    ax[0].set_xlabel("start phase / pi")
    ax[0].set_ylabel("S1 (max over theta0)")
    ax[0].set_title("oscillating interface speed (sign changes)")
    ax[0].legend(fontsize=7)
    # constant-nu reference: S1 at nu = |nu(0)|
    ref = []
    for nu in (0.01, 0.02, 0.05, 0.1, 0.2, 0.3):
        ref.append((nu, max(np.abs(rms.run(nu, th, 400)).sum() for th in (0.1, 0.37, 0.7))))
    ax[1].loglog([r[0] for r in ref], [r[1] for r in ref], "o-", label="constant nu (sup over theta0 in {0.1, 0.37, 0.7})")
    ax[1].set_xlabel("nu")
    ax[1].set_ylabel("S1")
    ax[1].set_title("constant-nu reference")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(PNG, dpi=110)
    print("constant-nu reference:", [(r[0], round(r[1], 1)) for r in ref])
    print(f"Plot saved: {PNG}")


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    main()
