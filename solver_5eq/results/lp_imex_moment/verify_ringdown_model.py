"""Section 76 (task 2): ring-down model of the interface trace S1 ~ S_c + A (1 - r^W)/(1 - r), r = 1 - delta.
 A     per-step amplitude of the trapped-mode ring (normalised trace) at nu = 0  (residue of the pole on |z| = 1)
 delta damping per step of the ring by the remap; model: grid-scale content is remapped by first-order upwind
       (minmod -> 0 on a sawtooth), amplification |1 - nu + nu e^{i pi}| = 1 - 2 nu  =>  delta = 2 nu.
Measured here with the real scheme: A(theta) at u0 = 0, delta from the envelope decay at u0 > 0, and the
predicted vs measured S1 over a nu scan."""
import importlib.util
import os
from multiprocessing import Pool

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("oi", os.path.join(HERE, "verify_open_items.py"))
oi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oi)
PNG = "results/lp_imex_moment/ringdown_model.png"


def series(args):
    N, u0, theta, W, eps = args
    h = 1.0 / N
    tau = h / oi.CG
    st = oi.tm.make(N, u0, theta, 0)
    b = oi.iface(st)
    m = np.array([z[3] for z in st.zones])
    MI = 0.5 * (m[b - 1] + m[b])
    st.w[b] += eps
    w = []
    for n in range(W):
        st, _ = oi.fs.step(st, tau, oi.PH, N)
        w.append(st.w[oi.iface(st)] - u0)
    norm = (oi.ZL + oi.ZR) * tau / (MI * eps)
    return np.array(w) * norm, u0 * tau / h


def amp(args):
    w, nu = series(args)
    return np.abs(w[10:]).mean(), nu


def decay(args):
    w, nu = series(args)
    a = np.abs(w)
    env = np.array([a[max(0, n - 3):n + 4].max() for n in range(len(a))])
    lo, hi = 8, min(len(a) - 4, 8 + int(3.0 / max(nu, 1e-3)))
    n = np.arange(lo, hi)
    keep = env[lo:hi] > 1e-4 * env[lo]
    slope = np.polyfit(n[keep], np.log(env[lo:hi][keep]), 1)[0]
    return -slope, nu


def main():
    with Pool(16) as pool:
        thetas = [0.05, 0.1, 0.3, 0.5, 0.7, 0.9]
        A = pool.map(amp, [(400, 0.0, th, 140, 1e-4) for th in thetas])
        print("A(theta) at nu=0 (N=400, mean |w_trace| per step after 10 steps):",
              {th: round(a[0], 3) for th, a in zip(thetas, A)}, flush=True)
        Ah = dict(zip(thetas, [a[0] for a in A]))
        u0s = [0.15, 0.3, 0.6, 1.0, 2.0]
        D = pool.map(decay, [(400, u, 0.5, 200, 1e-4) for u in u0s])
        print("envelope decay per step delta, delta/nu, delta vs 2 nu:",
              [(round(nu, 4), round(d, 4), round(d / nu, 2)) for d, nu in D], flush=True)
        u0scan = [0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0]
        W = 70
        S = pool.map(series, [(200, u, 0.37, W, 1e-4) for u in u0scan])
    nus = np.array([s[1] for s in S])
    s1 = np.array([np.abs(s[0]).sum() for s in S])
    Abar = float(np.mean([Ah[t] for t in thetas]))
    kappa = float(np.median([d[0] / d[1] for d in D if d[1] < 0.08]))
    print(f"theta-average of A(theta) = {Abar:.3f}; kappa = median(delta/nu, nu<0.08) = {kappa:.2f}; "
          f"closure S1*nu -> Abar/kappa = {Abar / kappa:.3f}")
    for nu, a in zip(nus, s1):
        print(f"  nu={nu:.4f}: measured S1 {a:.3f}  S1*nu = {a * nu:.3f}")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].plot(thetas, [Ah[t] for t in thetas], "o-")
    ax[0].set_xlabel("theta")
    ax[0].set_ylabel("ring amplitude per step A")
    ax[0].set_title("A(theta) at nu=0")
    nu_d = np.array([d[1] for d in D])
    ax[1].plot(nu_d, [d[0] for d in D], "o-", label="measured envelope decay")
    ax[1].plot(nu_d, 2 * nu_d, "k--", label="2 nu (first-order upwind, grid scale)")
    ax[1].set_xlabel("nu")
    ax[1].set_ylabel("delta per step")
    ax[1].legend(fontsize=7)
    ax[1].set_title("ring damping per step")
    ax[2].plot(nus, s1 * nus, "o-", label="measured S1*nu (N=200, W=70)")
    ax[2].axhline(Abar / kappa, color="k", ls="--", label=f"closure A_bar/kappa = {Abar / kappa:.2f}")
    ax[2].set_xlabel("nu")
    ax[2].set_ylabel("S1 * nu")
    ax[2].set_ylim(0, 0.8)
    ax[2].legend(fontsize=7)
    ax[2].set_title("closure S1 ~ A_bar / (kappa nu)")
    fig.tight_layout()
    fig.savefig(PNG, dpi=110)
    print(f"Plot saved: {PNG}")


if __name__ == "__main__":
    os.chdir(os.path.join(HERE, "..", ".."))
    main()
