"""Path (a) of section 41: is the discrete H1 seminorm of the difference of two
finite-amplitude solutions bounded in time, independently of the occupancy?
Reports G_e(t) = ||D_x dp||_2 / ||dp||_2 (dp = pressure difference, D_x = cell difference / h)
and the same for the nodal velocity difference."""
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


va = load("va", "verify_acoustic.py")
vk = load("vk", "verify_kapila_scheme.py")
vn = load("vn", "verify_nonlinear_lipschitz.py")
PNG = "results/lp_imex_moment/h1_error_occupancy.png"


def run(shift, N=100, lam=2.0, T=0.06, sigma=1e-6):
    h = 1.0 / N
    U = vn.make_state(N, shift, sigma)
    V = vn.make_state(N, shift, 0.0)
    cmax = np.sqrt((2 * 0.5 * 0.2 + 1.4 * (10.5 - 0.02)) / 0.2)
    tau = min(lam * h / cmax, 0.5 * h / 0.8)
    nst = int(np.ceil(T / tau))
    tau = T / nst
    Gp, Gw = [], []
    for n in range(nst + 1):
        if n:
            U, _, _ = vk.step_kapila(U, vn.PH, tau)
            V, _, _ = vk.step_kapila(V, vn.PH, tau)
        dp = vn.pressures_relaxed(U, h) - vn.pressures_relaxed(V, h)
        dw = U["w"] - V["w"]
        Gp.append(np.linalg.norm(np.diff(np.r_[dp, dp[0]]) / h) / np.linalg.norm(dp))
        Gw.append(np.linalg.norm(np.diff(np.r_[dw, dw[0]]) / h) / max(np.linalg.norm(dw), 1e-300))
    return np.array(Gp), np.array(Gw), tau * np.arange(nst + 1)


def main():
    res = {}
    for shift in (0.5, 1e-2, 1e-4, 1e-6):
        Gp, Gw, t = run(shift)
        res[shift] = (Gp, Gw, t)
        print(f"shift={shift:.0e}: G_p(0)={Gp[0]:.2f} max G_p={Gp.max():.2f} final={Gp[-1]:.2f} | "
              f"max G_w (t>0)={Gw[1:].max():.2f} final={Gw[-1]:.2f}   (smooth-mode reference 4*pi={4*np.pi:.2f})")
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    for shift, (Gp, Gw, t) in res.items():
        ax[0].plot(t, Gp, label=f"shift {shift:.0e}")
        ax[1].plot(t[1:], Gw[1:], label=f"shift {shift:.0e}")
    for a_, tt in zip(ax, ("pressure error ||D_x dp|| / ||dp||", "velocity error ||D_x dw|| / ||dw||")):
        a_.axhline(4 * np.pi, color="k", ls=":", label="initial mode wavenumber 4 pi")
        a_.set_xlabel("time")
        a_.set_title(tt)
        a_.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(PNG, dpi=115)
    print(f"Plot saved: {PNG}")


if __name__ == "__main__":
    main()
