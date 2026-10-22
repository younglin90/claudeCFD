"""Section 75 summary figure of the four feasibility tests (numbers = printed outputs of verify_open_items.py,
2026-10-23; T3 stronger gradients from the same pulse test with EPS = 0.04, 0.08 up to N = 1600)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PNG = "results/lp_imex_moment/open_items.png"
T1 = {"u0=0 (nu=0, no remap)": [27.065, 53.979, 106.595, 213.067], "nu=0.036": [11.007, 11.241, 11.279, 11.402],
      "nu=0.119": [3.421, 3.425, 3.458, 3.522], "nu=0.358": [1.422, 1.438, 1.449, 1.480],
      "nu=0.119, theta=0.05": [2.330, 2.330, 2.347, 2.374]}
T2 = [(0.012, 24.844), (0.043, 6.791), (0.073, 5.665), (0.104, 4.295), (0.125, 4.051), (0.135, 3.267), (0.165, 2.260),
      (0.185, 2.102), (0.196, 2.108), (0.226, 2.201), (0.250, 2.159), (0.257, 2.159), (0.288, 2.093), (0.318, 1.509),
      (0.333, 1.395), (0.349, 1.361), (0.380, 1.363), (0.410, 1.533), (0.441, 1.516), (0.472, 1.583), (0.500, 1.711),
      (0.502, 1.717), (0.533, 1.844), (0.564, 1.937), (0.594, 2.208), (0.625, 2.473), (0.656, 2.262), (0.686, 2.097),
      (0.717, 2.082)]
T3 = {"EPS=0.04": [1.00, 1.28, 1.60], "EPS=0.08": [0.77, 0.75, 0.83]}   # p-order per pair 100/200, 200/400, 400/800
T4 = {0.1: 0.9005, 0.3: 0.8360, 0.5: 0.8142, 0.7: 0.8360, 0.9: 0.9005}


def main():
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.3))
    Ns = [100, 200, 400, 800]
    for k, v in T1.items():
        ax[0].loglog(Ns, v, "o-", label=k)
    ax[0].loglog(Ns, 0.27 * np.array(Ns), "k:", label="slope 1")
    ax[0].set_xlabel("N")
    ax[0].set_ylabel("S1 = sum |w_trace| / continuum value")
    ax[0].set_title("T1 real scheme: interface impulse response")
    ax[0].legend(fontsize=6)
    nu, s1 = zip(*T2)
    ax[1].semilogy(nu, s1, "o-")
    ax[1].semilogy(nu, 0.4 / np.array(nu), "k:", label="0.4 / nu")
    ax[1].set_xlabel("nu = u tau / h")
    ax[1].set_ylabel("S1")
    ax[1].set_title("T2: scan over nu (N=200), no rational spikes")
    ax[1].legend(fontsize=7)
    for k, v in T3.items():
        ax[2].plot([100, 200, 400], v, "o-", label=k)
    ax[2].plot([100, 200, 400], [1.06, 1.52, 1.83], "s--", label="EPS=0.02 (baseline, d200/d400 -> d400/d800)")
    ax[2].axhline(2, color="k", ls="--")
    ax[2].set_xlabel("N of the coarser grid in the pair")
    ax[2].set_ylabel("observed p order")
    ax[2].set_title("T3: stronger gradients (pre-asymptotic)")
    ax[2].legend(fontsize=6)
    th = np.array(sorted(T4))
    ax[3].plot(th, [T4[t] for t in th], "o-", label="trapped eigenvalue arg z / pi")
    ax[3].plot(np.linspace(0.05, 0.95, 100), 2 * np.arctan(2.0 / np.minimum(np.linspace(0.05, 0.95, 100), 1 - np.linspace(0.05, 0.95, 100))) / np.pi,
               "k--", label="local band edge 2 arctan(Lam/min(theta,1-theta))/pi")
    ax[3].axhline(2 * np.arctan(2.0) / np.pi, color="gray", ls="--", label="global band edge")
    ax[3].set_xlabel("theta")
    ax[3].set_ylabel("arg z / pi")
    ax[3].set_title("T4: trapped mode = cavity mode of the short cut zones")
    ax[3].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(PNG, dpi=110)
    print(f"Plot saved: {PNG}")


if __name__ == "__main__":
    import os
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    main()
