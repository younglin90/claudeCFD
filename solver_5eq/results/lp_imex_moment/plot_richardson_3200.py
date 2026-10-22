"""Section 70: successive differences of the no-correction interface-fitted scheme (compatible pulse) up to
N=3200 (verify_pulse_richardson.py --parabolic --to3200, run 2026-10-22)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PNG = "results/lp_imex_moment/richardson_3200.png"
N = np.array([200, 400, 800, 1600])
D = {"rho L2": [6.422e-04, 2.233e-04, 6.285e-05, 1.648e-05], "p L2": [1.765e-02, 6.134e-03, 1.717e-03, 4.494e-04],
     "w L2": [5.047e-03, 1.701e-03, 4.681e-04, 1.218e-04], "rho Linf near interface": [6.399e-04, 2.795e-05, 1.191e-05, 7.325e-06]}


def main():
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    h = 1.0 / N
    for k, v in D.items():
        v = np.array(v)
        ax[0].loglog(h, v / v[0], "o-", label=k)
        ax[1].plot(N[1:], np.log2(v[:-1] / v[1:]), "o-", label=k)
    ax[0].loglog(h, (h / h[0]) ** 2, "k--", label="slope 2")
    ax[0].loglog(h, (h / h[0]) ** 1.5, "k:", label="slope 1.5")
    ax[0].set_xlabel("h")
    ax[0].set_ylabel("d_N / d_200")
    ax[0].legend(fontsize=7)
    ax[0].set_title("successive differences, full scheme with remap")
    ax[1].axhline(2, color="k", ls="--")
    ax[1].axhline(1.5, color="k", ls=":")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("N (pair N/2, N)")
    ax[1].set_ylabel("observed order")
    ax[1].legend(fontsize=7)
    ax[1].set_title("order trend")
    fig.tight_layout()
    fig.savefig(PNG, dpi=110)
    print(f"Plot saved: {PNG}")


if __name__ == "__main__":
    import os
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    main()
