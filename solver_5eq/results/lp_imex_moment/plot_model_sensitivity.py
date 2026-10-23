"""Section 81 figure: S1 of the real scheme vs the reduced linear model with first-order remap and with the unlimited
central-slope (second-order) remap, and the sensitivity of K(nu) to the remap order.  Numbers = outputs of
reduced_model_sparse.py runs (2026-10-23)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PNG = "results/lp_imex_moment/model_sensitivity.png"
CASES = ["nu=.036\nth=.37", "nu=.071\nth=.37", "nu=.119\nth=.37", "nu=.190\nth=.37", "nu=.036\nth=.5 W280", "nu=.119\nth=.5 W280", "nu=.356\nth=.5 W280"]
REAL = np.array([7.180, 5.574, 4.127, 2.083, 11.402, 3.522, 1.480])
M1 = np.array([6.766, 5.288, 3.922, 2.005, 11.774, 3.220, 1.263])
M2 = np.array([7.306, 5.421, 3.969, 2.161, 11.929, 3.127, 1.614])
KNU = [0.01, 0.035, 0.1, 0.3]
K1 = [47.78, 15.13, 4.69, 2.47]
K2 = [53.38, 20.83, 5.35, 2.89]


def main():
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    x = np.arange(len(CASES))
    ax[0].plot(x, M1 / REAL, "o-", label="first-order remap / real")
    ax[0].plot(x, M2 / REAL, "s-", label="central-slope remap / real")
    ax[0].axhline(1, color="k", ls="--")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(CASES, fontsize=7)
    ax[0].set_ylabel("S1 model / S1 real scheme")
    ax[0].set_title("reduced linear model vs real scheme (7 cases)")
    ax[0].legend(fontsize=8)
    ax[1].loglog(KNU, K1, "o-", label="K(nu), first-order remap")
    ax[1].loglog(KNU, K2, "s-", label="K(nu), central-slope remap")
    ax[1].set_xlabel("nu")
    ax[1].set_ylabel("K = sup_theta0 S1_inf (gas | water)")
    ax[1].set_title("sensitivity of the UKL constant to the remap order")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PNG, dpi=110)
    print(f"Plot saved: {PNG}")


if __name__ == "__main__":
    import os
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    main()
