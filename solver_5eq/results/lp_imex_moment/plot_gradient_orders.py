"""Section 76 (task 1) figure: successive-difference orders of the p field for the compatible pulse with
amplitude EPS = 0.02 (earlier N-sequence), 0.04 and 0.08 (this section), PARABOLIC on, no force correction.
Numbers = outputs of verify_gradient_asymptotics.py (2026-10-23)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PNG = "results/lp_imex_moment/gradient_orders.png"
ORD = {"EPS=0.02": ([200, 400, 800], [1.52, 1.83, 1.93]), "EPS=0.04": ([400, 800], [1.60, 1.81]),
       "EPS=0.08": ([400, 800], [0.83, 0.89])}
STEEP = {0.04: 2028.6, 0.08: 7134.1}


def main():
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for k, (n, o) in ORD.items():
        ax[0].semilogx(n, o, "o-", label=k)
    ax[0].axhline(2, color="k", ls="--")
    ax[0].axhline(1, color="gray", ls=":")
    ax[0].set_xlabel("N of the coarser grid in the pair (d_N / d_2N)")
    ax[0].set_ylabel("observed p order")
    ax[0].set_title("order vs resolution for stronger pulses")
    ax[0].legend(fontsize=8)
    ax[1].bar([str(k) for k in STEEP], list(STEEP.values()), color=["tab:orange", "tab:red"])
    ax[1].set_ylabel("max |dp/dx| at T away from interfaces / EPS (N=3200)")
    ax[1].set_title("steepening at T = 0.035 (larger = closer to shock formation)")
    fig.tight_layout()
    fig.savefig(PNG, dpi=110)
    print(f"Plot saved: {PNG}")


if __name__ == "__main__":
    import os
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    main()
