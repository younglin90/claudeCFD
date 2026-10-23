"""Section 80 figure: order of the p field vs resolution for EPS = 0.08 at three end times (steepening of the pulse
grows with T/t_b, t_b ~ 0.04).  Numbers = outputs of verify_gradient_asymptotics.py (2026-10-23)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PNG = "results/lp_imex_moment/gradient_T.png"
DATA = {"T=0.0175 (T/t_b~0.44)": ([200, 400], [1.51, 1.69]), "T=0.0245 (T/t_b~0.61)": ([200, 400], [1.24, 1.41]),
        "T=0.035 (T/t_b~0.88)": ([400, 800], [0.83, 0.89])}


def main():
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for k, (n, o) in DATA.items():
        ax.semilogx(n, o, "o-", label=k)
    ax.axhline(2, color="k", ls="--")
    ax.set_xlabel("N of the coarser grid in the pair")
    ax.set_ylabel("observed p order (EPS = 0.08)")
    ax.set_title("order recovers when the pulse is further from breaking")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PNG, dpi=110)
    print(f"Plot saved: {PNG}")


if __name__ == "__main__":
    import os
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    main()
