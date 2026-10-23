"""Section 82 figure: interface impulse response S1 on a non-uniform background (EPS_bg = 0.08 pulse on the interface,
u0 = 1) versus the impulse amplitude, with and without the limited reconstruction.  Numbers = verify_variable_coefficient_
interface.py output (2026-10-23)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PNG = "results/lp_imex_moment/variable_coefficient.png"
EPS = [1e-6, 1e-5, 1e-4, 1e-3]
WITH = [5244.795, 3.383, 54.381, 7.443]
WITHOUT = [2.665, 2.365, 2.345, 2.345]


def main():
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.loglog(EPS, WITH, "o-", label="limited linear reconstruction (minmod)")
    ax.loglog(EPS, WITHOUT, "s-", label="first-order remap (no reconstruction)")
    ax.axhline(3.425, color="k", ls="--", label="uniform background, limiter (3.425)")
    ax.set_xlabel("impulse amplitude eps")
    ax.set_ylabel("S1 (two-run difference)")
    ax.set_title("non-uniform background: response is Lipschitz only without the limiter")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(PNG, dpi=110)
    print(f"Plot saved: {PNG}")


if __name__ == "__main__":
    import os
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    main()
