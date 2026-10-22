"""Section 62 summary: force-correction variants of the interface-fitted scheme, accuracy (compatible
pulse, Richardson successive differences, verify_pulse_richardson.py) against linear energy stability
(verify_l3f_band.py).  Numbers are the printed outputs of those runs (2026-10-22)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PNG = "results/lp_imex_moment/corr_variants.png"
# name: (near-interface rho Linf d_400, d_800, global rho L2 order d400/d800, linear stability verdict)
V = {
    "no correction": (2.795e-5, 1.191e-5, 1.83, "proved (S=I)"),
    "lagged, all zones": (2.171e-5, 4.321e-6, 1.84, "unstable (u->0)"),
    "secant, raw stencil": (2.420e-5, 1.355e-5, 1.85, "unstable (theta~0.1)"),
    "secant, raw, gated": (2.279e-5, 3.866e-6, 1.84, "unstable (theta~0.1)"),
    "secant, fullonly": (2.559e-5, 1.353e-5, 1.85, "measured (sup 1.08)"),
    "secant, fullsym": (2.750e-5, 1.033e-5, 1.84, "proved, frozen geometry"),
    "secant, fullsym, gated": (1.555e-5, 1.148e-5, 1.81, "proved, frozen geometry"),
}


def main():
    names = list(V)
    near = np.array([np.log2(V[k][0] / V[k][1]) for k in names])
    glob = np.array([V[k][2] for k in names])
    col = ["tab:green" if v[3].startswith("proved") else ("tab:red" if v[3].startswith("unstable") else "tab:orange")
           for v in V.values()]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    y = np.arange(len(names))
    ax.barh(y - 0.2, near, 0.4, color=col, label="near-interface L-inf order (d400/d800)")
    ax.barh(y + 0.2, glob, 0.4, color="lightgray", edgecolor="k", label="global L2 order (d400/d800)")
    for i, k in enumerate(names):
        ax.text(max(near[i], glob[i]) + 0.05, y[i], V[k][3], va="center", fontsize=8)
    ax.axvline(2.0, color="k", ls="--", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlim(0, 4.2)
    ax.set_xlabel("observed order")
    ax.set_title("force correction: accuracy vs energy stability (green proved, orange measured, red unstable)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(PNG, dpi=115)
    print(f"Plot saved: {PNG}")


if __name__ == "__main__":
    import os
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    main()
