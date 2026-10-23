"""Section 77 summary figure: ring amplitude A(theta) from the pole residue, reduced linear model vs real scheme,
and the spread of the ring-damping estimates kappa (numbers = printed outputs of derive_ring_amplitude.py,
derive_ring_damping.py, reduced_linear_model.py, floquet_ring.py, 2026-10-23)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PNG = "results/lp_imex_moment/ring_derivation.png"
TH = [0.05, 0.1, 0.3, 0.5, 0.7, 0.9]
A_PRED = [0.1836, 0.1991, 0.3115, 0.7591, 1.7576, 2.9417]
A_MEAS = [0.188, 0.205, 0.320, 0.763, 1.759, 2.929]
NU_SERIES = 0.0717
MEAS = [0.6414, -1.0608, 0.0239, 0.8172, -0.5862, -0.4093, 0.7634, -0.4202, -0.1345, -0.0308, 0.0133, 0.0719, 0.0483, -0.0855, -0.0533, 0.0859, 0.0295, -0.0852, 0.0175, 0.0573, -0.0648, 0.0358, 0.0004, -0.0009, 0.0041, -0.0003, -0.0045, 0.0014, 0.0031, -0.0023, -0.0016, 0.0026, -0.0009, -0.0024, 0.0024, -0.0019, -0.0001, 0.0008, 0.0009, 0.0005, -0.0002, -0.0006, -0.0002, 0.0002, -0.0001, -0.0003, 0.0004, 0.0004, -0.0004, 0.0, -0.0002, -0.0002, -0.0002, -0.0001, 0.0, 0.0001, 0.0001, 0.0002, 0.0004, 0.0004, 0.0001, 0.0001, 0.0001, -0.0002, -0.0002, -0.0003, -0.0004, -0.0004, -0.0004, -0.0001]
MODEL = [0.6353, -1.0659, 0.0193, 0.8137, -0.5636, -0.402, 0.7455, -0.4144, -0.1315, -0.0229, 0.0043, 0.0777, 0.0383, -0.0853, -0.0324, 0.0786, 0.0098, -0.069, 0.0253, 0.0359, -0.0483, 0.0338, 0.0018, -0.003, 0.0031, -0.0004, -0.0036, 0.0019, 0.0021, -0.0024, -0.0005, 0.0025, -0.0014, -0.0013, 0.0024, -0.0017, -0.0003, 0.0002, 0.0, 0.0001, 0.0001, -0.0002, -0.0001, 0.0002, 0.0, -0.0002, 0.0, 0.0001, -0.0001, 0.0001, 0.0, -0.0, 0.0, -0.0, -0.0, 0.0, 0.0, -0.0, -0.0, 0.0, -0.0, -0.0, 0.0, -0.0, -0.0, 0.0, 0.0, -0.0, 0.0, -0.0]
S1 = [(0.03584150092693391, 7.180035165817667, 6.86477691546818), (0.07168300185386782, 5.574398678634442, 5.3824790013000845), (0.11947166975644637, 4.127397491082235, 3.94830420481836), (0.1911546716103142, 2.0834393433941134, 2.024253777125778)]     # (nu, real scheme, reduced model), N=200, W=70, theta0=0.37
KAPPA = [("pure Nyquist\nupwind 1-2nu", 2.0), ("single-branch\nlog-avg, nu=.036", 3.65), ("single-branch\nlog-avg, nu=.12", 3.33),
         ("envelope fit\n(measured)", 2.51), ("Floquet slowest\nnu=1/8", 1.25), ("Floquet slowest\nnu=1/4", 0.93)]


def main():
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.3))
    ax[0].plot(TH, A_MEAS, "o-", label="real scheme (nu=0)")
    ax[0].plot(TH, A_PRED, "s--", label="pole residue (4/pi)(Z_L+Z_R) tau |w0|^2/E")
    ax[0].set_xlabel("theta")
    ax[0].set_ylabel("ring amplitude A")
    ax[0].set_title("A(theta): residue formula vs real scheme")
    ax[0].legend(fontsize=7)
    ax[1].plot(np.arange(1, 71), MEAS, "o-", ms=3, label="real scheme")
    ax[1].plot(np.arange(1, 71), MODEL, "x--", ms=4, label="reduced linear model")
    ax[1].set_xlabel("step")
    ax[1].set_ylabel("normalised interface trace")
    ax[1].set_title(f"impulse response, nu={NU_SERIES:.3f}")
    ax[1].legend(fontsize=7)
    nus = [r[0] for r in S1]
    ax[2].plot(nus, [r[1] for r in S1], "o-", label="real scheme")
    ax[2].plot(nus, [r[2] for r in S1], "s--", label="reduced linear model")
    ax[2].set_xlabel("nu")
    ax[2].set_ylabel("S1 (N=200, W=70)")
    ax[2].set_title("S1: model reproduces the real scheme")
    ax[2].legend(fontsize=7)
    ax[3].bar(range(len(KAPPA)), [k[1] for k in KAPPA], color=["gray", "tab:red", "tab:red", "tab:blue", "tab:green", "tab:green"])
    ax[3].set_xticks(range(len(KAPPA)))
    ax[3].set_xticklabels([k[0] for k in KAPPA], fontsize=6)
    ax[3].set_ylabel("kappa = delta / nu")
    ax[3].set_title("ring damping: no single kappa")
    fig.tight_layout()
    fig.savefig(PNG, dpi=110)
    print(f"Plot saved: {PNG}")


if __name__ == "__main__":
    import os
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    main()
